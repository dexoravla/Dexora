#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
from collections import OrderedDict
from models.rdt.blocks import RDTBlock, TimestepEmbedder, get_multimodal_cond_pos_embed, get_1d_sincos_pos_embed_from_grid


class ScoringModel(nn.Module):
    """
    Scoring model that takes state, action chunks, and \\hat{log pi}_t as input
    and outputs a 0-1 score for episode quality assessment.

    Used for Positive-Unlabeled (PU) learning to distinguish expert vs
    non-expert episodes (Dexora §III-C). Tokens fed to the transformer follow
    the paper:

        [s_t ; a_{t:t+L-1} ; \\hat{log pi}_t]

    while language + multi-view image tokens form a separate condition stream.

    Implementation note about the log-pi token
    -----------------------------------------
    The paper treats \\hat{log pi}_t as a single scalar that becomes one token
    after a learned projection. Because the scalar's dynamic range is bounded
    (it is the z-scored denoising residual energy, see ``compute_logpi.py``)
    we project it through a small **sinusoidal positional-style encoding**
    before the linear layer:

        x -> [x, sin(2^0 pi x), cos(2^0 pi x), ..., sin(2^{F-1} pi x), cos(...)]
          -> Linear(2F + 1 -> hidden_size)

    This is mathematically equivalent to "a single learned linear projection
    of \\hat{log pi}_t" in expressivity (the linear layer can collapse the
    extra frequencies to zero) but in practice it makes the model much more
    robust to bf16 underflow on near-zero scores and helps the network pick
    up small differences between high-quality and low-quality clips. We use
    ``F = 8`` frequency bands and keep the raw input as the first feature.
    """

    def __init__(
        self,
        state_dim=36,
        action_dim=36,
        action_chunk_size=32,
        hidden_size=512,
        depth=12,
        num_heads=8,
        max_lang_cond_len=1024,
        img_cond_len=4096,
        lang_token_dim=4096,
        img_token_dim=1152,
        lang_pos_embed_config=None,
        img_pos_embed_config=None,
        dtype=torch.bfloat16,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.hidden_size = hidden_size
        self.max_lang_cond_len = max_lang_cond_len
        self.img_cond_len = img_cond_len
        self.lang_token_dim = lang_token_dim
        self.img_token_dim = img_token_dim
        self.dtype = dtype
        self.lang_pos_embed_config = lang_pos_embed_config
        self.img_pos_embed_config = img_pos_embed_config

        # Input projections
        self.state_proj = nn.Linear(state_dim, hidden_size)
        self.action_proj = nn.Linear(action_dim, hidden_size)

        # Sinusoidal positional-style encoding for scalar logpi.  High-frequency
        # bands under bf16 can overflow -> NaNs, so we keep a modest count and
        # run the trig in float32 inside _scalar_sincos_encoding.
        self.logpi_num_frequencies = 8
        self.logpi_include_input = True
        logpi_in_dim = (1 if self.logpi_include_input else 0) + 2 * self.logpi_num_frequencies
        self.logpi_proj = nn.Linear(logpi_in_dim, hidden_size)

        # Conditioning-stream adapters are registered up front so they appear
        # in `parameters()` and are picked up by Accelerator/DDP/optimizer.
        self.lang_proj = (
            nn.Linear(lang_token_dim, hidden_size, bias=False)
            if lang_token_dim != hidden_size else nn.Identity()
        )
        self.img_proj = (
            nn.Linear(img_token_dim, hidden_size, bias=False)
            if img_token_dim != hidden_size else nn.Identity()
        )
        
        # Positional embeddings
        # [state; action_chunk; logpi]
        seq_len = 1 + action_chunk_size + 1  # state + actions + single logpi
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, hidden_size))
        
        # Language and image condition embeddings (same as RDT)
        self.lang_cond_pos_embed = nn.Parameter(
            torch.zeros(1, max_lang_cond_len, hidden_size))
        self.img_cond_pos_embed = nn.Parameter(
            torch.zeros(1, img_cond_len, hidden_size))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            RDTBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        
        # Output head for scoring (0-1 probability)
        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        self.initialize_weights()

        print("Diffusion params: %e" % sum(
            [p.numel() for p in self.blocks.parameters()] + 
            [p.numel() for p in self.action_proj.parameters()] + 
            [p.numel() for p in self.logpi_proj.parameters()] + 
            [p.numel() for p in self.score_head.parameters()] + 
            [p.numel() for p in self.state_proj.parameters()]))
    
    def initialize_weights(self):
        """Initialize weights using similar strategy as RDT"""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize positional embeddings
        pos_embed = get_multimodal_cond_pos_embed(
            embed_dim=self.hidden_size,
            mm_cond_lens=OrderedDict([
                ('state', 1),
                ('action', self.action_chunk_size),
                ('logpi', 1),
            ])
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Language condition pos embed
        if self.lang_pos_embed_config is None:
            lang_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(
                self.hidden_size, torch.arange(self.max_lang_cond_len))
        else:
            lang_cond_pos_embed = get_multimodal_cond_pos_embed(
                embed_dim=self.hidden_size,
                mm_cond_lens=OrderedDict(self.lang_pos_embed_config),
                embed_modality=False
            )
        self.lang_cond_pos_embed.data.copy_(
            torch.from_numpy(lang_cond_pos_embed).float().unsqueeze(0))
        
        # Image condition pos embed
        if self.img_pos_embed_config is None:
            img_cond_pos_embed = get_1d_sincos_pos_embed_from_grid(
                self.hidden_size, torch.arange(self.img_cond_len))
        else:
            img_cond_pos_embed = get_multimodal_cond_pos_embed(
                embed_dim=self.hidden_size,
                mm_cond_lens=OrderedDict(self.img_pos_embed_config),
                embed_modality=False
            )
        self.img_cond_pos_embed.data.copy_(
            torch.from_numpy(img_cond_pos_embed).float().unsqueeze(0))
    
    def forward(self, state, action_chunk, logpi_chunk=None, lang_cond=None, img_cond=None):
        """
        Forward pass of the scoring model.
        
        Args:
            state: Current state [B, state_dim]
            action_chunk: Action sequence [B, action_chunk_size, action_dim]
            logpi_chunk: Log probability for the action chunk [B, 1] (optional)
            lang_cond: Language conditioning [B, max_lang_cond_len, hidden_size] (optional)
            img_cond: Image conditioning [B, img_cond_len, hidden_size] (optional)
            
        Returns:
            score: Quality score between 0 and 1 [B, 1]
        """
        B = state.shape[0]
        device = state.device
        
        # Project inputs to hidden dimension
        state_tokens = self.state_proj(state).unsqueeze(1)  # [B, 1, hidden_size]
        action_tokens = self.action_proj(action_chunk)  # [B, action_chunk_size, hidden_size]
        
        # Handle logpi with sinusoidal encoding (use zeros if not provided)
        if logpi_chunk is None:
            logpi_chunk = torch.zeros(B, 1, device=device, dtype=state.dtype)
        # logpi_chunk: [B, 1] -> sinusoidal encoding -> [B, D] -> proj -> [B, 1, hidden]
        logpi_encoded = self._scalar_sincos_encoding(
            logpi_chunk, num_frequencies=self.logpi_num_frequencies,
            include_input=self.logpi_include_input
        )
        logpi_tokens = self.logpi_proj(logpi_encoded).unsqueeze(1)  # [B, 1, hidden_size]
        
        # Concatenate sequence tokens
        x = torch.cat([state_tokens, action_tokens, logpi_tokens], dim=1)  # [B, seq_len, hidden_size]
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Prepare conditioning. The projectors are created eagerly in __init__
        # so they participate in optimizer/DDP/accelerator.prepare().
        cond_tokens = []
        if lang_cond is not None:
            assert lang_cond.shape[-1] == self.lang_token_dim, (
                f"lang_cond last dim {lang_cond.shape[-1]} != configured lang_token_dim {self.lang_token_dim}"
            )
            lang_cond = self.lang_proj(lang_cond)
            lang_cond = lang_cond + self.lang_cond_pos_embed[:, :lang_cond.shape[1]]
            cond_tokens.append(lang_cond)

        if img_cond is not None:
            assert img_cond.shape[-1] == self.img_token_dim, (
                f"img_cond last dim {img_cond.shape[-1]} != configured img_token_dim {self.img_token_dim}"
            )
            img_cond = self.img_proj(img_cond)
            img_cond = img_cond + self.img_cond_pos_embed[:, :img_cond.shape[1]]
            cond_tokens.append(img_cond)

        if len(cond_tokens) == 0:
            # Provide a minimal empty-cond tensor with the right shape so blocks can run.
            cond = state.new_zeros((B, 0, self.hidden_size))
        else:
            cond = torch.cat(cond_tokens, dim=1)
        
        # Apply transformer blocks with conditioning
        for block in self.blocks:
            x = block(x, cond)
        
        # Global average pooling over sequence dimension
        x_pooled = x.mean(dim=1)  # [B, hidden_size]
        
        # Generate score
        score = self.score_head(x_pooled)  # [B, 1]
        
        return score

    def _scalar_sincos_encoding(self, x, num_frequencies=32, include_input=True):
        """
        Sinusoidal positional-style encoding for scalar inputs.
        Args:
            x: [B, 1] tensor of scalars
            num_frequencies: number of frequency bands
            include_input: whether to concatenate the raw input
        Returns:
            [B, (1 if include_input else 0) + 2*num_frequencies]
        """
        # Ensure correct dtype/device and perform the trig in float32 to avoid bf16 overflow
        B = x.shape[0]
        device = x.device
        # Compute in float32 for numerical stability, then cast back
        x_f32 = x.to(device=device, dtype=torch.float32)
        # Frequencies: 2^k * pi, k=0..num_frequencies-1 (kept small for stability)
        k = torch.arange(num_frequencies, device=device, dtype=torch.float32)
        freqs = (2.0 ** k) * torch.pi  # [F]
        # Broadcast multiply -> [B, F]
        angles = x_f32 * freqs
        # Wrap angles to [-pi, pi] to avoid huge magnitudes before sin/cos
        two_pi = 2.0 * torch.pi
        angles = torch.remainder(angles + torch.pi, two_pi) - torch.pi
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        parts = [sin, cos]
        if include_input:
            parts.insert(0, x_f32)
        enc = torch.cat(parts, dim=-1)
        # Cast back to the model's compute dtype
        return enc.to(dtype=self.dtype)


class ScoringModelRunner:
    """
    Runner class for the scoring model, similar to RDTRunner.
    """

    def __init__(self, config):
        self.config = config
        # img_cond_len defaults to (img_history_size * num_cameras * 27 * 27)
        # matching SigLip-SO400M at 384x384 (729 patches per view).
        img_cond_len = (
            config['common'].get('img_history_size', 1)
            * config['common'].get('num_cameras', 4)
            * config['common'].get('num_patches_per_view', 27 * 27)
        )
        self.model = ScoringModel(
            state_dim=config['model']['state_token_dim'],
            action_dim=config['model']['state_token_dim'],  # Same as state_dim
            action_chunk_size=config['common']['action_chunk_size'],
            hidden_size=config['model']['scoring']['hidden_size'],
            depth=config['model']['scoring']['depth'],
            num_heads=config['model']['scoring']['num_heads'],
            max_lang_cond_len=config['dataset']['tokenizer_max_length'],
            img_cond_len=img_cond_len,
            lang_token_dim=config['model'].get('lang_token_dim', 4096),
            img_token_dim=config['model'].get('img_token_dim', 1152),
            dtype=torch.bfloat16,
        )
    
    def save_pretrained(self, save_directory):
        """Save the model to a directory"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model state dict
        torch.save(self.model.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        
        # Save config
        import json
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)
    
    def load_pretrained(self, model_path):
        """Load pretrained model"""
        import os
        if os.path.isfile(model_path):
            # Direct path to model file
            state_dict = torch.load(model_path, map_location="cpu")
        else:
            # Directory with pytorch_model.bin
            model_file = os.path.join(model_path, "pytorch_model.bin")
            state_dict = torch.load(model_file, map_location="cpu")

        # Tolerate optional wrappers from older training scripts.
        if isinstance(state_dict, dict):
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "module" in state_dict and isinstance(state_dict["module"], dict):
                state_dict = state_dict["module"]

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[ScoringModelRunner] load_pretrained: missing={len(missing)} unexpected={len(unexpected)}")
        return self.model

    @torch.no_grad()
    def score_episode(
        self,
        clips,
        device=None,
        aggregation: str = "mean",
    ):
        """
        Episode-level scoring by sub-clip aggregation (Dexora §III-C).

        Args:
            clips: A list of K dicts, each containing tensors for
                {"state", "action_chunk", "logpi_chunk", "lang_cond", "img_cond"}.
                Each tensor's leading dimension is **1** (single example per clip).
            device: Optional device override.
            aggregation: How to reduce per-clip scores to a single number.
                One of {"mean", "median", "min", "max"}.

        Returns:
            A python float ``d(τ) ∈ (0, 1]`` plus the per-clip score tensor.
        """
        import torch
        self.model.eval()
        scores = []
        for clip in clips:
            kwargs = {k: v.to(device) if device is not None and torch.is_tensor(v) else v
                      for k, v in clip.items()}
            s = self.model(**kwargs)
            scores.append(s.view(-1))
        scores_t = torch.cat(scores, dim=0)
        if aggregation == "mean":
            agg = scores_t.mean()
        elif aggregation == "median":
            agg = scores_t.median()
        elif aggregation == "min":
            agg = scores_t.min()
        elif aggregation == "max":
            agg = scores_t.max()
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        return float(agg.item()), scores_t
