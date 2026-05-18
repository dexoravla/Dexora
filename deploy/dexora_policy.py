#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
``DexoraPolicy`` — thin runtime wrapper around the Dexora policy
(``models.rdt_runner.RDTRunner``) that exposes the same ``get_action(obs_dict)``
interface used by the real-robot inference loop in
``deploy/dexora_inference_zmq.py``.

The wrapper takes care of:

* Loading the policy weights from a Stage-1 / Stage-3 checkpoint directory
  produced by ``train/train{,_posttrain}.py``.
* Loading the SigLIP-SO400M vision encoder and the T5-v1.1-XXL text encoder.
* Mapping the 4 raw camera images to per-camera SigLIP token sequences.
* Encoding the (single) language instruction with T5 to ``[lang_len, 4096]``
  token embeddings + attention mask.
* Concatenating the proprioceptive 36-D state into the ``state_tokens``
  expected by ``RDTRunner.predict_action``.

The output is a numpy array of shape ``[chunk_size, 36]`` in the canonical
order ``[left_arm(6) | right_arm(6) | left_hand(12) | right_hand(12)]``,
all in radians — exactly the layout consumed by ``mmk_forwarder`` (arms,
first 12 dims) and ``xhand_forwarder`` (hands, last 24 dims).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import yaml
from PIL import Image

from models.rdt_runner import RDTRunner
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.multimodal_encoder.t5_encoder import T5Embedder

# Canonical 4-camera order. The keys here MUST be the ones the inference loop
# passes in via ``obs['images']``; the values are passed to SigLIP one-by-one,
# producing ``num_cameras * num_patches_per_view`` tokens per timestep.
DEXORA_CAMERA_ORDER: tuple[str, ...] = (
    "cam_head",          # observation.images.top
    "cam_left_wrist",    # observation.images.wrist_left
    "cam_third_view",    # observation.images.front
    "cam_right_wrist",   # observation.images.wrist_right
)


@dataclass
class DexoraPolicyConfig:
    """Runtime config for ``DexoraPolicy``.

    Most of these fields default to the paper spec (``configs/base_400m.yaml``).
    Override anything you've tweaked at training time.
    """

    model_config_path: str = "configs/base_400m.yaml"
    text_encoder_path: str = "google/t5-v1_1-xxl"
    vision_encoder_path: str = "google/siglip-so400m-patch14-384"

    state_dim: int = 36
    chunk_size: int = 32
    img_history_size: int = 1
    cameras: Sequence[str] = DEXORA_CAMERA_ORDER

    # Inference dtype. bf16 matches training; fall back to fp32 if needed.
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"


class DexoraPolicy:
    """Thin wrapper around ``RDTRunner`` for real-robot inference."""

    def __init__(
        self,
        model_path: str,
        cfg: Optional[DexoraPolicyConfig] = None,
    ) -> None:
        self.cfg = cfg or DexoraPolicyConfig()
        self.device = torch.device(self.cfg.device)

        # ---- Load YAML config the policy was trained with ------------------
        with open(self.cfg.model_config_path, "r") as f:
            self.model_yaml = yaml.safe_load(f)

        # Validate basic dims match the YAML so we fail fast on mismatched ckpts.
        cfg_state_dim = int(self.model_yaml["common"]["state_dim"])
        cfg_chunk = int(self.model_yaml["common"]["action_chunk_size"])
        if cfg_state_dim != self.cfg.state_dim:
            logging.warning(
                f"[DexoraPolicy] config_state_dim={cfg_state_dim} != "
                f"runtime state_dim={self.cfg.state_dim}; slicing/padding "
                "may be required upstream."
            )
        if cfg_chunk != self.cfg.chunk_size:
            logging.warning(
                f"[DexoraPolicy] config_action_chunk={cfg_chunk} != "
                f"runtime chunk_size={self.cfg.chunk_size}; using YAML value."
            )
            self.cfg.chunk_size = cfg_chunk

        # ---- Load encoders -------------------------------------------------
        logging.info(f"Loading T5 text encoder from {self.cfg.text_encoder_path} ...")
        text_embedder = T5Embedder(
            from_pretrained=self.cfg.text_encoder_path,
            model_max_length=self.model_yaml["dataset"]["tokenizer_max_length"],
            device=self.device,
        )
        self.tokenizer = text_embedder.tokenizer
        self.text_encoder = text_embedder.model.to(self.device, dtype=self.cfg.dtype).eval()

        logging.info(f"Loading SigLIP vision encoder from {self.cfg.vision_encoder_path} ...")
        self.vision_encoder = SiglipVisionTower(
            vision_tower=self.cfg.vision_encoder_path, args=None
        )
        self.vision_encoder.vision_tower.to(self.device, dtype=self.cfg.dtype).eval()
        self.image_processor = self.vision_encoder.image_processor

        # ---- Load policy ---------------------------------------------------
        logging.info(f"Loading Dexora policy from {model_path} ...")
        self.policy = self._load_policy(model_path).to(self.device, dtype=self.cfg.dtype).eval()
        n_params = sum(p.numel() for p in self.policy.parameters())
        logging.info(f"[DexoraPolicy] policy params = {n_params/1e6:.1f}M")

        # Static action mask (all dims active for the 36-DoF embodiment).
        self._action_mask = torch.ones(
            (1, 1, self.cfg.state_dim), device=self.device, dtype=self.cfg.dtype
        )

        # Cache last language embedding to avoid re-running T5 every step.
        self._cached_instruction: Optional[str] = None
        self._cached_lang_tokens: Optional[torch.Tensor] = None
        self._cached_lang_mask: Optional[torch.Tensor] = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    @torch.inference_mode()
    def get_action(self, obs: dict) -> np.ndarray:
        """
        Run one diffusion sampling pass and return ``[chunk_size, state_dim]``.

        ``obs`` must contain:
          * ``state``        : np.ndarray ``[state_dim]`` (radians)
          * ``images``       : dict ``{cam_name -> np.ndarray [H, W, 3] RGB uint8}``
                               with keys matching ``cfg.cameras``
          * ``instruction``  : str  (single English language goal)
          * ``ctrl_freq``    : float, optional (defaults to config['common'] value)
        """
        # ---- 1. Language ---------------------------------------------------
        lang_tokens, lang_mask = self._encode_language(obs["instruction"])

        # ---- 2. Vision -----------------------------------------------------
        img_tokens = self._encode_images(obs["images"])

        # ---- 3. State + control frequency ---------------------------------
        state = torch.from_numpy(np.asarray(obs["state"], dtype=np.float32))[None, None, :]
        # Pad / truncate to expected state_dim (defensive against 39-D feeders).
        if state.shape[-1] > self.cfg.state_dim:
            state = state[..., : self.cfg.state_dim]
        elif state.shape[-1] < self.cfg.state_dim:
            pad = torch.zeros(1, 1, self.cfg.state_dim - state.shape[-1])
            state = torch.cat([state, pad], dim=-1)
        state = state.to(self.device, dtype=self.cfg.dtype)

        ctrl_freq = float(obs.get("ctrl_freq", 20.0))
        ctrl_freqs = torch.tensor([ctrl_freq], device=self.device, dtype=self.cfg.dtype)

        # ---- 4. Diffusion sampling ----------------------------------------
        action_pred = self.policy.predict_action(
            lang_tokens=lang_tokens,
            lang_attn_mask=lang_mask,
            img_tokens=img_tokens,
            state_tokens=state,
            action_mask=self._action_mask,
            ctrl_freqs=ctrl_freqs,
        )  # [1, chunk_size, state_dim]

        return action_pred[0].float().cpu().numpy()

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------
    def _load_policy(self, model_path: str) -> RDTRunner:
        """Try ``RDTRunner.from_pretrained`` first (HF format), then fall back
        to a raw ``state_dict`` / ``pytorch_model.bin``.
        """
        if os.path.isdir(model_path):
            try:
                return RDTRunner.from_pretrained(model_path)
            except Exception as e:
                logging.warning(
                    f"[DexoraPolicy] from_pretrained failed ({e}); "
                    "constructing from config + raw state_dict instead."
                )

        # Manual construction path.
        cfg = self.model_yaml
        img_cond_len = (
            cfg["common"]["img_history_size"]
            * cfg["common"]["num_cameras"]
            * self.vision_encoder.num_patches
        )
        policy = RDTRunner(
            action_dim=cfg["common"]["state_dim"],
            pred_horizon=cfg["common"]["action_chunk_size"],
            config=cfg["model"],
            lang_token_dim=cfg["model"]["lang_token_dim"],
            img_token_dim=cfg["model"]["img_token_dim"],
            state_token_dim=cfg["model"]["state_token_dim"],
            max_lang_cond_len=cfg["dataset"]["tokenizer_max_length"],
            img_cond_len=img_cond_len,
            img_pos_embed_config=[
                ("image", (cfg["common"]["img_history_size"],
                           cfg["common"]["num_cameras"],
                           -self.vision_encoder.num_patches)),
            ],
            lang_pos_embed_config=[
                ("lang", -cfg["dataset"]["tokenizer_max_length"]),
            ],
            dtype=self.cfg.dtype,
        )

        # Resolve raw state-dict path
        if os.path.isfile(model_path):
            sd_path = model_path
        else:
            sd_path = os.path.join(model_path, "pytorch_model.bin")
            if not os.path.exists(sd_path):
                raise FileNotFoundError(
                    f"No ``pytorch_model.bin`` under {model_path}; please pass "
                    "a HF-style checkpoint directory or a state_dict file."
                )
        sd = torch.load(sd_path, map_location="cpu")
        if isinstance(sd, dict):
            sd = sd.get("module", sd.get("model_state_dict", sd.get("state_dict", sd)))
        missing, unexpected = policy.load_state_dict(sd, strict=False)
        if missing or unexpected:
            logging.warning(
                f"[DexoraPolicy] state_dict load: missing={len(missing)} "
                f"unexpected={len(unexpected)}"
            )
        return policy

    def _encode_language(self, instruction: str) -> tuple[torch.Tensor, torch.Tensor]:
        if instruction == self._cached_instruction and self._cached_lang_tokens is not None:
            return self._cached_lang_tokens, self._cached_lang_mask

        max_len = self.model_yaml["dataset"]["tokenizer_max_length"]
        tokens = self.tokenizer(
            instruction,
            return_tensors="pt",
            padding="max_length",
            max_length=max_len,
            truncation=True,
        )
        input_ids = tokens["input_ids"].to(self.device)
        attn_mask = tokens["attention_mask"].to(self.device)
        lang_embeds = self.text_encoder(
            input_ids=input_ids, attention_mask=attn_mask
        )["last_hidden_state"].to(self.cfg.dtype)

        # Cache one entry (most deployments use a single instruction for a run).
        self._cached_instruction = instruction
        self._cached_lang_tokens = lang_embeds
        self._cached_lang_mask = attn_mask.to(torch.bool)
        return self._cached_lang_tokens, self._cached_lang_mask

    def _encode_images(self, images: dict) -> torch.Tensor:
        """Resize / pad / normalize each of the 4 cameras through SigLIP."""
        # Background colour for any missing camera (mirrors train/dataset.py).
        bg_color = np.array(
            [int(x * 255) for x in self.image_processor.image_mean], dtype=np.uint8
        ).reshape(1, 1, 3)

        pixel_values = []
        for cam in self.cfg.cameras:
            img = images.get(cam)
            if img is None:
                # Use the SigLIP mean-colour background as the "missing camera"
                # placeholder (same trick as the training dataloader).
                H = self.image_processor.size["height"]
                W = self.image_processor.size["width"]
                img = np.ones((H, W, 3), dtype=np.uint8) * bg_color
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            pil = Image.fromarray(img, mode="RGB")
            # Pad-to-square (image_aspect_ratio == 'pad') so wrist views aren't squashed.
            pil = _expand2square(pil, tuple(int(x * 255) for x in self.image_processor.image_mean))
            arr = self.image_processor.preprocess(
                pil, return_tensors="pt"
            )["pixel_values"][0]
            pixel_values.append(arr)
        batch = torch.stack(pixel_values, dim=0).to(self.device, dtype=self.cfg.dtype)
        # [N_cam, T_patch, hidden]
        img_embeds = self.vision_encoder(batch).detach()
        # Flatten to [1, N_cam * T_patch, hidden]
        return img_embeds.reshape(1, -1, self.vision_encoder.hidden_size)


def _expand2square(pil_img: Image.Image, bg_color):
    """Square-pad an image to its longest side (replicates train/dataset.py)."""
    w, h = pil_img.size
    if w == h:
        return pil_img
    if w > h:
        out = Image.new(pil_img.mode, (w, w), bg_color)
        out.paste(pil_img, (0, (w - h) // 2))
        return out
    out = Image.new(pil_img.mode, (h, h), bg_color)
    out.paste(pil_img, ((h - w) // 2, 0))
    return out
