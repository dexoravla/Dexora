#!/usr/bin/env python
# coding=utf-8

import copy
import logging
import math
import os
import json
from pathlib import Path
from typing import Optional

import diffusers
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import yaml
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import io

from models.scoring_model import ScoringModel, ScoringModelRunner
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.multimodal_encoder.t5_encoder import T5Embedder
from train.dataset import DataCollatorForVLAConsumerDataset, VLAConsumerDataset
from data.bson_vla_dataset import BsonVLADataset
from data.bson_vla_dataset_with_logpi import BsonVLADatasetWithLogpi
from data.lerobot_vla_dataset import LeRobotVLADataset
from data.lerobot_vla_dataset_with_logpi import LeRobotVLADatasetWithLogpi

if is_wandb_available():
    import wandb


class ScoringDataset(torch.utils.data.Dataset):
    """
    Dataset for scoring model training with PU learning.
    Wraps the original VLA dataset and adds episode quality labels.

    The positive set comes from Shigh (paper §III-C) when ``shigh_file`` is
    provided. Otherwise it falls back to Spre (the output of
    ``analyze_episode_quality.py``) — useful for ablations and for users who
    have not yet run ``replay_validate.py``.
    """

    def __init__(
        self,
        vla_dataset,
        valid_episodes_file: str = "new_lerobot_jerk/complete_analysis_results.json",
        shigh_file: Optional[str] = None,
    ):
        self.vla_dataset = vla_dataset

        if shigh_file is not None and os.path.exists(shigh_file):
            with open(shigh_file, 'r') as f:
                shigh_data = json.load(f)
            self.valid_episodes = set(int(x) for x in shigh_data["shigh_episodes"])
            print(
                f"[ScoringDataset] Positives = Shigh from {shigh_file} "
                f"({len(self.valid_episodes)} episodes)."
            )
        else:
            with open(valid_episodes_file, 'r') as f:
                analysis_results = json.load(f)
            self.valid_episodes = set(int(x) for x in analysis_results["filtering_thresholds"]["valid_episodes"])
            print(
                f"[ScoringDataset] Positives = Spre from {valid_episodes_file} "
                f"({len(self.valid_episodes)} episodes). "
                "Consider running replay_validate.py to produce Shigh."
            )
        
    def __len__(self):
        return len(self.vla_dataset)
    
    def __getitem__(self, idx):
        # Get original data from VLA dataset
        data = self.vla_dataset[idx]
        
        # Extract episode ID from metadata - handle different possible formats
        episode_id = None
        if 'meta' in data and isinstance(data['meta'], dict):
            # Prefer explicit episode index from meta (present in LeRobot wrapper)
            ep_idx = data['meta'].get('episode_idx', None)
            if ep_idx is not None:
                try:
                    episode_id = int(ep_idx)
                except Exception:
                    episode_id = ep_idx
            else:
                episode_id = data['meta'].get('episode_id', None)

        # If still not found, try BSON-specific path extraction
        if episode_id is None:
            # For BSON dataset, episode info is available on the base dataset directly
            if hasattr(self.vla_dataset, 'episode_infos'):
                try:
                    if 0 <= idx < len(self.vla_dataset.episode_infos):
                        episode_path = str(self.vla_dataset.episode_infos[idx].path)
                        import re
                        match = re.search(r'episode_(\d+)', episode_path)
                        if match:
                            episode_id = int(match.group(1))
                except Exception:
                    pass
        
        # Determine if this is an expert sample (positive)
        is_expert = episode_id in self.valid_episodes if episode_id is not None else False
        
        # Add label to data
        data['is_expert'] = torch.tensor(1.0 if is_expert else 0.0, dtype=torch.float32)
        data['episode_id'] = episode_id
        
        return data


def pu_loss_function(scores, is_expert, eta=0.5, variant="paper"):
    """
    Positive-Unlabeled discriminator loss for the Dexora discriminator (Eq.(7)).

    Two supported variants:

    * ``variant="paper"`` — Dexora paper Eq.(7):
        L_D = eta * E_{Shigh}[-log d]  +  E_{U}[-log(1 - d)]
      with ``eta = 0.5`` by default. This is the formulation the paper
      explicitly writes down (two BCE terms, positives & unlabeled).

    * ``variant="dwbc"`` — the original DWBC formulation of Xu et al. ICML'22
      (ref. [41] in the paper):
        L_D = eta * E_{Shigh}[-log d]
            +       E_{U}[-log(1 - d)]
            - eta * E_{Shigh}[-log(1 - d)]
      The third subtractive term debiases the unlabeled term using the
      positives. Empirically close to the paper variant when |Shigh| << |U|.

    Reduction
    ---------
    Each expectation is implemented as a **separate mean** over its sub-batch
    (positives vs unlabeled) rather than a single ``mean()`` over the full
    batch. This is a numerically safer estimator of the two expectations when
    Shigh is rare (|Shigh| << |U|, paper: ~15%) and matches what DWBC does in
    practice; it differs from a single batch mean by at most an O(|Shigh|/B)
    re-weighting, which is folded into the ``eta`` hyper-parameter.

    Clipping
    --------
    Following the paper's "apply clip scores to d ∈ [0.1, 0.9] for stability",
    we hard-clamp ``scores`` (already in (0, 1) after the sigmoid head) to that
    interval before taking logs, then add an additional ``tiny`` epsilon to
    keep ``log(1-d)`` finite at the boundary.

    Args:
        scores: Model output scores [B, 1] in (0, 1) (after sigmoid).
        is_expert: Binary labels [B] (1.0 for positives in Shigh, 0.0 for U).
        eta: PU weight (paper uses 0.5).
        variant: "paper" (default) or "dwbc".

    Returns:
        (loss, log_dict)
    """
    # Squeeze and upcast to float32 for numerically-stable logs under bf16
    scores = scores.squeeze(-1).to(dtype=torch.float32)  # [B]

    # Paper: "apply clip scores to d ∈ [0.1, 0.9] for stability".
    # A tiny epsilon keeps log(1 - d) finite at the boundary.
    tiny = 1e-6
    scores = torch.clamp(scores, 0.1, 0.9)
    scores = torch.clamp(scores, tiny, 1.0 - tiny)

    expert_mask = is_expert == 1.0
    unlabeled_mask = is_expert == 0.0

    log_d = torch.log(scores)
    log_1m_d = torch.log1p(-scores)

    if expert_mask.any():
        expert_pos_term = eta * (-log_d[expert_mask]).mean()
        expert_neg_term = eta * (-log_1m_d[expert_mask]).mean()
    else:
        expert_pos_term = torch.zeros((), device=scores.device)
        expert_neg_term = torch.zeros((), device=scores.device)

    if unlabeled_mask.any():
        unlabeled_term = (-log_1m_d[unlabeled_mask]).mean()
    else:
        unlabeled_term = torch.zeros((), device=scores.device)

    if variant == "paper":
        loss = expert_pos_term + unlabeled_term
    elif variant == "dwbc":
        loss = expert_pos_term + unlabeled_term - expert_neg_term
    else:
        raise ValueError(f"Unknown PU variant: {variant!r}; expected 'paper' or 'dwbc'.")

    return loss, {
        'expert_pos_term': float(expert_pos_term.item()),
        'unlabeled_term': float(unlabeled_term.item()),
        'expert_neg_term': float(expert_neg_term.item()),
        'num_expert': int(expert_mask.sum().item()),
        'num_unlabeled': int(unlabeled_mask.sum().item()),
        'variant': variant,
        'eta': float(eta),
    }


def train_scoring_model(args, logger):
    """Main training function for scoring model"""
    
    # Read the config
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)
    
    # Add scoring model config
    if 'scoring' not in config['model']:
        config['model']['scoring'] = {
            'hidden_size': 512,
            'depth': 12,
            'num_heads': 8
        }
    
    logging_dir = Path(args.output_dir, args.logging_dir)
    
    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
    accelerator = Accelerator(
        deepspeed_plugin=DeepSpeedPlugin(
            hf_ds_config=args.deepspeed
        ) if args.deepspeed is not None else None,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Initialize encoders
    text_encoder = T5Embedder(
        from_pretrained=args.pretrained_text_encoder_name_or_path,
        model_max_length=config['dataset']['tokenizer_max_length'],
        local_files_only=False,
        device=accelerator.device
    )
    
    vision_encoder = SiglipVisionTower(
        vision_tower=args.pretrained_vision_encoder_name_or_path,
        args=None,
        delay_load=False
    )

    # Determine dtype based on mixed precision and move encoders
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Ensure encoders are on the right device/dtype
    if hasattr(text_encoder, "model") and text_encoder.model is not None:
        text_encoder.model.to(accelerator.device, dtype=weight_dtype)
    if hasattr(vision_encoder, "vision_tower") and vision_encoder.vision_tower is not None:
        vision_encoder.vision_tower.to(accelerator.device, dtype=weight_dtype)

    # Initialize scoring model
    scoring_runner = ScoringModelRunner(config)
    scoring_model = scoring_runner.model
    
    # Load pretrained model if specified
    if args.pretrained_model_name_or_path is not None:
        logger.info(f"Loading pretrained scoring model from {args.pretrained_model_name_or_path}")
        scoring_runner.load_pretrained(args.pretrained_model_name_or_path)

    # Create dataset
    if args.load_from == "bson":
        bson_kwargs = dict(
            logpi_file=args.logpi_file,
            sub_sample=1.0,
            normalize_mode="min_max",
        )
        if getattr(args, "bson_root", None) is not None:
            bson_kwargs["bson_dir"] = args.bson_root
        if getattr(args, "stats_file", None) is not None:
            bson_kwargs["stats_file"] = args.stats_file
        base_dataset = BsonVLADatasetWithLogpi(**bson_kwargs)
        logger.info(f"Logpi statistics: {base_dataset.get_logpi_statistics()}")
    elif args.load_from == "lerobot":
        lerobot_kwargs = dict(logpi_file=args.logpi_file)
        if getattr(args, "lerobot_root", None) is not None:
            lerobot_kwargs["repo_dir"] = args.lerobot_root
        if getattr(args, "stats_file", None) is not None:
            lerobot_kwargs["stats_file"] = args.stats_file
        if getattr(args, "state_dim_keep", None) is not None:
            lerobot_kwargs["state_dim_keep"] = int(args.state_dim_keep)
        base_dataset = LeRobotVLADatasetWithLogpi(**lerobot_kwargs)
        logger.info(f"Logpi statistics: {base_dataset.get_logpi_statistics()}")
    else:
        raise ValueError(f"Unsupported dataset type: {args.load_from}")
    
    # Wrap with scoring dataset; positives = Shigh if provided, else Spre.
    train_dataset = ScoringDataset(
        base_dataset,
        valid_episodes_file=getattr(args, 'spre_file',
                                    "new_lerobot_jerk/complete_analysis_results.json"),
        shigh_file=getattr(args, 'shigh_file', None),
    )
    
    # Create VLA consumer dataset for compatibility
    vla_dataset = VLAConsumerDataset(
        config=config["dataset"],
        tokenizer=text_encoder.tokenizer,
        image_processor=vision_encoder.image_processor,
        num_cameras=config['common']['num_cameras'],
        img_history_size=config['common']['img_history_size'],
        image_aug=args.image_aug,
        dataset_type=args.dataset_type,
        cond_mask_prob=args.cond_mask_prob,
        cam_ext_mask_prob=args.cam_ext_mask_prob,
        state_noise_snr=args.state_noise_snr,
        use_hdf5=args.load_from,
        use_precomp_lang_embed=args.precomp_lang_embed,
        lerobot_root=getattr(args, "lerobot_root", None),
        bson_root=getattr(args, "bson_root", None),
        stats_file=getattr(args, "stats_file", None),
        state_dim_keep=getattr(args, "state_dim_keep", 36),
    )
    vla_dataset.dataset = train_dataset
    
    # Data collator
    data_collator = DataCollatorForVLAConsumerDataset(
        tokenizer=text_encoder.tokenizer
    )

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        vla_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.train_batch_size,
        # num_workers=args.dataloader_num_workers,
        num_workers=0,
    )

    # Optimizer
    # Use AdamW optimizer
    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        scoring_model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    scoring_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        scoring_model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("scoring_model", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(vla_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step - (first_epoch * len(train_dataloader))

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # Training loop
    for epoch in range(first_epoch, args.num_train_epochs):
        scoring_model.train()
        train_loss = 0.0
        expert_count = 0
        unlabeled_count = 0
        
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(scoring_model):
                # Extract inputs
                # Ensure consistent dtype with mixed precision settings
                states = batch['states'].to(dtype=weight_dtype)  # [B, T, state_dim]
                # Sanitize states to avoid NaN/Inf propagation
                states = torch.nan_to_num(states, nan=0.0, posinf=1e6, neginf=-1e6)
                # Use the last state as current state
                state = states[:, -1, :]  # [B, state_dim]
                actions = batch['actions'].to(dtype=weight_dtype)  # [B, action_chunk_size, action_dim]
                actions = torch.nan_to_num(actions, nan=0.0, posinf=1e6, neginf=-1e6)
                is_expert = batch['is_expert'].to(dtype=torch.float32)  # [B]

                # Prepare language and image conditioning aligned with train.py
                with torch.no_grad():
                    images = batch["images"].to(dtype=weight_dtype)
                    images = torch.nan_to_num(images, nan=0.0, posinf=1.0, neginf=0.0)
                    batch_size, _, C, H, W = images.shape
                    image_embeds = vision_encoder(images.reshape(-1, C, H, W)).detach()
                    image_embeds = image_embeds.reshape((batch_size, -1, vision_encoder.hidden_size))

                    lang_attn_mask = batch["lang_attn_mask"]
                    lang_cond = batch["lang_embeds"].to(dtype=weight_dtype) \
                        if args.precomp_lang_embed \
                        else text_encoder.model(
                            input_ids=batch["input_ids"],
                            attention_mask=lang_attn_mask
                        )["last_hidden_state"].detach()
                    # Sanitize conditions
                    image_embeds = torch.nan_to_num(image_embeds, nan=0.0, posinf=1e6, neginf=-1e6)
                    lang_cond = torch.nan_to_num(lang_cond, nan=0.0, posinf=1e6, neginf=-1e6)
                    img_cond = image_embeds

                # Get logpi values from batch data
                logpi_chunk = batch.get('logpi', torch.zeros(actions.shape[0], 1, device=actions.device)).to(dtype=weight_dtype)
                # Ensure shape [B, 1]
                if logpi_chunk.ndim == 1:
                    logpi_chunk = logpi_chunk.view(-1, 1)
                elif logpi_chunk.ndim > 2:
                    logpi_chunk = logpi_chunk.view(logpi_chunk.shape[0], -1)[:, :1]
                # Sanitize and clamp to a safe range
                logpi_chunk = torch.nan_to_num(logpi_chunk, nan=0.0, posinf=50.0, neginf=-50.0)
                logpi_chunk = torch.clamp(logpi_chunk, min=-50.0, max=50.0)
                
                # Forward pass
                scores = scoring_model(
                    state=state,
                    action_chunk=actions,
                    logpi_chunk=logpi_chunk,
                    lang_cond=lang_cond,
                    img_cond=img_cond
                )
                # Guard against non-finite scores
                if not torch.isfinite(scores).all():
                    bad = ~torch.isfinite(scores)
                    accelerator.print(f"[WARN] Non-finite scores detected at step {global_step}, replacing with 0.5. Count={bad.sum().item()}")
                    scores = torch.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
                
                # Calculate PU loss. We use the paper's Eq.(7) by default with eta=0.5.
                # Users can switch to the original DWBC three-term form via --pu_variant dwbc.
                loss, loss_components = pu_loss_function(
                    scores,
                    is_expert,
                    eta=getattr(args, 'eta', 0.5),
                    variant=getattr(args, 'pu_variant', 'paper'),
                )
                # If loss is non-finite, log diagnostics and skip this batch
                if not torch.isfinite(loss):
                    with torch.no_grad():
                        finite_mask = torch.isfinite(scores)
                        if finite_mask.any():
                            s_min = scores[finite_mask].min().item()
                            s_max = scores[finite_mask].max().item()
                        else:
                            s_min, s_max = float('nan'), float('nan')
                    accelerator.print(f"[WARN] Non-finite loss at step {global_step}. Skipping batch. score[min,max]=[{s_min:.6f},{s_max:.6f}] num_expert={int((is_expert==1.0).sum().item())} num_unlabeled={int((is_expert==0.0).sum().item())}")
                    optimizer.zero_grad()
                    continue
                
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                
                # Count expert vs unlabeled samples
                expert_count += (is_expert == 1.0).sum().item()
                unlabeled_count += (is_expert == 0.0).sum().item()

                # Backprop
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(scoring_model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                logs = {
                    "step_loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "expert_pos_term": loss_components['expert_pos_term'],
                    "unlabeled_term": loss_components['unlabeled_term'],
                    "expert_neg_term": loss_components['expert_neg_term'],
                    "expert_ratio": expert_count / (expert_count + unlabeled_count) if (expert_count + unlabeled_count) > 0 else 0.0
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step % args.checkpointing_period == 0:
                    if accelerator.is_main_process:
                        # Save model checkpoint
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        
                        # Also save just the model
                        unwrapped_model = accelerator.unwrap_model(scoring_model)
                        model_save_path = os.path.join(save_path, "scoring_model")
                        os.makedirs(model_save_path, exist_ok=True)
                        torch.save(unwrapped_model.state_dict(), 
                                 os.path.join(model_save_path, "pytorch_model.bin"))
                        
                        # Save config
                        with open(os.path.join(model_save_path, "config.json"), "w") as f:
                            json.dump(config, f, indent=2)

                if global_step >= args.max_train_steps:
                    break

        # Log epoch metrics
        if accelerator.is_main_process:
            epoch_logs = {
                "epoch": epoch,
                "train_loss": train_loss,
                "expert_samples": expert_count,
                "unlabeled_samples": unlabeled_count
            }
            accelerator.log(epoch_logs, step=global_step)
            logger.info(f"Epoch {epoch}: loss={train_loss:.4f}, expert={expert_count}, unlabeled={unlabeled_count}")

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Save final model
        unwrapped_model = accelerator.unwrap_model(scoring_model)
        final_save_path = os.path.join(args.output_dir, "final_model")
        os.makedirs(final_save_path, exist_ok=True)
        torch.save(unwrapped_model.state_dict(), 
                 os.path.join(final_save_path, "pytorch_model.bin"))
        
        # Save config
        with open(os.path.join(final_save_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Training completed! Model saved to {args.output_dir}")

    accelerator.end_training()
