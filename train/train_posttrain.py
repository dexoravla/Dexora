#!/usr/bin/env python
# coding=utf-8
"""
Stage-3: Data-quality-aware post-training (Dexora §III-D).

Starting from a pretrained policy (stage-1, sim corpus) and a trained
discriminator (stage-2, PU on Spre/Shigh), we fine-tune the policy on the
real-world dataset with a discriminator-weighted diffusion loss (Eq.(8)):

    L_pi = sum_i  w_i * || eps_theta(.) - eps ||^2

Per-sample weights w_i are computed online from the discriminator score d(.)
through the DWBC mapping (models/sample_weighting.py), with a short linear
warm-up. The discriminator is frozen during stage-3.

This file is intentionally close in shape to ``train/train.py`` for easy
side-by-side comparison and minimal duplication.
"""

import copy
import logging
import math
import os
from pathlib import Path
from typing import Optional

import diffusers
import torch
import torch.utils.checkpoint
import transformers
import yaml
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin, ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from safetensors.torch import load_model
from tqdm.auto import tqdm

from models.ema_model import EMAModel
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.multimodal_encoder.t5_encoder import T5Embedder
from models.rdt_runner import RDTRunner
from models.sample_weighting import scores_to_train_weights
from models.scoring_model import ScoringModel, ScoringModelRunner
from train.dataset import DataCollatorForVLAConsumerDataset, VLAConsumerDataset
from train.sample import log_sample_res


if is_wandb_available():
    import wandb


def _load_scoring_model(scoring_runner: ScoringModelRunner, ckpt_path: str, logger):
    """Load discriminator weights, supporting both bare state_dicts and
    'model_state_dict' / 'state_dict' wrappers."""
    logger.info(f"Loading frozen discriminator from {ckpt_path} ...")
    if os.path.isfile(ckpt_path):
        sd = torch.load(ckpt_path, map_location="cpu")
    else:
        # treat as directory with pytorch_model.bin
        sd = torch.load(os.path.join(ckpt_path, "pytorch_model.bin"), map_location="cpu")

    if isinstance(sd, dict):
        if "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        elif "state_dict" in sd:
            sd = sd["state_dict"]
        elif "module" in sd:
            sd = sd["module"]

    missing, unexpected = scoring_runner.model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning(f"Discriminator missing keys ({len(missing)}): {missing[:6]} ...")
    if unexpected:
        logger.warning(f"Discriminator unexpected keys ({len(unexpected)}): {unexpected[:6]} ...")
    scoring_runner.model.eval()
    for p in scoring_runner.model.parameters():
        p.requires_grad_(False)


def train_posttrain(args, logger):
    """
    Quality-aware post-training driver. The signature mirrors train.train(...)
    so the same `main.py`-style CLI works.
    """
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
    accelerator = Accelerator(
        deepspeed_plugin=DeepSpeedPlugin(hf_ds_config=args.deepspeed) if args.deepspeed is not None else None,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb" and not is_wandb_available():
        raise ImportError("wandb is requested but not installed.")

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

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # ---------------- Encoders ----------------
    if args.precomp_lang_embed:
        tokenizer, text_encoder = None, None
    else:
        text_embedder = T5Embedder(
            from_pretrained=args.pretrained_text_encoder_name_or_path,
            model_max_length=config["dataset"]["tokenizer_max_length"],
            device=accelerator.device,
        )
        tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model

    vision_encoder = SiglipVisionTower(
        vision_tower=args.pretrained_vision_encoder_name_or_path, args=None
    )
    image_processor = vision_encoder.image_processor

    # ---------------- Policy: load stage-1 weights ----------------
    img_cond_len = (
        config["common"]["img_history_size"]
        * config["common"]["num_cameras"]
        * vision_encoder.num_patches
    )
    assert args.stage1_ckpt is not None, "Post-training requires --stage1_ckpt (the pretrained policy)."
    if os.path.isdir(args.stage1_ckpt):
        logger.info(f"Constructing policy from pretrained directory: {args.stage1_ckpt}")
        rdt = RDTRunner.from_pretrained(args.stage1_ckpt)
    else:
        logger.info("Constructing policy from config; will load stage-1 state_dict below.")
        rdt = RDTRunner(
            action_dim=config["common"]["state_dim"],
            pred_horizon=config["common"]["action_chunk_size"],
            config=config["model"],
            lang_token_dim=config["model"]["lang_token_dim"],
            img_token_dim=config["model"]["img_token_dim"],
            state_token_dim=config["model"]["state_token_dim"],
            max_lang_cond_len=config["dataset"]["tokenizer_max_length"],
            img_cond_len=img_cond_len,
            img_pos_embed_config=[
                ("image", (config["common"]["img_history_size"],
                           config["common"]["num_cameras"],
                           -vision_encoder.num_patches)),
            ],
            lang_pos_embed_config=[
                ("lang", -config["dataset"]["tokenizer_max_length"]),
            ],
            dtype=weight_dtype,
        )
        if os.path.isfile(args.stage1_ckpt):
            sd = torch.load(args.stage1_ckpt, map_location="cpu")
            if isinstance(sd, dict) and "module" in sd:
                sd = sd["module"]
            missing, unexpected = rdt.load_state_dict(sd, strict=False)
            logger.info(f"Loaded stage-1 ckpt. missing={len(missing)} unexpected={len(unexpected)}")

    # EMA tracker for the (post-trained) policy.
    ema_rdt = copy.deepcopy(rdt)
    ema_model = EMAModel(
        ema_rdt,
        update_after_step=config["model"]["ema"]["update_after_step"],
        inv_gamma=config["model"]["ema"]["inv_gamma"],
        power=config["model"]["ema"]["power"],
        min_value=config["model"]["ema"]["min_value"],
        max_value=config["model"]["ema"]["max_value"],
    )

    # ---------------- Frozen discriminator ----------------
    # Skipped when --no_quality_weights is passed (ablation: vanilla post-train).
    scoring_model: Optional[ScoringModel] = None
    if not getattr(args, "no_quality_weights", False):
        # Build a scoring-model config (mirroring configs/scoring.yaml if not present).
        if "scoring" not in config["model"]:
            config["model"]["scoring"] = {"hidden_size": 512, "depth": 12, "num_heads": 8}
        scoring_runner = ScoringModelRunner(config)
        _load_scoring_model(scoring_runner, args.scoring_ckpt, logger)
        scoring_model = scoring_runner.model
    else:
        logger.info("--no_quality_weights set: running VANILLA post-training "
                    "(no discriminator, all w_i = 1).")

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                model_to_save = model.module if hasattr(model, "module") else model
                if isinstance(model_to_save, type(accelerator.unwrap_model(rdt))):
                    model_to_save.save_pretrained(output_dir)

    accelerator.register_save_state_pre_hook(save_model_hook)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps
            * args.train_batch_size * accelerator.num_processes
        )

    optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(
        rdt.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # ---------------- Dataset ----------------
    dataset_common_kwargs = dict(
        config=config["dataset"],
        tokenizer=tokenizer,
        image_processor=image_processor,
        num_cameras=config["common"]["num_cameras"],
        img_history_size=config["common"]["img_history_size"],
        dataset_type=args.dataset_type,
        use_hdf5=args.load_from,
        use_precomp_lang_embed=args.precomp_lang_embed,
        lerobot_root=getattr(args, "lerobot_root", None),
        bson_root=getattr(args, "bson_root", None),
        stats_file=getattr(args, "stats_file", None),
        state_dim_keep=getattr(args, "state_dim_keep", 36),
    )
    train_dataset = VLAConsumerDataset(
        image_aug=args.image_aug,
        cond_mask_prob=args.cond_mask_prob,
        cam_ext_mask_prob=args.cam_ext_mask_prob,
        state_noise_snr=args.state_noise_snr,
        **dataset_common_kwargs,
    )
    sample_dataset = VLAConsumerDataset(
        image_aug=False,
        cond_mask_prob=0,
        cam_ext_mask_prob=-1,
        state_noise_snr=None,
        **dataset_common_kwargs,
    )

    data_collator = DataCollatorForVLAConsumerDataset(tokenizer)

    # ---- Optional Fig.10-style data-composition ablation ----
    # `--real_data_fraction 0.0` -> sim-only (no real samples used; require
    # `--load_from=bson|lerobot` pointing at sim corpus). 0.5 / 1.0 reproduce
    # the "Sim + 50% Real" / "Sim + All Real" rows of Fig. 10.
    real_frac = float(getattr(args, "real_data_fraction", 1.0))
    real_frac = max(0.0, min(real_frac, 1.0))
    if real_frac < 1.0:
        n_keep = max(1, int(round(len(train_dataset) * real_frac)))
        # Deterministic stride keeps the subset balanced across episodes.
        import torch.utils.data as tud
        if n_keep == 0:
            train_subset = tud.Subset(train_dataset, [])
        else:
            stride = max(1, len(train_dataset) // n_keep)
            indices = list(range(0, len(train_dataset), stride))[:n_keep]
            train_subset = tud.Subset(train_dataset, indices)
        logger.info(
            f"--real_data_fraction={real_frac}: using {len(train_subset)}/"
            f"{len(train_dataset)} real samples for stage-3."
        )
        train_dataset_for_loader = train_subset
    else:
        train_dataset_for_loader = train_dataset

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_for_loader,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=(args.dataloader_num_workers > 0),
    )
    sample_dataloader = torch.utils.data.DataLoader(
        sample_dataset,
        batch_size=args.sample_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=(args.dataloader_num_workers > 0),
    )

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

    rdt, optimizer, train_dataloader, sample_dataloader, lr_scheduler = accelerator.prepare(
        rdt, optimizer, train_dataloader, sample_dataloader, lr_scheduler
    )

    ema_rdt.to(accelerator.device, dtype=weight_dtype)
    if scoring_model is not None:
        scoring_model.to(accelerator.device, dtype=weight_dtype)
    if text_encoder is not None:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    if vision_encoder is not None:
        vision_encoder.vision_tower.to(accelerator.device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("dexora-posttrain", config=vars(args))

    total_batch_size = (
        args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )
    logger.info("***** Running stage-3 post-training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches per epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Per-device batch size = {args.train_batch_size}")
    logger.info(f"  Total batch size (w. dist & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Max optimization steps = {args.max_train_steps}")
    logger.info(
        f"  DWBC mapping: eta={args.dwbc_eta}, "
        f"w in [{args.dwbc_w_min}, {args.dwbc_w_max}], warmup={args.dwbc_warmup_steps}"
    )

    global_step = 0
    first_epoch = 0
    resume_step = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if dirs else None
        if path is None:
            accelerator.print(f"No prior checkpoint at {args.resume_from_checkpoint}; starting fresh.")
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            try:
                accelerator.load_state(os.path.join(args.output_dir, path))
            except Exception:
                logger.info("Resuming failed via load_state; trying raw module reload.")
                ck = torch.load(os.path.join(args.output_dir, path, "pytorch_model",
                                             "mp_rank_00_model_states.pt"))
                rdt.module.load_state_dict(ck["module"])
            ema_path = os.path.join(args.output_dir, path, "ema", "model.safetensors")
            if os.path.exists(ema_path):
                load_model(ema_rdt, ema_path)
            global_step = int(path.split("-")[1])
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    progress_bar = tqdm(range(global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        rdt.train()
        if args.resume_from_checkpoint and epoch == first_epoch:
            progress_bar.update(resume_step // args.gradient_accumulation_steps)

        for batch in train_dataloader:
            with accelerator.accumulate(rdt):
                images = batch["images"].to(dtype=weight_dtype)
                states = batch["states"].to(dtype=weight_dtype)
                states = states[:, -1:, :]  # last proprio frame as the conditioning state
                actions = batch["actions"].to(dtype=weight_dtype)
                state_elem_mask = batch["state_elem_mask"].to(dtype=weight_dtype)
                ctrl_freqs = batch["ctrl_freqs"]

                with torch.no_grad():
                    B, _, C, H, W = images.shape
                    image_embeds = vision_encoder(images.reshape(-1, C, H, W)).detach()
                    image_embeds = image_embeds.reshape((B, -1, vision_encoder.hidden_size))
                    lang_attn_mask = batch["lang_attn_mask"]
                    text_embeds = (
                        batch["lang_embeds"].to(dtype=weight_dtype)
                        if args.precomp_lang_embed
                        else text_encoder(
                            input_ids=batch["input_ids"],
                            attention_mask=lang_attn_mask,
                        )["last_hidden_state"].detach()
                    )

                # ---------- Discriminator scoring (frozen) or vanilla baseline ----------
                if scoring_model is None:
                    sample_weights = torch.ones(
                        (actions.shape[0],), device=actions.device, dtype=weight_dtype,
                    )
                    scores = None
                else:
                    with torch.no_grad():
                        logpi_chunk = batch.get("logpi", None)
                        if logpi_chunk is None:
                            logpi_chunk = torch.zeros(B, 1, device=actions.device, dtype=weight_dtype)
                        else:
                            logpi_chunk = logpi_chunk.to(device=actions.device, dtype=weight_dtype)
                            if logpi_chunk.ndim == 1:
                                logpi_chunk = logpi_chunk.view(-1, 1)
                            elif logpi_chunk.ndim > 2:
                                logpi_chunk = logpi_chunk.view(logpi_chunk.shape[0], -1)[:, :1]
                            logpi_chunk = torch.nan_to_num(logpi_chunk, nan=0.0, posinf=50.0, neginf=-50.0)
                            logpi_chunk = torch.clamp(logpi_chunk, -50.0, 50.0)
                        scores = scoring_model(
                            state=states.squeeze(1),
                            action_chunk=actions,
                            logpi_chunk=logpi_chunk,
                            lang_cond=text_embeds,
                            img_cond=image_embeds,
                        )
                        sample_weights = scores_to_train_weights(
                            scores,
                            eta=args.dwbc_eta,
                            w_min=args.dwbc_w_min,
                            w_max=args.dwbc_w_max,
                            warmup_steps=args.dwbc_warmup_steps,
                            global_step=global_step,
                        ).view(-1)

                state_elem_mask = state_elem_mask.unsqueeze(1)

                loss, info = rdt(
                    lang_tokens=text_embeds,
                    lang_attn_mask=lang_attn_mask,
                    img_tokens=image_embeds,
                    state_tokens=states,
                    action_gt=actions,
                    action_mask=state_elem_mask,
                    ctrl_freqs=ctrl_freqs,
                    sample_weights=sample_weights,
                    return_dict=True,
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(rdt.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            ema_model.step(accelerator.unwrap_model(rdt))

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_period == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    ema_save_path = os.path.join(save_path, "ema")
                    accelerator.save_model(ema_rdt, ema_save_path)
                    logger.info(f"Saved state to {save_path}")

                if args.sample_period > 0 and global_step % args.sample_period == 0:
                    with torch.cuda.amp.autocast(enabled=True):
                        sample_loss_for_log = log_sample_res(
                            text_encoder,
                            vision_encoder,
                            rdt,
                            args,
                            accelerator,
                            weight_dtype,
                            sample_dataset.get_dataset_id2name(),
                            sample_dataloader,
                            logger,
                        )
                    logger.info(sample_loss_for_log)
                    accelerator.log(sample_loss_for_log, step=global_step)

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "mean_weight": info["mean_weight"].item() if torch.is_tensor(info["mean_weight"]) else float(info["mean_weight"]),
                "per_sample_mse_mean": info["per_sample_mse_mean"].item(),
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(rdt).save_pretrained(args.output_dir)
        ema_save_path = os.path.join(args.output_dir, "ema")
        accelerator.save_model(ema_rdt, ema_save_path)
        logger.info(f"Stage-3 finished. Final model saved to {args.output_dir}")

    accelerator.end_training()
