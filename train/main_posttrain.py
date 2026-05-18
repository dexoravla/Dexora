"""Entry point for Dexora stage-3 (data-quality-aware post-training)."""
import argparse
import os

from accelerate.logging import get_logger

from train.train_posttrain import train_posttrain


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Stage-3: data-quality-aware post-training of the Dexora policy."
    )

    # --- Config / encoders / outputs ---
    parser.add_argument("--config_path", type=str, default="configs/base_400m.yaml")
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--pretrained_text_encoder_name_or_path", type=str, default=None)
    parser.add_argument("--pretrained_vision_encoder_name_or_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="checkpoints/dexora-posttrain")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--seed", type=int, default=None)

    # --- Stage-3 specific ---
    parser.add_argument("--stage1_ckpt", type=str, required=True,
                        help="Path to stage-1 policy checkpoint (directory or .bin/.pt).")
    parser.add_argument("--scoring_ckpt", type=str, default=None,
                        help="Path to stage-2 discriminator checkpoint. "
                             "Required unless --no_quality_weights is set.")
    parser.add_argument("--dwbc_eta", type=float, default=0.5,
                        help="DWBC mapping eta (paper uses 0.5).")
    parser.add_argument("--dwbc_w_min", type=float, default=0.0,
                        help="Lower clamp for per-sample weights.")
    parser.add_argument("--dwbc_w_max", type=float, default=5.0,
                        help="Upper clamp for per-sample weights.")
    parser.add_argument("--dwbc_warmup_steps", type=int, default=1000,
                        help="Linear weight warm-up: 0 means use w_i from step 0.")
    parser.add_argument("--no_quality_weights", action="store_true",
                        help="Disable discriminator weighting (ablation: 'vanilla post-training' "
                             "row in Table III). Per-sample weights are fixed to 1.0.")
    parser.add_argument("--real_data_fraction", type=float, default=1.0,
                        help="Fraction of the real dataset to keep during post-training "
                             "(ablation Fig. 10: 0.0 = sim-only, 0.5 = sim + 50%% real, "
                             "1.0 = sim + all real). Implemented as a deterministic per-epoch "
                             "stride on the LeRobot/BSON dataset's __len__.")

    # --- Dataset / batch ---
    parser.add_argument("--load_from", type=str, default="lerobot",
                        choices=["hdf5", "bson", "egodex", "lerobot"])
    parser.add_argument("--lerobot_root", type=str, default=None,
                        help="Path to LeRobot v2.1 dataset root "
                             "(required when --load_from=lerobot).")
    parser.add_argument("--bson_root", type=str, default=None,
                        help="Path to BSON dataset root (legacy).")
    parser.add_argument("--stats_file", type=str, default=None,
                        help="Path to dataset_statistics.json for per-dim min-max "
                             "normalization (see `data/lerobot_vla_dataset.py --stat`).")
    parser.add_argument("--state_dim_keep", type=int, default=36,
                        help="Slice state/action to first N dims; default 36 matches "
                             "the paper layout (HF release stores 39 dims).")
    parser.add_argument("--dataset_type", type=str, default="finetune",
                        help="finetune or pretrain (passed to VLAConsumerDataset).")
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--sample_batch_size", type=int, default=4)
    parser.add_argument("--num_sample_batches", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)

    # --- Optimization ---
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--scale_lr", action="store_true")
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--set_grads_to_none", action="store_true")

    # --- Checkpointing ---
    parser.add_argument("--checkpointing_period", type=int, default=5000)
    parser.add_argument("--sample_period", type=int, default=-1)
    parser.add_argument("--checkpoints_total_limit", type=int, default=20)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    # --- Augmentation / masking ---
    parser.add_argument("--image_aug", action="store_true")
    parser.add_argument("--precomp_lang_embed", action="store_true")
    parser.add_argument("--cond_mask_prob", type=float, default=0.1)
    parser.add_argument("--cam_ext_mask_prob", type=float, default=-1.0)
    parser.add_argument("--state_noise_snr", type=float, default=None)

    parser.add_argument("--local_rank", type=int, default=-1)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main():
    """CLI entry-point used by ``pyproject.toml``'s ``dexora-posttrain`` script."""
    logger = get_logger(__name__)
    args = parse_args()
    if (not args.no_quality_weights) and (args.scoring_ckpt is None):
        raise SystemExit(
            "ERROR: --scoring_ckpt is required for data-quality-aware post-training. "
            "Pass --no_quality_weights to run the vanilla ablation instead."
        )
    train_posttrain(args, logger)


if __name__ == "__main__":
    main()
