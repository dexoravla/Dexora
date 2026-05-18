#!/usr/bin/env bash
# ============================================================================
# Stage 3 — Data-quality-aware post-training (paper §III-D, Eq.(8))
#
# Starts from the Stage-1 policy and fine-tunes on the real dataset with a
# discriminator-weighted diffusion loss
#
#     L_pi = sum_i w_i * || eps_theta(.) - eps ||^2
#
# where w_i = DWBC(d(xi_i)) is computed online from the frozen Stage-2
# discriminator score. Pass ``--no_quality_weights`` to reproduce the
# vanilla baseline (Tab. III "w/o discriminator" row).
# ============================================================================
set -Eeuo pipefail

: "${DEXORA_LEROBOT_ROOT:=data/Dexora_Real-World_Dataset/airbot_pick_and_place}"
: "${DEXORA_T5:=google/t5-v1_1-xxl}"
: "${DEXORA_SIGLIP:=google/siglip-so400m-patch14-384}"
: "${DEXORA_STATS:=new_lerobot_stats/dataset_statistics.json}"

: "${CONFIG_PATH:=configs/base_400m.yaml}"
: "${STAGE1_CKPT:=checkpoints/dexora-400m-pretrain}"
: "${SCORING_CKPT:=checkpoints/dexora-scoring/final_model/pytorch_model.bin}"
: "${OUTPUT_DIR:=checkpoints/dexora-400m-posttrain}"

: "${NUM_GPUS:=8}"
: "${TRAIN_BATCH_SIZE:=4}"
: "${GRAD_ACCUM:=1}"
: "${MAX_TRAIN_STEPS:=50000}"
: "${CHECKPOINTING_PERIOD:=5000}"
: "${LEARNING_RATE:=5e-5}"
: "${MIXED_PRECISION:=bf16}"
: "${LR_SCHEDULER:=constant}"

: "${DWBC_ETA:=0.5}"
: "${DWBC_W_MIN:=0.0}"
: "${DWBC_W_MAX:=5.0}"
: "${DWBC_WARMUP_STEPS:=1000}"

# ``--real_data_fraction`` reproduces Fig. 10 (0.0 = sim-only, 0.5 = sim+50%real,
# 1.0 = sim+all real; the default 1.0 matches the main paper run).
: "${REAL_DATA_FRACTION:=1.0}"

# Pass any extra flags via $EXTRA_FLAGS, e.g.
#     EXTRA_FLAGS="--no_quality_weights"  bash s3_post_train.sh
: "${EXTRA_FLAGS:=}"

export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export WANDB_PROJECT=${WANDB_PROJECT:-dexora-posttrain}
export WANDB_MODE=${WANDB_MODE:-offline}

mkdir -p "$OUTPUT_DIR"
echo "==> Stage-3 quality-aware post-training"
echo "    DEXORA_LEROBOT_ROOT  : $DEXORA_LEROBOT_ROOT"
echo "    STAGE1_CKPT          : $STAGE1_CKPT"
echo "    SCORING_CKPT         : $SCORING_CKPT"
echo "    OUTPUT_DIR           : $OUTPUT_DIR"
echo "    DWBC eta / w_max     : $DWBC_ETA / $DWBC_W_MAX"
echo "    real_data_fraction   : $REAL_DATA_FRACTION"

accelerate launch --num_processes="$NUM_GPUS" --multi_gpu \
    --mixed_precision="$MIXED_PRECISION" \
    -m train.main_posttrain \
    --config_path="$CONFIG_PATH" \
    --pretrained_text_encoder_name_or_path="$DEXORA_T5" \
    --pretrained_vision_encoder_name_or_path="$DEXORA_SIGLIP" \
    --output_dir="$OUTPUT_DIR" \
    --stage1_ckpt="$STAGE1_CKPT" \
    --scoring_ckpt="$SCORING_CKPT" \
    --dwbc_eta="$DWBC_ETA" \
    --dwbc_w_min="$DWBC_W_MIN" \
    --dwbc_w_max="$DWBC_W_MAX" \
    --dwbc_warmup_steps="$DWBC_WARMUP_STEPS" \
    --real_data_fraction="$REAL_DATA_FRACTION" \
    --load_from=lerobot \
    --lerobot_root="$DEXORA_LEROBOT_ROOT" \
    --stats_file="$DEXORA_STATS" \
    --state_dim_keep=36 \
    --dataset_type=finetune \
    --train_batch_size="$TRAIN_BATCH_SIZE" \
    --sample_batch_size=2 \
    --gradient_accumulation_steps="$GRAD_ACCUM" \
    --max_train_steps="$MAX_TRAIN_STEPS" \
    --checkpointing_period="$CHECKPOINTING_PERIOD" \
    --sample_period=-1 \
    --lr_scheduler="$LR_SCHEDULER" \
    --learning_rate="$LEARNING_RATE" \
    --mixed_precision="$MIXED_PRECISION" \
    --dataloader_num_workers=4 \
    --state_noise_snr=40 \
    --image_aug \
    --report_to=tensorboard \
    $EXTRA_FLAGS
