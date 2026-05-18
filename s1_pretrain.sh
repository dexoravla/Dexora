#!/usr/bin/env bash
# ============================================================================
# Stage 1 — Pretrain the Dexora policy (Dexora paper §III-D Stage 1)
#
# Trains the 400M Diffusion-Transformer policy from scratch on the Dexora
# real-world dataset. Replace the dataset path with a subset of
# ``Dexora/Dexora_Real-World_Dataset`` from HuggingFace.
#
#     huggingface-cli download Dexora/Dexora_Real-World_Dataset \
#         --repo-type dataset --local-dir data/Dexora_Real-World_Dataset
#
# Override any of the variables below via env vars, e.g.
#
#     DEXORA_LEROBOT_ROOT=/path/to/airbot_pick_and_place \
#     OUTPUT_DIR=checkpoints/dexora-400m-pretrain \
#         bash s1_pretrain.sh
# ============================================================================
set -Eeuo pipefail

# ---------- Required: dataset + encoders ----------
# Either the per-task LeRobot v2.1 root (e.g. ``data/Dexora_Real-World_Dataset/airbot_pick_and_place``)
# or one of the task families. See README "Dataset layout" for details.
: "${DEXORA_LEROBOT_ROOT:=data/Dexora_Real-World_Dataset/airbot_pick_and_place}"
: "${DEXORA_T5:=google/t5-v1_1-xxl}"
: "${DEXORA_SIGLIP:=google/siglip-so400m-patch14-384}"
# Stats file for per-dim min-max normalization. If missing we auto-generate it
# from the LeRobot root with ``data/lerobot_vla_dataset.py --stat``.
: "${DEXORA_STATS:=new_lerobot_stats/dataset_statistics.json}"

# ---------- Training knobs ----------
: "${CONFIG_PATH:=configs/base_400m.yaml}"
: "${OUTPUT_DIR:=checkpoints/dexora-400m-pretrain}"
: "${NUM_GPUS:=8}"
: "${TRAIN_BATCH_SIZE:=4}"
: "${GRAD_ACCUM:=1}"
: "${MAX_TRAIN_STEPS:=100000}"
: "${CHECKPOINTING_PERIOD:=5000}"
: "${LEARNING_RATE:=1e-4}"
: "${MIXED_PRECISION:=bf16}"
: "${LR_SCHEDULER:=constant}"
: "${DATALOADER_NUM_WORKERS:=4}"
: "${REPORT_TO:=tensorboard}"
: "${WANDB_PROJECT:=dexora}"

export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export WANDB_PROJECT
export WANDB_MODE=${WANDB_MODE:-offline}

mkdir -p "$OUTPUT_DIR"
echo "==> Stage-1 pretrain"
echo "    DEXORA_LEROBOT_ROOT : $DEXORA_LEROBOT_ROOT"
echo "    DEXORA_STATS        : $DEXORA_STATS"
echo "    OUTPUT_DIR          : $OUTPUT_DIR"
echo "    NUM_GPUS / bs       : $NUM_GPUS x $TRAIN_BATCH_SIZE  (grad-accum $GRAD_ACCUM)"
echo "    MAX_TRAIN_STEPS     : $MAX_TRAIN_STEPS"

# Auto-generate stats if missing (one-time, ~2 min)
if [[ ! -f "$DEXORA_STATS" ]]; then
    echo "==> Stats file $DEXORA_STATS not found; generating from $DEXORA_LEROBOT_ROOT ..."
    mkdir -p "$(dirname "$DEXORA_STATS")"
    python -m data.lerobot_vla_dataset --stat \
        --num_samples 5000 \
        --output_dir "$(dirname "$DEXORA_STATS")" \
        --repo_dir   "$DEXORA_LEROBOT_ROOT"
fi

# ----- Launch -----
accelerate launch --num_processes="$NUM_GPUS" --multi_gpu \
    --mixed_precision="$MIXED_PRECISION" \
    -m train.main \
    --config_path="$CONFIG_PATH" \
    --pretrained_text_encoder_name_or_path="$DEXORA_T5" \
    --pretrained_vision_encoder_name_or_path="$DEXORA_SIGLIP" \
    --output_dir="$OUTPUT_DIR" \
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
    --dataloader_num_workers="$DATALOADER_NUM_WORKERS" \
    --state_noise_snr=40 \
    --image_aug \
    --report_to="$REPORT_TO"
