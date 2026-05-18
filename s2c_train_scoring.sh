#!/usr/bin/env bash
# ============================================================================
# Stage 2c-2 — Discriminator PU training (paper §III-C, Eq.(7))
#
# Trains the 30M scoring model that scores each clip in (0, 1]. Positives
# come from the Shigh set produced by ``s2b_replay.sh`` (falling back to
# Spre from ``s2a_analyze_jerk.sh`` when Shigh is missing). The Stage-3
# post-trainer ingests the resulting checkpoint to derive per-sample
# DWBC weights (Eq.(8)).
# ============================================================================
set -Eeuo pipefail

: "${DEXORA_LEROBOT_ROOT:=data/Dexora_Real-World_Dataset/airbot_pick_and_place}"
: "${DEXORA_T5:=google/t5-v1_1-xxl}"
: "${DEXORA_SIGLIP:=google/siglip-so400m-patch14-384}"
: "${DEXORA_STATS:=new_lerobot_stats/dataset_statistics.json}"

: "${CONFIG_PATH:=configs/scoring.yaml}"
: "${OUTPUT_DIR:=checkpoints/dexora-scoring}"
: "${LOGPI_FILE:=runs/logpi/logpi.json}"
: "${SPRE_FILE:=runs/spre/complete_analysis_results.json}"
: "${SHIGH_FILE:=runs/shigh.json}"

: "${NUM_GPUS:=8}"
: "${TRAIN_BATCH_SIZE:=16}"
: "${GRAD_ACCUM:=2}"
: "${MAX_TRAIN_STEPS:=10000}"
: "${CHECKPOINTING_PERIOD:=2000}"
: "${LEARNING_RATE:=5e-5}"
: "${MIXED_PRECISION:=bf16}"
: "${LR_SCHEDULER:=constant}"
: "${ETA:=0.5}"                      # paper Eq.(7) eta
: "${PU_VARIANT:=paper}"             # paper | dwbc
: "${WANDB_PROJECT:=dexora-scoring}"

export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export WANDB_PROJECT
export WANDB_MODE=${WANDB_MODE:-offline}

mkdir -p "$OUTPUT_DIR"
echo "==> Stage-2c-2 discriminator PU training"
echo "    DEXORA_LEROBOT_ROOT  : $DEXORA_LEROBOT_ROOT"
echo "    LOGPI_FILE           : $LOGPI_FILE"
echo "    SPRE / SHIGH         : $SPRE_FILE / $SHIGH_FILE"
echo "    OUTPUT_DIR           : $OUTPUT_DIR"

accelerate launch --num_processes="$NUM_GPUS" --multi_gpu \
    --mixed_precision="$MIXED_PRECISION" \
    -m train.main_scoring \
    --config_path="$CONFIG_PATH" \
    --pretrained_text_encoder_name_or_path="$DEXORA_T5" \
    --pretrained_vision_encoder_name_or_path="$DEXORA_SIGLIP" \
    --output_dir="$OUTPUT_DIR" \
    --load_from=lerobot \
    --lerobot_root="$DEXORA_LEROBOT_ROOT" \
    --stats_file="$DEXORA_STATS" \
    --state_dim_keep=36 \
    --logpi_file="$LOGPI_FILE" \
    --spre_file="$SPRE_FILE" \
    --shigh_file="$SHIGH_FILE" \
    --train_batch_size="$TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps="$GRAD_ACCUM" \
    --max_train_steps="$MAX_TRAIN_STEPS" \
    --checkpointing_period="$CHECKPOINTING_PERIOD" \
    --lr_scheduler="$LR_SCHEDULER" \
    --learning_rate="$LEARNING_RATE" \
    --mixed_precision="$MIXED_PRECISION" \
    --dataloader_num_workers=4 \
    --eta="$ETA" \
    --pu_variant="$PU_VARIANT" \
    --report_to=tensorboard
