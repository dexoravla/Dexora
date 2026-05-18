#!/usr/bin/env bash
# ============================================================================
# Stage 2c-1 — log-π proxy from Stage-1 policy (paper §III-C, Eq.(4)-(5))
#
# For each frame in the dataset, computes the denoising-residual energy E_t
# under the Stage-1 policy, then writes \hat{logπ}_t = -zscore(E_t) to
# ``$LOGPI_FILE``. The discriminator (Stage-2c-2) consumes this file.
# ============================================================================
set -Eeuo pipefail

: "${DEXORA_LEROBOT_ROOT:=data/Dexora_Real-World_Dataset/airbot_pick_and_place}"
: "${DEXORA_T5:=google/t5-v1_1-xxl}"
: "${DEXORA_SIGLIP:=google/siglip-so400m-patch14-384}"
: "${DEXORA_STATS:=new_lerobot_stats/dataset_statistics.json}"
: "${CONFIG_PATH:=configs/base_400m.yaml}"
: "${STAGE1_CKPT:=checkpoints/dexora-400m-pretrain}"
: "${LOGPI_FILE:=runs/logpi/logpi.json}"
: "${BATCH_SIZE:=8}"
: "${NUM_NOISE_STEPS:=4}"
: "${FRAME_STRIDE:=10}"
: "${MAX_EPISODES:=-1}"              # -1 = all
: "${STATE_DIM_KEEP:=36}"
: "${CUDA_VISIBLE_DEVICES:=0}"

export CUDA_VISIBLE_DEVICES
mkdir -p "$(dirname "$LOGPI_FILE")"

echo "==> Stage-2c-1 log-π proxy"
echo "    STAGE1_CKPT          : $STAGE1_CKPT"
echo "    DEXORA_LEROBOT_ROOT  : $DEXORA_LEROBOT_ROOT"
echo "    LOGPI_FILE           : $LOGPI_FILE"

python scripts/compute_logpi.py \
    --config_path="$CONFIG_PATH" \
    --model_path="$STAGE1_CKPT" \
    --dataset_path="$DEXORA_LEROBOT_ROOT" \
    --load_from=lerobot \
    --stats_file="$DEXORA_STATS" \
    --state_dim_keep="$STATE_DIM_KEEP" \
    --output_file="$LOGPI_FILE" \
    --pretrained_text_encoder_name_or_path="$DEXORA_T5" \
    --pretrained_vision_encoder_name_or_path="$DEXORA_SIGLIP" \
    --batch_size="$BATCH_SIZE" \
    --num_noise_steps="$NUM_NOISE_STEPS" \
    --frame_stride="$FRAME_STRIDE" \
    --max_episodes="$MAX_EPISODES" \
    --normalize_mode=zscore

echo "==> log-π written to $LOGPI_FILE (raw E to ${LOGPI_FILE%.json}_raw_E.json)"
