#!/usr/bin/env bash
# ============================================================================
# Dexora end-to-end three-stage training pipeline.
#
# Chains:
#   1. Stage-1   pretrain on real data                  -> s1_pretrain.sh
#   2. Stage-2a  pre-screening (Aep, Jep) -> Spre       -> s2a_analyze_jerk.sh
#   3. Stage-2b  Spre -> Shigh post-validation          -> s2b_replay.sh
#   4. Stage-2c  log-π proxy (Eq.(5))                   -> s2c_compute_logpi.sh
#   5. Stage-2c  discriminator PU training (Eq.(7))     -> s2c_train_scoring.sh
#   6. Stage-3   quality-aware post-training (Eq.(8))   -> s3_post_train.sh
#
# Override stages with the START_STAGE / END_STAGE env vars, e.g.
#
#     START_STAGE=4 END_STAGE=6 bash run_all_stages.sh
#
# Override paths via env vars (mirrored in each stage script):
#
#     DEXORA_LEROBOT_ROOT=/path/to/airbot_pick_and_place \
#     RUN_DIR=./runs/dexora-paper-replication \
#         bash run_all_stages.sh
# ============================================================================
set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Where everything lives (each variable also matches the defaults inside the
# individual s*_*.sh scripts so partial reruns work the same way).
# ---------------------------------------------------------------------------
START_STAGE="${START_STAGE:-1}"
END_STAGE="${END_STAGE:-6}"
RUN_DIR="${RUN_DIR:-./runs/dexora-$(date +%Y%m%d-%H%M%S)}"

export DEXORA_LEROBOT_ROOT="${DEXORA_LEROBOT_ROOT:-data/Dexora_Real-World_Dataset/airbot_pick_and_place}"
export DEXORA_T5="${DEXORA_T5:-google/t5-v1_1-xxl}"
export DEXORA_SIGLIP="${DEXORA_SIGLIP:-google/siglip-so400m-patch14-384}"
export DEXORA_STATS="${DEXORA_STATS:-new_lerobot_stats/dataset_statistics.json}"

export OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/stage1-pretrain}"
export STAGE1_CKPT="${STAGE1_CKPT:-$RUN_DIR/stage1-pretrain}"
export SPRE_DIR="${SPRE_DIR:-$RUN_DIR/spre}"
export SHIGH_FILE="${SHIGH_FILE:-$RUN_DIR/shigh.json}"
export LOGPI_FILE="${LOGPI_FILE:-$RUN_DIR/logpi/logpi.json}"
export SCORING_OUT="${SCORING_OUT:-$RUN_DIR/stage2-scoring}"
export SCORING_CKPT="${SCORING_CKPT:-$SCORING_OUT/final_model/pytorch_model.bin}"
export STAGE3_OUT="${STAGE3_OUT:-$RUN_DIR/stage3-posttrain}"

export REPLAY_VERIFIER="${REPLAY_VERIFIER:-trust_spre}"

mkdir -p "$RUN_DIR"
echo "==> RUN_DIR=$RUN_DIR"
echo "==> Stages: $START_STAGE..$END_STAGE"

run_stage() {
    local n="$1" name="$2"
    if (( n < START_STAGE || n > END_STAGE )); then
        echo "==> [skip ] stage $n: $name"
        return 1
    fi
    echo
    echo "================================================================"
    echo "==> [stage] $n: $name"
    echo "================================================================"
    return 0
}

# ---------------------------------------------------------------------------
# 1. Stage-1: pretrain
# ---------------------------------------------------------------------------
if run_stage 1 "Stage-1 pretrain (s1_pretrain.sh)"; then
    OUTPUT_DIR="$STAGE1_CKPT" bash s1_pretrain.sh
fi

# ---------------------------------------------------------------------------
# 2. Stage-2a: pre-screening
# ---------------------------------------------------------------------------
if run_stage 2 "Stage-2a pre-screening (s2a_analyze_jerk.sh)"; then
    SPRE_DIR="$SPRE_DIR" bash s2a_analyze_jerk.sh
fi

# ---------------------------------------------------------------------------
# 3. Stage-2b: replay post-validation
# ---------------------------------------------------------------------------
if run_stage 3 "Stage-2b replay verification (s2b_replay.sh)"; then
    SPRE_DIR="$SPRE_DIR" SHIGH_FILE="$SHIGH_FILE" REPLAY_VERIFIER="$REPLAY_VERIFIER" \
        bash s2b_replay.sh
fi

# ---------------------------------------------------------------------------
# 4. Stage-2c-1: log-π proxy
# ---------------------------------------------------------------------------
if run_stage 4 "Stage-2c-1 log-π proxy (s2c_compute_logpi.sh)"; then
    STAGE1_CKPT="$STAGE1_CKPT" LOGPI_FILE="$LOGPI_FILE" \
        bash s2c_compute_logpi.sh
fi

# ---------------------------------------------------------------------------
# 5. Stage-2c-2: discriminator PU training
# ---------------------------------------------------------------------------
if run_stage 5 "Stage-2c-2 discriminator PU training (s2c_train_scoring.sh)"; then
    OUTPUT_DIR="$SCORING_OUT" LOGPI_FILE="$LOGPI_FILE" \
    SPRE_FILE="$SPRE_DIR/complete_analysis_results.json" SHIGH_FILE="$SHIGH_FILE" \
        bash s2c_train_scoring.sh
fi

# ---------------------------------------------------------------------------
# 6. Stage-3: quality-aware post-training
# ---------------------------------------------------------------------------
if run_stage 6 "Stage-3 quality-aware post-training (s3_post_train.sh)"; then
    STAGE1_CKPT="$STAGE1_CKPT" SCORING_CKPT="$SCORING_CKPT" \
    OUTPUT_DIR="$STAGE3_OUT" \
        bash s3_post_train.sh
fi

echo
echo "==> All requested stages finished. Artifacts in $RUN_DIR"
