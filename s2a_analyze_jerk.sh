#!/usr/bin/env bash
# ============================================================================
# Stage 2a — Pre-screen real demonstrations (Dexora paper §III-C, Eq.(1)-(3))
#
# For every episode in the LeRobot dataset, compute the episode-level
# acceleration / jerk RMS (``Aep`` / ``Jep``) under per-dim min-max
# normalization, then keep
#
#     Spre = Low-20%(Aep) ∩ Low-20%(Jep)
#
# as the input to Stage-2b (``s2b_replay.sh``). The script writes
# ``${SPRE_DIR}/complete_analysis_results.json``.
# ============================================================================
set -Eeuo pipefail

: "${DEXORA_LEROBOT_ROOT:=data/Dexora_Real-World_Dataset/airbot_pick_and_place}"
: "${DEXORA_STATS:=new_lerobot_stats/dataset_statistics.json}"
: "${NUM_EPISODES:=10000}"           # 10000 = "process all" if dataset is smaller
: "${TARGET_RATIO:=0.2}"             # paper: 20% per axis -> ~18% retained
: "${SPRE_DIR:=runs/spre}"
: "${FPS:=20}"                       # paper §III-A logs at 20 Hz
: "${STATE_DIM_KEEP:=36}"

# Auto-generate stats if missing.
if [[ ! -f "$DEXORA_STATS" ]]; then
    echo "==> Stats file $DEXORA_STATS not found; generating from $DEXORA_LEROBOT_ROOT ..."
    mkdir -p "$(dirname "$DEXORA_STATS")"
    python -m data.lerobot_vla_dataset --stat \
        --num_samples 5000 \
        --output_dir "$(dirname "$DEXORA_STATS")" \
        --repo_dir   "$DEXORA_LEROBOT_ROOT"
fi

mkdir -p "$SPRE_DIR"
echo "==> Stage-2a pre-screening (target_ratio=$TARGET_RATIO)"
python scripts/analyze_episode_quality.py "$NUM_EPISODES" \
    --output_dir="$SPRE_DIR" \
    --stats_file="$DEXORA_STATS" \
    --lerobot_root="$DEXORA_LEROBOT_ROOT" \
    --fps="$FPS" \
    --target_ratio="$TARGET_RATIO" \
    --state_dim_keep="$STATE_DIM_KEEP"

echo "==> Spre written to $SPRE_DIR/complete_analysis_results.json"
