#!/usr/bin/env bash
# ============================================================================
# Stage 2b — Replay-based post-validation Spre → Shigh (paper §III-C)
#
# Open-loop replays every Spre episode and keeps only the survivors
# (task completion + collision-free). The paper uses a MuJoCo digital
# twin; the released code ships three verifiers:
#
#   * ``trust_spre`` (default): no verification, every Spre episode passes.
#                                Use for smoke testing / when no simulator
#                                is available yet.
#   * ``energy``: cheap kinematic heuristic (out-of-range states + acc spikes).
#   * ``mujoco``: real MuJoCo replay. Plug in your own twin module via
#                 ``REPLAY_TWIN_MODULE`` (must expose ``replay(states, actions,
#                 task_id) -> {"success": bool, "collision_free": bool}``).
# ============================================================================
set -Eeuo pipefail

: "${DEXORA_LEROBOT_ROOT:=data/Dexora_Real-World_Dataset/airbot_pick_and_place}"
: "${SPRE_DIR:=runs/spre}"
: "${SHIGH_FILE:=runs/shigh.json}"
: "${REPLAY_VERIFIER:=trust_spre}"   # trust_spre | energy | mujoco
: "${REPLAY_TWIN_MODULE:=}"          # only used when REPLAY_VERIFIER=mujoco

extra_args=()
if [[ -n "$REPLAY_TWIN_MODULE" ]]; then
    extra_args+=(--twin_module "$REPLAY_TWIN_MODULE")
fi

mkdir -p "$(dirname "$SHIGH_FILE")"
echo "==> Stage-2b replay verification (verifier=$REPLAY_VERIFIER)"
python scripts/replay_validate.py \
    --pre_screening_file="$SPRE_DIR/complete_analysis_results.json" \
    --lerobot_root="$DEXORA_LEROBOT_ROOT" \
    --output_file="$SHIGH_FILE" \
    --verifier="$REPLAY_VERIFIER" \
    "${extra_args[@]}"

echo "==> Shigh written to $SHIGH_FILE"
