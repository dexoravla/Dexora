#!/usr/bin/env bash
# ============================================================================
# Dexora real-robot inference launcher.
#
# Brings up the three processes that make a full Dexora rollout:
#
#   1. xhand_forwarder  (in xhand_tele_env conda env, ZMQ tcp://*:5557)
#   2. mmk_forwarder    (in imitall      conda env, ZMQ tcp://*:5556)
#   3. dexora_inference (in dexora       conda env, GPU, talks to 1 + 2)
#
# This is just a convenience wrapper around three ``conda activate && python``
# calls; you can also run each command by hand in three separate terminals
# (recommended the first time you deploy on a new robot, so you can see each
# process's startup logs side-by-side).
# ============================================================================
set -Eeuo pipefail

# ---------- paths to override per host ----------
: "${DEPLOY_DIR:=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
: "${REPO_DIR:=$(cd "$DEPLOY_DIR/.." && pwd)}"

: "${CONFIG_PATH:=$DEPLOY_DIR/mmk_xhand_config.yaml}"
: "${MODEL_PATH:=$REPO_DIR/checkpoints/dexora-400m-posttrain}"
: "${MODEL_CONFIG_PATH:=$REPO_DIR/configs/base_400m.yaml}"
: "${TASK_DESCRIPTION:?TASK_DESCRIPTION env var is required, e.g. \"Pick the apple and put it on the plate.\"}"

# Three conda envs (override CONDA_BASE if your install isn't /opt/conda).
: "${CONDA_BASE:=$HOME/miniconda3}"
: "${XHAND_ENV:=xhand_tele_env}"
: "${MMK_ENV:=imitall}"
: "${DEXORA_ENV:=dexora}"

: "${LOG_DIR:=$REPO_DIR/logs/deploy-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$LOG_DIR"
echo "==> logs -> $LOG_DIR"

source "$CONDA_BASE/etc/profile.d/conda.sh"

# ---------- 1. XHand forwarder (background) ----------
echo "==> Starting XHand forwarder ($XHAND_ENV) ..."
conda activate "$XHAND_ENV"
python "$DEPLOY_DIR/xhand_forwarder.py" --config "$CONFIG_PATH" \
    > "$LOG_DIR/xhand_forwarder.log" 2>&1 &
XHAND_PID=$!
conda deactivate
sleep 2

# ---------- 2. MMK forwarder (background) ----------
echo "==> Starting MMK forwarder ($MMK_ENV) ..."
conda activate "$MMK_ENV"
python "$DEPLOY_DIR/mmk_forwarder.py" --config "$CONFIG_PATH" \
    > "$LOG_DIR/mmk_forwarder.log" 2>&1 &
MMK_PID=$!
conda deactivate
sleep 2

# Cleanup forwarders on Ctrl-C / exit so we don't leave RS485 buses hung.
trap 'echo "==> Cleaning up forwarders ..."; kill ${XHAND_PID} ${MMK_PID} 2>/dev/null || true' EXIT

# ---------- 3. Dexora inference (foreground) ----------
echo "==> Launching Dexora policy ($DEXORA_ENV) ..."
conda activate "$DEXORA_ENV"
python "$DEPLOY_DIR/dexora_inference_zmq.py" \
    --model-path "$MODEL_PATH" \
    --config-path "$CONFIG_PATH" \
    --model-config-path "$MODEL_CONFIG_PATH" \
    --task-description "$TASK_DESCRIPTION" \
    --save-logs --log-dir "$LOG_DIR" \
    --monitor-interval 1 \
    "$@"
