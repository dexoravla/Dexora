#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Example launcher for ``scripts/eval_action_curves.py`` (paper Fig. 11).
#
# Runs open-loop inference on a single episode from a LeRobot v2.1 dataset
# (the HuggingFace release of ``Dexora_Real-World_Dataset``) and dumps
# per-joint GT-vs-prediction plots under ``${OUTPUT_DIR}``.
#
# Override any of the variables below via the env, e.g.
#   MODEL_PATH=checkpoints/dexora-400m-posttrain \
#   REPO_DIR=data/Dexora_Real-World_Dataset/airbot_dexterous \
#   EPISODE_IDX=12 bash scripts/run_eval_example.sh
# -----------------------------------------------------------------------------
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ---- Required paths ---------------------------------------------------------
MODEL_PATH=${MODEL_PATH:-"checkpoints/dexora-400m-posttrain"}
REPO_DIR=${REPO_DIR:-"data/Dexora_Real-World_Dataset/airbot_pick_and_place"}
STATS_FILE=${STATS_FILE:-"new_lerobot_stats/dataset_statistics.json"}
MODEL_CONFIG=${MODEL_CONFIG:-"configs/base_400m.yaml"}

# ---- Encoders (override only if you renamed the default download locations) -
TEXT_ENCODER=${TEXT_ENCODER:-"google/t5-v1_1-xxl"}
VISION_ENCODER=${VISION_ENCODER:-"google/siglip-so400m-patch14-384"}

# ---- Eval knobs -------------------------------------------------------------
EPISODE_IDX=${EPISODE_IDX:-0}
INFERENCE_INTERVAL=${INFERENCE_INTERVAL:-32}      # one pass per chunk_size
MAX_STEPS=${MAX_STEPS:-}                          # leave empty to use full ep
INSTRUCTION=${INSTRUCTION:-}                      # leave empty to use dataset's
OUTPUT_DIR=${OUTPUT_DIR:-"eval_results/ep${EPISODE_IDX}"}

mkdir -p "${OUTPUT_DIR}"

EXTRA_FLAGS=()
if [[ -n "${MAX_STEPS}" ]]; then
    EXTRA_FLAGS+=(--max-steps "${MAX_STEPS}")
fi
if [[ -n "${INSTRUCTION}" ]]; then
    EXTRA_FLAGS+=(--instruction "${INSTRUCTION}")
fi
if [[ "${DUMP_JSON:-0}" == "1" ]]; then
    EXTRA_FLAGS+=(--dump-json)
fi

python scripts/eval_action_curves.py \
    --model-path        "${MODEL_PATH}" \
    --model-config-path "${MODEL_CONFIG}" \
    --text-encoder      "${TEXT_ENCODER}" \
    --vision-encoder    "${VISION_ENCODER}" \
    --repo-dir          "${REPO_DIR}" \
    --stats-file        "${STATS_FILE}" \
    --episode-idx       "${EPISODE_IDX}" \
    --inference-interval "${INFERENCE_INTERVAL}" \
    --output-dir        "${OUTPUT_DIR}" \
    "${EXTRA_FLAGS[@]}"

echo "Evaluation completed. Plots written to ${OUTPUT_DIR}/"
