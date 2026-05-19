<p align="center">
  <h1 align="center">Dexora: Open-Source VLA for High-DoF Bimanual Dexterity</h1>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.18722"><img src="https://img.shields.io/badge/arXiv-2605.18722-B31B1B.svg" alt="arXiv"></a>
  <a href="https://dexoravla.github.io"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"></a>
  <a href="https://huggingface.co/Dexora"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Hugging%20Face-yellow.svg" alt="Dataset"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
</p>

<p align="center">
  <i>Dexora is a Vision–Language–Action (VLA) system for <b>dual-arm, dual-hand, 36-DoF dexterous manipulation</b>,<br>
  accepted at <b>ICRA 2026</b> (<a href="ICRA26_0209_FI.pdf">paper PDF</a>).
  This repository releases the full <b>training</b>, <b>inference</b>, <b>data-processing</b> and <b>teleoperation</b> code.</i>
</p>

---

## 🔥 News & Updates

- **2026-05** — Public source release: training pipeline, real-robot inference stack, BSON → LeRobot v2.1 converters, Vision-Pro teleoperation tools.
- **2025-12-12** — Released the **task-level** view of the real-world dataset (one folder per high-level task) on [Hugging Face](https://huggingface.co/datasets/Dexora/Dexora_Real-World_Dataset).
- **2025-12-03** — Released the full **Real-World Dataset** (**12.2K episodes / 2.92M frames / 40.5 h**) on [Hugging Face](https://huggingface.co/datasets/Dexora/Dexora_Real-World_Dataset).

---

## ✨ Highlights

- **Hybrid teleoperation.** Gross arm kinematics from a custom exoskeleton backpack are combined with fine finger motion from markerless Apple Vision Pro tracking, driving both the physical platform and a MuJoCo digital twin.
- **Embodiment-matched corpus.** 100 K simulated trajectories and 12.2 K real-world teleoperated episodes share the same 36-DoF dual-arm dual-hand embodiment and the LeRobot v2.1 schema.
- **Quality-aware post-training.** A lightweight discriminator scores each demonstration clip, and the Diffusion Transformer policy is post-trained with a weighted denoising loss that down-weights low-quality demonstrations.
- **Production-ready inference stack.** A 3-process ZMQ split (policy / arms / hands) cleanly isolates the conflicting Python environments required by the GPU policy, the AIRBOT SDK, and the XHAND SDK.

---

## 📦 Repository Layout

```
Dexora-VLA/
├── configs/                       # YAML / JSON training configurations
├── models/                        # Diffusion-Transformer policy + discriminator
├── train/                         # Pretrain / discriminator / post-train entry points
├── data/                          # LeRobot v2.1 + legacy BSON / HDF5 adapters
├── scripts/                       # Pre-screening, log-π proxy, smoothness / open-loop eval
├── dataprocess/                   # BSON → LeRobot v2.1 conversion utilities
├── teleop/                        # Real-robot data collection + Vision-Pro teleoperation
├── deploy/                        # Real-robot inference (ZMQ split: policy / arms / hands)
├── tests/                         # CPU-only pytest suite
├── google/                        # SigLIP / T5 download targets
├── new_lerobot_stats/             # Per-dim min/max statistics
├── s{1,2a,2b,2c,3}_*.sh           # Per-stage launchers
├── run_all_stages.sh              # End-to-end pipeline driver
├── pyproject.toml / requirements*.txt
├── ICRA26_0209_FI.pdf             # ICRA 2026 paper
└── LICENSE / CITATION.cff / CONTRIBUTING.md / CODE_OF_CONDUCT.md
```

A more detailed breakdown of each top-level package is available in the corresponding sub-`README.md` files (e.g. [`deploy/README.md`](deploy/README.md), [`teleop/README.md`](teleop/README.md), [`dataprocess/README.md`](dataprocess/README.md)).

---

## 🛠️ Installation

```bash
# 1. Conda env (Python 3.10 is required)
conda create -n dexora python=3.10 -y
conda activate dexora

# 2. PyTorch — pick your own CUDA from pytorch.org (CUDA 12.1 example)
pip install torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 3. Project dependencies (see requirements.txt for the pin list)
pip install -r requirements.txt

# 4. Editable install — registers `configs` / `data` / `models` / `train`
#    as importable packages and adds the `dexora-train` console scripts.
pip install -e .

# 5. (Optional) developer tooling
pip install -r requirements-dev.txt
pre-commit install
pytest tests/ -q                # 57 CPU-only tests, ~5 s

# 6. (Optional) flash-attn — pure speed knob; the attention path falls back
#    to PyTorch SDPA if this is absent.
# pip install flash-attn --no-build-isolation
```

> **Why pinned versions?** We pin `transformers<5`, `huggingface_hub<0.26`, `diffusers<0.32`, `accelerate<1.0`, `lerobot<0.4` and `numpy<2.0`. Newer versions break the `is_offline_mode` / LeRobot v2.1 / `imgaug` interfaces that the training stack relies on.

---

## 📥 Data & Pretrained Encoders

### Real-World Dataset

The Dexora real-world dataset is hosted on Hugging Face in the LeRobot v2.1 standard:

```bash
huggingface-cli download Dexora/Dexora_Real-World_Dataset \
    --repo-type dataset \
    --local-dir data/Dexora_Real-World_Dataset
```

Total ≈ 240 GB. The four task families are released as separate LeRobot v2.1 datasets so you can start with whichever subset is most relevant:

```
data/Dexora_Real-World_Dataset/
├── airbot_pick_and_place/
├── airbot_assemble/
├── airbot_articulation/
└── airbot_dexterous/
    ├── data/    chunk-000/episode_000000.parquet ...
    ├── videos/  chunk-000/observation.images.{top,wrist_left,wrist_right,front}/episode_000000.mp4
    └── meta/    info.json  episodes.jsonl  tasks.jsonl  modality.json  stats.json
```

> **State / action dimensions.** The HF release stores **39-D** state and action vectors. The last three dimensions (`head_joint_1`, `head_joint_2`, `spine_joint`) are fixed values required by the AIRBOT SDK but are *not* modelled by the Dexora policy. The training loaders slice to the first **36** dims by default: `[left_arm(6) | right_arm(6) | left_hand(12) | right_hand(12)]`. Pass `--state_dim_keep 0` to retain the full 39 dims.

### Pretrained Encoders

| Asset | Size | Default path |
|---|---|---|
| SigLIP-SO400M (vision) | ~3.7 GB | `google/siglip-so400m-patch14-384/` |
| T5-v1.1-XXL (language) | ~44 GB  | `google/t5-v1_1-xxl/` |

```bash
huggingface-cli download google/siglip-so400m-patch14-384 \
    --local-dir google/siglip-so400m-patch14-384 --local-dir-use-symlinks False
huggingface-cli download google/t5-v1_1-xxl \
    --local-dir google/t5-v1_1-xxl              --local-dir-use-symlinks False
```

See [`google/README.md`](google/README.md) for symlink shortcuts when these encoders already live elsewhere on disk.

### Dataset Statistics (per-dim min–max)

`dataset_statistics.json` is **not** included in the HF release because it depends on which subset you train on. The shell launchers below auto-generate it once if missing; alternatively, pre-compute it explicitly:

```bash
python -m data.lerobot_vla_dataset --stat \
    --num_samples 5000 \
    --repo_dir   data/Dexora_Real-World_Dataset/airbot_pick_and_place \
    --output_dir new_lerobot_stats
```

This writes a 36-D `new_lerobot_stats/dataset_statistics.json` plus `state_distributions.png` / `action_distributions.png` for a quick sanity check. See [`new_lerobot_stats/README.md`](new_lerobot_stats/README.md).

---

## 🚀 Training Pipeline

The training procedure has three stages: **(1)** pretrain the policy, **(2)** train a quality discriminator that scores demonstrations, **(3)** fine-tune the policy with the discriminator-derived per-sample weights. Each stage is launched by a single shell script that reads its inputs from environment variables (with sensible defaults).

```bash
# Shared inputs for all stages (override via env vars as needed).
export DEXORA_LEROBOT_ROOT=data/Dexora_Real-World_Dataset/airbot_pick_and_place
export DEXORA_T5=google/t5-v1_1-xxl
export DEXORA_SIGLIP=google/siglip-so400m-patch14-384
export DEXORA_STATS=new_lerobot_stats/dataset_statistics.json
```

### Stage 1 — Policy pretraining

Trains the 400 M Diffusion Transformer policy for 100 K steps on the real corpus. Swap `DEXORA_LEROBOT_ROOT` for the simulation corpus to reproduce the sim-pretrain variant.

```bash
NUM_GPUS=8 MAX_TRAIN_STEPS=100000 \
OUTPUT_DIR=checkpoints/dexora-400m-pretrain \
    bash s1_pretrain.sh
```

Outputs land under `checkpoints/dexora-400m-pretrain/checkpoint-*/{pytorch_model.bin,config.json,ema/}`.

### Stage 2 — Quality discriminator

The discriminator turns each demonstration clip into a scalar quality score. It is trained in three sub-steps:

**2a · Pre-screening.** Compute per-episode normalized acceleration and jerk, then keep the intersection of the lowest 20 % on both metrics as a high-quality candidate set `S_pre` (≈ 18 % of episodes).

```bash
SPRE_DIR=runs/spre bash s2a_analyze_jerk.sh
# → runs/spre/complete_analysis_results.json
```

**2b · Replay-based post-validation.** Open-loop replay each candidate episode in the MuJoCo digital twin and keep the survivors that complete the task without collisions, yielding `S_high`.

```bash
SPRE_DIR=runs/spre SHIGH_FILE=runs/shigh.json \
REPLAY_VERIFIER=trust_spre \
    bash s2b_replay.sh
```

The bundled `--verifier trust_spre` is a stub that accepts every `S_pre` episode (smoke test). Switch to `--verifier energy` for a cheap kinematic heuristic, or to `--verifier mujoco --twin_module path.to.your.replay` to plug in the real MuJoCo replay. The plug-in module must expose `replay(states, actions, task_id) -> {"success": bool, "collision_free": bool}`.

**2c · Log-π proxy + discriminator training.** A per-chunk action-energy proxy `logπ̂_t = -zscore(E_t)` is computed from the Stage-1 checkpoint, then a small PU-loss discriminator is trained to distinguish `S_high` from the rest.

```bash
# (i) log-π proxy
STAGE1_CKPT=checkpoints/dexora-400m-pretrain \
LOGPI_FILE=runs/logpi/logpi.json \
    bash s2c_compute_logpi.sh

# (ii) discriminator
OUTPUT_DIR=checkpoints/dexora-scoring \
LOGPI_FILE=runs/logpi/logpi.json \
SPRE_FILE=runs/spre/complete_analysis_results.json \
SHIGH_FILE=runs/shigh.json \
    bash s2c_train_scoring.sh
```

The discriminator (`models/scoring_model.py`) ingests the scalar `logπ̂_t` through a small sinusoidal positional-style encoding (8 frequency bands + raw) before the linear projection. This is mathematically equivalent in capacity to `Linear(1 → hidden_size)` but more numerically robust under bf16 when the z-scored proxy sits near zero.

### Stage 3 — Quality-aware post-training

Loads the Stage-1 policy and the frozen Stage-2 discriminator, then fine-tunes the policy on the real corpus with a per-sample weighted denoising loss

$$\mathcal{L}_\pi \;=\; \sum_{i=1}^{L} w_i \, \lVert\, \varepsilon_\theta(\cdot) - \varepsilon \,\rVert_2^2,$$

where the per-sample weight `w_i` is produced online from the discriminator score via a DWBC-style mapping (with a short linear warm-up).

```bash
STAGE1_CKPT=checkpoints/dexora-400m-pretrain \
SCORING_CKPT=checkpoints/dexora-scoring/final_model/pytorch_model.bin \
OUTPUT_DIR=checkpoints/dexora-400m-posttrain \
    bash s3_post_train.sh
```

To reproduce the *no-discriminator* baseline, pass `EXTRA_FLAGS="--no_quality_weights"`.

### End-to-end pipeline

```bash
RUN_DIR=./runs/dexora-paper-rep \
DEXORA_LEROBOT_ROOT=data/Dexora_Real-World_Dataset/airbot_pick_and_place \
    bash run_all_stages.sh

# Chain a subset of stages with START_STAGE / END_STAGE, e.g.
# START_STAGE=4 END_STAGE=6 RUN_DIR=./runs/... bash run_all_stages.sh
```

---

## 🤖 Real-Robot Deployment

`deploy/` runs a trained Dexora policy on the physical robot. The integration is split into three single-purpose processes that talk over loopback ZMQ, so the conflicting Python environments for the policy (GPU + `torch`), the arms SDK (`airbot_py`) and the hands SDK (`xhand_tele_ops`, Python 3.8) can coexist without dependency conflicts:

```
+-----------------------------+   ZMQ tcp://*:5556    +------------------------+
| dexora_inference_zmq.py     | <------------------>  | mmk_forwarder.py       |
| (env: dexora, GPU)          |  arms, 12-D radians   | (env: imitall, 3.10)   |
|                             |   ZMQ tcp://*:5557    +------------------------+
|                             | <------------------>  | xhand_forwarder.py     |
|                             |  hands, 2×12-D rad    | (env: xhand_tele_env)  |
+-----------------------------+                       +------------------------+
```

`deploy/dexora_policy.py` wraps `RDTRunner.from_pretrained(...)` plus SigLIP-SO400M and T5-XXL into a single `policy.get_action(obs) -> [L, 36]` call. The inference loop follows a chunk-and-replay scheme: every `chunk_size` (= L) control ticks we sample a length-L action sequence and play it back with `action_buffer[t % L]`.

### Quick start (three terminals)

```bash
# Terminal A — XHand forwarder (env: xhand_tele_env, Python 3.8)
conda activate xhand_tele_env
python deploy/xhand_forwarder.py --config deploy/mmk_xhand_config.yaml

# Terminal B — MMK forwarder (env: imitall, Python 3.10)
conda activate imitall
python deploy/mmk_forwarder.py   --config deploy/mmk_xhand_config.yaml

# Terminal C — Dexora policy (env: dexora, GPU)
conda activate dexora
python deploy/dexora_inference_zmq.py \
    --model-path checkpoints/dexora-400m-posttrain \
    --config-path deploy/mmk_xhand_config.yaml \
    --task-description "Pick the apple and put it on the plate." \
    --save-logs --monitor-interval 1
```

### One-shell mode

```bash
TASK_DESCRIPTION="Pick the apple and put it on the plate." \
MODEL_PATH=checkpoints/dexora-400m-posttrain \
    bash deploy/inference.sh
```

Wire protocol, joint limits, RealSense fallback, and the full troubleshooting checklist are documented in [`deploy/README.md`](deploy/README.md).

> **Noise schedule and inference steps.** Training uses a 1000-step DDPM forward process with a cosine `squaredcos_cap_v2` beta schedule, predicting the action noise `ε̂_θ`. At inference we swap DDPM for **DPMSolver++** and run only `num_inference_timesteps = 5` solver steps. Increasing this to 10–20 marginally improves smoothness on dexterous tasks at a proportional latency cost.
>
> **Backward compatibility.** Earlier Dexora checkpoints were saved with `prediction_type=sample`. `RDTRunner.compute_loss` and `scripts/compute_logpi.py` both still handle the `sample` branch even though new training defaults to `epsilon`.

---

## 📊 Open-Loop Evaluation

`scripts/eval_action_curves.py` reproduces per-joint trajectory plots from a single LeRobot v2.1 episode. It triggers one diffusion pass every `--inference-interval` steps, then overlays the predicted action chunks on the ground-truth trajectory for every one of the 36 controlled joints (plus a 6 × 6 summary grid).

This is the **open-loop** protocol — we always condition on the ground-truth observation at each sampled timestep, never on the policy's own previous prediction. It is the cheapest sanity check that a trained checkpoint is producing physically plausible chunks before committing to a closed-loop rollout on the real robot.

Under the hood the script reuses the same `deploy/dexora_policy.py` wrapper as the on-robot inference loop, so the prediction path is bit-identical to what the robot would receive at runtime. Inputs (state, action) are normalized with the same `dataset_statistics.json` the policy was trained on, ensuring an apples-to-apples comparison.

```bash
MODEL_PATH=checkpoints/dexora-400m-posttrain \
REPO_DIR=data/Dexora_Real-World_Dataset/airbot_pick_and_place \
STATS_FILE=new_lerobot_stats/dataset_statistics.json \
EPISODE_IDX=0 INFERENCE_INTERVAL=32 \
OUTPUT_DIR=eval_results/airbot_pick_and_place_ep0 \
    bash scripts/run_eval_example.sh
```

Outputs 36 per-axis PNGs (`ep000000_axis_<i>_<joint_name>.png`) plus one `ep000000_summary.png` grid under `${OUTPUT_DIR}`.

<details>
<summary><b>Useful knobs</b></summary>

| Flag / env var          | Meaning |
|---|---|
| `--inference-interval`  | Cadence between diffusion passes; defaults to `chunk_size = 32`, i.e. non-overlapping chunks. Use `16` to visualize chunk consistency on overlap. |
| `--max-steps`           | Truncate to the first N steps of the episode (default: full episode). |
| `--instruction "..."`   | Override the dataset-derived language goal (default: read from `tasks.jsonl`). |
| `--state-dim 39`        | Keep the full 39-D AIRBOT state instead of slicing to the 36-D modelled layout. |
| `--no-normalize`        | Disable per-dim normalization (legacy checkpoints trained without `stats_file`). |
| `--dump-json`           | Also dump GT + predictions as a JSON consumable by `scripts/eval_smoothness.py`. |

</details>

> **Heads up.** The script needs the policy, SigLIP-SO400M, T5-v1.1-XXL *and* the LeRobot dataset all visible at the same time, so peak GPU memory matches the deploy stack (~30 GB on an A100 in bf16). For sanity checks on smaller GPUs you can set `--text-encoder` to a local T5-base or `--vision-encoder` to a smaller SigLIP — at the cost of breaking apples-to-apples comparison with the released checkpoints.

---

## 🎮 Teleoperation & Data Collection

The on-robot recording stack lives in [`teleop/`](teleop/README.md). It is the same kit used to capture the released `Dexora_Real-World_Dataset`, with paths anchored at `PROJECT_ROOT` so it ports cleanly to a new robot.

- `teleop/scripts/record_delete.py` — top-level orchestrator that forks the robot recorder and the Vision-Pro teleop simultaneously, then archives each episode under a configurable root.
- `teleop/imitate_all/record_4_rgb_cam.py` — robot + 4-camera recorder (USB / RealSense → BSON), adapted from [airbot Imitate-All](https://github.com/airbots-org/Imitate-All).
- `teleop/teleop_pkg/receive_from_vision_pro.py` — pulls the Apple Vision Pro hand skeleton, retargets to the 12-DoF XHAND joints, drives the hands and logs `xhand_control_data.bson`.
- `teleop/scripts/replay.py` — synchronized playback of a recorded episode on both arms and hands.
- `teleop/data_tools/`, `teleop/video_tools/`, `teleop/camera_tools/` — episode consistency checks, 2 × 2 review-video generator, and USB-camera bring-up.

Two conda environments are required (the same ones used by `deploy/`): `imitall` (Python 3.10, AIRBOT SDK) on the robot side and `xhand_tele_env` (Python 3.8, `xhand_tele_ops`) for the Vision-Pro hand side. The full setup — udev rules for the USB cameras, Vision-Pro IP configuration, secrets layout — is documented in [`teleop/README.md`](teleop/README.md). Once recorded, [`dataprocess/airbot_lerobot.py`](dataprocess/airbot_lerobot.py) converts the BSON session into the LeRobot v2.1 layout consumed by `data/lerobot_vla_dataset.py` and `s1_pretrain.sh`.

---


## 🔗 Related Work & Upstream Tooling

| Component | Used for | Link |
|---|---|---|
| LeRobot v2.1 | Real-world data format | <https://github.com/huggingface/lerobot> |
| DexMimicGen | Synthetic trajectory synthesis | <https://github.com/NVlabs/DexMimicGen> |
| Objaverse / Objaverse-XL | Source of 3D assets for simulation | <https://objaverse.allenai.org/> |
| Qwen2.5-VL | VLM-driven asset mining and physical-property assignment | <https://huggingface.co/Qwen> |
| MuJoCo | Digital twin and replay-based post-validation | <https://mujoco.org> |
| RDT-1B | Architectural reference for the Diffusion-Transformer policy | <https://github.com/thu-ml/RoboticsDiffusionTransformer> |
| DWBC | Score → weight mapping for the post-training loss | <https://github.com/ryanxhr/DWBC> |

---

## 📜 Citation

If you find Dexora useful in your research, please cite our ICRA 2026 paper:

```bibtex
@inproceedings{dexora2026,
  title     = {Dexora: Open-Source VLA for High-DoF Bimanual Dexterity},
  author    = {Zhang, Zongzheng and Pang, Jingrui and others},
  booktitle = {Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2026}
}
```

The full author list is available in [`CITATION.cff`](CITATION.cff).

---

## 📝 License

This codebase is released under the [MIT License](LICENSE). Third-party components (SigLIP, T5, LeRobot, RDT-1B reference) retain their original licenses.

For questions, collaborations, or feedback, please open an issue or reach the maintainers through the [project page](https://dexoravla.github.io).
