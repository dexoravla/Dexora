# Dexora: Open-source VLA for High-DoF Bimanual Dexterity

> 📝 **Paper**: *Dexora: Open-source VLA for High-DoF Bimanual Dexterity* (ICRA 2026 submission — see [`ICRA26_0209_FI.pdf`](./ICRA26_0209_FI.pdf))
> 🌐 **Project page**: <https://dexoravla.github.io>
> 🤗 **Dataset**: [Dexora/Dexora_Real-World_Dataset](https://huggingface.co/datasets/Dexora/Dexora_Real-World_Dataset) — 12.2 K teleoperated episodes / 2.92 M frames / 40.5 h, LeRobot v2.1 standard
> 🤖 **Hardware**: 2 × 6-DoF AIRBOT arms + 2 × 12-DoF XHAND (36 controlled DoF; +3 fixed head/spine dims for SDK compatibility)

Dexora is the first open-source Vision-Language-Action (VLA) system that
natively targets **dual-arm, dual-hand, high-DoF dexterous manipulation**.
This repository releases the full *training*, *inference* and *data-processing*
code used in the paper. Large data and pretrained weights are released on the
project page (and through the HuggingFace dataset above).

The system is built around three contributions:

1. **Hybrid teleoperation** — gross arm kinematics from a custom exoskeleton
   backpack + fine finger motion from markerless Apple Vision Pro tracking,
   driving both the physical platform and a MuJoCo digital twin (§III-A).
2. **Embodiment-matched corpus** — 100 K simulated trajectories (§III-B) and
   12.2 K real-world teleoperated episodes (released on HuggingFace), all in
   the LeRobot v2.1 standard.
3. **Discriminator-guided quality-aware training** — an offline discriminator
   (PU loss, §III-C, Eq. 7) scores each demonstration clip; the Diffusion
   Transformer policy is post-trained with a weighted loss
   (§III-D, Eq. 8) that down-weights low-quality demonstrations.

---

## Repository layout

```
Dexora-VLA/
├── configs/                       # YAML / JSON configuration
│   ├── base_400m.yaml             #   400M paper spec (28 / 1024 / 16)
│   ├── base.yaml                  #   1B variant (legacy)
│   ├── scoring.yaml               #   30M discriminator
│   ├── cross_embodiment/          #   EC-1 / EC-2 / EC-3 fine-tune configs
│   ├── zero2.json                 #   DeepSpeed stage-2 (optional)
│   ├── dataset_control_freq.json  #   per-dataset control freq (paper: 20 Hz)
│   ├── finetune_datasets.json     #   dataset names visible to the loader
│   └── ...
├── models/
│   ├── rdt/                       # Diffusion-Transformer backbone blocks
│   ├── rdt_runner.py              # Stage-1/3 policy (Eq. 8 weighted MSE)
│   ├── scoring_model.py           # Stage-2 30M discriminator (Eq. 7)
│   ├── sample_weighting.py        # DWBC score → weight + Eq. 8 helper
│   ├── ema_model.py / hub_mixin.py
│   └── multimodal_encoder/        # SigLIP + T5 thin wrappers
├── train/
│   ├── train.py / main.py                       # Stage-1 pretrain
│   ├── train_scoring.py / main_scoring.py       # Stage-2 discriminator (PU)
│   ├── train_posttrain.py / main_posttrain.py   # Stage-3 quality-aware post-train
│   ├── dataset.py                               # VLAConsumerDataset
│   ├── sample.py / image_corrupt.py             # eval sampler + light augs
├── data/                                        # Dexora dataset adapters
│   ├── lerobot_vla_dataset.py                   #   LeRobot v2.1 (HF release)
│   ├── lerobot_vla_dataset_with_logpi.py        #   + per-chunk log-π attach
│   ├── bson_vla_dataset.py / *_new.py / *_with_logpi.py   # legacy in-house BSON
│   ├── hdf5_vla_dataset.py                      # legacy HDF5 (RDT-era)
│   └── filelock.py
├── scripts/                                     # Pipeline + eval scripts
│   ├── analyze_episode_quality.py               #   §III-C Eq. (1)-(3) pre-screening → Spre
│   ├── replay_validate.py                       #   Spre → Shigh post-validation
│   ├── compute_logpi.py                         #   §III-C Eq. (4)-(5) log-π proxy
│   ├── eval_smoothness.py                       #   Tab. III Acc. / Jerk metrics
│   ├── eval_action_curves.py                    #   Fig. 11 open-loop per-joint curves
│   ├── encode_lang(_batch).py                   #   Optional T5 language pre-encoding
│   └── run_eval_example.sh                      #   Example launcher for eval_action_curves.py
├── dataprocess/                                 # BSON → LeRobot v2.1 conversion
│   ├── airbot.py / airbot_config.py             #   AIRBOT BSON reader + config
│   ├── airbot_lerobot.py                        #   BSON → LeRobot v2.1 converter
│   ├── lerobot_split_merge_prcessor-main/       #   LeRobot dataset surgery
│   ├── code/                                    #   embodiment configs (aloha, realman)
│   └── README.md
├── teleop/                                      # Real-robot data collection + Vision-Pro teleop
│   ├── scripts/                                 #   record_delete.py / replay.py launchers
│   ├── imitate_all/                             #   robot + 4-camera recorder (Imitate-All subset)
│   ├── teleop_pkg/                              #   Vision Pro → XHand teleop side
│   ├── data_tools/                              #   BSON ⇄ JSON, consistency checks
│   ├── video_tools/                             #   2×2 review video generator
│   ├── camera_tools/                            #   USB / RealSense camera bring-up
│   └── README.md
├── deploy/                                      # Real-robot inference (ZMQ split: policy / arms / hands)
│   ├── dexora_inference_zmq.py                  #   policy host (env: dexora, GPU)
│   ├── dexora_policy.py                         #   RDTRunner + SigLIP + T5 runtime wrapper
│   ├── mmk_forwarder.py                         #   arms forwarder (env: imitall)
│   ├── xhand_forwarder.py                       #   hands forwarder (env: xhand_tele_env)
│   ├── mmk_xhand_config.yaml                    #   shared runtime config
│   ├── mmk2_kdl_py-0.1.4/                       #   mmk2 KDL kinematics lib
│   ├── inference.sh                             #   3-process launcher
│   └── README.md
├── tests/                                       # CPU-only pytest suite
├── google/                                      # SigLIP / T5 download targets (see google/README.md)
├── new_lerobot_stats/                           # Per-dim min/max stats (see new_lerobot_stats/README.md)
├── s1_pretrain.sh                               # Stage 1 launcher
├── s2a_analyze_jerk.sh                          # Stage 2a launcher
├── s2b_replay.sh                                # Stage 2b launcher
├── s2c_compute_logpi.sh                         # Stage 2c-1 launcher
├── s2c_train_scoring.sh                         # Stage 2c-2 launcher
├── s3_post_train.sh                             # Stage 3 launcher
├── run_all_stages.sh                            # End-to-end pipeline
├── pyproject.toml + requirements{,-dev}.txt
├── ICRA26_0209_FI.pdf                           # The paper
└── LICENSE  +  CITATION.cff  +  CONTRIBUTING.md  +  CODE_OF_CONDUCT.md
```

---

## Installation

```bash
# 1. Conda env  (Python 3.10 is required)
conda create -n dexora python=3.10 -y
conda activate dexora

# 2. PyTorch (pick your own CUDA from pytorch.org; 12.1 example here)
pip install torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 3. The rest (see ``requirements.txt`` for the canonical pin list)
pip install -r requirements.txt

# 4. Editable install (registers ``configs`` / ``data`` / ``models`` / ``train``
#    as importable packages and adds ``dexora-train`` console scripts).
pip install -e .

# 5. (Optional, dev only) lint + tests
pip install -r requirements-dev.txt
pre-commit install
pytest tests/ -q                # 57 CPU-only tests, ~5 s

# 6. (Optional) flash-attn. The attention path falls back to PyTorch SDPA
#    if this is absent, so this is purely a speed knob.
# pip install flash-attn --no-build-isolation
```

We pin `transformers<5`, `huggingface_hub<0.26`, `diffusers<0.32`,
`accelerate<1.0`, `lerobot<0.4` and `numpy<2.0`. These are required: newer
versions break the `is_offline_mode` / LeRobot-v2.1 / `imgaug` interfaces
that the training stack depends on.

---

## Downloading the data

The Dexora real-world dataset is hosted on HuggingFace in the LeRobot v2.1
standard:

```bash
huggingface-cli download Dexora/Dexora_Real-World_Dataset \
    --repo-type dataset \
    --local-dir data/Dexora_Real-World_Dataset
```

Total ≈ 240 GB; the four task families
(`airbot_pick_and_place / airbot_assemble / airbot_articulation / airbot_dexterous`)
are released as separate LeRobot v2.1 datasets so you can pick one to start
with. Each subdirectory has the standard layout:

```
data/Dexora_Real-World_Dataset/
└── airbot_pick_and_place/
    ├── data/   chunk-000/episode_000000.parquet  ...
    ├── videos/ chunk-000/observation.images.{top,wrist_left,wrist_right,front}/episode_000000.mp4
    └── meta/   info.json  episodes.jsonl  tasks.jsonl  modality.json  stats.json  ...
```

> **State / action dimension.** The HF release stores **39-D** state and
> action vectors. The last 3 dims (`head_joint_1`, `head_joint_2`,
> `spine_joint`) are fixed values required by the AIRBOT SDK but are *not*
> modelled by the Dexora policy. The training loaders slice to the first
> **36** dims by default (`[left_arm(6) | right_arm(6) | left_hand(12) | right_hand(12)]`),
> matching paper §III-A. Set `--state_dim_keep 0` to keep the full 39 dims.

### Pretrained encoders

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

See [`google/README.md`](google/README.md) for symlink options if these
encoders already exist on your machine.

### Dataset statistics (per-dim min-max)

`dataset_statistics.json` is **not** in the HF release because it depends on
which subset you train on. Every shell launcher below auto-generates it
once if missing; or you can pre-compute it explicitly:

```bash
python -m data.lerobot_vla_dataset --stat \
    --num_samples 5000 \
    --repo_dir   data/Dexora_Real-World_Dataset/airbot_pick_and_place \
    --output_dir new_lerobot_stats
```

This writes a 36-D `new_lerobot_stats/dataset_statistics.json` plus
`state_distributions.png` / `action_distributions.png` for a quick sanity
check. See [`new_lerobot_stats/README.md`](new_lerobot_stats/README.md).

---

## Three-stage training recipe (paper §III-D)

Every stage is launched by a single shell script that reads its inputs from
env vars (with sensible defaults). The minimal invocation for a fresh user
who just downloaded the dataset:

```bash
# All paths can be overridden via env vars; defaults shown below match the
# repo's directory layout.
export DEXORA_LEROBOT_ROOT=data/Dexora_Real-World_Dataset/airbot_pick_and_place
export DEXORA_T5=google/t5-v1_1-xxl
export DEXORA_SIGLIP=google/siglip-so400m-patch14-384
export DEXORA_STATS=new_lerobot_stats/dataset_statistics.json
```

### Stage 1 — pretrain the 400M policy

Trains the Diffusion Transformer policy for 100 K steps on the real corpus
(or replace `DEXORA_LEROBOT_ROOT` with your sim corpus to reproduce the
paper's sim-pretrain).

```bash
NUM_GPUS=8 MAX_TRAIN_STEPS=100000 \
OUTPUT_DIR=checkpoints/dexora-400m-pretrain \
    bash s1_pretrain.sh
```

→ Writes `checkpoints/dexora-400m-pretrain/checkpoint-*/{pytorch_model.bin,config.json,ema/}`.

### Stage 2a — pre-screen real demonstrations (Eq. 1-3)

Computes per-episode acceleration `Aep` (Eq. 2) and jerk `Jep` (Eq. 3)
under per-dim min-max normalization, then keeps
`Spre = Low-20%(Aep) ∩ Low-20%(Jep)` (≈ 18 % of episodes per paper).

```bash
SPRE_DIR=runs/spre bash s2a_analyze_jerk.sh
# → runs/spre/complete_analysis_results.json
```

### Stage 2b — replay-based post-validation → `Shigh`

Open-loop replays each `Spre` episode in the MuJoCo digital twin and keeps
the survivors that complete the task without collisions.

```bash
SPRE_DIR=runs/spre SHIGH_FILE=runs/shigh.json \
REPLAY_VERIFIER=trust_spre \
    bash s2b_replay.sh
```

The bundled `--verifier trust_spre` is a stub for smoke testing — it
accepts every Spre episode. Switch to `--verifier energy` for a cheap
kinematic heuristic, or to `--verifier mujoco --twin_module path.to.your.replay`
for the real MuJoCo replay. The plug-in module must expose
`replay(states, actions, task_id) -> {"success": bool, "collision_free": bool}`.

### Stage 2c — log-π proxy + discriminator training

#### 2c-1 — `\hat{logπ}_t = -zscore(E_t)` (Eq. 4-5)

```bash
STAGE1_CKPT=checkpoints/dexora-400m-pretrain \
LOGPI_FILE=runs/logpi/logpi.json \
    bash s2c_compute_logpi.sh
# → runs/logpi/logpi.json          (\hat{log π} proxy per chunk)
# → runs/logpi/logpi_raw_E.json    (raw energies E_t)
```

The discriminator (`models/scoring_model.py`) ingests the scalar `\hat{logπ}_t`
through a small sinusoidal positional-style encoding (8 freq bands + raw)
before the linear projection. This is mathematically equivalent in capacity
to `Linear(1 → hidden_size)` but more numerically robust under bf16 when
the z-scored proxy sits near zero.

#### 2c-2 — discriminator PU training (Eq. 7)

```bash
OUTPUT_DIR=checkpoints/dexora-scoring \
LOGPI_FILE=runs/logpi/logpi.json \
SPRE_FILE=runs/spre/complete_analysis_results.json \
SHIGH_FILE=runs/shigh.json \
    bash s2c_train_scoring.sh
```

→ Writes `checkpoints/dexora-scoring/{checkpoint-*,final_model}/pytorch_model.bin`.

### Stage 3 — data-quality-aware post-training (Eq. 8)

Loads the Stage-1 policy and the frozen Stage-2 discriminator, then
fine-tunes the policy on the real corpus with

$$\mathcal{L}_\pi = \sum_{i=1}^{L} w_i \, \lVert\varepsilon_\theta(\cdot) - \varepsilon\rVert_2^2$$

where `w_i = DWBC(d(ξ_i))` is computed online from the discriminator score
via the DWBC mapping (with a short linear warm-up).

```bash
STAGE1_CKPT=checkpoints/dexora-400m-pretrain \
SCORING_CKPT=checkpoints/dexora-scoring/final_model/pytorch_model.bin \
OUTPUT_DIR=checkpoints/dexora-400m-posttrain \
    bash s3_post_train.sh
```

The vanilla baseline (Tab. III "w/o discriminator" row) is reproduced by
adding `EXTRA_FLAGS="--no_quality_weights"`.

### End-to-end pipeline

```bash
RUN_DIR=./runs/dexora-paper-rep \
DEXORA_LEROBOT_ROOT=data/Dexora_Real-World_Dataset/airbot_pick_and_place \
    bash run_all_stages.sh
# Chain stages with START_STAGE / END_STAGE, e.g.
#     START_STAGE=4 END_STAGE=6 RUN_DIR=./runs/... bash run_all_stages.sh
```

---

## Real-robot deployment (inference)

`deploy/` runs a trained Dexora policy on the physical robot. The integration
is split into three single-purpose processes that talk over loopback ZMQ, so
the conflicting Python environments for the policy (GPU + `torch`), the
arms SDK (`airbot_py`) and the hands SDK (`xhand_tele_ops`, Python 3.8) can
coexist without dependency hell:

```
+-----------------------------+   ZMQ tcp://*:5556    +------------------------+
| dexora_inference_zmq.py     | <------------------>  | mmk_forwarder.py       |
| (env: dexora, GPU)          |  arms, 12-D radians   | (env: imitall, 3.10)   |
|                             |   ZMQ tcp://*:5557    +------------------------+
|                             | <------------------>  | xhand_forwarder.py     |
|                             |  hands, 2×12-D rad    | (env: xhand_tele_env)  |
+-----------------------------+                       +------------------------+
```

`deploy/dexora_policy.py` wraps `RDTRunner.from_pretrained(...)` plus
SigLIP-SO400M and T5-XXL into a single `policy.get_action(obs) -> [L, 36]`
call. The inference loop follows the paper's chunk-and-replay scheme: every
`chunk_size` (= L) control ticks we sample a length-L action sequence then
play it back with `action_buffer[t % L]`.

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

Wire protocol, joint limits, RealSense fallback and the full troubleshooting
checklist are in [`deploy/README.md`](deploy/README.md).

### Noise schedule and inference steps

Training (Stage-1 / Stage-3) uses a 1000-step DDPM forward process with a
cosine `squaredcos_cap_v2` beta schedule, predicting the action noise
`\hat{ε}_θ` (paper §III-C). At inference we swap DDPM for **DPMSolver++**
and run only `num_inference_timesteps = 5` solver steps — the setting used
to produce every number in Tab. I / II / III. Increasing it to 10–20
marginally improves smoothness on the dexterous tasks (Tab. III Acc / Jerk)
at a proportional latency cost.

> **Backward compatibility.** Earlier Dexora checkpoints were saved with
> `prediction_type=sample`. `RDTRunner.compute_loss` and
> `scripts/compute_logpi.py` both still handle the `sample` branch even
> though new training defaults to `epsilon`.

---

## Open-loop evaluation (paper Fig. 11)

`scripts/eval_action_curves.py` reproduces the per-joint trajectory plots
shown in Fig. 11 of the paper. It walks a single LeRobot v2.1 episode and
triggers one diffusion pass every `--inference-interval` steps, then
overlays the predicted action chunks on the ground-truth trajectory for
every one of the 36 controlled joints (plus a 6×6 summary grid).

This is the **open-loop** protocol — we always condition on the
ground-truth observation at each sampled timestep, never on the policy's
own previous prediction. It's the cheapest sanity check that a trained
checkpoint is producing physically plausible chunks before you commit to a
closed-loop rollout on the real robot.

Under the hood the script reuses the same `deploy/dexora_policy.py`
wrapper as the on-robot inference loop, so the prediction path is
bit-identical to what the robot would receive at runtime. Inputs (state,
action) are normalized with the same `dataset_statistics.json` the policy
was trained on, ensuring an apples-to-apples comparison.

```bash
# All paths are environment-overridable; defaults shown.
MODEL_PATH=checkpoints/dexora-400m-posttrain \
REPO_DIR=data/Dexora_Real-World_Dataset/airbot_pick_and_place \
STATS_FILE=new_lerobot_stats/dataset_statistics.json \
EPISODE_IDX=0 INFERENCE_INTERVAL=32 \
OUTPUT_DIR=eval_results/airbot_pick_and_place_ep0 \
    bash scripts/run_eval_example.sh
```

→ Writes 36 per-axis PNGs (`ep000000_axis_<i>_<joint_name>.png`) plus one
`ep000000_summary.png` grid under `${OUTPUT_DIR}`.

Useful knobs:

| Flag / env var          | Meaning |
|---|---|
| `--inference-interval`  | Cadence between diffusion passes; defaults to `chunk_size = 32`, i.e. non-overlapping chunks. Use `16` to visualize chunk consistency on overlap. |
| `--max-steps`           | Truncate to the first N steps of the episode (default: full episode). |
| `--instruction "..."`   | Override the dataset-derived language goal (default: use `tasks.jsonl`). |
| `--state-dim 39`        | Keep the full 39-D AIRBOT state instead of slicing to the paper's 36-D layout. Only meaningful for whole-body experiments. |
| `--no-normalize`        | Disable per-dim normalization (legacy checkpoints trained without `stats_file`). |
| `--dump-json`           | Also dump GT + predictions as a single JSON, directly consumable by `scripts/eval_smoothness.py`. |

> **Heads up.** The script needs the policy, SigLIP-SO400M, T5-v1.1-XXL
> *and* the LeRobot dataset all visible at the same time, so peak GPU
> memory matches the deploy stack (~30 GB on an A100 in bf16). For sanity
> checks on smaller GPUs you can set `--text-encoder` to a local
> T5-base / `--vision-encoder` to a smaller SigLIP — at the cost of breaking
> apples-to-apples comparison with the released checkpoints.

---

## Real-robot data collection & teleoperation

The on-robot recording stack (paper §III-A) lives in
[`teleop/`](teleop/README.md). It is the same kit we used to capture the
released `Dexora_Real-World_Dataset`, with paths anchored at
`PROJECT_ROOT` so it ports cleanly to a new robot.

* `teleop/scripts/record_delete.py` — top-level orchestrator that forks the
  robot recorder + the Vision-Pro teleop simultaneously, then archives the
  episode under a configurable root (``ARCHIVE_ROOT`` constant at the top of
  the script).
* `teleop/imitate_all/record_4_rgb_cam.py` — robot + 4-camera recorder
  (4× USB / RealSense → BSON), lifted from
  [airbot Imitate-All](https://github.com/airbots-org/Imitate-All).
* `teleop/teleop_pkg/receive_from_vision_pro.py` — pulls the Apple Vision
  Pro hand skeleton, retargets to the 12-DoF XHand joints, drives the hands
  and logs `xhand_control_data.bson`.
* `teleop/scripts/replay.py` — synchronized playback of a recorded episode
  on both arms + hands.
* `teleop/data_tools/`, `teleop/video_tools/`, `teleop/camera_tools/` —
  episode consistency checks, 2×2 review-video generator, USB-camera bring-up.

Two conda envs are required (the same ones the `deploy/` stack uses):
`imitall` (Python 3.10, AIRBOT SDK) for the robot side and
`xhand_tele_env` (Python 3.8, `xhand_tele_ops`) for the Vision-Pro hand side.
See [`teleop/README.md`](teleop/README.md) for the full setup (udev rules
for the four USB cameras, Vision-Pro IP configuration, secrets layout), and
`dataprocess/airbot_lerobot.py` for the BSON → LeRobot v2.1 conversion that
turns a freshly recorded session into the exact layout consumed by
`data/lerobot_vla_dataset.py` and `s1_pretrain.sh`.

---

## Reproducing the paper numbers

| Table / Figure | How to run | Knob |
|---|---|---|
| Tab. I — Basic tasks (12) | Stage-1 → Stage-3 on each task; 20 rollouts | default |
| Tab. II — Dexterous tasks (6) | Same, on the 6 dexterous tasks | default |
| Tab. III — Discriminator ablation | Run Stage-3 with and without the discriminator | `EXTRA_FLAGS="--no_quality_weights"` |
| Fig. 10 — Data composition | Stage-3 with sim-only / sim+50% real / sim+all real | `REAL_DATA_FRACTION={0.0, 0.5, 1.0}` |
| Fig. 9, Tab. II EC rows — Cross-embodiment | Stage-3 ckpt + fine-tune under each EC config | `CONFIG_PATH=configs/cross_embodiment/{ec1_franka,ec2_aloha,ec3_g1_inspire}.yaml` |
| Fig. 11 — Per-joint trajectories | `bash scripts/run_eval_example.sh` (open-loop, see [Open-loop evaluation](#open-loop-evaluation-paper-fig-11)) | `EPISODE_IDX`, `INFERENCE_INTERVAL` |
| Tab. III smoothness (Acc.↓ / Jerk↓) | `scripts/eval_smoothness.py rollouts/*.json --stats_file new_lerobot_stats/dataset_statistics.json` | — |

---

## Upstream tooling referenced by the paper

| Component | Used for | Link |
|---|---|---|
| LeRobot v2.1 | Real-world data format | [github.com/huggingface/lerobot](https://github.com/huggingface/lerobot) |
| DexMimicGen | Synthetic trajectory synthesis (§III-B) | [github.com/NVlabs/DexMimicGen](https://github.com/NVlabs/DexMimicGen) |
| Objaverse / Objaverse-XL | Source of 3D assets for sim | [objaverse.allenai.org](https://objaverse.allenai.org/) |
| Qwen2.5-VL | VLM-driven asset mining + physical-property assignment | [huggingface.co/Qwen](https://huggingface.co/Qwen) |
| MuJoCo | Digital twin + replay post-validation | [mujoco.org](https://mujoco.org) |
| RDT-1B | Architectural reference for the Diffusion-Transformer policy | [github.com/thu-ml/RoboticsDiffusionTransformer](https://github.com/thu-ml/RoboticsDiffusionTransformer) |
| DWBC (Xu et al., ICML'22) | Score → weight mapping (§III-D, ref. [41]) | [github.com/ryanxhr/DWBC](https://github.com/ryanxhr/DWBC) |

---

## Citing

```bibtex
@inproceedings{dexora2026,
  title     = {Dexora: Open-source VLA for High-DoF Bimanual Dexterity},
  author    = {Zhang, Zongzheng and Pang, Jingrui and others},
  booktitle = {ICRA},
  year      = {2026}
}
```

## License

MIT — see [`LICENSE`](LICENSE). Third-party components (SigLIP, T5, LeRobot,
RDT-1B reference) keep their original licenses.
