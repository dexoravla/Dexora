# Dexora teleoperation & data-collection kit

This directory is the **on-robot** companion to the model code in the rest of
Dexora-VLA: it gathers the four-camera + 36-DoF demonstrations that the
training pipeline (`s1_pretrain.sh` → `s3_post_train.sh`) expects, plus the
playback / data-quality scripts referenced in the paper (§III-A / §III-B).

It bundles three originally-separate code bases — a fork of
[`airbots-org/Imitate-All`](https://github.com/airbots-org/Imitate-All)
(robot + 4-camera recording framework), the vendor XHand teleop SDK
(Vision Pro -> XHand teleoperation), and a thin set of top-level launchers
+ data triage scripts written for this project — with all hard-coded paths
replaced by `PROJECT_ROOT`-anchored ones, so this folder ports cleanly to
a new machine.

To move it onto a new robot you only need to:
① edit `scripts/*.py` to point `ROBOT_PYTHON_PATH` / `HAND_PYTHON_PATH` at the
right conda envs;
② drop the XHand `auth_info.json` / `key.dat` into `teleop_pkg/` (see
`teleop_pkg/SECRETS.md` — never checked in);
③ install the two conda envs (see below).

The recorded data lands in `LeRobot v2.1` format — the same layout that
`Dexora/Dexora_Real-World_Dataset` ships on HuggingFace and that
`data/lerobot_vla_dataset.py` consumes — so a fresh dataset you collect here
can be fed straight into `s1_pretrain.sh`.

---

## Directory layout

```
mmk_teleop_record_kit/
├── README.md                       <- the file you are reading
├── requirements.txt                <- top-level pointer (each env has its own requirements in its subdir)
├── .gitignore
│
├── scripts/                        <- "outer launchers" (orchestrators)
│   ├── record_delete.py            <- main program actually used for data collection
│   ├── record.py                   <- earlier version (copies but does not delete the source)
│   ├── record_intrpt.py            <- variant that uses ProcessManager
│   ├── replay.py                   <- synchronized robot + dexterous-hand playback
│   ├── replay_lerobot.py           <- playback with lerobot parquet support
│   └── replay_only_robot.py        <- robot-only playback
│
├── imitate_all/                    <- subset of mmk_dev/Imitate-All (actual recording / playback)
│   ├── record_4_rgb_cam.py         <- recording script for four USB cameras + MMK2
│   ├── mmk_replay.py               <- BSON trajectory playback
│   ├── mmk_replay_lerobot.py       <- Parquet trajectory playback
│   ├── habitats/  robots/  data_process/  configurations/  envs/  utils/
│   ├── requirements/               <- pip requirements for the imitall env
│   ├── 99-camera-symlinks.rules    <- udev rules mapping USB cameras to /dev/camera_*
│   └── install_camera_symlinks.sh
│
├── teleop_pkg/                     <- subset of teleop_software_pkg (dexterous-hand side)
│   ├── receive_from_vision_pro.py  <- pull gestures from Vision Pro -> drive XHand + record bson
│   ├── control_from_bson.py        <- dexterous-hand BSON playback
│   ├── config.yaml                 <- entry-point config for XHandTeleOps
│   ├── env.yaml                    <- conda env snapshot for xhand_tele_env
│   ├── xhand_tele_ops-*.whl        <- vendor-supplied SDK wheel (x86_64 / cp38 only)
│   ├── auth_info.example.json      <- auth placeholder (the real auth_info.json is not committed)
│   └── SECRETS.md                  <- how to configure the auth / key files
│
├── data_tools/                     <- data triage / BSON processing
│   ├── validate_data_consistency.py    <- check the 4-camera frame counts against the BSON frame counts
│   ├── swap_action_observation_bson.py <- swap action<->observation inside a BSON (+ degrees -> radians)
│   ├── bson_to_json_converter.py
│   └── sync_helper.py
│
├── video_tools/                    <- video / review helpers
│   ├── video.py                    <- 4-stream images -> 4 MP4 files
│   ├── video_rotate.py             <- images -> GIF (with rotation / frame subsampling)
│   ├── video_grid_merge.py         <- 4 MP4 files -> 2x2 grid MP4
│   └── video_review_generator.py   <- one-shot pipeline: images -> 2x2 review video
│
├── camera_tools/                   <- camera debugging
│   ├── usb_cameras.py              <- open multiple USB cameras simultaneously for preview
│   ├── camera_test.py              <- multi-camera synchronized capture / disk-write throughput test
│   ├── tools_camera.py             <- AprilTag-based pose-alignment helper (NCC / offset arrows)
│   └── v4.py                       <- AprilTag debugging demo
│
├── samples/                        <- sample data for debugging / playback
│   ├── episode_0.bson              <- one episode of the robot body
│   ├── episode_0.json              <- json mirror of the bson above
│   ├── xhand_control_data.bson     <- one episode of the dexterous hand
│   └── xhand_control_data.json
│
└── docs/
    └── validation_script_guide.md
```

---

## End-to-end data flow (one recording session)

```
┌────────────────────────────────────────────────────────────────────────┐
│   scripts/record_delete.py   <- the entry point you run in the terminal │
│   (forks two child processes inside the imitall env)                    │
└────────────────┬─────────────────────────────────────┬────────────────┘
                 │                                     │
       ROBOT_PYTHON_PATH                       HAND_PYTHON_PATH
                 │                                     │
     imitate_all/record_4_rgb_cam.py        teleop_pkg/receive_from_vision_pro.py
     ├─ open the 4 USB cameras              ├─ connect to the Vision Pro
     ├─ drive the robot body via airbot_py  ├─ retarget gestures -> XHand joints
     ├─ save to imitate_all/data/raw/       ├─ save to teleop_pkg/xhand_control_data.bson
     │     example/episode_0/{camera_*}     │     (contains action+observation+t)
     │     example/episode_0.bson           │
     └─ stop after enough frames or 's'     └─ press 's' to stop
                 │                                     │
                 └────────────── wait() ──────────────┘
                                  │
                          record_delete.py:copy()
                                  │
              cp every output to <ARCHIVE_ROOT>/episode_{N}/
              rm the source files
```

Playback direction:

```
scripts/replay.py
 ├─ imitate_all/mmk_replay.py        (read episode_0.bson -> drive MMK2 joints)
 └─ teleop_pkg/control_from_bson.py  (read xhand_control_data.bson -> drive XHand)
Both sides hand-shake to READY, wait for the user to press Enter, then fire START together.
```

Data triage:

```
data_tools/validate_data_consistency.py   <- iterates over every episode_* under all action_*,
                                             checks that the per-episode camera_*/ image count
                                             matches the frame count of the two BSONs, and
                                             writes any anomalies to logs/*.log
data_tools/swap_action_observation_bson.py <- some labeling stages require swapping action /
                                              observation inside the BSON (xhand will also
                                              convert degrees to radians)
```

---

## Environment installation

You need **two independent conda environments** because the robot side and the
dexterous-hand side use different major Python versions:

| Env name | Python | Purpose | Reference files |
| -------- | ------ | ------- | --------------- |
| `imitall`         | 3.10 | Robot body + 4-camera recording, body playback | `imitate_all/requirements/*.txt` (full env snapshot lives in the upstream repo's `environment.yml`) |
| `xhand_tele_env`  | 3.8  | XHand teleop, Vision Pro ingestion             | `teleop_pkg/requirements_x86_64.txt`, `teleop_pkg/env.yaml`, `teleop_pkg/xhand_tele_ops-*.whl` |

Minimal install steps (machine is x86_64 + CUDA 12.x):

```bash
# === imitall ===
conda create -n imitall python=3.10 -y
conda activate imitall
pip install -r imitate_all/requirements/data_collection.txt
pip install -r imitate_all/requirements/realsense.txt
# airbot-data / airbot-py / mmk2-types are vendor-supplied packages shipped
# as wheels by AIRBOT; install them according to the AIRBOT SDK documentation.

# === xhand_tele_env ===
conda create -n xhand_tele_env python=3.8 -y
conda activate xhand_tele_env
pip install teleop_pkg/xhand_tele_ops-1.1.5-cp38-cp38-linux_x86_64.whl
pip install -r teleop_pkg/requirements_x86_64.txt
# Let python listen to raw network packets (required by pynput to handle the Vision Pro stream)
sudo setcap cap_net_raw+ep "$(readlink -f "$(which python3)")"
```

Then update the following two lines at the top of `scripts/*.py` to point to your real paths:

```python
ROBOT_PYTHON_PATH = "/path/to/miniconda3/envs/imitall/bin/python"
HAND_PYTHON_PATH  = "/path/to/miniconda3/envs/xhand_tele_env/bin/python"
```

---

## System-level prerequisites

### 1. USB camera symlinks (4 USB cameras)

`imitate_all/record_4_rgb_cam.py` uses **fixed symbolic links**
`/dev/camera_{left,right,high,head}` instead of `/dev/video*`, so the numbering
does not shuffle after a replug:

```bash
cd imitate_all
sudo ./install_camera_symlinks.sh
# Then replug the cameras or reboot to let the rules take effect
ls -la /dev/camera_*
```

> Note: the default rules currently do not contain an entry for the `head`
> name (only left/right/high), so you have to add the head camera's devpath
> yourself following `camera_mapping_guide.md`.

### 2. Robot network

MMK2 defaults to gRPC at `192.168.11.200:50055`; make sure the host can reach
the robot body.

### 3. Vision Pro connection

In `teleop_pkg/config.yaml`, set `avp_ip` to your Vision Pro's actual IP and
confirm that the Vision Pro is already streaming (see
[avp-stream](https://pypi.org/project/avp-stream/)).

### 4. Auth files

See [`teleop_pkg/SECRETS.md`](teleop_pkg/SECRETS.md).

---

## Usage

### Data collection

```bash
conda activate imitall   # only to give the outer launcher an interpreter; the children switch to their own envs
cd mmk_teleop_record_kit

# Record the N-th episode (--order is required, otherwise the copy stage will error out)
python scripts/record_delete.py --order 0
python scripts/record_delete.py --order 1
...
```

Operating procedure:

1. After the program launches, the robot side will prompt you for a key press;
   press space to start recording this episode.
2. The dexterous-hand side has its own keyboard listener: space to start, `s`
   to save and exit, ESC to abort.
3. Once you press `s` on both sides, the outer launcher will automatically
   `copy()` the data to `ARCHIVE_ROOT/episode_{order}/`. Edit the
   `ARCHIVE_ROOT` constant at the top of `record_delete.py` to point at
   your own data directory before the first run.

### Playback

```bash
# By default plays back samples/episode_0.bson + samples/xhand_control_data.bson
python scripts/replay.py
# To use your own data: edit DEFAULT_EPISODE_BSON at the top of scripts/replay.py
```

### Data triage

```bash
# Validate every action_* folder
python data_tools/validate_data_consistency.py --path <ARCHIVE_ROOT>
# Validate a single action
python data_tools/validate_data_consistency.py --path <ARCHIVE_ROOT> --action action6 -v
```

### Video review

```bash
# Generate a 2x2-grid review video directly from the images in one step
python video_tools/video_review_generator.py \
    --input <ARCHIVE_ROOT> --output review_videos --fps 20
```

---

## Notes and limitations

This kit is the same pipeline used to collect the released
`Dexora_Real-World_Dataset`. A few rough edges are worth flagging before
you adapt it to a new platform:

* **Order argument validation.** `scripts/record_delete.py` parses
  `--order` inside `copy()`, so a missing value is only flagged after the
  recording finishes. Pass `--order N` on every run.
* **Process supervision.** `record_delete.py` only calls `terminate()` on
  `KeyboardInterrupt`. `scripts/record_intrpt.py` ships a `ProcessManager`
  variant if stricter cleanup is required.
* **Two-process synchronization.** The two child processes do not share a
  start/stop handshake or trial id; `sync_helper.py` exposes file-signal
  primitives (and `scripts/replay.py` already uses them in the playback
  direction) if you need stricter alignment during recording.
* **Timestamping.** `xhand_control_data.bson` is written in frame order
  without an explicit timestamp. Switch to
  `receive_from_vision_pro_timestemp.py` if you need time-based alignment
  of the arm and hand streams.
* **Hard-coded paths.** A handful of utility scripts
  (`video_tools/video.py`, `video_tools/video_rotate.py`,
  `camera_tools/v4.py`) still take dataset paths via top-of-file constants
  rather than CLI flags; edit them before running.
* **Auth / keys.** `teleop_pkg/auth_info.json` and `teleop_pkg/key.dat` are
  intentionally not distributed; see
  [`teleop_pkg/SECRETS.md`](teleop_pkg/SECRETS.md) for how to install
  vendor-supplied copies.

---

## Provenance and attribution

* `imitate_all/` is a local copy of
  [Airbot Imitate-All](https://github.com/airbots-org/Imitate-All)
  (see `imitate_all/LICENSE`).
* The `xhand_tele_ops` wheel and SDK interface in `teleop_pkg/` are
  third-party components from the XHand vendor.
* The top-level launchers under `scripts/`, the data-triage utilities
  under `data_tools/`, and the video-review helpers under `video_tools/`
  are part of the Dexora release; only the directory layout and paths were
  reorganized for this repository.
