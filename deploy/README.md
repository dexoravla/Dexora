# Dexora real-robot deployment

This directory contains everything needed to run a trained Dexora policy on
the physical robot (2 × AIRBOT arms + 2 × XHand). All on-robot communication
is split into three independent processes that talk over loopback ZMQ, so the
SDKs for the arms (Python 3.10, `airbot_py`) and the dexterous hands (Python
3.8, `xhand_tele_ops`) can run in their own conda envs without dragging the
policy's `torch` / `transformers` stack into either.

```
+-----------------------------+      ZMQ tcp://*:5556      +------------------------+
| dexora_inference_zmq.py     | <-------------------------> | mmk_forwarder.py       |
|                             |   (arms, 12-D joint pos)    |                        |
| env:  dexora  (GPU)         |      ZMQ tcp://*:5557      | env:  imitall (3.10)   |
|                             | <-------------------------> | xhand_forwarder.py     |
|                             |  (hands, 2×12-D radians)    | env:  xhand_tele_env   |
+-----------------------------+                             +------------------------+
        ^                                                              |
        |   4× USB / RealSense                                          |
        +---<>---  cam_head, cam_left_wrist, cam_third_view, cam_right_wrist
```

## Files

| File | Process | Role |
|---|---|---|
| `dexora_inference_zmq.py` | policy host (`dexora` env, GPU) | Loads the trained Dexora policy, grabs cameras + ZMQ obs, runs diffusion, sends actions. |
| `dexora_policy.py` | (imported by inference) | Thin wrapper around `models.rdt_runner.RDTRunner` + SigLIP + T5 → exposes `policy.get_action(obs)`. |
| `mmk_forwarder.py` | arms (`imitall` env) | Wraps `airbot_py.AIRBOTMMK2`. Replies to `{"command": "get_observations"/"execute_action"/"reset"}` over `tcp://*:5556`. |
| `xhand_forwarder.py` | hands (`xhand_tele_env`) | Wraps `xhand_tele_ops.XHandTeleOps`. Same wire schema as `mmk_forwarder` but on `tcp://*:5557`, plus joint-limit clamping. |
| `mmk_xhand_config.yaml` | shared | Runtime config: cameras, control freq, chunk size, SDK paths. **Read by all three processes.** |
| `inference.sh` | launcher | Brings up all three processes from a single shell. |
| `mmk2_kdl_py-0.1.4/` | hand SDK | Minimal KDL kinematics library for the mmk2 platform (pip-installable). |

## Required environments

| env | Python | Purpose |
|---|---|---|
| `dexora`         | 3.10 | This repo (`pip install -e .`) — runs the policy + cameras. |
| `imitall`        | 3.10 | AIRBOT SDK — runs `mmk_forwarder.py`. |
| `xhand_tele_env` | 3.8  | XHand SDK + Vision Pro deps — runs `xhand_forwarder.py`. |

The `imitall` and `xhand_tele_env` envs are the same ones the teleop kit uses
(see [`teleop/README.md`](../teleop/README.md) — they're shared between data
collection and inference).

## Configuration (`mmk_xhand_config.yaml`)

```yaml
camera_names: ['cam_head', 'cam_left_wrist', 'cam_third_view', 'cam_right_wrist']
ext_cam_ids:  ["/dev/camera_head", "/dev/camera_left", "/dev/camera_high", "/dev/camera_right"]

state_dim: 36          # paper layout: 6+6+12+12 (head/spine dropped at runtime)
chunk_size: 32         # diffusion action horizon L (matches configs/base_400m.yaml)
denoising_steps: 5     # DPMSolver++ steps at inference

control_frequency: 20.0   # paper §III-A: 20 Hz recording / control

# XHand details
xhand_obs_unit: "deg"     # the SDK returns degrees; we convert to radians before model input
xhand_action_unit: "rad"  # the policy emits radians

# SDK paths (override on each robot)
xhand_code_path: "/home/ubuntu/teleop_software_pkg"
xhand_config:    "/home/ubuntu/teleop_software_pkg/config.yaml"
mmk_code_path:   "/home/ubuntu/mmk_dev/Imitate-All"
mmk_ip: "192.168.11.200"
mmk_port: 50055
```

## Quick start

### Three-terminal mode (recommended for first deployment)

```bash
# Terminal A — XHand forwarder
conda activate xhand_tele_env
python deploy/xhand_forwarder.py --config deploy/mmk_xhand_config.yaml

# Terminal B — MMK forwarder
conda activate imitall
python deploy/mmk_forwarder.py   --config deploy/mmk_xhand_config.yaml

# Terminal C — Dexora policy
conda activate dexora
python deploy/dexora_inference_zmq.py \
    --model-path checkpoints/dexora-400m-posttrain \
    --config-path deploy/mmk_xhand_config.yaml \
    --task-description "Pick the apple and put it on the plate." \
    --save-logs --monitor-interval 1
```

### Single-shell mode

```bash
TASK_DESCRIPTION="Pick the apple and put it on the plate." \
MODEL_PATH=checkpoints/dexora-400m-posttrain \
    bash deploy/inference.sh
```

`inference.sh` writes the three processes' logs into a fresh
`logs/deploy-<timestamp>/` directory.

## Wire protocol (ZMQ schema)

Both forwarders use `zmq.REP` and accept JSON requests. The policy host uses
`zmq.REQ` and waits on each reply with a 5 s timeout. The schema below is the
contract — keep it stable if you swap policies or add a new SDK backend.

### MMK forwarder (`tcp://*:5556`)

| Request | Response |
|---|---|
| `{"command": "get_observations"}` | `{"qpos": [12 floats radians]}` (left_arm + right_arm) |
| `{"command": "execute_action", "action": [12 floats]}` | `{"status": "success", "send_result": "..."}` |
| `{"command": "reset"}` | `{"status": "success"}` (homes the robot to `default_action`) |

### XHand forwarder (`tcp://*:5557`)

| Request | Response |
|---|---|
| `{"command": "get_observations"}` | `{"left_hand": [12 floats], "right_hand": [12 floats]}` (degrees by default, see `xhand_obs_unit`) |
| `{"command": "execute_action", "action_data": {"left_hand": [...], "right_hand": [...]}}` | `{"status": "success"\|"skipped"\|"crc_failed", ...}` |

The 36-D action emitted by the policy is split as
`[left_arm(6) | right_arm(6) | left_hand(12) | right_hand(12)]`; the inference
script handles the routing — see `execute_action()` in `dexora_inference_zmq.py`.

## Control-loop logic (paper §III-C alignment)

* Diffusion chunk: every `chunk_size` (= 32 by default) control ticks we run
  one `policy.get_action()` pass, producing a `[chunk_size, 36]` action
  sequence. The next 32 ticks just index into this buffer (`action_buffer[t % L]`).
  This is the same scheme used by RDT / GR00T and matches the action-chunking
  trick the paper describes.
* DPMSolver++ at inference: 5 solver steps are enough for Tab. I / II / III's
  numbers (paper §III-C). Increase via `denoising_steps` in the YAML if you
  want smoother dexterous skills at the cost of latency.
* 20 Hz control: the policy was trained on 20 Hz recordings, so
  `control_frequency: 20.0` is the safe default. Don't change it without
  also retraining on data at the new rate.

## Known caveats / troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'zmq'` in `dexora` env | `pip install pyzmq>=25` (also in `requirements.txt`) |
| Forwarder hangs on startup | Wrong `mmk_code_path` / `xhand_code_path` in the YAML; both SDKs do `chdir` + dynamic import. |
| `RealSense init failed` warning | Either install `pyrealsense2` or unplug the RS — the policy falls back to the USB head camera. |
| Random `Skipped` from XHand | Built-in throttling (`xhand_min_send_interval_s`, `xhand_send_eps`); raise the eps if your motion is too jittery, lower it if responses are sluggish. |
| Hand jitter / arm overshoot | Bump `denoising_steps` (5→10) or check the per-joint limits in `xhand_forwarder.py:JOINT_LIMITS_RAD`. |

## Origin

The forwarders + ZMQ split were originally written for the GR00T deployment
of the same hardware platform; the script names and wire format are kept
identical here so you can drop in any other VLA policy that targets the
36-DoF AIRBOT + XHand robot. Only `dexora_inference_zmq.py` and
`dexora_policy.py` are Dexora-specific.
