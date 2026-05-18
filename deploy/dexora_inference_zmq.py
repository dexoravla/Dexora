#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dexora real-robot inference host.

This script loads a trained Dexora policy (Stage-1 or Stage-3 checkpoint) and
drives the AIRBOT / XHand platform via two ZMQ forwarders:

  +----------------------+      ZMQ tcp://*:5556      +-----------------------+
  | dexora_inference_zmq |  <----------------------->  | mmk_forwarder.py      |
  |    (this script)     |       (arms)               |   (env: imitall)      |
  |                      |      ZMQ tcp://*:5557      +-----------------------+
  |    env: dexora       |  <----------------------->  | xhand_forwarder.py    |
  +----------------------+       (hands)              |   (env: xhand_tele)   |
                                                       +-----------------------+

The wire protocol is intentionally identical to the GR00T deployment that
shipped with the AIRBOT teleop kit, so you can swap policies without touching
the forwarders or the on-robot middleware.

Logic mirrors RDT / paper §III-C: every ``chunk_size`` (= L) control ticks we
run one diffusion pass to obtain a length-L action sequence, then play it back
with ``action_buffer[t % L]`` indexing.

Usage:

    conda activate dexora
    python deploy/dexora_inference_zmq.py \\
        --model-path checkpoints/dexora-400m-posttrain \\
        --config-path deploy/mmk_xhand_config.yaml \\
        --task-description "Pick the apple and put it on the plate." \\
        --save-logs --monitor-interval 1
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
import zmq

# Ensure the repo root is importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deploy.dexora_policy import (  # noqa: E402
    DEXORA_CAMERA_ORDER,
    DexoraPolicy,
    DexoraPolicyConfig,
)

# Optional RealSense for the head camera. Skipped silently when absent.
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False


# ---------------------------------------------------------------------------
# ZMQ robot interface
# ---------------------------------------------------------------------------
class ZMQRobotInterface:
    """Thin REQ/REP wrapper around the two forwarders + local camera capture.

    Wire format matches ``deploy/mmk_forwarder.py`` and ``deploy/xhand_forwarder.py``
    in this repository (and the GR00T deployment kit they were ported from).
    """

    def __init__(
        self,
        config: dict,
        mmk_host: str = "localhost",
        mmk_port: int = 5556,
        xhand_host: str = "localhost",
        xhand_port: int = 5557,
        request_timeout_ms: int = 5000,
    ) -> None:
        self.config = config
        ctx = zmq.Context()

        self.mmk_socket = ctx.socket(zmq.REQ)
        self.mmk_socket.connect(f"tcp://{mmk_host}:{mmk_port}")
        self.mmk_socket.setsockopt(zmq.RCVTIMEO, request_timeout_ms)
        logging.info(f"Connected to MMK forwarder at {mmk_host}:{mmk_port}")

        self.xhand_socket = ctx.socket(zmq.REQ)
        self.xhand_socket.connect(f"tcp://{xhand_host}:{xhand_port}")
        self.xhand_socket.setsockopt(zmq.RCVTIMEO, request_timeout_ms)
        logging.info(f"Connected to XHand forwarder at {xhand_host}:{xhand_port}")
        self._ctx = ctx

        # ---- Cameras ------------------------------------------------------
        # The MMK forwarder no longer streams the head camera (lower latency
        # to grab locally); we open the RealSense head + the 3 USB cameras
        # right here.
        self.external_camera_names = ["cam_head", "cam_left_wrist", "cam_third_view", "cam_right_wrist"]
        self.external_camera_ids = list(config["ext_cam_ids"])
        self.realsense_pipeline = None
        self.external_cameras = {}
        self._initialize_cameras()

    # ----- cameras -----
    def _initialize_cameras(self) -> None:
        # USB cameras
        for name, dev in zip(self.external_camera_names, self.external_camera_ids):
            cap = cv2.VideoCapture(dev)
            if not cap.isOpened():
                logging.warning(f"Failed to open camera {name} at {dev}; will use mean-colour placeholder.")
                cap = None
            if cap is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.external_cameras[name] = cap
            if cap is not None:
                logging.info(f"Initialized camera {name} ({dev})")

        # Optional RealSense (override cam_head if available)
        if REALSENSE_AVAILABLE and "cam_head" in self.external_cameras:
            try:
                pipeline = rs.pipeline()
                rs_cfg = rs.config()
                rs_cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                pipeline.start(rs_cfg)
                self.realsense_pipeline = pipeline
                logging.info("Using RealSense head camera (overrides USB cam_head).")
            except Exception as e:
                logging.warning(f"RealSense init failed ({e}); falling back to USB head camera.")
                self.realsense_pipeline = None

    def _capture_images(self) -> dict[str, np.ndarray]:
        images: dict[str, np.ndarray] = {}
        for name, cap in self.external_cameras.items():
            if name == "cam_head" and self.realsense_pipeline is not None:
                continue  # handled below
            if cap is None:
                images[name] = np.zeros((480, 640, 3), dtype=np.uint8)
                continue
            ok, frame = cap.read()
            if not ok:
                logging.warning(f"Failed to read {name}; using placeholder.")
                images[name] = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                images[name] = frame  # BGR

        if self.realsense_pipeline is not None:
            frames = self.realsense_pipeline.wait_for_frames()
            color = frames.get_color_frame()
            images["cam_head"] = (
                np.asanyarray(color.get_data()) if color else np.zeros((480, 640, 3), dtype=np.uint8)
            )

        return images

    # ----- ZMQ requests -----
    def _req(self, sock: zmq.Socket, payload: dict, target: str) -> dict:
        sock.send_json(payload)
        try:
            return sock.recv_json()
        except zmq.error.Again as e:
            raise RuntimeError(f"{target} forwarder timeout") from e

    def get_mmk_qpos(self) -> np.ndarray:
        resp = self._req(self.mmk_socket, {"command": "get_observations"}, "MMK")
        if "error" in resp:
            raise RuntimeError(f"MMK error: {resp['error']}")
        return np.asarray(resp["qpos"], dtype=np.float32)

    def get_xhand(self) -> tuple[np.ndarray, np.ndarray]:
        resp = self._req(self.xhand_socket, {"command": "get_observations"}, "XHand")
        if "error" in resp:
            raise RuntimeError(f"XHand error: {resp['error']}")
        return (
            np.asarray(resp["left_hand"], dtype=np.float32),
            np.asarray(resp["right_hand"], dtype=np.float32),
        )

    def execute_mmk(self, arm_action_12d: np.ndarray) -> dict:
        return self._req(
            self.mmk_socket,
            {"command": "execute_action", "action": arm_action_12d.tolist()},
            "MMK",
        )

    def execute_xhand(self, left_12d: np.ndarray, right_12d: np.ndarray) -> dict:
        return self._req(
            self.xhand_socket,
            {
                "command": "execute_action",
                "action_data": {
                    "left_hand": left_12d.tolist(),
                    "right_hand": right_12d.tolist(),
                },
            },
            "XHand",
        )

    def reset_mmk(self) -> dict:
        return self._req(self.mmk_socket, {"command": "reset"}, "MMK")

    def close(self) -> None:
        for cap in self.external_cameras.values():
            if cap is not None:
                cap.release()
        if self.realsense_pipeline is not None:
            self.realsense_pipeline.stop()
        self.mmk_socket.close()
        self.xhand_socket.close()
        self._ctx.term()


# ---------------------------------------------------------------------------
# Observation / action helpers
# ---------------------------------------------------------------------------
def _bgr_to_rgb(images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for k, im in images.items():
        if im.ndim == 3 and im.shape[2] == 3:
            out[k] = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        else:
            out[k] = im
    return out


def gather_observation(
    interface: ZMQRobotInterface,
    instruction: str,
    xhand_obs_in_degrees: bool,
) -> dict:
    """Pull a single (state, images, instruction) frame from the robot."""
    qpos = interface.get_mmk_qpos()
    if qpos.shape[0] < 12:
        raise RuntimeError(f"MMK qpos has {qpos.shape[0]} dims, expected >= 12")
    left_arm = qpos[:6]
    right_arm = qpos[6:12]

    left_hand, right_hand = interface.get_xhand()
    if xhand_obs_in_degrees:
        left_hand = np.deg2rad(left_hand)
        right_hand = np.deg2rad(right_hand)

    # State layout: [left_arm(6) | right_arm(6) | left_hand(12) | right_hand(12)] = 36
    state = np.concatenate([left_arm, right_arm, left_hand, right_hand], axis=0)

    images = _bgr_to_rgb(interface._capture_images())

    return {
        "state": state.astype(np.float32),
        "images": images,
        "instruction": instruction,
    }


def execute_action(interface: ZMQRobotInterface, action_36d: np.ndarray) -> None:
    """Split [arm_L | arm_R | hand_L | hand_R] and route to forwarders."""
    arm_l = action_36d[0:6]
    arm_r = action_36d[6:12]
    hand_l = action_36d[12:24]
    hand_r = action_36d[24:36]
    interface.execute_xhand(hand_l, hand_r)
    interface.execute_mmk(np.concatenate([arm_l, arm_r]))


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------
def model_inference_loop(
    config: dict,
    interface: ZMQRobotInterface,
    policy: DexoraPolicy,
    task_description: str,
    *,
    monitor_interval: int = 1,
    show_camera: bool = False,
    save_logs: bool = False,
    log_dir: str = "logs",
) -> None:
    chunk_size = int(config["chunk_size"])
    max_steps = int(config["max_steps"])
    control_freq = float(config["control_frequency"])
    xhand_obs_in_degrees = str(config.get("xhand_obs_unit", "deg")).lower() == "deg"

    logging.info(f"Resetting MMK to default pose ...")
    interface.reset_mmk()

    log_file = None
    if save_logs:
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"dexora_inference_{ts}.log")
        log_file = open(log_path, "w")
        log_file.write(f"# Dexora inference log @ {datetime.now()}\n")
        log_file.write(f"# task: {task_description}\n")
        log_file.write(f"# chunk_size={chunk_size} control_freq={control_freq}\n")
        logging.info(f"Logging to {log_path}")

    if show_camera:
        cv2.namedWindow("Dexora Camera Views", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Dexora Camera Views", 1280, 480)

    t = 0
    action_buffer = np.zeros((chunk_size, 36), dtype=np.float32)
    prev_action: np.ndarray | None = None
    inference_times: list[float] = []
    loop_times: list[float] = []

    try:
        while t < max_steps:
            loop_t0 = time.time()
            obs = gather_observation(interface, task_description, xhand_obs_in_degrees)

            if show_camera and t % 5 == 0:
                _show_4cam_grid(obs["images"])

            inference_ms = 0.0
            if t % chunk_size == 0:
                t_inf = time.time()
                action_buffer = policy.get_action(obs).astype(np.float32)
                inference_ms = (time.time() - t_inf) * 1000.0
                inference_times.append(inference_ms / 1000.0)
                logging.info(
                    f"[Step {t:5d}] inference done: action_buffer={action_buffer.shape}, "
                    f"{inference_ms:.1f} ms"
                )
                if log_file is not None:
                    log_file.write(
                        f"step={t} inference={inference_ms:.2f}ms "
                        f"range=[{action_buffer.min():.4f},{action_buffer.max():.4f}]\n"
                    )

            raw_action = action_buffer[t % chunk_size]
            execute_action(interface, raw_action)

            loop_dt = time.time() - loop_t0
            loop_times.append(loop_dt)
            # Sleep to hit control_freq.
            remaining = 1.0 / control_freq - loop_dt
            if remaining > 0:
                time.sleep(remaining)

            if monitor_interval > 0 and t % monitor_interval == 0:
                diff = (
                    float(np.abs(raw_action - prev_action).mean())
                    if prev_action is not None
                    else 0.0
                )
                logging.info(
                    f"[Step {t:5d}] buf[{t % chunk_size:2d}] diff={diff:.4f} "
                    f"loop={loop_dt*1000:.1f}ms fps={1/max(loop_dt,1e-3):.1f} "
                    f"armL=[{raw_action[0:6].min():.2f},{raw_action[0:6].max():.2f}] "
                    f"armR=[{raw_action[6:12].min():.2f},{raw_action[6:12].max():.2f}]"
                )
            prev_action = raw_action.copy()
            t += 1

    finally:
        if inference_times:
            logging.info(
                f"\n== Inference stats == "
                f"steps={t} infer_n={len(inference_times)} "
                f"avg_infer={np.mean(inference_times)*1000:.1f}ms "
                f"avg_loop={np.mean(loop_times)*1000:.1f}ms "
                f"fps={1/np.mean(loop_times):.1f}"
            )
        if log_file is not None:
            log_file.write(f"# end @ {datetime.now()}, steps={t}\n")
            log_file.close()


def _show_4cam_grid(images: dict[str, np.ndarray]) -> None:
    h, w = 240, 320
    def _get(k):
        img = images.get(k, np.zeros((480, 640, 3), dtype=np.uint8))
        return cv2.resize(img, (w, h))
    grid = np.vstack([
        np.hstack([_get("cam_head"),       _get("cam_third_view")]),
        np.hstack([_get("cam_left_wrist"), _get("cam_right_wrist")]),
    ])
    cv2.imshow("Dexora Camera Views", grid)
    cv2.waitKey(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dexora real-robot inference (ZMQ)")
    # Model / config
    parser.add_argument("--model-path", required=True,
                        help="Stage-1 / Stage-3 checkpoint directory or pytorch_model.bin")
    parser.add_argument("--config-path", default="deploy/mmk_xhand_config.yaml",
                        help="Path to the runtime YAML (see deploy/mmk_xhand_config.yaml).")
    parser.add_argument("--model-config-path", default="configs/base_400m.yaml",
                        help="Path to the training-time policy config used to construct "
                             "RDTRunner when --model-path is a raw state_dict.")
    parser.add_argument("--task-description", required=True,
                        help="Language goal passed to the policy. Match the training tasks "
                             "for predictable behaviour.")
    parser.add_argument("--text-encoder",  default="google/t5-v1_1-xxl")
    parser.add_argument("--vision-encoder", default="google/siglip-so400m-patch14-384")

    # ZMQ
    parser.add_argument("--mmk-host", default="localhost")
    parser.add_argument("--mmk-port", type=int, default=5556)
    parser.add_argument("--xhand-host", default="localhost")
    parser.add_argument("--xhand-port", type=int, default=5557)

    # Monitoring
    parser.add_argument("--monitor-interval", type=int, default=1,
                        help="Print per-step summary every N steps (0 disables).")
    parser.add_argument("--show-camera", action="store_true",
                        help="Open a 2x2 OpenCV window with the live camera feeds.")
    parser.add_argument("--save-logs", action="store_true")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = get_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    with open(args.config_path, "r") as f:
        runtime_cfg = yaml.safe_load(f)

    logging.info("Initializing Dexora policy ...")
    policy = DexoraPolicy(
        model_path=args.model_path,
        cfg=DexoraPolicyConfig(
            model_config_path=args.model_config_path,
            text_encoder_path=args.text_encoder,
            vision_encoder_path=args.vision_encoder,
            state_dim=int(runtime_cfg.get("state_dim", 36)),
            chunk_size=int(runtime_cfg.get("chunk_size", 32)),
            img_history_size=int(runtime_cfg.get("img_history_size", 1)),
            cameras=tuple(runtime_cfg.get("camera_names", DEXORA_CAMERA_ORDER)),
        ),
    )

    logging.info("Connecting to forwarders ...")
    interface = ZMQRobotInterface(
        config=runtime_cfg,
        mmk_host=args.mmk_host,
        mmk_port=args.mmk_port,
        xhand_host=args.xhand_host,
        xhand_port=args.xhand_port,
    )

    try:
        logging.info("Probing forwarder connectivity ...")
        interface.get_mmk_qpos()
        interface.get_xhand()
        logging.info("Forwarders OK.")

        with torch.inference_mode():
            while input("Press <Enter> to run one episode (anything else to exit): ") == "":
                model_inference_loop(
                    runtime_cfg,
                    interface,
                    policy,
                    args.task_description,
                    monitor_interval=args.monitor_interval,
                    show_camera=args.show_camera,
                    save_logs=args.save_logs,
                    log_dir=args.log_dir,
                )
    except KeyboardInterrupt:
        logging.info("Interrupted by user; shutting down.")
    finally:
        interface.close()
        logging.info("Closed forwarder sockets.")


if __name__ == "__main__":
    main()
