#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Open-loop evaluation: compare ground-truth actions with Dexora policy
predictions over a single LeRobot v2.1 episode (paper Fig. 11).

The script:
  * loads a Stage-1 / Stage-3 Dexora checkpoint via ``deploy/dexora_policy.py``;
  * loads one episode from a LeRobot v2.1 dataset (the format released on
    HuggingFace as ``Dexora/Dexora_Real-World_Dataset``);
  * walks the episode and triggers an inference pass every
    ``--inference-interval`` steps, recording the full action chunk;
  * plots the GT action trajectory against the predicted chunks for every
    one of the 36 controlled joints, plus a 6x6 summary grid.

This is "open-loop": predictions are NOT replayed on the robot; we always
condition on the ground-truth observation at the chosen timestep. This is
the same protocol used in Dex-RDT/Dexora ablations to inspect per-joint
chunk consistency without committing to a closed-loop rollout.

Conventions follow ``configs/base_400m.yaml``:
  * state_dim = 36, layout
        [ left_arm(6) | right_arm(6) | left_hand(12) | right_hand(12) ]
  * chunk_size (L) = 32, control_freq = 20 Hz.

Normalization. The training pipeline normalizes states/actions per-dim with
``min_max`` (percentile_1 / percentile_99) using ``stats_file``. We do the
same here so the policy sees inputs in the distribution it was trained on,
and we plot predictions vs normalized GT for an apples-to-apples comparison.
Pass ``--no-normalize`` to skip normalization (only meaningful for raw-data
debugging of legacy checkpoints trained without stats).

Example
-------
    python scripts/eval_action_curves.py \
        --model-path checkpoints/dexora-400m-posttrain \
        --repo-dir   data/Dexora_Real-World_Dataset/airbot_pick_and_place \
        --stats-file new_lerobot_stats/dataset_statistics.json \
        --episode-idx 0 --inference-interval 32 \
        --output-dir  eval_results/airbot_pick_and_place_ep0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# Make the script runnable from the repo root or from anywhere else.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.lerobot_vla_dataset import LeRobotVLADataset  # noqa: E402
from deploy.dexora_policy import (  # noqa: E402
    DEXORA_CAMERA_ORDER,
    DexoraPolicy,
    DexoraPolicyConfig,
)


# ---------------------------------------------------------------------------
# Naming for the 36-D Dexora action vector. Order must match the paper layout
# (see configs/base_400m.yaml and deploy/dexora_inference_zmq.py).
# ---------------------------------------------------------------------------
def dexora_action_names(state_dim: int = 36) -> List[str]:
    names: List[str] = []
    for i in range(6):
        names.append(f"left_arm_joint_{i + 1}")
    for i in range(6):
        names.append(f"right_arm_joint_{i + 1}")
    for i in range(12):
        names.append(f"left_hand_joint_{i + 1}")
    for i in range(12):
        names.append(f"right_hand_joint_{i + 1}")
    if state_dim != len(names):
        # Pad / trim if a non-paper state_dim is requested (e.g. 39 with head).
        if state_dim > len(names):
            names += [f"extra_{j}" for j in range(state_dim - len(names))]
        else:
            names = names[:state_dim]
    return names


# ---------------------------------------------------------------------------
# Dataset access helpers
# ---------------------------------------------------------------------------
def episode_bounds(dataset: LeRobotVLADataset, episode_idx: int) -> Tuple[int, int]:
    """Return [from, to) global indices for ``episode_idx`` (LeRobot v2.1)."""
    ep_from = int(dataset.dataset.episode_data_index["from"][episode_idx].item())
    ep_to = int(dataset.dataset.episode_data_index["to"][episode_idx].item())
    return ep_from, ep_to


def gt_trajectories(
    dataset: LeRobotVLADataset, episode_idx: int
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Walk the episode and stack per-step (state, action[t]) trajectories.

    Returns (gt_states[T, D], gt_actions[T, D], instruction).
    Normalization (if any) follows ``dataset.normalize_mode``.
    """
    ep_from, ep_to = episode_bounds(dataset, episode_idx)
    length = ep_to - ep_from
    if length <= 0:
        raise ValueError(f"Episode {episode_idx} is empty (from={ep_from} to={ep_to}).")

    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    instruction: str = ""
    for local_idx in range(length):
        item = dataset.get_item(index=episode_idx, frame_index=local_idx, state_only=True)
        states.append(np.asarray(item["state"], dtype=np.float64).reshape(-1))
        # action is the chunk starting at this step; we keep action[0] = a_t.
        a = np.asarray(item["action"], dtype=np.float64)
        if a.ndim == 1:
            actions.append(a)
        else:
            actions.append(a[0])

    # Pull the instruction from a single full sample at the first step.
    head = dataset.get_item(index=episode_idx, frame_index=0, state_only=False)
    instruction = head["meta"]["instruction"]

    return np.stack(states, axis=0), np.stack(actions, axis=0), instruction


def fetch_obs_for_inference(
    dataset: LeRobotVLADataset, episode_idx: int, frame_idx: int
) -> Tuple[np.ndarray, Dict[str, np.ndarray], str]:
    """Build the ``obs`` dict expected by ``DexoraPolicy.get_action``.

    State is whatever the dataset returns (normalized if normalize_mode set).
    Images are the most recent frame per camera, RGB uint8, mapped onto the
    canonical ``DEXORA_CAMERA_ORDER`` keys (``cam_head`` / ``cam_left_wrist``
    / ``cam_third_view`` / ``cam_right_wrist``).
    """
    sample = dataset.get_item(index=episode_idx, frame_index=frame_idx, state_only=False)

    state = np.asarray(sample["state"], dtype=np.float32).reshape(-1)
    instruction = sample["meta"]["instruction"]

    # LeRobotVLADataset image keys are (cam_high / cam_left_wrist / cam_right_wrist / cam_third_view)
    # with shape [HIST, H, W, 3] uint8. DexoraPolicy expects DEXORA_CAMERA_ORDER:
    #   ("cam_head", "cam_left_wrist", "cam_third_view", "cam_right_wrist")
    cam_alias = {
        "cam_head": "cam_high",
        "cam_left_wrist": "cam_left_wrist",
        "cam_third_view": "cam_third_view",
        "cam_right_wrist": "cam_right_wrist",
    }
    images: Dict[str, np.ndarray] = {}
    for tgt_name in DEXORA_CAMERA_ORDER:
        src_name = cam_alias.get(tgt_name, tgt_name)
        arr = sample.get(src_name)
        if arr is None or not hasattr(arr, "shape") or arr.size == 0:
            images[tgt_name] = None  # DexoraPolicy fills with mean-colour bg.
            continue
        # arr is [HIST, H, W, 3]; take the latest frame as the current image.
        if arr.ndim == 4:
            images[tgt_name] = arr[-1]
        else:
            images[tgt_name] = arr

    return state, images, instruction


# ---------------------------------------------------------------------------
# Inference / plotting
# ---------------------------------------------------------------------------
def run_eval(args: argparse.Namespace) -> None:
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s - %(levelname)s - %(message)s")

    # ---- 1. Dataset ---------------------------------------------------------
    normalize_mode = None if args.no_normalize else "min_max"
    stats_file = args.stats_file if not args.no_normalize else None
    logging.info(
        f"Loading LeRobot dataset from {args.repo_dir} "
        f"(normalize_mode={normalize_mode}, stats_file={stats_file}, "
        f"state_dim_keep={args.state_dim})"
    )
    dataset = LeRobotVLADataset(
        repo_dir=args.repo_dir,
        normalize_mode=normalize_mode,
        stats_file=stats_file,
        load_imgs=True,
        config_path=args.model_config_path,
        chunk_size=args.chunk_size,
        img_history_size=args.img_history_size,
        state_dim_keep=args.state_dim,
    )

    num_episodes = len(dataset.dataset.episode_data_index["from"])
    if args.episode_idx < 0 or args.episode_idx >= num_episodes:
        raise ValueError(
            f"--episode-idx {args.episode_idx} out of range [0, {num_episodes})."
        )

    # ---- 2. GT trajectories (normalized if normalize_mode set) -------------
    logging.info(f"Loading GT trajectories for episode {args.episode_idx} ...")
    gt_states, gt_actions, instruction = gt_trajectories(dataset, args.episode_idx)
    num_steps = gt_actions.shape[0]
    logging.info(
        f"Episode {args.episode_idx}: T={num_steps}, "
        f"state dim={gt_states.shape[-1]}, action dim={gt_actions.shape[-1]}, "
        f"instruction={instruction!r}"
    )

    if args.max_steps is not None:
        num_steps = min(num_steps, int(args.max_steps))
        gt_states = gt_states[:num_steps]
        gt_actions = gt_actions[:num_steps]

    # ---- 3. Policy ----------------------------------------------------------
    logging.info(f"Loading Dexora policy from {args.model_path} ...")
    policy = DexoraPolicy(
        model_path=args.model_path,
        cfg=DexoraPolicyConfig(
            model_config_path=args.model_config_path,
            text_encoder_path=args.text_encoder,
            vision_encoder_path=args.vision_encoder,
            state_dim=int(args.state_dim),
            chunk_size=int(args.chunk_size),
            img_history_size=int(args.img_history_size),
            cameras=tuple(DEXORA_CAMERA_ORDER),
        ),
    )

    # ---- 4. Sweep inference timesteps --------------------------------------
    inference_steps = list(range(0, num_steps, args.inference_interval))
    predicted_actions: List[np.ndarray] = []
    inference_timesteps: List[int] = []

    override_instr = args.instruction.strip() if args.instruction else ""
    if override_instr:
        logging.info(f"Overriding instruction with: {override_instr!r}")

    logging.info(f"Running inference at {len(inference_steps)} timesteps ...")
    for step in inference_steps:
        if step >= num_steps:
            break
        state, images, instr = fetch_obs_for_inference(dataset, args.episode_idx, step)
        if override_instr:
            instr = override_instr

        obs = {
            "state": state,
            "images": images,
            "instruction": instr,
            "ctrl_freq": float(args.control_freq),
        }

        with torch.inference_mode():
            pred = policy.get_action(obs)  # [chunk_size, state_dim]

        predicted_actions.append(pred.astype(np.float64))
        inference_timesteps.append(step)
        logging.info(
            f"  step={step:5d}  pred_chunk={pred.shape}  "
            f"range=[{pred.min():.4f}, {pred.max():.4f}]"
        )

    # ---- 5. Plot ------------------------------------------------------------
    plot_action_curves(
        gt_actions=gt_actions,
        predicted_actions=predicted_actions,
        inference_timesteps=inference_timesteps,
        state_dim=int(args.state_dim),
        chunk_size=int(args.chunk_size),
        output_dir=args.output_dir,
        episode_tag=f"ep{args.episode_idx:06d}",
        title_suffix=f"  ({Path(args.repo_dir).name}, normalize={normalize_mode})",
    )

    # ---- 6. Optional dump for downstream tools (eval_smoothness.py etc.) ----
    if args.dump_json:
        dump_path = Path(args.output_dir) / f"ep{args.episode_idx:06d}_rollout.json"
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dump_path, "w") as f:
            json.dump(
                {
                    "control_freq": float(args.control_freq),
                    "instruction": instruction,
                    "episode_idx": int(args.episode_idx),
                    "inference_interval": int(args.inference_interval),
                    "inference_timesteps": inference_timesteps,
                    "episodes": [
                        {
                            "states": gt_states.tolist(),
                            "actions": gt_actions.tolist(),
                        }
                    ],
                    "predictions": [p.tolist() for p in predicted_actions],
                },
                f,
            )
        logging.info(f"Wrote rollout dump to {dump_path}")


def plot_action_curves(
    *,
    gt_actions: np.ndarray,
    predicted_actions: List[np.ndarray],
    inference_timesteps: List[int],
    state_dim: int,
    chunk_size: int,
    output_dir: str,
    episode_tag: str,
    title_suffix: str = "",
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = dexora_action_names(state_dim)
    T = gt_actions.shape[0]
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(predicted_actions), 1)))

    # Per-axis plots
    for axis in range(state_dim):
        plt.figure(figsize=(12, 5))
        plt.plot(np.arange(T), gt_actions[:, axis], "b-", linewidth=2,
                 label="Ground Truth", alpha=0.85)
        for i, (chunk, t0) in enumerate(zip(predicted_actions, inference_timesteps)):
            T_chunk = min(chunk_size, T - t0)
            if T_chunk <= 0:
                continue
            xs = np.arange(t0, t0 + T_chunk)
            plt.plot(xs, chunk[:T_chunk, axis], "--", color=colors[i % len(colors)],
                     linewidth=1.4, alpha=0.7,
                     label=f"pred @step {t0}")
        plt.xlabel("Timestep")
        plt.ylabel("Action value")
        plt.title(f"{episode_tag} — {names[axis]} (axis {axis}){title_suffix}")
        plt.grid(True, alpha=0.3)
        # Avoid blowing up the legend when many inference points are used.
        if len(predicted_actions) <= 12:
            plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        path = out_dir / f"{episode_tag}_axis_{axis:02d}_{names[axis]}.png"
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()

    # Summary 6x6 grid
    rows = 6
    cols = int(np.ceil(state_dim / rows))
    fig, ax_grid = plt.subplots(rows, cols, figsize=(4 * cols, 2.5 * rows))
    axes_flat = ax_grid.flatten()
    for axis in range(state_dim):
        ax = axes_flat[axis]
        ax.plot(np.arange(T), gt_actions[:, axis], "b-", linewidth=1, alpha=0.85)
        for i, (chunk, t0) in enumerate(zip(predicted_actions, inference_timesteps)):
            T_chunk = min(chunk_size, T - t0)
            if T_chunk <= 0:
                continue
            xs = np.arange(t0, t0 + T_chunk)
            ax.plot(xs, chunk[:T_chunk, axis], "--",
                    color=colors[i % len(colors)], linewidth=0.8, alpha=0.6)
        ax.set_title(f"{names[axis]}", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=6)
    for axis in range(state_dim, rows * cols):
        axes_flat[axis].set_visible(False)
    fig.suptitle(f"{episode_tag} — all {state_dim} action axes{title_suffix}",
                 fontsize=13)
    fig.tight_layout()
    summary_path = out_dir / f"{episode_tag}_summary.png"
    fig.savefig(summary_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    logging.info(f"Saved {state_dim} per-axis plots + summary to {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Open-loop action-curve evaluation for Dexora (paper Fig. 11).",
    )

    # Model
    p.add_argument("--model-path", required=True,
                   help="Stage-1 / Stage-3 checkpoint directory or pytorch_model.bin.")
    p.add_argument("--model-config-path", default="configs/base_400m.yaml",
                   help="Policy YAML used at training time.")
    p.add_argument("--text-encoder", default="google/t5-v1_1-xxl")
    p.add_argument("--vision-encoder", default="google/siglip-so400m-patch14-384")

    # Dataset
    p.add_argument("--repo-dir", required=True,
                   help="LeRobot v2.1 dataset directory (one task family, e.g. "
                        "data/Dexora_Real-World_Dataset/airbot_pick_and_place).")
    p.add_argument("--stats-file", default="new_lerobot_stats/dataset_statistics.json",
                   help="dataset_statistics.json used for per-dim min/max normalization "
                        "(must match what the checkpoint was trained on).")
    p.add_argument("--episode-idx", type=int, default=0,
                   help="Episode index within the LeRobot dataset.")
    p.add_argument("--instruction", default="",
                   help="Override the dataset-derived language goal (optional).")

    # Inference
    p.add_argument("--state-dim", type=int, default=36)
    p.add_argument("--chunk-size", type=int, default=32)
    p.add_argument("--img-history-size", type=int, default=1)
    p.add_argument("--control-freq", type=float, default=20.0)
    p.add_argument("--inference-interval", type=int, default=32,
                   help="Run one diffusion pass every N steps of the episode.")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Truncate the episode to this many steps (default: full).")
    p.add_argument("--no-normalize", action="store_true",
                   help="Disable per-dim normalization. Only safe for checkpoints "
                        "trained without stats_file; predictions will diverge "
                        "from GT otherwise.")

    # Output
    p.add_argument("--output-dir", default="eval_results",
                   help="Directory to write per-axis plots + summary grid.")
    p.add_argument("--dump-json", action="store_true",
                   help="Also dump the raw GT + predictions to a single JSON "
                        "file under --output-dir (consumable by eval_smoothness.py).")

    # Misc
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = get_args()
    run_eval(args)


if __name__ == "__main__":
    main()
