#!/usr/bin/env python3
"""
Evaluate the smoothness metrics reported in Tab. III of the Dexora paper
("Effect of the discriminator model"):

    * Mean normalized joint acceleration  (Acc. ↓)
    * Mean normalized joint jerk          (Jerk ↓)

The script reads one or more rollout files containing per-step joint states
(or actions), normalizes per-dimension using the dataset statistics, then
computes the RMS quantities defined by Dexora Eq.(2)–(3) over each episode
and averages across all episodes. We average over T-6 timesteps and D dims,
matching the paper.

Input rollout format
--------------------
Each file should contain a single JSON object:

    {
        "control_freq": 20,   # Hz (optional; default 20)
        "episodes": [
            {"states":  [[..36 floats..], ...]  },
            {"actions": [[..36 floats..], ...]  },
            ...
        ]
    }

For each episode we use ``states`` when present, otherwise fall back to
``actions``.

Stats file
----------
A ``dataset_statistics.json`` produced by the dataset preprocessing
pipeline. We use ``percentile_1`` and ``percentile_99`` for min–max
normalization (consistent with ``analyze_episode_quality.py`` and the
dataset loader).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

__all__ = [
    "load_stats",
    "normalize_states",
    "acc_jerk_rms",
    "evaluate",
]


def load_stats(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "r") as f:
        stats = json.load(f)
    s = stats.get("state", stats)
    lo = np.array(s["percentile_1"], dtype=np.float64)
    hi = np.array(s["percentile_99"], dtype=np.float64)
    span = np.where((hi - lo) == 0, 1.0, hi - lo)
    return lo, span


def normalize_states(states: np.ndarray, lo: np.ndarray, span: np.ndarray) -> np.ndarray:
    return (states - lo) / span


def acc_jerk_rms(states: np.ndarray, dt: float) -> Tuple[float, float]:
    """Centered finite differences per Eq.(1)-(3). Returns (Acc_ep, Jerk_ep)."""
    if states.shape[0] < 7:
        return float("nan"), float("nan")

    # v_t = (s_{t+1} - s_{t-1}) / (2dt)
    v = (states[2:] - states[:-2]) / (2.0 * dt)              # length T-2
    a = (v[2:] - v[:-2]) / (2.0 * dt)                         # length T-4
    j = (a[2:] - a[:-2]) / (2.0 * dt)                         # length T-6

    # RMS across (time, dim). Paper averages t in [4, T-3] and k in [1, D],
    # which is the inner block of length T-6 across all D dims.
    acc_rms = float(np.sqrt(np.mean(a[1:-1] ** 2)))           # align to inner T-6 slice
    jerk_rms = float(np.sqrt(np.mean(j ** 2)))
    return acc_rms, jerk_rms


def evaluate(rollout_file: str, lo: np.ndarray, span: np.ndarray) -> Dict[str, float]:
    with open(rollout_file, "r") as f:
        rollout = json.load(f)
    fps = float(rollout.get("control_freq", 20))
    dt = 1.0 / fps
    eps = rollout.get("episodes", [])

    acc_list, jerk_list = [], []
    for ep in eps:
        data = ep.get("states", ep.get("actions"))
        if data is None or len(data) < 7:
            continue
        states = np.asarray(data, dtype=np.float64)
        states = normalize_states(states, lo, span)
        a_rms, j_rms = acc_jerk_rms(states, dt)
        if np.isfinite(a_rms) and np.isfinite(j_rms):
            acc_list.append(a_rms)
            jerk_list.append(j_rms)

    return {
        "num_episodes": len(acc_list),
        "acc_mean": float(np.mean(acc_list)) if acc_list else float("nan"),
        "acc_std": float(np.std(acc_list)) if acc_list else float("nan"),
        "jerk_mean": float(np.mean(jerk_list)) if jerk_list else float("nan"),
        "jerk_std": float(np.std(jerk_list)) if jerk_list else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute Dexora Tab. III smoothness metrics (Acc., Jerk) on inference rollouts."
    )
    parser.add_argument(
        "rollouts", nargs="+",
        help="One or more JSON rollout files (see module docstring for the schema).",
    )
    parser.add_argument(
        "--stats_file", type=str, required=True,
        help="dataset_statistics.json used for per-dim min-max normalization.",
    )
    parser.add_argument(
        "--label", type=str, default=None,
        help="Optional label printed alongside each rollout (e.g. 'w/ disc').",
    )
    parser.add_argument(
        "--output_json", type=str, default=None,
        help="Optionally write the aggregated metrics to this JSON file.",
    )
    args = parser.parse_args()

    lo, span = load_stats(args.stats_file)
    print(f"Loaded stats with state_dim={lo.shape[0]} from {args.stats_file}")

    all_results: Dict[str, Dict[str, float]] = {}
    for f in args.rollouts:
        m = evaluate(f, lo, span)
        tag = f"{args.label}::" if args.label else ""
        print(
            f"{tag}{Path(f).name}: "
            f"episodes={m['num_episodes']}, "
            f"Acc={m['acc_mean']:.4f}±{m['acc_std']:.4f}, "
            f"Jerk={m['jerk_mean']:.4f}±{m['jerk_std']:.4f}"
        )
        all_results[f] = m

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump({"label": args.label, "results": all_results}, f, indent=2)
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
