#!/usr/bin/env python3
"""
Stage-2b: Post-validation of pre-screened demonstrations (Dexora §III-C).

The paper specifies a two-stage filter for assembling the positive set used in
discriminator training:

  Spre  = Low-20%(Aep) ∩ Low-20%(Jep)                    (pre-screening)
  Shigh = { τ ∈ Spre : Success(τ)=1  ∧  CollisionFree(τ)=1 }   (post-validation)

`analyze_episode_quality.py` produces Spre. This script then **replays** every
episode in Spre and applies a configurable verifier (default: a MuJoCo digital
twin via :func:`mujoco_replay_verifier`) to derive Shigh.

When a MuJoCo twin is unavailable (e.g. before the simulator is open-sourced),
two cheap built-in verifiers are provided:

* ``--verifier  trust_spre``  → assume every Spre episode is a positive
  (debugging fallback identical to what the old code did implicitly).
* ``--verifier  energy``  → use kinematic energy/state-norm heuristics
  (per-episode acceleration spikes & out-of-range states) as a proxy.

Output: a JSON listing the verified high-quality episode indices and the
per-episode verdicts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

__all__ = [
    "VerifierResult",
    "Verifier",
    "trust_spre_verifier",
    "energy_heuristic_verifier",
    "mujoco_replay_verifier_factory",
    "VERIFIER_REGISTRY",
]

# Optional dependency; only needed when reading LeRobot episodes for verifiers
# that look at per-frame state/action.
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lerobot", "src"))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
except Exception:
    LeRobotDataset = None  # type: ignore


VerifierResult = Dict[str, object]
Verifier = Callable[[int, np.ndarray, np.ndarray], VerifierResult]


# ---------------------------------------------------------------------------
# Built-in verifiers
# ---------------------------------------------------------------------------

def trust_spre_verifier(ep_idx: int, states: np.ndarray, actions: np.ndarray) -> VerifierResult:
    """Debug-only verifier that marks every Spre episode as high-quality."""
    return {
        "success": True,
        "collision_free": True,
        "note": "trust_spre (no real verification performed)",
    }


def energy_heuristic_verifier(
    ep_idx: int,
    states: np.ndarray,
    actions: np.ndarray,
    *,
    state_lo: float = -3.5,
    state_hi: float = 3.5,
    acc_spike_z: float = 6.0,
) -> VerifierResult:
    """
    Cheap proxy for "success + collision-free" when no simulator is available.

    Two rules:

    1.  No frame state should explode out of a typical normalized range.
        (Spike outside [state_lo, state_hi] suggests a tracking glitch or
        clipping artifact.)

    2.  No per-joint acceleration spike should exceed ``acc_spike_z`` standard
        deviations over the episode's own distribution. Sharp spikes correlate
        well with collisions or operator yanks in the teleop data.
    """
    if states.ndim != 2 or states.shape[0] < 5:
        return {"success": False, "collision_free": False, "note": "too short"}

    s_min = float(states.min())
    s_max = float(states.max())
    in_range = (s_min >= state_lo) and (s_max <= state_hi)

    accel = np.diff(states, n=2, axis=0)
    mu = accel.mean(axis=0, keepdims=True)
    sigma = accel.std(axis=0, keepdims=True) + 1e-6
    z = np.abs((accel - mu) / sigma)
    no_spike = bool((z.max() < acc_spike_z))

    return {
        "success": bool(in_range),
        "collision_free": no_spike,
        "max_abs_state": s_max,
        "min_state": s_min,
        "max_accel_zscore": float(z.max()),
    }


def mujoco_replay_verifier_factory(twin_module: Optional[str] = None) -> Verifier:
    """
    Construct a real verifier that loads a MuJoCo digital twin and open-loop
    replays each action sequence. Returns a stub when the user has not
    plugged in their MuJoCo bindings yet.

    To plug in:
        --verifier mujoco --twin_module path.to.your.replay_module

    The module is expected to expose:
        ``replay(states: np.ndarray, actions: np.ndarray,
                 task_id: int) -> Dict[str, object]``
    that returns at least ``{"success": bool, "collision_free": bool}``.
    """
    if twin_module is None:
        def _stub(ep_idx, states, actions):
            return {
                "success": True,
                "collision_free": True,
                "note": "mujoco verifier stub – plug in your --twin_module",
            }
        return _stub

    import importlib
    mod = importlib.import_module(twin_module)
    if not hasattr(mod, "replay"):
        raise ImportError(f"--twin_module {twin_module} must expose `replay(states, actions, task_id)`.")

    def _real(ep_idx, states, actions):
        return mod.replay(states=states, actions=actions, task_id=ep_idx)
    return _real


VERIFIER_REGISTRY: Dict[str, Callable[[argparse.Namespace], Verifier]] = {
    "trust_spre": lambda args: trust_spre_verifier,
    "energy": lambda args: energy_heuristic_verifier,
    "mujoco": lambda args: mujoco_replay_verifier_factory(args.twin_module),
}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _load_episode(dataset: "LeRobotDataset", ep_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (states, actions) numpy arrays for the given episode."""
    ep_from = int(dataset.episode_data_index["from"][ep_idx])
    ep_to = int(dataset.episode_data_index["to"][ep_idx])
    states, actions = [], []
    for i in range(ep_from, ep_to):
        sample = dataset[i]
        s = sample["states"]
        a = sample.get("actions", None)
        states.append(s.numpy() if hasattr(s, "numpy") else np.asarray(s))
        if a is not None:
            actions.append(a.numpy() if hasattr(a, "numpy") else np.asarray(a))
    states_np = np.stack(states, axis=0)
    actions_np = np.stack(actions, axis=0) if actions else np.zeros((0, states_np.shape[-1]))
    return states_np, actions_np


def main():
    parser = argparse.ArgumentParser(description="Validate Spre → Shigh by replay.")
    parser.add_argument("--pre_screening_file", type=str,
                        default="new_lerobot_jerk/complete_analysis_results.json",
                        help="Output of analyze_episode_quality.py.")
    parser.add_argument("--lerobot_root", type=str,
                        default="dataprocess/output/airbot_dexterous_bimanual_dexterous_manipulation",
                        help="LeRobot dataset directory (only needed for energy/mujoco verifiers).")
    parser.add_argument("--output_file", type=str, default="shigh_episodes.json")
    parser.add_argument("--verifier", type=str, default="trust_spre",
                        choices=list(VERIFIER_REGISTRY.keys()))
    parser.add_argument("--twin_module", type=str, default=None,
                        help="Python module that exposes replay(...) for --verifier mujoco.")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Process at most this many Spre episodes (-1 = all).")
    args = parser.parse_args()

    with open(args.pre_screening_file, "r") as f:
        screening = json.load(f)
    spre = list(screening["filtering_thresholds"]["valid_episodes"])
    if args.limit > 0:
        spre = spre[: args.limit]
    print(f"Loaded {len(spre)} episodes from Spre.")

    verifier = VERIFIER_REGISTRY[args.verifier](args)

    dataset = None
    if args.verifier in {"energy", "mujoco"}:
        if LeRobotDataset is None:
            raise RuntimeError("LeRobot is not importable but the chosen verifier needs it.")
        dataset = LeRobotDataset("", args.lerobot_root,
                                 delta_timestamps={"states": [0]},
                                 video_backend="pyav")
        # Disable video loading for faster iteration.
        object.__setattr__(dataset, "video_keys", [])
        object.__setattr__(dataset, "image_transforms", None)

    shigh: List[int] = []
    verdicts: Dict[int, VerifierResult] = {}

    t0 = time.time()
    for ep_idx in tqdm(spre, desc="Replaying Spre"):
        if dataset is None:
            verdict = verifier(ep_idx, np.zeros((0, 0)), np.zeros((0, 0)))
        else:
            try:
                states, actions = _load_episode(dataset, ep_idx)
            except Exception as e:
                verdict = {"success": False, "collision_free": False, "note": f"load error: {e}"}
                verdicts[ep_idx] = verdict
                continue
            verdict = verifier(ep_idx, states, actions)
        verdicts[ep_idx] = verdict
        if bool(verdict.get("success")) and bool(verdict.get("collision_free")):
            shigh.append(int(ep_idx))

    elapsed = time.time() - t0
    print(f"Verifier '{args.verifier}' done in {elapsed:.1f}s")
    print(f"  Spre  -> {len(spre)} episodes")
    print(f"  Shigh -> {len(shigh)} episodes ({100*len(shigh)/max(1,len(spre)):.1f}%)")

    output = {
        "verifier": args.verifier,
        "pre_screening_file": args.pre_screening_file,
        "num_spre": len(spre),
        "num_shigh": len(shigh),
        "shigh_episodes": shigh,
        "verdicts": {str(k): v for k, v in verdicts.items()},
    }
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"Saved Shigh to {args.output_file}")


if __name__ == "__main__":
    main()
