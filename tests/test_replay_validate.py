"""Tests for ``replay_validate.py`` verifiers (no MuJoCo, CPU-only)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from scripts.replay_validate import (  # noqa: E402
    VERIFIER_REGISTRY,
    energy_heuristic_verifier,
    mujoco_replay_verifier_factory,
    trust_spre_verifier,
)


class TestTrustSpre:
    def test_always_passes(self):
        v = trust_spre_verifier(0, np.zeros((10, 36)), np.zeros((10, 36)))
        assert v["success"] is True
        assert v["collision_free"] is True


class TestEnergyHeuristic:
    def test_smooth_episode_passes(self):
        # Smooth ramp inside [-1, 1] -> in-range and no acceleration spikes.
        t = np.linspace(0, 1, 50)
        states = np.stack([np.sin(2 * np.pi * t)] * 36, axis=-1)
        v = energy_heuristic_verifier(0, states, np.zeros_like(states))
        assert v["success"] is True
        assert v["collision_free"] is True

    def test_out_of_range_fails(self):
        states = np.full((20, 36), 100.0)
        v = energy_heuristic_verifier(0, states, np.zeros_like(states))
        assert v["success"] is False

    def test_spike_rejects_episode(self):
        # A glitchy episode with both an out-of-range value and a sharp spike
        # must be rejected (either `success=False` or `collision_free=False`).
        states = np.zeros((30, 36))
        states[15, 0] = 50.0
        v = energy_heuristic_verifier(0, states, np.zeros_like(states))
        assert (v["success"] is False) or (v["collision_free"] is False)

    def test_sharp_jitter_breaks_collision_free(self):
        # An alternating in-range jitter creates large per-step accelerations
        # without inflating the global std uniformly -> z >> 6 on average.
        rng = np.random.default_rng(0)
        # Mostly quiet trajectory with a small handful of large isolated swings.
        states = 0.01 * rng.standard_normal((60, 36))
        states[40, 0] = 3.0  # within in-range bound but very sharp vs. the noise floor
        v = energy_heuristic_verifier(0, states, np.zeros_like(states))
        assert v["success"] is True            # values stay within [-3.5, 3.5]
        assert v["collision_free"] is False    # but the spike is statistically anomalous

    def test_too_short(self):
        v = energy_heuristic_verifier(0, np.zeros((3, 36)), np.zeros((3, 36)))
        assert v["success"] is False
        assert v["collision_free"] is False


class TestMujocoFactory:
    def test_stub_passes_through(self):
        verifier = mujoco_replay_verifier_factory(twin_module=None)
        v = verifier(0, np.zeros((5, 36)), np.zeros((5, 36)))
        assert v["success"] is True
        assert "mujoco" in v["note"].lower()

    def test_missing_module_raises(self):
        with pytest.raises((ImportError, ModuleNotFoundError)):
            mujoco_replay_verifier_factory(twin_module="nonexistent.module.zzz")


class TestRegistry:
    def test_known_keys(self):
        assert set(VERIFIER_REGISTRY.keys()) == {"trust_spre", "energy", "mujoco"}


def test_end_to_end_trust_spre(tmp_path):
    """Smoke test: feed a small synthetic Spre JSON to replay_validate via its
    Python API, using the trust_spre verifier (no dataset / no MuJoCo)."""
    # Build a fake screening file.
    spre = {"filtering_thresholds": {"valid_episodes": [0, 1, 2]}}
    spre_path = tmp_path / "spre.json"
    spre_path.write_text(json.dumps(spre))

    # Recreate what main() does, minus argparse, with a hand-rolled call.
    import importlib

    rv = importlib.import_module("scripts.replay_validate")
    verifier = rv.VERIFIER_REGISTRY["trust_spre"](None)

    shigh: list = []
    for ep in spre["filtering_thresholds"]["valid_episodes"]:
        v = verifier(ep, np.zeros((0, 0)), np.zeros((0, 0)))
        if v["success"] and v["collision_free"]:
            shigh.append(ep)

    assert shigh == [0, 1, 2]
