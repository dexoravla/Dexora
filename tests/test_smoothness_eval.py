"""Tests for ``scripts/eval_smoothness.py``.

We construct synthetic rollouts whose Acc / Jerk we can verify analytically
and check that ``evaluate()`` recovers the right ballpark numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

import importlib.util  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "eval_smoothness", REPO_ROOT / "scripts" / "eval_smoothness.py"
)
es = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(es)  # type: ignore[union-attr]


def _make_stats(state_dim: int = 36, tmp_path: Path = Path("/tmp")) -> Path:
    stats = {
        "state": {
            "percentile_1": [-1.0] * state_dim,
            "percentile_99": [1.0] * state_dim,
        }
    }
    p = tmp_path / "stats.json"
    p.write_text(json.dumps(stats))
    return p


def _make_rollout(
    fps: float,
    n_steps: int,
    state_dim: int,
    motion: str,
    tmp_path: Path,
) -> Path:
    t = np.arange(n_steps) / fps
    if motion == "static":
        traj = np.zeros((n_steps, state_dim))
    elif motion == "linear":
        traj = np.tile(t[:, None], (1, state_dim))  # constant velocity
    elif motion == "sine":
        traj = np.tile(np.sin(2 * np.pi * t)[:, None], (1, state_dim))
    else:
        raise ValueError(motion)
    payload = {
        "control_freq": fps,
        "episodes": [{"states": traj.tolist()}],
    }
    p = tmp_path / f"rollout_{motion}.json"
    p.write_text(json.dumps(payload))
    return p


class TestAccJerkRms:
    def test_static_episode_is_zero(self):
        traj = np.zeros((50, 36))
        a, j = es.acc_jerk_rms(traj, dt=0.05)
        assert a == pytest.approx(0.0, abs=1e-8)
        assert j == pytest.approx(0.0, abs=1e-8)

    def test_short_returns_nan(self):
        traj = np.zeros((5, 36))
        a, j = es.acc_jerk_rms(traj, dt=0.05)
        assert np.isnan(a)
        assert np.isnan(j)

    def test_linear_motion_has_zero_acc(self):
        # Constant velocity -> acceleration and jerk both ~0 (up to FP error).
        n, d, fps = 60, 36, 20
        t = np.arange(n) / fps
        traj = np.tile(t[:, None], (1, d))
        a, j = es.acc_jerk_rms(traj, dt=1.0 / fps)
        assert a == pytest.approx(0.0, abs=1e-6)
        assert j == pytest.approx(0.0, abs=1e-6)

    def test_sine_motion_has_finite_acc(self):
        n, d, fps = 200, 36, 20
        t = np.arange(n) / fps
        traj = np.tile(np.sin(2 * np.pi * t)[:, None], (1, d))
        a, j = es.acc_jerk_rms(traj, dt=1.0 / fps)
        assert a > 0
        assert j > 0
        # For sin(2 pi t) the analytic peak acceleration is (2 pi)^2 ~ 39.5.
        # RMS over the inner slice will be ~ 39.5 / sqrt(2) ~ 27.9.
        assert 20.0 < a < 40.0


class TestEvaluate:
    def test_static_aggregate_is_zero(self, tmp_path):
        stats = _make_stats(tmp_path=tmp_path)
        roll = _make_rollout(20.0, 50, 36, "static", tmp_path)
        lo, span = es.load_stats(str(stats))
        m = es.evaluate(str(roll), lo, span)
        assert m["num_episodes"] == 1
        assert m["acc_mean"] == pytest.approx(0.0, abs=1e-6)
        assert m["jerk_mean"] == pytest.approx(0.0, abs=1e-6)

    def test_sine_aggregate_is_nonzero(self, tmp_path):
        stats = _make_stats(tmp_path=tmp_path)
        roll = _make_rollout(20.0, 200, 36, "sine", tmp_path)
        lo, span = es.load_stats(str(stats))
        m = es.evaluate(str(roll), lo, span)
        assert m["num_episodes"] == 1
        assert m["acc_mean"] > 0
        assert m["jerk_mean"] > 0
