"""Tests for ``models/sample_weighting.py`` (DWBC + warm-up)."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from models.sample_weighting import (  # noqa: E402
    dwbc_score_to_weight,
    scores_to_train_weights,
    warmup_weights,
)


class TestDWBCWeightMapping:
    """The DWBC mapping w = d / (eta*(1-d) + d) must satisfy a handful of
    mathematical properties that we rely on downstream."""

    def test_monotonic_in_score(self):
        s = torch.linspace(0.01, 0.99, steps=20)
        w = dwbc_score_to_weight(s, eta=0.5)
        diffs = w[1:] - w[:-1]
        assert torch.all(diffs >= 0), "DWBC weight must be monotonically non-decreasing in d"

    def test_endpoints(self):
        # d -> 0  =>  w -> 0   (after clamping but before w_min)
        # d -> 1  =>  w -> 1
        w_lo = dwbc_score_to_weight(torch.tensor([1e-6]), eta=0.5, w_min=0.0, w_max=10.0)
        w_hi = dwbc_score_to_weight(torch.tensor([1.0 - 1e-6]), eta=0.5, w_min=0.0, w_max=10.0)
        assert w_lo.item() < 1e-3
        assert math.isclose(w_hi.item(), 1.0, rel_tol=1e-3)

    def test_eta_equals_one_yields_identity(self):
        s = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        w = dwbc_score_to_weight(s, eta=1.0)
        # With eta = 1: w = d / (1 - d + d) = d.
        assert torch.allclose(w, s, atol=1e-5)

    def test_clamping(self):
        s = torch.tensor([0.999999])
        w = dwbc_score_to_weight(s, eta=0.01, w_min=0.0, w_max=2.0)
        assert w.item() <= 2.0 + 1e-6

    def test_dtype_round_trip(self):
        for dtype in (torch.float32, torch.float64):
            s = torch.tensor([0.5], dtype=dtype)
            w = dwbc_score_to_weight(s, eta=0.5)
            assert w.dtype == dtype


class TestWarmupWeights:
    def test_warmup_zero_is_identity(self):
        w = torch.tensor([0.1, 1.0, 3.0])
        out = warmup_weights(w, global_step=0, warmup_steps=0)
        assert torch.equal(out, w)

    def test_warmup_start_is_ones(self):
        w = torch.tensor([0.1, 1.0, 3.0])
        out = warmup_weights(w, global_step=0, warmup_steps=10)
        assert torch.allclose(out, torch.ones_like(w))

    def test_warmup_end_is_original(self):
        w = torch.tensor([0.1, 1.0, 3.0])
        out = warmup_weights(w, global_step=10, warmup_steps=10)
        assert torch.allclose(out, w)

    def test_warmup_midpoint_is_linear(self):
        w = torch.tensor([0.0, 2.0])
        out = warmup_weights(w, global_step=5, warmup_steps=10)
        # at progress=0.5: 1 + 0.5*(w - 1) = (1 + w)/2
        expected = (torch.ones_like(w) + w) / 2.0
        assert torch.allclose(out, expected, atol=1e-6)

    def test_after_warmup_clamps(self):
        w = torch.tensor([1.5])
        # global_step way past warmup -> still equal to w (no overshoot).
        out = warmup_weights(w, global_step=10_000, warmup_steps=100)
        assert torch.allclose(out, w)


class TestScoresToTrainWeights:
    def test_none_scores_returns_ones(self):
        w = scores_to_train_weights(None, fallback_shape=(4,), dtype=torch.float32)
        assert w.shape == (4,)
        assert torch.allclose(w, torch.ones(4))

    def test_composition_with_warmup(self):
        s = torch.tensor([0.1, 0.5, 0.9])
        w0 = scores_to_train_weights(s, eta=0.5, warmup_steps=10, global_step=0)
        w10 = scores_to_train_weights(s, eta=0.5, warmup_steps=10, global_step=10)
        assert torch.allclose(w0, torch.ones_like(s))
        # at step 10 we are at the un-warmed DWBC mapping
        expected = dwbc_score_to_weight(s, eta=0.5)
        assert torch.allclose(w10, expected, atol=1e-5)
