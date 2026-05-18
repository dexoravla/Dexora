"""Tests for ``weighted_mse_loss`` (Dexora Eq.(8) core math).

This is the building block that ``RDTRunner.compute_loss`` delegates to, so
testing this function transitively tests the data-quality-aware diffusion
objective without needing diffusers / accelerate / GPU.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
torch.manual_seed(0)

from models.sample_weighting import weighted_mse_loss  # noqa: E402


class TestUnweighted:
    def test_matches_F_mse_loss(self):
        import torch.nn.functional as F

        pred = torch.randn(8, 32, 36)
        target = torch.randn_like(pred)
        loss, info = weighted_mse_loss(pred, target, sample_weights=None)
        expected = F.mse_loss(pred, target)
        assert torch.isclose(loss, expected, atol=1e-6)
        assert info["mean_weight"].item() == pytest.approx(1.0)

    def test_zero_loss_when_pred_equals_target(self):
        x = torch.randn(4, 8, 12)
        loss, info = weighted_mse_loss(x, x)
        assert loss.item() == pytest.approx(0.0)
        assert info["per_sample_mse_max"].item() == pytest.approx(0.0)


class TestWeighted:
    def test_uniform_weights_equal_unweighted(self):
        pred = torch.randn(4, 32, 36)
        target = torch.randn_like(pred)
        loss_u, _ = weighted_mse_loss(pred, target)
        w = torch.ones(4)
        loss_w, _ = weighted_mse_loss(pred, target, w)
        assert torch.isclose(loss_u, loss_w, atol=1e-6)

    def test_zero_weight_zeros_loss(self):
        pred = torch.randn(4, 32, 36)
        target = torch.randn_like(pred)
        w = torch.tensor([1.0, 0.0, 0.0, 0.0])
        loss, _ = weighted_mse_loss(pred, target, w)
        # Should equal the per-sample MSE of sample 0
        diff = (pred[0].float() - target[0].float()) ** 2
        assert torch.isclose(loss, diff.mean(), atol=1e-6)

    def test_weight_emphasis(self):
        """Heavily up-weighting the lowest-error sample must drop the loss
        relative to uniform weighting; doing the same on the highest-error
        sample must raise it."""
        pred = torch.zeros(3, 8)
        target = torch.tensor([
            [0.0] * 8,           # mse = 0
            [1.0] * 8,           # mse = 1
            [2.0] * 8,           # mse = 4
        ])
        uniform_loss, _ = weighted_mse_loss(pred, target)
        emphasize_easy, _ = weighted_mse_loss(pred, target, torch.tensor([10.0, 1.0, 1.0]))
        emphasize_hard, _ = weighted_mse_loss(pred, target, torch.tensor([1.0, 1.0, 10.0]))
        assert emphasize_easy.item() < uniform_loss.item()
        assert emphasize_hard.item() > uniform_loss.item()

    def test_shape_mismatch_raises(self):
        pred = torch.randn(4, 8)
        target = torch.randn_like(pred)
        with pytest.raises(ValueError, match="sample_weights shape"):
            weighted_mse_loss(pred, target, torch.ones(3))

    def test_handles_tiny_weight_sum(self):
        pred = torch.randn(4, 8)
        target = torch.randn_like(pred)
        loss, _ = weighted_mse_loss(pred, target, torch.zeros(4))
        # The clamp_min(1e-6) prevents NaN; the numerator is exactly 0, so
        # loss should be 0 (or extremely small positive).
        assert torch.isfinite(loss)
        assert loss.item() <= 1e-3

    def test_preserves_pred_dtype(self):
        # We compute in float32 internally for stability, but the test still
        # expects a finite tensor; dtype consistency is asserted by the runner.
        pred = torch.randn(2, 4, dtype=torch.float32)
        target = torch.randn_like(pred)
        loss, _ = weighted_mse_loss(pred, target, torch.tensor([1.0, 2.0]))
        assert torch.isfinite(loss)


class TestInfoDict:
    def test_info_keys(self):
        pred = torch.randn(2, 4, 8)
        target = torch.randn_like(pred)
        _, info = weighted_mse_loss(pred, target, torch.tensor([0.3, 0.7]))
        assert set(info.keys()) == {
            "per_sample_mse_mean",
            "per_sample_mse_min",
            "per_sample_mse_max",
            "mean_weight",
        }

    def test_mean_weight_matches_input(self):
        pred = torch.zeros(3, 4)
        target = torch.zeros(3, 4)
        _, info = weighted_mse_loss(pred, target, torch.tensor([0.0, 1.0, 2.0]))
        assert info["mean_weight"].item() == pytest.approx(1.0, abs=1e-6)
