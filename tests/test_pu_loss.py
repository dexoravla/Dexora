"""Tests for the PU discriminator loss (Dexora Eq.(7) + DWBC variant).

We do not import ``train.train_scoring`` directly because it transitively
imports ``diffusers`` / ``accelerate`` / ``yaml`` etc., which may not be
available in a barebones CI environment. Instead we extract the function
via ``ast.get_source_segment``. This keeps the tests fast and avoids
unnecessary GPU/runtime deps.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE = (REPO_ROOT / "train" / "train_scoring.py").read_text()


def _extract(name: str):
    """Return ``name`` from train/train_scoring.py as a callable."""
    module = ast.parse(SOURCE)
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            src = ast.get_source_segment(SOURCE, node)
            assert src is not None
            ns: dict = {"torch": torch}
            exec(src, ns)
            return ns[name]
    raise RuntimeError(f"function {name!r} not found in train_scoring.py")


pu_loss_function = _extract("pu_loss_function")


def _make_batch(n_pos: int, n_neg: int, pos_score: float = 0.8, neg_score: float = 0.2):
    scores = torch.tensor([[pos_score]] * n_pos + [[neg_score]] * n_neg, dtype=torch.float32)
    labels = torch.tensor([1.0] * n_pos + [0.0] * n_neg, dtype=torch.float32)
    return scores, labels


class TestPaperVariant:
    def test_returns_scalar(self):
        scores, labels = _make_batch(2, 2)
        loss, info = pu_loss_function(scores, labels, eta=0.5, variant="paper")
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert info["variant"] == "paper"
        assert info["eta"] == 0.5
        assert info["num_expert"] == 2
        assert info["num_unlabeled"] == 2

    def test_paper_loss_is_non_negative(self):
        """The two-term BCE form (paper Eq.(7)) is a sum of -log probabilities
        and must always be >= 0."""
        for _ in range(20):
            scores = torch.rand(8, 1) * 0.8 + 0.1   # in [0.1, 0.9]
            labels = torch.randint(0, 2, (8,)).float()
            loss, _ = pu_loss_function(scores, labels, eta=0.5, variant="paper")
            assert loss.item() >= -1e-6

    def test_expert_pos_term_decreases_with_higher_score(self):
        scores_low, labels = _make_batch(4, 0, pos_score=0.2)
        scores_hi, _ = _make_batch(4, 0, pos_score=0.9)
        l_low, _ = pu_loss_function(scores_low, labels, eta=0.5, variant="paper")
        l_hi, _ = pu_loss_function(scores_hi, labels, eta=0.5, variant="paper")
        assert l_hi.item() < l_low.item()

    def test_unlabeled_term_decreases_with_lower_score(self):
        scores_hi = torch.full((4, 1), 0.9)
        scores_low = torch.full((4, 1), 0.2)
        labels = torch.zeros(4)
        l_hi, _ = pu_loss_function(scores_hi, labels, eta=0.5, variant="paper")
        l_low, _ = pu_loss_function(scores_low, labels, eta=0.5, variant="paper")
        assert l_low.item() < l_hi.item()


class TestDWBCVariant:
    def test_returns_scalar(self):
        scores, labels = _make_batch(2, 2)
        loss, info = pu_loss_function(scores, labels, eta=0.5, variant="dwbc")
        assert loss.ndim == 0
        assert info["variant"] == "dwbc"

    def test_dwbc_equals_paper_minus_subtractive_term(self):
        """DWBC == paper - eta * E_pos[-log(1-d)]. We check this identity
        directly using the returned breakdown."""
        scores, labels = _make_batch(4, 4)
        l_paper, info_p = pu_loss_function(scores, labels, eta=0.5, variant="paper")
        l_dwbc, info_d = pu_loss_function(scores, labels, eta=0.5, variant="dwbc")
        expected = l_paper.item() - info_p["expert_neg_term"]
        assert abs(l_dwbc.item() - expected) < 1e-5


class TestEdgeCases:
    def test_all_unlabeled(self):
        scores = torch.full((4, 1), 0.5)
        labels = torch.zeros(4)
        loss, info = pu_loss_function(scores, labels, eta=0.5, variant="paper")
        assert info["num_expert"] == 0
        assert info["expert_pos_term"] == 0.0
        assert info["expert_neg_term"] == 0.0
        assert torch.isfinite(loss)

    def test_all_expert(self):
        scores = torch.full((4, 1), 0.5)
        labels = torch.ones(4)
        loss, info = pu_loss_function(scores, labels, eta=0.5, variant="paper")
        assert info["num_unlabeled"] == 0
        assert info["unlabeled_term"] == 0.0
        assert torch.isfinite(loss)

    def test_invalid_variant(self):
        scores, labels = _make_batch(1, 1)
        with pytest.raises(ValueError, match="Unknown PU variant"):
            pu_loss_function(scores, labels, eta=0.5, variant="nonsense")

    def test_no_nan_on_saturated_scores(self):
        scores = torch.tensor([[0.0], [1.0]])
        labels = torch.tensor([1.0, 0.0])
        loss, _ = pu_loss_function(scores, labels, eta=0.5, variant="paper")
        # Scaling to [0.1, 0.9] inside the function should prevent NaNs.
        assert torch.isfinite(loss)
