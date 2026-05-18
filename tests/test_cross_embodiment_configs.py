"""Schema validation for ``configs/cross_embodiment/*.yaml``.

These configs are consumed at fine-tuning time to set ``state_elem_mask`` and
``camera_active_mask`` for the EC-1/EC-2/EC-3 experiments (Dexora §IV-C).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parent.parent
EC_DIR = REPO_ROOT / "configs" / "cross_embodiment"

EXPECTED = [
    ("ec1_franka.yaml", 7),
    ("ec2_aloha.yaml", 14),
    ("ec3_g1_inspire.yaml", 13),
]


@pytest.mark.parametrize("filename,expected_dof", EXPECTED)
def test_state_mask_shape_and_sum(filename: str, expected_dof: int) -> None:
    path = EC_DIR / filename
    cfg = yaml.safe_load(path.read_text())
    emb = cfg["embodiment"]
    mask: List[int] = emb["state_elem_mask"]

    assert isinstance(mask, list), f"{filename}: state_elem_mask must be a list"
    assert len(mask) == 36, f"{filename}: state_elem_mask must have 36 entries"
    assert all(v in (0, 1) for v in mask), f"{filename}: state_elem_mask must be 0/1"
    assert sum(mask) == emb["dof"], (
        f"{filename}: sum(state_elem_mask)={sum(mask)} != declared dof={emb['dof']}"
    )
    assert emb["dof"] == expected_dof, (
        f"{filename}: declared dof={emb['dof']} does not match expected {expected_dof}"
    )


@pytest.mark.parametrize("filename,_dof", EXPECTED)
def test_camera_mask_shape(filename: str, _dof: int) -> None:
    cfg = yaml.safe_load((EC_DIR / filename).read_text())
    cam = cfg["embodiment"]["camera_active_mask"]
    assert isinstance(cam, list)
    assert len(cam) == 4, f"{filename}: camera_active_mask must have 4 entries"
    assert all(v in (0, 1) for v in cam)
    assert sum(cam) >= 1, f"{filename}: at least one camera must be active"


@pytest.mark.parametrize("filename,_dof", EXPECTED)
def test_consistent_with_base_400m(filename: str, _dof: int) -> None:
    """Cross-embodiment configs must share the 400M architecture so that the
    Stage-3 weights can be loaded without resizing."""
    ec = yaml.safe_load((EC_DIR / filename).read_text())
    base = yaml.safe_load((REPO_ROOT / "configs" / "base_400m.yaml").read_text())
    assert ec["common"]["state_dim"] == base["common"]["state_dim"]
    assert ec["common"]["action_chunk_size"] == base["common"]["action_chunk_size"]
    assert ec["model"]["rdt"]["hidden_size"] == base["model"]["rdt"]["hidden_size"]
    assert ec["model"]["rdt"]["depth"] == base["model"]["rdt"]["depth"]
    assert ec["model"]["rdt"]["num_heads"] == base["model"]["rdt"]["num_heads"]
