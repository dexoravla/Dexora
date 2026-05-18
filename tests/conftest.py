"""Shared pytest fixtures and path setup."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Make the top-level packages importable when running ``pytest`` from any
# working directory.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Also expose lerobot/src for tests that exercise the vendored loader.
LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
if LEROBOT_SRC.is_dir() and str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

# Belt-and-braces: do not let tests accidentally hit the network or any GPU.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
