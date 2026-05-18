"""Pytest suite for Dexora.

The tests here are intentionally CPU-only and self-contained so they can run
in GitHub Actions without GPUs or downloaded model weights. Tests that need
real checkpoints / large data should live under ``tests/integration/`` and be
marked with ``@pytest.mark.slow``.
"""
