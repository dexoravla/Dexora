# Contributing to Dexora

Thank you for your interest in Dexora! This repository hosts the open-source
release of our ICRA 2026 paper *Dexora: Open-source VLA for High-DoF Bimanual
Dexterity*. Contributions are welcome — from bug reports and documentation
fixes to new dexterous tasks, additional embodiment configs, and training-
recipe ablations.

## Getting Set Up

```bash
git clone https://github.com/Augety7777/Dexora-VLA.git
cd Dexora-VLA

# Create an env (any flavour is fine; we test against 3.10/3.11)
conda create -n dexora python=3.10 -y
conda activate dexora

# Install the package + dev tooling
pip install -e ".[dev]"

# Hook the linter / formatter into your commits
pre-commit install
```

To pull the SigLip / T5 encoders + sample data, follow the `## Downloads`
section in [`README.md`](README.md).

## Repository Layout (TL;DR)

| Path | Purpose |
|---|---|
| `models/` | Diffusion-transformer policy + discriminator + DWBC weighting |
| `train/` | Stage-1 pretrain, stage-2 discriminator, stage-3 quality-aware post-train |
| `data/` | Dataset adapters (BSON, LeRobot, EgoDex, HDF5) |
| `configs/` | YAML hyperparameter files (`base_400m.yaml` is the paper spec) |
| `scripts/` | Eval & visualization (`eval_smoothness.py`, ...) |
| `tools/` | Repo maintenance scripts (`release_check.py`) |
| `tests/` | Pytest suite for the small, deterministic pieces |
| `lerobot/` | Vendored LeRobot copy (Apache-2.0, see its own LICENSE) |

## Coding Conventions

* **Style.** We use [ruff](https://docs.astral.sh/ruff/) for lint + import
  sorting and [black](https://github.com/psf/black) for formatting. Both are
  configured in [`pyproject.toml`](pyproject.toml). Hit `pre-commit run -a`
  before pushing.
* **Type hints.** New public APIs in `models/`, `train/`, `scripts/` and
  `tools/` should be annotated. We do not enforce strict typing for the
  vendored LeRobot tree or the legacy preprocessing scripts.
* **Docstrings.** Use Google or NumPy style. Always document the **shape**
  and **dtype** of tensor parameters — `compute_loss` is the gold standard
  to look at.
* **Side effects.** Do not put module-level code that touches `cuda` /
  network /filesystem; gate it behind `if __name__ == "__main__":` or
  inside a function.
* **Comments.** Prefer English; we keep a handful of Chinese inline comments
  for historical reasons, but new code should be English-only.

## Tests

The pytest suite stays deliberately *small and CPU-only* so it can run in
GitHub Actions without GPUs:

```bash
pytest                       # run everything
pytest -m "not slow"         # skip slow tests
pytest -k weight             # run a specific subset
pytest --cov                 # with coverage
```

When you add a feature, please add a corresponding unit test. We treat the
following as the "load-bearing" surface that must always be tested:

* `models/sample_weighting.py` — DWBC mapping monotonicity & warm-up.
* `train/train_scoring.py::pu_loss_function` — both `paper` and `dwbc` variants.
* `models/rdt_runner.py::RDTRunner.compute_loss` — sample-weighting shape /
  fallback behaviour.
* `configs/cross_embodiment/*.yaml` — schema validation (36-D mask, 4-cam
  mask, declared DoF).
* `replay_validate.py::*_verifier` — pure-python verifiers.

## Pull Request Checklist

- [ ] Code is formatted (`pre-commit run -a` clean)
- [ ] `pytest` is green
- [ ] New public APIs come with docstrings + type hints
- [ ] README / config docs are updated if behaviour changed
- [ ] Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
      (e.g. `feat:`, `fix:`, `docs:`, `refactor:`, `test:`).

## Filing Issues

Please include:

1. The Dexora commit SHA you reproduced on.
2. The exact command you ran (and the config file used).
3. The error message / stack trace, **with enough context** to reproduce.
4. GPU model + CUDA + PyTorch version when relevant.

## Releasing Data / Checkpoints

Large blobs (data, model weights, video) are **never** committed to the repo.
Follow the `Downloads` section in the README for the canonical hosting
locations. When you contribute a new asset link, update both the README and
[`tools/release_check.py`](tools/release_check.py) so the release script can
detect missing files.

## Code of Conduct

This project follows the
[Contributor Covenant](CODE_OF_CONDUCT.md). By participating you agree to
abide by its terms.
