# Spectral Disentangling

Disentangling **synthetic mixed 1D spectra** (NMR-like) with masked-pretrained
transformers. We generate synthetic mixtures with **known ground truth**, then study
whether self-supervised masked pretraining helps a small transformer separate and
identify the components - especially when labeled data is scarce.

This is a learning-first, reproducible research demo. It is **not** chasing
state-of-the-art or using real proprietary data. See [`DECISIONS.md`](DECISIONS.md) for
the running log of design choices.

## Status
- **Phase 0 - repo skeleton & sanity**: in progress.
- Later phases (data factory, baseline, pretraining, comparison, probing, robustness,
  report) are built and validated incrementally.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # installs the `spectral` package (editable)
```

## Sanity check (Phase 0 / Checkpoint 0)
```bash
python scripts/hello_gpu.py            # prints torch version + compute device
pytest tests/test_seeding.py           # confirms seeding is deterministic
```

## Layout
```
configs/        YAML configs (no magic numbers in code)
src/spectral/   the package: config, seeding, utils, and (later) data + model
scripts/        runnable entry points
experiments/    run outputs (gitignored)
notebooks/      exploration
report/         technical report (later)
tests/          pytest suite
```

## Design principles
- Fixed random seeds and config files - everything re-runnable with one command.
- Small, readable modules over cleverness.
- Every design decision is recorded in `DECISIONS.md`.
