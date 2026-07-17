# Spectral Disentangling

Disentangling **synthetic mixed 1D spectra** (NMR-like) with masked-pretrained
transformers. We generate synthetic mixtures with **known ground truth**, then study
whether self-supervised masked pretraining helps a small transformer separate and
identify the components - especially when labeled data is scarce.

This is a learning-first, reproducible research demo. It is **not** chasing
state-of-the-art or using real proprietary data. See [`REPORT.md`](REPORT.md) for the full
project writeup (motivation, methods, results, and findings), and [`DECISIONS.md`](DECISIONS.md)
for the running log of design choices.

## What we found

Three results, in the order they were discovered:

1. **Fine-tuning a pretrained encoder barely beats training from scratch** on the easy
   regime, at any label budget (biggest gain +0.030 macro-F1 at 10 labels, smaller than its
   own seed-to-seed spread). Taken alone this looks like "pretraining did not help".
2. **Frozen linear probes say the opposite.** The same pretrained encoder, frozen, is far
   more linearly decodable than a random one (presence macro-F1 0.872 vs 0.729;
   concentration MAE 0.150 vs 0.239, close to the 0.138 of a supervised encoder that never
   trained on concentration). Pretraining *does* learn real structure - full fine-tuning on
   an easy task simply lets a from-scratch model catch up and overwrite it.
3. **Push the task harder and the picture splits.** Sweeping a difficulty knob (SNR 40 -> 6,
   more overlapping components, more jitter), the pretraining gain at 40 labels grows to
   **+0.082 macro-F1** at the hard end, consistently across seeds. But the *frozen* SSL
   representation collapses there (presence 0.529) while a supervised encoder still reaches
   **0.921** on the same data - so the information survives the noise and the *pretext task*
   is what stops capturing it.
4. **The obvious explanation for that is wrong.** Suspecting the pretext was wasting itself
   predicting unpredictable noise, I swapped its target for the noise-free signal - an oracle
   a real setting could never provide. It recovered **2%** of the gap. Noise in the target is
   not the culprit, which rules out the noise-robust objectives (Noise2Noise and friends) you
   would naturally reach for next. What breaks it is still open; `REPORT.md` lays out the two
   live candidates and which to test first.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # installs the `spectral` package (editable)
```

## Sanity check
```bash
python scripts/hello_gpu.py            # prints torch version + compute device
pytest                                 # the full suite (~3s, no GPU needed)
```

## Reproducing everything

Every figure and number in the report comes from these scripts, in this order. The pipeline
driver runs them all:

```bash
python scripts/run_all.py --list          # stages, runtimes, outputs
python scripts/run_all.py                 # everything, start to finish
python scripts/run_all.py --skip-existing # resume: skip stages already done
python scripts/run_all.py --only probes   # just one stage
```

Or run any stage by hand - they are ordinary scripts with their own config:

| stage | command | ~min | what it produces |
|---|---|---|---|
| data | `python scripts/visualize_mixtures.py` | <1 | `library.png`, `mixtures.png` (Checkpoint 1) |
| baseline | `python scripts/train_baseline.py` | ~5 | `baseline_curve.png` |
| pretrain | `python scripts/pretrain.py` | ~7 | `pretrained_encoder.pt`, reconstructions |
| finetune | `python scripts/finetune_sweep.py` | ~25 | `label_efficiency.png` (result 1) |
| probes | `python scripts/probe.py` | ~10 | `probe_presence.png` (result 2) |
| robustness | `python scripts/robustness_sweep.py` | ~90 | `robustness_*.png` (result 3) |
| ablation | `python scripts/pretext_ablation.py` | ~30 | `pretext_ablation.png` |

Order matters: `finetune` and `probes` load the encoder that `pretrain` writes, and
`ablation` reuses the per-difficulty encoders that `robustness` caches. Runtimes are for an
M-series Mac (MPS); a single CUDA GPU is faster, CPU much slower. Outputs land in
`experiments/`, which is gitignored - the configs and seeds are the source of truth, not the
artifacts.

## Layout
```
configs/        YAML configs, one per phase (no magic numbers in code)
src/spectral/   the package: data factory, models, training, probing
scripts/        runnable entry points, one per phase + run_all.py
experiments/    run outputs (gitignored, regenerate with run_all.py)
notebooks/      exploration
tests/          pytest suite (46 tests)
```

## Design principles
- Fixed random seeds and config files - everything re-runnable with one command.
- Small, readable modules over cleverness.
- Every design decision, and every surprise, is recorded in `DECISIONS.md`.
- Negative and null results are reported as found, not buried.
