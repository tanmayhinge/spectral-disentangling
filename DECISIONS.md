# Decisions log

Running record of design choices (Working Agreement rule 6). Newest phase at the bottom.
Each entry: **what** we decided and **why**, so it can be defended later.

## Phase 0 - repo skeleton
- **Config system: dataclasses + PyYAML** (no Hydra/OmegaConf). Minimal deps, fully
  legible, one obvious place per parameter. Tradeoff: we hand-write a small loader
  instead of getting multirun/override sugar for free - fine at this scale.
- **Seeding: `seed_everything(seed)`** seeds `random`, `numpy`, and `torch` (CPU + CUDA),
  and sets `cudnn.deterministic=True`. Every entry point takes its seed from config so
  runs are reproducible from day one.
- **Compute device detection order: CUDA → MPS → CPU.** Development is on Apple Silicon
  (MPS); training will target a single CUDA GPU (A6000/4090-class). Same code path both
  places.

## Phase 1 - data factory (LOCKED / built)
- **Signal grid: N = 2048 points over 0–10 ppm** (1H-NMR range). Power of two for clean
  patching later; ~0.005 ppm/point resolves narrow peaks and J-splitting; small enough
  for a low-millions-param model on one GPU.
- **Component model: fixed, region-aware synthetic library of M = 12 compounds**, seeded
  once and frozen to disk. Peaks drawn with priors for realistic 1H regions (aromatic
  ~6.5–8.5, aliphatic ~0.5–3 ppm). A fixed reusable library is what makes "which
  components are present" a well-defined classification target. Behind a `ComponentSource`
  interface so real nmrshiftdb2 spectra can be dropped in later.
- **Lineshape: Lorentzian default** (physically correct for NMR); Gaussian/Voigt
  selectable.
- **Mixtures: K = 2–5 compounds**, positive random concentrations, weighted superposition
  then corruptions (noise via target SNR, baseline drift, peak jitter, optional phase).
- **Single difficulty knob** `∈ [0,1]` maps monotonically to {higher K, lower SNR, more
  overlap, more jitter}.
- **Generation: on-the-fly torch Dataset**, reproducible from (base_seed, index); a script
  dumps a fixed eval/visualization set for stable metrics/figures.

### Metrics (LOCKED)
- **Presence: macro-F1** (+ per-compound F1) - robust to class imbalance.
- **Concentration: MAE + relative error on present compounds** - interpretable across scale.
- **Separation: per-component MSE reconstruction** - simple, comparable across models.

### Implementation specifics (chosen while building)
- **SNR convention**: `snr` = tallest-peak-height / noise-std (NMR convention), so
  `noise_std = peak_height / snr`. Intuitive to eyeball on a plot.
- **Baseline**: sum of 3 long-wavelength sinusoids, amplitude = `baseline_frac` × tallest
  peak - a slow rolling drift.
- **Jitter** is applied at render time (a small shared per-compound ppm shift) so it lands
  *inside* the ground-truth component signals; this keeps the reconstruction invariant
  exact: `mixture = sum(components) + baseline + noise` (verified to 0 error, phase off).
- **Multiplets** split via Pascal-triangle intensities and conserve peak area.
- **Reproducibility**: each sample uses `np.random.default_rng([base_seed, index])`, an
  independent stream, so any sample is regenerable without storing the dataset.

### Scope
- **Synthetic-only through Phase 6**; real spectra optional later via a `render`-compatible
  library (the `CompoundLibrary` interface).

## Phase 2 - from-scratch baseline (built)
- **Signal representation: patched tokens** (the Open Question). The 2048-point spectrum is
  split into non-overlapping patches of 32 points (64 tokens + 1 CLS), each linearly
  embedded via a strided Conv1d. Chosen over (a) raw per-point tokens (length-2048 sequence,
  wasteful O(n^2) attention over mostly-empty baseline) and (b) peak-list input (needs a
  separate peak-picker, discards the raw signal and corruptions, and would not transfer to
  Phase-3 masked reconstruction). The patch layout is exactly what Phase 3 will mask.
- **Backbone**: pre-norm transformer encoder, d_model=128, 4 heads, 4 layers, ff=256,
  GELU, learned positional + CLS embeddings. ~544k params. Encoder returns per-token
  embeddings so later phases can attach reconstruction/regression heads.
- **Task/loss**: multi-label presence via `BCEWithLogitsLoss` on 12 logits (CLS pooling).
- **Optim**: AdamW, lr 3e-4, weight_decay 0.01, batch 64, 15 epochs. Train/val are disjoint
  reproducible mixture pools (base_seed 0 vs 10001); data pre-generated once and cached.
- **Result**: val macro-F1 = 0.997, exact-match = 0.978 with 4000 labels. NOTE: the task is
  nearly saturated in the abundant-label regime, which is expected (fixed distinct
  fingerprints make presence ~matched filtering). The meaningful test for pretraining is the
  Phase-4 low-label sweep (10/100/1000 labels) and/or higher difficulty; from-scratch has no
  headroom to beat at 4000 labels.

## Phase 3 - masked self-supervised pretraining (built)
- **Method: masked patch modeling** (BERT-style, not MAE). Replace masked patch embeddings
  with a learned `mask_token`, encode with the SAME `SpectralEncoder` as Phase 2, and a
  linear head predicts each patch's raw values. Loss = MSE on masked patches only. Chosen
  over MAE's encoder/decoder split for legibility and because it reuses the exact encoder we
  fine-tune later.
- **Masking: contiguous spans**, not isolated patches. `mask_ratio=0.5`, `span_len=4` patches
  (128 points, wide enough to hide a whole peak). Rationale: neighboring points are
  correlated, so single-patch masking is solvable by interpolation and teaches little; spans
  force use of longer-range structure.
- **Reconstruction target**: raw patch values (no per-patch normalization) so the sanity
  figure is directly interpretable.
- **Data**: 20000 UNLABELED mixtures on a disjoint stream (pretrain_seed=20000) from train
  (0) and val (10001), so pretraining never sees the fine-tuning inputs. AdamW lr 3e-4,
  batch 128, 20 epochs.
- **Transfer contract**: only `encoder.state_dict()` is saved (to
  experiments/pretrained_encoder.pt); a test asserts it loads strict=True into the Phase-2
  PresenceClassifier's encoder. This is the Phase-4 initialization.
- **Result**: held-out masked MSE 0.0089 -> 0.0038 (still decreasing). Reconstructions fill
  hidden spans with denoised peaks near ground truth -> evidence it learned peak
  location/shape structure without labels. Known limitation: MSE makes it under-predict peak
  heights (regresses toward the mean) and occasionally miss faint peaks.
