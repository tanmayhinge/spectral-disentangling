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
