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

## Phase 4 - fine-tuning and the headline comparison (built)
- **Protocol**: controlled full fine-tuning. For each label size and seed, train the same
  classifier twice (scratch vs pretrained encoder init) with identical head init, batch
  order, optimizer, and a fixed 600-step budget; report best val macro-F1. 3 seeds ->
  std bands. Labeled streams (seed 1000+s), val (10001), pretrain (20000) all disjoint.
- **HEADLINE RESULT (honest)**: pretraining gives only a marginal, mostly within-noise
  edge in this regime.
  | n_labels | scratch | pretrained | gain |
  |---|---|---|---|
  | 10 | 0.423 +/- .021 | 0.453 +/- .051 | +0.030 |
  | 40 | 0.711 | 0.711 | +0.000 |
  | 160 | 0.946 | 0.954 | +0.008 |
  | 640 | 0.994 | 0.995 | +0.001 |
  | 2560 | 0.998 | 0.998 | -0.000 |
  The +0.03 at n=10 is smaller than its std, so not conclusive.
- **Why (working hypotheses)**: (a) the task is easy - fixed distinct fingerprints make
  presence ~matched filtering, so from-scratch learns fast even with few labels, leaving
  little headroom; (b) 600 steps of full fine-tuning can overwrite pretrained features;
  (c) the denoise/inpaint pretext teaches peak location/shape, which the supervised task
  also picks up on its own here.
- **Planned investigations** (surfaced for Tanmay, not yet run): frozen-encoder LINEAR
  PROBES (Phase 5) to test representation quality without fine-tuning erasing it; and a
  HARDER difficulty regime (Phase 6: low SNR, more overlap/components) where from-scratch
  should struggle and pretraining has room to help. Also: more seeds to tighten n=10.

## Phase 5 - linear probes on frozen features (built)
- **Protocol**: freeze the encoder, mean-pool patch tokens (not the task-specific CLS),
  standardize, train only a linear layer. Compare three encoders - random (floor),
  pretrained (SSL), supervised (labels-trained ceiling) - on presence (macro-F1), component
  count K (accuracy), concentration (MAE over present entries). 3 seeds.
- **KEY RESULT: pretraining clearly DID learn structure.** Frozen pretrained features are
  much more linearly decodable than random ones:
  | probe | random | pretrained | supervised |
  |---|---|---|---|
  | presence macro-F1 @2560 | 0.729 | **0.872** | 0.998 |
  | count accuracy | 0.591 | **0.626** | 0.842 |
  | concentration MAE (lower=better) | 0.239 | **0.150** | 0.138 |
  Std ~0.001-0.013, so the gaps are real. Pretrained beats random at every presence label
  size (e.g. +0.118 at n=160). For concentration, pretrained (0.150) is close to the
  supervised ceiling (0.138) despite never seeing labels.
- **This resolves the Phase-4 null result.** Pretraining was not useless; FULL fine-tuning
  on an easy task simply lets from-scratch catch up and overwrite the advantage. Freeze the
  encoder and the advantage is large and consistent. Interpretation: the SSL representation
  linearly encodes compositional/physical structure (which compounds, how many, how much) -
  meaningful structure, not surface memorization. This is the "physics question" answer.
- **Note**: supervised is an oracle ceiling for presence (it trained on presence); more
  telling is concentration, which supervised did NOT train on, where pretrained nearly
  matches it. Next available: attention-on-peaks analysis (Phase 5 extension) and the
  harder-regime robustness sweep (Phase 6).

## Phase 6 - robustness sweep over difficulty (built)
- **The knob**: `DataConfig.at_difficulty(d)` linearly interpolates three things between an
  easy end (d=0: snr 40, k_max 2, jitter 0.002) and a hard end (d=1: snr 6, k_max 5, jitter
  0.010). Sweep is d = 0, 0.25, 0.5, 0.75, 1.0 with seeds 0/1/2 (matching Phases 4-5).
- **DECISION - pretrain a FRESH encoder at each difficulty** (rather than reusing the single
  Phase-3 encoder). Reusing it would confound two effects: the task getting harder, and the
  pretraining data no longer matching the test data. Matched pretraining also mirrors the
  realistic setting the KIT posting implies - abundant unlabeled spectra from your own
  instrument, few labels. Cost is ~7 min/encoder, so 5 encoders is cheap.
- **DECISION - one encoder per difficulty, shared across the 3 seeds** (as in Phases 3-5).
  LIMITATION: error bars therefore capture fine-tuning/probe variance but NOT
  pretraining-run variance. Worth revisiting if a difficulty lands close to a decision
  boundary.
- **Two arms per difficulty**, so Phase 6 re-runs both earlier measurements:
  (a) fine-tune arm: pretrained vs scratch at n = 40 and 160 labels (small budgets only -
  Phase 4 showed the arms converge once labels are plentiful). Tests the standing prediction
  that the Phase-4 null gap REAPPEARS once from-scratch can no longer catch up in 600 steps.
  (b) probe arm: frozen linear probes (random/pretrained/supervised) at 2560 probe labels,
  reporting presence macro-F1, count accuracy, concentration MAE. Tests where the Phase-5
  representation advantage breaks down.
- **Recipes are copied from Phases 3/4/5 unchanged** (mask_ratio 0.5, span_len 4, 20 epochs;
  600 fine-tune steps; 300 probe steps) so Phase 6 is comparable to them rather than being a
  new experiment. A test asserts the pretraining recipe still matches Phase 3.
- **GOTCHA found while building**: at d=0 the knob pins k_max to k_min=2, so EVERY mixture
  has exactly K=2 and the count probe has a single class - it scores 1.000 for any encoder,
  random included. That is an artifact of the knob, not a result. The reporting now prints
  "n/a (K constant)" there and omits the point from the plot instead of showing a perfect
  score. Same reason the count probe only becomes meaningful from d=0.25 on.
- **GOTCHA 2 - the difficulty axis is not uniform in K**: `round(lerp(2,5,d))` gives k_max =
  2, 3, 4, 4, 5 across the five points, so d=0.50 and d=0.75 have the SAME k_max=4 and differ
  only in SNR/jitter. Any flat stretch between those two points cannot be attributed to
  component count. Difficulty is a bundle of three knobs, not a scalar - do not read the
  x-axis as "amount of hardness".

### Phase 6 results
- **Prediction under test** (from Phase 4): the fine-tuning gap, absent in the easy regime,
  reappears once the task is hard enough. **Verdict: partially supported, and only at the
  smallest budget.** Paired per-seed gains (same seed = same head init and batch order):
  | d | n=40 gain (per-seed) | mean | n=160 gain (per-seed) | mean |
  |---|---|---|---|---|
  | 0.00 | +.028 +.048 +.045 | **+0.040** | +.003 -.001 -.003 | -0.000 |
  | 0.25 | -.042 +.001 -.015 | -0.018 (mixed) | -.007 -.001 -.016 | -0.008 |
  | 0.50 | +.014 +.029 +.029 | **+0.024** | +.020 +.033 -.006 | +0.016 (mixed) |
  | 0.75 | +.036 +.029 +.009 | **+0.025** | +.016 +.024 +.001 | +0.014 |
  | 1.00 | +.106 +.060 +.080 | **+0.082** | +.116 +.035 -.014 | +0.046 (mixed) |
- **The one solid claim**: at n=40, d=1.00, the gain is +0.082 with all three seeds positive
  and ~4x the seed std. That is the biggest, cleanest pretraining win in the whole project.
- **Honest caveats, do NOT overclaim**: (a) the curve is NOT monotone - d=0.00 also shows a
  consistent +0.040, so "no gap when easy" is false here; (b) the d=0.25 dip is mixed-sign
  and within noise, i.e. noise, not a real reversal; (c) the n=160 hard-end +0.046 is driven
  by ONE seed (+0.116) with another negative (-0.014) and std 0.054 > mean - **not reliable**;
  (d) d=0.00 here (snr 40, K=2 fixed) is a much easier regime than Phase 4's default (snr 30,
  K=2-5), so the +0.040 does not contradict Phase 4's +0.000 at n=40.
- **KEY FINDING (probes) - the SSL representation degrades much faster than the task does.**
  Presence macro-F1 on frozen features:
  | d | random | pretrained | supervised | pretrained-over-random |
  |---|---|---|---|---|
  | 0.00 | 0.875 | 0.933 | 1.000 | +0.058 |
  | 0.25 | 0.771 | 0.877 | 0.996 | +0.106 |
  | 0.50 | 0.659 | 0.811 | 0.991 | +0.152 |
  | 0.75 | 0.532 | 0.704 | 0.984 | **+0.172** |
  | 1.00 | 0.374 | 0.529 | 0.921 | +0.155 |
  Pretrained beats random at EVERY difficulty (stds ~0.001-0.009, so all real), and the
  advantage widens to d=0.75 before narrowing as everything collapses. Phase 5 replicates.
- **The interesting part**: at d=1.00 the SUPERVISED encoder still hits 0.921 presence F1 and
  0.145 concentration MAE (nearly flat from 0.133 at d=0), while pretrained falls to 0.529 /
  0.323. So the information is still there and still LINEARLY decodable - the task is not
  intrinsically impossible at SNR 6. It is the *pretext task* that stops capturing it.
- **Working hypothesis (needs testing, not established)**: at SNR 6 the raw-signal
  reconstruction target is dominated by noise the model cannot predict, so the masked-MSE
  gradient is mostly spent modelling noise rather than compound structure. Consistent with
  pretraining held-out MSE jumping 0.00676 -> 0.02734 between d=0.75 and d=1.00. **Testable
  fix for future work**: reconstruct the CLEAN component sum (a denoising objective) instead
  of the raw noisy signal, or weight the loss toward peak regions. The data factory already
  emits `clean_mixture`, so this is a small change.
- **Robustness shape**: supervised features degrade GRACEFULLY (1.000 -> 0.921 presence).
  Frozen SSL and random features degrade SHARPLY, with the steepest drop between d=0.75 and
  d=1.00. Fine-tuned models sit in between (n=160 scratch: 0.853 -> 0.645 over that step).

## Phase 6 follow-up - pretext-target ablation (built)
- **Question**: Phase 6 showed frozen SSL features collapse at d=1.00 (0.529) while a
  supervised encoder still hits 0.921 on the SAME data. So the information survives the noise
  and is linearly decodable - the pretext task is what stops capturing it. Hypothesis: at
  SNR 6 the reconstruction target is mostly unpredictable noise, so the masked-MSE gradient is
  spent modelling noise instead of compound structure.
- **Design**: swap ONLY the reconstruction target, hold everything else fixed.
  `raw` = reconstruct the observed noisy signal (the Phase-3 pretext); `clean` = reconstruct
  the noise-free component sum. Re-probe both frozen. Implemented as an optional `target` arg
  on `masked_mse` (default None = predict the input, so the Phase-3 path is untouched - a test
  asserts this, and the old-vs-new loop was checked bit-identical).
- **DECISION - `clean` is an ORACLE DIAGNOSTIC, not a method.** `clean_mixture` is ground
  truth that a real unlabeled setting would never hand you, so the clean arm is NOT
  self-supervised. It exists to isolate the effect of noise in the target: if it recovers the
  gap, the objective was the bottleneck and a noise-robust pretext is worth building; if not,
  the hypothesis is wrong. This caveat is repeated in the script docstring, the config header,
  and the `build_mixture_and_clean_tensors` docstring so it cannot get quietly reported as
  "our SSL method improved". A genuinely self-supervised follow-up would be Noise2Noise-style
  (predict one noisy realization from another).
- **Control worked**: at d=0.00 (SNR 40, almost no noise) raw and clean are indistinguishable
  (0.933 vs 0.936), exactly as predicted - the ablation is a no-op where there is no noise to
  remove. That is what licenses attributing any divergence at the hard end to noise rather
  than to some incidental difference between the two objectives.
- **Note**: held-out MSE is NOT comparable across arms (a clean target is intrinsically easier
  to hit than a noisy one). Only the downstream probe numbers are comparable.
- **RESULT: THE HYPOTHESIS IS REFUTED.** Presence macro-F1 on frozen features:
  | d | random | raw | clean | supervised | gap recovered by clean |
  |---|---|---|---|---|---|
  | 0.00 | 0.875 | 0.933 | 0.936 | 1.000 | +4% |
  | 0.50 | 0.659 | 0.811 | 0.815 | 0.991 | +2% |
  | 0.75 | 0.532 | 0.704 | 0.720 | 0.984 | +6% |
  | 1.00 | 0.374 | 0.529 | **0.539** | 0.921 | **+2%** |
  Even handed a PERFECT noise-free target - an oracle no real setting could provide - the
  pretext gains +0.010 at d=1.00 and recovers 2% of the 0.392 gap to supervised. Stds are
  0.001-0.009, so the non-effect is not noise. Concentration MAE tells the same story
  (0.323 -> 0.319 at d=1.00). **Noise in the reconstruction target is NOT why pretraining
  collapses at low SNR.** I wrote the hypothesis down before running this and it was wrong.
- **What this rules OUT, and what it costs**: it kills the obvious follow-up. Noise2Noise-style
  targets were the natural "genuinely self-supervised" version of this fix; if the ORACLE
  target buys ~nothing, a cleverer self-supervised target will not either. Do not spend time
  there. (This recommendation was already written into REPORT.md/paper.tex before the result
  landed and had to be corrected - a good argument for running the ablation before writing.)
- **What is still open (candidate explanations, none tested)**: (a) masked reconstruction may
  be solvable by LOCAL interpolation - continue the peak shape from neighbouring context -
  which never requires knowing WHICH compound produced it, so the objective simply does not
  incentivize identity; supervised training optimizes identity directly, which is why it holds
  up at 0.921. Low SNR would then hurt the local strategy without ever pushing the model
  toward the global one. (b) The probe reads MEAN-POOLED patch tokens; the information could
  be present nonlinearly, or concentrated in CLS, and simply invisible to this readout.
  (c) Capacity is spent on low-level signal statistics. Distinguishing these is the real next
  experiment - (b) is cheap to check (re-probe with CLS pooling and an MLP probe) and should
  come first, because it is a measurement artifact rather than a claim about learning.
- **What survives**: the Phase-6 fine-tuning gain at d=1.00, n=40 (+0.082, 3/3 seeds) is
  unaffected - pretraining still helps FINE-TUNING at the hard end even though its FROZEN
  features are weak there. Those two facts sit together and are worth keeping in view.

## Phase 7 - report and packaging (built)
- **`scripts/run_all.py`**: the "one command" from Working Agreement #7. Runs every stage in
  phase order with `--list`, `--only`, `--skip-existing`. Stage order is load-bearing:
  finetune/probes load the encoder pretrain writes, and the ablation reuses the encoders the
  robustness sweep caches. Uses `check=True` so a failed stage stops the run rather than
  letting later stages read stale artifacts and silently produce a wrong report.
- **DECISION - two writeups, on purpose**: `REPORT.md` is the readable narrative (linked from
  the README, easy to diff); `report/paper.tex` is the formal 4-6 page paper for the KIT
  application/arXiv. Some duplication is accepted; the paper is the citable artifact.
- **DECISION - figures are committed to `report/figures/`** even though `experiments/**` is
  gitignored, so the paper compiles from a clean clone without re-running hours of training.
  `scripts/build_paper.py` copies them across and fails loudly on a missing figure (a paper
  silently missing a figure is worse than no paper).
- **Toolchain**: tectonic (single self-contained binary, `brew install tectonic`) rather than
  a multi-GB MacTeX install. Build with `python scripts/build_paper.py` -> `report/paper.pdf`.
