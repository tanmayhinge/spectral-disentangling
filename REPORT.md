# Disentangling synthetic mixed spectra with a masked-pretrained transformer

This is a working report on a small research project. The question I set out to answer is
whether self-supervised masked pretraining, of the kind used for language and images, learns
anything useful about the structure of one-dimensional mixed spectra, and whether that helps
when labeled data is scarce. I work entirely with synthetic mixtures so that I have exact
ground truth for every signal, which is what makes the question measurable in the first place.

The short version of the result is that the answer depends on how you ask. Measured by
end-task accuracy after full fine-tuning, pretraining barely helps, because the task is easy
enough that a model trained from scratch catches up. Measured by what is linearly decodable
from the frozen representation, pretraining helps a lot: the pretrained encoder encodes which
compounds are present, how many there are, and at what concentration, far better than an
untrained encoder, and for concentration it comes close to an encoder that was trained
directly on labels. The two measurements together are the interesting part of the project.

The work is organized in phases, all of which are now complete. A formatted PDF version of
this writeup lives at `report/paper.pdf` (build it with `python scripts/build_paper.py`); this
file is the longer, more conversational version, including the reasoning and the dead ends.

## Motivation

Mixture analysis shows up whenever an instrument measures a sum of overlapping signals and
you want to recover the parts. Nuclear magnetic resonance spectra in drug screening are a
concrete case: a measured spectrum is a superposition of contributions from several compounds,
and the practical task is to say which compounds are present and in what amounts. Real
screening data is proprietary and, more importantly for a study like this, it does not come
with a trustworthy decomposition into ground-truth parts. Without that ground truth you can
train a model but you cannot cleanly measure whether a claimed separation is correct.

Synthetic data solves the measurement problem. If I build each mixture myself out of known
components, I hold the exact answer for every sample: the identity of each component, its
concentration, and its individual clean signal. The cost is realism, and I take that tradeoff
deliberately. The goal here is not a chemistry-grade tool. It is a clean, reproducible study
of a method, run in miniature, where every claim can be checked against the truth I generated.

The specific method under test is masked sequence modeling. Hide part of an input, ask the
model to reconstruct the hidden part, and use no labels while doing it. This is the objective
behind BERT and behind masked autoencoders for images. The hypothesis worth checking is that
the same idea, applied to raw spectra, teaches a model where peaks sit, how wide they are, and
how they cluster into per-compound patterns, and that this learned structure transfers to the
labeled tasks that actually matter.

## Synthetic data generation

Everything downstream is graded against the data factory, so I built it first and made
correctness the priority. A spectrum is represented as a fixed grid of 2048 intensity values
over a 0 to 10 ppm axis, which is the usual proton NMR range. That spacing, about 0.005 ppm
per point, is fine enough to resolve narrow peaks and their J-coupling splitting, and the grid
length is a power of two, which matters for the patch tokenization later.

A component is a compound with a fixed pattern of peaks. I generate a library of 12 compounds
once, from a fixed seed, and freeze it to disk, so every experiment sees the same 12
fingerprints. Peak positions are drawn with priors that place them in realistic proton
regions (aromatic around 6.5 to 8.5 ppm, an intermediate band, and aliphatic around 0.5 to
3 ppm). Each peak is rendered as a Lorentzian, which is the physically correct lineshape for
relaxation-limited NMR lines, and peaks can be split into multiplets by J-coupling, with the
split lines carrying Pascal-triangle intensities so the total area is conserved. Fixing the
library is what makes "which compounds are present" a well-defined classification target
rather than an ill-posed question about arbitrary peaks. The library sits behind a small
interface, so real single-compound spectra could be substituted later without touching the
rest of the pipeline.

A mixture is built by choosing K compounds, with K between 2 and 5, assigning each a positive
concentration, summing the scaled fingerprints, and then applying corruptions that stand in
for instrument imperfections: additive Gaussian noise at a target signal-to-noise ratio,
a slow rolling baseline, and a small random shift in peak positions. I define signal-to-noise
in the usual NMR sense, as the tallest peak height divided by the noise standard deviation,
so it is easy to read off a plot. The position jitter is applied while rendering each
component, which means it lands inside the stored ground-truth component signals. That detail
matters, because it keeps an exact identity intact:

```
mixture = sum(clean components) + baseline + noise
```

I verified this reconstruction identity numerically and it holds to zero floating-point error,
and there is a test that checks it on every generated sample. Each mixture ships with its full
label set: a multi-hot presence vector over the 12 compounds, a concentration vector, and the
clean per-component signals. Samples are generated on demand from a seed and an index rather
than stored, so any sample is reproducible and the dataset never has to be written to disk.

## Model and input representation

The model reads a spectrum as a sequence. The obvious encoding, one token per grid point, is
wasteful: attention scales with the square of the sequence length, and most of a spectrum is
empty baseline. The other obvious encoding, a list of picked peaks, throws away the raw signal
and the corruptions I went to the trouble of adding, and it would not transfer to a
reconstruction objective. So I patch the signal instead, in the same way a vision transformer
patches an image. The 2048 points are cut into 64 non-overlapping patches of 32 points, each
linearly embedded into a 128-dimensional vector by a strided convolution. A learned CLS token
is prepended, learned positional embeddings are added, and the sequence goes through four
pre-norm transformer encoder layers with four attention heads. The classifier reads the CLS
token through a linear layer to produce 12 presence logits. The whole model has about 544k
parameters, which trains in under a minute on a single laptop GPU.

The patch layout is deliberately the same layout that pretraining will mask, so the encoder is
shared across every phase and its weights carry over without modification.

## Masked pretraining

Pretraining hides random spans of a spectrum and asks the model to fill them back in. I patch
the signal as usual, replace the embeddings of the masked patches with a single learned mask
token, run the shared encoder, and predict the raw values of every patch through a linear
head. The loss is mean squared error on the masked patches only, so the model gets no credit
for copying patches it can already see. I mask contiguous spans rather than isolated patches,
with half the patches hidden in spans of four patches each, which is about 128 points and wide
enough to swallow a whole peak. Single-patch masking would be too easy, since a hidden point
can be guessed from its neighbors; spans force the model to use longer-range structure.

I pretrain on 20,000 unlabeled mixtures drawn from a stream that is disjoint from the labeled
data used later, so the pretraining never sees a fine-tuning input. Held-out reconstruction
error falls from 0.0089 to 0.0038 over 20 epochs and is still decreasing at the end. The
reconstructions are informative to look at. In the masked regions, where the model saw
nothing, it places peaks close to where the hidden truth has them, and it reconstructs a
denoised version of the signal rather than the noisy measurement. That the model recovers
clean peak structure from partial context, without labels, is the first piece of evidence that
it has learned something about how these spectra are built. The main limitation of the
objective is a height bias: mean squared error under partial information favors conservative
guesses, so the model tends to under-predict tall peaks and can miss faint ones.

## Experiments and results

### From-scratch baseline

Before any pretraining I trained the model from scratch on presence classification, with
binary cross-entropy on the 12 logits, AdamW, and a validation set held out on a separate
seed. With 4000 labeled examples it reaches a validation macro-F1 of 0.997 and gets all 12
labels correct on 97.8 percent of samples. Training and validation loss track each other, so
there is no overfitting.

This number is high, and it is worth being honest about why. With a fixed library of distinct
fingerprints, deciding which compounds are present is close to matched filtering, and a
capable model saturates it once labels are plentiful. That has a direct consequence for the
rest of the study: pretraining cannot improve on 0.997, so any benefit has to show up where
the baseline is weak, which means in the low-label regime or under harder conditions.

### Fine-tuning: pretrained versus from-scratch

The headline experiment is a controlled comparison. I fine-tune the full model on a range of
label budgets, initializing the encoder either from random weights or from the pretrained
weights, and I hold everything else fixed: the same head initialization, the same batch order,
the same optimizer, and the same 600-step budget, with the same seed. The only variable is the
encoder starting point. I report the best validation macro-F1 over training, averaged across
three seeds.

| labeled examples | from scratch | pretrained | difference |
| ---: | :--- | :--- | :---: |
| 10 | 0.423 +/- 0.021 | 0.453 +/- 0.051 | +0.030 |
| 40 | 0.711 +/- 0.021 | 0.711 +/- 0.013 | +0.000 |
| 160 | 0.946 +/- 0.010 | 0.954 +/- 0.005 | +0.008 |
| 640 | 0.994 +/- 0.001 | 0.995 +/- 0.001 | +0.001 |
| 2560 | 0.998 +/- 0.001 | 0.998 +/- 0.001 | -0.000 |

The result is close to a null. Pretraining gives a small edge at the smallest budget, but the
0.030 difference at 10 labels is smaller than its own standard deviation, so I do not read it
as a real effect. Everywhere else the two initializations are indistinguishable. Taken alone,
this experiment says pretraining does not help. My working explanation was that full
fine-tuning on an easy task lets the from-scratch model reach the same solution, which would
erase any advantage the pretrained features started with. The next experiment tests that
explanation directly.

### Linear probes on frozen features

To separate the quality of the learned representation from the effect of fine-tuning, I froze
each encoder and trained only a linear layer on top of its features. A linear probe is a sharp
test: a linear layer cannot compute structure that is not already present in the features, it
can only read structure that is. I compared three frozen encoders. The first is a random,
untrained encoder, which sets the floor for what is linearly readable by chance. The second is
the masked-pretrained encoder. The third is an encoder trained directly on 4000 presence
labels, which sets a ceiling. I probed three targets: presence, the number of components K,
and the per-compound concentrations. Features are mean-pooled over the patch tokens and
standardized before the probe.

Presence, as validation macro-F1, across probe-training budgets:

| probe labels | random | pretrained | supervised |
| ---: | :--- | :--- | :--- |
| 10 | 0.453 | 0.473 | 0.731 |
| 40 | 0.550 | 0.614 | 0.982 |
| 160 | 0.661 | 0.779 | 0.996 |
| 640 | 0.719 | 0.849 | 0.998 |
| 2560 | 0.729 | 0.872 | 0.998 |

Count and concentration, at the full probe budget:

| target | random | pretrained | supervised |
| :--- | :--- | :--- | :--- |
| component-count accuracy | 0.591 | 0.626 | 0.842 |
| concentration MAE (lower is better) | 0.239 | 0.150 | 0.138 |

The gaps are consistent across seeds, with standard deviations between 0.001 and 0.013, so
they are real rather than noise. The pretrained encoder beats the random one on every target
and at every budget. The presence gap reaches about 0.14 macro-F1 at the largest budget. The
concentration result is the one I find most convincing: the pretrained encoder, which never
saw a label, decodes concentration almost as well as the encoder that was trained on labels
(0.150 against 0.138), and much better than the random baseline (0.239).

### Robustness: sweeping the difficulty knob

Everything above runs on a single regime, and that regime is easy. The data factory exposes a
single difficulty knob that interpolates three things at once from an easy end to a hard one:
signal-to-noise from 40 down to 6, the maximum number of components from 2 up to 5, and peak
jitter from 0.002 to 0.010 ppm. I swept it at five points with three seeds.

One design choice matters here. At each difficulty I pretrain a *fresh* encoder on unlabeled
mixtures drawn from that same difficulty, rather than reusing the single encoder from earlier.
Reusing it would confound two different effects: the task getting harder, and the pretraining
data no longer matching the test data. Matched pretraining also happens to be the realistic
setting, where you have many unlabeled spectra from your own instrument and few labels.

The prediction I wrote down before running this was that the null result from fine-tuning
would reappear as a real gap once the task got hard enough that a from-scratch model could no
longer catch up in a fixed budget. That prediction is *partially* supported, and only at the
smallest label budget. Because the scratch and pretrained runs share a seed, the honest
statistic is the paired per-seed difference rather than the difference of the means:

| difficulty | gain at 40 labels | gain at 160 labels |
| ---: | :--- | :--- |
| 0.00 | +0.040 (3/3 seeds positive) | -0.000 (mixed) |
| 0.25 | -0.018 (mixed) | -0.008 |
| 0.50 | +0.024 (3/3 seeds positive) | +0.016 (mixed) |
| 0.75 | +0.025 (3/3 seeds positive) | +0.014 |
| 1.00 | **+0.082** (3/3 seeds positive) | +0.046 (mixed) |

The one claim I will stand behind is the +0.082 at the hard end with 40 labels: every seed
agrees on the sign and the effect is about four times the seed-to-seed spread. It is the
largest pretraining win anywhere in this project. But the tidy story ("no gap when easy, gap
when hard") is not what the data says, and I want to be precise about that. The curve is not
monotone, since the easy end also shows a consistent +0.040. The dip at 0.25 is mixed-sign and
within noise, so it is noise. And the +0.046 at 160 labels is driven by a single seed at
+0.116 while another seed goes negative, with a standard deviation larger than the mean, so I
do not report it as an effect at all.

The probe arm is where the sweep earns its keep. The pretrained encoder beats the random one at
every difficulty, and the advantage actually widens as the task gets harder, from +0.058 to
+0.172 at difficulty 0.75. But look at what happens at the hard end:

| difficulty | random | pretrained | supervised |
| ---: | :--- | :--- | :--- |
| 0.00 | 0.875 | 0.933 | 1.000 |
| 0.25 | 0.771 | 0.877 | 0.996 |
| 0.50 | 0.659 | 0.811 | 0.991 |
| 0.75 | 0.532 | 0.704 | 0.984 |
| 1.00 | 0.374 | 0.529 | **0.921** |

At the hard end the supervised encoder still reaches 0.921 presence macro-F1, and its
concentration error stays nearly flat across the entire sweep (0.133 to 0.145), while the
pretrained encoder falls to 0.529 and 0.323. This asymmetry is the most interesting thing I
found. The information is still present in the signal and still linearly decodable at SNR 6,
because supervised training demonstrably reads it. So the hard regime is not intrinsically
unsolvable. It is the *pretext task* that stops capturing the structure.

Two artifacts of the knob are worth naming, because both could be misread as findings. At
difficulty 0 the knob pins the maximum component count to the minimum, so every mixture has
exactly K=2, the count probe has a single class, and every encoder scores a perfect 1.000
including the random control. That is a broken metric, not a result, and it is reported as
"n/a" rather than as a perfect score. Separately, the knob's component count takes the values
2, 3, 4, 4, 5 across the five points, so difficulties 0.50 and 0.75 differ only in noise and
jitter. Difficulty is a bundle of three quantities, not a scalar, and the x-axis should not be
read as "amount of hardness".

### Which part breaks: a pretext-target ablation

The sweep left a specific hypothesis on the table, and I wrote it down before testing it. If
the supervised encoder can still read compound identity at SNR 6 but the pretrained one
cannot, maybe the problem is what the pretext is asked to predict. At SNR 6 the observed
signal is mostly noise, and noise is exactly the part that cannot be predicted, so perhaps the
masked-MSE gradient is spent modelling noise instead of compound structure. If that were true,
the fix would be to change the target.

So I changed only the target and held everything else fixed. The `raw` arm reconstructs the
observed noisy signal, which is the original pretext. The `clean` arm reconstructs the
noise-free component sum. Everything else is identical: same masking, same architecture, same
budget, same seeds, same probe. One honesty note that matters for reading this: the clean
signal is ground truth that a real unlabeled setting would never hand you, so the clean arm is
**not** self-supervised. It is an oracle, deliberately. The point is diagnostic, not
methodological: it puts an upper bound on what any better target could buy.

The control behaves as it should. At difficulty 0, where SNR is 40 and there is almost no
noise to remove, the two arms are indistinguishable (0.933 against 0.936). That is what
licenses reading a difference at the hard end as being about noise rather than about some
incidental difference between the objectives.

| difficulty | random | raw | clean | supervised | gap recovered by clean |
| ---: | :--- | :--- | :--- | :--- | :--- |
| 0.00 | 0.875 | 0.933 | 0.936 | 1.000 | +4% |
| 0.50 | 0.659 | 0.811 | 0.815 | 0.991 | +2% |
| 0.75 | 0.532 | 0.704 | 0.720 | 0.984 | +6% |
| 1.00 | 0.374 | 0.529 | **0.539** | 0.921 | **+2%** |

**The hypothesis is wrong.** Handed a perfect, noise-free target at the hard end, the pretext
gains 0.010 macro-F1, closing 2 percent of the 0.392 gap to the supervised encoder. The
standard deviations are between 0.001 and 0.009, so this is a real non-effect rather than a
noisy one, and concentration error tells the same story (0.323 to 0.319). Noise in the
reconstruction target is not why masked pretraining collapses at low SNR.

This was worth the half hour it cost, for two reasons. It kills the obvious follow-up: a
Noise2Noise-style target was the natural genuinely-self-supervised version of this fix, and if
the *oracle* target buys nothing then a cleverer self-supervised target will not either. That
is a direction I would otherwise have spent real time on. And it sharpens the open question
into something better than "pretraining degrades". My leading candidate explanation now, which
I want to be clear is untested, is that masked reconstruction can be solved *locally*: to fill
in a hidden span you can continue the peak shapes suggested by the neighbouring context,
without ever knowing which compound produced them. If so, the objective never rewards identity
at all, and supervised training holds up at 0.921 precisely because it optimizes identity
directly. Low SNR would then degrade the local strategy without ever pushing the model toward
a global one.

There is a duller possibility I cannot yet exclude, and it should be checked first because it
is a measurement artifact rather than a claim about learning: the probe reads mean-pooled patch
tokens, so information that is present nonlinearly, or concentrated in the CLS token, would be
invisible to it. Re-probing with CLS pooling and with a small MLP probe is cheap and would
settle it.

One thing survives all of this and is easy to lose sight of. The fine-tuning gain at the hard
end (+0.082 at 40 labels, every seed agreeing) is untouched by the ablation. Pretraining still
helps fine-tuning where labels are scarce and the task is hard, even though its frozen
features are weak there. Both facts are true at once.

## Discussion

The two experiments only make sense together. Fine-tuning says the final accuracy is the same
whether or not I pretrain. Probing says the pretrained representation is much better organized
for the properties that define a mixture. Both are true, and the tension between them is the
finding. When the whole model is free to move and the task is easy, the from-scratch model
learns whatever features it needs and reaches the same place, so the pretrained head start does
not change the destination. When the encoder is frozen, the head start is all that is
available, and the pretrained features win clearly.

This is a direct answer to the question the project was built around, which is whether the
model learns real structure or just memorizes a mapping. A linear probe cannot read out
component count or concentration unless the representation has arranged that information in a
simple form, and the pretrained encoder arranges it well enough that a linear readout of
concentration nearly matches a supervised encoder. That is structure, not surface
memorization. It also reframes the null result from the fine-tuning experiment as a statement
about the task and the training protocol rather than about the representation.

The difficulty sweep then adds a caution I did not expect, and one I would not have found by
staying on the easy regime. It is tempting to take "self-supervised pretraining learns
physical structure" as a general claim. It is not. The structure it learns is contingent on
the pretext objective staying informative, and a raw-reconstruction objective stops being
informative exactly where you would most want the help: in the low-SNR regime, where labels
are expensive and a good prior should matter most. The honest summary of the whole project is
narrower than the headline and more useful than it: masked pretraining on these spectra learns
real, linearly-decodable compositional structure; that structure is largely redundant when
labels are plentiful and the task is clean; and it degrades faster than the task itself does
as noise grows, for a reason we can point at.

## Limitations

The data is synthetic and, at the default settings, not very hard. Peaks are distinct, noise
is moderate, and mixtures contain at most five components, so the supervised task saturates
quickly. The model is small by design. The reconstruction objective has a height bias that
under-predicts tall peaks. The supervised encoder is an oracle ceiling for presence, since it
was trained on presence labels, so the fairer comparison is concentration, which it was not
trained on and where the pretrained encoder still nearly matches it. The random encoder is a
single draw rather than an average over initializations.

The robustness sweep adds two more. One encoder per difficulty is shared across the three
seeds, so the error bars there capture fine-tuning and probe variance but not
pretraining-run variance; a single unlucky pretraining run would be invisible to those bars,
and that is a real gap given how much weight the +0.082 carries. And the difficulty knob moves
three quantities at once, so a change along it cannot be attributed to noise, overlap, or
jitter individually. Finally, the entire study is one architecture at one size, and nothing
here speaks to what a larger model would do.

None of these invalidate the main comparisons, but they set the boundaries of what I am
claiming.

## Future work

The ablation redirects this section, which is the most useful thing it did. The obvious next
step *was* a noise-robust objective such as Noise2Noise, and that is now ruled out: if the
oracle clean target recovers 2 percent of the gap, no self-supervised approximation to it will
do better. I would have spent a week there.

The real next step is to distinguish the two live explanations, cheapest first. Re-probe the
existing encoders with CLS pooling and with a small MLP probe. That costs minutes, needs no
retraining, and tells you whether the collapse is a property of the representation or merely of
how I read it. Only if the representation genuinely lacks the information is the interesting
explanation on the table, namely that masked reconstruction is solvable locally and never
requires compound identity. That one predicts a specific fix: a pretext that cannot be solved
by local continuation. Masking whole contiguous regions much larger than a multiplet, or
masking one compound's peaks wherever they appear across the spectrum, would force the model to
integrate evidence globally instead of interpolating. Contrastive objectives over mixtures that
share components are another route to the same pressure.

Beyond that: swapping the parametric component library for real single-compound spectra from
nmrshiftdb2, which the library interface was designed to allow; a separation head that
reconstructs the individual component signals rather than classifying presence, which is the
version of the task closest to what a screening pipeline actually needs; an attention analysis
checking whether attention concentrates on peak regions rather than empty baseline, as a
second and independent view of what the model uses; and simply more seeds, since the paired
analysis shows that several of the marginal claims here would be settled one way or the other
by a handful more runs.

## Reproducibility

The project is Python and PyTorch with a short dependency list. Configuration lives in YAML
files that map onto typed dataclasses, so there are no hard-coded constants, and every entry
point seeds all random number generators from its config. The test suite has 46 tests covering
seeding determinism, the lineshapes, the reconstruction identity and other generator
invariants, model shapes and metrics, the masking and weight transfer, the fine-tuning paths,
the probes, the difficulty knob, and the pretext-target plumbing.

`python scripts/run_all.py` reruns every experiment in this report, in order, and regenerates
every figure; `--list` shows the stages and their runtimes, and `--skip-existing` resumes.
`python scripts/build_paper.py` rebuilds the PDF. Development ran on Apple Silicon (MPS); the
same code runs on a single CUDA GPU.

## Reproducibility

The project is Python and PyTorch with a small dependency list. Configuration lives in YAML
files that map onto typed dataclasses, so there are no hard-coded constants, and every entry
point seeds all random number generators from its config. Development ran on Apple Silicon
(MPS); the same code runs on a single CUDA GPU.

Setup:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Reproducing each result:

```
python scripts/visualize_mixtures.py --config configs/data/default.yaml   # data factory figures
python scripts/train_baseline.py    --config configs/train_baseline.yaml  # from-scratch baseline
python scripts/pretrain.py          --config configs/pretrain.yaml        # masked pretraining
python scripts/finetune_sweep.py    --config configs/finetune_sweep.yaml  # pretrained vs scratch
python scripts/probe.py             --config configs/probe.yaml           # linear probes
pytest                                                                    # 31 tests
```

Figures and logs are written under `experiments/`. The source is organized under
`src/spectral/` into `data` (the factory), `models` (the encoder, the classifier, and the
masked model), `training` (configs, metrics, and the fine-tuning loops), and `probing`
(feature extraction and the linear probes). Design decisions and their rationale are recorded
as they were made in `DECISIONS.md`.
