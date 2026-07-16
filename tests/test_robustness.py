"""Phase 6: the difficulty knob and the robustness-sweep plumbing.

The knob is load-bearing -- every Phase-6 number is indexed by it -- so these tests check
not just that the config fields move, but that the generated data actually gets harder:
more noise, more components. If the knob were silently a no-op, the sweep would produce a
flat, confidently wrong "the model is robust" curve.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from spectral.data.config import DataConfig
from spectral.data.generator import MixtureGenerator
from spectral.data.library import CompoundLibrary
from spectral.models.masked import make_batch_mask
from spectral.training.config import RobustnessExperimentConfig
from spectral.training.pretrain import pretrain_encoder


@pytest.fixture(scope="module")
def data_cfg() -> DataConfig:
    return DataConfig.from_yaml("configs/data/default.yaml")


@pytest.fixture(scope="module")
def library(data_cfg) -> CompoundLibrary:
    return CompoundLibrary.from_config(data_cfg.library)


def test_difficulty_endpoints_match_config(data_cfg):
    """d=0 and d=1 must land exactly on the declared easy/hard endpoints."""
    dc = data_cfg.difficulty
    easy, hard = data_cfg.at_difficulty(0.0), data_cfg.at_difficulty(1.0)

    assert easy.corruptions.snr == pytest.approx(dc.snr_easy)
    assert easy.corruptions.jitter_ppm == pytest.approx(dc.jitter_easy_ppm)
    assert easy.mixture.k_max == dc.kmax_easy

    assert hard.corruptions.snr == pytest.approx(dc.snr_hard)
    assert hard.corruptions.jitter_ppm == pytest.approx(dc.jitter_hard_ppm)
    assert hard.mixture.k_max == dc.kmax_hard


def test_difficulty_is_monotone(data_cfg):
    """Rising difficulty lowers SNR and raises jitter/component count -- no non-monotone dips."""
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    cfgs = [data_cfg.at_difficulty(d) for d in grid]

    snrs = [c.corruptions.snr for c in cfgs]
    jitters = [c.corruptions.jitter_ppm for c in cfgs]
    kmaxes = [c.mixture.k_max for c in cfgs]

    assert snrs == sorted(snrs, reverse=True) and snrs[0] > snrs[-1]
    assert jitters == sorted(jitters) and jitters[0] < jitters[-1]
    assert kmaxes == sorted(kmaxes) and kmaxes[0] < kmaxes[-1]


def test_at_difficulty_does_not_mutate_the_original(data_cfg):
    """The sweep calls at_difficulty in a loop; leaking state would corrupt later points."""
    before_snr = data_cfg.corruptions.snr
    before_kmax = data_cfg.mixture.k_max

    data_cfg.at_difficulty(1.0)

    assert data_cfg.corruptions.snr == before_snr
    assert data_cfg.mixture.k_max == before_kmax


@pytest.mark.parametrize("bad", [-0.1, 1.1, 2.0])
def test_at_difficulty_rejects_out_of_range(data_cfg, bad):
    with pytest.raises(ValueError):
        data_cfg.at_difficulty(bad)


def test_k_min_is_respected_when_kmax_would_fall_below_it(data_cfg):
    """kmax_easy can sit at/below k_min; clamping must never yield k_max < k_min."""
    cfg = data_cfg.at_difficulty(0.0)
    cfg.mixture.k_min = 4
    easy = cfg.at_difficulty(0.0)
    assert easy.mixture.k_max >= easy.mixture.k_min


def test_hard_data_is_actually_noisier_than_easy_data(data_cfg, library):
    """The empirical check: the knob must change the DATA, not just the config."""
    n = 64
    noise_std = {}
    for d in (0.0, 1.0):
        gen = MixtureGenerator(data_cfg.at_difficulty(d), library)
        noise_std[d] = float(np.mean([gen.generate(i).noise.std() for i in range(n)]))

    # Peak height is roughly comparable across regimes, so lower SNR => visibly more noise.
    assert noise_std[1.0] > 2 * noise_std[0.0]


def test_hard_data_has_more_components_on_average(data_cfg, library):
    n = 128
    mean_k = {}
    for d in (0.0, 1.0):
        gen = MixtureGenerator(data_cfg.at_difficulty(d), library)
        mean_k[d] = float(np.mean([gen.generate(i).present.sum() for i in range(n)]))

    assert mean_k[0.0] == pytest.approx(2.0)   # kmax_easy == k_min == 2 -> always exactly 2
    assert mean_k[1.0] > mean_k[0.0]


def test_robustness_config_loads(data_cfg):
    cfg = RobustnessExperimentConfig.from_yaml("configs/robustness.yaml")
    rcfg = cfg.robustness

    assert list(rcfg.difficulties) == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert all(0.0 <= d <= 1.0 for d in rcfg.difficulties)
    assert len(rcfg.seeds) >= 2                      # error bars need >1 seed
    assert rcfg.pool in {"mean", "cls"}
    # The pretraining recipe must match Phase 3's, or Phase 6 is not comparable to it.
    assert (rcfg.mask_ratio, rcfg.span_len) == (0.5, 4)


def test_robustness_streams_are_disjoint():
    """Pretrain / labeled / probe / supervised / val must not share mixture streams."""
    rcfg = RobustnessExperimentConfig.from_yaml("configs/robustness.yaml").robustness

    ft = {rcfg.labeled_seed_base + s for s in rcfg.seeds}
    probe = {rcfg.probe_seed_base + s for s in rcfg.seeds}
    fixed = {rcfg.pretrain_seed, rcfg.pretrain_seed + 1, rcfg.supervised_seed, rcfg.val_seed}

    assert not (ft & probe)
    assert not (ft & fixed)
    assert not (probe & fixed)
    assert len(fixed) == 4


def test_clean_target_is_noise_free_and_aligned(data_cfg, library):
    """The denoising target must be the same signal minus corruption, sample-for-sample."""
    from spectral.data.generator import MixtureGenerator
    from spectral.training.data import build_mixture_and_clean_tensors

    cfg = data_cfg.at_difficulty(1.0)
    observed, clean = build_mixture_and_clean_tensors(cfg, library, 20000, 8)
    gen = MixtureGenerator(_with_seed(cfg, 20000), library)

    assert observed.shape == clean.shape
    for i in range(8):
        s = gen.generate(i)
        # Rows line up with the generator, and clean really is the noise/baseline-free sum.
        assert np.allclose(clean[i].numpy(), s.clean_mixture, atol=1e-5)
        assert np.allclose(observed[i].numpy(), s.mixture, atol=1e-5)
    # At the hard end the two targets must differ substantially -- else the ablation is a no-op.
    assert (observed - clean).abs().mean() > 1e-3


def _with_seed(cfg, base_seed):
    import copy

    out = copy.deepcopy(cfg)
    out.base_seed = base_seed
    return out


def _masked_model():
    """Eval mode matters: with dropout on, two identical calls differ and the comparison
    below would be measuring dropout noise rather than target semantics."""
    from spectral.models.masked import MaskedSpectralModel

    torch.manual_seed(0)
    cfg = RobustnessExperimentConfig.from_yaml("configs/robustness.yaml").model
    return MaskedSpectralModel(cfg, n_points=2048).eval()


def test_masked_mse_default_target_is_the_input():
    """Passing target=x explicitly must equal the default -- guards the Phase-3 path."""
    model = _masked_model()
    x = torch.randn(4, 2048)
    mask = make_batch_mask(4, model.n_patches, 0.5, 4, np.random.default_rng(0))

    assert torch.allclose(model.masked_mse(x, mask), model.masked_mse(x, mask, x))


def test_masked_mse_with_a_different_target_changes_the_loss():
    model = _masked_model()
    x = torch.randn(4, 2048)
    clean = torch.zeros(4, 2048)
    mask = make_batch_mask(4, model.n_patches, 0.5, 4, np.random.default_rng(0))

    assert not torch.allclose(model.masked_mse(x, mask), model.masked_mse(x, mask, clean))


def test_pretrain_encoder_learns_and_is_reproducible(data_cfg, library):
    """Two runs at the same seed must agree exactly, and the loss must go down."""
    from spectral.training.data import build_mixture_tensor

    cfg = RobustnessExperimentConfig.from_yaml("configs/robustness.yaml")
    small = data_cfg.at_difficulty(0.5)
    x = build_mixture_tensor(small, library, 20000, 64)
    x_eval = build_mixture_tensor(small, library, 20001, 32)
    kwargs = dict(
        epochs=2, batch_size=32, lr=3e-4, weight_decay=0.01, mask_ratio=0.5,
        span_len=4, seed=0, device=torch.device("cpu"),
    )

    _, hist_a = pretrain_encoder(cfg.model, small.grid.n_points, x, x_eval, **kwargs)
    _, hist_b = pretrain_encoder(cfg.model, small.grid.n_points, x, x_eval, **kwargs)

    assert hist_a == hist_b                    # same seed -> bit-identical
    assert hist_a[-1][1] < hist_a[0][1]        # training loss decreases
