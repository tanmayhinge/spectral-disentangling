"""Phase 6: robustness sweep -- how do the Phase-4 and Phase-5 results move with difficulty?

The difficulty knob (see DataConfig.at_difficulty) interpolates SNR, max component count, and
peak jitter from an easy regime (d=0) to a hard one (d=1). At each difficulty we pretrain a
FRESH encoder on unlabeled mixtures from that same difficulty -- reusing one encoder would
confound "the task got harder" with "the pretraining data no longer matches the test data" --
and then run both measurements:

- fine-tune arm: pretrained vs from-scratch at small label budgets. Phase 4 found no gap in
  the easy regime. The prediction under test: the gap reappears once the task is hard enough
  that a from-scratch model can no longer catch up within a fixed budget.
- probe arm:     frozen linear probes (random / pretrained / supervised). Phase 5 found a
  large pretrained-over-random gap. The question: where does it break down?

Run:  python scripts/robustness_sweep.py --config configs/robustness.yaml
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from spectral.data.library import CompoundLibrary
from spectral.models.transformer import SpectralEncoder
from spectral.probing.features import extract_features
from spectral.probing.probes import (
    probe_concentration,
    probe_count,
    probe_presence,
    standardize,
)
from spectral.seeding import seed_everything
from spectral.training.config import RobustnessExperimentConfig
from spectral.training.data import build_labeled_tensors, build_mixture_tensor
from spectral.training.finetune import finetune_once, train_presence_encoder
from spectral.training.pretrain import pretrain_encoder
from spectral.utils import PROJECT_ROOT, get_device

INITS = ["scratch", "pretrained"]
INIT_COLORS = {"scratch": "tab:gray", "pretrained": "tab:red"}
ENCODER_COLORS = {"random": "tab:gray", "pretrained": "tab:red", "supervised": "tab:blue"}


def resolve_device(name: str) -> torch.device:
    return get_device() if name == "auto" else torch.device(name)


def encoder_path(rcfg, d: float):
    return PROJECT_ROOT / rcfg.encoder_dir / f"encoder_d{d:.2f}.pt"


def pretrain_at_difficulty(cfg, data_d, library, d: float, device) -> SpectralEncoder:
    """Pretrain (or reload) the encoder for one difficulty and return it frozen on CPU."""
    rcfg = cfg.robustness
    path = encoder_path(rcfg, d)
    n_points = cfg.data.grid.n_points

    if path.exists():  # resume support: the sweep is long, don't redo finished pretraining
        enc = SpectralEncoder(cfg.model, n_points)
        enc.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
        print(f"  [pretrain] reusing existing encoder -> {path.name}")
        return enc

    x_pre = build_mixture_tensor(data_d, library, rcfg.pretrain_seed, rcfg.n_pretrain)
    x_eval = build_mixture_tensor(data_d, library, rcfg.pretrain_seed + 1, rcfg.n_pretrain_eval)
    model, history = pretrain_encoder(
        cfg.model, n_points, x_pre, x_eval,
        epochs=rcfg.pretrain_epochs, batch_size=rcfg.pretrain_batch_size, lr=rcfg.pretrain_lr,
        weight_decay=rcfg.pretrain_weight_decay, mask_ratio=rcfg.mask_ratio,
        span_len=rcfg.span_len, seed=rcfg.pretrain_model_seed, device=device,
        log_fn=lambda s: print(f"    [pretrain d={d:.2f}] {s}"),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.encoder.state_dict(), path)

    with open(path.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_mse", "eval_mse"])
        w.writerows(history)
    print(f"  [pretrain] final held-out MSE={history[-1][2]:.5f}  saved -> {path.name}")
    return model.encoder.cpu()


def build_probe_encoders(cfg, data_d, library, pretrained_enc, n_classes, device):
    """The three frozen encoders to compare at one difficulty."""
    rcfg = cfg.robustness
    n_points = cfg.data.grid.n_points

    seed_everything(0)
    encoders = {"random": SpectralEncoder(cfg.model, n_points)}  # untrained control
    encoders["pretrained"] = pretrained_enc

    if rcfg.include_supervised:
        xs, ys, _ = build_labeled_tensors(data_d, library, rcfg.supervised_seed, rcfg.supervised_train_n)
        clf = train_presence_encoder(
            cfg.model, n_points, n_classes, xs, ys,
            steps=rcfg.supervised_steps, lr=rcfg.supervised_lr, batch_size=64,
            seed=0, device=device,
        )
        encoders["supervised"] = clf.encoder.cpu()
    return encoders


def run_finetune_arm(cfg, data_d, library, d, n_classes, device) -> list:
    """Pretrained vs scratch at each label budget and seed -> rows of (d, n, init, seed, f1)."""
    rcfg = cfg.robustness
    n_points = cfg.data.grid.n_points
    rows = []

    x_val, y_val = _presence_val(cfg, data_d, library)
    max_n = max(rcfg.ft_label_sizes)
    for seed in rcfg.seeds:
        full_x, full_y, _ = build_labeled_tensors(data_d, library, rcfg.labeled_seed_base + seed, max_n)
        for n in rcfg.ft_label_sizes:
            for init in INITS:
                f1 = finetune_once(
                    cfg.model, n_points, n_classes, full_x[:n], full_y[:n], x_val, y_val,
                    init=init, seed=seed, max_steps=rcfg.max_steps, eval_every=rcfg.eval_every,
                    batch_size=rcfg.batch_size, lr=rcfg.lr, weight_decay=rcfg.weight_decay,
                    threshold=rcfg.threshold, pretrained_encoder=str(encoder_path(rcfg, d)),
                    device=device,
                )
                rows.append((d, n, init, seed, f1))
                print(f"  [finetune] d={d:.2f}  n={n:4d}  {init:10s}  seed={seed}  macro_f1={f1:.3f}")
    return rows


def _presence_val(cfg, data_d, library):
    """Validation mixtures at this difficulty -> (x, presence)."""
    rcfg = cfg.robustness
    x, present, _ = build_labeled_tensors(data_d, library, rcfg.val_seed, rcfg.n_val)
    return x, present


def run_probe_arm(cfg, data_d, library, d, encoders, device) -> list:
    """Frozen linear probes -> rows of (d, encoder, seed, presence_f1, count_acc, conc_mae)."""
    rcfg = cfg.robustness
    rows = []

    xv, pv, cv = build_labeled_tensors(data_d, library, rcfg.val_seed, rcfg.probe_val_n)
    val_feats = {name: extract_features(enc, xv, device, rcfg.pool) for name, enc in encoders.items()}

    for seed in rcfg.seeds:
        xt, pt, ct = build_labeled_tensors(data_d, library, rcfg.probe_seed_base + seed, rcfg.probe_train_n)
        for name, enc in encoders.items():
            ftr, fva = standardize(extract_features(enc, xt, device, rcfg.pool), val_feats[name])
            f1 = probe_presence(ftr, pt, fva, pv, rcfg.probe_steps, rcfg.probe_lr, rcfg.threshold, seed)
            acc = probe_count(ftr, pt, fva, pv, rcfg.probe_steps, rcfg.probe_lr, seed)
            mae = probe_concentration(ftr, ct, fva, cv, rcfg.probe_steps, rcfg.probe_lr, seed)
            rows.append((d, name, seed, f1, acc, mae))
            print(f"  [probe]    d={d:.2f}  {name:11s}  seed={seed}  "
                  f"presence_f1={f1:.3f}  count_acc={acc:.3f}  conc_mae={mae:.3f}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6: robustness sweep over difficulty.")
    parser.add_argument("--config", default="configs/robustness.yaml")
    args = parser.parse_args()

    cfg = RobustnessExperimentConfig.from_yaml(args.config)
    rcfg = cfg.robustness
    device = resolve_device(rcfg.device)

    library = CompoundLibrary.from_config(cfg.data.library)
    n_classes = library.n_compounds

    print(f"device={device}  difficulties={rcfg.difficulties}  seeds={rcfg.seeds}\n"
          f"ft_sizes={rcfg.ft_label_sizes}  probe_n={rcfg.probe_train_n}\n")

    ft_rows, probe_rows, regimes = [], [], []
    for d in rcfg.difficulties:
        data_d = cfg.data.at_difficulty(d)
        c = data_d.corruptions
        print(f"=== difficulty {d:.2f}  snr={c.snr:.1f}  "
              f"k={data_d.mixture.k_min}..{data_d.mixture.k_max}  jitter={c.jitter_ppm:.4f} ppm ===")
        regimes.append((d, c.snr, data_d.mixture.k_min, data_d.mixture.k_max, c.jitter_ppm))

        pretrained_enc = pretrain_at_difficulty(cfg, data_d, library, d, device)
        ft_rows += run_finetune_arm(cfg, data_d, library, d, n_classes, device)
        encoders = build_probe_encoders(cfg, data_d, library, pretrained_enc, n_classes, device)
        probe_rows += run_probe_arm(cfg, data_d, library, d, encoders, device)
        print()

    _report(ft_rows, probe_rows, regimes, rcfg)


def _mean_std(vals):
    a = np.array(vals)
    return float(a.mean()), float(a.std())


def _report(ft_rows, probe_rows, regimes, rcfg) -> None:
    out = PROJECT_ROOT / "experiments"
    out.mkdir(exist_ok=True)
    ds = list(rcfg.difficulties)

    with open(out / "robustness_finetune.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["difficulty", "n_labels", "init", "seed", "macro_f1"])
        w.writerows(ft_rows)
    with open(out / "robustness_probe.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["difficulty", "encoder", "seed", "presence_f1", "count_acc", "conc_mae"])
        w.writerows(probe_rows)
    with open(out / "robustness_regimes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["difficulty", "snr", "k_min", "k_max", "jitter_ppm"])
        w.writerows(regimes)

    # Where the easy end pins k_max to k_min, every mixture has the same K, so the count
    # probe has a single class and scores 1.0 for any encoder (random included). That is an
    # artifact of the knob, not a result -- flag it rather than reporting it as a number.
    degenerate = {d for d, _snr, k_min, k_max, _j in regimes if k_max == k_min}

    _report_finetune(ft_rows, ds, rcfg, out)
    _report_probe(probe_rows, ds, out, degenerate)
    print(f"\nsaved -> {out / 'robustness_finetune.csv'}, {out / 'robustness_probe.csv'}")


def _report_finetune(ft_rows, ds, rcfg, out) -> None:
    buckets = defaultdict(list)
    for d, n, init, _seed, f1 in ft_rows:
        buckets[(n, init, d)].append(f1)

    print("\n=== fine-tuning: macro-F1 vs difficulty (mean +/- std over seeds) ===")
    for n in rcfg.ft_label_sizes:
        print(f"\n  n_labels={n}")
        print(f"  {'difficulty':>10} | {'scratch':>16} | {'pretrained':>16} | {'gain':>6}")
        for d in ds:
            sm, ss = _mean_std(buckets[(n, "scratch", d)])
            pm, ps = _mean_std(buckets[(n, "pretrained", d)])
            print(f"  {d:>10.2f} | {sm:6.3f} +/- {ss:5.3f} | {pm:6.3f} +/- {ps:5.3f} | {pm - sm:+.3f}")

    sizes = list(rcfg.ft_label_sizes)
    fig, axes = plt.subplots(1, len(sizes), figsize=(6 * len(sizes), 4.6), squeeze=False)
    for col, n in enumerate(sizes):
        ax = axes[0][col]
        for init in INITS:
            means = np.array([_mean_std(buckets[(n, init, d)])[0] for d in ds])
            stds = np.array([_mean_std(buckets[(n, init, d)])[1] for d in ds])
            ax.plot(ds, means, "o-", color=INIT_COLORS[init], label=init)
            ax.fill_between(ds, means - stds, means + stds, color=INIT_COLORS[init], alpha=0.2)
        ax.set_title(f"{n} labels")
        ax.set_xlabel("difficulty (0 = easy, 1 = hard)")
        ax.set_ylabel("best validation macro-F1")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Robustness: pretrained vs from-scratch across difficulty")
    fig.tight_layout()
    fig.savefig(out / "robustness_finetune.png", dpi=120)
    plt.close(fig)
    print(f"\nsaved -> {out / 'robustness_finetune.png'}")


def _report_probe(probe_rows, ds, out, degenerate) -> None:
    names = list(dict.fromkeys(r[1] for r in probe_rows))  # preserve encounter order
    metrics = [("presence_f1", 3, "presence macro-F1", True),
               ("count_acc", 4, "count accuracy", True),
               ("conc_mae", 5, "concentration MAE", False)]

    buckets = defaultdict(list)
    for row in probe_rows:
        d, name = row[0], row[1]
        for key, idx, _label, _up in metrics:
            buckets[(key, name, d)].append(row[idx])

    # The count probe is only meaningful where K actually varies.
    def valid(key, d) -> bool:
        return not (key == "count_acc" and d in degenerate)

    print("\n=== frozen linear probes vs difficulty (mean over seeds) ===")
    for key, _idx, label, higher_better in metrics:
        arrow = "higher=better" if higher_better else "lower=better"
        print(f"\n  {label} ({arrow})")
        print(f"  {'difficulty':>10} | " + " | ".join(f"{nm:>16}" for nm in names))
        for d in ds:
            if not valid(key, d):
                print(f"  {d:>10.2f} | " + " | ".join(f"{'n/a (K constant)':>16}" for _ in names))
                continue
            cells = " | ".join(
                f"{_mean_std(buckets[(key, nm, d)])[0]:6.3f} +/- {_mean_std(buckets[(key, nm, d)])[1]:5.3f}"
                for nm in names
            )
            print(f"  {d:>10.2f} | {cells}")
    if degenerate:
        print(f"\n  note: count probe omitted at difficulty {sorted(degenerate)} -- every mixture "
              f"there has the same K, so any encoder scores 1.0 and the metric says nothing.")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (key, _idx, label, higher_better) in zip(axes, metrics):
        xs = [d for d in ds if valid(key, d)]
        for nm in names:
            means = np.array([_mean_std(buckets[(key, nm, d)])[0] for d in xs])
            stds = np.array([_mean_std(buckets[(key, nm, d)])[1] for d in xs])
            ax.plot(xs, means, "o-", color=ENCODER_COLORS.get(nm), label=nm)
            ax.fill_between(xs, means - stds, means + stds, color=ENCODER_COLORS.get(nm), alpha=0.2)
        ax.set_xlabel("difficulty (0 = easy, 1 = hard)")
        ax.set_ylabel(label + ("" if higher_better else " (lower = better)"))
        ax.set_xlim(min(ds) - 0.05, max(ds) + 0.05)
        ax.grid(alpha=0.3)
        ax.legend()
        if higher_better:
            ax.set_ylim(0, 1)
        if key == "count_acc" and degenerate:
            ax.set_title("K constant at d=" + ", ".join(f"{d:g}" for d in sorted(degenerate)) + " -> omitted",
                         fontsize=8)
    fig.suptitle("Robustness: frozen linear probes across difficulty")
    fig.tight_layout()
    fig.savefig(out / "robustness_probe.png", dpi=120)
    plt.close(fig)
    print(f"\nsaved -> {out / 'robustness_probe.png'}")


if __name__ == "__main__":
    main()
