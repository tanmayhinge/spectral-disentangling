"""Phase 6 follow-up: is the PRETEXT TASK what breaks down at low SNR?

Phase 6 found that frozen pretrained features collapse at the hard end (presence macro-F1
0.529 at d=1.00) while a supervised encoder still reaches 0.921 on the same data. So the
information survives and is linearly decodable -- the masked-reconstruction pretext just
stops capturing it. The hypothesis: at SNR 6 the target is mostly unpredictable noise, so
the masked-MSE gradient is spent modelling noise instead of compound structure.

This ablation swaps ONLY the reconstruction target and re-probes the frozen encoder:
  raw   -- reconstruct the observed, noisy signal (the Phase-3 pretext)
  clean -- reconstruct the noise-free component sum (a denoising pretext)

IMPORTANT / FAIRNESS: `clean` is ground truth that a real unlabeled setting would not give
you, so the clean arm is NOT self-supervised. It is a DIAGNOSTIC UPPER BOUND that isolates
the effect of noise in the target -- it is not a deployable method. If it recovers the gap,
the pretext was the bottleneck and a noise-robust objective is worth building; if it does
not, the hypothesis is wrong. Either way the conclusion is about the diagnosis, not a method.

Reuses the raw-target encoders from scripts/robustness_sweep.py when present.

Run:  python scripts/pretext_ablation.py --config configs/pretext_ablation.yaml
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
from spectral.probing.probes import probe_concentration, probe_presence, standardize
from spectral.seeding import seed_everything
from spectral.training.config import RobustnessExperimentConfig
from spectral.training.data import build_labeled_tensors, build_mixture_and_clean_tensors
from spectral.training.finetune import train_presence_encoder
from spectral.training.pretrain import pretrain_encoder
from spectral.utils import PROJECT_ROOT, get_device

ARM_COLORS = {"random": "tab:gray", "raw": "tab:red", "clean": "tab:green", "supervised": "tab:blue"}


def resolve_device(name: str) -> torch.device:
    return get_device() if name == "auto" else torch.device(name)


def get_encoder(cfg, data_d, library, d, target: str, device) -> SpectralEncoder:
    """Pretrain (or reload) the encoder for one difficulty and one pretext target."""
    rcfg = cfg.robustness
    n_points = cfg.data.grid.n_points
    # The raw arm shares its cache with the Phase-6 sweep, so we do not retrain it.
    name = f"encoder_d{d:.2f}.pt" if target == "raw" else f"encoder_clean_d{d:.2f}.pt"
    path = PROJECT_ROOT / rcfg.encoder_dir / name

    if path.exists():
        enc = SpectralEncoder(cfg.model, n_points)
        enc.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
        print(f"  [{target:5s}] reusing {name}")
        return enc

    x_pre, clean_pre = build_mixture_and_clean_tensors(data_d, library, rcfg.pretrain_seed, rcfg.n_pretrain)
    x_ev, clean_ev = build_mixture_and_clean_tensors(data_d, library, rcfg.pretrain_seed + 1, rcfg.n_pretrain_eval)
    tgt_pre, tgt_ev = (None, None) if target == "raw" else (clean_pre, clean_ev)

    model, history = pretrain_encoder(
        cfg.model, n_points, x_pre, x_ev,
        epochs=rcfg.pretrain_epochs, batch_size=rcfg.pretrain_batch_size, lr=rcfg.pretrain_lr,
        weight_decay=rcfg.pretrain_weight_decay, mask_ratio=rcfg.mask_ratio,
        span_len=rcfg.span_len, seed=rcfg.pretrain_model_seed, device=device,
        x_pretrain_target=tgt_pre, x_eval_target=tgt_ev,
        log_fn=lambda s: print(f"    [{target} d={d:.2f}] {s}"),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.encoder.state_dict(), path)
    # Losses across arms are NOT comparable (a clean target is intrinsically easier to hit).
    print(f"  [{target:5s}] final held-out MSE={history[-1][2]:.5f} (not comparable across arms)")
    return model.encoder.cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretext-target ablation: raw vs clean.")
    parser.add_argument("--config", default="configs/pretext_ablation.yaml")
    args = parser.parse_args()

    cfg = RobustnessExperimentConfig.from_yaml(args.config)
    rcfg = cfg.robustness
    device = resolve_device(rcfg.device)
    n_points = cfg.data.grid.n_points

    library = CompoundLibrary.from_config(cfg.data.library)
    n_classes = library.n_compounds
    print(f"device={device}  difficulties={rcfg.difficulties}  seeds={rcfg.seeds}\n")

    rows = []  # (d, arm, seed, presence_f1, conc_mae)
    for d in rcfg.difficulties:
        data_d = cfg.data.at_difficulty(d)
        print(f"=== difficulty {d:.2f}  snr={data_d.corruptions.snr:.1f} ===")

        encoders = {arm: get_encoder(cfg, data_d, library, d, arm, device) for arm in ("raw", "clean")}
        seed_everything(0)
        encoders["random"] = SpectralEncoder(cfg.model, n_points)

        if rcfg.include_supervised:
            xs, ys, _ = build_labeled_tensors(data_d, library, rcfg.supervised_seed, rcfg.supervised_train_n)
            clf = train_presence_encoder(
                cfg.model, n_points, n_classes, xs, ys, steps=rcfg.supervised_steps,
                lr=rcfg.supervised_lr, batch_size=64, seed=0, device=device,
            )
            encoders["supervised"] = clf.encoder.cpu()

        xv, pv, cv = build_labeled_tensors(data_d, library, rcfg.val_seed, rcfg.probe_val_n)
        val_feats = {a: extract_features(e, xv, device, rcfg.pool) for a, e in encoders.items()}

        for seed in rcfg.seeds:
            xt, pt, ct = build_labeled_tensors(data_d, library, rcfg.probe_seed_base + seed, rcfg.probe_train_n)
            for arm, enc in encoders.items():
                ftr, fva = standardize(extract_features(enc, xt, device, rcfg.pool), val_feats[arm])
                f1 = probe_presence(ftr, pt, fva, pv, rcfg.probe_steps, rcfg.probe_lr, rcfg.threshold, seed)
                mae = probe_concentration(ftr, ct, fva, cv, rcfg.probe_steps, rcfg.probe_lr, seed)
                rows.append((d, arm, seed, f1, mae))
                print(f"  [probe] d={d:.2f}  {arm:11s} seed={seed}  presence_f1={f1:.3f}  conc_mae={mae:.3f}")
        print()

    _report(rows, list(rcfg.difficulties))


def _mean_std(vals):
    a = np.array(vals)
    return float(a.mean()), float(a.std())


def _report(rows, ds) -> None:
    out = PROJECT_ROOT / "experiments"
    out.mkdir(exist_ok=True)
    with open(out / "pretext_ablation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["difficulty", "arm", "seed", "presence_f1", "conc_mae"])
        w.writerows(rows)

    arms = list(dict.fromkeys(r[1] for r in rows))
    metrics = [("presence_f1", 3, "presence macro-F1", True), ("conc_mae", 4, "concentration MAE", False)]
    buckets = defaultdict(list)
    for row in rows:
        for key, idx, _l, _u in metrics:
            buckets[(key, row[1], row[0])].append(row[idx])

    for key, _idx, label, higher in metrics:
        print(f"\n=== {label} on frozen features ({'higher' if higher else 'lower'}=better) ===")
        print(f"  {'difficulty':>10} | " + " | ".join(f"{a:>16}" for a in arms))
        for d in ds:
            cells = " | ".join(
                f"{_mean_std(buckets[(key, a, d)])[0]:6.3f} +/- {_mean_std(buckets[(key, a, d)])[1]:5.3f}"
                for a in arms
            )
            print(f"  {d:>10.2f} | {cells}")

    print("\n=== the diagnostic: how much of the raw-vs-supervised gap does clean recover? ===")
    print(f"  {'difficulty':>10} | {'raw':>7} | {'clean':>7} | {'supervised':>10} | {'gap recovered':>13}")
    for d in ds:
        raw = _mean_std(buckets[("presence_f1", "raw", d)])[0]
        clean = _mean_std(buckets[("presence_f1", "clean", d)])[0]
        sup = _mean_std(buckets[("presence_f1", "supervised", d)])[0]
        gap = sup - raw
        frac = f"{(clean - raw) / gap:+.0%}" if abs(gap) > 1e-6 else "n/a"
        print(f"  {d:>10.2f} | {raw:7.3f} | {clean:7.3f} | {sup:10.3f} | {frac:>13}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (key, _idx, label, higher) in zip(axes, metrics):
        for a in arms:
            means = np.array([_mean_std(buckets[(key, a, d)])[0] for d in ds])
            stds = np.array([_mean_std(buckets[(key, a, d)])[1] for d in ds])
            ax.plot(ds, means, "o-", color=ARM_COLORS.get(a), label=a)
            ax.fill_between(ds, means - stds, means + stds, color=ARM_COLORS.get(a), alpha=0.2)
        ax.set_xlabel("difficulty (0 = easy, 1 = hard)")
        ax.set_ylabel(label + ("" if higher else " (lower = better)"))
        ax.grid(alpha=0.3)
        ax.legend()
        if higher:
            ax.set_ylim(0, 1)
    fig.suptitle("Pretext ablation: raw vs clean reconstruction target (clean = oracle diagnostic)")
    fig.tight_layout()
    fig.savefig(out / "pretext_ablation.png", dpi=120)
    plt.close(fig)
    print(f"\nsaved -> {out / 'pretext_ablation.csv'}, {out / 'pretext_ablation.png'}")


if __name__ == "__main__":
    main()
