"""Phase 5: linear probes on frozen features -- did pretraining learn real structure?

Freezes each encoder and trains only a linear layer on its features. Compares three
encoders -- random (floor), pretrained (SSL), supervised (labels-trained ceiling) -- on
three targets: presence (macro-F1), component count K (accuracy), concentration (MAE).
Produces a presence label-efficiency plot and a summary table.

Run:  python scripts/probe.py --config configs/probe.yaml
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
from spectral.training.config import ProbeExperimentConfig
from spectral.training.data import build_labeled_tensors
from spectral.training.finetune import train_presence_encoder
from spectral.utils import PROJECT_ROOT, get_device

ENCODER_COLORS = {"random": "tab:gray", "pretrained": "tab:red", "supervised": "tab:blue"}


def resolve_device(name: str) -> torch.device:
    return get_device() if name == "auto" else torch.device(name)


def build_encoders(cfg, library, n_points, n_classes, device) -> dict[str, SpectralEncoder]:
    """Return the frozen encoders to compare (random, pretrained, optional supervised)."""
    pcfg = cfg.probe
    encoders: dict[str, SpectralEncoder] = {}

    seed_everything(0)
    encoders["random"] = SpectralEncoder(cfg.model, n_points)  # untrained control

    pre = SpectralEncoder(cfg.model, n_points)
    path = PROJECT_ROOT / pcfg.pretrained_encoder
    pre.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    encoders["pretrained"] = pre

    if pcfg.include_supervised:
        xs, ys, _ = build_labeled_tensors(cfg.data, library, pcfg.supervised_seed, pcfg.supervised_train_n)
        clf = train_presence_encoder(
            cfg.model, n_points, n_classes, xs, ys,
            steps=pcfg.supervised_steps, lr=pcfg.supervised_lr, batch_size=64,
            seed=0, device=device,
        )
        encoders["supervised"] = clf.encoder.cpu()
    return encoders


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear probes on frozen features.")
    parser.add_argument("--config", default="configs/probe.yaml")
    args = parser.parse_args()

    cfg = ProbeExperimentConfig.from_yaml(args.config)
    pcfg = cfg.probe
    device = resolve_device(pcfg.device)
    n_points = cfg.data.grid.n_points

    library = CompoundLibrary.from_config(cfg.data.library)
    n_classes = library.n_compounds
    max_n = max(pcfg.probe_label_sizes)

    # Validation set + its frozen features per encoder (computed once).
    xv, pv, cv = build_labeled_tensors(cfg.data, library, pcfg.val_seed, pcfg.probe_val_n)
    encoders = build_encoders(cfg, library, n_points, n_classes, device)
    val_feats = {name: extract_features(enc, xv, device, pcfg.pool) for name, enc in encoders.items()}
    print(f"device={device}  encoders={list(encoders)}  sizes={pcfg.probe_label_sizes}  "
          f"seeds={pcfg.seeds}  pool={pcfg.pool}\n")

    presence_rows = []                 # (n, encoder, seed, macro_f1)
    count_rows, conc_rows = [], []     # (encoder, seed, metric) at full probe size
    for seed in pcfg.seeds:
        xt, pt, ct = build_labeled_tensors(cfg.data, library, pcfg.labeled_seed_base + seed, max_n)
        for name, enc in encoders.items():
            ftr = extract_features(enc, xt, device, pcfg.pool)
            fva = val_feats[name]
            ftr_s, fva_s = standardize(ftr, fva)

            for n in pcfg.probe_label_sizes:
                f1 = probe_presence(ftr_s[:n], pt[:n], fva_s, pv, pcfg.probe_steps, pcfg.probe_lr, pcfg.threshold, seed)
                presence_rows.append((n, name, seed, f1))

            acc = probe_count(ftr_s, pt, fva_s, pv, pcfg.probe_steps, pcfg.probe_lr, seed)
            mae = probe_concentration(ftr_s, ct, fva_s, cv, pcfg.probe_steps, pcfg.probe_lr, seed)
            count_rows.append((name, seed, acc))
            conc_rows.append((name, seed, mae))
            print(f"  seed={seed}  {name:11s}  count_acc={acc:.3f}  conc_mae={mae:.3f}")

    _report(presence_rows, count_rows, conc_rows, pcfg.probe_label_sizes, list(encoders))


def _mean_std(vals):
    a = np.array(vals)
    return float(a.mean()), float(a.std())


def _report(presence_rows, count_rows, conc_rows, sizes, names) -> None:
    out = PROJECT_ROOT / "experiments"
    out.mkdir(exist_ok=True)
    with open(out / "probe_presence.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["n_labels", "encoder", "seed", "macro_f1"]); w.writerows(presence_rows)

    # Presence label-efficiency curve.
    pres = defaultdict(list)
    for n, name, _s, f1 in presence_rows:
        pres[(name, n)].append(f1)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in names:
        means = np.array([_mean_std(pres[(name, n)])[0] for n in sizes])
        stds = np.array([_mean_std(pres[(name, n)])[1] for n in sizes])
        ax.plot(sizes, means, "o-", color=ENCODER_COLORS.get(name), label=name)
        ax.fill_between(sizes, means - stds, means + stds, color=ENCODER_COLORS.get(name), alpha=0.2)
    ax.set_xscale("log"); ax.set_ylim(0, 1)
    ax.set_xlabel("linear-probe training labels"); ax.set_ylabel("val macro-F1 (frozen features)")
    ax.set_title("Linear probe: presence, frozen features"); ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "probe_presence.png", dpi=120); plt.close(fig)

    # Count + concentration tables (at full probe size).
    count = defaultdict(list); conc = defaultdict(list)
    for name, _s, acc in count_rows:
        count[name].append(acc)
    for name, _s, mae in conc_rows:
        conc[name].append(mae)

    print("\n=== presence macro-F1 (frozen linear probe) ===")
    print(f"{'n_labels':>8} | " + " | ".join(f"{n:>16}" for n in names))
    for n in sizes:
        cells = " | ".join(f"{_mean_std(pres[(name, n)])[0]:6.3f} +/- {_mean_std(pres[(name, n)])[1]:5.3f}" for name in names)
        print(f"{n:>8} | {cells}")

    print("\n=== count accuracy / concentration MAE (frozen linear probe, full probe set) ===")
    for name in names:
        cm, cs = _mean_std(count[name]); mm, ms = _mean_std(conc[name])
        print(f"  {name:11s}  count_acc={cm:.3f} +/- {cs:.3f}   conc_mae={mm:.3f} +/- {ms:.3f}")
    print(f"\nsaved -> {out / 'probe_presence.png'}")
    print(f"saved -> {out / 'probe_presence.csv'}")


if __name__ == "__main__":
    main()
