"""Phase 4: the headline comparison -- pretrained vs from-scratch across a label budget.

For each label size and seed, fine-tune the full model twice (scratch init and pretrained
init) under an identical recipe, and record the best validation macro-F1. Produces the
label-efficiency plot (with std bands over seeds), a CSV, and a printed summary table.

Hypothesis: pretraining helps most when labels are scarce.

Run:  python scripts/finetune_sweep.py --config configs/finetune_sweep.yaml
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
from spectral.training.config import FinetuneExperimentConfig
from spectral.training.data import build_presence_tensors
from spectral.training.finetune import finetune_once
from spectral.utils import PROJECT_ROOT, get_device

INITS = ["scratch", "pretrained"]


def resolve_device(name: str) -> torch.device:
    return get_device() if name == "auto" else torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Label-efficiency sweep: pretrained vs scratch.")
    parser.add_argument("--config", default="configs/finetune_sweep.yaml")
    args = parser.parse_args()

    cfg = FinetuneExperimentConfig.from_yaml(args.config)
    fcfg = cfg.finetune
    device = resolve_device(fcfg.device)
    n_points = cfg.data.grid.n_points

    library = CompoundLibrary.from_config(cfg.data.library)
    n_classes = library.n_compounds
    x_val, y_val = build_presence_tensors(cfg.data, library, fcfg.val_seed, fcfg.n_val)
    max_n = max(fcfg.label_sizes)

    print(f"device={device}  sizes={fcfg.label_sizes}  seeds={fcfg.seeds}  "
          f"steps={fcfg.max_steps}  val={fcfg.n_val}\n")

    results = []  # (n, init, seed, macro_f1)
    for seed in fcfg.seeds:
        # One labeled pool per seed; smaller sizes are prefixes of the larger pool.
        full_x, full_y = build_presence_tensors(cfg.data, library, fcfg.labeled_seed_base + seed, max_n)
        for n in fcfg.label_sizes:
            xt, yt = full_x[:n], full_y[:n]
            for init in INITS:
                f1 = finetune_once(
                    cfg.model, n_points, n_classes, xt, yt, x_val, y_val,
                    init=init, seed=seed, max_steps=fcfg.max_steps, eval_every=fcfg.eval_every,
                    batch_size=fcfg.batch_size, lr=fcfg.lr, weight_decay=fcfg.weight_decay,
                    threshold=fcfg.threshold, pretrained_encoder=fcfg.pretrained_encoder, device=device,
                )
                results.append((n, init, seed, f1))
                print(f"  n={n:5d}  {init:10s}  seed={seed}  macro_f1={f1:.3f}")

    _report(results, fcfg.label_sizes)


def _aggregate(results, sizes):
    """-> {init: {n: (mean, std)}} over seeds."""
    buckets = defaultdict(list)
    for n, init, _seed, f1 in results:
        buckets[(init, n)].append(f1)
    agg = {init: {} for init in INITS}
    for init in INITS:
        for n in sizes:
            vals = np.array(buckets[(init, n)])
            agg[init][n] = (float(vals.mean()), float(vals.std()))
    return agg


def _report(results, sizes) -> None:
    out = PROJECT_ROOT / "experiments"
    out.mkdir(exist_ok=True)

    with open(out / "label_efficiency.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_labels", "init", "seed", "macro_f1"])
        w.writerows(results)

    agg = _aggregate(results, sizes)

    print("\n=== label efficiency (mean macro-F1 over seeds) ===")
    print(f"{'n_labels':>8} | {'scratch':>16} | {'pretrained':>16} | {'gain':>6}")
    for n in sizes:
        sm, ss = agg["scratch"][n]
        pm, ps = agg["pretrained"][n]
        print(f"{n:>8} | {sm:6.3f} +/- {ss:5.3f} | {pm:6.3f} +/- {ps:5.3f} | {pm - sm:+.3f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for init, color in [("scratch", "tab:gray"), ("pretrained", "tab:red")]:
        means = np.array([agg[init][n][0] for n in sizes])
        stds = np.array([agg[init][n][1] for n in sizes])
        ax.plot(sizes, means, "o-", color=color, label=init)
        ax.fill_between(sizes, means - stds, means + stds, color=color, alpha=0.2)
    ax.set_xscale("log")
    ax.set_xlabel("number of labeled examples")
    ax.set_ylabel("best validation macro-F1")
    ax.set_ylim(0, 1)
    ax.set_title("Label efficiency: pretrained vs from-scratch")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "label_efficiency.png", dpi=120)
    plt.close(fig)

    print(f"\nsaved -> {out / 'label_efficiency.csv'}")
    print(f"saved -> {out / 'label_efficiency.png'}")


if __name__ == "__main__":
    main()
