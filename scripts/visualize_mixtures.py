"""Checkpoint 1 visualization.

Generates a few mixtures and produces two figures:
  1. library.png     - the 12 compound "fingerprints" the mixtures are built from.
  2. mixtures.png     - per sample: (left) the mixture with its component signals
     overlaid and the clean sum dashed; (right) the reconstruction residual
     (mixture - clean sum), which should look like baseline drift + noise.

The left panels make the core claim visible: sum of components (+ corruptions) = mixture.
NMR convention: the ppm axis is drawn decreasing left-to-right.

Run:  python scripts/visualize_mixtures.py --config configs/data/default.yaml --n 4
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")  # no display needed; we save PNGs
import matplotlib.pyplot as plt

from spectral.data.config import DataConfig
from spectral.data.generator import MixtureGenerator
from spectral.data.library import CompoundLibrary
from spectral.seeding import seed_everything
from spectral.utils import PROJECT_ROOT


def plot_library(library: CompoundLibrary, grid, out_path) -> None:
    n = library.n_compounds
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.2 * rows), squeeze=False)
    for idx in range(n):
        ax = axes[idx // cols][idx % cols]
        ax.plot(grid, library.render(idx, grid), lw=1.0)
        ax.set_title(f"compound {idx}", fontsize=9)
        ax.invert_xaxis()
        ax.set_yticks([])
    for idx in range(n, rows * cols):  # hide any empty panels
        axes[idx // cols][idx % cols].axis("off")
    fig.suptitle("Compound library (fixed fingerprints)")
    fig.supxlabel("chemical shift (ppm)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_mixtures(gen: MixtureGenerator, n: int, out_path) -> None:
    fig, axes = plt.subplots(n, 2, figsize=(13, 2.6 * n), squeeze=False)
    for row in range(n):
        m = gen.generate(row)
        present = [j for j in range(gen.library.n_compounds) if m.present[j]]

        # Left: mixture + overlaid components + clean sum.
        ax = axes[row][0]
        ax.plot(m.grid, m.mixture, color="black", lw=1.0, label="mixture (observed)")
        ax.plot(m.grid, m.clean_mixture, color="tab:gray", lw=1.0, ls="--",
                label="clean sum")
        for j in present:
            ax.plot(m.grid, m.components[j], lw=0.9,
                    label=f"comp {j} (c={m.concentrations[j]:.2f})")
        ax.set_title(f"sample {row}: K={m.meta['k']}, compounds {present}", fontsize=9)
        ax.invert_xaxis()
        ax.legend(fontsize=6, ncol=2, loc="upper right")

        # Right: reconstruction residual.
        ax2 = axes[row][1]
        residual = m.mixture - m.clean_mixture
        ax2.plot(m.grid, residual, color="tab:red", lw=0.8, label="mixture - clean sum")
        ax2.plot(m.grid, m.baseline, color="tab:blue", lw=1.0, label="baseline (truth)")
        ax2.set_title("residual = baseline + noise", fontsize=9)
        ax2.invert_xaxis()
        ax2.legend(fontsize=6, loc="upper right")

    fig.supxlabel("chemical shift (ppm)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Checkpoint 1 mixture visualization.")
    parser.add_argument("--config", default="configs/data/default.yaml")
    parser.add_argument("--n", type=int, default=4, help="number of mixtures to show")
    args = parser.parse_args()

    cfg = DataConfig.from_yaml(args.config)
    seed_everything(cfg.base_seed)
    library = CompoundLibrary.from_config(cfg.library)  # freezes to disk on first run
    gen = MixtureGenerator(cfg, library)

    out_dir = PROJECT_ROOT / "experiments"
    out_dir.mkdir(exist_ok=True)
    lib_path = out_dir / "library.png"
    mix_path = out_dir / "mixtures.png"
    plot_library(library, gen.grid, lib_path)
    plot_mixtures(gen, args.n, mix_path)

    # Print labels + a numeric reconstruction check per sample.
    print(f"library: {library.n_compounds} compounds -> {lib_path}")
    print(f"mixtures: {args.n} samples -> {mix_path}\n")
    for i in range(args.n):
        m = gen.generate(i)
        present = [j for j in range(library.n_compounds) if m.present[j]]
        concs = {j: round(float(m.concentrations[j]), 3) for j in present}
        max_err = float(abs(m.mixture - (m.clean_mixture + m.baseline + m.noise)).max())
        print(f"sample {i}: K={m.meta['k']}  present={present}")
        print(f"           concentrations={concs}")
        print(f"           reconstruction max|error|={max_err:.2e}")


if __name__ == "__main__":
    main()
