"""Build the technical report PDF from report/paper.tex.

Copies the figures the paper cites out of experiments/ (which is gitignored, since it holds
regenerable run outputs) into report/figures/ (which is committed, so the paper stays
compilable from a clean clone without re-running hours of training), then compiles with
tectonic.

    python scripts/build_paper.py            # copy figures + compile -> report/paper.pdf
    python scripts/build_paper.py --figures-only

Requires tectonic (`brew install tectonic`). It is a single self-contained binary that
downloads only the LaTeX packages the document needs, rather than a multi-gigabyte TeX
distribution.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

from spectral.utils import PROJECT_ROOT

# Figures the paper includes, in the order they appear.
FIGURES = (
    "mixtures.png",
    "pretrain_reconstructions.png",
    "label_efficiency.png",
    "probe_presence.png",
    "robustness_finetune.png",
    "robustness_probe.png",
    "pretext_ablation.png",
)


def sync_figures() -> tuple[int, list[str]]:
    """Refresh report/figures/ from experiments/ where a fresh run exists.

    Returns (n_refreshed, missing_entirely). A figure absent from experiments/ is only a
    problem if it is also absent from report/figures/: on a clean clone experiments/ is
    empty (it is gitignored) and the committed figures are what the paper compiles against.
    """
    src_dir = PROJECT_ROOT / "experiments"
    dst_dir = PROJECT_ROOT / "report" / "figures"
    dst_dir.mkdir(parents=True, exist_ok=True)

    refreshed, missing = 0, []
    for name in FIGURES:
        src, dst = src_dir / name, dst_dir / name
        if src.exists():
            shutil.copy2(src, dst)
            refreshed += 1
        elif not dst.exists():
            missing.append(name)
    return refreshed, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the report PDF.")
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()

    refreshed, missing = sync_figures()
    if missing:
        # Fail loudly: a paper silently missing figures is worse than no paper.
        print(f"ERROR: figures missing from both experiments/ and report/figures/: "
              f"{', '.join(missing)}\n"
              f"Run the pipeline first:  python scripts/run_all.py --skip-existing", file=sys.stderr)
        raise SystemExit(1)
    print(f"figures: {refreshed} refreshed from experiments/, "
          f"{len(FIGURES) - refreshed} using the committed copy in report/figures/")
    if args.figures_only:
        return

    if shutil.which("tectonic") is None:
        print("ERROR: tectonic not found. Install it with:  brew install tectonic", file=sys.stderr)
        raise SystemExit(1)

    subprocess.run(["tectonic", "paper.tex"], cwd=PROJECT_ROOT / "report", check=True)
    print(f"\nbuilt -> {PROJECT_ROOT / 'report' / 'paper.pdf'}")


if __name__ == "__main__":
    main()
