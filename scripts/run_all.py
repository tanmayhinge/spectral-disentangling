"""Run the whole project end to end, in phase order, and regenerate every report figure.

This is the "one command" promised by the working agreement: from a clean clone, it takes
you from nothing to every figure and CSV the report cites. Each stage is an ordinary script
with its own config -- nothing happens here that you cannot run by hand.

    python scripts/run_all.py                 # everything (hours, mostly pretraining)
    python scripts/run_all.py --list          # show the stages and their outputs
    python scripts/run_all.py --only baseline probes
    python scripts/run_all.py --skip-existing # skip stages whose outputs are already there

Stage order matters: pretraining must precede fine-tuning and probing, which load the
encoder it writes. The robustness sweep is self-contained (it pretrains per difficulty), and
the pretext ablation reuses the encoders the sweep caches, so run the sweep first.

Runtimes below are from an M-series Mac (MPS). A CUDA GPU is faster; CPU is much slower.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass

from spectral.utils import PROJECT_ROOT


@dataclass(frozen=True)
class Stage:
    name: str
    script: str
    config: str | None
    outputs: tuple[str, ...]   # relative to experiments/; used by --skip-existing
    minutes: str
    blurb: str


STAGES: tuple[Stage, ...] = (
    Stage("sanity", "scripts/hello_gpu.py", None, (), "<1",
          "Phase 0: print torch version and the compute device."),
    Stage("data", "scripts/visualize_mixtures.py", "configs/data/default.yaml",
          ("library.png", "mixtures.png"), "<1",
          "Phase 1: plot the compound library and mixtures against their components."),
    Stage("baseline", "scripts/train_baseline.py", "configs/train_baseline.yaml",
          ("baseline_curve.png", "baseline_log.csv"), "~5",
          "Phase 2: from-scratch presence classifier + learning curve."),
    Stage("pretrain", "scripts/pretrain.py", "configs/pretrain.yaml",
          ("pretrained_encoder.pt", "pretrain_curve.png", "pretrain_reconstructions.png"), "~7",
          "Phase 3: masked self-supervised pretraining (writes the encoder others load)."),
    Stage("finetune", "scripts/finetune_sweep.py", "configs/finetune_sweep.yaml",
          ("label_efficiency.csv", "label_efficiency.png"), "~25",
          "Phase 4: the headline pretrained-vs-scratch label-efficiency sweep."),
    Stage("probes", "scripts/probe.py", "configs/probe.yaml",
          ("probe_presence.csv", "probe_presence.png"), "~10",
          "Phase 5: linear probes on frozen features."),
    Stage("robustness", "scripts/robustness_sweep.py", "configs/robustness.yaml",
          ("robustness_finetune.csv", "robustness_probe.csv",
           "robustness_finetune.png", "robustness_probe.png"), "~90",
          "Phase 6: difficulty sweep; pretrains a fresh encoder per difficulty."),
    Stage("ablation", "scripts/pretext_ablation.py", "configs/pretext_ablation.yaml",
          ("pretext_ablation.csv", "pretext_ablation.png"), "~30",
          "Phase 6 follow-up: raw vs clean pretext target (reuses the sweep's encoders)."),
)


def outputs_exist(stage: Stage) -> bool:
    exp = PROJECT_ROOT / "experiments"
    return bool(stage.outputs) and all((exp / o).exists() for o in stage.outputs)


def run(stage: Stage) -> float:
    cmd = [sys.executable, "-u", stage.script]
    if stage.config:
        cmd += ["--config", stage.config]
    print(f"\n{'=' * 78}\n>>> {stage.name}  (~{stage.minutes} min)  {stage.blurb}\n"
          f">>> {' '.join(cmd)}\n{'=' * 78}", flush=True)

    start = time.time()
    # check=True: a failing stage must stop the run rather than let later stages read
    # stale or missing artifacts and silently produce a wrong report.
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    return (time.time() - start) / 60


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full pipeline in phase order.")
    parser.add_argument("--list", action="store_true", help="show stages and exit")
    parser.add_argument("--only", nargs="+", metavar="STAGE", help="run just these stages")
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip stages whose outputs already exist")
    args = parser.parse_args()

    if args.list:
        print(f"{'stage':<12} {'~min':>5}  outputs")
        for s in STAGES:
            print(f"{s.name:<12} {s.minutes:>5}  {', '.join(s.outputs) or '(prints only)'}")
        return

    stages = STAGES
    if args.only:
        known = {s.name for s in STAGES}
        unknown = set(args.only) - known
        if unknown:
            parser.error(f"unknown stage(s): {', '.join(sorted(unknown))}. known: {', '.join(sorted(known))}")
        stages = tuple(s for s in STAGES if s.name in set(args.only))

    timings = []
    for stage in stages:
        if args.skip_existing and outputs_exist(stage):
            print(f"\n--- {stage.name}: outputs present, skipping")
            continue
        timings.append((stage.name, run(stage)))

    print(f"\n{'=' * 78}\nfinished")
    for name, mins in timings:
        print(f"  {name:<12} {mins:6.1f} min")
    print(f"  {'total':<12} {sum(m for _n, m in timings):6.1f} min")


if __name__ == "__main__":
    main()
