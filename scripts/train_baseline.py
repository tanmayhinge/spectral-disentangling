"""Phase 2: train the from-scratch supervised baseline.

The simplest full loop: patch-transformer -> CLS pooling -> 12 logits, trained with binary
cross-entropy to predict which compounds are present in each mixture. Reports macro-F1 on a
held-out validation set and saves a learning curve. This baseline is what Phase-3
pretraining must later beat.

Run:  python scripts/train_baseline.py --config configs/train_baseline.yaml
"""

from __future__ import annotations

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from spectral.data.library import CompoundLibrary
from spectral.models.transformer import PresenceClassifier, count_parameters
from spectral.seeding import seed_everything
from spectral.training.config import ExperimentConfig
from spectral.training.data import build_presence_tensors
from spectral.training.metrics import presence_scores
from spectral.utils import PROJECT_ROOT, get_device


def resolve_device(name: str) -> torch.device:
    return get_device() if name == "auto" else torch.device(name)


@torch.no_grad()
def evaluate(model, x, y, criterion, device, threshold, batch_size):
    """Compute val loss + presence metrics over the whole val set (batched)."""
    model.eval()
    logits_all = []
    loss_sum, n = 0.0, 0
    for i in range(0, x.size(0), batch_size):
        xb = x[i : i + batch_size].to(device)
        yb = y[i : i + batch_size].to(device)
        logits = model(xb)
        loss_sum += criterion(logits, yb).item() * xb.size(0)
        n += xb.size(0)
        logits_all.append(logits.cpu())
    scores = presence_scores(torch.cat(logits_all), y, threshold)
    return loss_sum / n, scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the from-scratch presence baseline.")
    parser.add_argument("--config", default="configs/train_baseline.yaml")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    tcfg = cfg.train
    seed_everything(tcfg.seed)
    device = resolve_device(tcfg.device)

    # Data: fixed library, disjoint train / val mixture pools.
    library = CompoundLibrary.from_config(cfg.data.library)
    x_train, y_train = build_presence_tensors(cfg.data, library, cfg.data.base_seed, tcfg.n_train)
    x_val, y_val = build_presence_tensors(cfg.data, library, tcfg.val_seed, tcfg.n_val)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=tcfg.batch_size, shuffle=True)

    # Model / optim / loss.
    n_points = cfg.data.grid.n_points
    model = PresenceClassifier(cfg.model, n_points, n_classes=library.n_compounds).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    print(f"device={device}  params={count_parameters(model):,}  "
          f"tokens={model.encoder.n_patches + 1}  train={tcfg.n_train}  val={tcfg.n_val}")

    history = []
    for epoch in range(1, tcfg.epochs + 1):
        model.train()
        train_loss, seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            seen += xb.size(0)
        train_loss /= seen

        val_loss, s = evaluate(model, x_val, y_val, criterion, device, tcfg.threshold, tcfg.batch_size)
        history.append((epoch, train_loss, val_loss, s.macro_f1, s.micro_f1, s.exact_match))
        print(f"epoch {epoch:2d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"macro_f1={s.macro_f1:.3f}  micro_f1={s.micro_f1:.3f}  exact={s.exact_match:.3f}")

    _save_outputs(history, s)


def _save_outputs(history, final_scores) -> None:
    out = PROJECT_ROOT / "experiments"
    out.mkdir(exist_ok=True)

    csv_path = out / "baseline_log.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "macro_f1", "micro_f1", "exact_match"])
        w.writerows(history)

    epochs = [h[0] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, [h[1] for h in history], label="train loss")
    ax1.plot(epochs, [h[2] for h in history], label="val loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("BCE loss"); ax1.legend(); ax1.set_title("Loss")
    ax2.plot(epochs, [h[3] for h in history], label="macro-F1")
    ax2.plot(epochs, [h[4] for h in history], label="micro-F1")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("F1"); ax2.set_ylim(0, 1); ax2.legend()
    ax2.set_title("Validation presence F1")
    fig.suptitle("From-scratch presence baseline")
    fig.tight_layout()
    fig.savefig(out / "baseline_curve.png", dpi=120)
    plt.close(fig)

    print(f"\nsaved log -> {csv_path}")
    print(f"saved curve -> {out / 'baseline_curve.png'}")
    print("final per-compound F1: " + ", ".join(f"{v:.2f}" for v in final_scores.per_class_f1))


if __name__ == "__main__":
    main()
