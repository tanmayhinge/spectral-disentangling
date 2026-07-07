"""Phase 3: self-supervised masked pretraining.

Hide random spans of each mixture and train the model to reconstruct the hidden raw signal.
No labels are used. Saves the pretrained encoder weights (for Phase-4 fine-tuning), a loss
curve, and a reconstruction sanity figure.

Run:  python scripts/pretrain.py --config configs/pretrain.yaml
"""

from __future__ import annotations

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from spectral.data.library import CompoundLibrary
from spectral.models.masked import MaskedSpectralModel, make_batch_mask
from spectral.models.transformer import count_parameters
from spectral.seeding import seed_everything
from spectral.training.config import PretrainExperimentConfig
from spectral.training.data import build_mixture_tensor
from spectral.utils import PROJECT_ROOT, get_device


def resolve_device(name: str) -> torch.device:
    return get_device() if name == "auto" else torch.device(name)


@torch.no_grad()
def eval_recon_loss(model, x, pcfg, device, mask_rng) -> float:
    model.eval()
    total, n = 0.0, 0
    for i in range(0, x.size(0), pcfg.batch_size):
        xb = x[i : i + pcfg.batch_size].to(device)
        mask = make_batch_mask(xb.size(0), model.n_patches, pcfg.mask_ratio, pcfg.span_len, mask_rng).to(device)
        total += model.masked_mse(xb, mask).item() * xb.size(0)
        n += xb.size(0)
    return total / n


def main() -> None:
    parser = argparse.ArgumentParser(description="Masked self-supervised pretraining.")
    parser.add_argument("--config", default="configs/pretrain.yaml")
    args = parser.parse_args()

    cfg = PretrainExperimentConfig.from_yaml(args.config)
    pcfg = cfg.pretrain
    seed_everything(pcfg.seed)
    device = resolve_device(pcfg.device)
    mask_rng = np.random.default_rng(pcfg.seed)  # separate stream for masks

    # Unlabeled pretraining pool + a small held-out pool for the reconstruction check.
    library = CompoundLibrary.from_config(cfg.data.library)
    x_pre = build_mixture_tensor(cfg.data, library, pcfg.pretrain_seed, pcfg.n_pretrain)
    x_eval = build_mixture_tensor(cfg.data, library, pcfg.pretrain_seed + 1, 512)
    loader = DataLoader(TensorDataset(x_pre), batch_size=pcfg.batch_size, shuffle=True)

    model = MaskedSpectralModel(cfg.model, cfg.data.grid.n_points).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=pcfg.lr, weight_decay=pcfg.weight_decay)
    print(f"device={device}  params={count_parameters(model):,}  patches={model.n_patches}  "
          f"pretrain={pcfg.n_pretrain}  mask_ratio={pcfg.mask_ratio}  span_len={pcfg.span_len}")

    history = []
    for epoch in range(1, pcfg.epochs + 1):
        model.train()
        run, seen = 0.0, 0
        for (xb,) in loader:
            xb = xb.to(device)
            mask = make_batch_mask(xb.size(0), model.n_patches, pcfg.mask_ratio, pcfg.span_len, mask_rng).to(device)
            optimizer.zero_grad()
            loss = model.masked_mse(xb, mask)
            loss.backward()
            optimizer.step()
            run += loss.item() * xb.size(0)
            seen += xb.size(0)
        train_mse = run / seen
        eval_mse = eval_recon_loss(model, x_eval, pcfg, device, np.random.default_rng(123))
        history.append((epoch, train_mse, eval_mse))
        print(f"epoch {epoch:2d}  train_mse={train_mse:.5f}  eval_mse={eval_mse:.5f}")

    _save_encoder(model, pcfg)
    _save_curve(history)
    _save_reconstructions(model, x_eval, cfg, pcfg, device)


def _save_encoder(model, pcfg) -> None:
    path = PROJECT_ROOT / pcfg.encoder_out
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.encoder.state_dict(), path)
    print(f"\nsaved pretrained encoder -> {path}")


def _save_curve(history) -> None:
    out = PROJECT_ROOT / "experiments"
    with open(out / "pretrain_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_mse", "eval_mse"])
        w.writerows(history)
    epochs = [h[0] for h in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [h[1] for h in history], label="train")
    ax.plot(epochs, [h[2] for h in history], label="held-out")
    ax.set_xlabel("epoch"); ax.set_ylabel("masked reconstruction MSE")
    ax.set_title("Masked pretraining"); ax.legend()
    fig.tight_layout()
    fig.savefig(out / "pretrain_curve.png", dpi=120)
    plt.close(fig)
    print(f"saved curve -> {out / 'pretrain_curve.png'}")


@torch.no_grad()
def _save_reconstructions(model, x_eval, cfg, pcfg, device) -> None:
    """Plot observed input (with masked gaps), the model's fill-in, and the truth."""
    model.eval()
    n = pcfg.n_recon_examples
    ps = model.patch_size
    grid = np.linspace(cfg.data.grid.ppm_min, cfg.data.grid.ppm_max, cfg.data.grid.n_points)
    rng = np.random.default_rng(7)

    xb = x_eval[:n].to(device)
    mask = make_batch_mask(n, model.n_patches, pcfg.mask_ratio, pcfg.span_len, rng).to(device)
    recon = model(xb, mask).cpu().numpy()          # (n, P, ps)
    truth = model.to_patches(xb.cpu(), ps).numpy()  # (n, P, ps)
    mask_np = mask.cpu().numpy()

    fig, axes = plt.subplots(n, 1, figsize=(12, 2.4 * n), squeeze=False)
    for r in range(n):
        ax = axes[r][0]
        # Visible input: original where unmasked, NaN (gap) where masked.
        visible = truth[r].copy()
        visible[mask_np[r]] = np.nan
        ax.plot(grid, truth[r].reshape(-1), color="tab:gray", lw=0.8, label="truth (hidden)")
        ax.plot(grid, visible.reshape(-1), color="black", lw=0.9, label="model input (visible)")
        # Reconstruction shown only on masked patches.
        recon_masked = np.full_like(recon[r], np.nan)
        recon_masked[mask_np[r]] = recon[r][mask_np[r]]
        ax.plot(grid, recon_masked.reshape(-1), color="tab:red", lw=1.0, label="reconstruction")
        ax.set_title(f"example {r}: {int(mask_np[r].sum())}/{model.n_patches} patches masked", fontsize=9)
        ax.invert_xaxis()
        if r == 0:
            ax.legend(fontsize=7, loc="upper right")
    fig.supxlabel("chemical shift (ppm)")
    fig.suptitle("Masked reconstruction (red fills the hidden spans)")
    fig.tight_layout()
    out = PROJECT_ROOT / "experiments" / "pretrain_reconstructions.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved reconstructions -> {out}")


if __name__ == "__main__":
    main()
