"""One fine-tuning run: train the full classifier on N labeled examples, return best val F1.

Shared by the Phase-4 sweep. The `init` argument selects the encoder starting point:
"scratch" (random) or "pretrained" (weights from Phase 3). Everything else -- head init,
batch order, optimizer, budget -- is held identical so the comparison is controlled.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from spectral.models.config import ModelConfig
from spectral.models.transformer import PresenceClassifier
from spectral.seeding import seed_everything
from spectral.training.metrics import presence_scores
from spectral.utils import PROJECT_ROOT


@torch.no_grad()
def _val_macro_f1(model, x, y, device, threshold, batch_size) -> float:
    model.eval()
    logits = [model(x[i : i + batch_size].to(device)).cpu() for i in range(0, x.size(0), batch_size)]
    return presence_scores(torch.cat(logits), y, threshold).macro_f1


def finetune_once(
    model_cfg: ModelConfig,
    n_points: int,
    n_classes: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    init: str,
    seed: int,
    max_steps: int,
    eval_every: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    threshold: float,
    pretrained_encoder: str,
    device: torch.device,
) -> float:
    """Fine-tune once and return the best validation macro-F1 seen during training."""
    # Reset all RNGs so scratch and pretrained runs get identical head init + batch order;
    # the encoder init is then the only variable.
    seed_everything(seed)
    model = PresenceClassifier(model_cfg, n_points, n_classes)
    if init == "pretrained":
        path = Path(pretrained_encoder)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        state = torch.load(path, map_location="cpu")
        model.encoder.load_state_dict(state, strict=True)
    elif init != "scratch":
        raise ValueError(f"init must be 'scratch' or 'pretrained', got {init!r}")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    bs = min(batch_size, x_train.size(0))
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=bs, shuffle=True, drop_last=False)

    best_f1, step = 0.0, 0
    while step < max_steps:
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
            step += 1
            if step % eval_every == 0 or step >= max_steps:
                best_f1 = max(best_f1, _val_macro_f1(model, x_val, y_val, device, threshold, batch_size))
            if step >= max_steps:
                break
    return best_f1
