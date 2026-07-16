"""The masked-pretraining loop (Phase 3), as a reusable function.

Lives here rather than in the script because two callers need it: `scripts/pretrain.py`
(the Phase-3 run that produces the headline encoder) and the Phase-6 robustness sweep,
which pretrains a fresh encoder at each difficulty level.

No labels are used anywhere in this file -- that is the whole point of the pretext task.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from spectral.models.config import ModelConfig
from spectral.models.masked import MaskedSpectralModel, make_batch_mask
from spectral.seeding import seed_everything

# (epoch, train_mse, eval_mse) per epoch.
History = list[tuple[int, float, float]]

_EVAL_MASK_SEED = 123  # fixed so the held-out masks are identical across epochs and runs


@torch.no_grad()
def eval_recon_loss(
    model: MaskedSpectralModel, x: torch.Tensor, *, mask_ratio: float, span_len: int,
    batch_size: int, device: torch.device, mask_rng: np.random.Generator,
    target: torch.Tensor | None = None,
) -> float:
    """Mean masked-reconstruction MSE over a held-out pool.

    `target` defaults to `x` (predict the observed signal). Note that losses computed against
    different targets are NOT comparable to each other -- a clean target is intrinsically
    easier to hit than a noisy one.
    """
    model.eval()
    total, n = 0.0, 0
    for i in range(0, x.size(0), batch_size):
        xb = x[i : i + batch_size].to(device)
        tb = None if target is None else target[i : i + batch_size].to(device)
        mask = make_batch_mask(xb.size(0), model.n_patches, mask_ratio, span_len, mask_rng).to(device)
        total += model.masked_mse(xb, mask, tb).item() * xb.size(0)
        n += xb.size(0)
    return total / n


def pretrain_encoder(
    model_cfg: ModelConfig,
    n_points: int,
    x_pretrain: torch.Tensor,
    x_eval: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    mask_ratio: float,
    span_len: int,
    seed: int,
    device: torch.device,
    log_fn: Callable[[str], None] | None = None,
    x_pretrain_target: torch.Tensor | None = None,
    x_eval_target: torch.Tensor | None = None,
) -> tuple[MaskedSpectralModel, History]:
    """Train a MaskedSpectralModel to fill in hidden spans. Returns (model, history).

    `x_pretrain` and `x_eval` are unlabeled mixtures (N, n_points). The mask stream gets its
    own RNG seeded from `seed`, so masking is reproducible independently of model init.

    The optional `*_target` tensors override what the model reconstructs (default: the input
    itself). Supplying the generator's clean signal makes this a denoising pretext -- see the
    fairness caveat on `build_mixture_and_clean_tensors`.
    """
    seed_everything(seed)
    mask_rng = np.random.default_rng(seed)

    model = MaskedSpectralModel(model_cfg, n_points).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Pair each input with its target so shuffling keeps them aligned.
    targets = x_pretrain if x_pretrain_target is None else x_pretrain_target
    loader = DataLoader(TensorDataset(x_pretrain, targets), batch_size=batch_size, shuffle=True)

    history: History = []
    for epoch in range(1, epochs + 1):
        model.train()
        run, seen = 0.0, 0
        for xb, tb in loader:
            xb, tb = xb.to(device), tb.to(device)
            mask = make_batch_mask(xb.size(0), model.n_patches, mask_ratio, span_len, mask_rng).to(device)
            optimizer.zero_grad()
            loss = model.masked_mse(xb, mask, tb)
            loss.backward()
            optimizer.step()
            run += loss.item() * xb.size(0)
            seen += xb.size(0)

        train_mse = run / seen
        eval_mse = eval_recon_loss(
            model, x_eval, mask_ratio=mask_ratio, span_len=span_len,
            batch_size=batch_size, device=device, mask_rng=np.random.default_rng(_EVAL_MASK_SEED),
            target=x_eval_target,
        )
        history.append((epoch, train_mse, eval_mse))
        if log_fn is not None:
            log_fn(f"epoch {epoch:2d}  train_mse={train_mse:.5f}  eval_mse={eval_mse:.5f}")
    return model, history
