"""Linear probes: train a single linear layer on frozen features to decode a property.

Three probes, each returning one number so encoders can be compared:
- presence:      macro-F1 of a 12-way multi-label linear classifier.
- count (K):     accuracy of decoding how many compounds are in the mixture (K in 2..5).
- concentration: mean absolute error of decoding the per-compound weights (present entries).

Features are standardized (train mean/std) before probing so scale differences between
encoders don't distort the linear fit.
"""

from __future__ import annotations

import torch
from torch import nn

from spectral.training.metrics import presence_scores

_K_MIN = 2  # component counts run 2..5 -> classes 0..3


def standardize(train: torch.Tensor, val: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu = train.mean(dim=0, keepdim=True)
    sd = train.std(dim=0, keepdim=True) + 1e-6
    return (train - mu) / sd, (val - mu) / sd


def _train_linear(feat: torch.Tensor, target: torch.Tensor, out_dim: int,
                  criterion, steps: int, lr: float, seed: int) -> nn.Linear:
    torch.manual_seed(seed)
    head = nn.Linear(feat.size(1), out_dim)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        criterion(head(feat), target).backward()
        opt.step()
    return head


def probe_presence(feat_tr, y_tr, feat_va, y_va, steps, lr, threshold, seed) -> float:
    """Multi-label linear classifier -> validation macro-F1."""
    head = _train_linear(feat_tr, y_tr, y_tr.size(1), nn.BCEWithLogitsLoss(), steps, lr, seed)
    with torch.no_grad():
        return presence_scores(head(feat_va), y_va, threshold).macro_f1


def probe_count(feat_tr, present_tr, feat_va, present_va, steps, lr, seed) -> float:
    """Decode component count K (2..5) -> validation accuracy."""
    k_tr = (present_tr.sum(dim=1).long() - _K_MIN).clamp(min=0)
    k_va = (present_va.sum(dim=1).long() - _K_MIN).clamp(min=0)
    n_classes = int(max(k_tr.max().item(), k_va.max().item())) + 1
    head = _train_linear(feat_tr, k_tr, n_classes, nn.CrossEntropyLoss(), steps, lr, seed)
    with torch.no_grad():
        pred = head(feat_va).argmax(dim=1)
        return float((pred == k_va).float().mean())


def probe_concentration(feat_tr, conc_tr, feat_va, conc_va, steps, lr, seed) -> float:
    """Regress per-compound concentrations -> MAE over the truly-present entries."""
    head = _train_linear(feat_tr, conc_tr, conc_tr.size(1), nn.MSELoss(), steps, lr, seed)
    with torch.no_grad():
        pred = head(feat_va)
        present = conc_va > 0
        return float((pred - conc_va).abs()[present].mean())
