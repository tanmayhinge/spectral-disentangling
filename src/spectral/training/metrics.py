"""Metrics for multi-label presence classification.

Locked in DECISIONS.md: presence is scored with macro-F1 (the unweighted mean of the
per-compound F1 scores), plus we report micro-F1 and exact-match for context. Macro-F1 is
robust to the class imbalance here (each compound is absent more often than present).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_EPS = 1e-8


@dataclass
class PresenceScores:
    macro_f1: float
    micro_f1: float
    exact_match: float          # fraction of samples with all 12 labels correct
    per_class_f1: list[float]


def presence_scores(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> PresenceScores:
    """Compute presence metrics from raw logits and {0,1} targets, both (N, n_classes)."""
    preds = (torch.sigmoid(logits) > threshold).float()

    tp = (preds * targets).sum(dim=0)
    fp = (preds * (1 - targets)).sum(dim=0)
    fn = ((1 - preds) * targets).sum(dim=0)

    per_class = 2 * tp / (2 * tp + fp + fn + _EPS)     # per-compound F1
    macro = per_class.mean()

    tp_t, fp_t, fn_t = tp.sum(), fp.sum(), fn.sum()    # pooled over all classes
    micro = 2 * tp_t / (2 * tp_t + fp_t + fn_t + _EPS)

    exact = (preds == targets).all(dim=1).float().mean()

    return PresenceScores(
        macro_f1=float(macro),
        micro_f1=float(micro),
        exact_match=float(exact),
        per_class_f1=[float(v) for v in per_class],
    )
