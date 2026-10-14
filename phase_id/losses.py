from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    tau: float = 0.5,
    normalize: bool = False,
) -> torch.Tensor:
    """Symmetric InfoNCE with sim = -||a - p||^2 / tau.

    Row i of ``anchor`` and ``positive`` form the positive pair; every other
    row in the batch is a negative. ``normalize=True`` puts z on the unit
    sphere first (the vMF setting of Zimmermann et al. 2021); False is the
    Gaussian / unconstrained-R^d variant.
    """
    if normalize:
        anchor, positive = F.normalize(anchor, dim=-1), F.normalize(positive, dim=-1)
    logits = -torch.cdist(anchor, positive).pow(2) / tau  # (B, B)
    labels = torch.arange(len(anchor), device=anchor.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
