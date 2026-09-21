"""Model-independent strictly proper scoring rules for typed decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CompositeScoreConfig:
    log_weight: float = 1.0
    spherical_weight: float = 0.5
    ranked_probability_weight: float = 1.0


def composite_proper_score(
    probabilities: Any,
    targets: Any,
    option_mask: Any,
    ordinal_mask: Any,
    config: CompositeScoreConfig = CompositeScoreConfig(),
):
    """Return log + spherical - ordinal RPS for each probability report.

    `probabilities` may be shaped `[batch, option]` or
    `[group, batch, option]`. Other tensors are batch-shaped and broadcast over
    an optional leading group dimension.
    """

    import torch

    masked = (probabilities * option_mask).float()
    targets = targets.float()
    log_probabilities = masked.clamp_min(torch.finfo(masked.dtype).tiny).log()
    log_score = (targets * log_probabilities).sum(-1)
    spherical = (targets * masked).sum(-1) / masked.norm(dim=-1).clamp_min(1e-9)
    score = config.log_weight * log_score + config.spherical_weight * spherical

    if bool(ordinal_mask.any()):
        option_count = option_mask.sum(-1).clamp(min=2).to(masked.dtype)
        predicted_cdf = masked.cumsum(-1)
        target_cdf = targets.cumsum(-1)
        ranked_probability = (
            ((predicted_cdf - target_cdf) ** 2) * option_mask
        ).sum(-1) / (option_count - 1)
        score = score - config.ranked_probability_weight * ranked_probability * ordinal_mask
    return score
