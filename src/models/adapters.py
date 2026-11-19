"""Projection components.

``LoRALinear`` is retained as a legacy utility but is not part of the
active Master's experiment.  The controlled study uses the same ordinary
trainable modality projections in every learned configuration so that its
fairness comparisons do not depend on an unresolved frozen base layer.
"""

from __future__ import annotations

import math
from numbers import Real

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Legacy frozen linear base plus a trainable rank-``r`` update.

    ``base`` is required and is registered by reference and frozen in place.
    Requiring it prevents the layer from silently creating and freezing a
    random projection while presenting that operation as adaptation of a
    pretrained weight.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        r: int = 8,
        alpha: float | None = None,
        base: nn.Linear | None = None,
    ) -> None:
        super().__init__()
        if isinstance(in_dim, bool) or not isinstance(in_dim, int) or in_dim <= 0:
            raise ValueError("in_dim must be a positive integer")
        if isinstance(out_dim, bool) or not isinstance(out_dim, int) or out_dim <= 0:
            raise ValueError("out_dim must be a positive integer")
        if isinstance(r, bool) or not isinstance(r, int) or r <= 0:
            raise ValueError("r must be a positive integer")
        if alpha is None:
            alpha = float(2 * r)
        elif isinstance(alpha, bool) or not isinstance(alpha, Real):
            raise TypeError("alpha must be a real scalar")
        alpha = float(alpha)
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")

        if base is None:
            raise ValueError(
                "base must be supplied explicitly; LoRA cannot silently adapt "
                "a newly randomised frozen projection"
            )
        if not isinstance(base, nn.Linear):
            raise TypeError("base must be torch.nn.Linear")
        if base.in_features != in_dim or base.out_features != out_dim:
            raise ValueError(
                "base dimensions do not match in_dim/out_dim: "
                f"got ({base.in_features}, {base.out_features})"
            )

        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.alpha = alpha
        self.A = nn.Parameter(self.base.weight.new_empty((r, in_dim)))
        nn.init.normal_(self.A, std=0.02)
        self.B = nn.Parameter(self.base.weight.new_zeros((out_dim, r)))
        self.scale = self.alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)
