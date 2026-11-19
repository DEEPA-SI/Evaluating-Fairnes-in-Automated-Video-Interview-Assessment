"""Prediction heads and the legacy gradient-reversal utility."""

from __future__ import annotations

import math
from numbers import Real

import torch
import torch.nn as nn
from torch.autograd import Function


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x, strength):
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.strength * grad, None


def grad_reverse(x: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """Apply the Gradient Reversal Layer with a finite non-negative strength."""
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if isinstance(strength, bool) or not isinstance(strength, Real):
        raise TypeError("strength must be a real scalar")
    strength = float(strength)
    if not math.isfinite(strength) or strength < 0:
        raise ValueError("strength must be finite and non-negative")
    return _GradReverse.apply(x, strength)


class RegressionHead(nn.Module):
    """Map a fused representation to the single [0, 1] primary score."""

    def __init__(self, dim: int = 512, hidden: int = 256, output_dim: int = 1) -> None:
        super().__init__()
        if output_dim != 1:
            raise ValueError("the governing study requires exactly one primary output")
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class AdversarialHead(nn.Module):
    """Predict the binary pseudo-attribute behind a unit-strength GRL.

    The adversarial objective coefficient is applied once by
    ``src.losses.fairness.total_loss``.  Keeping the GRL strength at one here
    avoids accidentally applying that coefficient twice to the encoder.
    """

    def __init__(self, dim: int = 512, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(grad_reverse(z)).squeeze(-1)
