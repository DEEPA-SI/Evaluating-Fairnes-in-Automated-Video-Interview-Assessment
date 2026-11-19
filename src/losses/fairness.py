"""Loss terms used by the fairness objective.

The terms are described in ``docs/research-overview.md``. This training
module deliberately accepts only the single primary score and a *balanced
binary* pseudo-group assignment.  Audit-time empirical W1 for unequal
protected groups is implemented separately in ``src.eval.metrics``.
"""

from __future__ import annotations

import math
from numbers import Real

import torch
import torch.nn.functional as F


def task_loss(pred: torch.Tensor, target: torch.Tensor, kind: str = "mse") -> torch.Tensor:
    """Return the configured prediction loss after validating its inputs."""
    _require_matching_finite_tensors(pred, target, "pred", "target")
    if kind == "mse":
        return F.mse_loss(pred, target)
    if kind == "bce":
        if not torch.all((pred >= 0) & (pred <= 1)):
            raise ValueError("pred must be in [0, 1] for BCE")
        if not torch.all((target >= 0) & (target <= 1)):
            raise ValueError("target must be in [0, 1] for BCE")
        return F.binary_cross_entropy(pred, target)
    raise ValueError(f"unsupported task loss: {kind!r}")


def wasserstein1(s0: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
    """Return W1 for two non-empty, equal-size one-dimensional samples."""
    _require_matching_finite_tensors(s0, s1, "s0", "s1")
    if s0.ndim != 1:
        raise ValueError(f"s0 and s1 must be one-dimensional; got shape {tuple(s0.shape)}")
    return (s0.sort().values - s1.sort().values).abs().mean()


def wasserstein_fairness(pred: torch.Tensor, pseudo_attr: torch.Tensor) -> torch.Tensor:
    """Return primary-score W1 over an exactly balanced binary-group batch.

    ``pred`` has shape ``(batch, 1)`` and ``pseudo_attr`` has shape
    ``(batch,)``.  Unequal, missing, invalid, or multiclass groups raise an
    error.  This enforces the balanced-sampler precondition instead of using
    the old order-dependent truncation.
    """
    _require_prediction_matrix(pred)
    if pred.shape[1] != 1:
        raise ValueError("W1 training requires exactly one primary prediction column")
    _require_binary_groups(pseudo_attr, pred.shape[0])
    if pseudo_attr.device != pred.device:
        raise ValueError("pred and pseudo_attr must be on the same device")

    g0, g1 = pseudo_attr == 0, pseudo_attr == 1
    n0, n1 = int(g0.sum().item()), int(g1.sum().item())
    if n0 != n1:
        raise ValueError(
            f"wasserstein_fairness requires a balanced batch; received group counts 0:{n0}, 1:{n1}"
        )

    return wasserstein1(pred[g0, 0], pred[g1, 0])


def adversarial_loss(logit: torch.Tensor, pseudo_attr: torch.Tensor) -> torch.Tensor:
    """Return binary adversarial BCE for a batch containing both groups."""
    if not isinstance(logit, torch.Tensor):
        raise TypeError("logit must be a torch.Tensor")
    if logit.ndim != 1:
        raise ValueError(f"logit must have shape (batch,); got {tuple(logit.shape)}")
    if logit.numel() == 0 or not torch.isfinite(logit).all():
        raise ValueError("logit must be non-empty and finite")
    _require_binary_groups(pseudo_attr, logit.shape[0])
    if pseudo_attr.device != logit.device:
        raise ValueError("logit and pseudo_attr must be on the same device")
    return F.binary_cross_entropy_with_logits(logit, pseudo_attr.float())


def total_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    adv_logit: torch.Tensor | None,
    pseudo_attr: torch.Tensor | None,
    lambda_w: float,
    lambda_a: float,
    task_kind: str = "mse",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the weighted objective and its three unweighted components.

    ``lambda_a`` is applied here exactly once.  ``AdversarialHead`` uses a
    unit-strength gradient reversal, so both head and representation receive
    a gradient whose magnitude is weighted by this coefficient (with opposite
    signs for the shared representation).

    A zero coefficient disables its component completely.  In particular, a
    task-only ERM call may pass ``None`` for both ``adv_logit`` and
    ``pseudo_attr``: the control must not inspect a pseudo-group or require a
    group-balanced batch.  A W1-only call needs ``pseudo_attr`` but may pass
    ``None`` for ``adv_logit``.
    """
    lambda_w = _require_nonnegative_weight(lambda_w, "lambda_w")
    lambda_a = _require_nonnegative_weight(lambda_a, "lambda_a")
    lt = task_loss(pred, target, task_kind)

    lw = pred.new_zeros(())
    if lambda_w > 0:
        if pseudo_attr is None:
            raise ValueError("pseudo_attr is required when lambda_w is positive")
        lw = wasserstein_fairness(pred, pseudo_attr)

    la = pred.new_zeros(())
    if lambda_a > 0:
        if adv_logit is None:
            raise ValueError("adv_logit is required when lambda_a is positive")
        if pseudo_attr is None:
            raise ValueError("pseudo_attr is required when lambda_a is positive")
        la = adversarial_loss(adv_logit, pseudo_attr)

    return lt + lambda_w * lw + lambda_a * la, {"task": lt, "w1": lw, "adv": la}


def _require_matching_finite_tensors(
    left: torch.Tensor,
    right: torch.Tensor,
    left_name: str,
    right_name: str,
) -> None:
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        raise TypeError(f"{left_name} and {right_name} must be torch.Tensor instances")
    if left.shape != right.shape:
        raise ValueError(
            f"{left_name} and {right_name} must have equal shapes; "
            f"got {tuple(left.shape)} and {tuple(right.shape)}"
        )
    if left.numel() == 0:
        raise ValueError(f"{left_name} and {right_name} must be non-empty")
    if left.device != right.device:
        raise ValueError(f"{left_name} and {right_name} must be on the same device")
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError(f"{left_name} and {right_name} must contain only finite values")


def _require_prediction_matrix(pred: torch.Tensor) -> None:
    if not isinstance(pred, torch.Tensor):
        raise TypeError("pred must be a torch.Tensor")
    if pred.ndim != 2 or pred.shape[0] == 0 or pred.shape[1] == 0:
        raise ValueError(
            f"pred must be a non-empty matrix with shape (batch, outputs); got {tuple(pred.shape)}"
        )
    if not torch.isfinite(pred).all():
        raise ValueError("pred must contain only finite values")
    if not torch.all((pred >= 0) & (pred <= 1)):
        raise ValueError("pred must be in [0, 1]")


def _require_binary_groups(pseudo_attr: torch.Tensor, batch_size: int) -> None:
    if not isinstance(pseudo_attr, torch.Tensor):
        raise TypeError("pseudo_attr must be a torch.Tensor")
    if pseudo_attr.ndim != 1 or pseudo_attr.shape[0] != batch_size:
        raise ValueError(
            f"pseudo_attr must have shape ({batch_size},); got {tuple(pseudo_attr.shape)}"
        )
    if pseudo_attr.numel() == 0 or not torch.isfinite(pseudo_attr).all():
        raise ValueError("pseudo_attr must be non-empty and finite")
    values = set(pseudo_attr.detach().cpu().tolist())
    if values != {0, 1}:
        raise ValueError(
            "pseudo_attr must contain both binary groups encoded as 0 and 1; "
            f"received {sorted(values)}"
        )


def _require_nonnegative_weight(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result
