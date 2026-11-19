"""Fairness and accuracy metrics.

Metric definitions are described in ``docs/research-overview.md``. Public
functions reject malformed inputs rather than returning ``nan``. The strict
Adverse Impact primitive rejects an ambiguous score tie; the study-level
sensitivity wrapper applies and records the predeclared clip-ID hash rule.
"""

from __future__ import annotations

import math
from hashlib import sha256

import numpy as np
from numpy.typing import ArrayLike

AI_Q_VALUES = (0.05, 0.10, 0.20)
AI_TIE_SALT = "fair-avi-ai-tie-v1"


def empirical_wasserstein1(sample0: ArrayLike, sample1: ArrayLike) -> float:
    """Return exact empirical one-dimensional W1 for non-empty samples.

    The samples may have different sizes. The implementation integrates the
    absolute difference between their empirical CDFs over the pooled support;
    it does not interpolate a threshold grid. This is the audit estimator,
    distinct from the balanced differentiable training-batch loss.
    """
    sample0 = _as_finite_vector("sample0", sample0)
    sample1 = _as_finite_vector("sample1", sample1)
    support = np.sort(np.concatenate((sample0, sample1)))
    widths = np.diff(support)
    if not widths.size:
        return 0.0
    cdf0 = np.searchsorted(np.sort(sample0), support[:-1], side="right") / sample0.size
    cdf1 = np.searchsorted(np.sort(sample1), support[:-1], side="right") / sample1.size
    return float(np.sum(np.abs(cdf0 - cdf1) * widths))


def spdd(pred: ArrayLike, attr: ArrayLike) -> float:
    """Return exact empirical SPDD for two non-empty binary audit groups."""
    pred, attr = _validated_scores_and_groups(pred, attr)
    return empirical_wasserstein1(pred[attr == 0], pred[attr == 1])


def speo(
    pred: ArrayLike,
    attr: ArrayLike,
    y_true: ArrayLike,
) -> float:
    """Return the binary-outcome special case of SPEO.

    Kim et al. (2023), equations 27--28, define SPEO for a continuous outcome
    by averaging the equalized-opportunity gap over *both* a prediction-score
    threshold and a ground-truth threshold.  A single positive-class cutoff is
    therefore not that published metric.  Until the finite-sample policy for
    ground-truth thresholds with an empty group is specified, this function
    fails explicitly for continuous outcomes rather than returning a
    mislabeled statistic.

    For outcomes encoded exactly as 0/1, conditioning on ``y_true == 1`` is the
    valid binary special case.  The conditioned subset must contain both
    groups.
    """
    pred, attr = _validated_scores_and_groups(pred, attr)
    y_true = _as_finite_vector("y_true", y_true)
    if y_true.shape != pred.shape:
        raise ValueError(f"y_true must have shape {pred.shape}; got {y_true.shape}")

    values = set(np.unique(y_true).tolist())
    if not values.issubset({0, 1}):
        raise NotImplementedError(
            "continuous-target SPEO requires integration over both prediction "
            "and ground-truth thresholds (Kim et al., 2023, eqs. 27--28); "
            "the finite-sample empty-cell policy is not yet specified"
        )
    pos = y_true == 1

    if not pos.any():
        raise ValueError("the positive-outcome subset is empty")
    return spdd(pred[pos], attr[pos])


def adverse_impact(pred: ArrayLike, attr: ArrayLike, k: float = 0.10) -> float:
    """Return the Adverse-Impact ratio for an unambiguous exact top-k set.

    The selection size is ``ceil(k * n)``.  A tie crossing the selection
    boundary has no unique exact top-k set, so this function raises instead
    of selecting extra clips via ``score >= quantile`` or breaking the
    tie by input order.  A tie entirely inside or outside the boundary is
    harmless.
    """
    pred, attr = _validated_scores_and_groups(pred, attr)
    if isinstance(k, (bool, np.bool_)):
        raise TypeError("k must be a real scalar")
    try:
        k = float(k)
    except (TypeError, ValueError) as exc:
        raise TypeError("k must be a real scalar") from exc
    if not math.isfinite(k) or not 0 < k <= 1:
        raise ValueError("k must be finite and in (0, 1]")

    n_select = math.ceil(k * pred.size)
    order = np.argsort(-pred, kind="stable")
    cutoff = pred[order[n_select - 1]]
    above = pred > cutoff
    tied = pred == cutoff
    remaining = n_select - int(above.sum())
    if remaining < int(tied.sum()):
        raise ValueError(
            "score tie crosses the exact top-k boundary; specify a tie-breaking "
            "policy outside this metric before computing adverse impact"
        )
    selected = above | tied
    if int(selected.sum()) != n_select:  # defensive guard for future changes
        raise RuntimeError("internal error: selection set is not exact top-k")

    rates = [float(selected[attr == group].mean()) for group in (0, 1)]
    return min(rates) / max(rates)


def adverse_impact_sensitivity(
    pred: ArrayLike,
    attr: ArrayLike,
    clip_ids: ArrayLike,
    q_values: tuple[float, ...] = AI_Q_VALUES,
    tie_salt: str = AI_TIE_SALT,
) -> list[dict[str, float | int | str]]:
    """Return the predeclared simulated shortlist-sensitivity report.

    Each shortlist contains exactly ``ceil(q * n)`` clips. Scores are ordered
    descending; a score tie that crosses the boundary is resolved by ordering
    the tied clips by ``SHA-256(tie_salt + NUL + clip_id)``. The fixed salt and
    clip IDs are independent of outcomes and protected annotations. Returned
    rows record the salt and number of clips in a boundary-crossing tie so the
    selection is auditable.

    This is a simulated selection diagnostic, not an observed hiring outcome
    or legal conclusion. The strict :func:`adverse_impact` primitive remains
    available for callers that require ambiguous boundary ties to raise.
    """
    pred, attr = _validated_scores_and_groups(pred, attr)
    clip_ids = _validated_clip_ids(clip_ids, pred.shape)
    if not isinstance(tie_salt, str) or not tie_salt:
        raise ValueError("tie_salt must be a non-empty string")
    if not isinstance(q_values, tuple) or not q_values:
        raise ValueError("q_values must be a non-empty tuple")

    reports: list[dict[str, float | int | str]] = []
    for q in q_values:
        q = _validated_fraction(q, "q")
        n_select = math.ceil(q * pred.size)
        tie_keys = np.array(
            [sha256(f"{tie_salt}\0{clip_id}".encode()).hexdigest() for clip_id in clip_ids]
        )
        order = np.lexsort((tie_keys, -pred))
        selected = np.zeros(pred.size, dtype=bool)
        selected[order[:n_select]] = True

        cutoff = pred[order[n_select - 1]]
        at_cutoff = pred == cutoff
        selected_at_cutoff = int((selected & at_cutoff).sum())
        cutoff_count = int(at_cutoff.sum())
        boundary_tie_count = cutoff_count if selected_at_cutoff < cutoff_count else 0

        rates = [float(selected[attr == group].mean()) for group in (0, 1)]
        reports.append(
            {
                "q": q,
                "adverse_impact": min(rates) / max(rates),
                "n_observations": int(pred.size),
                "n_selected": n_select,
                "boundary_tie_count": boundary_tie_count,
                "tie_break_salt": tie_salt,
            }
        )
    return reports


def audit_probe_auroc(locked_probability: ArrayLike, annotated_gender: ArrayLike) -> float:
    """Return AUROC for locked Stage-B probe probabilities at the Stage-C join.

    This metric never fits or cross-validates a probe. The segregated runner must
    implement the fixed train-only pipeline declared in ``configs/default.yaml``
    and supply its already-locked test probabilities here.
    """
    score, group = _validated_scores_and_groups(locked_probability, annotated_gender)
    order = np.argsort(score, kind="stable")
    ranks = np.empty(score.size, dtype=float)
    start = 0
    while start < score.size:
        stop = start + 1
        while stop < score.size and score[order[stop]] == score[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2
        start = stop
    n_pos = int(group.sum())
    n_neg = group.size - n_pos
    return float((ranks[group == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _validated_scores_and_groups(pred: ArrayLike, attr: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    pred = _as_finite_vector("pred", pred)
    attr = _as_finite_vector("attr", attr)
    if attr.shape != pred.shape:
        raise ValueError(f"attr must have shape {pred.shape}; got {attr.shape}")
    if np.any((pred < 0) | (pred > 1)):
        raise ValueError("pred must be in [0, 1]")
    values = set(np.unique(attr).tolist())
    if values != {0, 1}:
        raise ValueError(
            f"attr must contain both binary groups encoded as 0 and 1; received {sorted(values)}"
        )
    return pred, attr


def _as_finite_vector(name: str, values: ArrayLike) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
        raise TypeError(f"{name} must contain numeric values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validated_fraction(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real scalar") from exc
    if not math.isfinite(result) or not 0 < result <= 1:
        raise ValueError(f"{name} must be finite and in (0, 1]")
    return result


def _validated_clip_ids(clip_ids: ArrayLike, shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(clip_ids, dtype=object)
    if result.ndim != 1 or result.shape != shape:
        raise ValueError(f"clip_ids must have shape {shape}; got {result.shape}")
    if any(not isinstance(value, str) or not value for value in result.tolist()):
        raise ValueError("clip_ids must contain non-empty strings")
    if len(set(result.tolist())) != result.size:
        raise ValueError("clip_ids must be unique")
    return result
