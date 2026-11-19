"""Essential mathematical checks for the compact cached-feature experiment."""

import numpy as np
import pytest
from scipy.optimize import approx_fprime
from scipy.stats import wasserstein_distance
from sklearn.linear_model import Ridge

from src.study2.compact import (
    RowSpaceDesign,
    comparison_arms,
    empirical_w1_gradient,
    fit_linear_arm,
    partition_indices,
    task_weights,
)


def test_exact_unequal_sample_w1_and_gradient():
    scores = np.array([-1.4, -0.3, 0.9, -0.8, 0.4])
    groups = np.array([0, 0, 0, 1, 1])
    value, gradient = empirical_w1_gradient(scores, groups)
    assert value == pytest.approx(wasserstein_distance(scores[:3], scores[3:]))
    numerical = approx_fprime(scores, lambda x: empirical_w1_gradient(x, groups)[0], 1e-7)
    np.testing.assert_allclose(gradient, numerical, atol=2e-8)


def test_w1_tie_gradient_is_row_order_invariant():
    scores, groups = np.array([0.1, 0.1, 0.7, 0.3, 0.9]), np.array([0, 0, 0, 1, 1])
    value, gradient = empirical_w1_gradient(scores, groups)
    order = np.array([3, 1, 4, 0, 2])
    shuffled_value, shuffled_gradient = empirical_w1_gradient(scores[order], groups[order])
    assert value == pytest.approx(shuffled_value)
    np.testing.assert_allclose(gradient[order], shuffled_gradient)
    assert gradient[0] == gradient[1]


def test_task_only_rowspace_solution_matches_alpha_one_ridge():
    rng = np.random.default_rng(104)
    x, test = rng.normal(size=(18, 40)), rng.normal(size=(5, 40))
    y = rng.normal(size=18)
    design = RowSpaceDesign.fit(x)
    theta, receipt = fit_linear_arm(design, y, task_weights(None, len(y)), None, 0.0)
    reference = Ridge(alpha=1, solver="svd").fit(design.scaler.transform(x), y)
    np.testing.assert_allclose(
        design.transform(test) @ theta, reference.predict(design.scaler.transform(test)), atol=1e-10
    )
    assert receipt["optimizer_converged"]


def test_balanced_task_equivalence_preserves_explicit_regularisation():
    rng = np.random.default_rng(22)
    x, y = rng.normal(size=(15, 8)), rng.normal(size=15)
    groups = np.r_[np.zeros(5, dtype=int), np.ones(10, dtype=int)]
    design = RowSpaceDesign.fit(x)
    weights = task_weights(groups, len(y))
    theta, _ = fit_linear_arm(design, y, weights, groups, 0.0)
    # sklearn's summed weighted residuals require n*q to match alpha=1.
    reference = Ridge(alpha=1, solver="svd").fit(
        design.scaler.transform(x), y, sample_weight=len(y) * weights
    )
    np.testing.assert_allclose(
        design.train @ theta, reference.predict(design.scaler.transform(x)), atol=1e-10
    )


def test_participant_folds_never_split_sessions():
    rows = tuple(
        {"participant_id": str(p), "fold": p % 5, "phase": phase}
        for p in range(10)
        for phase in ("pre", "post")
    )
    roles = partition_indices(rows, 2)
    sets = [{rows[i]["participant_id"] for i in indices} for indices in roles.values()]
    assert all(not first & second for i, first in enumerate(sets) for second in sets[i + 1 :])
    corrupt = rows + ({"participant_id": "2", "fold": 1, "phase": "post"},)
    with pytest.raises(ValueError, match="different folds"):
        partition_indices(corrupt, 2)


def test_all_fifteen_arms_have_matching_zero_control():
    arms = comparison_arms()
    assert len(arms) == 15
    by_id = {arm["arm_id"]: arm for arm in arms}
    for arm in arms:
        control = by_id[arm["control_arm_id"]]
        assert control["lambda"] == 0
        assert control["exposure"] == arm["exposure"]
