"""Validation and edge-case tests for fairness metrics."""

import numpy as np
import pytest

from src.eval.metrics import (
    AI_Q_VALUES,
    AI_TIE_SALT,
    adverse_impact,
    adverse_impact_sensitivity,
    audit_probe_auroc,
    empirical_wasserstein1,
    spdd,
    speo,
)


def test_spdd_is_zero_for_identical_group_distributions():
    pred = np.array([0.1, 0.8, 0.1, 0.8])
    attr = np.array([0, 0, 1, 1])
    assert spdd(pred, attr) == pytest.approx(0.0)


def test_exact_empirical_w1_supports_unequal_sample_sizes():
    assert empirical_wasserstein1(
        np.array([0.0, 1.0]),
        np.array([0.5]),
    ) == pytest.approx(0.5)


def test_spdd_is_exact_and_supports_unequal_audit_groups():
    # Group 0 has point masses at 0 and 1; group 1 has one mass at 0.5.
    # The exact integral of the empirical CDF gap is 0.5.
    assert spdd(
        np.array([0.0, 1.0, 0.5]),
        np.array([0, 0, 1]),
    ) == pytest.approx(0.5)


@pytest.mark.parametrize("attr", [np.array([0, 0, 0]), np.array([0, 1, 2])])
def test_spdd_rejects_missing_or_multiclass_groups(attr):
    with pytest.raises(ValueError, match="both binary groups"):
        spdd(np.array([0.1, 0.2, 0.3]), attr)


def test_speo_marks_continuous_outcome_estimator_as_unimplemented():
    with pytest.raises(NotImplementedError, match="both prediction and ground-truth"):
        speo(
            np.array([0.1, 0.9, 0.2, 0.8]),
            np.array([0, 0, 1, 1]),
            np.array([0.2, 0.8, 0.9, 0.1]),
        )


def test_speo_supports_the_binary_outcome_special_case():
    result = speo(
        np.array([0.1, 0.9, 0.2, 0.8]),
        np.array([0, 0, 1, 1]),
        np.array([1, 1, 0, 1]),
    )
    assert result == pytest.approx(0.4)


def test_speo_rejects_a_positive_subset_missing_one_group():
    with pytest.raises(ValueError, match="both binary groups"):
        speo(
            np.array([0.9, 0.8, 0.2, 0.1]),
            np.array([0, 0, 1, 1]),
            np.array([1, 1, 0, 0]),
        )


def test_adverse_impact_uses_exact_selection_count_without_boundary_ties():
    pred = np.array([1.0, 0.9, 0.7, 0.5, 0.3, 0.8, 0.6, 0.4, 0.2, 0.1])
    attr = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    # Exactly two are selected, and both are from group 0.
    assert adverse_impact(pred, attr, k=0.2) == pytest.approx(0.0)


def test_adverse_impact_rejects_an_ambiguous_boundary_tie():
    with pytest.raises(ValueError, match="tie crosses"):
        adverse_impact(
            np.array([0.9, 0.8, 0.8, 0.1]),
            np.array([0, 0, 1, 1]),
            k=0.5,
        )


def test_adverse_impact_allows_a_tie_fully_inside_selection_set():
    assert adverse_impact(
        np.array([0.9, 0.9, 0.2, 0.1]),
        np.array([0, 1, 0, 1]),
        k=0.5,
    ) == pytest.approx(1.0)


def test_ai_sensitivity_uses_predeclared_fractions_and_exact_counts():
    pred = np.linspace(0.0, 1.0, 20)
    attr = np.tile([0, 1], 10)
    clip_ids = np.array([f"clip-{index:02d}" for index in range(20)])

    rows = adverse_impact_sensitivity(pred, attr, clip_ids)

    assert tuple(row["q"] for row in rows) == AI_Q_VALUES
    assert [row["n_selected"] for row in rows] == [1, 2, 4]
    assert all(row["tie_break_salt"] == AI_TIE_SALT for row in rows)


def test_ai_sensitivity_breaks_boundary_ties_stably_under_input_permutation():
    pred = np.full(20, 0.5)
    attr = np.tile([0, 1], 10)
    clip_ids = np.array([f"clip-{index:02d}" for index in range(20)])
    permutation = np.array([7, 2, 19, 0, 11, 5, 3, 17, 13, 1, 18, 8, 10, 4, 15, 6, 9, 16, 12, 14])

    original = adverse_impact_sensitivity(pred, attr, clip_ids)
    permuted = adverse_impact_sensitivity(
        pred[permutation],
        attr[permutation],
        clip_ids[permutation],
    )

    assert original == permuted
    assert [row["boundary_tie_count"] for row in original] == [20, 20, 20]


def test_ai_sensitivity_rejects_duplicate_clip_ids():
    with pytest.raises(ValueError, match="unique"):
        adverse_impact_sensitivity(
            np.array([0.9, 0.8, 0.7, 0.6]),
            np.array([0, 0, 1, 1]),
            np.array(["a", "b", "c", "c"]),
        )


def test_audit_probe_auroc_uses_locked_probabilities_without_fitting():
    assert audit_probe_auroc(
        np.array([0.1, 0.4, 0.4, 0.9]),
        np.array([0, 0, 1, 1]),
    ) == pytest.approx(0.875)
