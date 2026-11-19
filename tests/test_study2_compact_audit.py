"""Focused numerical checks for compact participant bootstrap audit statistics."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr, wasserstein_distance

from src.study2.compact_audit import (
    _axis_summary,
    _disparity_draws,
    _interval,
    bootstrap_utilities,
    empirical_w1_bootstrap,
    participant_bootstrap_weights,
)


def test_empirical_w1_matches_unequal_weighted_samples_and_counts_absence() -> None:
    prediction = np.array([0.1, 0.1, 0.7, 0.2, 0.3])
    categories = np.array(["a", "a", "a", "b", "b"])
    weights = np.array([[1, 2, 1, 2, 1], [2, 1, 3, 1, 4], [1, 1, 1, 0, 0]])
    actual = empirical_w1_bootstrap(prediction, categories, ("a", "b"), weights)
    for index in range(2):
        expected = wasserstein_distance(
            prediction[:3], prediction[3:], weights[index, :3], weights[index, 3:]
        )
        np.testing.assert_allclose(actual[index], expected, atol=1e-14)
    assert np.isnan(actual[2])
    interval = _interval(actual, float(actual[0]))
    assert interval["valid_draws"] == 2
    assert interval["undefined_draws"] == 1
    assert interval["interval_basis"] == "valid_draws_only"


def test_bootstrap_spearman_equals_explicit_repeated_rows_with_score_ties() -> None:
    target = np.array([0.1, 0.2, 0.2, 0.9])
    prediction = np.array([0.8, 0.8, 0.3, 0.4])
    weights = np.array([[2, 0, 1, 3], [1, 2, 1, 0], [0, 1, 1, 0]])
    actual = bootstrap_utilities(target, prediction, weights)
    for index, counts in enumerate(weights):
        expected_target = np.repeat(target, counts)
        expected_prediction = np.repeat(prediction, counts)
        expected_srcc = spearmanr(expected_target, expected_prediction).statistic
        np.testing.assert_allclose(actual["srcc"][index], expected_srcc, equal_nan=True)
        np.testing.assert_allclose(
            actual["mae"][index], np.abs(expected_prediction - expected_target).mean()
        )


def test_participant_bootstrap_carries_paired_sessions() -> None:
    weights = participant_bootstrap_weights(["1", "1", "2", "2", "3", "3"], replicates=17)
    np.testing.assert_array_equal(weights[:, 0::2], weights[:, 1::2])
    np.testing.assert_array_equal(weights[:, 0::2].sum(axis=1), np.full(17, 3))


def test_maximum_disparity_keeps_fixed_group_universe_on_missing_draw() -> None:
    scores = np.array([0.1, 0.2, 0.6, 0.9])
    labels = np.array(["1", "2", "3", "3"])
    weights = np.array([[1, 1, 1, 1], [1, 0, 1, 2]])
    actual = _disparity_draws(scores, labels, ["1", "2", "3"], weights)
    np.testing.assert_allclose(actual[0], 0.65)
    assert np.isnan(actual[1])


def test_cannot_determine_is_unknown_even_when_its_count_reaches_support() -> None:
    labels = ["a"] * 10 + ["b"] * 10 + ["cannot_determine"] * 10
    rows = [
        {"target": 0.5, "prediction": index / 30, "source_session_id": str(index)}
        for index in range(30)
    ]
    annotations = {
        str(index): {"perceived_gender_presentation": label} for index, label in enumerate(labels)
    }
    result, _, supported = _axis_summary(rows, annotations, "perceived_gender_presentation")
    assert supported == ["a", "b"]
    assert result["unknown_count"] == 10
    assert result["all_category_counts"]["cannot_determine"] == 10
