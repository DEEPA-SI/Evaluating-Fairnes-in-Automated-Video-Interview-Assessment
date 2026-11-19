"""Participant-level proxy contracts."""

from __future__ import annotations

import numpy as np
import pytest

from src.features.pseudo_attributes import (
    RANDOM_PROXY_SALT,
    BalancedParticipantSampler,
    estimate_audio_session_pseudo_attr,
    estimate_face_pseudo_attr,
    random_proxy_groups,
    validity_check,
)


def test_face_and_audio_session_proxies_fit_training_participants_only_and_are_stable():
    features = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    ids = np.array(["4", "3", "2", "1"])
    for fit in (estimate_face_pseudo_attr, estimate_audio_session_pseudo_attr):
        proxy = fit(features, ids)
        before = proxy.manifest()
        assert set(proxy.transform(np.array([[-3.0], [3.0]])).tolist()) == {0, 1}
        assert proxy.manifest() == before
        assert before["join_unit"] == "participant"
        assert before["kmeans"]["n_init"] == 20
        assert sorted(before["kmeans"]["raw_to_canonical"]) == [0, 1]


def test_proxy_fit_is_invariant_to_participant_row_permutation():
    features = np.array([[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]])
    ids = np.array(["7", "1", "12", "2"])
    p = np.array([2, 0, 3, 1])
    a = estimate_face_pseudo_attr(features, ids)
    b = estimate_face_pseudo_attr(features[p], ids[p])
    assert a.manifest() == b.manifest()
    assert np.array_equal(a.transform(features), b.transform(features))


def test_global_random_proxy_is_fixed_near_balanced_and_permutation_invariant():
    ids = np.array([str(i) for i in range(1, 70)])
    permutation = np.arange(69)[::-1]
    original = random_proxy_groups(ids)
    permuted = random_proxy_groups(ids[permutation])
    restored = np.empty_like(permuted)
    restored[permutation] = permuted
    assert RANDOM_PROXY_SALT == "fair-avi-mit-random-proxy-v1"
    assert np.bincount(original, minlength=2).tolist() == [34, 35]
    assert np.array_equal(original, restored)


def test_global_random_proxy_rejects_partial_or_duplicate_population():
    with pytest.raises(ValueError, match="all 69"):
        random_proxy_groups(np.array(["1", "2"]))
    with pytest.raises(ValueError, match="unique"):
        random_proxy_groups(np.array(["1"] * 69))


def test_validity_check_is_descriptive_only():
    assert validity_check(np.array([0, 0, 1, 1]), np.array([1, 1, 0, 0])) == {
        "nmi": 1.0,
        "ari": 1.0,
    }


def test_canonical_cluster_ids_survive_raw_sklearn_label_swap():
    features = np.array([[-3.0], [-2.0], [2.0], [3.0]])
    proxy = estimate_face_pseudo_attr(features, np.array(["1", "2", "3", "4"]))
    expected = proxy.transform(features)
    proxy.clusterer.cluster_centers_ = proxy.clusterer.cluster_centers_[::-1].copy()
    swapped = type(proxy)(
        proxy.scaler,
        proxy.clusterer,
        tuple(reversed(proxy.raw_to_canonical)),
    )
    assert np.array_equal(expected, swapped.transform(features))


def test_balanced_sampler_cycles_deterministically_and_reports_repeats():
    ids = np.asarray([str(value) for value in range(1, 13)])
    groups = np.asarray([0] * 9 + [1] * 3)
    first = BalancedParticipantSampler(ids, groups, seed=11)
    second = BalancedParticipantSampler(ids, groups, seed=11)
    draws = []
    for _ in range(3):
        batch, metadata = first.draw()
        expected, expected_metadata = second.draw()
        assert np.array_equal(batch, expected)
        assert metadata == expected_metadata
        assert batch.shape == (16,)
        assert len(set(batch[:8])) == 8
        assert metadata["repeated_draw_count"] > 0
        draws.extend(batch[:8].tolist())
    assert set(draws[:9]) == set(ids[:9])


def test_balanced_sampler_rejects_empty_group():
    with pytest.raises(ValueError, match="non-empty"):
        BalancedParticipantSampler(np.array(["1", "2"]), np.array([0, 0]), seed=1)

    with pytest.raises(ValueError, match="exactly 8"):
        BalancedParticipantSampler(
            np.array(["1", "2"]),
            np.array([0, 1]),
            seed=1,
            participants_per_group=4,
        )


def test_balanced_sampler_is_invariant_to_input_row_order():
    ids = np.asarray([str(value) for value in range(1, 17)])
    groups = np.asarray([0] * 8 + [1] * 8)
    permutation = np.asarray([15, 0, 9, 3, 8, 2, 14, 1, 13, 4, 12, 5, 11, 6, 10, 7])
    first = BalancedParticipantSampler(ids, groups, seed=21)
    second = BalancedParticipantSampler(ids[permutation], groups[permutation], seed=21)
    assert np.array_equal(first.draw()[0], second.draw()[0])


def test_balanced_sampler_emits_deterministic_paired_schedule_and_group_evidence():
    ids = np.asarray([str(value) for value in range(1, 13)])
    groups = np.asarray([0] * 9 + [1] * 3)
    pairs = {
        participant_id: (f"P{participant_id}", f"PP{participant_id}")
        for participant_id in ids
    }
    first = BalancedParticipantSampler(ids, groups, seed=11).paired_schedule(
        pairs, batch_count=3
    )
    second = BalancedParticipantSampler(ids, groups, seed=11).paired_schedule(
        dict(reversed(list(pairs.items()))), batch_count=3
    )

    assert first == second
    for batch_index, batch in enumerate(first):
        assert batch["batch_index"] == batch_index
        assert batch["participant_count"] == 16
        assert batch["session_count"] == 32
        assert batch["groups"]["0"] == {
            "group_size": 9,
            "draw_count": 8,
            "unique_participant_count": 8,
            "repeated_draw_count": 0,
        }
        assert batch["groups"]["1"] == {
            "group_size": 3,
            "draw_count": 8,
            "unique_participant_count": 3,
            "repeated_draw_count": 5,
        }
        for index, participant_id in enumerate(batch["participant_ids"]):
            assert batch["session_ids"][2 * index : 2 * index + 2] == [
                f"P{participant_id}",
                f"PP{participant_id}",
            ]


def test_balanced_paired_schedule_fails_closed_on_incomplete_or_mismatched_pairs():
    sampler = BalancedParticipantSampler(
        np.asarray(["1", "2"]), np.asarray([0, 1]), seed=1
    )
    with pytest.raises(ValueError, match="exactly cover"):
        sampler.paired_schedule({"1": ("P1", "PP1")}, batch_count=1)
    with pytest.raises(ValueError, match="identity"):
        sampler.paired_schedule(
            {"1": ("P1", "PP2"), "2": ("P2", "PP2")}, batch_count=1
        )
