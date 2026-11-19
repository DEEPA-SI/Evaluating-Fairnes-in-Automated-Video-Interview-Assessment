"""Participant-level binary proxy assignment for MIT Interview."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

RANDOM_PROXY_SALT = "fair-avi-mit-random-proxy-v1"
KMEANS_RANDOM_STATE = 0


@dataclass(frozen=True)
class FittedBinaryProxy:
    """Training-participant scaler and K-means used unchanged on held-out participants."""

    scaler: StandardScaler
    clusterer: KMeans
    raw_to_canonical: tuple[int, int]

    def transform(self, features: np.ndarray) -> np.ndarray:
        matrix = _validated_feature_matrix(features, expected_columns=self.scaler.n_features_in_)
        raw = self.clusterer.predict(self.scaler.transform(matrix))
        mapping = np.asarray(self.raw_to_canonical, dtype=np.int8)
        return mapping[raw]

    def manifest(self) -> dict[str, Any]:
        raw_centres = np.asarray(self.clusterer.cluster_centers_, dtype=np.float64)
        canonical_centres = raw_centres[np.argsort(np.asarray(self.raw_to_canonical))]
        arrays = {
            "scaler_mean": np.asarray(self.scaler.mean_),
            "scaler_scale": np.asarray(self.scaler.scale_),
            "cluster_centres_canonical": canonical_centres,
            "raw_to_canonical": np.asarray(self.raw_to_canonical, dtype=np.int8),
        }
        checksums = {f"{name}_sha256": _array_sha256(value) for name, value in arrays.items()}
        return {
            "join_unit": "participant",
            "fit_scope": "training_participants_only",
            "standard_scaler": {"training_row_order": "ascending_numeric_participant_id"},
            "scaler_values": {
                "mean": arrays["scaler_mean"].astype(float).tolist(),
                "scale": arrays["scaler_scale"].astype(float).tolist(),
            },
            "kmeans": {
                "n_clusters": 2,
                "init": "k-means++",
                "n_init": 20,
                "max_iter": 300,
                "tol": 1e-4,
                "algorithm": "lloyd",
                "random_state": 0,
                "canonical_order": (
                    "ascending_sha256_little_endian_float64_centre_bytes_then_lexicographic_values"
                ),
                "raw_to_canonical": list(self.raw_to_canonical),
                "centres_canonical": canonical_centres.astype(float).tolist(),
            },
            **checksums,
            "combined_sha256": sha256("\n".join(checksums.values()).encode("ascii")).hexdigest(),
        }


def estimate_face_pseudo_attr(
    participant_features_train: np.ndarray, participant_ids_train: np.ndarray
) -> FittedBinaryProxy:
    """Fit a face-context proxy on training participants only."""
    return _fit_binary_proxy(participant_features_train, participant_ids_train)


def estimate_audio_session_pseudo_attr(
    participant_features_train: np.ndarray, participant_ids_train: np.ndarray
) -> FittedBinaryProxy:
    """Fit a whole-interview-audio-context proxy on training participants only."""
    return _fit_binary_proxy(participant_features_train, participant_ids_train)


def random_proxy_groups(participant_ids: np.ndarray) -> np.ndarray:
    """Return the one global, participant-stable 34/35 negative-control assignment."""
    identifiers = _validated_participant_ids(participant_ids)
    if identifiers.size != 69:
        raise ValueError("the frozen global random proxy requires all 69 participants")
    order = sorted(
        range(identifiers.size),
        key=lambda index: (
            sha256(
                RANDOM_PROXY_SALT.encode("ascii") + b"\0" + identifiers[index].encode("ascii")
            ).digest(),
            identifiers[index].encode("ascii"),
        ),
    )
    groups = np.ones(identifiers.size, dtype=np.int8)
    groups[np.asarray(order[:34], dtype=int)] = 0
    return groups


def validity_check(pseudo_attr: np.ndarray, audit_label: np.ndarray) -> dict[str, float]:
    """Return descriptive association after firewalled choices are locked."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    return {
        "nmi": float(normalized_mutual_info_score(audit_label, pseudo_attr)),
        "ari": float(adjusted_rand_score(audit_label, pseudo_attr)),
    }


def _fit_binary_proxy(features: np.ndarray, participant_ids: np.ndarray) -> FittedBinaryProxy:
    matrix = _validated_feature_matrix(features)
    identifiers = _validated_participant_ids(participant_ids, matrix.shape[0])
    if matrix.shape[0] < 2:
        raise ValueError("at least two training participants are required")
    order = np.asarray(
        sorted(range(len(identifiers)), key=lambda i: int(identifiers[i])), dtype=int
    )
    scaler = StandardScaler().fit(matrix[order])
    clusterer = KMeans(
        n_clusters=2,
        init="k-means++",
        n_init=20,
        max_iter=300,
        tol=1e-4,
        algorithm="lloyd",
        random_state=0,
    ).fit(scaler.transform(matrix[order]))
    if np.unique(clusterer.labels_).size != 2:
        raise ValueError("training features did not yield both required proxy groups")
    raw_to_canonical = _canonical_cluster_mapping(clusterer.cluster_centers_)
    return FittedBinaryProxy(scaler, clusterer, raw_to_canonical)


def _canonical_cluster_mapping(centres: np.ndarray) -> tuple[int, int]:
    """Map raw K-means IDs to stable IDs using the frozen centre-byte ordering."""
    matrix = np.asarray(centres, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != 2 or not np.isfinite(matrix).all():
        raise ValueError("binary proxy centres must be a finite 2D matrix with two rows")
    canonical_rows = [np.ascontiguousarray(row, dtype="<f8") for row in matrix]
    order = sorted(
        range(2),
        key=lambda raw: (
            sha256(canonical_rows[raw].tobytes()).digest(),
            tuple(float(value) for value in canonical_rows[raw]),
        ),
    )
    mapping = [0, 0]
    for canonical, raw in enumerate(order):
        mapping[raw] = canonical
    return mapping[0], mapping[1]


class BalancedParticipantSampler:
    """Seeded cyclic 8+8 participant sampler with explicit duplication metadata."""

    def __init__(
        self,
        participant_ids: np.ndarray,
        groups: np.ndarray,
        *,
        seed: int,
        participants_per_group: int = 8,
    ) -> None:
        identifiers = _validated_participant_ids(participant_ids)
        labels = np.asarray(groups)
        if labels.shape != identifiers.shape or not np.isin(labels, (0, 1)).all():
            raise ValueError("groups must contain one binary label per participant")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if participants_per_group != 8:
            raise ValueError(
                "the frozen balanced sampler requires exactly 8 participants per group"
            )
        members = [
            np.asarray(
                sorted(identifiers[labels == group].tolist(), key=int),
                dtype=object,
            )
            for group in (0, 1)
        ]
        if any(values.size == 0 for values in members):
            raise ValueError("both proxy groups must be non-empty")
        self._half_size = participants_per_group
        self._group_sizes = tuple(int(values.size) for values in members)
        self._group_by_participant = {
            str(participant_id): group
            for group, values in enumerate(members)
            for participant_id in values
        }
        self._samplers = tuple(
            _CyclicGroupSampler(values, np.random.default_rng(seed + group))
            for group, values in enumerate(members)
        )

    def draw(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Draw one balanced participant batch and report unique/repeated counts."""
        halves = [sampler.draw(self._half_size) for sampler in self._samplers]
        batch = np.concatenate(halves)
        unique = int(np.unique(batch).size)
        repeated = int(batch.size - unique)
        group_evidence = {
            str(group): {
                "group_size": self._group_sizes[group],
                "draw_count": self._half_size,
                "unique_participant_count": int(np.unique(half).size),
                "repeated_draw_count": int(half.size - np.unique(half).size),
            }
            for group, half in enumerate(halves)
        }
        return batch, {
            "participant_count": int(batch.size),
            "unique_participant_count": unique,
            "repeated_draw_count": repeated,
            "unique_participant_rate": unique / batch.size,
            "duplication_rate": repeated / batch.size,
            "group_counts": [self._half_size, self._half_size],
            "groups": group_evidence,
        }

    def paired_schedule(
        self,
        session_pairs: Mapping[str, Sequence[str]],
        *,
        batch_count: int,
    ) -> tuple[dict[str, Any], ...]:
        """Return a deterministic schedule with both sessions for every participant draw."""
        if isinstance(batch_count, bool) or not isinstance(batch_count, int) or batch_count <= 0:
            raise ValueError("batch_count must be a positive integer")
        pairs = _validated_session_pairs(session_pairs, frozenset(self._group_by_participant))
        schedule: list[dict[str, Any]] = []
        for batch_index in range(batch_count):
            participant_ids, evidence = self.draw()
            session_ids = [
                session_id
                for participant_id in participant_ids.tolist()
                for session_id in pairs[str(participant_id)]
            ]
            if len(session_ids) != 2 * participant_ids.size:
                raise RuntimeError("participant-paired schedule lost one or more sessions")
            schedule.append(
                {
                    "batch_index": batch_index,
                    "participant_ids": participant_ids.tolist(),
                    "proxy_groups": [
                        self._group_by_participant[str(participant_id)]
                        for participant_id in participant_ids
                    ],
                    "session_ids": session_ids,
                    "session_count": len(session_ids),
                    **evidence,
                }
            )
        return tuple(schedule)


class _CyclicGroupSampler:
    def __init__(self, members: np.ndarray, rng: np.random.Generator) -> None:
        self._members = np.asarray(members, dtype=object)
        self._rng = rng
        self._order = self._new_order(frozenset())
        self._cursor = 0

    def _new_order(self, avoid_first: frozenset[str]) -> np.ndarray:
        shuffled = self._members[self._rng.permutation(self._members.size)]
        if avoid_first:
            allowed = [value for value in shuffled if value not in avoid_first]
            delayed = [value for value in shuffled if value in avoid_first]
            shuffled = np.asarray([*allowed, *delayed], dtype=object)
        return shuffled

    def draw(self, count: int) -> np.ndarray:
        selected: list[str] = []
        while len(selected) < count:
            if self._cursor == self._order.size:
                avoid = frozenset(selected) if self._members.size >= count else frozenset()
                self._order = self._new_order(avoid)
                self._cursor = 0
            selected.append(str(self._order[self._cursor]))
            self._cursor += 1
        if self._members.size >= count and len(set(selected)) != count:
            raise RuntimeError("cyclic sampler violated within-half uniqueness")
        return np.asarray(selected, dtype=object)


def _validated_participant_ids(values: np.ndarray, expected: int | None = None) -> np.ndarray:
    from src.data.mit_interview import canonical_participant_id

    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("participant_ids must be a non-empty one-dimensional array")
    result = np.asarray([canonical_participant_id(value) for value in array.tolist()], dtype=object)
    if expected is not None and result.size != expected:
        raise ValueError("participant_ids must contain one ID per feature row")
    if len(set(result.tolist())) != result.size:
        raise ValueError("participant_ids must be unique")
    return result


def _validated_session_pairs(
    values: Mapping[str, Sequence[str]], expected_participants: frozenset[str]
) -> dict[str, tuple[str, str]]:
    from src.data.mit_interview import canonical_participant_id, parse_session_id

    if not isinstance(values, Mapping):
        raise TypeError("session_pairs must be a participant-to-session mapping")
    pairs: dict[str, tuple[str, str]] = {}
    for supplied_participant_id, supplied_sessions in values.items():
        if not isinstance(supplied_participant_id, str):
            raise TypeError("session_pairs keys must be canonical participant IDs")
        participant_id = canonical_participant_id(supplied_participant_id)
        if participant_id != supplied_participant_id or participant_id in pairs:
            raise ValueError("session_pairs keys must be unique canonical participant IDs")
        if (
            not isinstance(supplied_sessions, Sequence)
            or isinstance(supplied_sessions, (str, bytes))
            or len(supplied_sessions) != 2
        ):
            raise ValueError("each participant must have exactly one pre/post session pair")
        parsed: dict[str, str] = {}
        for supplied_session_id in supplied_sessions:
            if not isinstance(supplied_session_id, str):
                raise TypeError("session pair values must be canonical session IDs")
            session_id, parsed_participant_id, condition = parse_session_id(supplied_session_id)
            if (
                session_id != supplied_session_id
                or parsed_participant_id != participant_id
                or condition in parsed
            ):
                raise ValueError("session pair identity is invalid or duplicated")
            parsed[condition] = session_id
        if set(parsed) != {"pre", "post"}:
            raise ValueError("each participant must have exactly one pre/post session pair")
        pairs[participant_id] = (parsed["pre"], parsed["post"])
    if frozenset(pairs) != expected_participants:
        raise ValueError("session_pairs must exactly cover the sampler participants")
    return pairs


def _validated_feature_matrix(value: np.ndarray, expected_columns: int | None = None) -> np.ndarray:
    matrix = np.asarray(value)
    if matrix.ndim != 2 or 0 in matrix.shape or not np.issubdtype(matrix.dtype, np.number):
        raise ValueError("features must be a non-empty numeric matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("features must contain only finite values")
    if expected_columns is not None and matrix.shape[1] != expected_columns:
        raise ValueError(f"features must have {expected_columns} columns")
    return matrix.astype(float, copy=False)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()
