"""Restricted, deterministic evidence for auditing MIT source-session identity joins.

This module does not alter dataset records and its output is not an approved participant map.
It re-embeds a fixed subset of already validated candidate-face crops, solves a global one-to-one
pre/post assignment, and emits only crop hashes and aggregate similarity evidence. Pixels and
embedding vectors are deliberately absent from the report.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.features.extraction import canonical_json_bytes, file_sha256

from .annotation_stimuli import FACE_MANIFEST_SCHEMA_VERSION, load_face_crop_manifest

IDENTITY_JOIN_AUDIT_SCHEMA_VERSION = "mit-identity-join-audit-2"
IDENTITY_JOIN_AUDIT_METHOD_VERSION = "fair-avi-mit-identity-join-audit-2"
SCRFD_ARTIFACT_FILE = "det_10g.onnx"
ARCFACE_ARTIFACT_FILE = "w600k_r50.onnx"
ENDPOINT_CROP_COUNT = 12
MAX_REDETECTION_FALLBACK_OFFSET = 6
EMBEDDING_DIMENSION = 512
SIMILARITY_DECIMAL_PLACES = 12

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_SESSION_PATTERN = re.compile(r"^(PP|P)([1-9][0-9]*)$", re.ASCII)
_TOP_FIELDS = {
    "schema_version",
    "status",
    "mapping_approved",
    "training_input_authorized",
    "purpose",
    "bindings",
    "method",
    "coverage",
    "sessions",
    "pairwise_cosine",
    "pre_to_post_evidence",
    "assignment",
    "proposed_source_session_to_canonical_participant",
    "changed_nominal_pair_count",
    "changed_nominal_pairs",
    "report_payload_sha256",
}


@dataclass(frozen=True, slots=True)
class EvidenceThresholds:
    """Explicit, recorded acceptance thresholds for identity-join evidence."""

    minimum_match_similarity: float
    minimum_match_margin: float
    minimum_crop_to_session_similarity: float
    tie_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        values = (
            self.minimum_match_similarity,
            self.minimum_match_margin,
            self.minimum_crop_to_session_similarity,
            self.tie_tolerance,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError("identity-join evidence thresholds must be finite numbers")
        if not 0 <= self.minimum_match_similarity <= 1:
            raise ValueError("minimum match similarity must be in [0,1]")
        if not 0 < self.minimum_match_margin <= 2:
            raise ValueError("minimum match margin must be in (0,2]")
        if not 0 <= self.minimum_crop_to_session_similarity <= 1:
            raise ValueError("minimum crop-to-session similarity must be in [0,1]")
        if not 0 < self.tie_tolerance < self.minimum_match_margin:
            raise ValueError("tie tolerance must be positive and below the minimum match margin")

    def as_dict(self) -> dict[str, float]:
        """Return a JSON-safe exact threshold record."""
        return {
            "minimum_match_similarity": self.minimum_match_similarity,
            "minimum_match_margin": self.minimum_match_margin,
            "minimum_crop_to_session_similarity": self.minimum_crop_to_session_similarity,
            "tie_tolerance": self.tie_tolerance,
        }


class CropEmbedder(Protocol):
    """Injectable crop-embedding boundary used by production and synthetic tests."""

    @property
    def runtime_provenance(self) -> Mapping[str, str]:
        """Return exact runtime versions used to produce embeddings."""

    def embed(self, crop_path: Path, *, expected_sha256: str) -> np.ndarray:
        """Return one finite 512-dimensional candidate-face embedding."""


class FaceRedetectionUnavailableError(ValueError):
    """A valid crop has no unique acceptable SCRFD candidate-face re-detection."""


@dataclass(frozen=True, slots=True)
class _VerifiedArtifact:
    path: Path
    binding: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _VerifiedImplementation:
    paths: dict[str, Path]
    hashes: dict[str, str]


def endpoint_rank_indices(count: int, maximum: int = ENDPOINT_CROP_COUNT) -> tuple[int, ...]:
    """Select an exact, endpoint-spanning rank grid from an ordered sequence."""
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or isinstance(maximum, bool)
        or not isinstance(maximum, int)
    ):
        raise TypeError("endpoint-rank count and maximum must be integers")
    if count < maximum or maximum < 2:
        raise ValueError("endpoint-rank selection requires count >= maximum >= 2")
    return tuple((rank * (count - 1)) // (maximum - 1) for rank in range(maximum))


def fallback_rank_indices(
    target_rank: int,
    frame_count: int,
    maximum_offset: int = MAX_REDETECTION_FALLBACK_OFFSET,
) -> tuple[int, ...]:
    """Return bounded nearest-rank candidates, breaking equal-distance ties downward."""
    for label, value in (
        ("target_rank", target_rank),
        ("frame_count", frame_count),
        ("maximum_offset", maximum_offset),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{label} must be an integer")
    if frame_count <= 0 or not 0 <= target_rank < frame_count or maximum_offset < 0:
        raise ValueError("fallback rank search bounds are invalid")
    candidates = [target_rank]
    for offset in range(1, maximum_offset + 1):
        lower = target_rank - offset
        upper = target_rank + offset
        if lower >= 0:
            candidates.append(lower)
        if upper < frame_count:
            candidates.append(upper)
    return tuple(candidates)


def aggregate_session_embeddings(
    crop_embeddings: Sequence[np.ndarray],
    *,
    minimum_crop_to_session_similarity: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Average twelve individually L2-normalized 512-dimensional crop embeddings."""
    if (
        isinstance(minimum_crop_to_session_similarity, bool)
        or not isinstance(minimum_crop_to_session_similarity, (int, float))
        or not math.isfinite(minimum_crop_to_session_similarity)
        or not 0 <= minimum_crop_to_session_similarity <= 1
    ):
        raise ValueError("minimum crop-to-session similarity must be finite and in [0,1]")
    if len(crop_embeddings) != ENDPOINT_CROP_COUNT:
        raise ValueError(f"each session must provide exactly {ENDPOINT_CROP_COUNT} embeddings")
    rows = []
    for embedding in crop_embeddings:
        row = _validated_crop_embedding(embedding)
        rows.append(row / float(np.linalg.norm(row)))
    matrix = np.stack(rows)
    mean = matrix.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean))
    if not math.isfinite(mean_norm) or mean_norm <= np.finfo(np.float64).eps:
        raise ValueError("normalized crop embeddings do not form a stable session mean")
    unit_mean = mean / mean_norm
    crop_similarities = _rounded(matrix @ unit_mean)
    minimum = float(np.min(crop_similarities))
    average = float(np.mean(crop_similarities))
    if minimum < minimum_crop_to_session_similarity:
        raise ValueError("session crop embeddings fall below the declared consistency threshold")
    return mean, {
        "minimum_crop_to_session_similarity": minimum,
        "mean_crop_to_session_similarity": _round_float(average),
    }


def _validated_crop_embedding(value: object) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in {"f", "i"}:
        raise ValueError("crop embedding must contain only numeric values")
    row = np.asarray(raw, dtype=np.float64)
    if row.shape != (EMBEDDING_DIMENSION,) or not np.isfinite(row).all():
        raise ValueError("crop embedding must be a finite 512-dimensional vector")
    norm = float(np.linalg.norm(row))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("crop embedding must have a positive finite L2 norm")
    return row.copy()


def cosine_similarity_matrix(
    pre_embeddings: Sequence[np.ndarray], post_embeddings: Sequence[np.ndarray]
) -> np.ndarray:
    """Return a finite, twelve-decimal pre-to-post cosine matrix."""
    if not pre_embeddings or len(pre_embeddings) != len(post_embeddings):
        raise ValueError("pre/post embedding sets must be non-empty and equally sized")
    pre = _embedding_matrix(pre_embeddings, "pre")
    post = _embedding_matrix(post_embeddings, "post")
    similarities = (pre / np.linalg.norm(pre, axis=1, keepdims=True)) @ (
        post / np.linalg.norm(post, axis=1, keepdims=True)
    ).T
    if not np.isfinite(similarities).all():
        raise ValueError("cosine similarity matrix contains non-finite values")
    if np.any(similarities < -1 - 1e-12) or np.any(similarities > 1 + 1e-12):
        raise ValueError("cosine similarity escaped its numerical range")
    return _rounded(np.clip(similarities, -1.0, 1.0))


def analyse_identity_similarity(
    pre_session_ids: Sequence[str],
    post_session_ids: Sequence[str],
    similarities: np.ndarray,
    *,
    thresholds: EvidenceThresholds,
) -> dict[str, Any]:
    """Rank nominal joins and solve a unique maximum-cosine Hungarian assignment."""
    pre_ids = tuple(pre_session_ids)
    post_ids = tuple(post_session_ids)
    _validate_ordered_sessions(pre_ids, "P")
    _validate_ordered_sessions(post_ids, "PP")
    if tuple(_session_number(value) for value in pre_ids) != tuple(
        _session_number(value) for value in post_ids
    ):
        raise ValueError("pre/post source sessions must have the same nominal numeric coverage")
    raw_matrix = np.asarray(similarities)
    if raw_matrix.dtype.kind not in {"f", "i"}:
        raise ValueError("identity similarity matrix must contain only numeric values")
    matrix = np.asarray(raw_matrix, dtype=np.float64)
    expected_shape = (len(pre_ids), len(post_ids))
    if matrix.shape != expected_shape or not np.isfinite(matrix).all():
        raise ValueError("identity similarity matrix has an unexpected shape or non-finite value")
    if np.any(matrix < -1) or np.any(matrix > 1):
        raise ValueError("identity similarity matrix values must be in [-1,1]")
    if not np.array_equal(matrix, _rounded(matrix)):
        raise ValueError("identity similarities must use the frozen decimal precision")
    if len(pre_ids) < 2:
        raise ValueError("identity-join assignment requires at least two participants")

    ranked_columns: list[np.ndarray] = []
    row_best: list[int] = []
    for row_index, row in enumerate(matrix):
        order = np.argsort(-row, kind="stable")
        ordered_scores = row[order]
        if ordered_scores[0] - ordered_scores[1] <= thresholds.tie_tolerance:
            raise ValueError("identity similarity row contains an unresolved top-match tie")
        nominal_column = post_ids.index(f"PP{_session_number(pre_ids[row_index])}")
        nominal_score = row[nominal_column]
        nominal_ties = np.flatnonzero(np.abs(row - nominal_score) <= thresholds.tie_tolerance)
        if len(nominal_ties) != 1:
            raise ValueError("identity similarity row contains an unresolved nominal-rank tie")
        margin = float(ordered_scores[0] - ordered_scores[1])
        if ordered_scores[0] < thresholds.minimum_match_similarity:
            raise ValueError("identity similarity falls below the declared evidence threshold")
        if margin < thresholds.minimum_match_margin:
            raise ValueError(
                "identity similarity margin falls below the declared evidence threshold"
            )
        ranked_columns.append(order)
        row_best.append(int(order[0]))
    if len(set(row_best)) != len(row_best):
        raise ValueError("row-best post-session matches are not one-to-one")

    row_indices, assigned_columns = linear_sum_assignment(-matrix)
    if not np.array_equal(row_indices, np.arange(len(pre_ids))):
        raise ValueError("Hungarian assignment did not cover every pre session exactly once")
    if len(set(int(value) for value in assigned_columns)) != len(post_ids):
        raise ValueError("Hungarian assignment did not cover every post session exactly once")
    if [int(value) for value in assigned_columns] != row_best:
        raise ValueError("global assignment conflicts with a locally supported row-best match")

    total = float(matrix[row_indices, assigned_columns].sum())
    alternative_total = _best_alternative_assignment_total(matrix, assigned_columns)
    alternative_gap = total - alternative_total
    if alternative_gap <= thresholds.tie_tolerance:
        raise ValueError("Hungarian assignment optimum is tied or numerically unresolved")

    evidence: list[dict[str, Any]] = []
    assignment_pairs: list[dict[str, Any]] = []
    changed: list[dict[str, str]] = []
    proposed_mapping: list[dict[str, str]] = []
    for row_index, pre_id in enumerate(pre_ids):
        number = _session_number(pre_id)
        nominal_post = f"PP{number}"
        nominal_column = post_ids.index(nominal_post)
        assigned_column = int(assigned_columns[row_index])
        order = ranked_columns[row_index]
        assigned_rank = int(np.flatnonzero(order == assigned_column)[0]) + 1
        nominal_rank = int(np.flatnonzero(order == nominal_column)[0]) + 1
        best_column = int(order[0])
        best_margin = float(matrix[row_index, best_column] - matrix[row_index, int(order[1])])
        assigned_post = post_ids[assigned_column]
        row = {
            "pre_source_session_id": pre_id,
            "nominal_post_source_session_id": nominal_post,
            "nominal_similarity": _round_float(matrix[row_index, nominal_column]),
            "nominal_rank": nominal_rank,
            "best_post_source_session_id": post_ids[best_column],
            "best_similarity": _round_float(matrix[row_index, best_column]),
            "best_margin": _round_float(best_margin),
            "assigned_post_source_session_id": assigned_post,
            "assigned_similarity": _round_float(matrix[row_index, assigned_column]),
            "assigned_rank": assigned_rank,
            "assignment_equals_row_best": assigned_column == best_column,
        }
        evidence.append(row)
        assignment_pairs.append(
            {
                "pre_source_session_id": pre_id,
                "post_source_session_id": assigned_post,
                "cosine_similarity": row["assigned_similarity"],
            }
        )
        proposed_mapping.extend(
            (
                {
                    "source_session_id": pre_id,
                    "canonical_participant_id": number,
                    "phase": "pre",
                    "basis": "pre_source_session_anchor",
                },
                {
                    "source_session_id": assigned_post,
                    "canonical_participant_id": number,
                    "phase": "post",
                    "basis": "hungarian_assignment_to_pre_source_session_anchor",
                },
            )
        )
        if assigned_post != nominal_post:
            changed.append(
                {
                    "pre_source_session_id": pre_id,
                    "nominal_post_source_session_id": nominal_post,
                    "proposed_post_source_session_id": assigned_post,
                }
            )

    proposed_mapping.sort(
        key=lambda row: (
            int(row["canonical_participant_id"]),
            0 if row["phase"] == "pre" else 1,
            row["source_session_id"].encode("ascii"),
        )
    )
    return {
        "pre_to_post_evidence": evidence,
        "assignment": {
            "algorithm": "scipy_linear_sum_assignment_maximum_cosine",
            "one_to_one": True,
            "unique_optimum_verified": True,
            "total_similarity": _round_float(total),
            "best_alternative_total_similarity": _round_float(alternative_total),
            "best_alternative_gap": _round_float(alternative_gap),
            "pairs": assignment_pairs,
        },
        "proposed_mapping": proposed_mapping,
        "changed_nominal_pairs": changed,
    }


def build_identity_join_audit_report(
    *,
    session_embeddings: Mapping[str, np.ndarray],
    session_crop_evidence: Mapping[str, Mapping[str, Any]],
    bindings: Mapping[str, Any],
    thresholds: EvidenceThresholds,
    expected_participant_count: int,
) -> dict[str, Any]:
    """Build a checksum-bound proposed mapping from already computed session evidence."""
    if (
        isinstance(expected_participant_count, bool)
        or not isinstance(expected_participant_count, int)
        or expected_participant_count < 2
    ):
        raise ValueError("expected participant count must be an integer of at least two")
    pre_ids, post_ids = _partition_sessions(tuple(session_embeddings))
    if len(pre_ids) != expected_participant_count:
        raise ValueError("identity-join session coverage does not match the expected corpus")
    if set(session_crop_evidence) != set(session_embeddings):
        raise ValueError("session crop evidence must exactly cover the session embeddings")
    sessions = _validated_session_evidence(session_crop_evidence, thresholds)
    matrix = cosine_similarity_matrix(
        [session_embeddings[value] for value in pre_ids],
        [session_embeddings[value] for value in post_ids],
    )
    analysis = analyse_identity_similarity(pre_ids, post_ids, matrix, thresholds=thresholds)
    selected_crop_binding = _selected_crop_binding(sessions)
    validated_bindings = _validate_bindings(
        {**dict(bindings), "selected_crop_file_set_sha256": selected_crop_binding}
    )
    unsigned: dict[str, Any] = {
        "schema_version": IDENTITY_JOIN_AUDIT_SCHEMA_VERSION,
        "status": "proposed_not_approved",
        "mapping_approved": False,
        "training_input_authorized": False,
        "purpose": "restricted_source_session_identity_join_audit_only",
        "bindings": validated_bindings,
        "method": {
            "method_version": IDENTITY_JOIN_AUDIT_METHOD_VERSION,
            "subject_interpretation": "candidate_face_hypothesis_not_biometric_identity_truth",
            "crops_per_session": ENDPOINT_CROP_COUNT,
            "crop_selection": "twelve_endpoint_targets_with_nearest_unused_fallback",
            "redetection_fallback": {
                "trigger": "FaceRedetectionUnavailableError_only",
                "candidate_order": "increasing_absolute_manifest_rank_offset_lower_then_higher",
                "maximum_absolute_manifest_rank_offset": MAX_REDETECTION_FALLBACK_OFFSET,
                "selected_rank_reuse": "prohibited_within_session",
                "failed_redetection_rank_cache": "deterministic_within_session",
            },
            "crop_embedding": "scrfd_redetect_five_point_align_arcface_w600k_r50",
            "embedding_dimension": EMBEDDING_DIMENSION,
            "session_aggregation": "l2_each_then_arithmetic_mean_then_cosine_normalize",
            "similarity": "pre_to_post_cosine",
            "similarity_decimal_places": SIMILARITY_DECIMAL_PLACES,
            "nominal_rank": "one_based_descending_cosine",
            "assignment": "exact_hungarian_maximum_total_cosine",
            "proposal_boundary": "audit_evidence_only_requires_separate_approval_before_use",
            "thresholds": thresholds.as_dict(),
        },
        "coverage": {
            "expected_participant_count": expected_participant_count,
            "pre_session_count": len(pre_ids),
            "post_session_count": len(post_ids),
            "session_count": len(sessions),
            "selected_crop_file_count": len(sessions) * ENDPOINT_CROP_COUNT,
            "substituted_crop_count": sum(row["substitution_count"] for row in sessions),
            "all_face_manifest_sessions_successful": True,
            "complete_pre_post_coverage_verified": True,
            "complete_temporal_target_coverage_verified": True,
            "unique_selected_crops_per_session_verified": True,
        },
        "sessions": sessions,
        "pairwise_cosine": {
            "pre_source_session_ids": list(pre_ids),
            "post_source_session_ids": list(post_ids),
            "values": matrix.tolist(),
            "float64_little_endian_sha256": hashlib.sha256(
                np.asarray(matrix, dtype="<f8").tobytes(order="C")
            ).hexdigest(),
        },
        "pre_to_post_evidence": analysis["pre_to_post_evidence"],
        "assignment": analysis["assignment"],
        "proposed_source_session_to_canonical_participant": analysis["proposed_mapping"],
        "changed_nominal_pair_count": len(analysis["changed_nominal_pairs"]),
        "changed_nominal_pairs": analysis["changed_nominal_pairs"],
    }
    report = {**unsigned, "report_payload_sha256": _canonical_sha256(unsigned)}
    validate_identity_join_audit_report(report)
    return report


def _select_session_crops(
    frames: Sequence[Mapping[str, Any]],
    *,
    manifest_root: Path,
    embedder: CropEmbedder,
) -> tuple[list[np.ndarray], list[dict[str, Any]], int]:
    """Embed twelve endpoint targets with bounded nearest unused redetection fallback."""
    target_ranks = endpoint_rank_indices(len(frames))
    used_ranks: set[int] = set()
    redetection_unavailable_ranks: set[int] = set()
    crop_vectors: list[np.ndarray] = []
    selected_rows: list[dict[str, Any]] = []
    substitution_count = 0
    for endpoint_rank, target_rank in enumerate(target_ranks):
        selected = False
        for selected_rank in fallback_rank_indices(target_rank, len(frames)):
            if selected_rank in used_ranks:
                continue
            frame = frames[selected_rank]
            crop_path = _manifest_crop_path(manifest_root, frame["path"])
            observed_hash = file_sha256(crop_path)
            if observed_hash != frame["crop_sha256"]:
                raise ValueError("selected face crop changed after manifest validation")
            if selected_rank in redetection_unavailable_ranks:
                continue
            try:
                embedding = embedder.embed(
                    crop_path,
                    expected_sha256=frame["crop_sha256"],
                )
            except FaceRedetectionUnavailableError:
                if file_sha256(crop_path) != observed_hash:
                    raise ValueError(
                        "face crop changed during failed identity re-detection"
                    ) from None
                redetection_unavailable_ranks.add(selected_rank)
                continue
            if file_sha256(crop_path) != observed_hash:
                raise ValueError("selected face crop changed during identity embedding")
            crop_vectors.append(_validated_crop_embedding(embedding))
            used_ranks.add(selected_rank)
            offset = selected_rank - target_rank
            substitution_count += int(offset != 0)
            selected_rows.append(
                {
                    "endpoint_rank": endpoint_rank,
                    "target_manifest_frame_rank": target_rank,
                    "selected_manifest_frame_rank": selected_rank,
                    "rank_offset": offset,
                    "grid_index": frame["grid_index"],
                    "crop_sha256": observed_hash,
                }
            )
            selected = True
            break
        if not selected:
            raise FaceRedetectionUnavailableError(
                "bounded nearest-rank search could not provide twelve unique valid face crops"
            )
    if len(used_ranks) != ENDPOINT_CROP_COUNT:
        raise ValueError("identity fallback did not select twelve unique manifest crops")
    return crop_vectors, selected_rows, substitution_count


def audit_face_crop_manifest(
    *,
    face_crop_manifest: str | Path,
    detector_model: str | Path,
    detector_sha256: str,
    recognition_model: str | Path,
    recognition_sha256: str,
    thresholds: EvidenceThresholds,
    expected_participant_count: int = 69,
    embedder_factory: Callable[[Path, Path, Mapping[str, Any], float], CropEmbedder] | None = None,
    implementation_files: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Audit a complete strict face manifest without changing any dataset mapping."""
    manifest_file = _regular_file(face_crop_manifest, "face-crop manifest")
    manifest_bytes = manifest_file.read_bytes()
    manifest = load_face_crop_manifest(manifest_file)
    if canonical_json_bytes(manifest) != manifest_bytes:
        raise ValueError("face-crop manifest must use the canonical JSON serialization")
    pre_ids, post_ids = _partition_sessions(tuple(manifest["clips"]))
    if len(pre_ids) != expected_participant_count or len(post_ids) != expected_participant_count:
        raise ValueError("face-crop manifest does not contain the complete expected MIT corpus")

    detector = _verify_artifact(detector_model, detector_sha256, "SCRFD detector")
    recognition = _verify_artifact(recognition_model, recognition_sha256, "ArcFace recognition")
    selector = manifest["selector"]
    if detector.binding["file_name"] != selector["artifact_file"]:
        raise ValueError("SCRFD detector filename does not match face-manifest provenance")
    if detector.binding["file_sha256"] != selector["artifact_sha256"]:
        raise ValueError("SCRFD detector checksum does not match face-manifest provenance")
    if detector.binding["file_size_bytes"] != selector["artifact_size_bytes"]:
        raise ValueError("SCRFD detector size does not match face-manifest provenance")
    if recognition.binding["file_name"] != ARCFACE_ARTIFACT_FILE:
        raise ValueError("ArcFace recognition artifact must be w600k_r50.onnx")
    implementation = _verify_implementation_files(implementation_files)

    factory = embedder_factory or InsightFaceIdentityEmbedder
    embedder = factory(detector.path, recognition.path, selector, thresholds.tie_tolerance)
    runtime = _validate_runtime_provenance(embedder.runtime_provenance)
    manifest_root = manifest_file.parent.resolve()
    embeddings: dict[str, np.ndarray] = {}
    crop_evidence: dict[str, dict[str, Any]] = {}
    for session_id in (*pre_ids, *post_ids):
        clip = manifest["clips"][session_id]
        if clip["status"] != "success" or clip["failure"] is not None:
            raise ValueError("identity-join audit rejects incomplete or failed face sessions")
        frames = clip["frames"]
        crop_vectors, selected_rows, substitution_count = _select_session_crops(
            frames,
            manifest_root=manifest_root,
            embedder=embedder,
        )
        mean, diagnostics = aggregate_session_embeddings(
            crop_vectors,
            minimum_crop_to_session_similarity=(thresholds.minimum_crop_to_session_similarity),
        )
        embeddings[session_id] = mean
        crop_evidence[session_id] = {
            "source_session_id": session_id,
            "phase": "pre" if not session_id.startswith("PP") else "post",
            "nominal_participant_id": _session_number(session_id),
            "manifest_frame_count": len(frames),
            "selected_crop_count": ENDPOINT_CROP_COUNT,
            "substitution_count": substitution_count,
            "selected_crops": selected_rows,
            **diagnostics,
        }

    if file_sha256(manifest_file) != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError("face-crop manifest changed during identity-join audit")
    detector_after = _verify_artifact(detector.path, detector_sha256, "SCRFD detector")
    if detector_after.binding != detector.binding:
        raise ValueError("SCRFD detector artifact changed during identity-join audit")
    if (
        _verify_artifact(recognition.path, recognition_sha256, "ArcFace recognition").binding
        != recognition.binding
    ):
        raise ValueError("ArcFace recognition artifact changed during identity-join audit")
    implementation_after = _verify_implementation_files(implementation.paths)
    if implementation_after.hashes != implementation.hashes:
        raise ValueError("identity-join implementation changed during audit execution")

    bindings = {
        "face_crop_manifest_schema_version": FACE_MANIFEST_SCHEMA_VERSION,
        "face_crop_manifest_file_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "face_crop_manifest_payload_sha256": _canonical_sha256(manifest),
        "face_selector_payload_sha256": _canonical_sha256(selector),
        "detector_artifact": detector.binding,
        "recognition_artifact": recognition.binding,
        "implementation_file_sha256": implementation.hashes,
        "runtime": runtime,
    }
    return build_identity_join_audit_report(
        session_embeddings=embeddings,
        session_crop_evidence=crop_evidence,
        bindings=bindings,
        thresholds=thresholds,
        expected_participant_count=expected_participant_count,
    )


class InsightFaceIdentityEmbedder:
    """Deterministic CPU SCRFD re-detection and ArcFace aligned embedding runtime."""

    def __init__(
        self,
        detector_model: Path,
        recognition_model: Path,
        selector: Mapping[str, Any],
        tie_tolerance: float,
    ) -> None:
        try:
            import cv2
            import insightface
            import onnxruntime
            from insightface.model_zoo.arcface_onnx import ArcFaceONNX
            from insightface.model_zoo.scrfd import SCRFD
        except ImportError as error:
            raise RuntimeError(
                "pinned InsightFace, ONNX Runtime and OpenCV are required for identity audit"
            ) from error
        if "CPUExecutionProvider" not in onnxruntime.get_available_providers():
            raise RuntimeError("ONNX Runtime CPUExecutionProvider is unavailable")
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)

        detector_session = _onnx_session(onnxruntime, detector_model)
        recognition_session = _onnx_session(onnxruntime, recognition_model)
        self._detector = SCRFD(model_file=str(detector_model), session=detector_session)
        self._detector.prepare(
            -1,
            input_size=tuple(selector["input_size"]),
            det_thresh=float(selector["confidence_threshold"]),
            nms_thresh=float(selector["nms_threshold"]),
        )
        self._recognizer = ArcFaceONNX(
            model_file=str(recognition_model), session=recognition_session
        )
        self._recognizer.prepare(-1)
        if self._recognizer.input_size != (112, 112):
            raise RuntimeError("ArcFace recognition input shape drifted from 112x112")
        output_shape = tuple(self._recognizer.output_shape)
        if not output_shape or output_shape[-1] != EMBEDDING_DIMENSION:
            raise RuntimeError("ArcFace recognition output dimension drifted from 512")
        self._cv2 = cv2
        self._selector = dict(selector)
        self._tie_tolerance = tie_tolerance
        self._runtime_provenance = {
            "insightface_version": str(insightface.__version__),
            "onnxruntime_version": str(onnxruntime.__version__),
            "opencv_version": str(cv2.__version__),
            "numpy_version": str(np.__version__),
            "scipy_version": _scipy_version(),
            "execution_provider": "CPUExecutionProvider",
        }

    @property
    def runtime_provenance(self) -> Mapping[str, str]:
        """Return exact runtime versions without host-specific fields."""
        return dict(self._runtime_provenance)

    def embed(self, crop_path: Path, *, expected_sha256: str) -> np.ndarray:
        """Re-detect, five-point align and embed one checksum-bound crop."""
        raw = crop_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("face crop checksum changed before identity embedding")
        image = self._cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), self._cv2.IMREAD_COLOR)
        if (
            image is None
            or image.ndim != 3
            or image.shape[2] != 3
            or image.dtype != np.uint8
            or image.shape[0] <= 0
            or image.shape[1] <= 0
        ):
            raise ValueError("face crop is not a decodable non-empty uint8 colour image")
        boxes, keypoints = self._detector.detect(
            image, input_size=tuple(self._selector["input_size"])
        )
        landmarks = select_candidate_landmarks(
            boxes,
            keypoints,
            image_width=image.shape[1],
            image_height=image.shape[0],
            maximum_centrality=float(self._selector["candidate_centrality_max_ratio"]),
            minimum_centrality_margin=float(self._selector["candidate_centrality_margin_ratio"]),
            tie_tolerance=self._tie_tolerance,
        )
        from insightface.utils import face_align

        aligned = face_align.norm_crop(image, landmark=landmarks, image_size=112)
        values = np.asarray(self._recognizer.get_feat(aligned), dtype=np.float64)
        if values.shape != (1, EMBEDDING_DIMENSION) or not np.isfinite(values).all():
            raise ValueError("ArcFace output must be one finite 512-dimensional embedding")
        return values[0]


def select_candidate_landmarks(
    boxes: object,
    keypoints: object,
    *,
    image_width: int,
    image_height: int,
    maximum_centrality: float,
    minimum_centrality_margin: float,
    tie_tolerance: float,
) -> np.ndarray:
    """Select the uniquely most-central SCRFD detection and return its five landmarks."""
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (image_width, image_height)
    ):
        raise ValueError("identity crop dimensions must be positive integers")
    threshold_values = (maximum_centrality, minimum_centrality_margin, tie_tolerance)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in threshold_values
    ):
        raise ValueError("identity crop selection thresholds must be finite numbers")
    if (
        not 0 <= maximum_centrality <= 1
        or not 0 < minimum_centrality_margin <= 1
        or not 0 < tie_tolerance < minimum_centrality_margin
    ):
        raise ValueError("identity crop selection thresholds are outside their valid ranges")
    raw_detections = np.asarray(boxes)
    raw_landmarks = np.asarray(keypoints)
    if raw_detections.dtype.kind not in {"f", "i"} or raw_landmarks.dtype.kind not in {
        "f",
        "i",
    }:
        raise ValueError("SCRFD identity-audit output must contain only numeric values")
    detections = np.asarray(raw_detections, dtype=np.float64)
    landmarks = np.asarray(raw_landmarks, dtype=np.float64)
    if detections.ndim != 2 or detections.shape[1] != 5:
        raise ValueError("SCRFD detection output has an unexpected shape")
    if landmarks.shape != (detections.shape[0], 5, 2):
        raise ValueError("SCRFD five-point landmark output has an unexpected shape")
    if not np.isfinite(detections).all() or not np.isfinite(landmarks).all():
        raise ValueError("SCRFD identity-audit output contains non-finite values")
    if detections.shape[0] == 0:
        raise FaceRedetectionUnavailableError(
            "SCRFD could not re-detect a face in the selected crop"
        )
    x0 = np.clip(detections[:, 0], 0, image_width)
    y0 = np.clip(detections[:, 1], 0, image_height)
    x1 = np.clip(detections[:, 2], 0, image_width)
    y1 = np.clip(detections[:, 3], 0, image_height)
    if np.any(x1 <= x0) or np.any(y1 <= y0):
        raise ValueError("SCRFD identity-audit boxes have invalid clipped geometry")
    if np.any(detections[:, 4] < 0) or np.any(detections[:, 4] > 1):
        raise ValueError("SCRFD identity-audit confidence is outside [0,1]")
    diagonal = math.hypot(image_width, image_height)
    centrality = np.hypot((x0 + x1) / 2 - image_width / 2, (y0 + y1) / 2 - image_height / 2)
    centrality /= diagonal
    order = np.argsort(centrality, kind="stable")
    best = int(order[0])
    if centrality[best] > maximum_centrality:
        raise FaceRedetectionUnavailableError(
            "re-detected candidate face is outside the centrality boundary"
        )
    if len(order) > 1:
        margin = float(centrality[int(order[1])] - centrality[best])
        if margin <= tie_tolerance or margin < minimum_centrality_margin:
            raise FaceRedetectionUnavailableError(
                "re-detected candidate face is tied or spatially ambiguous"
            )
    return landmarks[best].copy()


def canonical_identity_join_audit_json(payload: Mapping[str, Any]) -> str:
    """Return stable human-readable JSON after complete semantic validation."""
    validate_identity_join_audit_report(payload)
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"


def load_identity_join_audit_report(path: str | Path) -> dict[str, Any]:
    """Load a duplicate-key-safe, checksum-bound identity-join audit report."""
    source = _regular_file(path, "identity-join audit report")
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("cannot read a valid identity-join audit report") from error
    validate_identity_join_audit_report(payload)
    return dict(payload)


def validate_identity_join_audit_report(payload: Mapping[str, Any]) -> None:
    """Fail closed on schema, checksum, coverage, assignment or proposal drift."""
    if not isinstance(payload, Mapping) or set(payload) != _TOP_FIELDS:
        raise ValueError("identity-join audit report has an unexpected top-level schema")
    if payload["schema_version"] != IDENTITY_JOIN_AUDIT_SCHEMA_VERSION:
        raise ValueError("unsupported identity-join audit schema version")
    if (
        payload["status"] != "proposed_not_approved"
        or payload["mapping_approved"] is not False
        or payload["training_input_authorized"] is not False
        or payload["purpose"] != "restricted_source_session_identity_join_audit_only"
    ):
        raise ValueError("identity-join proposal boundary was weakened")
    bindings = _validate_bindings(payload["bindings"])
    method = payload["method"]
    expected_method_fields = {
        "method_version",
        "subject_interpretation",
        "crops_per_session",
        "crop_selection",
        "redetection_fallback",
        "crop_embedding",
        "embedding_dimension",
        "session_aggregation",
        "similarity",
        "similarity_decimal_places",
        "nominal_rank",
        "assignment",
        "proposal_boundary",
        "thresholds",
    }
    if not isinstance(method, Mapping) or set(method) != expected_method_fields:
        raise ValueError("identity-join audit method schema drifted")
    for field in ("crops_per_session", "embedding_dimension", "similarity_decimal_places"):
        if isinstance(method[field], bool) or not isinstance(method[field], int):
            raise ValueError(f"identity-join audit method {field} must be an integer")
    fixed_method = {
        "method_version": IDENTITY_JOIN_AUDIT_METHOD_VERSION,
        "subject_interpretation": "candidate_face_hypothesis_not_biometric_identity_truth",
        "crops_per_session": ENDPOINT_CROP_COUNT,
        "crop_selection": "twelve_endpoint_targets_with_nearest_unused_fallback",
        "crop_embedding": "scrfd_redetect_five_point_align_arcface_w600k_r50",
        "embedding_dimension": EMBEDDING_DIMENSION,
        "session_aggregation": "l2_each_then_arithmetic_mean_then_cosine_normalize",
        "similarity": "pre_to_post_cosine",
        "similarity_decimal_places": SIMILARITY_DECIMAL_PLACES,
        "nominal_rank": "one_based_descending_cosine",
        "assignment": "exact_hungarian_maximum_total_cosine",
        "proposal_boundary": "audit_evidence_only_requires_separate_approval_before_use",
    }
    if any(method[field] != value for field, value in fixed_method.items()):
        raise ValueError("identity-join audit frozen method value drifted")
    fallback = method["redetection_fallback"]
    expected_fallback = {
        "trigger": "FaceRedetectionUnavailableError_only",
        "candidate_order": "increasing_absolute_manifest_rank_offset_lower_then_higher",
        "maximum_absolute_manifest_rank_offset": MAX_REDETECTION_FALLBACK_OFFSET,
        "selected_rank_reuse": "prohibited_within_session",
        "failed_redetection_rank_cache": "deterministic_within_session",
    }
    if fallback != expected_fallback:
        raise ValueError("identity-join redetection fallback contract drifted")
    threshold_values = method["thresholds"]
    if not isinstance(threshold_values, Mapping) or set(threshold_values) != {
        "minimum_match_similarity",
        "minimum_match_margin",
        "minimum_crop_to_session_similarity",
        "tie_tolerance",
    }:
        raise ValueError("identity-join evidence threshold schema drifted")
    if any(not isinstance(threshold_values[field], float) for field in threshold_values):
        raise ValueError("identity-join report thresholds must be JSON floats")
    thresholds = EvidenceThresholds(**threshold_values)

    coverage = payload["coverage"]
    coverage_fields = {
        "expected_participant_count",
        "pre_session_count",
        "post_session_count",
        "session_count",
        "selected_crop_file_count",
        "substituted_crop_count",
        "all_face_manifest_sessions_successful",
        "complete_pre_post_coverage_verified",
        "complete_temporal_target_coverage_verified",
        "unique_selected_crops_per_session_verified",
    }
    if not isinstance(coverage, Mapping) or set(coverage) != coverage_fields:
        raise ValueError("identity-join coverage schema drifted")
    participant_count = coverage["expected_participant_count"]
    count_fields = (
        "expected_participant_count",
        "pre_session_count",
        "post_session_count",
        "session_count",
        "selected_crop_file_count",
        "substituted_crop_count",
    )
    if (
        any(
            isinstance(coverage[field], bool) or not isinstance(coverage[field], int)
            for field in count_fields
        )
        or participant_count < 2
        or coverage["pre_session_count"] != participant_count
        or coverage["post_session_count"] != participant_count
        or coverage["session_count"] != 2 * participant_count
        or coverage["selected_crop_file_count"] != 2 * participant_count * ENDPOINT_CROP_COUNT
        or not 0 <= coverage["substituted_crop_count"] <= coverage["selected_crop_file_count"]
        or coverage["all_face_manifest_sessions_successful"] is not True
        or coverage["complete_pre_post_coverage_verified"] is not True
        or coverage["complete_temporal_target_coverage_verified"] is not True
        or coverage["unique_selected_crops_per_session_verified"] is not True
    ):
        raise ValueError("identity-join report does not prove complete successful coverage")

    sessions = _validated_session_evidence_from_report(
        payload["sessions"], participant_count, thresholds
    )
    if coverage["substituted_crop_count"] != sum(row["substitution_count"] for row in sessions):
        raise ValueError("identity-join substitution coverage does not match session evidence")
    if _selected_crop_binding(sessions) != bindings["selected_crop_file_set_sha256"]:
        raise ValueError("identity-join selected crop binding does not match session evidence")

    pairwise = payload["pairwise_cosine"]
    if not isinstance(pairwise, Mapping) or set(pairwise) != {
        "pre_source_session_ids",
        "post_source_session_ids",
        "values",
        "float64_little_endian_sha256",
    }:
        raise ValueError("identity-join pairwise similarity schema drifted")
    if not isinstance(pairwise["pre_source_session_ids"], list) or not isinstance(
        pairwise["post_source_session_ids"], list
    ):
        raise ValueError("identity-join pairwise axes must be JSON arrays")
    pre_ids = tuple(pairwise["pre_source_session_ids"])
    post_ids = tuple(pairwise["post_source_session_ids"])
    _validate_ordered_sessions(pre_ids, "P")
    _validate_ordered_sessions(post_ids, "PP")
    if len(pre_ids) != participant_count or len(post_ids) != participant_count:
        raise ValueError("identity-join pairwise matrix coverage drifted")
    session_pre_ids = tuple(row["source_session_id"] for row in sessions if row["phase"] == "pre")
    session_post_ids = tuple(row["source_session_id"] for row in sessions if row["phase"] == "post")
    if pre_ids != session_pre_ids or post_ids != session_post_ids:
        raise ValueError("identity-join matrix axes do not match checksum-bound crop sessions")
    matrix = _json_numeric_matrix(pairwise["values"])
    if matrix.shape != (participant_count, participant_count) or not np.isfinite(matrix).all():
        raise ValueError("identity-join pairwise matrix is incomplete or non-finite")
    observed_matrix_sha = hashlib.sha256(
        np.asarray(matrix, dtype="<f8").tobytes(order="C")
    ).hexdigest()
    if pairwise["float64_little_endian_sha256"] != observed_matrix_sha:
        raise ValueError("identity-join pairwise matrix checksum does not match")
    derived = analyse_identity_similarity(pre_ids, post_ids, matrix, thresholds=thresholds)
    if not _same_json_value(payload["pre_to_post_evidence"], derived["pre_to_post_evidence"]):
        raise ValueError("identity-join rank evidence is inconsistent with its matrix")
    if not _same_json_value(payload["assignment"], derived["assignment"]):
        raise ValueError("identity-join assignment is inconsistent with its matrix")
    if not _same_json_value(
        payload["proposed_source_session_to_canonical_participant"],
        derived["proposed_mapping"],
    ):
        raise ValueError("identity-join proposed mapping is inconsistent with its assignment")
    if (
        not _same_json_value(payload["changed_nominal_pairs"], derived["changed_nominal_pairs"])
        or isinstance(payload["changed_nominal_pair_count"], bool)
        or not isinstance(payload["changed_nominal_pair_count"], int)
        or payload["changed_nominal_pair_count"] != len(derived["changed_nominal_pairs"])
    ):
        raise ValueError("identity-join changed-pair summary is inconsistent")

    checksum = payload["report_payload_sha256"]
    if not isinstance(checksum, str) or _SHA256_PATTERN.fullmatch(checksum) is None:
        raise ValueError("identity-join report checksum is invalid")
    unsigned = {key: value for key, value in payload.items() if key != "report_payload_sha256"}
    if _canonical_sha256(unsigned) != checksum:
        raise ValueError("identity-join report payload checksum does not match")


def _validated_session_evidence(
    values: Mapping[str, Mapping[str, Any]], thresholds: EvidenceThresholds
) -> list[dict[str, Any]]:
    ordered_ids = (*_partition_sessions(tuple(values))[0], *_partition_sessions(tuple(values))[1])
    rows = [dict(values[session_id]) for session_id in ordered_ids]
    return _validated_session_evidence_from_report(rows, len(rows) // 2, thresholds)


def _validated_session_evidence_from_report(
    values: object, participant_count: int, thresholds: EvidenceThresholds
) -> list[dict[str, Any]]:
    if not isinstance(values, list) or len(values) != 2 * participant_count:
        raise ValueError("identity-join session evidence coverage is incomplete")
    expected_fields = {
        "source_session_id",
        "phase",
        "nominal_participant_id",
        "manifest_frame_count",
        "selected_crop_count",
        "substitution_count",
        "selected_crops",
        "minimum_crop_to_session_similarity",
        "mean_crop_to_session_similarity",
    }
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in values:
        if not isinstance(row, Mapping) or set(row) != expected_fields:
            raise ValueError("identity-join session evidence schema drifted")
        session_id = row["source_session_id"]
        phase, number = _parse_session_id(session_id)
        if session_id in seen or row["phase"] != phase or row["nominal_participant_id"] != number:
            raise ValueError("identity-join session identity/phase evidence is inconsistent")
        frame_count = row["manifest_frame_count"]
        if (
            isinstance(frame_count, bool)
            or not isinstance(frame_count, int)
            or frame_count < ENDPOINT_CROP_COUNT
            or isinstance(row["selected_crop_count"], bool)
            or not isinstance(row["selected_crop_count"], int)
            or row["selected_crop_count"] != ENDPOINT_CROP_COUNT
            or isinstance(row["substitution_count"], bool)
            or not isinstance(row["substitution_count"], int)
            or not 0 <= row["substitution_count"] <= ENDPOINT_CROP_COUNT
        ):
            raise ValueError("identity-join session has insufficient selected crop evidence")
        selected = row["selected_crops"]
        if not isinstance(selected, list) or len(selected) != ENDPOINT_CROP_COUNT:
            raise ValueError("identity-join selected crop evidence must contain twelve rows")
        expected_ranks = endpoint_rank_indices(frame_count)
        selected_frame_ranks: set[int] = set()
        selected_grid_indices: set[int] = set()
        observed_substitutions = 0
        for endpoint_rank, crop in enumerate(selected):
            if not isinstance(crop, Mapping) or set(crop) != {
                "endpoint_rank",
                "target_manifest_frame_rank",
                "selected_manifest_frame_rank",
                "rank_offset",
                "grid_index",
                "crop_sha256",
            }:
                raise ValueError("identity-join selected crop row schema drifted")
            endpoint_value = crop["endpoint_rank"]
            target_rank = crop["target_manifest_frame_rank"]
            selected_rank = crop["selected_manifest_frame_rank"]
            rank_offset = crop["rank_offset"]
            grid_index = crop["grid_index"]
            if (
                isinstance(endpoint_value, bool)
                or not isinstance(endpoint_value, int)
                or endpoint_value != endpoint_rank
                or isinstance(target_rank, bool)
                or not isinstance(target_rank, int)
                or target_rank != expected_ranks[endpoint_rank]
                or isinstance(selected_rank, bool)
                or not isinstance(selected_rank, int)
                or not 0 <= selected_rank < frame_count
                or selected_rank in selected_frame_ranks
                or isinstance(rank_offset, bool)
                or not isinstance(rank_offset, int)
                or rank_offset != selected_rank - target_rank
                or abs(rank_offset) > MAX_REDETECTION_FALLBACK_OFFSET
                or isinstance(crop["grid_index"], bool)
                or not isinstance(grid_index, int)
                or grid_index < 0
                or grid_index in selected_grid_indices
                or not isinstance(crop["crop_sha256"], str)
                or _SHA256_PATTERN.fullmatch(crop["crop_sha256"]) is None
            ):
                raise ValueError("identity-join selected crop rank/hash evidence is invalid")
            selected_frame_ranks.add(selected_rank)
            selected_grid_indices.add(grid_index)
            observed_substitutions += int(rank_offset != 0)
        if (
            len(selected_frame_ranks) != ENDPOINT_CROP_COUNT
            or len(selected_grid_indices) != ENDPOINT_CROP_COUNT
            or row["substitution_count"] != observed_substitutions
        ):
            raise ValueError("identity-join substitution/uniqueness evidence is inconsistent")
        if not isinstance(row["minimum_crop_to_session_similarity"], float) or not isinstance(
            row["mean_crop_to_session_similarity"], float
        ):
            raise ValueError("identity-join crop consistency values must be JSON floats")
        minimum = _finite_float(
            row["minimum_crop_to_session_similarity"], "minimum crop similarity"
        )
        average = _finite_float(row["mean_crop_to_session_similarity"], "mean crop similarity")
        if (
            not -1 <= minimum <= average <= 1
            or minimum < thresholds.minimum_crop_to_session_similarity
        ):
            raise ValueError("identity-join crop consistency evidence is insufficient")
        validated.append(
            {
                **dict(row),
                "selected_crops": [dict(value) for value in selected],
                "minimum_crop_to_session_similarity": minimum,
                "mean_crop_to_session_similarity": average,
            }
        )
        seen.add(session_id)
    pre_ids, post_ids = _partition_sessions(tuple(seen))
    expected_order = (*pre_ids, *post_ids)
    if tuple(row["source_session_id"] for row in validated) != expected_order:
        raise ValueError("identity-join session evidence order is not canonical")
    return validated


def _partition_sessions(session_ids: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if len(session_ids) != len(set(session_ids)):
        raise ValueError("identity-join source session IDs must be unique")
    pre: list[tuple[int, str]] = []
    post: list[tuple[int, str]] = []
    for session_id in session_ids:
        phase, number = _parse_session_id(session_id)
        target = pre if phase == "pre" else post
        target.append((int(number), session_id))
    pre.sort()
    post.sort()
    pre_ids = tuple(value for _, value in pre)
    post_ids = tuple(value for _, value in post)
    if len(pre_ids) != len(post_ids) or tuple(value for value, _ in pre) != tuple(
        value for value, _ in post
    ):
        raise ValueError("identity-join source sessions lack exact nominal pre/post coverage")
    return pre_ids, post_ids


def _validate_ordered_sessions(values: Sequence[str], prefix: str) -> None:
    expected_phase = "pre" if prefix == "P" else "post"
    observed_numbers: list[int] = []
    for value in values:
        phase, number = _parse_session_id(value)
        if phase != expected_phase:
            raise ValueError("identity similarity axis contains the wrong source phase")
        observed_numbers.append(int(number))
    if len(observed_numbers) != len(set(observed_numbers)) or observed_numbers != sorted(
        observed_numbers
    ):
        raise ValueError("identity similarity axes must be unique and numeric-ID sorted")


def _parse_session_id(value: object) -> tuple[str, str]:
    if not isinstance(value, str):
        raise ValueError("source session ID must be canonical ASCII text")
    match = _SESSION_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError("source session ID must be canonical P# or PP# without leading zeros")
    return ("post" if match.group(1) == "PP" else "pre", match.group(2))


def _session_number(value: str) -> str:
    return _parse_session_id(value)[1]


def _embedding_matrix(values: Sequence[np.ndarray], label: str) -> np.ndarray:
    raw_rows = [np.asarray(value) for value in values]
    if any(row.dtype.kind not in {"f", "i"} for row in raw_rows):
        raise ValueError(f"{label} session embeddings must contain only numeric values")
    rows = [np.asarray(value, dtype=np.float64) for value in raw_rows]
    if any(row.shape != (EMBEDDING_DIMENSION,) for row in rows):
        raise ValueError(f"{label} session embeddings must be 512-dimensional")
    matrix = np.stack(rows)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label} session embeddings contain non-finite values")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError(f"{label} session embeddings must have positive finite norms")
    return matrix


def _best_alternative_assignment_total(matrix: np.ndarray, assignment: np.ndarray) -> float:
    if matrix.shape[0] < 2:
        raise ValueError("alternative assignment requires at least two participants")
    best = -math.inf
    for row, column in enumerate(assignment):
        cost = -matrix.copy()
        cost[row, int(column)] = 1e9
        alternative_rows, alternative_columns = linear_sum_assignment(cost)
        if int(alternative_columns[row]) == int(column):
            raise ValueError("cannot construct an alternative complete assignment")
        score = float(matrix[alternative_rows, alternative_columns].sum())
        best = max(best, score)
    if not math.isfinite(best):
        raise ValueError("cannot compute a finite alternative assignment score")
    return best


def _manifest_crop_path(root: Path, reference: object) -> Path:
    if not isinstance(reference, str):
        raise ValueError("manifest crop reference must be text")
    pure = PurePosixPath(reference)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError("manifest crop reference must be a portable relative path")
    candidate = root.joinpath(*pure.parts)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("manifest crop reference must resolve to a regular non-symlink file")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("manifest crop reference escapes its manifest directory")
    return resolved


def _verify_artifact(path: str | Path, expected_sha256: str, label: str) -> _VerifiedArtifact:
    expected = _validated_sha256(expected_sha256, f"{label} expected checksum")
    source = _regular_file(path, f"{label} artifact")
    observed = file_sha256(source)
    if observed != expected:
        raise ValueError(f"{label} artifact checksum does not match the supplied exact hash")
    return _VerifiedArtifact(
        path=source,
        binding={
            "file_name": source.name,
            "file_size_bytes": source.stat().st_size,
            "file_sha256": observed,
        },
    )


def _validate_bindings(value: object) -> dict[str, Any]:
    expected_fields = {
        "face_crop_manifest_schema_version",
        "face_crop_manifest_file_sha256",
        "face_crop_manifest_payload_sha256",
        "face_selector_payload_sha256",
        "detector_artifact",
        "recognition_artifact",
        "implementation_file_sha256",
        "runtime",
        "selected_crop_file_set_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError("identity-join audit binding schema drifted")
    if value["face_crop_manifest_schema_version"] != FACE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("identity-join face-manifest schema binding drifted")
    for field in (
        "face_crop_manifest_file_sha256",
        "face_crop_manifest_payload_sha256",
        "face_selector_payload_sha256",
        "selected_crop_file_set_sha256",
    ):
        _validated_sha256(value[field], f"identity-join {field}")
    detector = _validate_artifact_binding(value["detector_artifact"], "detector")
    recognition = _validate_artifact_binding(value["recognition_artifact"], "recognition")
    if detector["file_name"] != SCRFD_ARTIFACT_FILE:
        raise ValueError("identity-join detector binding must name det_10g.onnx")
    if recognition["file_name"] != ARCFACE_ARTIFACT_FILE:
        raise ValueError("identity-join recognition binding must name w600k_r50.onnx")
    implementation = _validate_implementation_hashes(value["implementation_file_sha256"])
    runtime = _validate_runtime_provenance(value["runtime"])
    return {
        "face_crop_manifest_schema_version": FACE_MANIFEST_SCHEMA_VERSION,
        "face_crop_manifest_file_sha256": value["face_crop_manifest_file_sha256"],
        "face_crop_manifest_payload_sha256": value["face_crop_manifest_payload_sha256"],
        "face_selector_payload_sha256": value["face_selector_payload_sha256"],
        "detector_artifact": detector,
        "recognition_artifact": recognition,
        "implementation_file_sha256": implementation,
        "runtime": runtime,
        "selected_crop_file_set_sha256": value["selected_crop_file_set_sha256"],
    }


def _validate_artifact_binding(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "file_name",
        "file_size_bytes",
        "file_sha256",
    }:
        raise ValueError(f"identity-join {label} artifact binding schema drifted")
    file_name = value["file_name"]
    size = value["file_size_bytes"]
    if (
        not isinstance(file_name, str)
        or Path(file_name).name != file_name
        or not file_name
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
    ):
        raise ValueError(f"identity-join {label} artifact binding is invalid")
    return {
        "file_name": file_name,
        "file_size_bytes": size,
        "file_sha256": _validated_sha256(
            value["file_sha256"], f"identity-join {label} artifact checksum"
        ),
    }


def _validate_implementation_hashes(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {
        "identity_join_audit_module",
        "identity_join_audit_cli",
    }:
        raise ValueError("identity-join implementation file bindings are incomplete")
    return {
        key: _validated_sha256(value[key], f"identity-join implementation {key}")
        for key in sorted(value)
    }


def _verify_implementation_files(
    value: Mapping[str, str | Path] | None,
) -> _VerifiedImplementation:
    repository_root = Path(__file__).resolve().parents[2]
    supplied: Mapping[str, str | Path] = value or {
        "identity_join_audit_module": Path(__file__).resolve(),
        "identity_join_audit_cli": repository_root / "scripts" / "audit_identity_joins.py",
    }
    if set(supplied) != {
        "identity_join_audit_module",
        "identity_join_audit_cli",
    }:
        raise ValueError("identity-join implementation file set is incomplete")
    paths = {
        key: _regular_file(supplied[key], f"identity-join implementation {key}")
        for key in sorted(supplied)
    }
    hashes = {key: file_sha256(path) for key, path in paths.items()}
    return _VerifiedImplementation(paths=paths, hashes=hashes)


def _validate_runtime_provenance(value: object) -> dict[str, str]:
    fields = {
        "insightface_version",
        "onnxruntime_version",
        "opencv_version",
        "numpy_version",
        "scipy_version",
        "execution_provider",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("identity-join runtime provenance schema drifted")
    result: dict[str, str] = {}
    for field in sorted(fields):
        item = value[field]
        if not isinstance(item, str) or not item or item != item.strip() or "\x00" in item:
            raise ValueError(f"identity-join runtime {field} must be clean non-empty text")
        result[field] = item
    if result["execution_provider"] != "CPUExecutionProvider":
        raise ValueError("identity-join audit must use the CPU execution provider")
    return result


def _selected_crop_binding(sessions: Sequence[Mapping[str, Any]]) -> str:
    rows = [
        {
            "source_session_id": session["source_session_id"],
            "selected_crops": session["selected_crops"],
        }
        for session in sessions
    ]
    return _canonical_sha256({"sessions": rows})


def _regular_file(value: str | Path, label: str) -> Path:
    candidate = Path(value)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return candidate.resolve()


def _validated_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be 64 lowercase hexadecimal digits")
    return value


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _same_json_value(left: object, right: object) -> bool:
    return canonical_json_bytes({"value": left}) == canonical_json_bytes({"value": right})


def _rounded(values: np.ndarray) -> np.ndarray:
    result = np.round(np.asarray(values, dtype=np.float64), SIMILARITY_DECIMAL_PLACES)
    result[result == 0] = 0.0
    return result


def _round_float(value: float | np.floating[Any]) -> float:
    result = round(float(value), SIMILARITY_DECIMAL_PLACES)
    return 0.0 if result == 0 else result


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _json_numeric_matrix(value: object) -> np.ndarray:
    if not isinstance(value, list) or not value:
        raise ValueError("identity-join similarity values must be a non-empty matrix")
    for row in value:
        if not isinstance(row, list) or any(
            isinstance(item, bool) or not isinstance(item, float) for item in row
        ):
            raise ValueError("identity-join similarity values must be canonical finite JSON floats")
    try:
        return np.asarray(value, dtype=np.float64)
    except ValueError as error:
        raise ValueError("identity-join similarity matrix must be rectangular") from error


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _onnx_session(onnxruntime: Any, model_path: Path) -> Any:
    options = onnxruntime.SessionOptions()
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.use_deterministic_compute = True
    return onnxruntime.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def _scipy_version() -> str:
    import scipy

    return str(scipy.__version__)
