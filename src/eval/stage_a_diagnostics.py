"""Annotation-free diagnostics for a checksum-validated Stage-A release.

The module is deliberately post-lock and read-only.  It reuses the shared
Stage-A and feature-closure validators, never opens protected annotations, and
publishes only aggregate counts, distribution summaries and neutral proxy-group
comparisons.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.data import load_verified_session_mapping
from src.eval.metrics import empirical_wasserstein1
from src.features.closure import load_verified_feature_extraction_closure
from src.secure_io import AnchoredDirectory
from src.training.crossfit import canonical_json_bytes
from src.training.proxy_artifacts import PROXY_ARTIFACT_SCHEMA_VERSION
from src.training.proxy_training import FROZEN_FINAL_SEEDS, FROZEN_OUTER_ROTATIONS

from .stage_c import StageARelease, load_stage_a_release

JSON_FILENAME = "stage_a_diagnostics_v2.json"
MARKDOWN_FILENAME = "STAGE_A_DIAGNOSTICS_V2.md"
DIAGNOSTICS_SCHEMA_VERSION = "fair-avi-mit-stage-a-diagnostics-2"

_PROXY_CONFIGURATIONS = {
    "visual": ("face_sampling_only", "face_w1"),
    "interview_audio": ("audio_session_sampling_only", "audio_session_w1"),
    "random": ("random_proxy_w1",),
}
_REPRESENTATIVE_CONFIGURATION = {
    source: configurations[-1] for source, configurations in _PROXY_CONFIGURATIONS.items()
}
_PREDICTION_COMPARISONS = {
    "visual": (
        ("erm", "reference_erm"),
        ("face_sampling_only", "matched_sampling_only"),
        ("face_w1", "w1_intervention"),
    ),
    "interview_audio": (
        ("erm", "reference_erm"),
        ("audio_session_sampling_only", "matched_sampling_only"),
        ("audio_session_w1", "w1_intervention"),
    ),
    "random": (
        ("erm", "reference_erm"),
        ("random_proxy_w1", "w1_intervention"),
    ),
}
_PROHIBITED_PUBLIC_KEYS = {
    "candidate_id",
    "clip_id",
    "participant_id",
    "session_id",
    "source_session_id",
    "eligible_participants",
    "input_sessions",
    "truncated_post_session_ids",
    "prediction_lock_id",
    "model_fit_sha256",
}
_PROHIBITED_PUBLIC_KEY_SUFFIXES = (
    "_path",
    "_paths",
    "_reference",
    "_references",
)
_SOURCE_BINDING_KEYS = frozenset(
    {
        "stage_a_closure_file_sha256",
        "stage_a_closure_payload_sha256",
        "stage_a_context_payload_sha256",
        "feature_closure_file_sha256",
        "feature_closure_payload_sha256",
        "session_mapping_file_sha256",
        "face_crop_manifest_file_sha256",
        "face_crop_qc_file_sha256",
        "inventory_file_sha256",
        "eligibility_file_sha256",
        "eligibility_payload_sha256",
        "diagnostics_source_file_sha256",
    }
)
_PINNED_INPUT_BINDING_KEYS = frozenset(
    {
        "face_crop_qc_file_sha256",
        "inventory_file_sha256",
        "eligibility_file_sha256",
    }
)
_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "scope",
        "execution_mode",
        "source_bindings",
        "participant_count",
        "session_count",
        "protected_annotation_fields_accessed",
        "proxy_diagnostics",
        "boundary_and_representation_quality",
        "deferred",
        "interpretation",
        "diagnostics_payload_sha256",
    }
)
_SAFE_PUBLIC_FAILURE_CODES = frozenset(
    {
        "audio_active_interval_unavailable",
        "audio_decode_failed",
        "audio_invalid",
        "audio_missing",
        "audio_source_alignment_invalid",
        "audio_source_timing_unavailable",
        "audio_too_short",
        "code_provenance_unavailable",
        "encoder_shape_invalid",
        "encoder_values_invalid",
        "face_active_start_missing",
        "face_candidate_ambiguous",
        "face_candidate_incomplete",
        "face_candidate_insufficient",
        "face_candidate_off_centre",
        "face_crop_failed",
        "face_crop_invalid",
        "face_crop_manifest_mutated",
        "face_crop_manifest_unavailable",
        "face_crop_missing",
        "face_crop_mutated_during_extraction",
        "face_crop_path_invalid",
        "face_crop_publish_collision",
        "face_crop_too_small",
        "face_detector_artifact_hash_mismatch",
        "face_detector_artifact_unavailable",
        "face_detector_output_invalid",
        "face_detector_runtime_unavailable",
        "face_frame_invalid",
        "face_media_missing",
        "face_not_detected",
        "face_video_decode_failed",
        "face_video_not_cfr",
        "face_video_probe_failed",
        "input_revalidation_failed",
        "media_missing",
        "media_reference_unsafe",
        "media_runtime_unavailable",
        "reference_unsafe",
        "schema_invalid",
        "text_shape_invalid",
        "transcript_empty",
        "transcript_load_failed",
        "transcript_missing",
        "transcript_provenance_invalid",
        "transcript_reference_unsafe",
        "unexpected_error",
        "video_duration_probe_failed",
        "video_shape_invalid",
    }
)
_SAFE_PUBLIC_STRING_VALUES = frozenset(
    {
        DIAGNOSTICS_SCHEMA_VERSION,
        JSON_FILENAME,
        MARKDOWN_FILENAME,
        "annotation_free_post_lock_stage_a_diagnostics",
        "research_read_only",
        "reported_without_protected_annotations",
        "canonical_cluster_or_random_labels_only_not_demographic_identity",
        "visual",
        "interview_audio",
        "random",
        "proxy_group_0",
        "proxy_group_1",
        "train",
        "validation",
        "test",
        "pre",
        "post",
        "locked_target",
        "oof_proxy_target_association",
        "erm",
        "reference_erm",
        "face_sampling_only",
        "audio_session_sampling_only",
        "matched_sampling_only",
        "face_w1",
        "audio_session_w1",
        "random_proxy_w1",
        "w1_intervention",
        "success",
        "failure",
        "terminal_invalid_run",
        "end_of_stream",
        "video",
        "candidate_text",
        "forbidden",
        "defective_legacy_derived_features_inventory_only",
        "legacy_shore_truncation_inventory_only",
        "malformed_legacy_schema_inventory_only",
        "deferred_not_computed",
        "deferred_until_a_separate_predeclared_analysis",
        "not_accessed_or_required",
        (
            "proxy_groups_are_neutral_model_diagnostics_not_demographic_"
            "identities_or_protected_attribute_inference"
        ),
        "published",
        *_SAFE_PUBLIC_FAILURE_CODES,
    }
)
_SAFE_PUBLIC_KEY = re.compile(r"[a-z][a-z0-9_]*", re.ASCII)
_IDENTIFIER_LIKE_VALUE = re.compile(
    r"(?:p|participant|session|source_session|clip|candidate)[-_]?[a-z0-9.-]+",
    re.IGNORECASE | re.ASCII,
)
_UUID_VALUE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
    re.IGNORECASE | re.ASCII,
)
_IMPORTED_DIAGNOSTICS_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_PUBLICATION_ISSUER = object()


@dataclass(frozen=True, slots=True)
class _PublicationAuthorization:
    """In-process evidence that the research builder issued these exact bytes."""

    issuer: object
    source_root: Path
    participant_count: int
    session_count: int
    diagnostics_payload_sha256: str
    source_bindings: tuple[tuple[str, str], ...]
    pinned_inputs: tuple[tuple[str, Path, str], ...]
    json_sha256: str
    markdown_sha256: str


@dataclass(frozen=True, slots=True)
class StageADiagnostics:
    """Deterministic aggregate-only diagnostics ready for safe publication."""

    source_root: Path
    pinned_input_paths: tuple[tuple[str, Path], ...]
    payload: Mapping[str, Any]
    json_bytes: bytes
    markdown_bytes: bytes
    _publication_authorization: _PublicationAuthorization | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True, slots=True)
class PublishedStageADiagnostics:
    """Checksums of safely published diagnostics files."""

    json_sha256: str
    markdown_sha256: str


def build_stage_a_diagnostics(
    stage_a_dir: str | Path,
    feature_dir: str | Path,
    *,
    expected_stage_a_closure_file_sha256: str,
    expected_stage_a_closure_payload_sha256: str,
    expected_face_crop_qc_file_sha256: str,
    expected_inventory_file_sha256: str,
    expected_eligibility_file_sha256: str,
    session_mapping_path: str | Path,
    identity_join_audit_path: str | Path,
    face_crop_manifest_path: str | Path,
    face_crop_qc_path: str | Path,
    inventory_path: str | Path,
    eligibility_path: str | Path,
) -> StageADiagnostics:
    """Validate locked evidence and build annotation-free aggregate diagnostics."""
    _require_sha256(
        expected_stage_a_closure_file_sha256,
        "expected Stage-A closure file SHA-256",
    )
    _require_sha256(
        expected_stage_a_closure_payload_sha256,
        "expected Stage-A closure payload SHA-256",
    )
    _require_sha256(
        expected_face_crop_qc_file_sha256,
        "expected face-crop QC file SHA-256",
    )
    _require_sha256(
        expected_inventory_file_sha256,
        "expected inventory file SHA-256",
    )
    _require_sha256(
        expected_eligibility_file_sha256,
        "expected eligibility file SHA-256",
    )
    release = load_stage_a_release(stage_a_dir)
    if (
        release.closure_file_sha256 != expected_stage_a_closure_file_sha256
        or release.closure_payload_sha256 != expected_stage_a_closure_payload_sha256
    ):
        raise ValueError("Stage-A closure does not match the externally recorded checksums")
    context = release.closure.get("context")
    if not isinstance(context, Mapping):
        raise ValueError("Stage-A closure lacks its bound context")
    if release.closure.get("protected_annotation_fields_accessed") != []:
        raise ValueError("Stage-A closure unexpectedly accessed protected annotations")

    session_mapping = load_verified_session_mapping(
        session_mapping_path,
        identity_join_audit_report_path=identity_join_audit_path,
        face_crop_manifest_path=face_crop_manifest_path,
    )
    if session_mapping.provenance != context.get("session_mapping"):
        raise ValueError("current session mapping does not match the Stage-A closure")

    feature_file_sha256 = _require_sha256(
        context.get("feature_extraction_closure_file_sha256"),
        "Stage-A feature closure file SHA-256",
    )
    feature_payload_sha256 = _require_sha256(
        context.get("feature_extraction_closure_payload_sha256"),
        "Stage-A feature closure payload SHA-256",
    )
    face_qc_source = _validated_source_path(face_crop_qc_path, "face-crop QC")
    inventory_source = _validated_source_path(inventory_path, "dataset inventory")
    eligibility_source = _validated_source_path(eligibility_path, "eligibility manifest")
    face_qc, face_qc_sha256 = _read_json_object(
        face_qc_source,
        "face-crop QC",
        expected_file_sha256=expected_face_crop_qc_file_sha256,
    )
    inventory, inventory_sha256 = _read_json_object(
        inventory_source,
        "dataset inventory",
        expected_file_sha256=expected_inventory_file_sha256,
    )
    eligibility, eligibility_file_sha256 = _read_json_object(
        eligibility_source,
        "eligibility manifest",
        expected_file_sha256=expected_eligibility_file_sha256,
    )
    _validate_eligibility(release, eligibility, context)
    _validate_inventory_binding(inventory, context)
    if (
        face_qc.get("face_crop_manifest_sha256")
        != session_mapping.provenance["face_crop_manifest_file_sha256"]
    ):
        raise ValueError("face-crop QC is not bound to the approved face manifest")

    with load_verified_feature_extraction_closure(
        feature_dir,
        closure_file_sha256=feature_file_sha256,
        closure_payload_sha256=feature_payload_sha256,
        session_mapping=session_mapping,
    ) as feature_closure:
        feature_closure.verify_current_files()
        boundary = _boundary_diagnostics(
            release,
            feature_closure,
            face_qc=face_qc,
            inventory=inventory,
            eligibility=eligibility,
        )
        feature_closure.verify_current_files()
    session_mapping.verify_current_files()

    proxy = _proxy_diagnostics(release)
    source_bindings = {
        "stage_a_closure_file_sha256": release.closure_file_sha256,
        "stage_a_closure_payload_sha256": release.closure_payload_sha256,
        "stage_a_context_payload_sha256": _require_sha256(
            release.closure.get("context_payload_sha256"),
            "Stage-A context payload SHA-256",
        ),
        "feature_closure_file_sha256": feature_file_sha256,
        "feature_closure_payload_sha256": feature_payload_sha256,
        "session_mapping_file_sha256": session_mapping.artifact_file_sha256,
        "face_crop_manifest_file_sha256": session_mapping.face_crop_manifest_file_sha256,
        "face_crop_qc_file_sha256": face_qc_sha256,
        "inventory_file_sha256": inventory_sha256,
        "eligibility_file_sha256": eligibility_file_sha256,
        "eligibility_payload_sha256": str(eligibility["manifest_payload_sha256"]),
        "diagnostics_source_file_sha256": _diagnostics_source_sha256(),
    }
    unsigned = {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "scope": "annotation_free_post_lock_stage_a_diagnostics",
        "execution_mode": "research_read_only",
        "source_bindings": source_bindings,
        "participant_count": release.participant_count,
        "session_count": len(release.session_keys),
        "protected_annotation_fields_accessed": [],
        "proxy_diagnostics": proxy,
        "boundary_and_representation_quality": boundary,
        "deferred": {
            "legacy_quality_error_association": ("deferred_until_a_separate_predeclared_analysis"),
            "protected_attribute_analysis": "not_accessed_or_required",
        },
        "interpretation": (
            "proxy_groups_are_neutral_model_diagnostics_not_demographic_"
            "identities_or_protected_attribute_inference"
        ),
    }
    payload = {**unsigned, "diagnostics_payload_sha256": _payload_sha256(unsigned)}
    _assert_aggregate_only(payload)
    json_bytes = canonical_json_bytes(payload)
    markdown_bytes = _markdown(payload)
    pinned_input_paths = (
        ("face_crop_qc_file_sha256", face_qc_source),
        ("inventory_file_sha256", inventory_source),
        ("eligibility_file_sha256", eligibility_source),
    )
    report = StageADiagnostics(
        source_root=release.root,
        pinned_input_paths=pinned_input_paths,
        payload=payload,
        json_bytes=json_bytes,
        markdown_bytes=markdown_bytes,
    )
    object.__setattr__(
        report,
        "_publication_authorization",
        _PublicationAuthorization(
            issuer=_PUBLICATION_ISSUER,
            source_root=release.root.resolve(strict=False),
            participant_count=release.participant_count,
            session_count=len(release.session_keys),
            diagnostics_payload_sha256=str(payload["diagnostics_payload_sha256"]),
            source_bindings=tuple(sorted(source_bindings.items())),
            pinned_inputs=tuple(
                (
                    name,
                    path.resolve(strict=False),
                    str(source_bindings[name]),
                )
                for name, path in pinned_input_paths
            ),
            json_sha256=hashlib.sha256(json_bytes).hexdigest(),
            markdown_sha256=hashlib.sha256(markdown_bytes).hexdigest(),
        ),
    )
    return report


def publish_stage_a_diagnostics(
    report: StageADiagnostics, output_dir: str | Path
) -> PublishedStageADiagnostics:
    """Publish separate files atomically, idempotently and without replacement."""
    _validate_report_for_publication(report)
    output = _validated_separate_output(report.source_root, output_dir)
    contents = {
        JSON_FILENAME: report.json_bytes,
        MARKDOWN_FILENAME: report.markdown_bytes,
    }
    with AnchoredDirectory.open(output, create=True, directory_mode=0o755) as anchor:
        with anchor.exclusive_lock():
            names = set(anchor.list_names())
            for filename, content in contents.items():
                if filename not in names:
                    continue
                existing = anchor.read_bytes(filename, expected_mode=0o644)
                if existing.content != content:
                    raise FileExistsError(
                        f"refusing to replace non-identical diagnostics {filename!r}"
                    )
            written = {
                filename: anchor.publish_bytes(filename, content, mode=0o644)
                for filename, content in contents.items()
            }
            for filename, observed in written.items():
                anchor.verify_file(filename, observed, expected_mode=0o644)
    return PublishedStageADiagnostics(
        json_sha256=written[JSON_FILENAME].sha256,
        markdown_sha256=written[MARKDOWN_FILENAME].sha256,
    )


def publication_summary(report: StageADiagnostics, published: PublishedStageADiagnostics) -> bytes:
    """Return a canonical aggregate-only CLI receipt."""
    _require_sha256(published.json_sha256, "published diagnostics JSON SHA-256")
    _require_sha256(published.markdown_sha256, "published diagnostics Markdown SHA-256")
    payload = {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "status": "published",
        "diagnostics_payload_sha256": report.payload["diagnostics_payload_sha256"],
        "json": {"filename": JSON_FILENAME, "sha256": published.json_sha256},
        "markdown": {
            "filename": MARKDOWN_FILENAME,
            "sha256": published.markdown_sha256,
        },
        "protected_annotation_fields_accessed": [],
    }
    _assert_aggregate_only(payload)
    return canonical_json_bytes(payload) + b"\n"


def _validate_report_for_publication(report: StageADiagnostics) -> None:
    """Revalidate the complete public contract before any filesystem mutation."""
    if not isinstance(report, StageADiagnostics):
        raise TypeError("diagnostics publication requires a StageADiagnostics report")
    authorization = report._publication_authorization
    if authorization is None or authorization.issuer is not _PUBLICATION_ISSUER:
        raise ValueError("only builder-issued research diagnostics may be published")
    if not isinstance(report.json_bytes, bytes) or not isinstance(report.markdown_bytes, bytes):
        raise ValueError("diagnostics report contents must be immutable bytes")
    parsed = _canonical_object(report.json_bytes, "public diagnostics")
    try:
        payload_bytes = canonical_json_bytes(report.payload)
    except (TypeError, ValueError) as error:
        raise ValueError("diagnostics payload is not canonical-JSON compatible") from error
    if payload_bytes != report.json_bytes or parsed != report.payload:
        raise ValueError("diagnostics payload and JSON bytes disagree")
    if set(parsed) != _REPORT_KEYS:
        raise ValueError("public diagnostics top-level schema drifted")
    _validate_signed_payload(parsed, "diagnostics_payload_sha256", "public diagnostics")
    if (
        parsed.get("schema_version") != DIAGNOSTICS_SCHEMA_VERSION
        or parsed.get("scope") != "annotation_free_post_lock_stage_a_diagnostics"
        or parsed.get("execution_mode") != "research_read_only"
        or parsed.get("protected_annotation_fields_accessed") != []
    ):
        raise ValueError("public diagnostics research scope or annotation firewall drifted")
    _positive_int(parsed.get("participant_count"), "public participant count")
    _positive_int(parsed.get("session_count"), "public session count")
    bindings = _mapping(parsed.get("source_bindings"), "public source bindings")
    if set(bindings) != _SOURCE_BINDING_KEYS:
        raise ValueError("public diagnostics source-binding schema drifted")
    for name, digest in bindings.items():
        _require_sha256(digest, f"public {name}")
    if bindings["diagnostics_source_file_sha256"] != _diagnostics_source_sha256():
        raise ValueError("public diagnostics are not bound to the current evaluator source")

    release = load_stage_a_release(report.source_root)
    context = _mapping(release.closure.get("context"), "current Stage-A context")
    if (
        release.root.resolve() != Path(report.source_root).resolve()
        or release.closure_file_sha256 != bindings["stage_a_closure_file_sha256"]
        or release.closure_payload_sha256 != bindings["stage_a_closure_payload_sha256"]
        or release.closure.get("context_payload_sha256")
        != bindings["stage_a_context_payload_sha256"]
        or context.get("feature_extraction_closure_file_sha256")
        != bindings["feature_closure_file_sha256"]
        or context.get("feature_extraction_closure_payload_sha256")
        != bindings["feature_closure_payload_sha256"]
        or release.closure.get("protected_annotation_fields_accessed") != []
    ):
        raise ValueError("public diagnostics no longer match the current Stage-A closure")

    path_pairs = report.pinned_input_paths
    if not isinstance(path_pairs, tuple) or any(
        not isinstance(pair, tuple)
        or len(pair) != 2
        or not isinstance(pair[0], str)
        or not isinstance(pair[1], Path)
        for pair in path_pairs
    ):
        raise ValueError("diagnostics pinned-input path registry is invalid")
    pinned_paths = dict(path_pairs)
    if len(pinned_paths) != len(path_pairs) or set(pinned_paths) != _PINNED_INPUT_BINDING_KEYS:
        raise ValueError("diagnostics pinned-input path registry drifted")
    for binding_name, label in (
        ("face_crop_qc_file_sha256", "face-crop QC"),
        ("inventory_file_sha256", "dataset inventory"),
        ("eligibility_file_sha256", "eligibility manifest"),
    ):
        _read_json_object(
            pinned_paths[binding_name],
            label,
            expected_file_sha256=str(bindings[binding_name]),
        )

    _assert_aggregate_only(parsed)
    try:
        expected_markdown = _markdown(parsed)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("public diagnostics Markdown source schema is invalid") from error
    if report.markdown_bytes != expected_markdown:
        raise ValueError("diagnostics payload and Markdown bytes disagree")
    observed_json_sha256 = hashlib.sha256(report.json_bytes).hexdigest()
    observed_markdown_sha256 = hashlib.sha256(report.markdown_bytes).hexdigest()
    authorized_pinned_inputs = tuple(
        (
            name,
            path.resolve(strict=False),
            str(bindings[name]),
        )
        for name, path in path_pairs
    )
    if (
        authorization.source_root != Path(report.source_root).resolve(strict=False)
        or authorization.participant_count != parsed["participant_count"]
        or authorization.session_count != parsed["session_count"]
        or authorization.diagnostics_payload_sha256 != parsed["diagnostics_payload_sha256"]
        or authorization.source_bindings != tuple(sorted(bindings.items()))
        or authorization.pinned_inputs != authorized_pinned_inputs
        or authorization.json_sha256 != observed_json_sha256
        or authorization.markdown_sha256 != observed_markdown_sha256
    ):
        raise ValueError("diagnostics publication no longer matches its builder authorization")


def _align_fold_proxy_assignments(
    *,
    assignments: Mapping[tuple[str, int, str], int],
    participant_folds: Mapping[str, int],
    source: str,
    reference_rotation: int,
    rotations: Sequence[int],
) -> dict[str, int]:
    """Align fold-local binary proxy IDs using shared training participants only.

    The reference rotation retains its canonical IDs. Each other rotation is
    mapped to the reference by the permutation that maximises agreement on
    participants who are training-role members in both rotations.

    No protected annotations, outcomes, predictions or audit labels are used.
    For an exact tie between the identity and swapped permutation, identity is
    retained deterministically. Each participant receives only the aligned
    assignment from their own held-out test rotation.
    """

    if reference_rotation not in rotations:
        raise ValueError("reference rotation is not present")
    if not set(participant_folds.values()).issubset(rotations):
        raise ValueError("participant test rotation is not present")

    participant_ids = tuple(participant_folds)

    def role(participant: str, rotation: int) -> str:
        return _rotation_role(participant_folds[participant], rotation)

    # The reference rotation defines the arbitrary common ID convention.
    swaps = {reference_rotation: False}

    for rotation in rotations:
        if rotation == reference_rotation:
            continue

        shared_training = [
            participant
            for participant in participant_ids
            if role(participant, reference_rotation) == "train"
            and role(participant, rotation) == "train"
        ]

        if not shared_training:
            raise ValueError(
                "proxy rotations have no shared training participants for alignment"
            )

        identity_agreement = sum(
            assignments[(source, reference_rotation, participant)]
            == assignments[(source, rotation, participant)]
            for participant in shared_training
        )

        swapped_agreement = sum(
            assignments[(source, reference_rotation, participant)]
            != assignments[(source, rotation, participant)]
            for participant in shared_training
        )

        # Exact tie: retain the rotation's existing canonical IDs.
        swaps[rotation] = swapped_agreement > identity_agreement

    return {
        participant: (
            1 - assignments[(source, participant_folds[participant], participant)]
            if swaps[participant_folds[participant]]
            else assignments[(source, participant_folds[participant], participant)]
        )
        for participant in participant_ids
    }


def _proxy_diagnostics(release: StageARelease) -> dict[str, Any]:
    assignments: dict[tuple[str, int, str], int] = {}
    bindings: list[dict[str, Any]] = []
    separations: list[dict[str, Any]] = []
    with AnchoredDirectory.open(release.root, directory_mode=0o700) as anchor:
        for source, representative in _REPRESENTATIVE_CONFIGURATION.items():
            for rotation in FROZEN_OUTER_ROTATIONS:
                filename = f"{source}.fold-{rotation}.proxy.json"
                manifest = release.lock_manifests[(representative, FROZEN_FINAL_SEEDS[0])]
                record = _artifact_record(manifest, filename)
                observed = anchor.read_bytes(filename, expected_mode=0o600)
                if observed.sha256 != record.get("sha256"):
                    raise ValueError("Stage-A proxy artifact checksum failed")
                payload = _canonical_object(observed.content, "proxy manifest")
                _validate_signed_payload(payload, "manifest_payload_sha256", "proxy")
                _validate_proxy_identity(release, payload, source, rotation)
                rows = payload.get("participants")
                if not isinstance(rows, list) or len(rows) != release.participant_count:
                    raise ValueError("proxy participant coverage drifted")
                seen: set[str] = set()
                for row in rows:
                    participant = row.get("participant_id") if isinstance(row, Mapping) else None
                    group = row.get("proxy_group") if isinstance(row, Mapping) else None
                    if (
                        participant not in release.participant_folds
                        or participant in seen
                        or isinstance(group, bool)
                        or group not in {0, 1}
                        or row.get("fixed_fold") != release.participant_folds[participant]
                        or row.get("rotation_role")
                        != _rotation_role(release.participant_folds[participant], rotation)
                    ):
                        raise ValueError("proxy assignment or fold binding drifted")
                    seen.add(str(participant))
                    assignments[(source, rotation, str(participant))] = int(group)
                if seen != set(release.participant_ids):
                    raise ValueError("proxy assignments do not exactly cover Stage A")

                peer_hashes: list[str] = []
                for seed in FROZEN_FINAL_SEEDS:
                    for configuration in _PROXY_CONFIGURATIONS[source]:
                        peer = _artifact_record(
                            release.lock_manifests[(configuration, seed)], filename
                        )
                        peer_hashes.append(str(peer.get("sha256")))
                if set(peer_hashes) != {observed.sha256}:
                    raise ValueError("proxy artifact changed across configurations or seeds")
                bindings.append(
                    {
                        "source": source,
                        "outer_rotation": rotation,
                        "file_sha256": observed.sha256,
                        "payload_sha256": payload["manifest_payload_sha256"],
                        "frozen_seed_count": len(FROZEN_FINAL_SEEDS),
                        "configuration_count": len(_PROXY_CONFIGURATIONS[source]),
                        "distinct_file_sha256_count": len(set(peer_hashes)),
                        "identical_across_frozen_seeds_and_configurations": True,
                    }
                )
                if source != "random":
                    separations.append(_centre_separation(payload, source, rotation))
        anchor.verify_path_identity()

    counts = [
        {
            "source": source,
            "outer_rotation": rotation,
            "group_ids": ["proxy_group_0", "proxy_group_1"],
            "all": _group_counts(release, assignments, source, rotation, None),
            "roles": [
                {
                    "role": role,
                    **_group_counts(release, assignments, source, rotation, role),
                }
                for role in ("train", "validation", "test")
            ],
        }
        for source in _PROXY_CONFIGURATIONS
        for rotation in FROZEN_OUTER_ROTATIONS
    ]
    stability = [
        {
            "source": source,
            "rotation_a": left,
            "rotation_b": right,
            **_agreement(
                [assignments[(source, left, p)] for p in release.participant_ids],
                [assignments[(source, right, p)] for p in release.participant_ids],
            ),
        }
        for source in _PROXY_CONFIGURATIONS
        for left in FROZEN_OUTER_ROTATIONS
        for right in FROZEN_OUTER_ROTATIONS
        if right > left
    ]
    visual_audio = [
        {
            "outer_rotation": rotation,
            **_agreement(
                [assignments[("visual", rotation, p)] for p in release.participant_ids],
                [assignments[("interview_audio", rotation, p)] for p in release.participant_ids],
            ),
        }
        for rotation in FROZEN_OUTER_ROTATIONS
    ]
    oof = {
        source: _align_fold_proxy_assignments(
            assignments=assignments,
            participant_folds=release.participant_folds,
            source=source,
            reference_rotation=FROZEN_OUTER_ROTATIONS[0],
            rotations=FROZEN_OUTER_ROTATIONS,
        )
        for source in _PROXY_CONFIGURATIONS
    }
    reference_rows = release.learned_rows[("erm", FROZEN_FINAL_SEEDS[0])]
    target_w1 = [
        _score_w1_row(
            source=source,
            configuration="locked_target",
            comparison_role="oof_proxy_target_association",
            seed=None,
            rows=reference_rows,
            groups=oof[source],
            value_field="target",
        )
        for source in _PROXY_CONFIGURATIONS
    ]
    prediction_w1 = [
        _score_w1_row(
            source=source,
            configuration=configuration,
            comparison_role=role,
            seed=seed,
            rows=release.learned_rows[(configuration, seed)],
            groups=oof[source],
            value_field="prediction",
        )
        for source, comparisons in _PREDICTION_COMPARISONS.items()
        for configuration, role in comparisons
        for seed in FROZEN_FINAL_SEEDS
    ]
    return {
        "status": "reported_without_protected_annotations",
        "neutral_group_semantics": (
            "canonical_cluster_or_random_labels_only_not_demographic_identity"
        ),
        "artifact_and_seed_identity": bindings,
        "balance_by_rotation_and_role": counts,
        "standardised_centre_separation": separations,
        "cross_rotation_stability": stability,
        "visual_audio_agreement_by_rotation": visual_audio,
        "oof_visual_audio_agreement": _agreement(
            [oof["visual"][p] for p in release.participant_ids],
            [oof["interview_audio"][p] for p in release.participant_ids],
        ),
        "oof_proxy_target_w1": target_w1,
        "oof_proxy_conditioned_prediction_w1": prediction_w1,
    }


def _boundary_diagnostics(
    release: StageARelease,
    feature_closure: Any,
    *,
    face_qc: Mapping[str, Any],
    inventory: Mapping[str, Any],
    eligibility: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = feature_closure.manifest
    if (
        manifest.get("outcome_fields_accessed") != []
        or manifest.get("protected_annotation_fields_accessed") != []
    ):
        raise ValueError("feature closure violated the outcome/annotation firewall")
    records = manifest.get("records")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("feature closure lacks its record registry")
    feature_sessions = {str(record.get("session_id")) for record in records}
    feature_source_sessions = {str(record.get("source_session_id")) for record in records}
    expected_sessions = {session for _, _, session in release.session_keys}
    if feature_sessions != expected_sessions:
        raise ValueError("feature closure and Stage-A session rosters disagree")
    if len(feature_source_sessions) != len(records):
        raise ValueError("feature closure source-session roster is not one-to-one")

    status_counts: Counter[str] = Counter()
    feature_failure_codes: Counter[str] = Counter()
    close_reasons: Counter[str] = Counter()
    modality_status = {name: Counter() for name in ("video", "audio", "text")}
    values: dict[str, list[float]] = {
        "active_interval_seconds": [],
        "selected_face_samples": [],
        "valid_candidate_face_seconds": [],
        "audio_active_frames": [],
        "audio_window_count": [],
        "audio_end_shortfall_seconds": [],
        "text_chunk_count": [],
        "video_summary_l2_norm": [],
        "audio_summary_l2_norm": [],
        "text_summary_l2_norm": [],
        "video_valid_element_count": [],
        "audio_valid_element_count": [],
        "text_valid_element_count": [],
        "candidate_text_token_count": [],
    }
    face_cap = 0
    audio_window_cap: int | None = None
    audio_candidate_windows_below_cap = 0
    audio_candidate_windows_equal_to_cap = 0
    audio_candidate_windows_above_cap = 0
    audio_selected_windows_equal_to_cap = 0
    audio_window_cap_truncated = 0
    complete_audio = audio_eof_shortfall = 0
    successful_face_counts: dict[str, tuple[int, int]] = {}
    zero_summary_counts: Counter[str] = Counter()
    for record in records:
        state = str(record.get("state"))
        status_counts[state] += 1
        if state != "success":
            feature_failure_codes[str(record.get("failure_code") or "unspecified")] += 1
            continue
        artifacts = feature_closure.read_record_artifacts(record)
        sidecar = artifacts.metadata_payload
        if artifacts.shard is None:
            raise ValueError("successful feature record lacks its closure-bound shard")
        quality = _representation_quality(artifacts.shard.content)
        for modality in ("video", "audio", "text"):
            norm = quality[f"{modality}_summary_l2_norm"]
            values[f"{modality}_summary_l2_norm"].append(norm)
            values[f"{modality}_valid_element_count"].append(
                quality[f"{modality}_valid_element_count"]
            )
            zero_summary_counts[modality] += int(norm == 0.0)
        values["candidate_text_token_count"].append(quality["candidate_text_token_count"])
        modalities = sidecar.get("modality")
        arrays = sidecar.get("arrays")
        if not isinstance(modalities, Mapping) or not isinstance(arrays, Mapping):
            raise ValueError("successful feature sidecar lacks modality/array evidence")
        for name in modality_status:
            detail = modalities.get(name)
            if not isinstance(detail, Mapping):
                raise ValueError("successful feature sidecar lacks modality status")
            modality_status[name][str(detail.get("status"))] += 1
        video = modalities["video"]
        audio = modalities["audio"]
        crop = video.get("crop_provider")
        if not isinstance(crop, Mapping):
            raise ValueError("successful feature sidecar lacks crop boundary evidence")
        selected = _shape_length(arrays, "video")
        valid = _sequence_length(crop.get("valid_candidate_grid_indices"), "valid faces")
        start = _finite_number(crop.get("active_start_grid_index"), "active start")
        end = _finite_number(crop.get("active_end_grid_index_exclusive"), "active end")
        maximum = _positive_int(crop.get("maximum_retained_samples"), "face cap")
        values["active_interval_seconds"].append(end - start)
        values["selected_face_samples"].append(float(selected))
        values["valid_candidate_face_seconds"].append(float(valid))
        close_reasons[str(crop.get("active_close_reason"))] += 1
        face_cap += int(selected == maximum)
        successful_face_counts[str(record.get("source_session_id"))] = (selected, valid)

        audio_frames = _shape_length(arrays, "audio")
        windows = _positive_int(audio.get("window_count_active"), "audio windows")
        audio_max = _positive_int(sidecar["preprocessing"]["audio"]["maximum_windows"], "audio cap")
        selected_windows = _sequence_length(audio.get("selected_windows"), "selected audio windows")
        if audio_window_cap is None:
            audio_window_cap = audio_max
        elif audio_window_cap != audio_max:
            raise ValueError("audio maximum-window contract drifted across feature sidecars")
        if selected_windows != min(windows, audio_max):
            raise ValueError("selected audio-window count disagrees with the bounded-window policy")
        complete_interval = audio.get("complete_active_interval_representation")
        if not isinstance(complete_interval, bool) or complete_interval is not (
            windows <= audio_max
        ):
            raise ValueError("audio complete-interval flag disagrees with the window cap")
        shortfall = _finite_number(
            audio.get("end_alignment_shortfall_seconds"), "audio EOF shortfall"
        )
        values["audio_active_frames"].append(float(audio_frames))
        values["audio_window_count"].append(float(windows))
        values["audio_end_shortfall_seconds"].append(shortfall)
        values["text_chunk_count"].append(float(_shape_length(arrays, "text")))
        audio_candidate_windows_below_cap += int(windows < audio_max)
        audio_candidate_windows_equal_to_cap += int(windows == audio_max)
        audio_candidate_windows_above_cap += int(windows > audio_max)
        audio_selected_windows_equal_to_cap += int(selected_windows == audio_max)
        audio_window_cap_truncated += int(windows > audio_max)
        complete_audio += int(complete_interval)
        audio_eof_shortfall += int(shortfall > 0)

    qc_rows = face_qc.get("sessions")
    if not isinstance(qc_rows, list) or len(qc_rows) != len(expected_sessions):
        raise ValueError("face-crop QC coverage drifted")
    if any(not isinstance(row, Mapping) for row in qc_rows):
        raise ValueError("face-crop QC contains an invalid session row")
    qc_sessions = {str(row.get("session_id")) for row in qc_rows if isinstance(row, Mapping)}
    if len(qc_sessions) != len(qc_rows) or qc_sessions != feature_source_sessions:
        raise ValueError("face-crop QC and feature source-session rosters disagree")
    qc_status = Counter(str(row.get("status")) for row in qc_rows)
    qc_failures = Counter(
        str(row.get("failure_code")) for row in qc_rows if row.get("failure_code") is not None
    )
    qc_selected = [
        _finite_number(row.get("selected_grid_count"), "QC selected-grid count") for row in qc_rows
    ]
    qc_valid = [
        _finite_number(row.get("valid_candidate_grid_count"), "QC valid-grid count")
        for row in qc_rows
    ]
    for row in qc_rows:
        session = str(row.get("session_id"))
        if session not in successful_face_counts:
            continue
        expected_selected, expected_valid = successful_face_counts[session]
        if (
            row.get("selected_grid_count") != expected_selected
            or row.get("valid_candidate_grid_count") != expected_valid
        ):
            raise ValueError("face-crop QC disagrees with closure-bound feature sidecars")
    if (
        face_qc.get("schema_version") != "mit-candidate-face-crop-qc-1"
        or face_qc.get("session_count") != len(expected_sessions)
        or face_qc.get("succeeded") != qc_status["success"]
        or face_qc.get("failed") != len(qc_rows) - qc_status["success"]
    ):
        raise ValueError("face-crop QC aggregate contract drifted")

    transcripts = inventory.get("transcripts")
    legacy = inventory.get("supplied_derived_inputs")
    if not isinstance(transcripts, Mapping) or not isinstance(legacy, Mapping):
        raise ValueError("dataset inventory lacks transcript or legacy-input evidence")
    if transcripts.get("session_count") != len(expected_sessions):
        raise ValueError("transcript inventory and Stage-A session counts disagree")
    if (
        manifest.get("selected_session_count") != len(records)
        or manifest.get("completed_session_count") != len(records)
        or manifest.get("success_session_count") != status_counts["success"]
        or manifest.get("failure_session_count") != len(records) - status_counts["success"]
    ):
        raise ValueError("feature closure aggregate counts disagree with its records")
    return {
        "feature_extraction": {
            "selected_session_count": manifest.get("selected_session_count"),
            "completed_session_count": manifest.get("completed_session_count"),
            "success_session_count": manifest.get("success_session_count"),
            "failure_session_count": manifest.get("failure_session_count"),
            "observed_state_counts": _counter_rows(status_counts, "state"),
            "failure_code_counts": _counter_rows(feature_failure_codes, "code"),
            "modality_status_counts": {
                name: _counter_rows(counts, "status") for name, counts in modality_status.items()
            },
        },
        "candidate_face_boundary": {
            "close_reason_counts": _counter_rows(close_reasons, "reason"),
            "active_interval_seconds_quantiles": _quantiles(values["active_interval_seconds"]),
            "selected_sample_count_quantiles": _quantiles(values["selected_face_samples"]),
            "valid_candidate_second_count_quantiles": _quantiles(
                values["valid_candidate_face_seconds"]
            ),
            "sample_cap_reached_session_count": face_cap,
        },
        "face_crop_qc": {
            "session_count": face_qc.get("session_count"),
            "succeeded": face_qc.get("succeeded"),
            "failed": face_qc.get("failed"),
            "status_counts": _counter_rows(qc_status, "status"),
            "failure_code_counts": _counter_rows(qc_failures, "code"),
            "selected_grid_count_quantiles": _quantiles(qc_selected),
            "valid_candidate_grid_count_quantiles": _quantiles(qc_valid),
            "crop_storage_bytes": face_qc.get("crop_storage_bytes"),
            "peak_rss_bytes": face_qc.get("peak_rss_bytes"),
        },
        "audio_boundary": {
            "active_frame_count_quantiles": _quantiles(values["audio_active_frames"]),
            "window_count_quantiles": _quantiles(values["audio_window_count"]),
            "maximum_selected_windows_per_session": audio_window_cap,
            "candidate_window_count_below_cap_session_count": (audio_candidate_windows_below_cap),
            "candidate_window_count_equal_to_cap_session_count": (
                audio_candidate_windows_equal_to_cap
            ),
            "candidate_window_count_above_cap_session_count": (audio_candidate_windows_above_cap),
            "selected_window_count_equal_to_cap_session_count": (
                audio_selected_windows_equal_to_cap
            ),
            "audio_window_cap_truncated_session_count": audio_window_cap_truncated,
            "complete_active_interval_session_count": complete_audio,
            "positive_eof_shortfall_session_count": audio_eof_shortfall,
            "eof_shortfall_seconds_quantiles": _quantiles(values["audio_end_shortfall_seconds"]),
        },
        "candidate_text": {
            "session_count": transcripts.get("session_count"),
            "all_sessions_nonempty": transcripts.get("all_sessions_nonempty"),
            "candidate_only_parser": transcripts.get("candidate_only_parser"),
            "candidate_segment_count": transcripts.get("candidate_segment_count"),
            "excluded_interviewer_segments": transcripts.get("excluded_interviewer_segments"),
            "retained_candidate_continuation_fragments": transcripts.get(
                "retained_candidate_continuation_fragments"
            ),
            "embedding_chunk_count_quantiles": _quantiles(values["text_chunk_count"]),
            "token_count_quantiles": _quantiles(values["candidate_text_token_count"]),
        },
        "representation_quality": {
            "closure_semantic_validation_session_count": status_counts["success"],
            "summary_l2_norm_quantiles": {
                modality: _quantiles(values[f"{modality}_summary_l2_norm"])
                for modality in ("video", "audio", "text")
            },
            "valid_encoded_element_count_quantiles": {
                modality: _quantiles(values[f"{modality}_valid_element_count"])
                for modality in ("video", "audio", "text")
            },
            "zero_summary_session_counts": {
                modality: zero_summary_counts[modality] for modality in ("video", "audio", "text")
            },
        },
        "eligibility": {
            "input_participant_count": eligibility.get("input_participant_count"),
            "input_session_count": eligibility.get("input_session_count"),
            "eligible_participant_count": eligibility.get("eligible_participant_count"),
            "eligible_session_count": eligibility.get("eligible_session_count"),
            "excluded_participant_count": len(eligibility.get("exclusions", [])),
            "eligible_participants_per_fold": eligibility.get("eligible_participants_per_fold"),
            "required_modalities": eligibility.get("required_modalities"),
            "imputation": eligibility.get("imputation"),
        },
        "verified_legacy_input_defects": _legacy_defects(legacy),
        "legacy_quality_error_association": "deferred_not_computed",
    }


def _legacy_defects(legacy: Mapping[str, Any]) -> dict[str, Any]:
    facial = _mapping(legacy.get("facial_features"), "legacy facial features")
    shore = _mapping(legacy.get("shore_smile_data"), "legacy SHORE data")
    prosody = _mapping(legacy.get("prosody"), "legacy prosody")
    rows_by_phase = _mapping(shore.get("rows_by_phase"), "SHORE phase rows")
    return {
        "primary_pipeline_uses_any": legacy.get("primary_pipeline_uses_any"),
        "facial_features": {
            key: facial.get(key)
            for key in (
                "available",
                "files",
                "rows",
                "numeric_columns",
                "files_with_wholly_blank_rows",
                "wholly_blank_rows",
                "partially_blank_rows",
                "trailing_empty_sentinel",
                "used_by_primary_pipeline",
                "reason",
            )
        },
        "shore_smile_data": {
            "available": shore.get("available"),
            "files": shore.get("files"),
            "numeric_columns": shore.get("numeric_columns"),
            "pre_rows": rows_by_phase.get("pre"),
            "post_rows": rows_by_phase.get("post"),
            "truncated_post_files": shore.get("truncated_post_files"),
            "truncated_post_total_frame_shortfall": shore.get(
                "truncated_post_total_frame_shortfall"
            ),
            "used_by_primary_pipeline": shore.get("used_by_primary_pipeline"),
            "reason": shore.get("reason"),
        },
        "prosody": {
            key: prosody.get(key)
            for key in (
                "available",
                "rows",
                "unique_keys",
                "header_columns",
                "physical_row_widths",
                "records_with_undefined_cells",
                "undefined_cells",
                "used_by_primary_pipeline",
                "reason",
            )
        },
    }


def _validate_proxy_identity(
    release: StageARelease,
    payload: Mapping[str, Any],
    source: str,
    rotation: int,
) -> None:
    context = release.closure["context"]
    if (
        payload.get("schema_version") != PROXY_ARTIFACT_SCHEMA_VERSION
        or payload.get("proxy_source") != source
        or payload.get("outer_rotation") != rotation
        or payload.get("outcome_fields_accessed") != []
        or payload.get("protected_annotation_fields_accessed") != []
        or payload.get("eligibility_manifest_sha256") != context.get("eligibility_manifest_sha256")
    ):
        raise ValueError("proxy identity or firewall binding drifted")


def _centre_separation(payload: Mapping[str, Any], source: str, rotation: int) -> dict[str, Any]:
    estimator = _mapping(payload.get("estimator"), "proxy estimator")
    kmeans = _mapping(estimator.get("kmeans"), "proxy k-means evidence")
    try:
        centres = np.asarray(kmeans.get("centres_canonical"), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("learned proxy lacks finite canonical centres") from error
    if (
        centres.ndim != 2
        or centres.shape[0] != 2
        or centres.shape[1] == 0
        or not np.isfinite(centres).all()
    ):
        raise ValueError("learned proxy lacks finite canonical centres")
    return {
        "source": source,
        "outer_rotation": rotation,
        "dimension": int(centres.shape[1]),
        "standardised_euclidean_centre_separation": float(np.linalg.norm(centres[0] - centres[1])),
    }


def _score_w1_row(
    *,
    source: str,
    configuration: str,
    comparison_role: str,
    seed: int | None,
    rows: Sequence[Mapping[str, Any]],
    groups: Mapping[str, int],
    value_field: str,
) -> dict[str, Any]:
    by_phase = {"pre": [[], []], "post": [[], []]}
    pooled: list[list[float]] = [[], []]
    participant_groups = Counter(groups.values())
    for row in rows:
        participant = row.get("participant_id")
        phase = row.get("session_phase")
        if participant not in groups or phase not in by_phase:
            raise ValueError("OOF row does not align with proxy assignment roster")
        group = groups[str(participant)]
        value = _finite_number(row.get(value_field), f"OOF {value_field}")
        pooled[group].append(value)
        by_phase[str(phase)][group].append(value)
    if any(not values for values in pooled):
        raise ValueError("OOF proxy group is empty")
    phase_rows = []
    for phase in ("pre", "post"):
        if any(not values for values in by_phase[phase]):
            raise ValueError("phase-specific OOF proxy group is empty")
        phase_rows.append(
            {
                "phase": phase,
                "group_0_session_count": len(by_phase[phase][0]),
                "group_1_session_count": len(by_phase[phase][1]),
                "wasserstein1": empirical_wasserstein1(by_phase[phase][0], by_phase[phase][1]),
            }
        )
    return {
        "source": source,
        "configuration": configuration,
        "comparison_role": comparison_role,
        "seed": seed,
        "group_ids": ["proxy_group_0", "proxy_group_1"],
        "group_0_participant_count": participant_groups[0],
        "group_1_participant_count": participant_groups[1],
        "group_0_session_count": len(pooled[0]),
        "group_1_session_count": len(pooled[1]),
        "pooled_session_wasserstein1": empirical_wasserstein1(pooled[0], pooled[1]),
        "by_phase": phase_rows,
    }


def _group_counts(
    release: StageARelease,
    assignments: Mapping[tuple[str, int, str], int],
    source: str,
    rotation: int,
    role: str | None,
) -> dict[str, int]:
    selected = [
        participant
        for participant in release.participant_ids
        if role is None or _rotation_role(release.participant_folds[participant], rotation) == role
    ]
    return {
        "participant_count": len(selected),
        "group_0_count": sum(
            assignments[(source, rotation, participant)] == 0 for participant in selected
        ),
        "group_1_count": sum(
            assignments[(source, rotation, participant)] == 1 for participant in selected
        ),
    }


def _rotation_role(fixed_fold: int, outer_rotation: int) -> str:
    if fixed_fold == outer_rotation:
        return "test"
    if fixed_fold == (outer_rotation + 1) % len(FROZEN_OUTER_ROTATIONS):
        return "validation"
    return "train"


def _agreement(left: Sequence[Any], right: Sequence[Any]) -> dict[str, float]:
    table, row_counts, column_counts = _contingency(left, right)
    n = int(table.sum())
    if n < 2:
        ari = 1.0
    else:

        def combinations(values: np.ndarray) -> float:
            return float(np.sum(values * (values - 1) / 2))

        observed = combinations(table)
        row_sum = combinations(row_counts)
        column_sum = combinations(column_counts)
        total = n * (n - 1) / 2
        expected = row_sum * column_sum / total
        maximum = 0.5 * (row_sum + column_sum)
        ari = 1.0 if maximum == expected else (observed - expected) / (maximum - expected)
    probabilities = table / float(n)
    row_probabilities = row_counts / float(n)
    column_probabilities = column_counts / float(n)
    mutual = 0.0
    for row, column in zip(*np.nonzero(table), strict=True):
        mutual += probabilities[row, column] * math.log(
            probabilities[row, column] / (row_probabilities[row] * column_probabilities[column])
        )
    positive_rows = row_probabilities[row_probabilities > 0]
    positive_columns = column_probabilities[column_probabilities > 0]
    entropy_left = -float(np.sum(positive_rows * np.log(positive_rows)))
    entropy_right = -float(np.sum(positive_columns * np.log(positive_columns)))
    denominator = (entropy_left + entropy_right) / 2
    nmi = 1.0 if denominator == 0 else mutual / denominator
    return {
        "adjusted_rand_index": float(ari),
        "normalised_mutual_information": float(nmi),
    }


def _contingency(
    left: Sequence[Any], right: Sequence[Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(left) != len(right) or not left:
        raise ValueError("association vectors must be non-empty and aligned")
    left_values = sorted(set(left), key=lambda value: (type(value).__name__, str(value)))
    right_values = sorted(set(right), key=lambda value: (type(value).__name__, str(value)))
    left_index = {value: index for index, value in enumerate(left_values)}
    right_index = {value: index for index, value in enumerate(right_values)}
    table = np.zeros((len(left_values), len(right_values)), dtype=np.float64)
    for first, second in zip(left, right, strict=True):
        table[left_index[first], right_index[second]] += 1
    return table, table.sum(axis=1), table.sum(axis=0)


def _validate_eligibility(
    release: StageARelease,
    eligibility: Mapping[str, Any],
    context: Mapping[str, Any],
) -> None:
    _validate_signed_payload(eligibility, "manifest_payload_sha256", "eligibility manifest")
    if (
        eligibility.get("manifest_payload_sha256") != context.get("eligibility_manifest_sha256")
        or eligibility.get("feature_extraction_closure_file_sha256")
        != context.get("feature_extraction_closure_file_sha256")
        or eligibility.get("feature_extraction_closure_payload_sha256")
        != context.get("feature_extraction_closure_payload_sha256")
        or eligibility.get("session_mapping") != context.get("session_mapping")
        or eligibility.get("outcome_fields_accessed") != []
        or eligibility.get("protected_annotation_fields_accessed") != []
        or eligibility.get("eligible_participant_count") != release.participant_count
        or eligibility.get("eligible_session_count") != len(release.session_keys)
    ):
        raise ValueError("eligibility manifest does not match the Stage-A closure")
    participants = eligibility.get("eligible_participants")
    if not isinstance(participants, list):
        raise ValueError("eligibility participant registry is missing")
    observed_participants: set[str] = set()
    observed_sessions: set[tuple[str, str, str]] = set()
    for row in participants:
        if not isinstance(row, Mapping):
            raise ValueError("eligibility participant row is invalid")
        participant = str(row.get("participant_id"))
        sessions = row.get("sessions")
        if not isinstance(sessions, list):
            raise ValueError("eligibility session pair is missing")
        observed_participants.add(participant)
        for session in sessions:
            if not isinstance(session, Mapping):
                raise ValueError("eligibility session row is invalid")
            observed_sessions.add(
                (participant, str(session.get("condition")), str(session.get("session_id")))
            )
    if observed_participants != set(release.participant_ids) or observed_sessions != set(
        release.session_keys
    ):
        raise ValueError("eligibility roster does not exactly match Stage A")


def _validate_inventory_binding(inventory: Mapping[str, Any], context: Mapping[str, Any]) -> None:
    legacy = _mapping(inventory.get("supplied_derived_inputs"), "legacy inventory")
    media = _mapping(inventory.get("media"), "media inventory")
    if (
        inventory.get("corpus") != "mit_interview"
        or inventory.get("inventory_schema_version") != 3
        or inventory.get("mode") != "validated_restricted_corpus"
        or inventory.get("session_mapping") != context.get("session_mapping")
        or _mapping(inventory.get("protected_annotations"), "inventory annotation policy").get(
            "ordinary_inventory_inspects_manual_annotations"
        )
        is not False
        or legacy.get("primary_pipeline_uses_any") is not False
        or media.get("supplied_facial_shore_prosody_used_by_primary_pipeline") is not False
    ):
        raise ValueError("dataset inventory provenance or firewall policy drifted")


def _artifact_record(lock_manifest: Mapping[str, Any], filename: str) -> Mapping[str, Any]:
    artifacts = lock_manifest.get("artifacts")
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)):
        raise ValueError("prediction lock lacks its artifact inventory")
    matches = [
        row for row in artifacts if isinstance(row, Mapping) and row.get("filename") == filename
    ]
    if len(matches) != 1:
        raise ValueError(f"prediction lock does not bind exactly one {filename!r}")
    return matches[0]


def _validated_source_path(path: str | Path, label: str) -> Path:
    source = Path(path).expanduser()
    if source.name in {"", ".", ".."} or any(
        component in {".", ".."} for component in source.parts
    ):
        raise ValueError(f"{label} path is unsafe")
    if not source.is_absolute():
        source = Path.cwd() / source
    return source


def _read_json_object(
    path: str | Path,
    label: str,
    *,
    expected_file_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    source = _validated_source_path(path, label)
    expected = (
        None
        if expected_file_sha256 is None
        else _require_sha256(expected_file_sha256, f"expected {label} file SHA-256")
    )
    with AnchoredDirectory.open(source.parent, directory_mode=None) as anchor:
        observed = anchor.read_bytes(source.name, expected_mode=0o600)
        if expected is not None and observed.sha256 != expected:
            raise ValueError(f"{label} does not match the caller-pinned file SHA-256")
        payload = _json_object(observed.content, label)
        anchor.verify_file(source.name, observed, expected_mode=0o600)
    return payload, observed.sha256


def _json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"cannot read valid {label} JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return payload


def _canonical_object(raw: bytes, label: str) -> dict[str, Any]:
    payload = _json_object(raw, label)
    if canonical_json_bytes(payload) != raw:
        raise ValueError(f"{label} JSON is not byte-canonical")
    return payload


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _validate_signed_payload(payload: Mapping[str, Any], field: str, label: str) -> None:
    observed = _require_sha256(payload.get(field), f"{label} payload SHA-256")
    unsigned = {key: value for key, value in payload.items() if key != field}
    if observed != _payload_sha256(unsigned):
        raise ValueError(f"{label} payload checksum failed")


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _quantiles(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "q25": None,
            "median": None,
            "q75": None,
            "maximum": None,
        }
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("quality distribution contains a non-finite value")
    quantiles = np.quantile(array, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "count": int(array.size),
        "minimum": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "maximum": float(quantiles[4]),
    }


def _counter_rows(counter: Counter[str], label: str) -> list[dict[str, Any]]:
    return [{label: key, "count": int(counter[key])} for key in sorted(counter)]


def _representation_quality(shard_bytes: bytes) -> dict[str, float]:
    required = {
        "video_mean",
        "audio_mean",
        "text_mean",
        "video_mask",
        "audio_mask",
        "text_mask",
        "text_token_count",
    }
    try:
        with np.load(io.BytesIO(shard_bytes), allow_pickle=False) as archive:
            if not required.issubset(archive.files):
                raise ValueError("feature shard lacks representation-quality arrays")
            result: dict[str, float] = {}
            for modality in ("video", "audio", "text"):
                summary = np.asarray(archive[f"{modality}_mean"], dtype=np.float64)
                mask = np.asarray(archive[f"{modality}_mask"])
                if (
                    summary.ndim != 1
                    or not summary.size
                    or not np.isfinite(summary).all()
                    or mask.ndim != 1
                    or mask.dtype != np.bool_
                    or not mask.size
                ):
                    raise ValueError("feature shard representation-quality array is invalid")
                result[f"{modality}_summary_l2_norm"] = float(np.linalg.norm(summary))
                result[f"{modality}_valid_element_count"] = float(mask.sum())
            token_counts = np.asarray(archive["text_token_count"])
            if (
                token_counts.ndim != 1
                or token_counts.size != np.asarray(archive["text_mask"]).size
                or not np.issubdtype(token_counts.dtype, np.integer)
                or np.any(token_counts < 0)
            ):
                raise ValueError("feature shard text-token evidence is invalid")
            result["candidate_text_token_count"] = float(token_counts.sum())
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("feature shard"):
            raise
        raise ValueError("cannot read closure-bound feature shard") from error
    return result


def _shape_length(arrays: Mapping[str, Any], name: str) -> int:
    detail = _mapping(arrays.get(name), f"{name} array metadata")
    shape = detail.get("shape")
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)) or not shape:
        raise ValueError(f"{name} array shape is invalid")
    return _positive_int(shape[0], f"{name} array length")


def _sequence_length(value: Any, label: str) -> int:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    return len(value)


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _diagnostics_source_sha256() -> str:
    current = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if current != _IMPORTED_DIAGNOSTICS_SOURCE_SHA256:
        raise RuntimeError("diagnostics evaluator source changed after module import")
    return current


def _assert_aggregate_only(value: Any, *, _context_key: str | None = None) -> None:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("public diagnostics contain a non-string object key")
        keys = set(value)
        forbidden = keys & _PROHIBITED_PUBLIC_KEYS
        if forbidden:
            raise ValueError(f"public diagnostics contain identifier field(s): {sorted(forbidden)}")
        for key, nested in value.items():
            if key.isdigit():
                if _context_key != "eligible_participants_per_fold" or key not in {
                    str(rotation) for rotation in FROZEN_OUTER_ROTATIONS
                }:
                    raise ValueError("public diagnostics contain an unapproved numeric object key")
            elif _SAFE_PUBLIC_KEY.fullmatch(key) is None:
                raise ValueError("public diagnostics contain an unsafe object key")
            if key != "group_ids" and (
                key.endswith(("_id", "_ids")) or key.endswith(_PROHIBITED_PUBLIC_KEY_SUFFIXES)
            ):
                raise ValueError(f"public diagnostics contain identifier/path field {key!r}")
            _assert_aggregate_only(nested, _context_key=key)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            _assert_aggregate_only(nested, _context_key=_context_key)
    elif isinstance(value, str):
        _assert_safe_public_string(value, _context_key)


def _assert_safe_public_string(value: str, context_key: str | None) -> None:
    if context_key == "sha256" or (context_key is not None and context_key.endswith("_sha256")):
        _require_sha256(value, f"public {context_key}")
        return
    if value in _SAFE_PUBLIC_STRING_VALUES:
        return
    if (
        "/" in value
        or "\\" in value
        or "\x00" in value
        or value.startswith(("~", "."))
        or "://" in value
        or re.search(r"(?:^|[._-])(?:json|npz|csv|tsv|wav|mp[34])$", value, re.IGNORECASE)
    ):
        raise ValueError("public diagnostics contain a path-like string value")
    if (
        _IDENTIFIER_LIKE_VALUE.fullmatch(value) is not None
        or _UUID_VALUE.fullmatch(value) is not None
        or "@" in value
    ):
        raise ValueError("public diagnostics contain an identifier-like string value")
    raise ValueError("public diagnostics contain an unapproved string value")


def _markdown(payload: Mapping[str, Any]) -> bytes:
    proxy = payload["proxy_diagnostics"]
    boundary = payload["boundary_and_representation_quality"]
    feature = boundary["feature_extraction"]
    eligibility = boundary["eligibility"]
    lines = [
        "# Stage A Annotation-Free Diagnostics",
        "",
        "This report was generated only from checksum-validated Stage-A, feature, "
        "face-QC, eligibility and inventory evidence. It contains no participant- or "
        "session-level rows and accessed no protected annotations.",
        "",
        "## Evidence scope",
        "",
        f"- Participants: {payload['participant_count']}",
        f"- Sessions: {payload['session_count']}",
        f"- Successful feature sessions: {feature['success_session_count']}",
        f"- Failed feature sessions: {feature['failure_session_count']}",
        f"- Eligible complete pairs: {eligibility['eligible_participant_count']}",
        "",
        "## Proxy diagnostics",
        "",
        f"- Proxy rotation/role balance rows: {len(proxy['balance_by_rotation_and_role'])}",
        f"- Learned centre-separation rows: {len(proxy['standardised_centre_separation'])}",
        f"- Cross-rotation stability comparisons: {len(proxy['cross_rotation_stability'])}",
        "- Proxy-conditioned prediction W1 rows: "
        f"{len(proxy['oof_proxy_conditioned_prediction_w1'])}",
        "- Every proxy artifact was required to be byte-identical across its fixed "
        "seeds and peer configurations.",
        "- Proxy groups 0 and 1 are neutral algorithmic labels. They are not demographic "
        "identities or protected-attribute inferences.",
        "",
        "## Boundary and representation quality",
        "",
        "- Candidate-face close reasons: "
        f"{_display_counts(boundary['candidate_face_boundary']['close_reason_counts'], 'reason')}",
        "- Face sample cap reached: "
        f"{boundary['candidate_face_boundary']['sample_cap_reached_session_count']} sessions",
        "- Audio candidate-window counts relative to the "
        f"{boundary['audio_boundary']['maximum_selected_windows_per_session']}-window cap: "
        f"below={boundary['audio_boundary']['candidate_window_count_below_cap_session_count']}, "
        f"equal={boundary['audio_boundary']['candidate_window_count_equal_to_cap_session_count']}, "
        f"above={boundary['audio_boundary']['candidate_window_count_above_cap_session_count']}",
        "- Audio sessions selecting the maximum window count: "
        f"{boundary['audio_boundary']['selected_window_count_equal_to_cap_session_count']} "
        "sessions",
        "- Audio sessions temporally subsampled because candidate windows exceeded the cap: "
        f"{boundary['audio_boundary']['audio_window_cap_truncated_session_count']} sessions",
        "- Positive audio end-of-stream shortfall: "
        f"{boundary['audio_boundary']['positive_eof_shortfall_session_count']} sessions",
        "- Candidate transcripts all non-empty: "
        f"{boundary['candidate_text']['all_sessions_nonempty']}",
        "- Verified legacy derived-feature defects are inventory evidence only; those "
        "inputs were not used by the primary pipeline.",
        "- Association between legacy quality defects and prediction error was not computed "
        "in this release.",
        "",
        "## Provenance",
        "",
        "| Binding | SHA-256 |",
        "|---|---|",
        *(
            f"| {name.replace('_', ' ')} | `{digest}` |"
            for name, digest in sorted(payload["source_bindings"].items())
        ),
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _display_counts(rows: Sequence[Mapping[str, Any]], key: str) -> str:
    return ", ".join(f"{row[key]}={row['count']}" for row in rows) or "none"


def _validated_separate_output(source_root: Path, output_dir: str | Path) -> Path:
    output = Path(output_dir).expanduser()
    if any(part in {".", ".."} for part in output.parts):
        raise ValueError("public diagnostics output path may not contain dot components")
    if not output.is_absolute():
        output = Path.cwd() / output
    resolved = output.resolve(strict=False)
    source = source_root.resolve()
    if resolved == source or resolved.is_relative_to(source) or source.is_relative_to(resolved):
        raise ValueError("public diagnostics must be outside the closed Stage-A tree")
    return resolved
