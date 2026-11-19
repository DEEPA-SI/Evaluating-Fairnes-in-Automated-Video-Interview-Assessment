"""Strict candidate-only crop manifests and blinded participant annotation packets."""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import math
import re
import subprocess
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any

from .mit_interview import MITInterviewDataset, MITInterviewRosterEntry, parse_session_id
from .mit_session_mapping import (
    MIT_SESSION_MAPPING_SCHEMA_VERSION,
    SESSION_MAPPING_PROVENANCE_FIELDS,
    SessionMappingContext,
)

FACE_MANIFEST_SCHEMA_VERSION = "mit-candidate-face-crops-2"
STIMULUS_SCHEMA_VERSION = "mit-annotation-stimuli-3"
RATER_DISPLAY_SCHEMA_VERSION = "mit-annotation-rater-display-1"
RATER_LINKAGE_SCHEMA_VERSION = "mit-annotation-rater-linkage-1"
STIMULUS_SALT = "fair-avi-mit-annotation-stimulus-v1"
PARTICIPANT_SHUFFLE_SALT = "fair-avi-mit-annotation-participant-order-v1"
RATER_PACKET_SALT = "fair-avi-mit-annotation-rater-packet-v1"
TARGET_FRACTIONS = (Decimal("0.2"), Decimal("0.5"), Decimal("0.8"))

FACE_GENERATOR_FIELDS = (
    "name",
    "version",
    "source_commit_sha",
    "source_tree_sha256",
    "source_tree_dirty",
    "config_sha256",
    "source_date_epoch",
    "insightface_version",
    "onnxruntime_version",
    "opencv_version",
    "numpy_version",
    "pillow_version",
    "ffprobe_version",
    "ffprobe_build_sha256",
)
FACE_SELECTOR_FIELDS = (
    "name",
    "revision",
    "artifact_release",
    "artifact_url",
    "archive_sha256",
    "artifact_file",
    "artifact_sha256",
    "artifact_size_bytes",
    "input_size",
    "preprocessing",
    "decoding",
    "principal_face_rule",
    "confidence_threshold",
    "nms_threshold",
    "candidate_centrality_max_ratio",
    "candidate_centrality_margin_ratio",
    "scan_fps",
    "first_valid_max_seconds",
    "minimum_active_elapsed_seconds",
    "terminal_invalid_run_seconds",
    "minimum_valid_candidate_seconds",
    "active_interval_rule",
    "retained_grid_rule",
    "maximum_retained_samples",
    "crop_rule",
    "crop_scale",
    "minimum_crop_side_pixels",
    "subject_role",
)
FACE_CLIP_FIELDS = (
    "source_video_sha256",
    "duration_seconds",
    "stream_duration_seconds",
    "full_one_hz_grid_count",
    "frame_width",
    "frame_height",
    "frame_rate_numerator",
    "frame_rate_denominator",
    "nominal_frame_rate_numerator",
    "nominal_frame_rate_denominator",
    "time_base_numerator",
    "time_base_denominator",
    "frame_count",
    "stream_start_time_seconds",
    "constant_frame_rate",
    "scan",
    "active_start_grid_index",
    "active_end_grid_index_exclusive",
    "active_close_reason",
    "valid_candidate_grid_indices",
    "selected_grid_indices",
    "status",
    "failure",
    "frames",
)
FACE_FRAME_FIELDS = (
    "path",
    "crop_sha256",
    "timestamp_seconds",
    "grid_index",
    "source_frame_index",
    "decoded_timestamp_seconds",
    "box_xyxy",
    "crop_box_xyxy",
    "confidence",
)
FACE_FAILURE_FIELDS = ("code", "message")
FACE_SCAN_FIELDS = (
    "grid_index",
    "timestamp_seconds",
    "status",
    "reason",
    "detection_count",
    "box_xyxy",
    "confidence",
    "centrality_ratio",
)
_TOP_FACE_FIELDS = {"schema_version", "generator", "selector", "clips"}
_STIMULUS_TOP_FIELDS = {
    "schema_version",
    "session_mapping",
    "subject_role",
    "packet_size",
    "target_fractions",
    "selection_provenance",
    "rater_display_prohibited_fields",
    "participants",
}
_STIMULUS_PACKET_FIELDS = {
    "participant_id",
    "status",
    "assessment_reasons",
    "session_qc",
    "offset_30s_sensitivity_eligible",
    "stimuli",
}
_STIMULUS_ROW_FIELDS = {
    "stimulus_id",
    "session_id",
    "source_session_id",
    "crop_reference",
    "crop_sha256",
    "grid_index",
    "target_fraction",
    "target_seconds",
    "actual_seconds",
    "absolute_offset_seconds",
    "display_order",
}
_RATER_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$", re.ASCII)
_LOWER_SHA40 = re.compile(r"^[0-9a-f]{40}$", re.ASCII)
_LOWER_SHA64 = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_DURATION_TOLERANCE_SECONDS = Decimal("0.001")


def bounded_visual_grid_indices(grid_count: int, maximum: int = 128) -> tuple[int, ...]:
    """Return the fixed endpoint-spanning integer selection over a one-second grid."""
    if isinstance(grid_count, bool) or isinstance(maximum, bool):
        raise TypeError("grid_count and maximum must be integers")
    if grid_count <= 0 or maximum <= 0:
        raise ValueError("grid_count and maximum must be positive")
    if grid_count <= maximum:
        return tuple(range(grid_count))
    if maximum == 1:
        return (0,)
    return tuple((rank * (grid_count - 1)) // (maximum - 1) for rank in range(maximum))


def validate_face_selector(value: object) -> dict[str, Any]:
    """Validate the selector provenance shared by extraction and annotation code."""
    if not isinstance(value, dict) or tuple(sorted(value)) != tuple(sorted(FACE_SELECTOR_FIELDS)):
        raise ValueError("face selector does not match the exact versioned provenance schema")
    result = dict(value)
    for field in (
        "name",
        "artifact_release",
        "artifact_url",
        "artifact_file",
        "preprocessing",
        "decoding",
        "principal_face_rule",
        "active_interval_rule",
        "retained_grid_rule",
        "crop_rule",
    ):
        if not isinstance(result[field], str) or not _clean_text(result[field]):
            raise ValueError(f"face selector {field} must be trimmed non-empty text")
    if not result["artifact_url"].startswith("https://"):
        raise ValueError("face selector artifact_url must use HTTPS")
    if (
        not isinstance(result["revision"], str)
        or _LOWER_SHA40.fullmatch(result["revision"]) is None
    ):
        raise ValueError("face selector revision must be a full lowercase commit SHA")
    if result["subject_role"] != "candidate":
        raise ValueError("face selector subject_role must be candidate")
    for field in ("archive_sha256", "artifact_sha256"):
        if not isinstance(result[field], str) or _LOWER_SHA64.fullmatch(result[field]) is None:
            raise ValueError(f"face selector {field} must be a lowercase SHA-256")
    artifact_size = result["artifact_size_bytes"]
    minimum_side = result["minimum_crop_side_pixels"]
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in (artifact_size, minimum_side)
    ):
        raise ValueError("face selector artifact size and minimum crop side must be positive")
    input_size = result["input_size"]
    if (
        not isinstance(input_size, list)
        or len(input_size) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in input_size
        )
    ):
        raise ValueError("face selector input_size must contain two positive integers")
    for field in (
        "confidence_threshold",
        "nms_threshold",
        "candidate_centrality_max_ratio",
        "candidate_centrality_margin_ratio",
    ):
        numeric = _finite_number(result[field], f"selector {field}")
        if not 0 < numeric <= 1:
            raise ValueError(f"face selector {field} must be in (0,1]")
        result[field] = numeric
    crop_scale = _finite_number(result["crop_scale"], "selector crop_scale")
    if crop_scale < 1:
        raise ValueError("face selector crop_scale must be at least one")
    result["crop_scale"] = crop_scale
    for field in (
        "scan_fps",
        "first_valid_max_seconds",
        "minimum_active_elapsed_seconds",
        "terminal_invalid_run_seconds",
        "minimum_valid_candidate_seconds",
        "maximum_retained_samples",
    ):
        numeric = result[field]
        minimum = 0 if field == "first_valid_max_seconds" else 1
        if isinstance(numeric, bool) or not isinstance(numeric, int) or numeric < minimum:
            raise ValueError(f"face selector {field} is invalid")
    if result["scan_fps"] != 1 or result["maximum_retained_samples"] != 128:
        raise ValueError("face selector scan rate/cap drifted from the frozen contract")
    if result["minimum_valid_candidate_seconds"] > result["maximum_retained_samples"]:
        raise ValueError("face selector minimum valid count exceeds the retained cap")
    return result


def validate_face_generator(value: object) -> dict[str, Any]:
    """Validate generator source/runtime provenance without accepting host paths."""
    if not isinstance(value, dict) or tuple(sorted(value)) != tuple(sorted(FACE_GENERATOR_FIELDS)):
        raise ValueError("face generator does not match the exact versioned provenance schema")
    result = dict(value)
    for field in (
        "name",
        "version",
        "insightface_version",
        "onnxruntime_version",
        "opencv_version",
        "numpy_version",
        "pillow_version",
        "ffprobe_version",
    ):
        if not isinstance(result[field], str) or not _clean_text(result[field]):
            raise ValueError(f"face generator {field} must be trimmed non-empty text")
    if not result["ffprobe_version"].startswith("ffprobe version "):
        raise ValueError("face generator ffprobe_version is invalid")
    if _LOWER_SHA40.fullmatch(str(result["source_commit_sha"])) is None:
        raise ValueError("face generator source_commit_sha must be a full lowercase commit SHA")
    for field in ("source_tree_sha256", "config_sha256", "ffprobe_build_sha256"):
        if _LOWER_SHA64.fullmatch(str(result[field])) is None:
            raise ValueError(f"face generator {field} must be a lowercase SHA-256")
    if not isinstance(result["source_tree_dirty"], bool):
        raise ValueError("face generator source_tree_dirty must be boolean")
    epoch = result["source_date_epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("face generator source_date_epoch must be a non-negative integer")
    return result


def load_face_crop_manifest(
    path: str | Path, *, expected_session_ids: tuple[str, ...] | None = None
) -> dict[str, Any]:
    """Load a duplicate-key-safe, non-symlink candidate crop manifest.

    Crop paths are portable relative references beneath the manifest directory.  Symlink crops
    and symlink path components are rejected.  Missing detections are represented by absent frame
    rows, not by dropping a session or changing its fixed selected-grid provenance.
    """
    source = Path(path)
    payload, _ = _load_unique_json_file(source, "face-crop manifest")
    if not isinstance(payload, dict) or set(payload) != _TOP_FACE_FIELDS:
        raise ValueError("face-crop manifest must have the exact versioned top-level schema")
    if payload["schema_version"] != FACE_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported face-crop manifest schema_version")
    generator = validate_face_generator(payload["generator"])
    selector = validate_face_selector(payload["selector"])
    clips = payload["clips"]
    if not isinstance(clips, dict):
        raise ValueError("face-crop manifest clips must be an object")
    if expected_session_ids is not None:
        expected = set(expected_session_ids)
        if len(expected) != len(expected_session_ids):
            raise ValueError("expected session IDs must be unique")
        missing = sorted(expected - set(clips))
        unexpected = sorted(set(clips) - expected)
        if missing or unexpected:
            raise ValueError(
                f"face-crop session coverage mismatch: missing={missing!r}; "
                f"unexpected={unexpected!r}"
            )
    root = source.parent.resolve()
    validated_clips: dict[str, Any] = {}
    for session_id in sorted(clips, key=lambda item: item.encode("utf-8")):
        if not isinstance(session_id, str) or not _clean_text(session_id):
            raise ValueError("face-crop session IDs must be trimmed non-empty text")
        validated_clips[session_id] = _validate_face_clip(
            clips[session_id], root=root, selector=selector
        )
    return {
        "schema_version": FACE_MANIFEST_SCHEMA_VERSION,
        "generator": generator,
        "selector": selector,
        "clips": validated_clips,
    }


def build_stimulus_manifest(
    dataset: MITInterviewDataset | tuple[MITInterviewRosterEntry, ...],
    *,
    face_crop_manifest: str | Path,
    dataset_root: str | Path,
    session_mapping: SessionMappingContext | None = None,
) -> dict[str, Any]:
    """Build paired six-image packets from a strict candidate-only crop manifest.

    Source videos are rehashed against the strict face manifest before any annotation packet is
    emitted. This prevents crops or session labels being silently substituted across sessions.
    """
    dataset_directory = Path(dataset_root)
    if dataset_directory.is_symlink() or not dataset_directory.is_dir():
        raise ValueError("dataset root must be a regular non-symlink directory")
    dataset_directory = dataset_directory.resolve()
    records = dataset.records if isinstance(dataset, MITInterviewDataset) else tuple(dataset)
    if not records:
        raise ValueError("annotation roster must be non-empty")
    if session_mapping is not None:
        session_mapping.verify_current_files()
        if any(
            session_mapping.lookup.get(row.session_id) != row.source_session_id
            for row in records
        ):
            raise ValueError("annotation roster disagrees with approved session mapping")
    manifest_path = Path(face_crop_manifest)
    payload = load_face_crop_manifest(
        manifest_path,
        expected_session_ids=tuple(row.source_session_id for row in records),
    )
    face_manifest_sha256 = _file_sha256(manifest_path)
    if (
        session_mapping is not None
        and face_manifest_sha256
        != session_mapping.provenance["face_crop_manifest_file_sha256"]
    ):
        raise ValueError("face-crop manifest disagrees with approved session mapping")
    selector_sha256 = _canonical_sha256(payload["selector"])
    ffprobe_version, ffprobe_build_sha256 = _ffprobe_provenance()

    packets: list[dict[str, Any]] = []
    participant_ids = tuple(sorted({row.participant_id for row in records}, key=int))
    for participant_id in participant_ids:
        sessions = sorted(
            (row for row in records if row.participant_id == participant_id),
            key=lambda row: row.session_id.encode("ascii"),
        )
        if len(sessions) != 2:
            raise ValueError("every annotation participant must have exactly two sessions")
        selected: list[dict[str, Any]] = []
        reasons: list[str] = []
        session_qc: list[dict[str, Any]] = []
        for session in sessions:
            clip = payload["clips"][session.source_session_id]
            source_video = _regular_source_video(dataset_directory, session.video_reference)
            if _file_sha256(source_video) != clip["source_video_sha256"]:
                raise ValueError(
                    f"face-crop source video checksum mismatch for {session.session_id}"
                )
            measured_duration = _ffprobe_duration(source_video)
            declared_duration = Decimal(clip["duration_seconds"])
            if abs(measured_duration - declared_duration) > _DURATION_TOLERANCE_SECONDS:
                raise ValueError(
                    f"face-crop duration mismatch for {session.session_id}: "
                    f"declared={declared_duration}, measured={measured_duration}"
                )
            rows = clip["frames"]
            active_start = clip["active_start_grid_index"]
            active_end = clip["active_end_grid_index_exclusive"]
            active_duration = (
                active_end - active_start
                if isinstance(active_start, int)
                and not isinstance(active_start, bool)
                and isinstance(active_end, int)
                and not isinstance(active_end, bool)
                else None
            )
            session_qc.append(
                {
                    "session_id": session.session_id,
                    "source_session_id": session.source_session_id,
                    "face_boundary_status": clip["status"],
                    "failure_code": (
                        clip["failure"]["code"] if clip["failure"] is not None else None
                    ),
                    "active_start_seconds": active_start,
                    "active_end_seconds_exclusive": active_end,
                    "active_duration_seconds": active_duration,
                    "valid_distinct_crop_count": len(rows),
                    "required_crop_count": 3,
                }
            )
            if clip["status"] != "success":
                reasons.append(f"{session.session_id}:{clip['failure']['code']}")
                continue
            session_rows = _nearest_distinct(rows, active_start, active_end)
            if session_rows is None:
                reasons.append(f"{session.session_id}:fewer_than_3_distinct_valid_crops")
                continue
            interval_start = Decimal(active_start)
            interval_duration = Decimal(active_end - active_start)
            for target_index, row in enumerate(session_rows):
                target = interval_start + interval_duration * TARGET_FRACTIONS[target_index]
                actual = Decimal(row["timestamp_seconds"])
                opaque = hashlib.sha256(
                    STIMULUS_SALT.encode("ascii")
                    + b"\0"
                    + participant_id.encode("ascii")
                    + b"\0"
                    + session.session_id.encode("ascii")
                    + b"\0"
                    + session.source_session_id.encode("ascii")
                    + b"\0"
                    + (
                        _canonical_sha256(session_mapping.provenance).encode("ascii")
                        if session_mapping is not None
                        else b"synthetic-no-mapping"
                    )
                    + b"\0"
                    + str(target_index).encode("ascii")
                ).hexdigest()[:24]
                selected.append(
                    {
                        "stimulus_id": opaque,
                        "session_id": session.session_id,
                        "source_session_id": session.source_session_id,
                        "crop_reference": row["path"],
                        "crop_sha256": row["crop_sha256"],
                        "grid_index": row["grid_index"],
                        "target_fraction": str(TARGET_FRACTIONS[target_index]),
                        "target_seconds": _decimal_string(target),
                        "actual_seconds": row["timestamp_seconds"],
                        "absolute_offset_seconds": _decimal_string(abs(actual - target)),
                    }
                )
        assessable = not reasons and len(selected) == 6
        if not assessable:
            selected = []
        else:
            selected.sort(
                key=lambda row: _salted_digest(
                    STIMULUS_SALT,
                    "display",
                    participant_id,
                    row["stimulus_id"],
                )
            )
            for display_order, row in enumerate(selected, start=1):
                row["display_order"] = display_order
        sensitivity_eligible = bool(
            assessable and all(Decimal(row["absolute_offset_seconds"]) <= 30 for row in selected)
        )
        packets.append(
            {
                "participant_id": participant_id,
                "status": "assessable" if assessable else "not_assessable",
                "assessment_reasons": reasons,
                "session_qc": session_qc,
                "offset_30s_sensitivity_eligible": sensitivity_eligible,
                "stimuli": selected,
            }
        )
    result = {
        "schema_version": STIMULUS_SCHEMA_VERSION,
        "session_mapping": (
            session_mapping.provenance if session_mapping is not None else None
        ),
        "subject_role": "candidate",
        "packet_size": 6,
        "target_fractions": [str(value) for value in TARGET_FRACTIONS],
        "selection_provenance": {
            "face_crop_manifest_schema_version": FACE_MANIFEST_SCHEMA_VERSION,
            "face_crop_manifest_sha256": face_manifest_sha256,
            "selector_sha256": selector_sha256,
            "selection_algorithm": "nearest_distinct_timestamp_to_each_active_interval_target",
            "time_basis": "candidate_face_active_interval_half_open",
            "selection_tie_break": "absolute_offset_then_timestamp_then_utf8_path",
            "stimulus_id_algorithm": "sha256_prefix_24_hex",
            "stimulus_id_salt": STIMULUS_SALT,
            "within_packet_shuffle_algorithm": "sha256_rank",
            "within_packet_shuffle_salt": STIMULUS_SALT,
            "duration_probe": "ffprobe_format_duration",
            "duration_probe_version": ffprobe_version,
            "duration_probe_build_sha256": ffprobe_build_sha256,
            "duration_tolerance_seconds": str(_DURATION_TOLERANCE_SECONDS),
        },
        "rater_display_prohibited_fields": [
            "participant_id",
            "session_id",
            "source_session_id",
            "crop_reference",
            "condition",
            "phase",
            "transcript",
            "outcome",
            "prediction",
            "proxy",
        ],
        "participants": packets,
    }
    if session_mapping is not None:
        session_mapping.verify_current_files()
    _validate_stimulus_payload(result)
    return result


def build_rater_presentation_manifests(
    stimulus_manifest: dict[str, Any] | str | Path,
    *,
    rater_id: str,
    blinding_key: bytes,
    face_crop_manifest: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create a path-free display payload and a separate restricted linkage payload."""
    if _RATER_ID_PATTERN.fullmatch(rater_id) is None:
        raise ValueError("rater_id must be a pseudonymous token")
    if not isinstance(blinding_key, bytes) or len(blinding_key) < 32:
        raise ValueError("blinding_key must contain at least 32 secret bytes")
    if isinstance(stimulus_manifest, (str, Path)):
        payload = load_stimulus_manifest(stimulus_manifest)
    else:
        payload = stimulus_manifest
    _validate_stimulus_payload(payload)
    source_sha256 = _canonical_sha256(payload)
    face_path = Path(face_crop_manifest)
    face_payload = load_face_crop_manifest(face_path)
    expected_face_sha256 = payload["selection_provenance"]["face_crop_manifest_sha256"]
    if _file_sha256(face_path) != expected_face_sha256:
        raise ValueError("face-crop manifest changed after stimulus selection")
    mapping_provenance = payload["session_mapping"]
    if (
        mapping_provenance is not None
        and mapping_provenance["face_crop_manifest_file_sha256"] != expected_face_sha256
    ):
        raise ValueError("stimulus face manifest disagrees with its approved session mapping")
    if (
        _canonical_sha256(face_payload["selector"])
        != payload["selection_provenance"]["selector_sha256"]
    ):
        raise ValueError("face selector changed after stimulus selection")
    assessable = [row for row in payload["participants"] if row["status"] == "assessable"]
    assessable.sort(
        key=lambda row: _keyed_digest(
            blinding_key, PARTICIPANT_SHUFFLE_SALT, rater_id, row["participant_id"]
        )
    )
    display_packets: list[dict[str, Any]] = []
    linkage_packets: list[dict[str, Any]] = []
    for packet_order, packet in enumerate(assessable, start=1):
        participant_id = packet["participant_id"]
        packet_id = _keyed_digest(blinding_key, RATER_PACKET_SALT, rater_id, participant_id).hex()[
            :32
        ]
        stimuli = sorted(packet["stimuli"], key=lambda row: row["display_order"])
        for row in stimuli:
            face_rows = face_payload["clips"].get(row["source_session_id"], {}).get("frames", [])
            matches = [
                face_row for face_row in face_rows if face_row["grid_index"] == row["grid_index"]
            ]
            if (
                len(matches) != 1
                or matches[0]["path"] != row["crop_reference"]
                or matches[0]["crop_sha256"] != row["crop_sha256"]
            ):
                raise ValueError(
                    "stimulus crop no longer matches the checksum-locked face manifest"
                )
        display_ids = {
            row["stimulus_id"]: _keyed_digest(
                blinding_key,
                RATER_PACKET_SALT,
                "stimulus",
                rater_id,
                row["stimulus_id"],
            ).hex()[:32]
            for row in stimuli
        }
        display_packets.append(
            {
                "packet_id": packet_id,
                "packet_order": packet_order,
                "stimuli": [
                    {
                        "stimulus_id": display_ids[row["stimulus_id"]],
                        "display_order": row["display_order"],
                        "image_asset": f"{display_ids[row['stimulus_id']]}.png",
                    }
                    for row in stimuli
                ],
            }
        )
        linkage_packets.append(
            {
                "packet_id": packet_id,
                "participant_id": participant_id,
                "stimuli": [
                    {
                        "display_stimulus_id": display_ids[row["stimulus_id"]],
                        "source_stimulus_id": row["stimulus_id"],
                        "session_id": row["session_id"],
                        "source_session_id": row["source_session_id"],
                        "crop_reference": row["crop_reference"],
                        "crop_sha256": row["crop_sha256"],
                        "image_asset": f"{display_ids[row['stimulus_id']]}.png",
                    }
                    for row in stimuli
                ],
            }
        )
    display = {
        "schema_version": RATER_DISPLAY_SCHEMA_VERSION,
        "rater_manifest_id": _keyed_digest(
            blinding_key, RATER_PACKET_SALT, "rater", rater_id
        ).hex()[:32],
        "packet_count": len(display_packets),
        "source_manifest_sha256": source_sha256,
        "asset_resolution": "opaque_basenames_in_separately_delivered_rater_asset_directory",
        "packet_order_provenance": {
            "algorithm": "hmac_sha256_rank",
            "domain_separator": PARTICIPANT_SHUFFLE_SALT,
            "key_sha256": hashlib.sha256(blinding_key).hexdigest(),
            "rater_specific": True,
        },
        "packets": display_packets,
    }
    linkage = {
        "schema_version": RATER_LINKAGE_SCHEMA_VERSION,
        "rater_id": rater_id,
        "source_manifest_sha256": source_sha256,
        "rater_manifest_id": display["rater_manifest_id"],
        "face_crop_manifest_sha256": expected_face_sha256,
        "participant_order_algorithm": "hmac_sha256_rank",
        "participant_order_domain_separator": PARTICIPANT_SHUFFLE_SALT,
        "packet_id_algorithm": "hmac_sha256_prefix_32_hex",
        "packet_id_domain_separator": RATER_PACKET_SALT,
        "blinding_key_sha256": hashlib.sha256(blinding_key).hexdigest(),
        "packets": linkage_packets,
        "not_assessable_participant_ids": [
            row["participant_id"]
            for row in payload["participants"]
            if row["status"] == "not_assessable"
        ],
    }
    _validate_blinded_display(display)
    validate_rater_linkage_manifest(linkage)
    return display, linkage


def load_stimulus_manifest(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate a restricted master stimulus manifest."""
    payload, _ = _load_unique_json_file(Path(path), "stimulus manifest")
    _validate_stimulus_payload(payload)
    return payload


def load_rater_linkage_manifest(path: str | Path) -> dict[str, Any]:
    """Load a restricted per-rater participant/source linkage manifest."""
    payload, _ = _load_unique_json_file(Path(path), "rater linkage manifest")
    validate_rater_linkage_manifest(payload)
    return payload


def validate_rater_linkage_manifest(payload: object) -> None:
    """Validate exact linkage structure before resolving blinded responses or assets."""
    top_fields = {
        "schema_version",
        "rater_id",
        "rater_manifest_id",
        "source_manifest_sha256",
        "face_crop_manifest_sha256",
        "participant_order_algorithm",
        "participant_order_domain_separator",
        "packet_id_algorithm",
        "packet_id_domain_separator",
        "blinding_key_sha256",
        "packets",
        "not_assessable_participant_ids",
    }
    if not isinstance(payload, dict) or set(payload) != top_fields:
        raise ValueError("rater linkage manifest must have the exact versioned schema")
    if payload["schema_version"] != RATER_LINKAGE_SCHEMA_VERSION:
        raise ValueError("unsupported rater linkage schema_version")
    if (
        not isinstance(payload["rater_id"], str)
        or _RATER_ID_PATTERN.fullmatch(payload["rater_id"]) is None
    ):
        raise ValueError("rater linkage rater_id is invalid")
    for field in (
        "source_manifest_sha256",
        "face_crop_manifest_sha256",
        "blinding_key_sha256",
    ):
        if not isinstance(payload[field], str) or _LOWER_SHA64.fullmatch(payload[field]) is None:
            raise ValueError(f"rater linkage {field} is invalid")
    if (
        not isinstance(payload["rater_manifest_id"], str)
        or re.fullmatch(r"[0-9a-f]{32}", payload["rater_manifest_id"]) is None
    ):
        raise ValueError("rater linkage rater_manifest_id is invalid")
    packets = payload["packets"]
    if not isinstance(packets, list):
        raise ValueError("rater linkage packets must be a list")
    seen_packets: set[str] = set()
    seen_participants: set[str] = set()
    for packet in packets:
        if not isinstance(packet, dict) or set(packet) != {
            "packet_id",
            "participant_id",
            "stimuli",
        }:
            raise ValueError("rater linkage packet schema is invalid")
        packet_id = packet["packet_id"]
        participant_id = packet["participant_id"]
        if not isinstance(packet_id, str) or re.fullmatch(r"[0-9a-f]{32}", packet_id) is None:
            raise ValueError("rater linkage packet_id is invalid")
        if (
            not isinstance(participant_id, str)
            or not participant_id.isascii()
            or not participant_id.isdecimal()
        ):
            raise ValueError("rater linkage participant_id is invalid")
        if packet_id in seen_packets or participant_id in seen_participants:
            raise ValueError("rater linkage packet/participant IDs must be unique")
        seen_packets.add(packet_id)
        seen_participants.add(participant_id)
        if not isinstance(packet["stimuli"], list) or len(packet["stimuli"]) != 6:
            raise ValueError("rater linkage packet must contain six stimuli")
        for row in packet["stimuli"]:
            if not isinstance(row, dict) or set(row) != {
                "display_stimulus_id",
                "source_stimulus_id",
                "session_id",
                "source_session_id",
                "crop_reference",
                "crop_sha256",
                "image_asset",
            }:
                raise ValueError("rater linkage stimulus schema is invalid")
            _portable_crop_reference(row["crop_reference"])
            if (
                not isinstance(row["crop_sha256"], str)
                or _LOWER_SHA64.fullmatch(row["crop_sha256"]) is None
            ):
                raise ValueError("rater linkage crop checksum is invalid")
            _opaque_asset_name(row["image_asset"])
    not_assessable = payload["not_assessable_participant_ids"]
    if not isinstance(not_assessable, list) or any(
        not isinstance(value, str) or not value.isascii() or not value.isdecimal()
        for value in not_assessable
    ):
        raise ValueError("rater linkage not-assessable participant IDs are invalid")


def blinded_asset_payloads(
    linkage_manifest: dict[str, Any] | str | Path,
    *,
    face_crop_manifest: str | Path,
) -> dict[str, bytes]:
    """Return checksum-verified crops re-encoded as metadata-free opaque RGB PNGs."""
    if isinstance(linkage_manifest, (str, Path)):
        linkage = load_rater_linkage_manifest(linkage_manifest)
    else:
        linkage = linkage_manifest
        validate_rater_linkage_manifest(linkage)
    face_path = Path(face_crop_manifest)
    load_face_crop_manifest(face_path)
    if _file_sha256(face_path) != linkage["face_crop_manifest_sha256"]:
        raise ValueError("face-crop manifest changed before rater asset export")
    root = face_path.parent.resolve()
    output: dict[str, bytes] = {}
    for packet in linkage["packets"]:
        for row in packet["stimuli"]:
            source = _regular_crop_path(root, row["crop_reference"])
            content = source.read_bytes()
            if hashlib.sha256(content).hexdigest() != row["crop_sha256"]:
                raise ValueError("crop changed before rater asset export")
            asset = _opaque_asset_name(row["image_asset"])
            try:
                from PIL import Image

                with Image.open(io.BytesIO(content)) as image:
                    pixels = image.convert("RGB")
                    encoded = io.BytesIO()
                    pixels.save(
                        encoded,
                        format="PNG",
                        optimize=False,
                        compress_level=9,
                    )
                    delivered = encoded.getvalue()
            except (OSError, ValueError) as error:
                raise ValueError("crop is not a decodable image for blinded export") from error
            if asset in output and output[asset] != delivered:
                raise ValueError("opaque rater asset collision")
            output[asset] = delivered
    return output


def canonical_stimulus_json(payload: dict[str, Any]) -> str:
    """Return canonical human-readable JSON after strict stimulus validation."""
    _validate_stimulus_payload(payload)
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def canonical_presentation_json(payload: dict[str, Any]) -> str:
    """Return deterministic JSON for a display or linkage manifest."""
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _bounded_candidate_valid_indices(
    valid_indices: tuple[int, ...], maximum: int
) -> tuple[int, ...]:
    if not valid_indices or valid_indices != tuple(sorted(set(valid_indices))):
        raise ValueError("candidate-valid grid indices must be increasing and unique")
    if len(valid_indices) <= maximum:
        return valid_indices
    ranks = bounded_visual_grid_indices(len(valid_indices), maximum)
    return tuple(valid_indices[index] for index in ranks)


def _validate_face_scan(
    value: object,
    *,
    grid_count: int,
    frame_width: int,
    frame_height: int,
    selector: dict[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("face-crop one-Hz scan must be a list")
    validated: list[dict[str, Any]] = []
    for expected_index, row in enumerate(value):
        if not isinstance(row, dict) or tuple(sorted(row)) != tuple(sorted(FACE_SCAN_FIELDS)):
            raise ValueError("face-crop scan row has an unexpected field set")
        grid_index = row["grid_index"]
        if (
            isinstance(grid_index, bool)
            or not isinstance(grid_index, int)
            or grid_index != expected_index
            or grid_index >= grid_count
        ):
            raise ValueError("face-crop scan indices must be contiguous from zero")
        timestamp = _nonnegative_decimal(row["timestamp_seconds"], "scan timestamp_seconds")
        if timestamp != grid_index:
            raise ValueError("face-crop scan timestamp must equal its integer-second index")
        detection_count = row["detection_count"]
        if (
            isinstance(detection_count, bool)
            or not isinstance(detection_count, int)
            or detection_count < 0
        ):
            raise ValueError("face-crop scan detection_count must be non-negative")
        status = row["status"]
        if status == "valid":
            if row["reason"] is not None or detection_count < 1:
                raise ValueError("valid face-crop scan row has invalid status metadata")
            box = row["box_xyxy"]
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError("valid face-crop scan row requires one candidate box")
            x1, y1, x2, y2 = (_finite_number(item, "scan box coordinate") for item in box)
            if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > frame_width or y2 > frame_height:
                raise ValueError("face-crop scan candidate box geometry is invalid")
            confidence = _finite_number(row["confidence"], "scan confidence")
            if not selector["confidence_threshold"] <= confidence <= 1:
                raise ValueError("face-crop scan candidate confidence is invalid")
            centrality = _finite_number(row["centrality_ratio"], "scan centrality_ratio")
            expected_centrality = math.hypot(
                (x1 + x2) / 2 - frame_width / 2,
                (y1 + y2) / 2 - frame_height / 2,
            ) / math.hypot(frame_width, frame_height)
            if not 0 <= centrality <= selector[
                "candidate_centrality_max_ratio"
            ] or not math.isclose(centrality, expected_centrality, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("face-crop scan candidate centrality is invalid")
            validated.append(
                {
                    "grid_index": grid_index,
                    "timestamp_seconds": _decimal_string(timestamp),
                    "status": "valid",
                    "reason": None,
                    "detection_count": detection_count,
                    "box_xyxy": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "centrality_ratio": centrality,
                }
            )
            continue
        if status != "invalid":
            raise ValueError("face-crop scan status must be valid or invalid")
        reason = row["reason"]
        minimum_detections = {
            "face_not_detected": 0,
            "face_candidate_off_centre": 1,
            "face_candidate_ambiguous": 2,
        }
        if reason not in minimum_detections:
            raise ValueError("face-crop scan invalid reason is unsupported")
        if reason == "face_not_detected":
            valid_count = detection_count == 0
        else:
            valid_count = detection_count >= minimum_detections[reason]
        if not valid_count or any(
            row[field] is not None for field in ("box_xyxy", "confidence", "centrality_ratio")
        ):
            raise ValueError("invalid face-crop scan row has inconsistent detection metadata")
        validated.append(
            {
                "grid_index": grid_index,
                "timestamp_seconds": _decimal_string(timestamp),
                "status": "invalid",
                "reason": reason,
                "detection_count": detection_count,
                "box_xyxy": None,
                "confidence": None,
                "centrality_ratio": None,
            }
        )
    return validated


def _validate_face_clip(value: object, *, root: Path, selector: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or tuple(sorted(value)) != tuple(sorted(FACE_CLIP_FIELDS)):
        raise ValueError("face-crop clip must have the exact duration/grid/frames schema")
    source_video_sha256 = value["source_video_sha256"]
    if (
        not isinstance(source_video_sha256, str)
        or _LOWER_SHA64.fullmatch(source_video_sha256) is None
    ):
        raise ValueError("face-crop source_video_sha256 must be 64 lowercase hex digits")
    duration = _positive_decimal(value["duration_seconds"], "clip duration_seconds")
    stream_duration = _positive_decimal(
        value["stream_duration_seconds"], "clip stream_duration_seconds"
    )
    grid_count = value["full_one_hz_grid_count"]
    if isinstance(grid_count, bool) or not isinstance(grid_count, int) or grid_count <= 0:
        raise ValueError("full_one_hz_grid_count must be a positive integer")
    if grid_count != max(1, int(duration)):
        raise ValueError("full_one_hz_grid_count does not match floor(duration_seconds)")
    frame_width = value["frame_width"]
    frame_height = value["frame_height"]
    frame_rate_numerator = value["frame_rate_numerator"]
    frame_rate_denominator = value["frame_rate_denominator"]
    nominal_numerator = value["nominal_frame_rate_numerator"]
    nominal_denominator = value["nominal_frame_rate_denominator"]
    time_base_numerator = value["time_base_numerator"]
    time_base_denominator = value["time_base_denominator"]
    frame_count = value["frame_count"]
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in (
            frame_width,
            frame_height,
            frame_rate_numerator,
            frame_rate_denominator,
            nominal_numerator,
            nominal_denominator,
            time_base_numerator,
            time_base_denominator,
            frame_count,
        )
    ):
        raise ValueError("face-crop source frame geometry/rate must be positive integers")
    stream_start = _nonnegative_decimal(
        value["stream_start_time_seconds"], "clip stream_start_time_seconds"
    )
    if stream_start != 0 or value["constant_frame_rate"] is not True:
        raise ValueError("face-crop source stream must be the frozen zero-start CFR stream")
    if (
        nominal_numerator * frame_rate_denominator != frame_rate_numerator * nominal_denominator
        or time_base_numerator * frame_rate_numerator
        != time_base_denominator * frame_rate_denominator
    ):
        raise ValueError("face-crop nominal/average rate and time base are inconsistent")
    expected_frame_count = (
        stream_duration * Decimal(frame_rate_numerator) / Decimal(frame_rate_denominator)
    )
    if abs(expected_frame_count - Decimal(frame_count)) > Decimal("0.001"):
        raise ValueError("face-crop frame count does not match the CFR stream duration")

    scan = _validate_face_scan(
        value["scan"],
        grid_count=grid_count,
        frame_width=frame_width,
        frame_height=frame_height,
        selector=selector,
    )
    active_start = value["active_start_grid_index"]
    active_end = value["active_end_grid_index_exclusive"]
    active_close_reason = value["active_close_reason"]
    valid_indices = value["valid_candidate_grid_indices"]
    selected = value["selected_grid_indices"]
    if not isinstance(valid_indices, list) or not isinstance(selected, list):
        raise ValueError("face-crop valid/selected grid indices must be lists")
    if any(
        isinstance(index, bool) or not isinstance(index, int)
        for index in (*valid_indices, *selected)
    ):
        raise ValueError("face-crop valid/selected grid indices must be integers")
    if active_start is None or active_end is None or active_close_reason is None:
        if not (active_start is None and active_end is None and active_close_reason is None):
            raise ValueError("face-crop active interval must be wholly present or absent")
        if valid_indices or selected or any(row["status"] == "valid" for row in scan):
            raise ValueError("face-crop missing active interval cannot retain candidate points")
        expected = ()
    else:
        if (
            isinstance(active_start, bool)
            or not isinstance(active_start, int)
            or isinstance(active_end, bool)
            or not isinstance(active_end, int)
            or not 0 <= active_start < active_end <= grid_count
        ):
            raise ValueError("face-crop active interval indices are invalid")
        if active_start > selector["first_valid_max_seconds"]:
            raise ValueError("face-crop active start exceeds the frozen boundary")
        if active_close_reason not in {"terminal_invalid_run", "end_of_stream"}:
            raise ValueError("face-crop active close reason is invalid")
        observed_valid = tuple(
            row["grid_index"]
            for row in scan
            if row["status"] == "valid" and active_start <= row["grid_index"] < active_end
        )
        if observed_valid != tuple(row["grid_index"] for row in scan if row["status"] == "valid"):
            raise ValueError("face-crop scan contains valid points outside the active interval")
        if (
            not observed_valid
            or active_start != observed_valid[0]
            or active_end != observed_valid[-1] + 1
        ):
            raise ValueError("face-crop active interval does not bind its valid endpoints")
        if tuple(valid_indices) != observed_valid:
            raise ValueError("face-crop valid candidate indices do not match the one-Hz scan")
        if active_close_reason == "end_of_stream" and len(scan) != grid_count:
            raise ValueError("end-of-stream active closure requires a complete source scan")
        if active_close_reason == "terminal_invalid_run":
            terminal = selector["terminal_invalid_run_seconds"]
            if len(scan) < terminal or any(row["status"] != "invalid" for row in scan[-terminal:]):
                raise ValueError("terminal active closure lacks the frozen invalid run")
            if (
                scan[-1]["grid_index"] - active_start + 1
                < selector["minimum_active_elapsed_seconds"]
            ):
                raise ValueError("terminal active closure occurred before the frozen minimum")
        if len(observed_valid) >= selector["minimum_valid_candidate_seconds"]:
            expected = _bounded_candidate_valid_indices(
                observed_valid, selector["maximum_retained_samples"]
            )
        else:
            expected = ()
        if tuple(selected) != expected:
            raise ValueError("face-crop selected indices drifted from valid endpoint ranks")
    status = value["status"]
    failure = value["failure"]
    if status not in {"success", "failure"}:
        raise ValueError("face-crop clip status must be success or failure")
    if status == "success" and failure is not None:
        raise ValueError("successful face-crop clip cannot contain a failure")
    if status == "failure":
        if not isinstance(failure, dict) or tuple(sorted(failure)) != tuple(
            sorted(FACE_FAILURE_FIELDS)
        ):
            raise ValueError("failed face-crop clip requires an exact structured failure")
        if not all(
            isinstance(failure[field], str) and _clean_text(failure[field])
            for field in FACE_FAILURE_FIELDS
        ):
            raise ValueError("face-crop failure values must be trimmed non-empty text")
    if status == "success" and (
        active_start is None
        or len(valid_indices) < selector["minimum_valid_candidate_seconds"]
        or not expected
    ):
        raise ValueError("successful face-crop clip lacks a valid active interview")
    threshold = float(selector["confidence_threshold"])
    scan_by_index = {row["grid_index"]: row for row in scan}
    frames = value["frames"]
    if not isinstance(frames, list):
        raise ValueError("face-crop clip frames must be a list")
    validated: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    seen_timestamps: set[str] = set()
    seen_grid_indices: set[int] = set()
    previous_grid_index = -1
    for row in frames:
        if not isinstance(row, dict) or tuple(sorted(row)) != tuple(sorted(FACE_FRAME_FIELDS)):
            raise ValueError("face-crop frame must have the exact versioned row schema")
        path = _portable_crop_reference(row["path"])
        if path in seen_paths:
            raise ValueError("face-crop paths must be unique within a session")
        _regular_crop_path(root, path)
        grid_index = row["grid_index"]
        if (
            isinstance(grid_index, bool)
            or not isinstance(grid_index, int)
            or grid_index not in expected
            or grid_index <= previous_grid_index
        ):
            raise ValueError("face-crop grid indices must be selected, unique and increasing")
        timestamp = _nonnegative_decimal(row["timestamp_seconds"], "frame timestamp_seconds")
        timestamp_text = _decimal_string(timestamp)
        if timestamp != grid_index:
            raise ValueError("face-crop timestamp_seconds must equal its one-second grid index")
        if timestamp_text in seen_timestamps or grid_index in seen_grid_indices:
            raise ValueError("face-crop timestamps and grid indices must be distinct")
        confidence = _finite_number(row["confidence"], "frame confidence")
        if not threshold <= confidence <= 1:
            raise ValueError("face-crop confidence is below the selector threshold or above one")
        box = row["box_xyxy"]
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("face-crop box_xyxy must contain four values")
        x1, y1, x2, y2 = (_finite_number(item, "box coordinate") for item in box)
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > frame_width or y2 > frame_height:
            raise ValueError("face-crop box_xyxy geometry is invalid")
        scan_row = scan_by_index.get(grid_index)
        if (
            scan_row is None
            or scan_row["status"] != "valid"
            or scan_row["box_xyxy"] != [x1, y1, x2, y2]
            or scan_row["confidence"] != confidence
        ):
            raise ValueError("face-crop frame does not match its candidate-valid scan row")
        crop_box = row["crop_box_xyxy"]
        if (
            not isinstance(crop_box, list)
            or len(crop_box) != 4
            or any(isinstance(item, bool) or not isinstance(item, int) for item in crop_box)
        ):
            raise ValueError("face-crop crop_box_xyxy must contain four integers")
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
        if (
            crop_x1 < 0
            or crop_y1 < 0
            or crop_x2 <= crop_x1
            or crop_y2 <= crop_y1
            or crop_x2 > frame_width
            or crop_y2 > frame_height
            or crop_x2 - crop_x1 != crop_y2 - crop_y1
        ):
            raise ValueError("face-crop crop_box_xyxy geometry is invalid")
        raw_side = max(x2 - x1, y2 - y1)
        if raw_side < selector["minimum_crop_side_pixels"]:
            raise ValueError("face-crop detector box is below the minimum crop side")
        expected_side = min(
            int(math.ceil(raw_side * selector["crop_scale"])), frame_width, frame_height
        )
        expected_left = min(
            max(int(math.floor((x1 + x2) / 2 - expected_side / 2)), 0),
            frame_width - expected_side,
        )
        expected_top = min(
            max(int(math.floor((y1 + y2) / 2 - expected_side / 2)), 0),
            frame_height - expected_side,
        )
        if crop_box != [
            expected_left,
            expected_top,
            expected_left + expected_side,
            expected_top + expected_side,
        ]:
            raise ValueError("face-crop box drifted from the frozen square geometry")
        source_frame_index = row["source_frame_index"]
        if (
            isinstance(source_frame_index, bool)
            or not isinstance(source_frame_index, int)
            or source_frame_index < 0
        ):
            raise ValueError("face-crop source_frame_index must be non-negative")
        scaled_numerator = grid_index * frame_rate_numerator
        expected_frame_index = (2 * scaled_numerator + frame_rate_denominator) // (
            2 * frame_rate_denominator
        )
        if source_frame_index != expected_frame_index or source_frame_index >= frame_count:
            raise ValueError("face-crop source frame index drifted from the CFR grid")
        decoded_timestamp = _nonnegative_decimal(
            row["decoded_timestamp_seconds"], "frame decoded_timestamp_seconds"
        )
        expected_decoded = Decimal(source_frame_index * frame_rate_denominator) / Decimal(
            frame_rate_numerator
        )
        if abs(decoded_timestamp - expected_decoded) > Decimal("0.000000000001"):
            raise ValueError("face-crop decoded timestamp does not match source frame/rate")
        crop_sha256 = row["crop_sha256"]
        if not isinstance(crop_sha256, str) or _LOWER_SHA64.fullmatch(crop_sha256) is None:
            raise ValueError("face-crop crop_sha256 must be a lowercase SHA-256")
        crop_file = _regular_crop_path(root, path)
        if _file_sha256(crop_file) != crop_sha256:
            raise ValueError("face-crop checksum does not match the crop file")
        validated.append(
            {
                "path": path,
                "crop_sha256": crop_sha256,
                "timestamp_seconds": timestamp_text,
                "grid_index": grid_index,
                "source_frame_index": source_frame_index,
                "decoded_timestamp_seconds": _decimal_string(decoded_timestamp),
                "box_xyxy": [x1, y1, x2, y2],
                "crop_box_xyxy": list(crop_box),
                "confidence": confidence,
            }
        )
        seen_paths.add(path)
        seen_timestamps.add(timestamp_text)
        seen_grid_indices.add(grid_index)
        previous_grid_index = grid_index
    if status == "success" and [row["grid_index"] for row in validated] != list(expected):
        raise ValueError("successful face-crop clip must cover every selected grid point")
    if status == "failure" and validated:
        raise ValueError("failed face-crop clip cannot publish partial candidate crops")
    return {
        "source_video_sha256": source_video_sha256,
        "duration_seconds": _decimal_string(duration),
        "stream_duration_seconds": _decimal_string(stream_duration),
        "full_one_hz_grid_count": grid_count,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "frame_rate_numerator": frame_rate_numerator,
        "frame_rate_denominator": frame_rate_denominator,
        "nominal_frame_rate_numerator": nominal_numerator,
        "nominal_frame_rate_denominator": nominal_denominator,
        "time_base_numerator": time_base_numerator,
        "time_base_denominator": time_base_denominator,
        "frame_count": frame_count,
        "stream_start_time_seconds": _decimal_string(stream_start),
        "constant_frame_rate": True,
        "scan": scan,
        "active_start_grid_index": active_start,
        "active_end_grid_index_exclusive": active_end,
        "active_close_reason": active_close_reason,
        "valid_candidate_grid_indices": list(valid_indices),
        "selected_grid_indices": list(expected),
        "status": status,
        "failure": dict(failure) if isinstance(failure, dict) else None,
        "frames": validated,
    }


def _nearest_distinct(
    rows: list[dict[str, Any]], active_start: int, active_end: int
) -> tuple[dict[str, Any], ...] | None:
    if len(rows) < 3 or len({row["timestamp_seconds"] for row in rows}) < 3:
        return None
    if (
        isinstance(active_start, bool)
        or not isinstance(active_start, int)
        or isinstance(active_end, bool)
        or not isinstance(active_end, int)
        or active_start >= active_end
    ):
        raise ValueError("annotation active interval is invalid")
    remaining = list(rows)
    selected = []
    start_decimal = Decimal(active_start)
    duration_decimal = Decimal(active_end - active_start)
    for fraction in TARGET_FRACTIONS:
        target = start_decimal + duration_decimal * fraction
        row = min(
            remaining,
            key=lambda candidate: (
                abs(Decimal(candidate["timestamp_seconds"]) - target),
                Decimal(candidate["timestamp_seconds"]),
                candidate["path"].encode("utf-8"),
            ),
        )
        selected.append(row)
        remaining.remove(row)
    return tuple(selected)


def _validate_stimulus_payload(payload: object) -> None:
    if not isinstance(payload, dict) or set(payload) != _STIMULUS_TOP_FIELDS:
        raise ValueError("stimulus manifest must have the exact versioned top-level schema")
    if payload["schema_version"] != STIMULUS_SCHEMA_VERSION:
        raise ValueError("unsupported stimulus manifest schema_version")
    mapping_provenance = payload["session_mapping"]
    if mapping_provenance is not None:
        if not isinstance(mapping_provenance, dict) or set(mapping_provenance) != set(
            SESSION_MAPPING_PROVENANCE_FIELDS
        ):
            raise ValueError("stimulus session mapping provenance is invalid")
        if mapping_provenance["schema_version"] != MIT_SESSION_MAPPING_SCHEMA_VERSION:
            raise ValueError("stimulus session mapping schema version is invalid")
        for field in SESSION_MAPPING_PROVENANCE_FIELDS[1:]:
            if _LOWER_SHA64.fullmatch(str(mapping_provenance[field])) is None:
                raise ValueError("stimulus session mapping checksum is invalid")
    if payload["subject_role"] != "candidate" or payload["packet_size"] != 6:
        raise ValueError("stimulus manifest candidate/packet contract is invalid")
    if payload["target_fractions"] != ["0.2", "0.5", "0.8"]:
        raise ValueError("stimulus target fractions are invalid")
    provenance = payload["selection_provenance"]
    expected_provenance = {
        "face_crop_manifest_schema_version",
        "face_crop_manifest_sha256",
        "selector_sha256",
        "selection_algorithm",
        "time_basis",
        "selection_tie_break",
        "stimulus_id_algorithm",
        "stimulus_id_salt",
        "within_packet_shuffle_algorithm",
        "within_packet_shuffle_salt",
        "duration_probe",
        "duration_probe_version",
        "duration_probe_build_sha256",
        "duration_tolerance_seconds",
    }
    if not isinstance(provenance, dict) or set(provenance) != expected_provenance:
        raise ValueError("stimulus selection provenance is required")
    if _LOWER_SHA64.fullmatch(str(provenance["face_crop_manifest_sha256"])) is None:
        raise ValueError("stimulus face-crop manifest checksum is invalid")
    if _LOWER_SHA64.fullmatch(str(provenance["selector_sha256"])) is None:
        raise ValueError("stimulus selector checksum is invalid")
    if _LOWER_SHA64.fullmatch(str(provenance["duration_probe_build_sha256"])) is None:
        raise ValueError("stimulus duration-probe build checksum is invalid")
    if provenance["duration_probe"] != "ffprobe_format_duration":
        raise ValueError("stimulus duration-probe method is invalid")
    if not isinstance(provenance["duration_probe_version"], str) or not provenance[
        "duration_probe_version"
    ].startswith("ffprobe version "):
        raise ValueError("stimulus duration-probe version is invalid")
    if provenance["duration_tolerance_seconds"] != str(_DURATION_TOLERANCE_SECONDS):
        raise ValueError("stimulus duration tolerance is invalid")
    if (
        provenance["selection_algorithm"]
        != "nearest_distinct_timestamp_to_each_active_interval_target"
        or provenance["time_basis"] != "candidate_face_active_interval_half_open"
    ):
        raise ValueError("stimulus active-interval selection contract is invalid")
    prohibited = payload["rater_display_prohibited_fields"]
    expected_prohibited = {
        "participant_id",
        "session_id",
        "source_session_id",
        "crop_reference",
        "condition",
        "phase",
        "transcript",
        "outcome",
        "prediction",
        "proxy",
    }
    if (
        not isinstance(prohibited, list)
        or len(prohibited) != len(set(prohibited))
        or set(prohibited) != expected_prohibited
    ):
        raise ValueError("rater display prohibited fields must be explicit")
    participants = payload["participants"]
    if not isinstance(participants, list) or not participants:
        raise ValueError("stimulus participants must be a non-empty list")
    seen_participants: set[str] = set()
    seen_stimuli: set[str] = set()
    seen_source_sessions: set[str] = set()
    for packet in participants:
        if not isinstance(packet, dict) or set(packet) != _STIMULUS_PACKET_FIELDS:
            raise ValueError("stimulus packet schema is invalid")
        participant_id = packet["participant_id"]
        if (
            not isinstance(participant_id, str)
            or not participant_id.isascii()
            or not participant_id.isdecimal()
        ):
            raise ValueError("stimulus participant_id must be canonical ASCII decimal")
        if participant_id in seen_participants:
            raise ValueError("stimulus participant IDs must be unique")
        seen_participants.add(participant_id)
        if packet["status"] not in {"assessable", "not_assessable"}:
            raise ValueError("stimulus packet status is invalid")
        reasons = packet["assessment_reasons"]
        if not isinstance(reasons, list) or not all(
            isinstance(value, str) and _clean_text(value) for value in reasons
        ):
            raise ValueError("stimulus assessment_reasons are invalid")
        session_qc = packet["session_qc"]
        if not isinstance(session_qc, list) or len(session_qc) != 2:
            raise ValueError("stimulus session_qc must contain exactly two sessions")
        qc_by_session: dict[str, dict[str, Any]] = {}
        for row in session_qc:
            if not isinstance(row, dict) or set(row) != {
                "session_id",
                "source_session_id",
                "face_boundary_status",
                "failure_code",
                "active_start_seconds",
                "active_end_seconds_exclusive",
                "active_duration_seconds",
                "valid_distinct_crop_count",
                "required_crop_count",
            }:
                raise ValueError("stimulus session_qc schema is invalid")
            if (
                not isinstance(row["session_id"], str)
                or not isinstance(row["source_session_id"], str)
                or isinstance(row["valid_distinct_crop_count"], bool)
                or not isinstance(row["valid_distinct_crop_count"], int)
                or row["valid_distinct_crop_count"] < 0
                or row["required_crop_count"] != 3
            ):
                raise ValueError("stimulus session_qc values are invalid")
            if row["session_id"] in qc_by_session:
                raise ValueError("stimulus session_qc IDs must be unique")
            qc_by_session[row["session_id"]] = row
            status = row["face_boundary_status"]
            failure_code = row["failure_code"]
            start = row["active_start_seconds"]
            end = row["active_end_seconds_exclusive"]
            duration = row["active_duration_seconds"]
            if status == "success":
                if (
                    failure_code is not None
                    or isinstance(start, bool)
                    or not isinstance(start, int)
                    or isinstance(end, bool)
                    or not isinstance(end, int)
                    or not 0 <= start < end
                    or duration != end - start
                ):
                    raise ValueError("successful stimulus session_qc active interval is invalid")
            elif status == "failure":
                if not isinstance(failure_code, str) or not _clean_text(failure_code):
                    raise ValueError("failed stimulus session_qc requires a failure code")
                if (start, end, duration) != (None, None, None) and (
                    isinstance(start, bool)
                    or not isinstance(start, int)
                    or isinstance(end, bool)
                    or not isinstance(end, int)
                    or not 0 <= start < end
                    or duration != end - start
                ):
                    raise ValueError("failed stimulus session_qc active interval is invalid")
            else:
                raise ValueError("stimulus session_qc face boundary status is invalid")
        if set(qc_by_session) != {f"P{participant_id}", f"PP{participant_id}"}:
            raise ValueError("stimulus session_qc IDs do not match the participant pair")
        source_ids = [row["source_session_id"] for row in session_qc]
        if len(set(source_ids)) != 2:
            raise ValueError("stimulus source session IDs must be unique")
        if seen_source_sessions.intersection(source_ids):
            raise ValueError("stimulus source session IDs must be globally unique")
        seen_source_sessions.update(source_ids)
        for row in session_qc:
            try:
                canonical_id, canonical_participant, canonical_phase = parse_session_id(
                    row["session_id"]
                )
                source_id, _source_participant, source_phase = parse_session_id(
                    row["source_session_id"]
                )
            except ValueError as error:
                raise ValueError("stimulus source session ID is invalid") from error
            if (
                canonical_id != row["session_id"]
                or source_id != row["source_session_id"]
                or canonical_participant != participant_id
                or source_phase != canonical_phase
            ):
                raise ValueError("stimulus source session mapping is invalid")
        assessable_qc = all(
            row["face_boundary_status"] == "success"
            and row["valid_distinct_crop_count"] >= row["required_crop_count"]
            for row in session_qc
        )
        if (packet["status"] == "assessable") != (assessable_qc and not reasons):
            raise ValueError("stimulus packet status does not match session QC")
        if packet["status"] == "not_assessable" and not reasons:
            raise ValueError("not-assessable stimulus packet requires an assessment reason")
        stimuli = packet["stimuli"]
        expected_count = 6 if packet["status"] == "assessable" else 0
        if not isinstance(stimuli, list) or len(stimuli) != expected_count:
            raise ValueError("stimulus count does not match packet status")
        orders: set[int] = set()
        observed_targets: set[tuple[str, str]] = set()
        offsets: list[Decimal] = []
        for row in stimuli:
            if not isinstance(row, dict) or set(row) != _STIMULUS_ROW_FIELDS:
                raise ValueError("stimulus row schema is invalid")
            if row["stimulus_id"] in seen_stimuli:
                raise ValueError("stimulus IDs must be globally unique")
            seen_stimuli.add(row["stimulus_id"])
            if (
                not isinstance(row["crop_sha256"], str)
                or _LOWER_SHA64.fullmatch(row["crop_sha256"]) is None
            ):
                raise ValueError("stimulus crop_sha256 is invalid")
            _portable_crop_reference(row["crop_reference"])
            session_id = row["source_session_id"]
            canonical_session_id = row["session_id"]
            fraction_text = row["target_fraction"]
            if (
                canonical_session_id not in qc_by_session
                or qc_by_session[canonical_session_id]["source_session_id"] != session_id
                or fraction_text not in payload["target_fractions"]
            ):
                raise ValueError("stimulus source session or target fraction is invalid")
            target_key = (canonical_session_id, fraction_text)
            if target_key in observed_targets:
                raise ValueError("stimulus session target fractions must be unique")
            observed_targets.add(target_key)
            grid_index = row["grid_index"]
            if isinstance(grid_index, bool) or not isinstance(grid_index, int) or grid_index < 0:
                raise ValueError("stimulus grid_index must be a non-negative integer")
            numeric: dict[str, Decimal] = {}
            for field in ("target_seconds", "actual_seconds", "absolute_offset_seconds"):
                raw = row[field]
                if not isinstance(raw, str):
                    raise ValueError(f"stimulus {field} must be canonical decimal text")
                value = _nonnegative_decimal(raw, f"stimulus {field}")
                if raw != _decimal_string(value):
                    raise ValueError(f"stimulus {field} must be canonical decimal text")
                numeric[field] = value
            qc = qc_by_session[canonical_session_id]
            start = Decimal(qc["active_start_seconds"])
            end = Decimal(qc["active_end_seconds_exclusive"])
            fraction = Decimal(fraction_text)
            expected_target = start + (end - start) * fraction
            if (
                numeric["target_seconds"] != expected_target
                or numeric["actual_seconds"] != Decimal(grid_index)
                or not start <= numeric["actual_seconds"] < end
                or numeric["absolute_offset_seconds"]
                != abs(numeric["actual_seconds"] - expected_target)
            ):
                raise ValueError("stimulus timing is not bound to the active interview interval")
            target_index = payload["target_fractions"].index(fraction_text)
            expected_stimulus_id = hashlib.sha256(
                STIMULUS_SALT.encode("ascii")
                + b"\0"
                + participant_id.encode("ascii")
                + b"\0"
                + canonical_session_id.encode("ascii")
                + b"\0"
                + session_id.encode("ascii")
                + b"\0"
                + (
                    _canonical_sha256(mapping_provenance).encode("ascii")
                    if mapping_provenance is not None
                    else b"synthetic-no-mapping"
                )
                + b"\0"
                + str(target_index).encode("ascii")
            ).hexdigest()[:24]
            if row["stimulus_id"] != expected_stimulus_id:
                raise ValueError("stimulus ID is not bound to its active-interval target")
            offsets.append(numeric["absolute_offset_seconds"])
            if isinstance(row["display_order"], bool) or not isinstance(row["display_order"], int):
                raise ValueError("stimulus display_order must be an integer")
            orders.add(row["display_order"])
        if orders != set(range(1, expected_count + 1)):
            raise ValueError("stimulus display_order must be contiguous")
        expected_targets = (
            {
                (session_id, fraction)
                for session_id in qc_by_session
                for fraction in payload["target_fractions"]
            }
            if packet["status"] == "assessable"
            else set()
        )
        if observed_targets != expected_targets:
            raise ValueError("stimulus packet does not cover every active-interval target")
        sensitivity_eligible = bool(stimuli) and all(value <= 30 for value in offsets)
        if (
            not isinstance(packet["offset_30s_sensitivity_eligible"], bool)
            or packet["offset_30s_sensitivity_eligible"] != sensitivity_eligible
        ):
            raise ValueError("stimulus offset sensitivity eligibility is invalid")


def _validate_blinded_display(payload: dict[str, Any]) -> None:
    prohibited_key_tokens = {
        "participant",
        "session",
        "condition",
        "phase",
        "reference",
        "transcript",
        "outcome",
        "prediction",
        "proxy",
    }

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                parts = set(re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_").split("_"))
                if parts & prohibited_key_tokens:
                    raise ValueError(f"blinded display contains prohibited key {key!r}")
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str) and ("/" in value or "\\" in value):
            raise ValueError("blinded display contains a path-like string")

    visit(payload)


def _load_unique_json_file(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    try:
        raw = path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"cannot read a valid {label}") from error
    return payload, raw


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _portable_crop_reference(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or "\0" in value:
        raise ValueError("crop reference must be trimmed non-empty text")
    if "\\" in value:
        raise ValueError("crop reference must use portable forward slashes")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("crop reference must be a canonical relative path")
    if path.as_posix() != value:
        raise ValueError("crop reference must be a canonical relative path")
    return value


def _regular_crop_path(root: Path, reference: str) -> Path:
    relative = PurePosixPath(reference)
    current = root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError("crop references may not contain symlinks")
    resolved = current.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("crop reference is unavailable beneath the manifest root")
    return resolved


def _regular_source_video(root: Path, reference: str) -> Path:
    portable = _portable_crop_reference(reference)
    return _regular_crop_path(root, portable)


def _opaque_asset_name(value: object) -> str:
    if not isinstance(value, str) or PurePosixPath(value).name != value or "\\" in value:
        raise ValueError("rater image asset must be an opaque basename")
    stem = PurePosixPath(value).stem
    suffix = PurePosixPath(value).suffix.casefold()
    if re.fullmatch(r"[0-9a-f]{32}", stem) is None or suffix != ".png":
        raise ValueError("rater image asset must be an opaque metadata-free PNG basename")
    return value


def _ffprobe_duration(path: Path) -> Decimal:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return _positive_decimal(result.stdout.strip(), "ffprobe duration")
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("cannot verify source video duration with ffprobe") from error


def _ffprobe_provenance() -> tuple[str, str]:
    try:
        result = subprocess.run(["ffprobe", "-version"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("cannot record ffprobe version") from error
    first_line = result.stdout.splitlines()[0].strip() if result.stdout else ""
    if not first_line.startswith("ffprobe version ") or "\0" in result.stdout:
        raise ValueError("ffprobe returned an invalid version string")
    return first_line, hashlib.sha256(result.stdout.encode("utf-8")).hexdigest()


def _positive_decimal(value: object, label: str) -> Decimal:
    result = _decimal(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _nonnegative_decimal(value: object, label: str) -> Decimal:
    result = _decimal(value, label)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _decimal(value: object, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{label} must be a finite decimal number")
    try:
        result = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f"{label} must be a finite decimal number") from error
    if not result.is_finite():
        raise ValueError(f"{label} must be a finite decimal number")
    return result


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _clean_text(value: str) -> bool:
    return bool(value) and value == value.strip() and "\0" not in value


def _salted_digest(salt: str, *values: str) -> bytes:
    return hashlib.sha256(
        salt.encode("ascii") + b"\0" + b"\0".join(value.encode("utf-8") for value in values)
    ).digest()


def _keyed_digest(key: bytes, domain_separator: str, *values: str) -> bytes:
    message = (
        domain_separator.encode("ascii")
        + b"\0"
        + b"\0".join(value.encode("utf-8") for value in values)
    )
    return hmac.new(key, message, hashlib.sha256).digest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _decimal_string(value: Decimal) -> str:
    return format(value.normalize(), "f")
