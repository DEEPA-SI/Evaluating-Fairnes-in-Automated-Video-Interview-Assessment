"""Deterministic, privacy-safe inventory for the restricted MIT Interview corpus."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .mit_interview import (
    CORPUS_ID,
    EXPECTED_PARTICIPANTS,
    EXPECTED_SESSIONS,
    MITInterviewDataset,
    fold_manifest_sha256,
    load_mit_interview_dataset,
    parse_session_id,
)
from .mit_session_mapping import (
    MIT_SESSION_MAPPING_SCHEMA_VERSION,
    SESSION_MAPPING_PROVENANCE_FIELDS,
)

if TYPE_CHECKING:
    from .mit_session_mapping import SessionMappingContext

INVENTORY_SCHEMA_VERSION = 3
SMALL_PROVENANCE_FILES = (
    "README.txt",
    "PROVENANCE.md",
    "Labels/interview_transcripts_by_turkers.csv",
    "Labels/turker_scores_full_interview.csv",
)


def build_inventory(
    dataset_root: str | Path,
    *,
    session_mapping: SessionMappingContext,
) -> dict[str, Any]:
    """Validate the local distribution and return aggregate metadata without local paths.

    Raw media are checked for exact session coverage and regular-file status but are not
    content-hashed on every inventory run.  The compact source, label and provenance files are
    content-hashed.  No transcript text, worker ID, protected annotation or access URL is emitted.
    """
    if session_mapping is None:
        raise ValueError("an approved session mapping context is required")
    session_mapping.verify_current_files()
    mapping_provenance = _validated_mapping_provenance(session_mapping.provenance)
    mapping_lookup = dict(session_mapping.canonical_to_source)

    root = Path(dataset_root)
    dataset = load_mit_interview_dataset(root, session_mapping=session_mapping)
    session_mapping.verify_current_files()
    identity_relationship = _validate_mapped_dataset(dataset, mapping_lookup)
    videos = _regular_file_inventory(root, (row.video_reference for row in dataset.records))
    audios = _regular_file_inventory(root, (row.audio_reference for row in dataset.records))
    provenance = {
        reference: _small_file_summary(root / reference, root)
        for reference in SMALL_PROVENANCE_FILES
    }
    condition_counts = {
        condition: sum(row.condition == condition for row in dataset.records)
        for condition in ("pre", "post")
    }
    primary = [row.outcome.recommend_hiring for row in dataset.records]
    raw = [row.outcome.recommend_hiring_raw for row in dataset.records]
    candidate_segments = [row.transcript.candidate_segment_count for row in dataset.records]
    rating_counts = dict(dataset.outcome_inventory.individual_ratings_per_session)
    source_session_ids = {row.source_session_id for row in dataset.records}
    if set(rating_counts) != source_session_ids:
        raise ValueError("outcome inventory must retain source-session rating keys")
    inventory = {
        "corpus": CORPUS_ID,
        "inventory_schema_version": INVENTORY_SCHEMA_VERSION,
        "mode": "validated_restricted_corpus",
        "repository_data_policy": "references_and_aggregate_metadata_only",
        "session_mapping": mapping_provenance,
        "session_identity": identity_relationship,
        "join": {
            "canonical_to_source_one_to_one": True,
            "canonical_participants": len(dataset.participant_ids),
            "canonical_sessions": len(dataset.records),
            "source_sessions": len(source_session_ids),
            "complete_pre_post_pairs": len(dataset.participant_ids),
            "condition_counts": condition_counts,
            "video_source_sessions": videos["file_count"],
            "audio_source_sessions": audios["file_count"],
            "transcript_source_sessions": len(dataset.records),
            "aggregate_outcome_source_sessions": len(dataset.records),
        },
        "participant_cross_fitting": {
            "fold_count": 5,
            "fold_sizes": list(dataset.fold_sizes),
            "fixed_manifest_sha256": fold_manifest_sha256(dataset),
            "pre_post_share_fold": True,
            "role_rotation": "test=f; validation=(f+1)%5; train=remaining_three",
        },
        "media": {
            "video": videos,
            "audio": audios,
            "fresh_foundation_features_required": True,
            "supplied_facial_shore_prosody_used_by_primary_pipeline": False,
        },
        "distribution": _tree_inventory(root),
        "supplied_derived_inputs": _derived_input_inventory(root, dataset),
        "transcripts": {
            "candidate_only_parser": True,
            "session_count": len(dataset.records),
            "candidate_segment_count": sum(candidate_segments),
            "all_sessions_nonempty": all(value > 0 for value in candidate_segments),
            "excluded_unclassified_colon_markers": sum(
                row.transcript.excluded_unclassified_marker_count for row in dataset.records
            ),
            "excluded_interviewer_segments": sum(
                row.transcript.excluded_interviewer_segment_count for row in dataset.records
            ),
            "excluded_colonless_fragments": sum(
                row.transcript.excluded_colonless_fragment_count for row in dataset.records
            ),
            "retained_candidate_continuation_fragments": sum(
                row.transcript.retained_candidate_continuation_count for row in dataset.records
            ),
            "retained_candidate_continuation_characters": sum(
                row.transcript.retained_candidate_continuation_characters for row in dataset.records
            ),
            "excluded_colonless_characters": sum(
                row.transcript.excluded_colonless_characters for row in dataset.records
            ),
        },
        "outcome": {
            "primary": "RecommendHiring",
            "rater_source": "Mechanical_Turk_EM_aggregate",
            "construct": "crowd_perceived_hireability_not_actual_hiring",
            "raw_scale": [1.0, 7.0],
            "normalisation": "(raw-1)/6",
            "observed_raw_min": min(raw),
            "observed_raw_max": max(raw),
            "observed_normalised_min": min(primary),
            "observed_normalised_max": max(primary),
            "missing_count": 0,
            "source_schema": list(dataset.outcome_inventory.columns),
            "source_column_count": len(dataset.outcome_inventory.columns),
            "total_rows": dataset.outcome_inventory.total_rows,
            "aggregate_rows": dataset.outcome_inventory.aggregate_rows,
            "individual_rows": dataset.outcome_inventory.individual_rows,
            "individual_ratings_per_session": {
                "minimum": min(rating_counts.values()),
                "maximum": max(rating_counts.values()),
                "sessions_with_8": sum(value == 8 for value in rating_counts.values()),
                "sessions_with_9": sum(value == 9 for value in rating_counts.values()),
            },
            "missing_rating_cells": dataset.outcome_inventory.missing_rating_cells,
            "rating_cells_outside_documented_1_to_7_scale": (
                dataset.outcome_inventory.rating_cells_outside_documented_scale
            ),
            "rating_scale_exceptions": [
                {"row_type": row_type, "field": field, "count": count}
                for row_type, field, count in (
                    dataset.outcome_inventory.rating_scale_exception_counts
                )
            ],
            "aggregate_rating_scale_exceptions": sum(
                count
                for row_type, _, count in dataset.outcome_inventory.rating_scale_exception_counts
                if row_type == "aggregate"
            ),
            "primary_aggregate_recommend_hiring_all_on_1_to_7_scale": True,
        },
        "protected_annotations": {
            "included_in_corpus": False,
            "ordinary_inventory_inspects_manual_annotations": False,
            "manual_audit_join_unit": "participant",
        },
        "provenance_files": provenance,
    }
    if any(
        _small_file_summary(root / reference, root) != provenance[reference]
        for reference in SMALL_PROVENANCE_FILES
    ):
        raise ValueError("a compact corpus input changed during inventory construction")
    session_mapping.verify_current_files()
    return inventory


def _validated_mapping_provenance(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(SESSION_MAPPING_PROVENANCE_FIELDS):
        raise ValueError("session mapping provenance schema drifted")
    provenance: dict[str, str] = {}
    for name in SESSION_MAPPING_PROVENANCE_FIELDS:
        item = value[name]
        if not isinstance(item, str) or not item:
            raise ValueError(f"session mapping provenance field {name!r} is invalid")
        if name != "schema_version" and (
            len(item) != 64 or any(character not in "0123456789abcdef" for character in item)
        ):
            raise ValueError(f"session mapping provenance field {name!r} is not SHA-256")
        provenance[name] = item
    if provenance["schema_version"] != MIT_SESSION_MAPPING_SCHEMA_VERSION:
        raise ValueError("session mapping provenance version drifted")
    return provenance


def _validate_mapped_dataset(
    dataset: MITInterviewDataset, mapping_lookup: dict[str, str]
) -> dict[str, Any]:
    canonical_ids = tuple(row.session_id for row in dataset.records)
    source_ids = tuple(row.source_session_id for row in dataset.records)
    if (
        len(dataset.participant_ids) != EXPECTED_PARTICIPANTS
        or len(canonical_ids) != EXPECTED_SESSIONS
    ):
        raise ValueError("mapped dataset must contain exactly 69 canonical pairs/138 sessions")
    if len(mapping_lookup) != len(dataset.records) or set(mapping_lookup) != set(canonical_ids):
        raise ValueError("mapped dataset canonical-session coverage disagrees with its mapping")
    if len(canonical_ids) != len(set(canonical_ids)):
        raise ValueError("mapped dataset contains duplicate canonical session IDs")
    if len(source_ids) != len(set(source_ids)) or set(source_ids) != set(mapping_lookup.values()):
        raise ValueError("mapped dataset source-session coverage is not the approved bijection")

    non_identity: list[dict[str, str]] = []
    for row in dataset.records:
        expected_source = mapping_lookup[row.session_id]
        if row.source_session_id != expected_source:
            raise ValueError(
                f"canonical session {row.session_id} does not use its approved source session"
            )
        canonical, participant_id, condition = parse_session_id(row.session_id)
        source, _, source_condition = parse_session_id(row.source_session_id)
        if (
            canonical != row.session_id
            or source != row.source_session_id
            or row.participant_id != participant_id
            or row.condition != condition
            or source_condition != condition
        ):
            raise ValueError("mapped dataset session identity or phase contract drifted")
        expected_video = f"Videos/{row.source_session_id}.avi"
        expected_audio = f"Audio/{row.source_session_id}.wav"
        if row.video_reference != expected_video or row.audio_reference != expected_audio:
            raise ValueError("mapped dataset media references must use source_session_id")
        if row.session_id != row.source_session_id:
            non_identity.append(
                {
                    "session_id": row.session_id,
                    "source_session_id": row.source_session_id,
                }
            )

    non_identity.sort(key=lambda row: row["session_id"].encode("ascii"))
    if non_identity != [
        {"session_id": "PP32", "source_session_id": "PP33"},
        {"session_id": "PP33", "source_session_id": "PP32"},
    ]:
        raise ValueError("mapped dataset does not implement the approved MIT correction")
    return {
        "mapping_relationship": "approved_canonical_to_source_bijection",
        "canonical_ids_are_downstream_analysis_ids": True,
        "source_ids_select_raw_references": True,
        "folds_and_participant_groups_use_canonical_ids": True,
        "filename_numbers_used_as_participant_identity_evidence": False,
        "phase_preserving": True,
        "canonical_session_count": len(canonical_ids),
        "source_session_count": len(source_ids),
        "non_identity_mapping_count": len(non_identity),
        "non_identity_mappings": non_identity,
    }


def canonical_json(inventory: dict[str, Any]) -> str:
    """Serialise an inventory deterministically with a final newline."""
    return json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_inventory(inventory: dict[str, Any], output_path: str | Path) -> None:
    """Atomically write canonical JSON."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as handle:
            handle.write(canonical_json(inventory))
            temporary = Path(handle.name)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _regular_file_inventory(root: Path, references: Any) -> dict[str, Any]:
    rows: list[tuple[str, int]] = []
    identities: set[tuple[int, int]] = set()
    for reference in references:
        path = root / reference
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(root.resolve())
        ):
            raise ValueError(f"dataset reference {reference!r} must be a regular non-symlink file")
        status = path.stat()
        identity = (status.st_dev, status.st_ino)
        if identity in identities:
            raise ValueError("two sessions resolve to the same physical media file")
        identities.add(identity)
        rows.append((reference, status.st_size))
    digest = hashlib.sha256()
    for reference, byte_count in sorted(rows, key=lambda value: value[0].encode("utf-8")):
        digest.update(reference.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(byte_count).encode("ascii"))
        digest.update(b"\n")
    return {
        "file_count": len(rows),
        "total_bytes": sum(value for _, value in rows),
        "reference_size_manifest_sha256": digest.hexdigest(),
        "physical_targets_unique": True,
    }


def _small_file_summary(path: Path, root: Path) -> dict[str, int | str | bool]:
    if path.is_symlink() or (path.exists() and not path.resolve().is_relative_to(root.resolve())):
        raise ValueError(f"dataset file is unsafe: {path.name}")
    if not path.is_file():
        if path.name == "PROVENANCE.md":
            return {"available": False, "bytes": 0, "sha256": ""}
        raise ValueError(f"required dataset file is unavailable: {path.name}")
    content = path.read_bytes()
    return {
        "available": True,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _tree_inventory(root: Path) -> dict[str, Any]:
    rows: list[tuple[str, int]] = []
    resolved_root = root.resolve()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("dataset distribution must not contain symlinks")
        if not path.is_file():
            continue
        if not path.resolve().is_relative_to(resolved_root):
            raise ValueError("dataset file escapes the corpus root")
        rows.append((path.relative_to(root).as_posix(), path.stat().st_size))
    original = [row for row in rows if row[0] != "PROVENANCE.md"]
    return {
        "file_count_including_local_provenance": len(rows),
        "bytes_including_local_provenance": sum(size for _, size in rows),
        "original_distribution_file_count": len(original),
        "original_distribution_bytes": sum(size for _, size in original),
    }


def _derived_input_inventory(root: Path, dataset: MITInterviewDataset) -> dict[str, Any]:
    sessions = {row.source_session_id for row in dataset.records}
    return {
        "primary_pipeline_uses_any": False,
        "facial_features": _facial_inventory(root / "Facial_Features", root, sessions),
        "shore_smile_data": _shore_inventory(root / "SmileData", root, sessions),
        "prosody": _prosody_inventory(root / "Prosody" / "prosodic_features.csv", root, sessions),
    }


def _facial_inventory(directory: Path, root: Path, sessions: set[str]) -> dict[str, Any]:
    if not directory.exists():
        return {"available": False, "used_by_primary_pipeline": False}
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Facial_Features must be a regular directory")
    files = sorted(directory.glob("*.csv"), key=lambda path: path.name)
    observed = {path.stem for path in files}
    if observed != sessions:
        raise ValueError("Facial_Features session coverage mismatch")
    rows = blank = partial = affected = 0
    for path in files:
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Facial_Features file is unsafe")
        file_blank = 0
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration as error:
                raise ValueError("Facial_Features file is empty") from error
            if len(header) != 37 or header[-1] != "" or len(set(header[:-1])) != 36:
                raise ValueError("Facial_Features schema drifted")
            for row in reader:
                if len(row) != 37 or row[-1] != "":
                    raise ValueError("Facial_Features row width drifted")
                values = row[:-1]
                missing = sum(value == "" for value in values)
                for value in values:
                    if value == "":
                        continue
                    try:
                        numeric_value = float(value)
                    except ValueError as error:
                        raise ValueError("Facial_Features contains non-numeric data") from error
                    if not math.isfinite(numeric_value):
                        raise ValueError("Facial_Features contains non-finite data")
                if missing == 36:
                    file_blank += 1
                elif missing:
                    partial += 1
                rows += 1
        blank += file_blank
        affected += bool(file_blank)
    return {
        "available": True,
        "used_by_primary_pipeline": False,
        "reason": "defective_legacy_derived_features_inventory_only",
        "files": len(files),
        "rows": rows,
        "numeric_columns": 36,
        "trailing_empty_sentinel": True,
        "wholly_blank_rows": blank,
        "partially_blank_rows": partial,
        "files_with_wholly_blank_rows": affected,
    }


def _shore_inventory(directory: Path, root: Path, sessions: set[str]) -> dict[str, Any]:
    if not directory.exists():
        return {"available": False, "used_by_primary_pipeline": False}
    rows_by_phase: dict[str, int] = {}
    truncated_post_files: list[str] = []
    total_post_shortfall = 0
    observed: set[str] = set()
    file_count = 0
    for phase, session_prefix in (("pre", "P"), ("post", "PP")):
        phase_dir = directory / phase
        if phase_dir.is_symlink() or not phase_dir.is_dir():
            raise ValueError("SmileData phase directory is unsafe")
        phase_rows = 0
        for path in sorted(phase_dir.glob("Smoothed-features-P*.txt")):
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("SmileData file is unsafe")
            number = path.stem.removeprefix("Smoothed-features-P")
            session = f"{session_prefix}{int(number)}"
            observed.add(session)
            file_count += 1
            with path.open("r", encoding="utf-8") as handle:
                file_rows = 0
                for line in handle:
                    values = line.split()
                    if len(values) != 5:
                        raise ValueError("SmileData row width drifted")
                    parsed = [float(value) for value in values]
                    if not all(math.isfinite(value) for value in parsed):
                        raise ValueError("SmileData contains non-finite data")
                    phase_rows += 1
                    file_rows += 1
            expected_rows = _video_frame_count(root / "Videos" / f"{session}.avi") - 1
            if file_rows != expected_rows:
                if phase != "post" or file_rows >= expected_rows:
                    raise ValueError("SmileData row count differs unexpectedly from video frames")
                truncated_post_files.append(session)
                total_post_shortfall += expected_rows - file_rows
        rows_by_phase[phase] = phase_rows
    if observed != sessions:
        raise ValueError("SmileData session coverage mismatch")
    return {
        "available": True,
        "used_by_primary_pipeline": False,
        "reason": "legacy_shore_truncation_inventory_only",
        "files": file_count,
        "numeric_columns": 5,
        "rows_by_phase": rows_by_phase,
        "truncated_post_files": len(truncated_post_files),
        "truncated_post_session_ids": sorted(truncated_post_files),
        "truncated_post_total_frame_shortfall": total_post_shortfall,
    }


def _prosody_inventory(path: Path, root: Path, sessions: set[str]) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "used_by_primary_pipeline": False}
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Prosody file is unsafe")
    keys: set[str] = set()
    session_keys: set[str] = set()
    undefined = 0
    physical_widths: set[int] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if len(header) != 60 or len(set(header)) != 60:
            raise ValueError("Prosody header must contain 60 unique columns")
        affected_records = 0
        for row in reader:
            physical_widths.add(len(row))
            if not row:
                raise ValueError("Prosody contains an empty row")
            key = row[0]
            if key in keys:
                raise ValueError("Prosody contains duplicate question keys")
            keys.add(key)
            match = __import__("re").fullmatch(r"(PP|P)([1-9][0-9]*)Q([1-5])", key)
            if match is None:
                raise ValueError("Prosody question key is invalid")
            session_keys.add(f"{match.group(1)}{int(match.group(2))}")
            row_undefined = sum(value == "--undefined--" for value in row)
            undefined += row_undefined
            affected_records += bool(row_undefined)
            for value in row[1:]:
                if value == "--undefined--":
                    continue
                try:
                    numeric = float(value)
                except ValueError as error:
                    raise ValueError("Prosody contains unexpected non-numeric data") from error
                if not math.isfinite(numeric):
                    raise ValueError("Prosody contains non-finite data")
    if session_keys != sessions:
        raise ValueError("Prosody session coverage mismatch")
    expected_keys = {f"{session}Q{question}" for session in sessions for question in range(1, 6)}
    if keys != expected_keys or len(keys) != 690 or physical_widths != {59}:
        raise ValueError("Prosody 138-session by five-question/schema contract drifted")
    return {
        "available": True,
        "used_by_primary_pipeline": False,
        "reason": "malformed_legacy_schema_inventory_only",
        "rows": len(keys),
        "unique_keys": len(keys),
        "header_columns": len(header),
        "physical_row_widths": sorted(physical_widths),
        "undefined_cells": undefined,
        "records_with_undefined_cells": affected_records,
    }


def _video_frame_count(path: Path) -> int:
    if path.is_symlink() or not path.is_file():
        raise ValueError("video needed for SHORE QC is unavailable or unsafe")
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        result = int(completed.stdout.strip())
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        raise ValueError("cannot obtain video frame count for SHORE QC") from error
    if result <= 0:
        raise ValueError("video frame count must be positive")
    return result
