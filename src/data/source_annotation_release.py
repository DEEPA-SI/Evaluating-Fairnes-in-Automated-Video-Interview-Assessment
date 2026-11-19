"""Accepted release of the delivered single-rater MIT perception annotations.

The received workbook is not rewritten or represented as a multi-rater consensus.
This module binds the preserved intake evidence, retains every supplied session row,
and creates two analysis views:

* phase-aligned session ratings for Stage C (post primary; available pre sensitivity);
* post-session perceived gender for the exploratory Stage-B oracle.

Confidence was not supplied or recorded in the workbook.  It is never imputed.  Source age is
retained only in the immutable intake and is deliberately absent from this release.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from src.frozen_json import freeze_json
from src.secure_io import AnchoredDirectory, AnchoredFile

from .annotation_intake import (
    GENDER_LABELS,
    INTAKE_SCHEMA_VERSION,
    NORMALIZED_COLUMNS,
    RACIALISED_APPEARANCE_LABELS,
    RECEIPT_SCHEMA_VERSION,
    SKIN_TONE_SCALE,
    SOURCE_RECEIPT_SCHEMA_VERSION,
    audit_annotation_workbook,
    build_normalized_intake,
)
from .mit_session_mapping import (
    canonical_mit_session_mapping_json,
    canonical_to_source_lookup,
    load_mit_session_mapping,
    load_verified_session_mapping,
)

SOURCE_CODEBOOK_SCHEMA_VERSION = "mit-source-annotation-codebook-1"
SOURCE_CODEBOOK_VERSION = "mit-interview-perceived-source-2026-09-03-v1"
SOURCE_SESSION_RELEASE_SCHEMA_VERSION = "mit-source-session-annotation-release-1"
SOURCE_STAGE_B_RELEASE_SCHEMA_VERSION = "mit-source-stage-b-gender-release-1"
SOURCE_RELEASE_AUDIT_SCHEMA_VERSION = "mit-source-annotation-release-audit-1"
SOURCE_RELEASE_MANIFEST_SCHEMA_VERSION = "mit-source-annotation-release-manifest-1"
SOURCE_RELEASE_METHOD = "single_rater_phase_aligned_source_session"
SOURCE_STAGE_B_DESIGNATION = "exploratory_post_designated_not_preregistered"
DESIGNATED_STAGE_B_FIELD = "perceived_gender_presentation"

EXPECTED_SESSION_COUNT = 126
EXPECTED_POST_COUNT = 69
EXPECTED_PRE_COUNT = 57

SOURCE_SESSION_COLUMNS = (
    "schema_version",
    "source_codebook_version",
    "participant_id",
    "phase",
    "canonical_session_id",
    "source_session_id",
    "source_row_number",
    "perceived_gender_presentation",
    "perceived_racialised_appearance",
    "perceived_skin_tone",
)
SOURCE_STAGE_B_COLUMNS = (
    "schema_version",
    "source_codebook_version",
    "participant_id",
    "perceived_gender_presentation",
    "release_basis",
    "designation",
)
SOURCE_ANALYSIS_FIELDS = (
    "perceived_gender_presentation",
    "perceived_racialised_appearance",
    "perceived_skin_tone",
)

GENDER_ANALYSIS_CATEGORIES = (
    "feminine_presentation",
    "masculine_presentation",
)
APPEARANCE_ANALYSIS_CATEGORIES = (
    "white_appearance_source_category",
    "black_appearance_source_category",
    "east_asian_appearance_source_category",
    "south_asian_appearance_source_category",
    "southeast_asian_appearance_source_category",
    "hispanic_or_latino_appearance_source_category",
    "middle_eastern_or_north_african_appearance_source_category",
    "ambiguous_or_multiple_appearance_source_category",
)
SKIN_TONE_VALUES = tuple(range(1, 6))
SOURCE_RELEASE_OWNED_SOURCE_PATHS = (
    "scripts/release_source_annotations.py",
    "src/data/annotation_intake.py",
    "src/data/identity_join_audit.py",
    "src/data/mit_session_mapping.py",
    "src/data/source_annotation_release.py",
    "src/features/extraction.py",
    "src/frozen_json.py",
    "src/secure_io.py",
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_PARTICIPANT_ID = re.compile(r"^[1-9][0-9]*$", re.ASCII)
_CANONICAL_SESSION_ID = re.compile(r"^(P|PP)([1-9][0-9]*)$", re.ASCII)
_CODEBOOK_CHECKSUM = "codebook_payload_sha256"
_AUDIT_CHECKSUM = "audit_payload_sha256"
_MANIFEST_CHECKSUM = "manifest_payload_sha256"


@dataclass(frozen=True, slots=True)
class SourceAnnotationCodebook:
    """Retrospective, source-faithful vocabulary used by the accepted release."""

    codebook_version: str
    status: str
    gender_categories: tuple[str, ...]
    racialised_appearance_categories: tuple[str, ...]
    skin_tone_values: tuple[int, ...]
    confidence_values: tuple[int, ...]
    sha256: str
    designated_stage_b_field: str
    stage_b_designation: str
    release_method: str


@dataclass(frozen=True, slots=True)
class SourceSessionAnnotation:
    """One accepted source rating aligned to the exact pre/post session."""

    participant_id: str
    phase: str
    canonical_session_id: str
    source_session_id: str
    source_row_number: int
    perceived_gender_presentation: str
    perceived_racialised_appearance: str | None
    perceived_skin_tone: int


@dataclass(frozen=True, slots=True)
class SourceAnnotationRelease:
    """Validated source release returned to the Stage-B/C compatibility layer."""

    codebook: SourceAnnotationCodebook
    session_annotations: Mapping[tuple[str, str], SourceSessionAnnotation]
    post_annotations: Mapping[str, SourceSessionAnnotation]
    audit_report: Mapping[str, Any]
    bindings: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "session_annotations", MappingProxyType(dict(self.session_annotations))
        )
        object.__setattr__(self, "post_annotations", MappingProxyType(dict(self.post_annotations)))
        object.__setattr__(self, "audit_report", freeze_json(self.audit_report))
        object.__setattr__(self, "bindings", freeze_json(self.bindings))


@dataclass(frozen=True, slots=True)
class SourceAnnotationPublication:
    """Paths and checksums of one immutable accepted source release."""

    output_directory: Path
    codebook: Path
    session_release: Path
    stage_b_gender_release: Path
    audit: Path
    manifest: Path
    manifest_sha256: str


def publish_source_annotation_release(
    *,
    source_workbook_path: str | Path,
    source_receipt_path: str | Path,
    normalized_session_csv_path: str | Path,
    intake_receipt_path: str | Path,
    session_mapping_path: str | Path,
    identity_join_audit_path: str | Path,
    face_crop_manifest_path: str | Path,
    output_directory: str | Path,
    allow_dirty_code: bool = False,
) -> SourceAnnotationPublication:
    """Validate the preserved intake and immutably publish an accepted release bundle."""
    if not isinstance(allow_dirty_code, bool):
        raise TypeError("allow_dirty_code must be a boolean")
    source_workbook = _read_external_file(source_workbook_path, "source workbook")
    source_receipt = _read_external_file(source_receipt_path, "source receipt")
    normalized_csv = _read_external_file(normalized_session_csv_path, "normalized session CSV")
    intake_receipt = _read_external_file(intake_receipt_path, "intake receipt")
    session_mapping = _read_external_file(session_mapping_path, "session mapping")
    repository_root = Path(__file__).resolve().parents[2]
    module_file = _read_external_file(Path(__file__).resolve(), "release implementation")
    cli_file = _read_external_file(
        repository_root / "scripts" / "release_source_annotations.py",
        "release command",
    )
    implementation_provenance = _release_code_provenance(repository_root)
    if implementation_provenance.dirty and not allow_dirty_code:
        raise ValueError("source annotation release requires clean release-owned code")

    mapping_payload = load_mit_session_mapping(session_mapping_path)
    if (
        canonical_mit_session_mapping_json(mapping_payload).encode("utf-8")
        != session_mapping.content
    ):
        raise ValueError("session mapping changed across canonical validation")
    verified_mapping = load_verified_session_mapping(
        session_mapping_path,
        identity_join_audit_report_path=identity_join_audit_path,
        face_crop_manifest_path=face_crop_manifest_path,
    )

    evidence = _validate_intake_evidence(
        source_workbook=source_workbook,
        source_receipt=source_receipt,
        normalized_csv=normalized_csv,
        intake_receipt=intake_receipt,
        session_mapping=session_mapping,
        mapping_payload=mapping_payload,
        verified_mapping=verified_mapping,
    )
    rows = evidence["rows"]
    source_receipt_payload = evidence["source_receipt"]

    codebook_unsigned = _source_codebook_payload(source_workbook.sha256)
    codebook_payload = _signed_payload(codebook_unsigned, _CODEBOOK_CHECKSUM)
    codebook_bytes = _canonical_json_bytes(codebook_payload)

    accepted_rows = _accepted_session_rows(rows)
    session_bytes = _canonical_csv_bytes(SOURCE_SESSION_COLUMNS, accepted_rows)
    stage_b_rows = _stage_b_gender_rows(accepted_rows)
    stage_b_bytes = _canonical_csv_bytes(SOURCE_STAGE_B_COLUMNS, stage_b_rows)

    input_bindings = {
        "source_workbook_file_sha256": source_workbook.sha256,
        "source_receipt_file_sha256": source_receipt.sha256,
        "normalized_session_intake_file_sha256": normalized_csv.sha256,
        "intake_receipt_file_sha256": intake_receipt.sha256,
        "intake_schema_version": INTAKE_SCHEMA_VERSION,
        "intake_receipt_schema_version": RECEIPT_SCHEMA_VERSION,
        "source_receipt_schema_version": SOURCE_RECEIPT_SCHEMA_VERSION,
        "session_mapping_file_sha256": verified_mapping.artifact_file_sha256,
        "session_mapping_payload_sha256": verified_mapping.artifact_payload_sha256,
        "identity_join_audit_file_sha256": verified_mapping.identity_join_audit_file_sha256,
        "identity_join_audit_report_payload_sha256": (
            verified_mapping.identity_join_audit_report_payload_sha256
        ),
        "face_crop_manifest_file_sha256": verified_mapping.face_crop_manifest_file_sha256,
        "face_crop_manifest_payload_sha256": verified_mapping.face_crop_manifest_payload_sha256,
        "release_module_file_sha256": module_file.sha256,
        "release_cli_file_sha256": cli_file.sha256,
        "release_code_commit_sha": implementation_provenance.commit_sha,
        "release_code_source_tree_sha256": implementation_provenance.source_tree_sha256,
        "release_code_working_tree_dirty": implementation_provenance.dirty,
        "release_owned_source_paths": list(SOURCE_RELEASE_OWNED_SOURCE_PATHS),
    }
    output_bindings = {
        "codebook_file_sha256": _sha256(codebook_bytes),
        "codebook_payload_sha256": str(codebook_payload[_CODEBOOK_CHECKSUM]),
        "session_release_file_sha256": _sha256(session_bytes),
        "stage_b_gender_release_file_sha256": _sha256(stage_b_bytes),
    }
    audit_unsigned = _source_release_audit(
        accepted_rows,
        input_bindings=input_bindings,
        output_bindings=output_bindings,
    )
    audit_payload = _signed_payload(audit_unsigned, _AUDIT_CHECKSUM)
    audit_bytes = _canonical_json_bytes(audit_payload)

    filenames = {
        "codebook": "source_annotation_codebook.json",
        "session_release": "source_session_annotations.csv",
        "stage_b_gender_release": "source_stage_b_gender_annotations.csv",
        "audit": "source_annotation_release_audit.json",
    }
    artifacts = {
        "codebook": {
            "filename": filenames["codebook"],
            "file_sha256": _sha256(codebook_bytes),
            "file_bytes": len(codebook_bytes),
            "schema_version": SOURCE_CODEBOOK_SCHEMA_VERSION,
        },
        "session_release": {
            "filename": filenames["session_release"],
            "file_sha256": _sha256(session_bytes),
            "file_bytes": len(session_bytes),
            "schema_version": SOURCE_SESSION_RELEASE_SCHEMA_VERSION,
            "row_count": len(accepted_rows),
        },
        "stage_b_gender_release": {
            "filename": filenames["stage_b_gender_release"],
            "file_sha256": _sha256(stage_b_bytes),
            "file_bytes": len(stage_b_bytes),
            "schema_version": SOURCE_STAGE_B_RELEASE_SCHEMA_VERSION,
            "row_count": len(stage_b_rows),
        },
        "audit": {
            "filename": filenames["audit"],
            "file_sha256": _sha256(audit_bytes),
            "file_bytes": len(audit_bytes),
            "schema_version": SOURCE_RELEASE_AUDIT_SCHEMA_VERSION,
            "payload_sha256": audit_payload[_AUDIT_CHECKSUM],
        },
    }
    manifest_unsigned = {
        "schema_version": SOURCE_RELEASE_MANIFEST_SCHEMA_VERSION,
        "status": "accepted_single_rater_source_release",
        "release_method": SOURCE_RELEASE_METHOD,
        "source_codebook_version": SOURCE_CODEBOOK_VERSION,
        "inputs": input_bindings,
        "artifacts": artifacts,
        "analysis_contract": _manifest_analysis_contract(),
    }
    manifest_payload = _signed_payload(manifest_unsigned, _MANIFEST_CHECKSUM)
    manifest_bytes = _canonical_json_bytes(manifest_payload)

    verified_mapping.verify_current_files()
    output = _absolute_path(output_directory)
    with AnchoredDirectory.open(output, create=True, directory_mode=0o700) as anchor:
        with anchor.exclusive_lock():
            published = {
                "codebook": anchor.publish_bytes(filenames["codebook"], codebook_bytes),
                "session_release": anchor.publish_bytes(
                    filenames["session_release"], session_bytes
                ),
                "stage_b_gender_release": anchor.publish_bytes(
                    filenames["stage_b_gender_release"], stage_b_bytes
                ),
                "audit": anchor.publish_bytes(filenames["audit"], audit_bytes),
            }
            # The manifest is the trust marker and is therefore always published last.
            manifest_file = anchor.publish_bytes(
                "source_annotation_release_manifest.json", manifest_bytes
            )
            for name, record in published.items():
                anchor.verify_file(filenames[name], record)
            anchor.verify_file("source_annotation_release_manifest.json", manifest_file)

    verified_mapping.verify_current_files()

    # Reopen the complete bundle through the same validator before returning it.
    manifest_path = output / "source_annotation_release_manifest.json"
    loaded = load_source_annotation_release(manifest_path)
    if loaded.bindings["manifest_file_sha256"] != _sha256(manifest_bytes):
        raise RuntimeError("published source annotation manifest did not revalidate")
    if source_receipt_payload["source_file_sha256"] != source_workbook.sha256:
        raise RuntimeError("source receipt binding changed during publication")
    return SourceAnnotationPublication(
        output_directory=output,
        codebook=output / filenames["codebook"],
        session_release=output / filenames["session_release"],
        stage_b_gender_release=output / filenames["stage_b_gender_release"],
        audit=output / filenames["audit"],
        manifest=manifest_path,
        manifest_sha256=_sha256(manifest_bytes),
    )


def load_source_annotation_release(
    manifest_path: str | Path,
    *,
    expected_participant_ids: Iterable[str] | None = None,
) -> SourceAnnotationRelease:
    """Load and cross-check every member of an accepted source release bundle."""
    manifest_file_path = _absolute_path(manifest_path)
    with AnchoredDirectory.open(
        manifest_file_path.parent, create=False, directory_mode=0o700
    ) as anchor:
        with anchor.exclusive_lock():
            manifest_file = anchor.read_bytes(manifest_file_path.name)
            manifest = _canonical_json_object(manifest_file.content, "source release manifest")
            _validate_signed_payload(
                manifest,
                schema_version=SOURCE_RELEASE_MANIFEST_SCHEMA_VERSION,
                checksum_field=_MANIFEST_CHECKSUM,
                label="source release manifest",
            )
            if (
                manifest.get("status") != "accepted_single_rater_source_release"
                or manifest.get("release_method") != SOURCE_RELEASE_METHOD
                or manifest.get("source_codebook_version") != SOURCE_CODEBOOK_VERSION
                or manifest.get("analysis_contract") != _manifest_analysis_contract()
            ):
                raise ValueError("source release manifest contract is not accepted")
            _validate_manifest_inputs(manifest.get("inputs"))
            artifacts = manifest.get("artifacts")
            if not isinstance(artifacts, dict) or set(artifacts) != {
                "codebook",
                "session_release",
                "stage_b_gender_release",
                "audit",
            }:
                raise ValueError("source release manifest artifact set is invalid")
            _validate_artifact_descriptors(artifacts)
            records: dict[str, AnchoredFile] = {}
            for name, descriptor in artifacts.items():
                if not isinstance(descriptor, dict):
                    raise ValueError("source release manifest artifact descriptor is invalid")
                filename = descriptor.get("filename")
                digest = descriptor.get("file_sha256")
                size = descriptor.get("file_bytes")
                if (
                    not isinstance(filename, str)
                    or not isinstance(digest, str)
                    or _HEX64.fullmatch(digest) is None
                    or isinstance(size, bool)
                    or not isinstance(size, int)
                    or size < 0
                ):
                    raise ValueError("source release manifest artifact binding is invalid")
                record = anchor.read_bytes(filename)
                if record.sha256 != digest or record.size != size:
                    raise ValueError("source release artifact byte count drifted")
                records[name] = record

            codebook_payload = _canonical_json_object(
                records["codebook"].content, "source annotation codebook"
            )
            if (
                codebook_payload.get("source_workbook_file_sha256")
                != manifest["inputs"]["source_workbook_file_sha256"]
            ):
                raise ValueError("source codebook does not bind the preserved workbook")
            expected_codebook = _signed_payload(
                _source_codebook_payload(
                    str(manifest["inputs"]["source_workbook_file_sha256"])
                ),
                _CODEBOOK_CHECKSUM,
            )
            if codebook_payload != expected_codebook:
                raise ValueError("source annotation codebook is not the exact frozen instrument")
            codebook = _load_source_codebook(codebook_payload)
            session_rows = _load_session_release(records["session_release"].content, codebook)
            stage_b_rows = _load_stage_b_release(
                records["stage_b_gender_release"].content, codebook
            )
            audit = _canonical_json_object(
                records["audit"].content, "source annotation release audit"
            )
            _validate_source_audit(
                audit,
                manifest=manifest,
                codebook=codebook,
                session_rows=session_rows,
                stage_b_rows=stage_b_rows,
            )
            for name, record in records.items():
                anchor.verify_file(str(artifacts[name]["filename"]), record)
            anchor.verify_file(manifest_file_path.name, manifest_file)

    post = {row.participant_id: row for row in session_rows if row.phase == "post"}
    if len(post) != EXPECTED_POST_COUNT:
        raise ValueError("source release does not contain 69 unique post-session participants")
    expected = (
        None
        if expected_participant_ids is None
        else {str(value) for value in expected_participant_ids}
    )
    if expected is not None and set(post) != expected:
        raise ValueError("source release post participants do not exactly match Stage A")
    if {row["participant_id"]: row["perceived_gender_presentation"] for row in stage_b_rows} != {
        participant_id: row.perceived_gender_presentation for participant_id, row in post.items()
    }:
        raise ValueError("Stage-B gender release does not equal the post-session source view")

    bindings = {
        "manifest_file_sha256": manifest_file.sha256,
        "manifest_payload_sha256": str(manifest[_MANIFEST_CHECKSUM]),
        "codebook_file_sha256": records["codebook"].sha256,
        "codebook_payload_sha256": codebook.sha256,
        "session_release_file_sha256": records["session_release"].sha256,
        "stage_b_gender_release_file_sha256": records["stage_b_gender_release"].sha256,
        "annotation_audit_file_sha256": records["audit"].sha256,
        "annotation_audit_payload_sha256": str(audit[_AUDIT_CHECKSUM]),
    }
    by_session = {(row.participant_id, row.phase): row for row in session_rows}
    return SourceAnnotationRelease(
        codebook=codebook,
        session_annotations=by_session,
        post_annotations=post,
        audit_report=audit,
        bindings=bindings,
    )


def _validate_intake_evidence(
    *,
    source_workbook: AnchoredFile,
    source_receipt: AnchoredFile,
    normalized_csv: AnchoredFile,
    intake_receipt: AnchoredFile,
    session_mapping: AnchoredFile,
    mapping_payload: Mapping[str, Any],
    verified_mapping: Any,
) -> dict[str, Any]:
    source_receipt_payload = _canonical_json_object(
        source_receipt.content, "source workbook receipt"
    )
    intake_receipt_payload = _canonical_json_object(
        intake_receipt.content, "normalized intake receipt"
    )
    if (
        source_receipt_payload.get("schema_version") != SOURCE_RECEIPT_SCHEMA_VERSION
        or source_receipt_payload.get("status") != "source_preserved_and_verified"
        or source_receipt_payload.get("source_file_sha256") != source_workbook.sha256
        or source_receipt_payload.get("source_file_bytes") != source_workbook.size
        or source_receipt_payload.get("byte_for_byte_copy_verified") is not True
    ):
        raise ValueError("source workbook receipt does not bind the supplied workbook")
    if (
        intake_receipt_payload.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or intake_receipt_payload.get("intake_schema_version") != INTAKE_SCHEMA_VERSION
        or intake_receipt_payload.get("status")
        != "verified_provisional_session_level_intake_not_stage_bc_ready"
        or intake_receipt_payload.get("source", {}).get("file_sha256") != source_workbook.sha256
        or intake_receipt_payload.get("source", {}).get("source_receipt_sha256")
        != source_receipt.sha256
        or intake_receipt_payload.get("output", {}).get("file_sha256") != normalized_csv.sha256
        or intake_receipt_payload.get("output", {}).get("file_bytes") != normalized_csv.size
    ):
        raise ValueError("normalized intake receipt does not bind all supplied evidence")
    mapping = intake_receipt_payload.get("mapping")
    conversion = intake_receipt_payload.get("conversion")
    if (
        not isinstance(mapping, dict)
        or mapping.get("schema_version") != verified_mapping.schema_version
        or mapping.get("canonical_mapping_applied") is not True
        or mapping.get("live_evidence_bindings_verified") is not True
        or mapping.get("file_sha256") != session_mapping.sha256
        or mapping.get("payload_sha256") != verified_mapping.artifact_payload_sha256
        or mapping.get("identity_join_audit_file_sha256")
        != verified_mapping.identity_join_audit_file_sha256
        or mapping.get("identity_join_audit_report_payload_sha256")
        != verified_mapping.identity_join_audit_report_payload_sha256
        or mapping.get("face_crop_manifest_file_sha256")
        != verified_mapping.face_crop_manifest_file_sha256
        or mapping.get("face_crop_manifest_payload_sha256")
        != verified_mapping.face_crop_manifest_payload_sha256
        or not isinstance(conversion, dict)
        or conversion.get("source_skin_tone_scale") != SKIN_TONE_SCALE
        or conversion.get("source_skin_tone_numeric_scale_matches_project_contract") is not True
        or conversion.get("no_category_collapsing_or_skin_tone_rescaling") is not True
    ):
        raise ValueError("normalized intake receipt mapping or scale contract is invalid")

    if (
        verified_mapping.artifact_file_sha256 != session_mapping.sha256
        or verified_mapping.lookup != canonical_to_source_lookup(mapping_payload)
    ):
        raise ValueError("session mapping changed across evidence-bound validation")

    workbook = audit_annotation_workbook(source_workbook.content)
    expected_normalized_csv, validation = build_normalized_intake(workbook, dict(mapping_payload))
    if normalized_csv.content != expected_normalized_csv:
        raise ValueError(
            "normalized intake does not equal the workbook under the verified session mapping"
        )
    rows = _parse_normalized_intake(normalized_csv.content, workbook.rows)
    output = intake_receipt_payload["output"]
    if (
        output.get("row_count") != len(rows)
        or intake_receipt_payload.get("validation") != validation
    ):
        raise ValueError("normalized intake receipt row count is inconsistent")
    return {
        "rows": rows,
        "source_receipt": source_receipt_payload,
        "intake_receipt": intake_receipt_payload,
    }


def _validate_manifest_inputs(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("source release manifest inputs are invalid")
    digest_fields = {
        "source_workbook_file_sha256",
        "source_receipt_file_sha256",
        "normalized_session_intake_file_sha256",
        "intake_receipt_file_sha256",
        "session_mapping_file_sha256",
        "session_mapping_payload_sha256",
        "identity_join_audit_file_sha256",
        "identity_join_audit_report_payload_sha256",
        "face_crop_manifest_file_sha256",
        "face_crop_manifest_payload_sha256",
        "release_module_file_sha256",
        "release_cli_file_sha256",
        "release_code_source_tree_sha256",
    }
    required = {
        *digest_fields,
        "intake_schema_version",
        "intake_receipt_schema_version",
        "source_receipt_schema_version",
        "release_code_commit_sha",
        "release_code_working_tree_dirty",
        "release_owned_source_paths",
    }
    if (
        set(value) != required
        or any(_HEX64.fullmatch(str(value.get(field, ""))) is None for field in digest_fields)
        or re.fullmatch(r"[0-9a-f]{40}", str(value.get("release_code_commit_sha", ""))) is None
        or not isinstance(value.get("release_code_working_tree_dirty"), bool)
        or value.get("release_owned_source_paths") != list(SOURCE_RELEASE_OWNED_SOURCE_PATHS)
        or value.get("intake_schema_version") != INTAKE_SCHEMA_VERSION
        or value.get("intake_receipt_schema_version") != RECEIPT_SCHEMA_VERSION
        or value.get("source_receipt_schema_version") != SOURCE_RECEIPT_SCHEMA_VERSION
    ):
        raise ValueError("source release manifest input provenance is invalid")


def _validate_artifact_descriptors(artifacts: Mapping[str, Any]) -> None:
    expected = {
        "codebook": (SOURCE_CODEBOOK_SCHEMA_VERSION, None),
        "session_release": (SOURCE_SESSION_RELEASE_SCHEMA_VERSION, EXPECTED_SESSION_COUNT),
        "stage_b_gender_release": (
            SOURCE_STAGE_B_RELEASE_SCHEMA_VERSION,
            EXPECTED_POST_COUNT,
        ),
        "audit": (SOURCE_RELEASE_AUDIT_SCHEMA_VERSION, None),
    }
    for name, (schema_version, row_count) in expected.items():
        descriptor = artifacts.get(name)
        if (
            not isinstance(descriptor, dict)
            or descriptor.get("schema_version") != schema_version
            or (row_count is not None and descriptor.get("row_count") != row_count)
            or (row_count is None and "row_count" in descriptor)
        ):
            raise ValueError("source release artifact schema descriptor is invalid")
    audit = artifacts["audit"]
    if _HEX64.fullmatch(str(audit.get("payload_sha256", ""))) is None:
        raise ValueError("source release audit payload binding is invalid")


def _parse_normalized_intake(content: bytes, workbook_rows: Iterable[Any]) -> list[dict[str, str]]:
    rows = _canonical_csv_rows(content, NORMALIZED_COLUMNS, "normalized session intake")
    if len(rows) != EXPECTED_SESSION_COUNT:
        raise ValueError("accepted source intake must contain exactly 126 session rows")
    raw_by_source = {row.source_session_id: row for row in workbook_rows}
    if len(raw_by_source) != len(rows) or set(raw_by_source) != {
        row["source_session_id"] for row in rows
    }:
        raise ValueError("normalized intake does not exactly preserve source workbook rows")
    observed_keys: set[tuple[str, str]] = set()
    for row in rows:
        match = _CANONICAL_SESSION_ID.fullmatch(row["canonical_session_id"])
        if match is None:
            raise ValueError("normalized intake contains an invalid canonical session")
        prefix, participant_id = match.groups()
        phase = "post" if prefix == "PP" else "pre"
        key = (participant_id, phase)
        raw = raw_by_source[row["source_session_id"]]
        if (
            key in observed_keys
            or row["participant_id"] != participant_id
            or row["phase"] != phase
            or row["source_row_number"] != str(raw.source_row_number)
            or row["perceived_gender_presentation_source_code"] != raw.gender_code
            or row["perceived_gender_presentation_source_label"] != GENDER_LABELS[raw.gender_code]
            or row["perceived_skin_tone_source_value"] != str(raw.skin_tone_value)
            or row["perceived_skin_tone_source_scale"] != SKIN_TONE_SCALE
            or row["perceived_racialised_appearance_source_code"] != raw.racialised_appearance_code
            or row["perceived_racialised_appearance_source_label"]
            != RACIALISED_APPEARANCE_LABELS[raw.racialised_appearance_code]
            or not row["perceived_age_band_source_code"]
            or not row["perceived_age_band_source_label"]
        ):
            raise ValueError("normalized intake row does not match workbook or canonical identity")
        observed_keys.add(key)
    post = {participant for participant, phase in observed_keys if phase == "post"}
    pre = {participant for participant, phase in observed_keys if phase == "pre"}
    if len(post) != EXPECTED_POST_COUNT or len(pre) != EXPECTED_PRE_COUNT or not pre.issubset(post):
        raise ValueError("source intake must contain 69 post and 57 paired pre sessions")
    return rows


def _accepted_session_rows(rows: Iterable[Mapping[str, str]]) -> list[dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    for source in rows:
        gender = source["perceived_gender_presentation_source_label"]
        if source["phase"] == "post" and gender not in GENDER_ANALYSIS_CATEGORIES:
            raise ValueError("post-session gender source code is outside exact binary F/M release")
        skin = int(source["perceived_skin_tone_source_value"])
        if skin not in SKIN_TONE_VALUES:
            raise ValueError("source skin tone is outside the exact ordinal 1-5 release")
        appearance = source["perceived_racialised_appearance_source_label"]
        if appearance == "cannot_determine":
            appearance = ""
        elif appearance not in APPEARANCE_ANALYSIS_CATEGORIES:
            raise ValueError("source appearance is outside the frozen release vocabulary")
        accepted.append(
            {
                "schema_version": SOURCE_SESSION_RELEASE_SCHEMA_VERSION,
                "source_codebook_version": SOURCE_CODEBOOK_VERSION,
                "participant_id": source["participant_id"],
                "phase": source["phase"],
                "canonical_session_id": source["canonical_session_id"],
                "source_session_id": source["source_session_id"],
                "source_row_number": int(source["source_row_number"]),
                "perceived_gender_presentation": gender,
                "perceived_racialised_appearance": appearance,
                "perceived_skin_tone": skin,
            }
        )
    return sorted(
        accepted,
        key=lambda row: (int(row["participant_id"]), 0 if row["phase"] == "pre" else 1),
    )


def _stage_b_gender_rows(
    accepted_rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = [row for row in accepted_rows if row["phase"] == "post"]
    if len(rows) != EXPECTED_POST_COUNT:
        raise ValueError("Stage-B source release requires all 69 post-session ratings")
    return [
        {
            "schema_version": SOURCE_STAGE_B_RELEASE_SCHEMA_VERSION,
            "source_codebook_version": SOURCE_CODEBOOK_VERSION,
            "participant_id": row["participant_id"],
            "perceived_gender_presentation": row["perceived_gender_presentation"],
            "release_basis": "post_session_rating",
            "designation": SOURCE_STAGE_B_DESIGNATION,
        }
        for row in rows
    ]


def _source_codebook_payload(source_workbook_sha256: str) -> dict[str, Any]:
    source_gender = [
        {"source_code": code, "source_label": label} for code, label in GENDER_LABELS.items()
    ]
    source_appearance = [
        {"source_code": code, "source_label": label}
        for code, label in RACIALISED_APPEARANCE_LABELS.items()
    ]
    return {
        "schema_version": SOURCE_CODEBOOK_SCHEMA_VERSION,
        "codebook_version": SOURCE_CODEBOOK_VERSION,
        "status": "retrospectively_documented_from_embedded_source_legend",
        "source_workbook_file_sha256": source_workbook_sha256,
        "source_vocabulary": {
            "perceived_gender_presentation": source_gender,
            "perceived_racialised_appearance": source_appearance,
            "perceived_skin_tone": {
                "values": list(SKIN_TONE_VALUES),
                "scale": SKIN_TONE_SCALE,
                "ordering": "ordinal_lightest_to_darkest",
            },
        },
        "analysis_vocabulary": {
            "perceived_gender_presentation": list(GENDER_ANALYSIS_CATEGORIES),
            "perceived_racialised_appearance": list(APPEARANCE_ANALYSIS_CATEGORIES),
            "perceived_skin_tone": list(SKIN_TONE_VALUES),
            "cannot_determine": "missing_not_a_group",
        },
        "release_contract": {
            "release_method": SOURCE_RELEASE_METHOD,
            "stage_c_primary_phase": "post",
            "stage_c_sensitivity_phase": "pre_when_available",
            "post_values_never_carried_to_missing_pre": True,
            "designated_stage_b_field": DESIGNATED_STAGE_B_FIELD,
            "stage_b_designation": SOURCE_STAGE_B_DESIGNATION,
            "stage_b_gender_non_binary_source_value_handling": "fail_closed_no_mapping_or_drop",
            "skin_tone_use": "stage_c_ordinal_only",
            "confidence": "not_supplied_in_source_workbook_no_imputation",
            "source_age": "excluded_from_analysis_retained_in_immutable_intake",
        },
    }


def _manifest_analysis_contract() -> dict[str, Any]:
    return {
        "stage_c_primary": "post_session_rating_joined_to_matching_post_prediction",
        "stage_c_sensitivity": "available_pre_session_rating_joined_to_matching_pre_prediction",
        "stage_c_pre_coverage": (
            "57_of_69_incomplete_missingness_not_assumed_random"
        ),
        "missing_pre_handling": "missing_no_post_rating_carry_forward",
        "stage_b_field": DESIGNATED_STAGE_B_FIELD,
        "stage_b_release_basis": "post_session_rating_per_participant",
        "stage_b_designation": SOURCE_STAGE_B_DESIGNATION,
        "confidence": "not_supplied_in_source_workbook_no_imputation",
        "source_age": "excluded_from_analysis_retained_in_immutable_intake",
    }


def _source_release_audit(
    rows: list[dict[str, Any]],
    *,
    input_bindings: Mapping[str, Any],
    output_bindings: Mapping[str, str],
) -> dict[str, Any]:
    by_key = {(str(row["participant_id"]), str(row["phase"])): row for row in rows}
    post = [row for row in rows if row["phase"] == "post"]
    pre = [row for row in rows if row["phase"] == "pre"]
    paired = [
        (by_key[(str(row["participant_id"]), "pre")], row)
        for row in post
        if (str(row["participant_id"]), "pre") in by_key
    ]
    skin_differences = Counter(
        abs(int(before["perceived_skin_tone"]) - int(after["perceived_skin_tone"]))
        for before, after in paired
    )
    fields = SOURCE_ANALYSIS_FIELDS
    exact_matches = {
        field: sum(before[field] == after[field] for before, after in paired) for field in fields
    }
    post_counts = {
        field: {
            str(key): value
            for key, value in sorted(Counter(row[field] or "missing" for row in post).items())
        }
        for field in fields
    }
    pre_counts = {
        field: {
            str(key): value
            for key, value in sorted(Counter(row[field] or "missing" for row in pre).items())
        }
        for field in fields
    }
    return {
        "schema_version": SOURCE_RELEASE_AUDIT_SCHEMA_VERSION,
        "status": "accepted_single_rater_source_release_validated",
        "release_method": SOURCE_RELEASE_METHOD,
        "source_codebook_version": SOURCE_CODEBOOK_VERSION,
        "bindings": {**dict(input_bindings), **dict(output_bindings)},
        "coverage": {
            "source_session_count": len(rows),
            "post_primary_session_count": len(post),
            "post_primary_participant_count": len(post),
            "pre_sensitivity_session_count": len(pre),
            "pre_sensitivity_participant_count": len(pre),
            "paired_pre_post_participant_count": len(paired),
            "post_participants_without_pre_rating_count": len(post) - len(pre),
            "pre_sensitivity_coverage": (
                "57_of_69_incomplete_missingness_not_assumed_random"
            ),
        },
        "post_primary_category_counts": post_counts,
        "pre_sensitivity_category_counts": pre_counts,
        "paired_stability": {
            "interpretation": "descriptive_within_rater_session_stability_only",
            "pair_count": len(paired),
            "gender_exact_match_count": exact_matches["perceived_gender_presentation"],
            "racialised_appearance_exact_match_count": exact_matches[
                "perceived_racialised_appearance"
            ],
            "skin_tone_exact_match_count": exact_matches["perceived_skin_tone"],
            "skin_tone_absolute_difference_counts": {
                str(key): value for key, value in sorted(skin_differences.items())
            },
        },
        "measurement_limitations": {
            "rater_design": "single_rater_source_workbook",
            "inter_rater_reliability": "unavailable",
            "confidence": "not_supplied_in_source_workbook",
            "confidence_imputation": "forbidden",
            "source_codebook_timing": "retrospectively_documented_not_preregistered",
            "pre_sensitivity_missingness": (
                "12_of_69_incomplete_missingness_not_assumed_random"
            ),
            "subjective_label_correctness_against_video": "not_independently_verified",
        },
        "analysis_contract": {
            "stage_c_post": "primary_phase_aligned",
            "stage_c_pre": "sensitivity_phase_aligned_when_available",
            "stage_b_gender": SOURCE_STAGE_B_DESIGNATION,
            "stage_b_other_fields": "forbidden",
            "skin_tone": "ordinal_integer_1_to_5_stage_c_only",
            "appearance_cannot_determine": "missing",
            "source_age": "excluded",
        },
    }


def _load_source_codebook(payload: Mapping[str, Any]) -> SourceAnnotationCodebook:
    _validate_signed_payload(
        payload,
        schema_version=SOURCE_CODEBOOK_SCHEMA_VERSION,
        checksum_field=_CODEBOOK_CHECKSUM,
        label="source annotation codebook",
    )
    source_vocabulary = payload.get("source_vocabulary")
    analysis = payload.get("analysis_vocabulary")
    contract = payload.get("release_contract")
    if (
        payload.get("codebook_version") != SOURCE_CODEBOOK_VERSION
        or payload.get("status") != "retrospectively_documented_from_embedded_source_legend"
        or not isinstance(source_vocabulary, dict)
        or not isinstance(analysis, dict)
        or not isinstance(contract, dict)
        or analysis.get("perceived_gender_presentation") != list(GENDER_ANALYSIS_CATEGORIES)
        or analysis.get("perceived_racialised_appearance") != list(APPEARANCE_ANALYSIS_CATEGORIES)
        or analysis.get("perceived_skin_tone") != list(SKIN_TONE_VALUES)
        or contract.get("designated_stage_b_field") != DESIGNATED_STAGE_B_FIELD
        or contract.get("stage_b_designation") != SOURCE_STAGE_B_DESIGNATION
        or contract.get("confidence")
        != "not_supplied_in_source_workbook_no_imputation"
        or contract.get("skin_tone_use") != "stage_c_ordinal_only"
    ):
        raise ValueError("source annotation codebook contract drifted")
    return SourceAnnotationCodebook(
        codebook_version=SOURCE_CODEBOOK_VERSION,
        status=str(payload["status"]),
        gender_categories=GENDER_ANALYSIS_CATEGORIES,
        racialised_appearance_categories=APPEARANCE_ANALYSIS_CATEGORIES,
        skin_tone_values=SKIN_TONE_VALUES,
        confidence_values=(),
        sha256=str(payload[_CODEBOOK_CHECKSUM]),
        designated_stage_b_field=DESIGNATED_STAGE_B_FIELD,
        stage_b_designation=SOURCE_STAGE_B_DESIGNATION,
        release_method=SOURCE_RELEASE_METHOD,
    )


def _load_session_release(
    content: bytes, codebook: SourceAnnotationCodebook
) -> tuple[SourceSessionAnnotation, ...]:
    rows = _canonical_csv_rows(content, SOURCE_SESSION_COLUMNS, "source session release")
    if len(rows) != EXPECTED_SESSION_COUNT:
        raise ValueError("source session release must contain exactly 126 rows")
    result: list[SourceSessionAnnotation] = []
    observed: set[tuple[str, str]] = set()
    observed_source_sessions: set[str] = set()
    observed_source_rows: set[int] = set()
    for row in rows:
        participant_id = row["participant_id"]
        phase = row["phase"]
        key = (participant_id, phase)
        match = _CANONICAL_SESSION_ID.fullmatch(row["canonical_session_id"])
        expected_phase = "post" if match is not None and match.group(1) == "PP" else "pre"
        appearance = row["perceived_racialised_appearance"] or None
        try:
            source_row = int(row["source_row_number"])
            skin = int(row["perceived_skin_tone"])
        except ValueError as error:
            raise ValueError("source session release numeric field is invalid") from error
        if (
            row["schema_version"] != SOURCE_SESSION_RELEASE_SCHEMA_VERSION
            or row["source_codebook_version"] != codebook.codebook_version
            or _PARTICIPANT_ID.fullmatch(participant_id) is None
            or phase not in {"pre", "post"}
            or match is None
            or match.group(2) != participant_id
            or expected_phase != phase
            or key in observed
            or not row["source_session_id"]
            or row["source_session_id"] in observed_source_sessions
            or source_row <= 1
            or source_row in observed_source_rows
            or row["perceived_gender_presentation"] not in codebook.gender_categories
            or (
                appearance is not None
                and appearance not in codebook.racialised_appearance_categories
            )
            or skin not in codebook.skin_tone_values
        ):
            raise ValueError("source session release row violates the frozen codebook")
        observed.add(key)
        observed_source_sessions.add(row["source_session_id"])
        observed_source_rows.add(source_row)
        result.append(
            SourceSessionAnnotation(
                participant_id=participant_id,
                phase=phase,
                canonical_session_id=row["canonical_session_id"],
                source_session_id=row["source_session_id"],
                source_row_number=source_row,
                perceived_gender_presentation=row["perceived_gender_presentation"],
                perceived_racialised_appearance=appearance,
                perceived_skin_tone=skin,
            )
        )
    expected_sort = sorted(
        result,
        key=lambda row: (int(row.participant_id), 0 if row.phase == "pre" else 1),
    )
    if result != expected_sort:
        raise ValueError("source session release is not in canonical participant/phase order")
    post = {row.participant_id for row in result if row.phase == "post"}
    pre = {row.participant_id for row in result if row.phase == "pre"}
    if len(post) != EXPECTED_POST_COUNT or len(pre) != EXPECTED_PRE_COUNT or not pre <= post:
        raise ValueError("source session release phase coverage drifted")
    return tuple(result)


def _load_stage_b_release(
    content: bytes, codebook: SourceAnnotationCodebook
) -> tuple[dict[str, str], ...]:
    rows = _canonical_csv_rows(content, SOURCE_STAGE_B_COLUMNS, "Stage-B gender release")
    if len(rows) != EXPECTED_POST_COUNT:
        raise ValueError("Stage-B gender release must contain 69 participant rows")
    observed: set[str] = set()
    for row in rows:
        participant_id = row["participant_id"]
        if (
            row["schema_version"] != SOURCE_STAGE_B_RELEASE_SCHEMA_VERSION
            or row["source_codebook_version"] != codebook.codebook_version
            or _PARTICIPANT_ID.fullmatch(participant_id) is None
            or participant_id in observed
            or row["perceived_gender_presentation"] not in codebook.gender_categories
            or row["release_basis"] != "post_session_rating"
            or row["designation"] != SOURCE_STAGE_B_DESIGNATION
        ):
            raise ValueError("Stage-B gender release violates the frozen source contract")
        observed.add(participant_id)
    if rows != sorted(rows, key=lambda row: int(row["participant_id"])):
        raise ValueError("Stage-B gender release is not in canonical participant order")
    return tuple(rows)


def _validate_source_audit(
    audit: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    codebook: SourceAnnotationCodebook,
    session_rows: tuple[SourceSessionAnnotation, ...],
    stage_b_rows: tuple[dict[str, str], ...],
) -> None:
    _validate_signed_payload(
        audit,
        schema_version=SOURCE_RELEASE_AUDIT_SCHEMA_VERSION,
        checksum_field=_AUDIT_CHECKSUM,
        label="source annotation release audit",
    )
    artifacts = manifest["artifacts"]
    inputs = manifest.get("inputs")
    bindings = audit.get("bindings")
    coverage = audit.get("coverage")
    limitations = audit.get("measurement_limitations")
    post_counts = audit.get("post_primary_category_counts")
    pre_counts = audit.get("pre_sensitivity_category_counts")
    if (
        audit.get("status") != "accepted_single_rater_source_release_validated"
        or audit.get("release_method") != SOURCE_RELEASE_METHOD
        or audit.get("source_codebook_version") != codebook.codebook_version
        or not isinstance(inputs, dict)
        or not isinstance(bindings, dict)
        or any(bindings.get(key) != value for key, value in inputs.items())
        or bindings.get("codebook_file_sha256") != artifacts["codebook"]["file_sha256"]
        or bindings.get("codebook_payload_sha256") != codebook.sha256
        or bindings.get("session_release_file_sha256")
        != artifacts["session_release"]["file_sha256"]
        or bindings.get("stage_b_gender_release_file_sha256")
        != artifacts["stage_b_gender_release"]["file_sha256"]
        or not isinstance(coverage, dict)
        or coverage.get("source_session_count") != len(session_rows)
        or coverage.get("post_primary_session_count") != EXPECTED_POST_COUNT
        or coverage.get("pre_sensitivity_session_count") != EXPECTED_PRE_COUNT
        or not _valid_phase_category_counts(post_counts, EXPECTED_POST_COUNT)
        or not _valid_phase_category_counts(pre_counts, EXPECTED_PRE_COUNT)
        or not isinstance(limitations, dict)
        or limitations.get("confidence") != "not_supplied_in_source_workbook"
        or limitations.get("confidence_imputation") != "forbidden"
        or limitations.get("inter_rater_reliability") != "unavailable"
        or len(stage_b_rows) != EXPECTED_POST_COUNT
    ):
        raise ValueError("source annotation release audit is inconsistent")
    if artifacts["audit"].get("payload_sha256") != audit[_AUDIT_CHECKSUM]:
        raise ValueError("source release manifest audit payload binding drifted")
    accepted_rows = [
        {
            "participant_id": row.participant_id,
            "phase": row.phase,
            "perceived_gender_presentation": row.perceived_gender_presentation,
            "perceived_racialised_appearance": row.perceived_racialised_appearance or "",
            "perceived_skin_tone": row.perceived_skin_tone,
        }
        for row in session_rows
    ]
    output_bindings = {
        "codebook_file_sha256": artifacts["codebook"]["file_sha256"],
        "codebook_payload_sha256": codebook.sha256,
        "session_release_file_sha256": artifacts["session_release"]["file_sha256"],
        "stage_b_gender_release_file_sha256": artifacts["stage_b_gender_release"][
            "file_sha256"
        ],
    }
    expected_audit = _signed_payload(
        _source_release_audit(
            accepted_rows,
            input_bindings=inputs,
            output_bindings=output_bindings,
        ),
        _AUDIT_CHECKSUM,
    )
    if audit != expected_audit:
        raise ValueError("source annotation release audit is not exactly reproducible")


def _valid_phase_category_counts(value: Any, expected_count: int) -> bool:
    if not isinstance(value, Mapping) or set(value) != set(SOURCE_ANALYSIS_FIELDS):
        return False
    for categories in value.values():
        if (
            not isinstance(categories, Mapping)
            or not categories
            or any(not isinstance(key, str) or not key for key in categories)
            or any(
                isinstance(count, bool) or not isinstance(count, int) or count <= 0
                for count in categories.values()
            )
            or sum(categories.values()) != expected_count
        ):
            return False
    return True


def _canonical_csv_rows(
    content: bytes, columns: tuple[str, ...], label: str
) -> list[dict[str, str]]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not UTF-8") from error
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        if tuple(reader.fieldnames or ()) != columns:
            raise ValueError(f"{label} columns do not match the exact schema")
        rows = list(reader)
    except csv.Error as error:
        raise ValueError(f"{label} is not valid CSV") from error
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{label} contains malformed rows")
    if _canonical_csv_bytes(columns, rows) != content:
        raise ValueError(f"{label} is not canonically serialized")
    return rows


def _canonical_csv_bytes(columns: tuple[str, ...], rows: Iterable[Mapping[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=columns,
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(dict(row))
    return output.getvalue().encode("utf-8")


def _canonical_json_object(content: bytes, label: str) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(payload, dict) or _canonical_json_bytes(payload) != content:
        raise ValueError(f"{label} is not one canonical JSON object")
    return payload


def _validate_signed_payload(
    payload: Mapping[str, Any],
    *,
    schema_version: str,
    checksum_field: str,
    label: str,
) -> None:
    checksum = payload.get(checksum_field)
    if (
        payload.get("schema_version") != schema_version
        or not isinstance(checksum, str)
        or _HEX64.fullmatch(checksum) is None
    ):
        raise ValueError(f"{label} schema or checksum field is invalid")
    unsigned = {key: value for key, value in payload.items() if key != checksum_field}
    if _sha256(_canonical_json_bytes(unsigned)) != checksum:
        raise ValueError(f"{label} payload checksum is invalid")


def _signed_payload(payload: Mapping[str, Any], checksum_field: str) -> dict[str, Any]:
    unsigned = dict(payload)
    if checksum_field in unsigned:
        raise ValueError("checksum field already exists")
    return {**unsigned, checksum_field: _sha256(_canonical_json_bytes(unsigned))}


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _read_external_file(path: str | Path, label: str) -> AnchoredFile:
    source = _absolute_path(path)
    if source.name in {"", ".", ".."}:
        raise ValueError(f"{label} must name one regular file")
    try:
        with AnchoredDirectory.open(source.parent, directory_mode=None) as anchor:
            return anchor.read_bytes(source.name, expected_mode=None)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"cannot safely read {label}") from error


def _absolute_path(path: str | Path) -> Path:
    supplied = Path(path).expanduser()
    if any(part in {".", ".."} for part in supplied.parts):
        raise ValueError("source annotation paths may not contain dot components")
    return supplied if supplied.is_absolute() else Path.cwd() / supplied


def _release_code_provenance(repository_root: Path) -> Any:
    """Return the exact release-owned source identity without burdening release loading."""
    from src.features.extraction import code_provenance

    return code_provenance(repository_root, SOURCE_RELEASE_OWNED_SOURCE_PATHS)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
