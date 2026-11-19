"""Approved canonical-to-source session mapping contract for MIT Interview.

The restricted identity-join audit deliberately emits a proposal that cannot be consumed as a
training join.  This module provides the separate, tracked acceptance artefact.  It preserves the
supplied source-session identifiers, anchors every canonical participant to its pre-interview
source session, and permits only an exact post-session permutation supported by the validated
audit report.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.secure_io import AnchoredDirectory

MIT_SESSION_MAPPING_SCHEMA_VERSION = "mit-canonical-source-session-map-1"
IDENTITY_JOIN_AUDIT_SCHEMA_VERSION = "mit-identity-join-audit-2"
FACE_CROP_MANIFEST_SCHEMA_VERSION = "mit-candidate-face-crops-2"
APPROVED_CANONICAL_PAIRING_STATUS = "approved_canonical_pairing"
EXPECTED_PARTICIPANT_COUNT = 69
EXPECTED_SOURCE_SESSION_COUNT = 138
SESSION_MAPPING_PROVENANCE_FIELDS = (
    "schema_version",
    "artifact_file_sha256",
    "artifact_payload_sha256",
    "identity_join_audit_file_sha256",
    "identity_join_audit_report_payload_sha256",
    "face_crop_manifest_file_sha256",
    "face_crop_manifest_payload_sha256",
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_SESSION_PATTERN = re.compile(r"^(PP|P)([1-9][0-9]*)$", re.ASCII)
_TOP_FIELDS = {
    "schema_version",
    "status",
    "mapping_approved",
    "training_input_authorized",
    "purpose",
    "bindings",
    "coverage",
    "source_session_ids",
    "post_overrides",
    "canonical_to_source_session",
    "artifact_payload_sha256",
}


@dataclass(frozen=True, slots=True)
class SessionMappingContext:
    """Approved canonical/source lookup plus non-path evidence provenance.

    Resolved evidence paths stay private and are never serialized downstream.  Call
    :meth:`verify_current_files` at execution boundaries to prove that the approved mapping and
    both bound evidence files still match this context.
    """

    canonical_to_source: tuple[tuple[str, str], ...]
    schema_version: str
    artifact_file_sha256: str
    artifact_payload_sha256: str
    identity_join_audit_file_sha256: str
    identity_join_audit_report_payload_sha256: str
    face_crop_manifest_file_sha256: str
    face_crop_manifest_payload_sha256: str
    _mapping_path: Path = field(repr=False, compare=False)
    _identity_join_audit_report_path: Path = field(repr=False, compare=False)
    _face_crop_manifest_path: Path = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != MIT_SESSION_MAPPING_SCHEMA_VERSION:
            raise ValueError("session mapping context schema version drifted")
        for name in SESSION_MAPPING_PROVENANCE_FIELDS[1:]:
            _validated_sha256(getattr(self, name), name)
        if len(self.canonical_to_source) != EXPECTED_SOURCE_SESSION_COUNT:
            raise ValueError("session mapping context must contain all 138 sessions")
        canonical_ids: set[str] = set()
        source_ids: set[str] = set()
        for canonical, source in self.canonical_to_source:
            canonical_phase, _ = _parse_session_id(canonical)
            source_phase, _ = _parse_session_id(source)
            if (
                canonical_phase != source_phase
                or canonical in canonical_ids
                or source in source_ids
            ):
                raise ValueError("session mapping context is not a phase-preserving bijection")
            canonical_ids.add(canonical)
            source_ids.add(source)
        if len(canonical_ids) != EXPECTED_SOURCE_SESSION_COUNT:
            raise ValueError("session mapping context coverage drifted")

    @property
    def lookup(self) -> dict[str, str]:
        """Return a fresh canonical-session to source-session lookup."""
        return dict(self.canonical_to_source)

    @property
    def provenance(self) -> dict[str, str]:
        """Return the exact seven-field non-path provenance contract."""
        return {name: getattr(self, name) for name in SESSION_MAPPING_PROVENANCE_FIELDS}

    def verify_current_files(self) -> None:
        """Revalidate all three private evidence paths against this immutable context."""
        observed = load_verified_session_mapping(
            self._mapping_path,
            identity_join_audit_report_path=self._identity_join_audit_report_path,
            face_crop_manifest_path=self._face_crop_manifest_path,
        )
        if (
            observed.canonical_to_source != self.canonical_to_source
            or observed.provenance != self.provenance
        ):
            raise ValueError("session mapping evidence changed after validation")


def build_approved_mit_session_mapping(
    source_session_ids: Sequence[str],
    post_overrides: Mapping[str, str],
    *,
    identity_join_audit_report_path: str | Path,
    face_crop_manifest_path: str | Path,
    approval_status: str,
) -> dict[str, Any]:
    """Build a deterministic approved join from an explicit complete source roster.

    ``post_overrides`` is canonical-to-source, for example ``PP32 -> PP33``.  Identity defaults
    are expanded for every other source session.  Calling this function is an approval action, so
    the caller must supply the exact ``approved_canonical_pairing`` status explicitly.
    """
    if approval_status != APPROVED_CANONICAL_PAIRING_STATUS:
        raise ValueError("canonical pairing approval status was not supplied explicitly")
    roster = _validated_source_roster(source_session_ids)
    overrides = _validated_override_mapping(post_overrides, roster)
    mapping_rows = _expanded_mapping_rows(roster, overrides)

    report, report_bytes = _read_identity_join_audit(identity_join_audit_report_path)
    face_manifest, face_manifest_bytes = _read_canonical_json_object(
        face_crop_manifest_path, "face-crop manifest"
    )
    bindings = _bindings_from_evidence(
        report,
        report_bytes,
        face_manifest,
        face_manifest_bytes,
        roster,
        mapping_rows,
    )
    unsigned: dict[str, Any] = {
        "schema_version": MIT_SESSION_MAPPING_SCHEMA_VERSION,
        "status": APPROVED_CANONICAL_PAIRING_STATUS,
        "mapping_approved": True,
        "training_input_authorized": True,
        "purpose": "approved_canonical_to_source_join_for_mit_interview",
        "bindings": bindings,
        "coverage": {
            "participant_count": EXPECTED_PARTICIPANT_COUNT,
            "source_session_count": EXPECTED_SOURCE_SESSION_COUNT,
            "canonical_session_count": EXPECTED_SOURCE_SESSION_COUNT,
            "pre_session_count": EXPECTED_PARTICIPANT_COUNT,
            "post_session_count": EXPECTED_PARTICIPANT_COUNT,
            "equal_numeric_pre_post_coverage_verified": True,
            "complete_source_coverage_verified": True,
            "exact_bijection_verified": True,
        },
        "source_session_ids": list(roster),
        "post_overrides": [
            {"canonical_session_id": canonical, "source_session_id": source}
            for canonical, source in overrides.items()
        ],
        "canonical_to_source_session": mapping_rows,
    }
    artifact = {
        **unsigned,
        "artifact_payload_sha256": _canonical_sha256(unsigned),
    }
    validate_mit_session_mapping(artifact)
    verify_mit_session_mapping_bindings(
        artifact,
        identity_join_audit_report_path=identity_join_audit_report_path,
        face_crop_manifest_path=face_crop_manifest_path,
    )
    return artifact


def validate_mit_session_mapping(payload: Mapping[str, Any]) -> None:
    """Fail closed on schema, approval state, mapping, coverage, bindings or checksum drift."""
    if not isinstance(payload, Mapping) or set(payload) != _TOP_FIELDS:
        raise ValueError("MIT session mapping has an unexpected top-level schema")
    if payload["schema_version"] != MIT_SESSION_MAPPING_SCHEMA_VERSION:
        raise ValueError("unsupported MIT session mapping schema version")
    if (
        payload["status"] != APPROVED_CANONICAL_PAIRING_STATUS
        or payload["mapping_approved"] is not True
        or payload["training_input_authorized"] is not True
        or payload["purpose"] != "approved_canonical_to_source_join_for_mit_interview"
    ):
        raise ValueError("MIT canonical pairing approval boundary was weakened")

    roster_value = payload["source_session_ids"]
    if not isinstance(roster_value, list):
        raise ValueError("MIT source-session roster must be a JSON array")
    roster = _validated_source_roster(roster_value)
    if list(roster) != roster_value:
        raise ValueError("MIT source-session roster is not in canonical order")

    overrides_value = payload["post_overrides"]
    if not isinstance(overrides_value, list):
        raise ValueError("MIT post-session overrides must be a JSON array")
    override_pairs: list[tuple[str, str]] = []
    for row in overrides_value:
        if not isinstance(row, Mapping) or set(row) != {
            "canonical_session_id",
            "source_session_id",
        }:
            raise ValueError("MIT post-session override row schema drifted")
        canonical = row["canonical_session_id"]
        source = row["source_session_id"]
        if not isinstance(canonical, str) or not isinstance(source, str):
            raise ValueError("MIT post-session override IDs must be strings")
        override_pairs.append((canonical, source))
    if len(override_pairs) != len({canonical for canonical, _ in override_pairs}):
        raise ValueError("MIT post-session overrides contain duplicate canonical IDs")
    overrides = _validated_override_mapping(dict(override_pairs), roster)
    if override_pairs != list(overrides.items()):
        raise ValueError("MIT post-session overrides are not in canonical order")

    expected_mapping = _expanded_mapping_rows(roster, overrides)
    observed_mapping = payload["canonical_to_source_session"]
    if not isinstance(observed_mapping, list) or observed_mapping != expected_mapping:
        raise ValueError("MIT canonical-to-source mapping does not match its roster and overrides")

    _validate_coverage(payload["coverage"])
    bindings = _validate_bindings(payload["bindings"])
    proposal_rows = _audit_proposal_rows(expected_mapping)
    if bindings["identity_join_audit"]["proposed_mapping_payload_sha256"] != (
        _canonical_sha256({"proposed_mapping": proposal_rows})
    ):
        raise ValueError("MIT mapping is not cross-bound to its recorded audit proposal")

    checksum = _validated_sha256(
        payload["artifact_payload_sha256"], "MIT session mapping payload checksum"
    )
    unsigned = {key: value for key, value in payload.items() if key != "artifact_payload_sha256"}
    if _canonical_sha256(unsigned) != checksum:
        raise ValueError("MIT session mapping payload checksum does not match")


def verify_mit_session_mapping_bindings(
    payload: Mapping[str, Any],
    *,
    identity_join_audit_report_path: str | Path,
    face_crop_manifest_path: str | Path,
) -> None:
    """Verify an accepted mapping against the exact restricted audit and face manifest files."""
    validate_mit_session_mapping(payload)
    roster = tuple(payload["source_session_ids"])
    mapping_rows = [dict(row) for row in payload["canonical_to_source_session"]]
    report, report_bytes = _read_identity_join_audit(identity_join_audit_report_path)
    face_manifest, face_manifest_bytes = _read_canonical_json_object(
        face_crop_manifest_path, "face-crop manifest"
    )
    expected = _bindings_from_evidence(
        report,
        report_bytes,
        face_manifest,
        face_manifest_bytes,
        roster,
        mapping_rows,
    )
    if payload["bindings"] != expected:
        raise ValueError("MIT session mapping evidence bindings do not match the supplied files")


def canonical_mit_session_mapping_json(payload: Mapping[str, Any]) -> str:
    """Return canonical UTF-8 JSON after complete standalone validation."""
    validate_mit_session_mapping(payload)
    return _canonical_json_bytes(payload).decode("utf-8")


def load_mit_session_mapping(path: str | Path) -> dict[str, Any]:
    """Load an exact-canonical, duplicate-key-safe MIT session mapping artefact."""
    payload, _ = _read_canonical_json_object(path, "MIT session mapping")
    validate_mit_session_mapping(payload)
    return payload


def load_verified_session_mapping(
    session_mapping_path: str | Path,
    *,
    identity_join_audit_report_path: str | Path,
    face_crop_manifest_path: str | Path,
) -> SessionMappingContext:
    """Load an approved mapping and verify its exact restricted evidence bindings."""
    mapping_path = _lexical_file_path(session_mapping_path, "MIT session mapping")
    audit_path = _lexical_file_path(
        identity_join_audit_report_path, "identity-join audit report"
    )
    face_path = _lexical_file_path(face_crop_manifest_path, "face-crop manifest")
    payload, mapping_bytes = _read_canonical_json_object(mapping_path, "MIT session mapping")
    validate_mit_session_mapping(payload)
    verify_mit_session_mapping_bindings(
        payload,
        identity_join_audit_report_path=audit_path,
        face_crop_manifest_path=face_path,
    )
    audit_binding = payload["bindings"]["identity_join_audit"]
    face_binding = payload["bindings"]["face_crop_manifest"]
    context = SessionMappingContext(
        canonical_to_source=tuple(canonical_to_source_lookup(payload).items()),
        schema_version=payload["schema_version"],
        artifact_file_sha256=hashlib.sha256(mapping_bytes).hexdigest(),
        artifact_payload_sha256=payload["artifact_payload_sha256"],
        identity_join_audit_file_sha256=audit_binding["file_sha256"],
        identity_join_audit_report_payload_sha256=audit_binding[
            "report_payload_sha256"
        ],
        face_crop_manifest_file_sha256=face_binding["file_sha256"],
        face_crop_manifest_payload_sha256=face_binding["payload_sha256"],
        _mapping_path=mapping_path,
        _identity_join_audit_report_path=audit_path,
        _face_crop_manifest_path=face_path,
    )
    if set(context.provenance) != set(SESSION_MAPPING_PROVENANCE_FIELDS):
        raise ValueError("session mapping provenance field set drifted")
    return context


def write_mit_session_mapping_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically publish one immutable, deterministic tracked mapping artefact."""
    content = canonical_mit_session_mapping_json(payload).encode("utf-8")
    requested = Path(path)
    if requested.name in {"", ".", ".."}:
        raise ValueError("MIT session mapping output must name a file")
    parent = _resolved_non_symlink_directory(requested.parent)
    destination = parent / requested.name

    descriptor, temporary_name = tempfile.mkstemp(
        dir=parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    published_identity: tuple[int, int] | None = None
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        load_mit_session_mapping(temporary)
        try:
            # A hard link publishes the fully written inode atomically but, unlike os.replace,
            # cannot clobber an artefact that another writer created after our initial checks.
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            pass
        else:
            temporary_stat = temporary.stat(follow_symlinks=False)
            published_identity = (temporary_stat.st_dev, temporary_stat.st_ino)
    finally:
        temporary.unlink(missing_ok=True)
        _fsync_directory(parent)

    _validate_existing_mapping_output(
        destination,
        expected_content=content,
        expected_identity=published_identity,
    )
    return destination


def canonical_to_source_lookup(payload: Mapping[str, Any]) -> dict[str, str]:
    """Return a validated canonical-session-to-source-session lookup."""
    validate_mit_session_mapping(payload)
    return {
        row["canonical_session_id"]: row["source_session_id"]
        for row in payload["canonical_to_source_session"]
    }


def _validated_source_roster(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("MIT source-session roster must be a sequence of strings")
    if len(values) != EXPECTED_SOURCE_SESSION_COUNT:
        raise ValueError("MIT source-session roster must contain exactly 138 sessions")
    parsed: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    pre_numbers: set[int] = set()
    post_numbers: set[int] = set()
    for value in values:
        phase, number = _parse_session_id(value)
        if value in seen:
            raise ValueError("MIT source-session roster contains a duplicate session ID")
        seen.add(value)
        (pre_numbers if phase == "pre" else post_numbers).add(number)
        parsed.append((number, 0 if phase == "pre" else 1, value))
    if (
        len(pre_numbers) != EXPECTED_PARTICIPANT_COUNT
        or len(post_numbers) != EXPECTED_PARTICIPANT_COUNT
        or pre_numbers != post_numbers
    ):
        raise ValueError("MIT source roster lacks equal complete numeric pre/post coverage")
    return tuple(value for _, _, value in sorted(parsed))


def _validated_override_mapping(values: Mapping[str, str], roster: Sequence[str]) -> dict[str, str]:
    if not isinstance(values, Mapping):
        raise TypeError("MIT post-session overrides must be a mapping")
    roster_set = set(roster)
    validated: list[tuple[int, str, str]] = []
    for canonical, source in values.items():
        canonical_phase, canonical_number = _parse_session_id(canonical)
        source_phase, _ = _parse_session_id(source)
        if canonical_phase != "post" or source_phase != "post":
            raise ValueError("MIT post-session overrides cannot cross session phases")
        if canonical not in roster_set or source not in roster_set:
            raise ValueError("MIT post-session override is outside the explicit source roster")
        if canonical == source:
            raise ValueError("MIT identity defaults must not be repeated as post overrides")
        validated.append((canonical_number, canonical, source))
    validated.sort()
    result = {canonical: source for _, canonical, source in validated}
    mapping_rows = _expanded_mapping_rows_without_validation(roster, result)
    source_values = [row["source_session_id"] for row in mapping_rows]
    if len(source_values) != len(set(source_values)) or set(source_values) != roster_set:
        raise ValueError("MIT canonical-to-source expansion is not an exact bijection")
    return result


def _expanded_mapping_rows(
    roster: Sequence[str], overrides: Mapping[str, str]
) -> list[dict[str, str]]:
    rows = _expanded_mapping_rows_without_validation(roster, overrides)
    source_values = [row["source_session_id"] for row in rows]
    if len(rows) != EXPECTED_SOURCE_SESSION_COUNT or set(source_values) != set(roster):
        raise ValueError("MIT canonical-to-source mapping lacks exact source coverage")
    if len(source_values) != len(set(source_values)):
        raise ValueError("MIT canonical-to-source mapping is not one-to-one")
    return rows


def _expanded_mapping_rows_without_validation(
    roster: Sequence[str], overrides: Mapping[str, str]
) -> list[dict[str, str]]:
    return [
        {
            "canonical_session_id": canonical,
            "source_session_id": overrides.get(canonical, canonical),
        }
        for canonical in roster
    ]


def _bindings_from_evidence(
    report: Mapping[str, Any],
    report_bytes: bytes,
    face_manifest: Mapping[str, Any],
    face_manifest_bytes: bytes,
    roster: Sequence[str],
    mapping_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    report_roster = _audit_source_roster(report)
    if tuple(roster) != report_roster:
        raise ValueError("MIT source roster does not exactly match the identity-join audit")
    proposal_rows = _audit_proposal_rows(mapping_rows)
    if report["proposed_source_session_to_canonical_participant"] != proposal_rows:
        raise ValueError("approved MIT mapping does not exactly match the validated audit proposal")

    report_binding = report["bindings"]
    manifest_schema = face_manifest.get("schema_version")
    if not isinstance(manifest_schema, str):
        raise ValueError("face-crop manifest schema version must be a string")
    face_file_sha256 = hashlib.sha256(face_manifest_bytes).hexdigest()
    face_payload_sha256 = _canonical_sha256(face_manifest)
    if (
        manifest_schema != report_binding["face_crop_manifest_schema_version"]
        or face_file_sha256 != report_binding["face_crop_manifest_file_sha256"]
        or face_payload_sha256 != report_binding["face_crop_manifest_payload_sha256"]
    ):
        raise ValueError("face-crop manifest does not match the validated audit binding")

    return {
        "identity_join_audit": {
            "schema_version": IDENTITY_JOIN_AUDIT_SCHEMA_VERSION,
            "status": "proposed_not_approved",
            "file_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "report_payload_sha256": report["report_payload_sha256"],
            "proposed_mapping_payload_sha256": _canonical_sha256(
                {"proposed_mapping": proposal_rows}
            ),
        },
        "face_crop_manifest": {
            "schema_version": manifest_schema,
            "file_sha256": face_file_sha256,
            "payload_sha256": face_payload_sha256,
        },
    }


def _audit_source_roster(report: Mapping[str, Any]) -> tuple[str, ...]:
    coverage = report["coverage"]
    if (
        coverage["expected_participant_count"] != EXPECTED_PARTICIPANT_COUNT
        or coverage["session_count"] != EXPECTED_SOURCE_SESSION_COUNT
    ):
        raise ValueError("identity-join audit does not cover the complete 138-session corpus")
    values = [row["source_session_id"] for row in report["sessions"]]
    return _validated_source_roster(values)


def _audit_proposal_rows(
    mapping_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for mapping in mapping_rows:
        canonical = mapping["canonical_session_id"]
        source = mapping["source_session_id"]
        phase, number = _parse_session_id(canonical)
        source_phase, _ = _parse_session_id(source)
        if source_phase != phase:
            raise ValueError("MIT canonical-to-source mapping cannot cross session phases")
        if phase == "pre" and source != canonical:
            raise ValueError("MIT canonical pre sessions must retain their source IDs")
        rows.append(
            {
                "source_session_id": source,
                "canonical_participant_id": str(number),
                "phase": phase,
                "basis": (
                    "pre_source_session_anchor"
                    if phase == "pre"
                    else "hungarian_assignment_to_pre_source_session_anchor"
                ),
            }
        )
    return rows


def _validate_coverage(value: object) -> None:
    expected = {
        "participant_count": EXPECTED_PARTICIPANT_COUNT,
        "source_session_count": EXPECTED_SOURCE_SESSION_COUNT,
        "canonical_session_count": EXPECTED_SOURCE_SESSION_COUNT,
        "pre_session_count": EXPECTED_PARTICIPANT_COUNT,
        "post_session_count": EXPECTED_PARTICIPANT_COUNT,
        "equal_numeric_pre_post_coverage_verified": True,
        "complete_source_coverage_verified": True,
        "exact_bijection_verified": True,
    }
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError("MIT session mapping coverage schema drifted")
    count_fields = {
        "participant_count",
        "source_session_count",
        "canonical_session_count",
        "pre_session_count",
        "post_session_count",
    }
    if any(
        isinstance(value[field], bool) or not isinstance(value[field], int)
        for field in count_fields
    ):
        raise ValueError("MIT session mapping coverage counts must be integers")
    if any(value[field] is not True for field in set(expected) - count_fields):
        raise ValueError("MIT session mapping coverage verification was weakened")
    if any(value[field] != expected[field] for field in expected):
        raise ValueError("MIT session mapping coverage does not match the complete corpus")


def _validate_bindings(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "identity_join_audit",
        "face_crop_manifest",
    }:
        raise ValueError("MIT session mapping binding schema drifted")
    audit = value["identity_join_audit"]
    if not isinstance(audit, Mapping) or set(audit) != {
        "schema_version",
        "status",
        "file_sha256",
        "report_payload_sha256",
        "proposed_mapping_payload_sha256",
    }:
        raise ValueError("MIT identity-join audit binding schema drifted")
    if (
        audit["schema_version"] != IDENTITY_JOIN_AUDIT_SCHEMA_VERSION
        or audit["status"] != "proposed_not_approved"
    ):
        raise ValueError("MIT session mapping is not bound to the frozen proposed audit schema")
    validated_audit = {
        **dict(audit),
        "file_sha256": _validated_sha256(audit["file_sha256"], "identity audit file checksum"),
        "report_payload_sha256": _validated_sha256(
            audit["report_payload_sha256"], "identity audit payload checksum"
        ),
        "proposed_mapping_payload_sha256": _validated_sha256(
            audit["proposed_mapping_payload_sha256"], "identity audit proposal checksum"
        ),
    }
    face = value["face_crop_manifest"]
    if not isinstance(face, Mapping) or set(face) != {
        "schema_version",
        "file_sha256",
        "payload_sha256",
    }:
        raise ValueError("MIT face-crop manifest binding schema drifted")
    if face["schema_version"] != FACE_CROP_MANIFEST_SCHEMA_VERSION:
        raise ValueError("MIT face-crop manifest binding schema version drifted")
    validated_face = {
        **dict(face),
        "file_sha256": _validated_sha256(face["file_sha256"], "face manifest file checksum"),
        "payload_sha256": _validated_sha256(
            face["payload_sha256"], "face manifest payload checksum"
        ),
    }
    return {"identity_join_audit": validated_audit, "face_crop_manifest": validated_face}


def _read_identity_join_audit(path: str | Path) -> tuple[dict[str, Any], bytes]:
    # The restricted audit implementation depends on SciPy and feature-extraction code.  Import
    # it only at the evidence boundary so normal mapping loads remain dependency-light and can be
    # used by the core MIT dataset loader without creating an import cycle.
    from .identity_join_audit import (
        canonical_identity_join_audit_json,
        validate_identity_join_audit_report,
    )

    source = _lexical_file_path(path, "identity-join audit report")
    raw = _anchored_regular_bytes(source, "identity-join audit report")
    payload = _decode_json_object(raw, "identity-join audit report")
    validate_identity_join_audit_report(payload)
    if canonical_identity_join_audit_json(payload).encode("utf-8") != raw:
        raise ValueError("identity-join audit report does not use its canonical serialization")
    return payload, raw


def _read_canonical_json_object(path: str | Path, label: str) -> tuple[dict[str, Any], bytes]:
    source = _lexical_file_path(path, label)
    raw = _anchored_regular_bytes(source, label)
    payload = _decode_json_object(raw, label)
    if _canonical_json_bytes(payload) != raw:
        raise ValueError(f"{label} does not use canonical JSON serialization")
    return payload, raw


def _decode_json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
        payload = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"cannot read a valid {label}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return payload


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _lexical_file_path(value: str | Path, label: str) -> Path:
    source = Path(value).expanduser()
    absolute = source if source.is_absolute() else Path.cwd() / source
    if absolute.name in {"", ".", ".."}:
        raise ValueError(f"{label} must name a regular non-symlink file")
    return absolute


def _anchored_regular_bytes(path: Path, label: str) -> bytes:
    """Read exact same-open bytes beneath a continuously verified parent handle."""
    try:
        with AnchoredDirectory.open(path.parent, directory_mode=None) as parent:
            return parent.read_bytes(path.name, expected_mode=None).content
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(
            f"{label} must be stable and its path cannot contain symlink components"
        ) from error


def _resolved_non_symlink_directory(value: Path) -> Path:
    """Resolve a directory only after rejecting every lexical symlink component."""
    absolute = value if value.is_absolute() else Path.cwd() / value
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        if part in {"", "."}:
            continue
        if part == "..":
            current = current.parent
            continue
        candidate = current / part
        try:
            metadata = candidate.lstat()
        except OSError as error:
            raise ValueError(
                "MIT session mapping output parent must be a real directory"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("MIT session mapping output parent cannot contain symlink components")
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("MIT session mapping output parent must be a real directory")
        current = candidate
    return current


def _validate_existing_mapping_output(
    destination: Path,
    *,
    expected_content: bytes,
    expected_identity: tuple[int, int] | None,
) -> None:
    """Validate an existing publication through one non-following file descriptor."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(destination, flags)
    except OSError as error:
        raise ValueError("MIT session mapping output may not be a symlink") from error
    try:
        metadata = os.fstat(descriptor)
        observed_identity = (metadata.st_dev, metadata.st_ino)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("existing MIT session mapping output is not a regular file")
        if metadata.st_nlink != 1:
            raise ValueError("existing MIT session mapping output may not be hard-linked")
        if expected_identity is not None and observed_identity != expected_identity:
            raise ValueError("published MIT session mapping changed during atomic writing")
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        observed = b"".join(chunks)
        if observed != expected_content:
            raise ValueError("MIT session mapping output already exists with different content")
        payload = _decode_json_object(observed, "MIT session mapping")
        if _canonical_json_bytes(payload) != observed:
            raise ValueError("MIT session mapping does not use canonical JSON serialization")
        validate_mit_session_mapping(payload)
        os.fchmod(descriptor, 0o644)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

    try:
        final_metadata = destination.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError("published MIT session mapping changed during atomic writing") from error
    if (final_metadata.st_dev, final_metadata.st_ino) != observed_identity:
        raise ValueError("published MIT session mapping changed during atomic writing")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _parse_session_id(value: object) -> tuple[str, int]:
    if not isinstance(value, str):
        raise ValueError("MIT session ID must be a string")
    match = _SESSION_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError("MIT session ID must be canonical P# or PP# text")
    return ("post" if match.group(1) == "PP" else "pre", int(match.group(2)))


def _validated_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be 64 lowercase hexadecimal digits")
    return value


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
