"""Immutable full-run closure for outcome-blind MIT feature extraction."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np

from src.data import (
    TRANSCRIPT_PARSER_REVISION,
    MITInterviewFeatureSession,
    MITInterviewSession,
    SessionMappingContext,
    assign_participant_folds,
    parse_session_id,
)
from src.frozen_json import freeze_json
from src.secure_io import AnchoredDirectory, AnchoredFile

from .extraction import (
    EXTRACTOR_VERSION,
    SCHEMA_VERSION,
    SHARD_KEYS,
    ExtractionSettings,
    _npz_bytes,
    _validate_arrays,
    _validate_modality_sidecar,
    canonical_json_bytes,
    safe_clip_stem,
)

FEATURE_EXTRACTION_CLOSURE_FILENAME = "feature-extraction-closure.json"
FEATURE_EXTRACTION_CLOSURE_SCHEMA_VERSION = "fair-avi-mit-feature-extraction-closure-1"
EXPECTED_MIT_FEATURE_SESSION_COUNT = 138

_TOP_FIELDS = {
    "schema_version",
    "state",
    "execution_mode",
    "corpus",
    "feature_schema_version",
    "extractor_version",
    "script",
    "seed",
    "source_date_epoch",
    "config_sha256",
    "code",
    "session_mapping",
    "runtime",
    "encoders",
    "preprocessing",
    "feature_run_contract_sha256",
    "feature_contract_sha256",
    "preprocessing_sha256",
    "outcome_fields_accessed",
    "protected_annotation_fields_accessed",
    "recovery_contract",
    "selected_roster_sha256",
    "selected_session_count",
    "completed_session_count",
    "success_session_count",
    "failure_session_count",
    "records",
    "closure_payload_sha256",
}
_RECORD_FIELDS = {
    "visit_index",
    "clip_id",
    "session_id",
    "source_session_id",
    "participant_id",
    "condition",
    "fold",
    "state",
    "metadata_filename",
    "metadata_file_sha256",
    "metadata_payload_sha256",
    "extraction_fingerprint",
    "shard_filename",
    "shard_file_sha256",
    "failure_code",
}
_COMMON_METADATA_FIELDS = {
    "schema_version",
    "extractor_version",
    "state",
    "clip_id",
    "session_id",
    "source_session_id",
    "participant_id",
    "condition",
    "fold",
    "input",
    "script",
    "seed",
    "source_date_epoch",
    "config_sha256",
    "extraction_fingerprint",
    "session_mapping",
    "runtime",
    "code",
    "encoders",
    "preprocessing",
}
_RECOVERY_CONTRACT = {
    "success_resume": "full_semantic_validation_against_current_inputs",
    "failure_resume": "forbidden_reexecute_each_selected_session",
    "publication": "anchored_o_excl_no_clobber_byte_identical_reuse_only",
    "closure": "publish_only_after_exact_selected_roster_visited",
}
_ENCODER_CORE = {
    "video": {
        "name": "facebook/dinov2-large",
        "revision": "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
        "output_dim": 1024,
        "weight_sha256": (
            "399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23"
        ),
    },
    "audio": {
        "name": "microsoft/wavlm-base-plus",
        "revision": "4c66d4806a428f2e922ccfa1a962776e232d487b",
        "output_dim": 768,
        "weight_sha256": (
            "3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51"
        ),
    },
    "text": {
        "name": "sentence-transformers/all-mpnet-base-v2",
        "revision": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
        "output_dim": 768,
        "weight_sha256": (
            "78c0197b6159d92658e319bc1d72e4c73a9a03dd03815e70e555c5ef05615658"
        ),
    },
}


@dataclass(frozen=True, slots=True)
class FeatureRecordArtifacts:
    """Stable bytes for one closure-bound success or failure record."""

    record: Mapping[str, Any]
    metadata: AnchoredFile
    metadata_payload: Mapping[str, Any]
    shard: AnchoredFile | None


class FeatureExtractionClosure:
    """Validated closure capability retaining one private anchored directory."""

    def __init__(
        self,
        anchor: AnchoredDirectory,
        manifest: Mapping[str, Any],
        manifest_bytes: bytes,
        closure_file_sha256: str,
        closure_payload_sha256: str,
    ) -> None:
        self._anchor = anchor
        self.manifest = freeze_json(manifest)
        self.manifest_bytes = manifest_bytes
        self.closure_file_sha256 = closure_file_sha256
        self.closure_payload_sha256 = closure_payload_sha256
        self._closed = False

    def __enter__(self) -> FeatureExtractionClosure:
        self._require_open()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __copy__(self) -> None:
        raise TypeError("feature extraction closures cannot be copied")

    def __deepcopy__(self, _memo: object) -> None:
        raise TypeError("feature extraction closures cannot be copied")

    def close(self) -> None:
        if not self._closed:
            self._anchor.close()
            self._closed = True

    def verify_current_files(self) -> None:
        """Revalidate closure, exact inventory and every referenced immutable byte."""
        self._require_open()
        closure_file = self._anchor.read_bytes(
            FEATURE_EXTRACTION_CLOSURE_FILENAME, expected_mode=0o600
        )
        if (
            closure_file.content != self.manifest_bytes
            or closure_file.sha256 != self.closure_file_sha256
        ):
            raise ValueError("feature extraction closure file changed")
        _validate_manifest_structure(
            self.manifest,
            expected_file_sha256=self.closure_file_sha256,
            manifest_bytes=self.manifest_bytes,
        )
        expected_names = {FEATURE_EXTRACTION_CLOSURE_FILENAME}
        for record in self.manifest["records"]:
            artifacts = self._read_record_artifacts(record)
            expected_names.add(record["metadata_filename"])
            if artifacts.shard is not None:
                expected_names.add(record["shard_filename"])
        if set(self._anchor.list_names()) != expected_names:
            raise ValueError("feature extraction directory inventory is not closure-exact")
        self._anchor.verify_path_identity()

    def read_record_artifacts(
        self, record_or_visit_index: Mapping[str, Any] | int
    ) -> FeatureRecordArtifacts:
        """Return stable bytes for one record without exposing or reopening a path."""
        self._require_open()
        if isinstance(record_or_visit_index, int) and not isinstance(
            record_or_visit_index, bool
        ):
            records = self.manifest["records"]
            if not 0 <= record_or_visit_index < len(records):
                raise IndexError("feature record visit index is out of range")
            record = records[record_or_visit_index]
        elif isinstance(record_or_visit_index, Mapping):
            visit_index = record_or_visit_index.get("visit_index")
            if (
                isinstance(visit_index, bool)
                or not isinstance(visit_index, int)
                or not 0 <= visit_index < len(self.manifest["records"])
                or dict(self.manifest["records"][visit_index])
                != dict(record_or_visit_index)
            ):
                raise ValueError("feature record is not bound to this closure")
            record = self.manifest["records"][visit_index]
        else:
            raise TypeError("feature record selector must be a record or visit index")
        return self._read_record_artifacts(record)

    def _read_record_artifacts(
        self, record: Mapping[str, Any]
    ) -> FeatureRecordArtifacts:
        self._anchor.verify_path_identity()
        metadata = self._anchor.read_bytes(
            record["metadata_filename"], expected_mode=0o600
        )
        if metadata.sha256 != record["metadata_file_sha256"]:
            raise ValueError("closure-bound feature metadata checksum failed")
        payload = _load_canonical_object(metadata.content, "feature metadata")
        signed_field = (
            "sidecar_payload_sha256"
            if record["state"] == "success"
            else "failure_payload_sha256"
        )
        if payload.get(signed_field) != record["metadata_payload_sha256"]:
            raise ValueError("closure-bound feature metadata payload checksum failed")
        unsigned = dict(payload)
        unsigned.pop(signed_field, None)
        if _payload_sha256(unsigned) != record["metadata_payload_sha256"]:
            raise ValueError("closure-bound feature metadata signature failed")
        _validate_metadata_against_record(payload, record, self.manifest)
        shard: AnchoredFile | None = None
        if record["state"] == "success":
            shard = self._anchor.read_bytes(record["shard_filename"], expected_mode=0o600)
            if shard.sha256 != record["shard_file_sha256"]:
                raise ValueError("closure-bound feature shard checksum failed")
            shard_binding = payload.get("shard")
            if (
                not isinstance(shard_binding, Mapping)
                or shard_binding.get("filename") != record["shard_filename"]
                or shard_binding.get("sha256") != shard.sha256
            ):
                raise ValueError("closure-bound feature shard metadata drifted")
            _validate_success_shard_semantics(payload, shard.content, self.manifest)
        self._anchor.verify_path_identity()
        return FeatureRecordArtifacts(
            record=freeze_json(record),
            metadata=metadata,
            metadata_payload=freeze_json(payload),
            shard=shard,
        )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("feature extraction closure is closed")


def write_feature_extraction_closure(
    output_dir: str | Path,
    *,
    records: Sequence[Mapping[str, Any]],
    selected_sessions: Sequence[MITInterviewSession | MITInterviewFeatureSession],
    session_mapping: SessionMappingContext,
    expected_run_context: Mapping[str, Any],
    verify_current_input: Callable[
        [MITInterviewSession | MITInterviewFeatureSession, Mapping[str, Any]], None
    ],
) -> Mapping[str, Any]:
    """Publish a closure only after the exact full selected roster was visited."""
    if len(selected_sessions) != EXPECTED_MIT_FEATURE_SESSION_COUNT:
        raise ValueError("research closure requires the exact 138-session MIT roster")
    if len(records) != len(selected_sessions):
        raise ValueError("research closure requires one visit record per selected session")
    session_mapping.verify_current_files()
    lookup = session_mapping.lookup
    if any(
        lookup.get(clip.session_id)
        != getattr(clip, "source_session_id", clip.session_id)
        for clip in selected_sessions
    ):
        raise ValueError("selected feature roster disagrees with the approved mapping")
    _validate_exact_roster([_identity_row(clip) for clip in selected_sessions])
    if [
        (clip.session_id, getattr(clip, "source_session_id", clip.session_id))
        for clip in selected_sessions
    ] != _ordered_mapping_pairs(lookup):
        raise ValueError("selected feature roster does not exactly cover the mapping")
    ordered_records: list[dict[str, Any]] = []
    roster: list[dict[str, Any]] = []
    with AnchoredDirectory.open(output_dir, directory_mode=0o700) as output:
        common: dict[str, Any] | None = None
        for visit_index, (clip, supplied_record) in enumerate(
            zip(selected_sessions, records, strict=True)
        ):
            expected_identity = _identity_row(clip)
            record = {
                "visit_index": visit_index,
                **{
                    key: supplied_record[key]
                    for key in sorted(_RECORD_FIELDS - {"visit_index"})
                },
            }
            if any(record.get(field) != value for field, value in expected_identity.items()):
                raise ValueError("feature visit record does not match the selected roster")
            _validate_record_shape(record)
            metadata = output.read_bytes(record["metadata_filename"], expected_mode=0o600)
            if metadata.sha256 != record["metadata_file_sha256"]:
                raise ValueError("feature visit metadata checksum failed")
            payload = _load_canonical_object(metadata.content, "feature metadata")
            verify_current_input(clip, payload)
            signed_field = (
                "sidecar_payload_sha256"
                if record["state"] == "success"
                else "failure_payload_sha256"
            )
            signed = payload.get(signed_field)
            unsigned = dict(payload)
            unsigned.pop(signed_field, None)
            if signed != record["metadata_payload_sha256"] or _payload_sha256(unsigned) != signed:
                raise ValueError("feature visit metadata signature failed")
            observed_common = _common_metadata(payload)
            if common is None:
                common = observed_common
            elif observed_common != common:
                raise ValueError("feature records do not share one extraction run contract")
            ordered_records.append(record)
            roster.append(expected_identity)
        if common is None:
            raise ValueError("feature extraction closure cannot be empty")
        if common != dict(expected_run_context):
            raise ValueError("feature run context changed before closure publication")
        if common["session_mapping"] != session_mapping.provenance:
            raise ValueError("feature records do not bind the approved session mapping")
        run_contract = {
            "schema_version": common["feature_schema_version"],
            "extractor_version": common["extractor_version"],
            "script": common["script"],
            "seed": common["seed"],
            "source_date_epoch": common["source_date_epoch"],
            "config_sha256": common["config_sha256"],
            "code": common["code"],
            "session_mapping": common["session_mapping"],
            "runtime": common["runtime"],
            "encoders": common["encoders"],
            "preprocessing": common["preprocessing"],
        }
        feature_contract = {
            "schema_version": common["feature_schema_version"],
            "extractor_version": common["extractor_version"],
            "config_sha256": common["config_sha256"],
            "code_source_tree_sha256": common["code"]["source_tree_sha256"],
            "encoders": common["encoders"],
            "preprocessing": common["preprocessing"],
            "runtime": common["runtime"],
            "session_mapping": common["session_mapping"],
        }
        success_count = sum(record["state"] == "success" for record in ordered_records)
        unsigned_closure = {
            "schema_version": FEATURE_EXTRACTION_CLOSURE_SCHEMA_VERSION,
            "state": "closed",
            "execution_mode": "research",
            "corpus": "mit_interview",
            **common,
            "feature_run_contract_sha256": _payload_sha256(run_contract),
            "feature_contract_sha256": _payload_sha256(feature_contract),
            "preprocessing_sha256": _payload_sha256(common["preprocessing"]),
            "outcome_fields_accessed": [],
            "protected_annotation_fields_accessed": [],
            "recovery_contract": dict(_RECOVERY_CONTRACT),
            "selected_roster_sha256": _payload_sha256({"selected_roster": roster}),
            "selected_session_count": len(roster),
            "completed_session_count": len(ordered_records),
            "success_session_count": success_count,
            "failure_session_count": len(ordered_records) - success_count,
            "records": ordered_records,
        }
        closure_payload_sha256 = _payload_sha256(unsigned_closure)
        closure = {**unsigned_closure, "closure_payload_sha256": closure_payload_sha256}
        closure_bytes = canonical_json_bytes(closure)
        _validate_manifest_structure(
            closure,
            expected_file_sha256=hashlib.sha256(closure_bytes).hexdigest(),
            manifest_bytes=closure_bytes,
        )
        for record in ordered_records:
            metadata = output.read_bytes(
                record["metadata_filename"], expected_mode=0o600
            )
            payload = _load_canonical_object(metadata.content, "feature metadata")
            _validate_metadata_against_record(payload, record, closure)
            if record["state"] == "success":
                shard = output.read_bytes(record["shard_filename"], expected_mode=0o600)
                binding = payload.get("shard")
                if (
                    shard.sha256 != record["shard_file_sha256"]
                    or not isinstance(binding, Mapping)
                    or binding.get("filename") != record["shard_filename"]
                    or binding.get("sha256") != shard.sha256
                ):
                    raise ValueError("feature visit shard binding failed")
                _validate_success_shard_semantics(payload, shard.content, closure)
            elif payload.get("failure", {}).get("code") != record["failure_code"]:
                raise ValueError("feature visit failure binding failed")
        expected_names = {FEATURE_EXTRACTION_CLOSURE_FILENAME}
        for record in ordered_records:
            expected_names.add(record["metadata_filename"])
            if record["shard_filename"] is not None:
                expected_names.add(record["shard_filename"])
        before_publication = set(output.list_names())
        artifact_names = expected_names - {FEATURE_EXTRACTION_CLOSURE_FILENAME}
        if frozenset(before_publication) not in {
            frozenset(artifact_names),
            frozenset(expected_names),
        }:
            raise ValueError("feature output inventory contains stale or extraneous artifacts")
        output.publish_bytes(
            FEATURE_EXTRACTION_CLOSURE_FILENAME,
            closure_bytes,
            mode=0o600,
        )
        if set(output.list_names()) != expected_names:
            raise ValueError("feature output inventory drifted during closure publication")
        output.verify_path_identity()
    session_mapping.verify_current_files()
    return freeze_json(closure)


def load_verified_feature_extraction_closure(
    output_dir: str | Path,
    *,
    closure_file_sha256: str,
    closure_payload_sha256: str,
    session_mapping: SessionMappingContext,
) -> FeatureExtractionClosure:
    """Open a caller-hash-bound full-run closure as a private capability."""
    _require_sha256(closure_file_sha256, "closure file checksum")
    _require_sha256(closure_payload_sha256, "closure payload checksum")
    session_mapping.verify_current_files()
    anchor = AnchoredDirectory.open(output_dir, directory_mode=0o700)
    try:
        closure_file = anchor.read_bytes(
            FEATURE_EXTRACTION_CLOSURE_FILENAME, expected_mode=0o600
        )
        if closure_file.sha256 != closure_file_sha256:
            raise ValueError("feature extraction closure file checksum failed")
        manifest = _load_canonical_object(
            closure_file.content, "feature extraction closure"
        )
        _validate_manifest_structure(
            manifest,
            expected_file_sha256=closure_file_sha256,
            manifest_bytes=closure_file.content,
        )
        if manifest["closure_payload_sha256"] != closure_payload_sha256:
            raise ValueError("feature extraction closure payload checksum failed")
        if manifest["session_mapping"] != session_mapping.provenance:
            raise ValueError("feature extraction closure mapping binding drifted")
        if any(
            session_mapping.lookup.get(record["session_id"])
            != record["source_session_id"]
            for record in manifest["records"]
        ):
            raise ValueError("feature extraction closure routing disagrees with mapping")
        if [
            (record["session_id"], record["source_session_id"])
            for record in manifest["records"]
        ] != _ordered_mapping_pairs(session_mapping.lookup):
            raise ValueError("feature extraction closure does not exactly cover mapping")
        capability = FeatureExtractionClosure(
            anchor,
            manifest,
            closure_file.content,
            closure_file_sha256,
            closure_payload_sha256,
        )
        capability.verify_current_files()
    except Exception:
        anchor.close()
        raise
    session_mapping.verify_current_files()
    return capability


def _validate_manifest_structure(
    manifest: Mapping[str, Any], *, expected_file_sha256: str, manifest_bytes: bytes
) -> None:
    if set(manifest) != _TOP_FIELDS:
        raise ValueError("feature extraction closure field set drifted")
    if (
        manifest.get("schema_version") != FEATURE_EXTRACTION_CLOSURE_SCHEMA_VERSION
        or manifest.get("state") != "closed"
        or manifest.get("execution_mode") != "research"
        or manifest.get("corpus") != "mit_interview"
        or manifest.get("feature_schema_version") != SCHEMA_VERSION
        or manifest.get("extractor_version") != EXTRACTOR_VERSION
        or manifest.get("script") != "scripts/extract_features.py"
        or manifest.get("seed") != 2026
        or manifest.get("outcome_fields_accessed") != []
        or manifest.get("protected_annotation_fields_accessed") != []
        or manifest.get("recovery_contract") != _RECOVERY_CONTRACT
    ):
        raise ValueError("feature extraction closure fixed contract drifted")
    _require_sha256(expected_file_sha256, "closure file checksum")
    if (
        canonical_json_bytes(manifest) != manifest_bytes
        or hashlib.sha256(manifest_bytes).hexdigest() != expected_file_sha256
    ):
        raise ValueError("feature extraction closure file checksum failed")
    signed = manifest.get("closure_payload_sha256")
    unsigned = dict(manifest)
    unsigned.pop("closure_payload_sha256", None)
    if not isinstance(signed, str) or _payload_sha256(unsigned) != signed:
        raise ValueError("feature extraction closure signature failed")
    for field in (
        "feature_run_contract_sha256",
        "feature_contract_sha256",
        "preprocessing_sha256",
        "selected_roster_sha256",
    ):
        _require_sha256(manifest.get(field), field)
    if (
        manifest.get("selected_session_count") != EXPECTED_MIT_FEATURE_SESSION_COUNT
        or manifest.get("completed_session_count") != EXPECTED_MIT_FEATURE_SESSION_COUNT
        or not isinstance(manifest.get("records"), list)
        or len(manifest["records"]) != EXPECTED_MIT_FEATURE_SESSION_COUNT
    ):
        raise ValueError("feature extraction closure does not cover all 138 sessions")
    records = manifest["records"]
    for index, record in enumerate(records):
        _validate_record_shape(record)
        if record["visit_index"] != index:
            raise ValueError("feature extraction closure visit order drifted")
    if len({record["session_id"] for record in records}) != len(records) or len(
        {record["source_session_id"] for record in records}
    ) != len(records):
        raise ValueError("feature extraction closure identities are not bijective")
    success_count = sum(record["state"] == "success" for record in records)
    if (
        manifest.get("success_session_count") != success_count
        or manifest.get("failure_session_count") != len(records) - success_count
    ):
        raise ValueError("feature extraction closure state counts drifted")
    roster = [
        {field: record[field] for field in _identity_fields()}
        for record in records
    ]
    _validate_exact_roster(roster)
    if manifest["selected_roster_sha256"] != _payload_sha256(
        {"selected_roster": roster}
    ):
        raise ValueError("feature extraction closure roster checksum failed")
    run_contract = {
        "schema_version": manifest["feature_schema_version"],
        "extractor_version": manifest["extractor_version"],
        "script": manifest["script"],
        "seed": manifest["seed"],
        "source_date_epoch": manifest["source_date_epoch"],
        "config_sha256": manifest["config_sha256"],
        "code": manifest["code"],
        "session_mapping": manifest["session_mapping"],
        "runtime": manifest["runtime"],
        "encoders": manifest["encoders"],
        "preprocessing": manifest["preprocessing"],
    }
    feature_contract = {
        "schema_version": manifest["feature_schema_version"],
        "extractor_version": manifest["extractor_version"],
        "config_sha256": manifest["config_sha256"],
        "code_source_tree_sha256": manifest["code"].get("source_tree_sha256"),
        "encoders": manifest["encoders"],
        "preprocessing": manifest["preprocessing"],
        "runtime": manifest["runtime"],
        "session_mapping": manifest["session_mapping"],
    }
    if (
        manifest["feature_run_contract_sha256"] != _payload_sha256(run_contract)
        or manifest["feature_contract_sha256"] != _payload_sha256(feature_contract)
        or manifest["preprocessing_sha256"]
        != _payload_sha256(manifest["preprocessing"])
    ):
        raise ValueError("feature extraction closure contract checksum failed")


def _validate_record_shape(record: Mapping[str, Any]) -> None:
    if not isinstance(record, Mapping) or set(record) != _RECORD_FIELDS:
        raise ValueError("feature extraction closure record fields drifted")
    for field in ("metadata_file_sha256", "metadata_payload_sha256"):
        _require_sha256(record.get(field), field)
    state = record.get("state")
    if state == "success":
        _require_sha256(record.get("extraction_fingerprint"), "extraction fingerprint")
        _require_sha256(record.get("shard_file_sha256"), "shard checksum")
        if (
            record.get("metadata_filename")
            != f"{safe_clip_stem(record['clip_id'])}.json"
            or record.get("shard_filename")
            != f"{safe_clip_stem(record['clip_id'])}.npz"
            or record.get("failure_code") is not None
        ):
            raise ValueError("feature success record binding drifted")
    elif state == "failure":
        fingerprint = record.get("extraction_fingerprint")
        if fingerprint is not None:
            _require_sha256(fingerprint, "extraction fingerprint")
        if (
            record.get("metadata_filename")
            != f"{safe_clip_stem(record['clip_id'])}.failure.json"
            or record.get("shard_filename") is not None
            or record.get("shard_file_sha256") is not None
            or not isinstance(record.get("failure_code"), str)
            or not record["failure_code"]
        ):
            raise ValueError("feature failure record binding drifted")
    else:
        raise ValueError("feature closure record state is invalid")


def _validate_metadata_against_record(
    payload: Mapping[str, Any],
    record: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    state_specific = (
        {"modality", "arrays", "shard", "sidecar_payload_sha256"}
        if record["state"] == "success"
        else {"failure", "failure_payload_sha256"}
    )
    if set(payload) != _COMMON_METADATA_FIELDS | state_specific:
        raise ValueError("feature metadata schema differs from the frozen contract")
    if any(payload.get(field) != record[field] for field in _identity_fields()):
        raise ValueError("feature metadata identity differs from closure")
    if payload.get("state") != record["state"]:
        raise ValueError("feature metadata state differs from closure")
    common = _common_metadata(payload)
    expected_common = {
        "feature_schema_version": manifest["feature_schema_version"],
        "extractor_version": manifest["extractor_version"],
        "script": manifest["script"],
        "seed": manifest["seed"],
        "source_date_epoch": manifest["source_date_epoch"],
        "config_sha256": manifest["config_sha256"],
        "code": manifest["code"],
        "session_mapping": manifest["session_mapping"],
        "runtime": manifest["runtime"],
        "encoders": manifest["encoders"],
        "preprocessing": manifest["preprocessing"],
    }
    if common != expected_common:
        raise ValueError("feature metadata run contract differs from closure")
    _validate_input_metadata(payload.get("input"), record["source_session_id"], record["state"])
    if record["state"] == "success":
        if payload.get("extraction_fingerprint") != record["extraction_fingerprint"]:
            raise ValueError("feature metadata fingerprint differs from closure")
    else:
        failure = payload.get("failure")
        if (
            not isinstance(failure, Mapping)
            or set(failure) != {"code", "message", "type", "modality"}
            or failure.get("code") != record["failure_code"]
            or not isinstance(failure.get("message"), str)
            or not failure["message"]
            or not isinstance(failure.get("type"), str)
            or not failure["type"]
            or (
                failure.get("modality") is not None
                and failure.get("modality") not in {"video", "audio", "text"}
            )
        ):
            raise ValueError("feature failure code differs from closure")


def _validate_input_metadata(
    value: object, source_session_id: str, state: str
) -> None:
    expected_fields = {
        "video_reference",
        "video_sha256",
        "audio_reference",
        "audio_sha256",
        "source_dialogue_sha256",
        "candidate_transcript_sha256",
        "transcript_parser_revision",
        "transcript_parser_audit",
        "face_crop_fingerprint",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError("feature input metadata schema drifted")
    if (
        value.get("video_reference") != f"Videos/{source_session_id}.avi"
        or value.get("audio_reference") != f"Audio/{source_session_id}.wav"
    ):
        raise ValueError("feature media routing differs from its source-session identity")
    for field in (
        "source_dialogue_sha256",
        "face_crop_fingerprint",
    ):
        _require_sha256(value.get(field), field)
    for field in ("video_sha256", "audio_sha256", "candidate_transcript_sha256"):
        observed = value.get(field)
        if observed is None and state == "failure":
            continue
        _require_sha256(observed, field)
    if value.get("transcript_parser_revision") != TRANSCRIPT_PARSER_REVISION:
        raise ValueError("feature transcript parser revision is invalid")
    audit = value.get("transcript_parser_audit")
    expected_audit_fields = {
        "explicit_candidate_segments",
        "retained_candidate_continuations",
        "retained_candidate_continuation_characters",
        "excluded_colonless_fragments",
        "excluded_colonless_characters",
        "excluded_unclassified_markers",
        "excluded_interviewer_segments",
    }
    if (
        not isinstance(audit, Mapping)
        or set(audit) != expected_audit_fields
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in audit.values()
        )
    ):
        raise ValueError("feature transcript parser audit is invalid")


def _validate_success_shard_semantics(
    payload: Mapping[str, Any], shard_bytes: bytes, manifest: Mapping[str, Any]
) -> None:
    try:
        with np.load(io.BytesIO(shard_bytes), allow_pickle=False) as archive:
            if tuple(archive.files) != SHARD_KEYS:
                raise ValueError("feature shard keys drifted")
            arrays = {key: archive[key] for key in SHARD_KEYS}
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError("feature shard cannot be decoded safely") from error
    settings = ExtractionSettings(
        seed=2026,
        video_fps=1,
        video_dim=1024,
        audio_sample_rate=16_000,
        audio_dim=768,
        audio_pooled_dim=1536,
        audio_conv_stride=320,
        audio_receptive_field=400,
        text_dim=768,
        text_max_sequence_tokens=384,
        video_max_samples=128,
        audio_window_samples=320_000,
        audio_max_windows=16,
        source_date_epoch=manifest["source_date_epoch"],
    )
    _validate_arrays(arrays, settings)
    if _npz_bytes(arrays) != shard_bytes:
        raise ValueError("feature shard is not the deterministic NPZ representation")
    metadata = payload.get("arrays")
    if not isinstance(metadata, Mapping) or set(metadata) != set(SHARD_KEYS):
        raise ValueError("feature array metadata registry drifted")
    for key, array in arrays.items():
        entry = metadata.get(key)
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"shape", "dtype"}
            or entry.get("shape") != list(array.shape)
            or entry.get("dtype") != str(array.dtype)
        ):
            raise ValueError("feature array metadata differs from shard bytes")
    expected_base = {
        field: payload[field] for field in _COMMON_METADATA_FIELDS
    }
    _validate_modality_sidecar(payload.get("modality"), arrays, expected_base, settings)


def _common_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not _COMMON_METADATA_FIELDS.issubset(payload):
        raise ValueError("feature metadata common contract is incomplete")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("extractor_version") != EXTRACTOR_VERSION
        or payload.get("script") != "scripts/extract_features.py"
        or payload.get("seed") != 2026
    ):
        raise ValueError("feature metadata fixed contract drifted")
    _require_sha256(payload.get("config_sha256"), "feature configuration checksum")
    source_date_epoch = payload.get("source_date_epoch")
    if (
        isinstance(source_date_epoch, bool)
        or not isinstance(source_date_epoch, int)
        or source_date_epoch < 0
    ):
        raise ValueError("feature source-date epoch is invalid")
    code = payload.get("code")
    if not isinstance(code, Mapping) or set(code) != {
        "commit_sha",
        "source_tree_sha256",
        "dirty",
    }:
        raise ValueError("feature code provenance schema drifted")
    commit = code.get("commit_sha")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or not isinstance(code.get("dirty"), bool)
    ):
        raise ValueError("feature code provenance is invalid")
    _require_sha256(code.get("source_tree_sha256"), "feature source-tree checksum")
    for field in ("session_mapping", "runtime", "encoders", "preprocessing"):
        if not isinstance(payload.get(field), Mapping) or not payload[field]:
            raise ValueError(f"feature {field} provenance is unavailable")
    runtime = payload["runtime"]
    determinism = runtime.get("torch_determinism")
    if (
        runtime.get("backend") != "huggingface"
        or runtime.get("dtype") != "float32"
        or not isinstance(runtime.get("device"), str)
        or not runtime["device"]
        or not isinstance(determinism, Mapping)
        or determinism.get("deterministic_algorithms_enabled") is not True
    ):
        raise ValueError("feature runtime is not the deterministic research backend")
    encoders = payload["encoders"]
    if set(encoders) != set(_ENCODER_CORE):
        raise ValueError("feature encoder registry drifted")
    for modality, expected in _ENCODER_CORE.items():
        observed = encoders.get(modality)
        if not isinstance(observed, Mapping) or any(
            observed.get(field) != expected_value
            for field, expected_value in expected.items()
        ):
            raise ValueError(f"feature {modality} encoder pin drifted")
    return {
        "feature_schema_version": payload["schema_version"],
        "extractor_version": payload["extractor_version"],
        "script": payload["script"],
        "seed": payload["seed"],
        "source_date_epoch": source_date_epoch,
        "config_sha256": payload["config_sha256"],
        "code": copy.deepcopy(payload["code"]),
        "session_mapping": copy.deepcopy(payload["session_mapping"]),
        "runtime": copy.deepcopy(payload["runtime"]),
        "encoders": copy.deepcopy(payload["encoders"]),
        "preprocessing": copy.deepcopy(payload["preprocessing"]),
    }


def _identity_row(
    clip: MITInterviewSession | MITInterviewFeatureSession,
) -> dict[str, Any]:
    return {
        "clip_id": clip.clip_id,
        "session_id": clip.session_id,
        "source_session_id": getattr(clip, "source_session_id", clip.session_id),
        "participant_id": clip.participant_id,
        "condition": clip.condition,
        "fold": clip.fold,
    }


def _identity_fields() -> tuple[str, ...]:
    return (
        "clip_id",
        "session_id",
        "source_session_id",
        "participant_id",
        "condition",
        "fold",
    )


def _validate_exact_roster(roster: Sequence[Mapping[str, Any]]) -> None:
    participant_values: set[str] = set()
    for row in roster:
        try:
            _prefix, participant_id, _condition = parse_session_id(row.get("session_id"))
        except (TypeError, ValueError) as error:
            raise ValueError("feature canonical session identity is invalid") from error
        participant_values.add(participant_id)
    participants = sorted(participant_values, key=int)
    if len(participants) != 69:
        raise ValueError("feature extraction closure must contain exactly 69 participants")
    folds = assign_participant_folds(participants)
    expected: list[dict[str, Any]] = []
    for participant_id in participants:
        for prefix, condition in (("P", "pre"), ("PP", "post")):
            session_id = f"{prefix}{participant_id}"
            expected.append(
                {
                    "clip_id": session_id,
                    "session_id": session_id,
                    "participant_id": participant_id,
                    "condition": condition,
                    "fold": folds[participant_id],
                }
            )
    observed_without_source = [
        {
            "clip_id": row.get("clip_id"),
            "session_id": row.get("session_id"),
            "participant_id": row.get("participant_id"),
            "condition": row.get("condition"),
            "fold": row.get("fold"),
        }
        for row in roster
    ]
    if observed_without_source != expected:
        raise ValueError("feature extraction closure roster or fold assignment drifted")
    for row in roster:
        try:
            _prefix, _participant, source_condition = parse_session_id(
                row.get("source_session_id")
            )
        except (TypeError, ValueError) as error:
            raise ValueError("feature source-session identity is invalid") from error
        if source_condition != row.get("condition"):
            raise ValueError("feature source-session routing crosses interview phase")


def _ordered_mapping_pairs(lookup: Mapping[str, str]) -> list[tuple[str, str]]:
    return [
        (session_id, lookup[session_id])
        for session_id in sorted(
            lookup,
            key=lambda value: (
                int(parse_session_id(value)[1]),
                parse_session_id(value)[2] == "post",
            ),
        )
    ]


def _load_canonical_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{label} is not valid duplicate-safe JSON") from error
    if not isinstance(payload, dict) or canonical_json_bytes(payload) != raw:
        raise ValueError(f"{label} does not use canonical JSON serialization")
    return payload


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value
