"""Strict Stage-B oracle lock consumed by the one-way Stage-C evaluator.

No training is implemented here.  The contract admits exactly two immutable
dispositions: all fifteen predeclared oracle fits are locked, or all fifteen
are omitted for the exact predeclared eligibility reasons.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from types import MappingProxyType
from typing import Any

from src.frozen_json import freeze_json
from src.secure_io import AnchoredDirectory, AnchoredFile
from src.training.crossfit import TARGET_TRANSFORM, RunProvenance, canonical_json_bytes
from src.training.proxy_training import FROZEN_FINAL_SEEDS, FROZEN_OUTER_ROTATIONS

STAGE_B_ORACLE_LOCK_FILENAME = "stage-b-oracle-lock.json"
STAGE_B_ORACLE_LOCK_SCHEMA_VERSION = "fair-avi-mit-stage-b-oracle-lock-3"
STAGE_B_ORACLE_ACCESS_LOG_SCHEMA_VERSION = "fair-avi-mit-stage-b-oracle-access-log-2"
STAGE_B_ORACLE_PREDICTION_LOCK_SCHEMA_VERSION = "fair-avi-mit-stage-b-oracle-prediction-lock-3"
STAGE_B_ORACLE_PREDICTION_SCHEMA_VERSION = "fair-avi-mit-stage-b-oracle-oof-predictions-1"
STAGE_B_EXECUTION_PROVENANCE_SCHEMA_VERSION = "fair-avi-mit-stage-b-execution-provenance-2"
STAGE_B_EXECUTION_OWNED_SOURCE_PATHS = (
    "pyproject.toml",
    "requirements.lock",
    "scripts/run_stage_b_oracle.py",
    "src/__init__.py",
    "src/data/__init__.py",
    "src/data/annotation_audit.py",
    "src/data/annotation_intake.py",
    "src/data/annotation_stimuli.py",
    "src/data/identity_join_audit.py",
    "src/data/mit_interview.py",
    "src/data/mit_session_mapping.py",
    "src/data/protected_annotations.py",
    "src/data/source_annotation_release.py",
    "src/eval/__init__.py",
    "src/eval/metrics.py",
    "src/eval/stage_b_oracle.py",
    "src/eval/stage_c.py",
    "src/features/__init__.py",
    "src/features/closure.py",
    "src/features/extraction.py",
    "src/features/face_selector.py",
    "src/features/pseudo_attributes.py",
    "src/frozen_json.py",
    "src/losses/__init__.py",
    "src/losses/fairness.py",
    "src/models/__init__.py",
    "src/models/fusion.py",
    "src/secure_io.py",
    "src/training/__init__.py",
    "src/training/crossfit.py",
    "src/training/proxy_artifacts.py",
    "src/training/proxy_training.py",
    "src/training/stage_a_ledger.py",
    "src/training/stage_b_oracle.py",
)
ORACLE_LOCK_CHECKSUM_FIELD = "oracle_lock_payload_sha256"
ORACLE_ACCESS_CHECKSUM_FIELD = "access_log_payload_sha256"
ORACLE_PREDICTION_LOCK_CHECKSUM_FIELD = "manifest_payload_sha256"
ORACLE_PREDICTION_CHECKSUM_FIELD = "prediction_payload_sha256"
ORACLE_EXPECTED_FIT_COUNT = 15
ORACLE_EXPECTED_BUNDLE_COUNT = 3

_HEX64 = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_HEX40 = re.compile(r"^[0-9a-f]{40}$", re.ASCII)

_COMMON_FIELDS = {
    "schema_version",
    "state",
    "configuration",
    "firewall_stage",
    "stage_a_closure_file_sha256",
    "stage_a_closure_payload_sha256",
    "oracle_eligibility_payload_sha256",
    "stage_b_execution_provenance",
    "annotation_bindings",
    "eligible_participant_count",
    "frozen_outer_folds",
    "frozen_seeds",
    "expected_fit_count",
    "authorized_fit_count",
    "completed_fit_count",
    "expected_bundle_count",
    "completed_bundle_count",
    "expected_oof_rows_per_seed",
    "protected_annotation_access_contract",
    "bundles",
    "omission_reasons",
    ORACLE_LOCK_CHECKSUM_FIELD,
}
_ANNOTATION_BINDING_FIELDS = {
    "codebook_file_sha256",
    "codebook_payload_sha256",
    "participant_annotation_release_file_sha256",
    "stage_b_gender_release_file_sha256",
    "codebook_version",
    "oracle_field",
    "codebook_categories",
    "annotation_release_method",
    "stage_b_designation",
    "designated_stage_b_field",
    "source_release_manifest_file_sha256",
}
_ACCESS_CONTRACT = {
    "release_unit": "outer_rotation_training_participants_only",
    "released_field_count": 1,
    "protected_validation_annotations_accessed": [],
    "protected_test_annotations_accessed": [],
    "checkpoint_selection": "ordinary_validation_srcc_without_protected_annotations",
    "category_merge": "forbidden",
    "category_drop": "forbidden",
    "post_hoc_binarisation": "forbidden",
    "all_15_or_none": True,
}
_BUNDLE_FIELDS = {
    "configuration",
    "seed",
    "prediction_lock_filename",
    "prediction_lock_file_sha256",
    "prediction_lock_payload_sha256",
    "prediction_lock_id",
    "prediction_filename",
    "prediction_file_sha256",
    "prediction_payload_sha256",
    "oof_row_count",
    "access_log_filename",
    "access_log_file_sha256",
    "access_log_payload_sha256",
    "fit_artifacts",
}
_ACCESS_LOG_FIELDS = {
    "schema_version",
    "configuration",
    "seed",
    "stage_a_closure_file_sha256",
    "stage_a_closure_payload_sha256",
    "oracle_eligibility_payload_sha256",
    "annotation_bindings",
    "entries",
    "protected_validation_participant_ids_released",
    "protected_test_participant_ids_released",
    ORACLE_ACCESS_CHECKSUM_FIELD,
}
_ACCESS_ENTRY_FIELDS = {
    "outer_rotation",
    "release_scope",
    "released_fields",
    "released_participant_ids",
    "released_participant_count",
    "released_annotations_sha256",
    "category_counts",
    "annotation_release_event_index",
    "prediction_lock_event_index",
    "entry_payload_sha256",
}
_PREDICTION_LOCK_FIELDS = {
    "schema_version",
    "state",
    "configuration",
    "seed",
    "firewall_stage",
    "stage_a_closure_file_sha256",
    "stage_a_closure_payload_sha256",
    "oracle_eligibility_payload_sha256",
    "stage_b_execution_provenance",
    "annotation_bindings",
    "annotation_access_log",
    "protected_training_annotation_scope",
    "protected_validation_annotation_fields_accessed",
    "protected_test_annotation_fields_accessed",
    "checkpoint_selection",
    "target_manifest_sha256",
    "eligibility_manifest_sha256",
    "feature_contract_sha256",
    "preprocessing_sha256",
    "feature_extraction_closure_file_sha256",
    "feature_extraction_closure_payload_sha256",
    "session_mapping",
    "config_sha256",
    "code_commit_sha",
    "code_source_tree_sha256",
    "source_date_epoch",
    "model_config_sha256",
    "eligible_participant_count",
    "expected_oof_session_count",
    "observed_oof_session_count",
    "rotations",
    "predictions",
    "prediction_lock_id",
    ORACLE_PREDICTION_LOCK_CHECKSUM_FIELD,
}
_ROTATION_FIELDS = {
    "outer_rotation",
    "training_annotation_count",
    "access_log_entry_payload_sha256",
    "checkpoint_epoch",
    "fit_artifact",
}
_PREDICTION_PAYLOAD_FIELDS = {
    "schema_version",
    "prediction_lock_id",
    "rows",
    ORACLE_PREDICTION_CHECKSUM_FIELD,
}
ORACLE_PREDICTION_ROW_FIELDS = {
    "schema_version",
    "corpus",
    "participant_id",
    "session_id",
    "source_session_id",
    "session_phase",
    "fixed_fold",
    "outer_rotation",
    "rotation_role",
    "seed",
    "configuration",
    "model_config_sha256",
    "model_fit_sha256",
    "target_transform",
    "target_manifest_sha256",
    "target",
    "prediction",
    "feature_contract_sha256",
    "preprocessing_sha256",
    "checkpoint_epoch",
    "firewall_stage",
    "eligibility_manifest_sha256",
    "session_mapping",
    "oracle_field",
    "annotation_access_log_payload_sha256",
    "prediction_lock_id",
}
_LOCK_IDENTITY_FIELDS = (
    "schema_version",
    "configuration",
    "seed",
    "firewall_stage",
    "stage_a_closure_file_sha256",
    "stage_a_closure_payload_sha256",
    "oracle_eligibility_payload_sha256",
    "stage_b_execution_provenance",
    "annotation_bindings",
    "annotation_access_log",
    "target_manifest_sha256",
    "eligibility_manifest_sha256",
    "feature_contract_sha256",
    "preprocessing_sha256",
    "feature_extraction_closure_file_sha256",
    "feature_extraction_closure_payload_sha256",
    "session_mapping",
    "config_sha256",
    "code_commit_sha",
    "code_source_tree_sha256",
    "source_date_epoch",
    "model_config_sha256",
    "rotations",
)

_STAGE_B_EXECUTION_PROVENANCE_FIELDS = {
    "schema_version",
    "owned_source_paths",
    "config_sha256",
    "code_commit_sha",
    "code_source_tree_sha256",
    "source_date_epoch",
    "runtime_versions",
}
_RUNTIME_VERSION_FIELDS = {
    "python",
    "numpy",
    "scipy",
    "scikit-learn",
    "torch",
}


@dataclass(frozen=True, slots=True)
class StageBOracleDisposition:
    """Validated locked predictions or the validated all-fits omission."""

    state: str
    payload: Mapping[str, Any]
    file_sha256: str
    payload_sha256: str
    prediction_rows: Mapping[int, tuple[Mapping[str, Any], ...]]

    def __post_init__(self) -> None:
        """Detach validated Stage-B evidence from caller-owned mutable objects."""
        object.__setattr__(self, "payload", freeze_json(self.payload))
        object.__setattr__(
            self,
            "prediction_rows",
            MappingProxyType(
                {
                    int(seed): tuple(freeze_json(row) for row in rows)
                    for seed, rows in self.prediction_rows.items()
                }
            ),
        )

    @property
    def public_summary(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "file_sha256": self.file_sha256,
            "payload_sha256": self.payload_sha256,
            "configuration": self.payload["configuration"],
            "oracle_field": self.payload["annotation_bindings"]["oracle_field"],
            "authorized_fit_count": self.payload["authorized_fit_count"],
            "completed_fit_count": self.payload["completed_fit_count"],
            "completed_bundle_count": self.payload["completed_bundle_count"],
            "expected_oof_rows_per_seed": self.payload["expected_oof_rows_per_seed"],
            "completed_oof_rows_per_seed": (
                self.payload["expected_oof_rows_per_seed"] if self.state == "locked" else 0
            ),
            "omission_reasons": list(self.payload["omission_reasons"]),
        }


def build_stage_b_execution_provenance(
    provenance: RunProvenance,
) -> dict[str, Any]:
    """Build the current Stage-B source/config identity recorded in every lock."""
    if not isinstance(provenance, RunProvenance):
        raise TypeError("Stage-B execution provenance requires frozen run provenance")
    return validate_stage_b_execution_provenance(
        {
            "schema_version": STAGE_B_EXECUTION_PROVENANCE_SCHEMA_VERSION,
            "owned_source_paths": list(STAGE_B_EXECUTION_OWNED_SOURCE_PATHS),
            "config_sha256": provenance.config_sha256,
            "code_commit_sha": provenance.code_commit_sha,
            "code_source_tree_sha256": provenance.code_source_tree_sha256,
            "source_date_epoch": provenance.source_date_epoch,
            "runtime_versions": _runtime_versions(),
        }
    )


def validate_stage_b_execution_provenance(value: object) -> dict[str, Any]:
    """Return a detached, structurally exact Stage-B execution identity."""
    if not isinstance(value, Mapping) or set(value) != _STAGE_B_EXECUTION_PROVENANCE_FIELDS:
        raise ValueError("Stage-B execution provenance schema drifted")
    paths = value.get("owned_source_paths")
    if (
        value.get("schema_version") != STAGE_B_EXECUTION_PROVENANCE_SCHEMA_VERSION
        or not isinstance(paths, list)
        or paths != list(STAGE_B_EXECUTION_OWNED_SOURCE_PATHS)
    ):
        raise ValueError("Stage-B execution provenance source inventory drifted")
    for relative in paths:
        if (
            not relative
            or Path(relative).is_absolute()
            or Path(relative).as_posix() != relative
            or any(part in {"", ".", ".."} for part in Path(relative).parts)
        ):
            raise ValueError("Stage-B execution provenance has an unsafe source path")
    if not isinstance(value.get("config_sha256"), str) or not _HEX64.fullmatch(
        str(value["config_sha256"])
    ):
        raise ValueError("Stage-B execution provenance config checksum is invalid")
    if not isinstance(value.get("code_commit_sha"), str) or not _HEX40.fullmatch(
        str(value["code_commit_sha"])
    ):
        raise ValueError("Stage-B execution provenance commit is invalid")
    if not isinstance(value.get("code_source_tree_sha256"), str) or not _HEX64.fullmatch(
        str(value["code_source_tree_sha256"])
    ):
        raise ValueError("Stage-B execution provenance source-tree checksum is invalid")
    epoch = value.get("source_date_epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("Stage-B execution provenance timestamp is invalid")
    runtime_versions = value.get("runtime_versions")
    if (
        not isinstance(runtime_versions, Mapping)
        or set(runtime_versions) != _RUNTIME_VERSION_FIELDS
        or any(
            not isinstance(runtime_versions[name], str) or not runtime_versions[name]
            for name in _RUNTIME_VERSION_FIELDS
        )
    ):
        raise ValueError("Stage-B execution provenance runtime versions are invalid")
    return {
        "schema_version": str(value["schema_version"]),
        "owned_source_paths": list(paths),
        "config_sha256": str(value["config_sha256"]),
        "code_commit_sha": str(value["code_commit_sha"]),
        "code_source_tree_sha256": str(value["code_source_tree_sha256"]),
        "source_date_epoch": epoch,
        "runtime_versions": {
            name: str(runtime_versions[name]) for name in sorted(_RUNTIME_VERSION_FIELDS)
        },
    }


def _runtime_versions() -> dict[str, str]:
    """Return the numerical runtime versions that can affect Stage-B results."""
    return {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "scikit-learn": version("scikit-learn"),
        "torch": version("torch"),
    }


@dataclass(frozen=True, slots=True)
class OracleAnnotationBinding:
    """Controller-issued hashes and eligibility with no participant labels."""

    annotation_bindings: Mapping[str, Any]
    eligibility: Mapping[str, Any]
    eligibility_payload_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "annotation_bindings", freeze_json(self.annotation_bindings))
        object.__setattr__(self, "eligibility", freeze_json(self.eligibility))
        if self.eligibility_payload_sha256 != _payload_sha256(self.eligibility):
            raise ValueError("oracle annotation binding eligibility checksum failed")


@dataclass(frozen=True, slots=True)
class OracleTrainingAnnotationRelease:
    """One rotation's only authorized protected-label capability."""

    outer_rotation: int
    field: str
    rows: tuple[Mapping[str, Any], ...]
    released_annotations_sha256: str
    category_counts: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        _validate_fold(self.outer_rotation)
        if not isinstance(self.field, str) or not self.field:
            raise ValueError("oracle training release field is invalid")
        frozen_rows = tuple(freeze_json(row) for row in self.rows)
        frozen_counts = tuple(freeze_json(row) for row in self.category_counts)
        object.__setattr__(self, "rows", frozen_rows)
        object.__setattr__(self, "category_counts", frozen_counts)
        expected_sha256 = _payload_sha256({"field": self.field, "rows": list(frozen_rows)})
        if self.released_annotations_sha256 != expected_sha256:
            raise ValueError("oracle training release checksum failed")


def oracle_eligibility_payload_sha256(eligibility: Mapping[str, Any]) -> str:
    """Hash the exact Stage-C eligibility report bound by Stage B."""
    if not isinstance(eligibility, Mapping):
        raise TypeError("oracle eligibility must be a mapping")
    return _payload_sha256(eligibility)


def prepare_oracle_annotation_binding(
    *, release: Any, annotations: Any, eligibility: Mapping[str, Any]
) -> OracleAnnotationBinding:
    """Validate full annotation evidence once and return only binding metadata."""
    _require_exact_eligibility(release, annotations, eligibility)
    return OracleAnnotationBinding(
        annotation_bindings=oracle_annotation_bindings(annotations, eligibility),
        eligibility=eligibility,
        eligibility_payload_sha256=oracle_eligibility_payload_sha256(eligibility),
    )


def prepare_oracle_training_releases(
    *, release: Any, annotations: Any, eligibility: Mapping[str, Any]
) -> tuple[OracleAnnotationBinding, tuple[OracleTrainingAnnotationRelease, ...]]:
    """Issue exactly five deep-frozen training-only label capabilities."""
    binding = prepare_oracle_annotation_binding(
        release=release,
        annotations=annotations,
        eligibility=eligibility,
    )
    _validate_eligibility(binding.eligibility, expected_status="eligible_all_15")
    field = str(binding.eligibility["field"])
    categories = tuple(binding.eligibility["codebook_categories"])
    releases = []
    for outer_rotation in FROZEN_OUTER_ROTATIONS:
        rows = oracle_training_release_rows(
            release=release,
            annotations=annotations,
            field=field,
            outer_rotation=outer_rotation,
        )
        counts = Counter(row["value"] for row in rows)
        releases.append(
            OracleTrainingAnnotationRelease(
                outer_rotation=outer_rotation,
                field=field,
                rows=tuple(rows),
                released_annotations_sha256=_payload_sha256({"field": field, "rows": rows}),
                category_counts=tuple(
                    {"category": category, "count": counts[category]} for category in categories
                ),
            )
        )
    return binding, tuple(releases)


def build_stage_b_oracle_omission(
    *,
    release: Any,
    annotation_binding: OracleAnnotationBinding,
    stage_b_execution_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the only valid zero-fit disposition from an ineligible report."""
    eligibility = annotation_binding.eligibility
    _validate_eligibility(eligibility, expected_status="omitted_all_15")
    reasons = eligibility.get("reasons")
    if not isinstance(reasons, list) or not reasons or reasons != sorted(set(reasons)):
        raise ValueError("oracle omission requires sorted unique predeclared reasons")
    unsigned = _common_unsigned(
        release=release,
        annotation_binding=annotation_binding,
        stage_b_execution_provenance=stage_b_execution_provenance,
        state="omitted",
        authorized_fit_count=0,
        completed_fit_count=0,
        completed_bundle_count=0,
        bundles=[],
        omission_reasons=list(reasons),
    )
    return _signed(unsigned, ORACLE_LOCK_CHECKSUM_FIELD)


def write_stage_b_oracle_lock(
    path: str | Path | AnchoredDirectory, payload: Mapping[str, Any]
) -> Path:
    """Write one canonical Stage-B disposition without replacement."""
    _validate_signed(payload, ORACLE_LOCK_CHECKSUM_FIELD, "Stage-B oracle lock")
    content = canonical_json_bytes(payload)
    if isinstance(path, AnchoredDirectory):
        path.verify_path_identity()
        path.publish_bytes(
            STAGE_B_ORACLE_LOCK_FILENAME,
            content,
            mode=0o600,
            allow_existing=False,
        )
        return path.path / STAGE_B_ORACLE_LOCK_FILENAME

    destination = _lexical_output_path(path)
    if destination.name != STAGE_B_ORACLE_LOCK_FILENAME:
        raise ValueError(f"Stage-B oracle disposition must be named {STAGE_B_ORACLE_LOCK_FILENAME}")
    with AnchoredDirectory.open(destination.parent, create=True, directory_mode=0o700) as anchor:
        anchor.publish_bytes(
            destination.name,
            content,
            mode=0o600,
            allow_existing=False,
        )
    return destination


def validate_stage_b_oracle_lock(
    path: str | Path,
    *,
    release: Any,
    annotations: Any,
    eligibility: Mapping[str, Any],
) -> StageBOracleDisposition:
    """Reload annotations, then validate every Stage-B byte through one anchor."""
    source = _lexical_output_path(path)
    if source.name != STAGE_B_ORACLE_LOCK_FILENAME:
        raise ValueError(f"Stage-B oracle disposition must be named {STAGE_B_ORACLE_LOCK_FILENAME}")
    research_release = (
        release.closure.get("execution_mode") == "research"
        and release.closure.get("state") == "closed"
    )
    anchor_context = (
        AnchoredDirectory.open(source.parent, create=False, directory_mode=0o700)
        if research_release
        else _open_existing_artifact_directory(source.parent)
    )
    with anchor_context as anchor:
        if research_release:
            top_record = anchor.read_bytes(STAGE_B_ORACLE_LOCK_FILENAME, expected_mode=0o600)
            file_mode = 0o600
        else:
            top_record, file_mode = _read_existing_artifact(anchor, STAGE_B_ORACLE_LOCK_FILENAME)
        payload = _canonical_object_from_bytes(top_record.content, "Stage-B oracle lock")
        state = payload.get("state")
        annotation_binding = prepare_oracle_annotation_binding(
            release=release,
            annotations=annotations,
            eligibility=eligibility,
        )
        training_releases: tuple[OracleTrainingAnnotationRelease, ...] = ()
        if state == "locked":
            annotation_binding, training_releases = prepare_oracle_training_releases(
                release=release,
                annotations=annotations,
                eligibility=eligibility,
            )
        return _validate_stage_b_oracle_lock_on_anchor(
            anchor,
            top_record=top_record,
            file_mode=file_mode,
            release=release,
            annotation_binding=annotation_binding,
            training_releases=training_releases,
        )


def validate_stage_b_oracle_lock_from_binding(
    path_or_anchor: str | Path | AnchoredDirectory,
    *,
    release: Any,
    annotation_binding: OracleAnnotationBinding,
    training_releases: Sequence[OracleTrainingAnnotationRelease],
) -> StageBOracleDisposition:
    """Anchor-native controller readback with no full annotation object retained."""
    if not isinstance(path_or_anchor, AnchoredDirectory):
        source = _lexical_output_path(path_or_anchor)
        if source.name != STAGE_B_ORACLE_LOCK_FILENAME:
            raise ValueError(
                f"Stage-B oracle disposition must be named {STAGE_B_ORACLE_LOCK_FILENAME}"
            )
        with AnchoredDirectory.open(source.parent, create=False, directory_mode=0o700) as opened:
            return validate_stage_b_oracle_lock_from_binding(
                opened,
                release=release,
                annotation_binding=annotation_binding,
                training_releases=training_releases,
            )
    anchor = path_or_anchor
    anchor.verify_path_identity()
    top_record = anchor.read_bytes(STAGE_B_ORACLE_LOCK_FILENAME, expected_mode=0o600)
    return _validate_stage_b_oracle_lock_on_anchor(
        anchor,
        top_record=top_record,
        file_mode=0o600,
        release=release,
        annotation_binding=annotation_binding,
        training_releases=training_releases,
    )


def _validate_stage_b_oracle_lock_on_anchor(
    anchor: AnchoredDirectory,
    *,
    top_record: AnchoredFile,
    file_mode: int,
    release: Any,
    annotation_binding: OracleAnnotationBinding,
    training_releases: Sequence[OracleTrainingAnnotationRelease],
) -> StageBOracleDisposition:
    payload = _canonical_object_from_bytes(top_record.content, "Stage-B oracle lock")
    _validate_signed(payload, ORACLE_LOCK_CHECKSUM_FIELD, "Stage-B oracle lock")
    if set(payload) != _COMMON_FIELDS:
        raise ValueError("Stage-B oracle lock top-level schema drifted")
    state = payload.get("state")
    expected_status = "eligible_all_15" if state == "locked" else "omitted_all_15"
    if state not in {"locked", "omitted"}:
        raise ValueError("Stage-B oracle state must be locked or omitted")
    eligibility = annotation_binding.eligibility
    stage_b_execution_provenance = validate_stage_b_execution_provenance(
        payload.get("stage_b_execution_provenance")
    )
    if state == "locked":
        training_releases = _validate_training_release_capabilities(
            release=release,
            annotation_binding=annotation_binding,
            training_releases=training_releases,
        )
    elif tuple(training_releases):
        raise ValueError("omitted Stage-B disposition cannot retain training releases")
    _validate_eligibility(eligibility, expected_status=expected_status)
    expected_common = _common_unsigned(
        release=release,
        annotation_binding=annotation_binding,
        stage_b_execution_provenance=stage_b_execution_provenance,
        state=state,
        authorized_fit_count=ORACLE_EXPECTED_FIT_COUNT if state == "locked" else 0,
        completed_fit_count=ORACLE_EXPECTED_FIT_COUNT if state == "locked" else 0,
        completed_bundle_count=ORACLE_EXPECTED_BUNDLE_COUNT if state == "locked" else 0,
        bundles=payload.get("bundles"),
        omission_reasons=([] if state == "locked" else list(eligibility.get("reasons", []))),
    )
    expected_signed = _signed(expected_common, ORACLE_LOCK_CHECKSUM_FIELD)
    if any(
        payload.get(field) != expected_signed.get(field)
        for field in _COMMON_FIELDS - {"bundles", ORACLE_LOCK_CHECKSUM_FIELD}
    ):
        raise ValueError("Stage-B oracle lock input, eligibility or disposition binding drifted")

    rows_by_seed: dict[int, tuple[Mapping[str, Any], ...]] = {}
    expected_oracle_artifacts: set[str] = set()
    if state == "omitted":
        if payload["bundles"] != [] or payload["omission_reasons"] != eligibility["reasons"]:
            raise ValueError("Stage-B oracle omission must contain zero bundles and exact reasons")
    else:
        bundles = payload["bundles"]
        if not isinstance(bundles, list) or len(bundles) != ORACLE_EXPECTED_BUNDLE_COUNT:
            raise ValueError("locked Stage-B oracle must bind exactly three seed bundles")
        if [row.get("seed") for row in bundles if isinstance(row, Mapping)] != list(
            FROZEN_FINAL_SEEDS
        ):
            raise ValueError("locked Stage-B oracle seed bundle registry drifted")
        for bundle in bundles:
            seed, rows = _validate_bundle(
                anchor,
                bundle,
                file_mode=file_mode,
                release=release,
                eligibility=eligibility,
                annotation_binding=annotation_binding,
                training_releases=training_releases,
                top_payload=payload,
            )
            if seed in rows_by_seed:
                raise ValueError("duplicate Stage-B oracle seed bundle")
            rows_by_seed[seed] = rows
            expected_oracle_artifacts.update(
                {
                    str(bundle["prediction_lock_filename"]),
                    str(bundle["prediction_filename"]),
                    str(bundle["access_log_filename"]),
                    *(str(record["filename"]) for record in bundle["fit_artifacts"]),
                }
            )
    _validate_oracle_inventory(anchor, expected_oracle_artifacts)
    anchor.verify_path_identity()
    return StageBOracleDisposition(
        state=state,
        payload=payload,
        file_sha256=top_record.sha256,
        payload_sha256=str(payload[ORACLE_LOCK_CHECKSUM_FIELD]),
        prediction_rows=rows_by_seed,
    )


def build_stage_b_oracle_lock(
    *,
    release: Any,
    annotation_binding: OracleAnnotationBinding,
    stage_b_execution_provenance: Mapping[str, Any],
    bundles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Sign a complete bundle registry; validation still reopens every referenced file."""
    eligibility = annotation_binding.eligibility
    _validate_eligibility(eligibility, expected_status="eligible_all_15")
    bundle_rows = [dict(row) for row in bundles]
    if len(bundle_rows) != ORACLE_EXPECTED_BUNDLE_COUNT:
        raise ValueError("complete Stage-B oracle lock requires exactly three bundles")
    if [row.get("seed") for row in bundle_rows] != list(FROZEN_FINAL_SEEDS):
        raise ValueError("complete Stage-B oracle lock requires frozen seed order")
    unsigned = _common_unsigned(
        release=release,
        annotation_binding=annotation_binding,
        stage_b_execution_provenance=stage_b_execution_provenance,
        state="locked",
        authorized_fit_count=ORACLE_EXPECTED_FIT_COUNT,
        completed_fit_count=ORACLE_EXPECTED_FIT_COUNT,
        completed_bundle_count=ORACLE_EXPECTED_BUNDLE_COUNT,
        bundles=bundle_rows,
        omission_reasons=[],
    )
    return _signed(unsigned, ORACLE_LOCK_CHECKSUM_FIELD)


def oracle_annotation_bindings(annotations: Any, eligibility: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact protected-release binding copied into every Stage-B object."""
    required = {"codebook_file_sha256", "codebook_payload_sha256"}
    if not isinstance(annotations.bindings, Mapping) or not required.issubset(annotations.bindings):
        raise ValueError("Stage-B oracle requires codebook and annotation-release bindings")
    categories = eligibility.get("codebook_categories")
    field = eligibility.get("field")
    if not isinstance(categories, list) or not categories or not isinstance(field, str):
        raise ValueError("Stage-B oracle binding requires one frozen codebook field")
    result = {
        "codebook_file_sha256": _required_sha256(
            annotations.bindings["codebook_file_sha256"], "codebook file checksum"
        ),
        "codebook_payload_sha256": _required_sha256(
            annotations.bindings["codebook_payload_sha256"], "codebook payload checksum"
        ),
        "participant_annotation_release_file_sha256": annotations.bindings.get(
            "participant_annotation_release_file_sha256"
        ),
        "stage_b_gender_release_file_sha256": annotations.bindings.get(
            "stage_b_gender_release_file_sha256"
        ),
        "codebook_version": annotations.codebook.codebook_version,
        "oracle_field": field,
        "codebook_categories": list(categories),
        "annotation_release_method": annotations.release_method,
        "stage_b_designation": annotations.stage_b_designation,
        "designated_stage_b_field": annotations.designated_stage_b_field,
        "source_release_manifest_file_sha256": annotations.bindings.get("manifest_file_sha256"),
    }
    if set(result) != _ANNOTATION_BINDING_FIELDS:
        raise RuntimeError("internal Stage-B annotation binding schema drifted")
    return result


def oracle_training_release_rows(
    *, release: Any, annotations: Any, field: str, outer_rotation: int
) -> list[dict[str, Any]]:
    """Return the exact sorted training-only participant/value rows used only for hashing."""
    _validate_fold(outer_rotation)
    rows = []
    for participant_id in release.participant_ids:
        if release.participant_folds[participant_id] in {
            outer_rotation,
            (outer_rotation + 1) % len(FROZEN_OUTER_ROTATIONS),
        }:
            continue
        annotation = annotations.participant_annotations.get(participant_id)
        value = None if annotation is None else getattr(annotation, field)
        if value is None:
            raise ValueError("eligible Stage-B oracle training participant has unknown annotation")
        rows.append({"participant_id": participant_id, "value": value})
    return sorted(rows, key=lambda row: int(row["participant_id"]))


def _validate_training_release_capabilities(
    *,
    release: Any,
    annotation_binding: OracleAnnotationBinding,
    training_releases: Sequence[OracleTrainingAnnotationRelease],
) -> tuple[OracleTrainingAnnotationRelease, ...]:
    eligibility = _validate_annotation_binding(annotation_binding)
    frozen = tuple(training_releases)
    if (
        len(frozen) != len(FROZEN_OUTER_ROTATIONS)
        or tuple(row.outer_rotation for row in frozen) != FROZEN_OUTER_ROTATIONS
    ):
        raise ValueError("Stage-B requires exactly five ordered training releases")
    field = str(eligibility["field"])
    categories = tuple(eligibility["codebook_categories"])
    for training_release in frozen:
        expected_ids = tuple(
            sorted(
                (
                    participant_id
                    for participant_id in release.participant_ids
                    if release.participant_folds[participant_id]
                    not in {
                        training_release.outer_rotation,
                        (training_release.outer_rotation + 1) % len(FROZEN_OUTER_ROTATIONS),
                    }
                ),
                key=int,
            )
        )
        observed_ids = tuple(str(row.get("participant_id")) for row in training_release.rows)
        values = [row.get("value") for row in training_release.rows]
        if (
            training_release.field != field
            or any(set(row) != {"participant_id", "value"} for row in training_release.rows)
            or observed_ids != expected_ids
            or any(value not in categories for value in values)
        ):
            raise ValueError("Stage-B training release scope or values drifted")
        counts = Counter(values)
        expected_counts = tuple(
            {"category": category, "count": counts[category]} for category in categories
        )
        if training_release.category_counts != expected_counts:
            raise ValueError("Stage-B training release category counts drifted")
    return frozen


def _validate_annotation_binding(
    annotation_binding: OracleAnnotationBinding,
) -> Mapping[str, Any]:
    if not isinstance(annotation_binding, OracleAnnotationBinding):
        raise TypeError("Stage-B requires a controller-issued annotation binding")
    bindings = annotation_binding.annotation_bindings
    eligibility = annotation_binding.eligibility
    if (
        set(bindings) != _ANNOTATION_BINDING_FIELDS
        or annotation_binding.eligibility_payload_sha256
        != oracle_eligibility_payload_sha256(eligibility)
        or bindings.get("oracle_field") != eligibility.get("field")
        or bindings.get("codebook_version") != eligibility.get("codebook_version")
        or bindings.get("codebook_categories") != eligibility.get("codebook_categories")
        or bindings.get("annotation_release_method") != eligibility.get("release_method")
        or bindings.get("stage_b_designation") != eligibility.get("stage_b_designation")
        or bindings.get("designated_stage_b_field") != eligibility.get("designated_stage_b_field")
    ):
        raise ValueError("Stage-B annotation binding schema or eligibility drifted")
    for field in (
        "codebook_file_sha256",
        "codebook_payload_sha256",
    ):
        _required_sha256(bindings[field], f"Stage-B annotation binding {field}")
    participant_release = bindings.get("participant_annotation_release_file_sha256")
    source_gender_release = bindings.get("stage_b_gender_release_file_sha256")
    manifest_sha256 = bindings.get("source_release_manifest_file_sha256")
    if bindings.get("stage_b_designation") == "exploratory_post_designated_not_preregistered":
        _required_sha256(source_gender_release, "Stage-B source gender release checksum")
        if participant_release is not None:
            raise ValueError("source Stage-B binding contains a legacy participant release")
        _required_sha256(manifest_sha256, "Stage-B source release manifest checksum")
    else:
        _required_sha256(participant_release, "Stage-B participant annotation checksum")
        if source_gender_release is not None or manifest_sha256 is not None:
            raise ValueError("legacy Stage-B binding contains source-release artifacts")
    return eligibility


def build_oracle_access_log(
    *,
    release: Any,
    annotation_binding: OracleAnnotationBinding,
    training_releases: Sequence[OracleTrainingAnnotationRelease],
    seed: int,
) -> dict[str, Any]:
    """Build a deterministic proof of fold-training-only annotation release."""
    if seed not in FROZEN_FINAL_SEEDS:
        raise ValueError("Stage-B oracle seed is not frozen")
    eligibility = annotation_binding.eligibility
    _validate_eligibility(eligibility, expected_status="eligible_all_15")
    releases = _validate_training_release_capabilities(
        release=release,
        annotation_binding=annotation_binding,
        training_releases=training_releases,
    )
    field = str(eligibility["field"])
    entries = []
    for training_release in releases:
        fold = training_release.outer_rotation
        released = [dict(row) for row in training_release.rows]
        unsigned_entry = {
            "outer_rotation": fold,
            "release_scope": "training_participants_only",
            "released_fields": [field],
            "released_participant_ids": [row["participant_id"] for row in released],
            "released_participant_count": len(released),
            "released_annotations_sha256": training_release.released_annotations_sha256,
            "category_counts": [dict(row) for row in training_release.category_counts],
            "annotation_release_event_index": 2 * fold,
            "prediction_lock_event_index": 2 * fold + 1,
        }
        entries.append(_signed(unsigned_entry, "entry_payload_sha256"))
    unsigned = {
        "schema_version": STAGE_B_ORACLE_ACCESS_LOG_SCHEMA_VERSION,
        "configuration": f"oracle_{field}",
        "seed": seed,
        "stage_a_closure_file_sha256": release.closure_file_sha256,
        "stage_a_closure_payload_sha256": release.closure_payload_sha256,
        "oracle_eligibility_payload_sha256": oracle_eligibility_payload_sha256(eligibility),
        "annotation_bindings": dict(annotation_binding.annotation_bindings),
        "entries": entries,
        "protected_validation_participant_ids_released": [],
        "protected_test_participant_ids_released": [],
    }
    return _signed(unsigned, ORACLE_ACCESS_CHECKSUM_FIELD)


def oracle_prediction_lock_id(manifest_without_ids: Mapping[str, Any]) -> str:
    """Compute the lock ID from the frozen non-prediction identity fields."""
    if any(field not in manifest_without_ids for field in _LOCK_IDENTITY_FIELDS):
        raise ValueError("Stage-B oracle prediction-lock identity fields are incomplete")
    identity = {field: manifest_without_ids[field] for field in _LOCK_IDENTITY_FIELDS}
    return _payload_sha256(identity)


def _common_unsigned(
    *,
    release: Any,
    annotation_binding: OracleAnnotationBinding,
    stage_b_execution_provenance: Mapping[str, Any],
    state: str,
    authorized_fit_count: int,
    completed_fit_count: int,
    completed_bundle_count: int,
    bundles: object,
    omission_reasons: list[str],
) -> dict[str, Any]:
    eligibility = _validate_annotation_binding(annotation_binding)
    field = eligibility.get("field")
    execution = validate_stage_b_execution_provenance(stage_b_execution_provenance)
    context = release.closure.get("context")
    if (
        not isinstance(context, Mapping)
        or execution["config_sha256"] != context.get("config_sha256")
        or execution["source_date_epoch"] != context.get("source_date_epoch")
    ):
        raise ValueError(
            "Stage-B execution provenance does not preserve the Stage-A config/timestamp binding"
        )
    return {
        "schema_version": STAGE_B_ORACLE_LOCK_SCHEMA_VERSION,
        "state": state,
        "configuration": f"oracle_{field}",
        "firewall_stage": "B",
        "stage_a_closure_file_sha256": release.closure_file_sha256,
        "stage_a_closure_payload_sha256": release.closure_payload_sha256,
        "oracle_eligibility_payload_sha256": oracle_eligibility_payload_sha256(eligibility),
        "stage_b_execution_provenance": execution,
        "annotation_bindings": dict(annotation_binding.annotation_bindings),
        "eligible_participant_count": release.participant_count,
        "frozen_outer_folds": list(FROZEN_OUTER_ROTATIONS),
        "frozen_seeds": list(FROZEN_FINAL_SEEDS),
        "expected_fit_count": ORACLE_EXPECTED_FIT_COUNT,
        "authorized_fit_count": authorized_fit_count,
        "completed_fit_count": completed_fit_count,
        "expected_bundle_count": ORACLE_EXPECTED_BUNDLE_COUNT,
        "completed_bundle_count": completed_bundle_count,
        "expected_oof_rows_per_seed": 2 * release.participant_count,
        "protected_annotation_access_contract": {
            **_ACCESS_CONTRACT,
            "released_field": field,
        },
        "bundles": bundles,
        "omission_reasons": omission_reasons,
    }


def _validate_eligibility(eligibility: Mapping[str, Any], *, expected_status: str) -> None:
    if not isinstance(eligibility, Mapping):
        raise TypeError("oracle eligibility must be a mapping")
    categories = eligibility.get("codebook_categories")
    if (
        eligibility.get("status") != expected_status
        or not isinstance(eligibility.get("field"), str)
        or not isinstance(categories, list)
    ):
        raise ValueError("oracle eligibility status or schema drifted")
    if expected_status == "eligible_all_15":
        if (
            len(categories) != 2
            or eligibility.get("authorized_fit_count") != ORACLE_EXPECTED_FIT_COUNT
            or eligibility.get("omitted_fit_count") != 0
            or eligibility.get("reasons") != []
            or not isinstance(eligibility.get("rotations"), list)
            or len(eligibility["rotations"]) != 5
            or any(row.get("eligible") is not True for row in eligibility["rotations"])
        ):
            raise ValueError("oracle eligible report does not authorize exactly all 15 fits")
    elif (
        eligibility.get("authorized_fit_count") != 0
        or eligibility.get("omitted_fit_count") != ORACLE_EXPECTED_FIT_COUNT
        or not eligibility.get("reasons")
    ):
        raise ValueError("oracle omission report does not omit exactly all 15 fits")


def _require_exact_eligibility(
    release: Any,
    annotations: Any,
    eligibility: Mapping[str, Any],
) -> None:
    from .stage_c import validate_exact_binary_oracle_eligibility

    field = eligibility.get("field") if isinstance(eligibility, Mapping) else None
    if not isinstance(field, str):
        raise ValueError("oracle eligibility lacks one frozen field")
    expected = validate_exact_binary_oracle_eligibility(
        release,
        annotations,
        field=field,
    )
    if dict(eligibility) != expected:
        raise ValueError("oracle eligibility is not the exact locally recomputed report")


def _validate_bundle(
    anchor: AnchoredDirectory,
    bundle: object,
    *,
    file_mode: int,
    release: Any,
    eligibility: Mapping[str, Any],
    annotation_binding: OracleAnnotationBinding,
    training_releases: Sequence[OracleTrainingAnnotationRelease],
    top_payload: Mapping[str, Any],
) -> tuple[int, tuple[Mapping[str, Any], ...]]:
    if not isinstance(bundle, Mapping) or set(bundle) != _BUNDLE_FIELDS:
        raise ValueError("Stage-B oracle bundle record schema drifted")
    seed = bundle["seed"]
    configuration = top_payload["configuration"]
    if bundle["configuration"] != configuration or seed not in FROZEN_FINAL_SEEDS:
        raise ValueError("Stage-B oracle bundle identity drifted")
    expected_names = {
        "prediction_lock_filename": f"{configuration}.seed-{seed}.prediction-lock.json",
        "prediction_filename": f"{configuration}.seed-{seed}.predictions.json",
        "access_log_filename": f"{configuration}.seed-{seed}.access-log.json",
    }
    if any(bundle[field] != value for field, value in expected_names.items()):
        raise ValueError("Stage-B oracle bundle filename registry drifted")
    manifest_record = anchor.read_bytes(
        str(bundle["prediction_lock_filename"]), expected_mode=file_mode
    )
    prediction_record = anchor.read_bytes(
        str(bundle["prediction_filename"]), expected_mode=file_mode
    )
    access_record = anchor.read_bytes(str(bundle["access_log_filename"]), expected_mode=file_mode)
    if (
        manifest_record.sha256 != bundle["prediction_lock_file_sha256"]
        or prediction_record.sha256 != bundle["prediction_file_sha256"]
        or access_record.sha256 != bundle["access_log_file_sha256"]
    ):
        raise ValueError("Stage-B oracle bundle file checksum failed")

    access = _canonical_object_from_bytes(access_record.content, "Stage-B oracle access log")
    _validate_access_log(
        access,
        release=release,
        annotation_binding=annotation_binding,
        training_releases=training_releases,
        seed=seed,
    )
    if access[ORACLE_ACCESS_CHECKSUM_FIELD] != bundle["access_log_payload_sha256"]:
        raise ValueError("Stage-B oracle access-log payload binding failed")

    manifest = _canonical_object_from_bytes(
        manifest_record.content, "Stage-B oracle prediction lock"
    )
    _validate_prediction_lock(
        manifest,
        anchor=anchor,
        file_mode=file_mode,
        release=release,
        annotation_binding=annotation_binding,
        training_releases=training_releases,
        seed=seed,
        access=access,
        bundle=bundle,
        top_payload=top_payload,
    )
    prediction = _canonical_object_from_bytes(
        prediction_record.content, "Stage-B oracle predictions"
    )
    rows = _validate_prediction_payload(
        prediction,
        manifest=manifest,
        release=release,
        eligibility=eligibility,
        seed=seed,
    )
    if (
        prediction[ORACLE_PREDICTION_CHECKSUM_FIELD] != bundle["prediction_payload_sha256"]
        or len(rows) != bundle["oof_row_count"]
    ):
        raise ValueError("Stage-B oracle prediction payload binding failed")
    return int(seed), rows


def _validate_access_log(
    payload: Mapping[str, Any],
    *,
    release: Any,
    annotation_binding: OracleAnnotationBinding,
    training_releases: Sequence[OracleTrainingAnnotationRelease],
    seed: int,
) -> None:
    _validate_signed(payload, ORACLE_ACCESS_CHECKSUM_FIELD, "Stage-B oracle access log")
    expected = build_oracle_access_log(
        release=release,
        annotation_binding=annotation_binding,
        training_releases=training_releases,
        seed=seed,
    )
    if set(payload) != _ACCESS_LOG_FIELDS or payload != expected:
        raise ValueError("Stage-B access log is not exact training-only release proof")
    if (
        payload["protected_validation_participant_ids_released"] != []
        or payload["protected_test_participant_ids_released"] != []
    ):
        raise ValueError("Stage-B access log released validation/test annotations")


def _validate_prediction_lock(
    manifest: Mapping[str, Any],
    *,
    anchor: AnchoredDirectory,
    file_mode: int,
    release: Any,
    annotation_binding: OracleAnnotationBinding,
    training_releases: Sequence[OracleTrainingAnnotationRelease],
    seed: int,
    access: Mapping[str, Any],
    bundle: Mapping[str, Any],
    top_payload: Mapping[str, Any],
) -> None:
    _validate_signed(
        manifest,
        ORACLE_PREDICTION_LOCK_CHECKSUM_FIELD,
        "Stage-B oracle prediction lock",
    )
    eligibility = annotation_binding.eligibility
    context = release.closure["context"]
    access_record = manifest.get("annotation_access_log")
    prediction_record = manifest.get("predictions")
    rotations = manifest.get("rotations")
    if set(manifest) != _PREDICTION_LOCK_FIELDS:
        raise ValueError("Stage-B oracle prediction-lock schema drifted")
    execution = validate_stage_b_execution_provenance(manifest.get("stage_b_execution_provenance"))
    if execution != top_payload.get("stage_b_execution_provenance"):
        raise ValueError("Stage-B top lock and prediction lock provenance disagree")
    if (
        manifest["schema_version"] != STAGE_B_ORACLE_PREDICTION_LOCK_SCHEMA_VERSION
        or manifest["state"] != "locked"
        or manifest["configuration"] != bundle["configuration"]
        or manifest["seed"] != seed
        or manifest["firewall_stage"] != "B"
        or manifest["stage_a_closure_file_sha256"] != release.closure_file_sha256
        or manifest["stage_a_closure_payload_sha256"] != release.closure_payload_sha256
        or manifest["oracle_eligibility_payload_sha256"]
        != oracle_eligibility_payload_sha256(eligibility)
        or manifest["annotation_bindings"] != bundle_annotation_bindings(bundle, access)
        or manifest["protected_training_annotation_scope"]
        != "outer_rotation_training_participants_only"
        or manifest["protected_validation_annotation_fields_accessed"] != []
        or manifest["protected_test_annotation_fields_accessed"] != []
        or manifest["checkpoint_selection"]
        != "ordinary_validation_srcc_without_protected_annotations"
        or manifest["target_manifest_sha256"] != context["target_manifest_sha256"]
        or manifest["eligibility_manifest_sha256"] != context["eligibility_manifest_sha256"]
        or manifest["feature_contract_sha256"] != context["feature_contract_sha256"]
        or manifest["preprocessing_sha256"] != context["preprocessing_sha256"]
        or manifest["feature_extraction_closure_file_sha256"]
        != context["feature_extraction_closure_file_sha256"]
        or manifest["feature_extraction_closure_payload_sha256"]
        != context["feature_extraction_closure_payload_sha256"]
        or manifest["session_mapping"] != context["session_mapping"]
        or manifest["config_sha256"] != execution["config_sha256"]
        or manifest["config_sha256"] != context["config_sha256"]
        or manifest["code_commit_sha"] != execution["code_commit_sha"]
        or manifest["code_source_tree_sha256"] != execution["code_source_tree_sha256"]
        or manifest["source_date_epoch"] != execution["source_date_epoch"]
        or manifest["source_date_epoch"] != context["source_date_epoch"]
        or manifest["eligible_participant_count"] != release.participant_count
        or manifest["expected_oof_session_count"] != 2 * release.participant_count
        or manifest["observed_oof_session_count"] != 2 * release.participant_count
        or not _is_sha256(manifest["model_config_sha256"])
        or not isinstance(access_record, Mapping)
        or access_record
        != {
            "filename": bundle["access_log_filename"],
            "file_sha256": bundle["access_log_file_sha256"],
            "payload_sha256": bundle["access_log_payload_sha256"],
        }
        or not isinstance(prediction_record, Mapping)
        or prediction_record
        != {
            "filename": bundle["prediction_filename"],
            "file_sha256": bundle["prediction_file_sha256"],
            "payload_sha256": bundle["prediction_payload_sha256"],
            "row_count": bundle["oof_row_count"],
        }
    ):
        raise ValueError("Stage-B oracle prediction-lock identity or firewall drifted")
    if manifest["prediction_lock_id"] != oracle_prediction_lock_id(manifest):
        raise ValueError("Stage-B oracle prediction-lock ID checksum failed")
    if (
        manifest["prediction_lock_id"] != bundle["prediction_lock_id"]
        or manifest[ORACLE_PREDICTION_LOCK_CHECKSUM_FIELD]
        != bundle["prediction_lock_payload_sha256"]
    ):
        raise ValueError("Stage-B top lock and prediction lock disagree")
    _validate_rotations(
        rotations,
        access,
        bundle,
        anchor=anchor,
        file_mode=file_mode,
        release=release,
        annotation_binding=annotation_binding,
        training_releases=training_releases,
        manifest=manifest,
    )


def bundle_annotation_bindings(
    bundle: Mapping[str, Any], access: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Return the access-log binding while keeping bundle validation expression explicit."""
    del bundle
    return access["annotation_bindings"]


def _validate_rotations(
    rotations: object,
    access: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    anchor: AnchoredDirectory,
    file_mode: int,
    release: Any,
    annotation_binding: OracleAnnotationBinding,
    training_releases: Sequence[OracleTrainingAnnotationRelease],
    manifest: Mapping[str, Any],
) -> None:
    from src.training.stage_b_oracle import validate_stage_b_oracle_fit_bytes

    if not isinstance(rotations, list) or len(rotations) != 5:
        raise ValueError("Stage-B oracle prediction lock must contain five rotations")
    access_entries = {row["outer_rotation"]: row for row in access["entries"]}
    bundle_fits = bundle.get("fit_artifacts")
    if not isinstance(bundle_fits, list) or len(bundle_fits) != 5:
        raise ValueError("Stage-B oracle top lock lacks five fit artifacts")
    for fold, (rotation, bundle_fit) in enumerate(zip(rotations, bundle_fits, strict=True)):
        if (
            not isinstance(rotation, Mapping)
            or set(rotation) != _ROTATION_FIELDS
            or rotation["outer_rotation"] != fold
            or rotation["training_annotation_count"]
            != access_entries[fold]["released_participant_count"]
            or rotation["access_log_entry_payload_sha256"]
            != access_entries[fold]["entry_payload_sha256"]
            or isinstance(rotation["checkpoint_epoch"], bool)
            or not isinstance(rotation["checkpoint_epoch"], int)
            or rotation["checkpoint_epoch"] <= 0
            or not isinstance(rotation["fit_artifact"], Mapping)
            or set(rotation["fit_artifact"]) != {"filename", "sha256"}
            or rotation["fit_artifact"] != bundle_fit
        ):
            raise ValueError("Stage-B oracle rotation/access/fit binding drifted")
        fit_record = anchor.read_bytes(str(bundle_fit["filename"]), expected_mode=file_mode)
        if fit_record.sha256 != bundle_fit["sha256"]:
            raise ValueError("Stage-B oracle fit artifact checksum failed")
        validate_stage_b_oracle_fit_bytes(
            fit_record.content,
            filename=str(bundle_fit["filename"]),
            release=release,
            annotation_binding=annotation_binding,
            training_release=training_releases[fold],
            seed=int(bundle["seed"]),
            outer_rotation=fold,
            access_log=access,
            prediction_lock_manifest=manifest,
        )


def _validate_prediction_payload(
    payload: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    release: Any,
    eligibility: Mapping[str, Any],
    seed: int,
) -> tuple[Mapping[str, Any], ...]:
    _validate_signed(
        payload,
        ORACLE_PREDICTION_CHECKSUM_FIELD,
        "Stage-B oracle predictions",
    )
    rows = payload.get("rows")
    if (
        set(payload) != _PREDICTION_PAYLOAD_FIELDS
        or payload["schema_version"] != STAGE_B_ORACLE_PREDICTION_SCHEMA_VERSION
        or payload["prediction_lock_id"] != manifest["prediction_lock_id"]
        or not isinstance(rows, list)
        or len(rows) != 2 * release.participant_count
    ):
        raise ValueError("Stage-B oracle prediction payload schema or count drifted")
    baseline = {row["session_id"]: row for row in release.learned_rows[("erm", seed)]}
    rotations = {row["outer_rotation"]: row for row in manifest["rotations"]}
    seen: set[str] = set()
    for row in rows:
        session_id = row.get("session_id") if isinstance(row, Mapping) else None
        expected = baseline.get(session_id)
        prediction = row.get("prediction") if isinstance(row, Mapping) else None
        fold = expected.get("fixed_fold") if isinstance(expected, Mapping) else None
        rotation = rotations.get(fold)
        if (
            not isinstance(row, Mapping)
            or set(row) != ORACLE_PREDICTION_ROW_FIELDS
            or expected is None
            or session_id in seen
            or row["schema_version"] != STAGE_B_ORACLE_PREDICTION_SCHEMA_VERSION
            or row["corpus"] != "mit_interview"
            or row["participant_id"] != expected["participant_id"]
            or row["source_session_id"] != expected["source_session_id"]
            or row["session_phase"] != expected["session_phase"]
            or row["fixed_fold"] != fold
            or row["outer_rotation"] != fold
            or row["rotation_role"] != "test"
            or row["seed"] != seed
            or row["configuration"] != manifest["configuration"]
            or row["model_config_sha256"] != manifest["model_config_sha256"]
            or row["model_fit_sha256"] != rotation["fit_artifact"]["sha256"]
            or row["target_transform"] != TARGET_TRANSFORM
            or row["target_manifest_sha256"] != manifest["target_manifest_sha256"]
            or row["target"] != expected["target"]
            or isinstance(prediction, bool)
            or not isinstance(prediction, (int, float))
            or not math.isfinite(float(prediction))
            or not 0 <= float(prediction) <= 1
            or row["feature_contract_sha256"] != manifest["feature_contract_sha256"]
            or row["preprocessing_sha256"] != manifest["preprocessing_sha256"]
            or row["checkpoint_epoch"] != rotation["checkpoint_epoch"]
            or row["firewall_stage"] != "B"
            or row["eligibility_manifest_sha256"] != manifest["eligibility_manifest_sha256"]
            or row["session_mapping"] != manifest["session_mapping"]
            or row["oracle_field"] != eligibility["field"]
            or row["annotation_access_log_payload_sha256"]
            != manifest["annotation_access_log"]["payload_sha256"]
            or row["prediction_lock_id"] != manifest["prediction_lock_id"]
        ):
            raise ValueError("Stage-B oracle OOF row identity, target or provenance drifted")
        seen.add(str(session_id))
    if seen != set(baseline):
        raise ValueError("Stage-B oracle OOF rows do not exactly cover the Stage-A roster")
    return tuple(rows)


def _validate_fold(value: Any) -> int:
    if isinstance(value, bool) or value not in FROZEN_OUTER_ROTATIONS:
        raise ValueError("outer rotation is not frozen")
    return int(value)


def _validate_oracle_inventory(anchor: AnchoredDirectory, expected: set[str]) -> None:
    observed = {name for name in anchor.list_names() if name.startswith("oracle_")}
    if observed != expected:
        raise ValueError("Stage-B oracle artifact inventory is not exactly all fifteen or zero")


def _signed(unsigned: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {**unsigned, field: _payload_sha256(unsigned)}


def _validate_signed(payload: Mapping[str, Any], field: str, label: str) -> None:
    observed = _required_sha256(payload.get(field), f"{label} checksum")
    unsigned = {key: value for key, value in payload.items() if key != field}
    if observed != _payload_sha256(unsigned):
        raise ValueError(f"{label} payload checksum failed")


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _required_sha256(value: Any, label: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return str(value)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _HEX64.fullmatch(value) is not None


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _canonical_object_from_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"cannot read valid {label} JSON") from error
    if not isinstance(payload, dict) or canonical_json_bytes(payload) != raw:
        raise ValueError(f"{label} JSON is not byte-canonical")
    return payload


def _open_existing_artifact_directory(path: Path) -> AnchoredDirectory:
    errors = []
    for mode in (0o700, 0o755):
        try:
            return AnchoredDirectory.open(path, create=False, directory_mode=mode)
        except ValueError as error:
            errors.append(error)
    raise ValueError("Stage-B oracle directory is unavailable or unsafe") from errors[-1]


def _read_existing_artifact(anchor: AnchoredDirectory, filename: str) -> tuple[AnchoredFile, int]:
    errors = []
    for mode in (0o600, 0o644):
        try:
            return anchor.read_bytes(filename, expected_mode=mode), mode
        except ValueError as error:
            errors.append(error)
    raise ValueError("Stage-B oracle lock is unavailable or unsafe") from errors[-1]


def _lexical_output_path(path: str | Path) -> Path:
    destination = Path(path).expanduser()
    if any(component in {".", ".."} for component in destination.parts):
        raise ValueError("Stage-B oracle output path contains unsafe dot components")
    if not destination.is_absolute():
        destination = Path.cwd() / destination
    return destination
