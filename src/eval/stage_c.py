"""One-way Stage-C audit of immutable out-of-fold predictions.

This module deliberately has no model-fitting entry point.  It accepts only a
mechanically closed Stage-A directory, checksum-bound annotation inputs and,
optionally, separately locked inference-only modality-mask predictions.  The
participant-level join is restricted; every other output is aggregate.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from importlib.metadata import version
from pathlib import Path
from types import MappingProxyType
from typing import Any
from weakref import WeakKeyDictionary

import numpy as np

from src.data.annotation_audit import validate_annotation_audit_report
from src.data.annotation_stimuli import load_stimulus_manifest
from src.data.protected_annotations import (
    CONSENSUS_METHOD,
    SKIN_TONE_CONSENSUS_VALUES,
    AnnotationCodebook,
    ConsensusAnnotation,
    build_consensus_annotations,
    load_annotation_codebook,
    load_consensus_annotations,
    load_rater_annotations,
)
from src.data.source_annotation_release import (
    EXPECTED_PRE_COUNT as EXPECTED_SOURCE_PRE_COUNT,
)
from src.data.source_annotation_release import (
    SOURCE_RELEASE_METHOD,
    SOURCE_STAGE_B_DESIGNATION,
    SourceAnnotationCodebook,
    SourceSessionAnnotation,
    load_source_annotation_release,
)
from src.features.extraction import code_provenance
from src.frozen_json import freeze_json
from src.secure_io import AnchoredDirectory, AnchoredFile
from src.training.crossfit import (
    DIAGNOSTIC_FLOOR_CONFIGURATIONS,
    FIREWALLED_LEARNED_CONFIGURATIONS,
    PREDICTION_LOCK_SCHEMA_VERSION,
    PREDICTION_ROW_FIELDS,
    PREDICTION_SCHEMA_VERSION,
    TARGET_TRANSFORM,
    canonical_json_bytes,
)
from src.training.proxy_artifacts import PROXY_ARTIFACT_SCHEMA_VERSION
from src.training.proxy_training import FROZEN_FINAL_SEEDS, FROZEN_OUTER_ROTATIONS
from src.training.stage_a_ledger import (
    STAGE_A_CLOSURE_FILENAME,
    require_stage_b_release,
)

from .metrics import empirical_wasserstein1
from .stage_b_oracle import (
    StageBOracleDisposition,
    validate_stage_b_oracle_lock,
)

STAGE_C_RESTRICTED_JOIN_SCHEMA_VERSION = "fair-avi-mit-stage-c-restricted-join-2"
STAGE_C_AGGREGATE_SCHEMA_VERSION = "fair-avi-mit-stage-c-aggregate-2"
STAGE_C_CLAIM_LEDGER_SCHEMA_VERSION = "fair-avi-mit-stage-c-claim-ledger-1"
STAGE_C_NO_RERUN_SCHEMA_VERSION = "fair-avi-mit-stage-c-no-rerun-1"
STAGE_C_MASK_SCHEMA_VERSION = "fair-avi-mit-stage-c-modality-mask-1"
STAGE_C_EVALUATOR_VERSION = "stage-c-evaluator-v4"
STAGE_C_EVALUATOR_PROVENANCE_SCHEMA_VERSION = "fair-avi-mit-stage-c-evaluator-provenance-1"
STAGE_C_EVALUATOR_OWNED_SOURCE_PATHS = (
    "pyproject.toml",
    "requirements.lock",
    "scripts/evaluate_stage_c.py",
    "src/__init__.py",
    "src/data/__init__.py",
    "src/data/annotation_audit.py",
    "src/data/annotation_intake.py",
    "src/data/annotation_stimuli.py",
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

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 2026
CONFIDENCE_LEVEL = 0.95
MINIMUM_REPORTABLE_PARTICIPANTS = 10
RAW_SCALE_WIDTH = 6.0
MASKED_MODALITIES = ("video", "interview_audio", "candidate_text")
PERCEIVED_FIELDS = (
    "perceived_gender_presentation",
    "perceived_racialised_appearance",
    "perceived_skin_tone",
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_HEX40 = re.compile(r"^[0-9a-f]{40}$", re.ASCII)
_PARTICIPANT_ID = re.compile(r"^[1-9][0-9]*$", re.ASCII)
_PROHIBITED_PUBLIC_KEYS = {
    "participant_id",
    "rater_id",
    "session_id",
    "source_session_id",
    "prediction_lock_id",
    "model_fit_sha256",
}
_RESTRICTED_JOIN_FIELDS = {
    "schema_version",
    "classification",
    "evaluator_version",
    "stage_a_closure_file_sha256",
    "stage_a_closure_payload_sha256",
    "annotation_bindings",
    "stage_b_oracle",
    "participant_count",
    "rows",
    "join_payload_sha256",
}
_RESTRICTED_ROW_COMMON_FIELDS = {
    "participant_id",
    "session_id",
    "source_session_id",
    "session_phase",
    "fixed_fold",
    "target",
    "predictions",
    "assessability",
    "proxy_assignments",
}
_LEGACY_RESTRICTED_ROW_FIELDS = {*_RESTRICTED_ROW_COMMON_FIELDS, "consensus"}
_SOURCE_RESTRICTED_ROW_FIELDS = {
    *_RESTRICTED_ROW_COMMON_FIELDS,
    "perceived_attributes",
}
_AGGREGATE_FIELDS = {
    "schema_version",
    "scope",
    "evaluator_version",
    "evaluator_provenance",
    "execution_mode",
    "stage_a",
    "annotations",
    "utility",
    "three_seed_sensitivity_summary",
    "phase_stratified_utility",
    "paired_change",
    "audited_disparity",
    "group_residual_and_mae",
    "pre_session_sensitivity",
    "contrasts",
    "participant_cluster_bootstrap",
    "proxy_diagnostics",
    "modality_masks",
    "exact_binary_oracle_eligibility",
    "stage_b_oracle",
    "interpretation_contract",
    "aggregate_payload_sha256",
}
_CLAIM_LEDGER_FIELDS = {
    "schema_version",
    "aggregate_payload_sha256",
    "claim_rows",
    "claim_language_review_required",
    "claim_ledger_payload_sha256",
}
_NO_RERUN_FIELDS = {
    "schema_version",
    "state",
    "firewall_transition",
    "evaluator_version",
    "stage_a_closure_file_sha256",
    "stage_a_closure_payload_sha256",
    "annotation_bindings",
    "stage_b_oracle",
    "restricted_join",
    "aggregate",
    "claim_ledger",
    "ordinary_model_rerun_after_audit",
    "checkpoint_or_proxy_selection_after_audit",
    "new_model_seed_fold_or_sweep_after_audit",
    "permitted_follow_up",
    "no_rerun_payload_sha256",
}

CONTRASTS: tuple[dict[str, str], ...] = (
    {
        "contrast_id": "visual_w1_total",
        "intervention": "face_w1",
        "reference": "erm",
        "family": "confirmatory_total_effect",
    },
    {
        "contrast_id": "audio_session_w1_total",
        "intervention": "audio_session_w1",
        "reference": "erm",
        "family": "confirmatory_total_effect",
    },
    {
        "contrast_id": "visual_w1_incremental",
        "intervention": "face_w1",
        "reference": "face_sampling_only",
        "family": "secondary_component",
    },
    {
        "contrast_id": "audio_session_w1_incremental",
        "intervention": "audio_session_w1",
        "reference": "audio_session_sampling_only",
        "family": "secondary_component",
    },
    {
        "contrast_id": "random_proxy_w1_control",
        "intervention": "random_proxy_w1",
        "reference": "erm",
        "family": "mechanism_control",
    },
    {
        "contrast_id": "arl_control",
        "intervention": "arl",
        "reference": "erm",
        "family": "mechanism_control",
    },
)

_STAGE_C_EVALUATOR_COMPONENT_PATHS = STAGE_C_EVALUATOR_OWNED_SOURCE_PATHS
_STAGE_C_EVALUATOR_PROVENANCE_FIELDS = {
    "schema_version",
    "owned_source_paths",
    "component_source_sha256",
    "code_commit_sha",
    "code_source_tree_sha256",
    "working_tree_dirty",
    "runtime_versions",
}
_STAGE_C_RUNTIME_VERSION_FIELDS = {
    "python",
    "numpy",
    "scipy",
    "scikit-learn",
    "torch",
}


def _current_evaluator_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    components: dict[str, str] = {}
    for relative in _STAGE_C_EVALUATOR_COMPONENT_PATHS:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Stage-C evaluator component is unavailable: {relative}")
        components[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    code = code_provenance(root, STAGE_C_EVALUATOR_OWNED_SOURCE_PATHS)
    result = {
        "schema_version": STAGE_C_EVALUATOR_PROVENANCE_SCHEMA_VERSION,
        "owned_source_paths": list(STAGE_C_EVALUATOR_OWNED_SOURCE_PATHS),
        "component_source_sha256": components,
        "code_commit_sha": code.commit_sha,
        "code_source_tree_sha256": code.source_tree_sha256,
        "working_tree_dirty": code.dirty,
        "runtime_versions": _runtime_versions(),
    }
    return validate_stage_c_evaluator_provenance(result)


def validate_stage_c_evaluator_provenance(value: object) -> dict[str, Any]:
    """Validate recorded Stage-C code identity without consulting a later checkout."""
    if not isinstance(value, Mapping) or set(value) != _STAGE_C_EVALUATOR_PROVENANCE_FIELDS:
        raise ValueError("Stage-C evaluator provenance schema drifted")
    owned = value.get("owned_source_paths")
    components = value.get("component_source_sha256")
    if (
        value.get("schema_version") != STAGE_C_EVALUATOR_PROVENANCE_SCHEMA_VERSION
        or owned != list(STAGE_C_EVALUATOR_OWNED_SOURCE_PATHS)
        or not isinstance(components, Mapping)
        or tuple(components) != _STAGE_C_EVALUATOR_COMPONENT_PATHS
    ):
        raise ValueError("Stage-C evaluator provenance source inventory drifted")
    component_hashes: dict[str, str] = {}
    for relative, digest in components.items():
        if not isinstance(digest, str) or not _HEX64.fullmatch(digest):
            raise ValueError(f"Stage-C evaluator component checksum is invalid: {relative}")
        component_hashes[str(relative)] = digest
    commit = value.get("code_commit_sha")
    source_tree = value.get("code_source_tree_sha256")
    if not isinstance(commit, str) or not _HEX40.fullmatch(commit):
        raise ValueError("Stage-C evaluator provenance commit is invalid")
    if not isinstance(source_tree, str) or not _HEX64.fullmatch(source_tree):
        raise ValueError("Stage-C evaluator provenance source-tree checksum is invalid")
    if source_tree != _stage_c_source_tree_sha256(component_hashes):
        raise ValueError("Stage-C evaluator provenance component/tree binding failed")
    dirty = value.get("working_tree_dirty")
    if not isinstance(dirty, bool):
        raise ValueError("Stage-C evaluator provenance worktree state is invalid")
    runtime_versions = value.get("runtime_versions")
    if (
        not isinstance(runtime_versions, Mapping)
        or set(runtime_versions) != _STAGE_C_RUNTIME_VERSION_FIELDS
        or any(
            not isinstance(runtime_versions[name], str) or not runtime_versions[name]
            for name in _STAGE_C_RUNTIME_VERSION_FIELDS
        )
    ):
        raise ValueError("Stage-C evaluator provenance runtime versions are invalid")
    return {
        "schema_version": STAGE_C_EVALUATOR_PROVENANCE_SCHEMA_VERSION,
        "owned_source_paths": list(STAGE_C_EVALUATOR_OWNED_SOURCE_PATHS),
        "component_source_sha256": component_hashes,
        "code_commit_sha": commit,
        "code_source_tree_sha256": source_tree,
        "working_tree_dirty": dirty,
        "runtime_versions": {
            name: str(runtime_versions[name]) for name in sorted(_STAGE_C_RUNTIME_VERSION_FIELDS)
        },
    }


def _stage_c_source_tree_sha256(components: Mapping[str, str]) -> str:
    """Reconstruct the exact code_provenance tree hash from recorded components."""
    digest = hashlib.sha256()
    for relative in STAGE_C_EVALUATOR_OWNED_SOURCE_PATHS:
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(bytes.fromhex(components[relative]))
    return digest.hexdigest()


def _runtime_versions() -> dict[str, str]:
    """Return the numerical runtime versions that can affect Stage-C results."""
    return {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "scikit-learn": version("scikit-learn"),
        "torch": version("torch"),
    }


_IMPORTED_EVALUATOR_PROVENANCE = freeze_json(_current_evaluator_provenance())


@dataclass(frozen=True, slots=True)
class StageARelease:
    """Checksum-validated Stage-A predictions arranged on one paired roster."""

    root: Path
    closure: Mapping[str, Any]
    closure_file_sha256: str
    closure_payload_sha256: str
    learned_rows: Mapping[tuple[str, int], tuple[Mapping[str, Any], ...]]
    floor_rows: Mapping[str, tuple[Mapping[str, Any], ...]]
    lock_manifests: Mapping[tuple[str, int | None], Mapping[str, Any]]
    participant_ids: tuple[str, ...]
    participant_folds: Mapping[str, int]
    session_keys: tuple[tuple[str, str, str], ...]

    def __post_init__(self) -> None:
        """Detach every validated value from caller-owned mutable containers."""
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "closure", freeze_json(self.closure))
        object.__setattr__(
            self,
            "learned_rows",
            MappingProxyType(
                {
                    key: tuple(freeze_json(row) for row in rows)
                    for key, rows in self.learned_rows.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "floor_rows",
            MappingProxyType(
                {
                    key: tuple(freeze_json(row) for row in rows)
                    for key, rows in self.floor_rows.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "lock_manifests",
            MappingProxyType(
                {key: freeze_json(value) for key, value in self.lock_manifests.items()}
            ),
        )
        object.__setattr__(self, "participant_ids", tuple(self.participant_ids))
        object.__setattr__(
            self, "participant_folds", MappingProxyType(dict(self.participant_folds))
        )
        object.__setattr__(self, "session_keys", tuple(self.session_keys))

    @property
    def participant_count(self) -> int:
        return len(self.participant_ids)


@dataclass(frozen=True, slots=True)
class AnnotationRelease:
    """Validated restricted annotation inputs and their exact file bindings."""

    codebook: AnnotationCodebook | SourceAnnotationCodebook
    participant_annotations: Mapping[str, ConsensusAnnotation | SourceSessionAnnotation]
    assessability: Mapping[str, str]
    audit_report: Mapping[str, Any]
    bindings: Mapping[str, str]
    release_method: str = CONSENSUS_METHOD
    session_annotations: Mapping[tuple[str, str], ConsensusAnnotation | SourceSessionAnnotation] = (
        dataclass_field(default_factory=dict)
    )
    designated_stage_b_field: str | None = None
    stage_b_designation: str = "legacy_frozen_codebook_axis"

    def __post_init__(self) -> None:
        """Detach protected evidence from every caller-owned mutable mapping."""
        object.__setattr__(
            self,
            "participant_annotations",
            MappingProxyType(dict(self.participant_annotations)),
        )
        object.__setattr__(self, "assessability", MappingProxyType(dict(self.assessability)))
        object.__setattr__(self, "audit_report", freeze_json(self.audit_report))
        object.__setattr__(self, "bindings", freeze_json(self.bindings))
        object.__setattr__(
            self, "session_annotations", MappingProxyType(dict(self.session_annotations))
        )
        if self.release_method == SOURCE_RELEASE_METHOD:
            if (
                self.designated_stage_b_field != "perceived_gender_presentation"
                or self.stage_b_designation != SOURCE_STAGE_B_DESIGNATION
                or not self.session_annotations
            ):
                raise ValueError("source annotation release metadata is incomplete")


@dataclass(frozen=True, slots=True)
class StageCSourcePaths:
    """Exact production paths reopened at every Stage-C evidence boundary."""

    stage_a_dir: Path
    stage_b_oracle_disposition: Path
    modality_mask_predictions: Path
    codebook: Path | None = None
    consensus: Path | None = None
    annotation_audit: Path | None = None
    rater_annotations: Path | None = None
    stimulus_manifest: Path | None = None
    source_annotation_release: Path | None = None

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if value is None:
                continue
            supplied = Path(value).expanduser()
            if any(component in {".", ".."} for component in supplied.parts):
                raise ValueError("Stage-C source paths may not contain dot components")
            if not supplied.is_absolute():
                supplied = Path.cwd() / supplied
            object.__setattr__(self, name, supplied)
        legacy = (
            self.codebook,
            self.consensus,
            self.annotation_audit,
            self.rater_annotations,
            self.stimulus_manifest,
        )
        if self.source_annotation_release is None and any(value is None for value in legacy):
            raise ValueError("Stage-C requires all legacy annotation paths or a source manifest")
        if self.source_annotation_release is not None and any(
            value is not None for value in legacy
        ):
            raise ValueError("Stage-C annotation source branches are mutually exclusive")
        if self.stage_b_oracle_disposition is None or self.modality_mask_predictions is None:
            raise ValueError("Stage-C requires Stage-B disposition and modality-mask paths")


@dataclass(frozen=True, slots=True)
class _ProductionSnapshot:
    release: StageARelease
    annotations: AnnotationRelease
    oracle_eligibility: Mapping[str, Any]
    oracle_disposition: StageBOracleDisposition
    modality_masks: Mapping[str, Any]
    fingerprint_bytes: bytes


@dataclass(frozen=True, slots=True)
class _CapturedProtectedSource:
    """One exact protected source held beneath an anchored lexical parent."""

    path: Path
    record: AnchoredFile


@dataclass(frozen=True, slots=True)
class _StageCResultState:
    restricted_join_bytes: bytes
    aggregate_bytes: bytes
    release: StageARelease
    annotations: AnnotationRelease
    source_paths: StageCSourcePaths | None
    source_fingerprint_bytes: bytes | None


def _build_result_capability():
    """Keep result state in a closure so callers cannot replace or re-sign it."""
    states: WeakKeyDictionary[StageCEvaluationResult, _StageCResultState] = WeakKeyDictionary()

    class StageCEvaluationResult:
        """Opaque canonical result produced only by a completed Stage-C evaluation."""

        __slots__ = ("__weakref__",)

        def __new__(cls, *_args: object, **_kwargs: object) -> StageCEvaluationResult:
            raise TypeError("Stage-C results may be constructed only by the evaluator")

        @property
        def restricted_join(self) -> dict[str, Any]:
            """Return a detached copy; mutation cannot alter publication bytes."""
            state = require(self)
            return _canonical_object_from_bytes(
                state.restricted_join_bytes, "Stage-C restricted result"
            )

        @property
        def aggregate(self) -> dict[str, Any]:
            """Return a detached copy; mutation cannot alter publication bytes."""
            state = require(self)
            return _canonical_object_from_bytes(state.aggregate_bytes, "Stage-C aggregate result")

        def __iter__(self):
            """Preserve synthetic-test unpacking without exposing internal state."""
            yield self.restricted_join
            yield self.aggregate

    def create(state: _StageCResultState) -> StageCEvaluationResult:
        result = object.__new__(StageCEvaluationResult)
        states[result] = state
        return result

    def require(result: object) -> _StageCResultState:
        if not isinstance(result, StageCEvaluationResult):
            raise TypeError("Stage-C publication requires an opaque evaluator result")
        try:
            return states[result]
        except KeyError:
            raise TypeError(
                "Stage-C publication requires an internally constructed evaluator result"
            ) from None

    return StageCEvaluationResult, create, require


StageCEvaluationResult, _create_stage_c_result, _stage_c_result_state = _build_result_capability()


def _make_stage_c_result(
    restricted_join: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    *,
    release: StageARelease,
    annotations: AnnotationRelease,
    source_paths: StageCSourcePaths | None,
    source_fingerprint_bytes: bytes | None,
) -> StageCEvaluationResult:
    """Validate once, canonicalize once and seal all publication state."""
    restricted_bytes = canonical_json(restricted_join)
    aggregate_bytes = canonical_json(aggregate)
    restricted = _canonical_object_from_bytes(restricted_bytes, "Stage-C restricted result")
    public = _canonical_object_from_bytes(aggregate_bytes, "Stage-C aggregate result")
    _validate_stage_c_payload_pair(
        restricted,
        public,
        release=release,
        annotations=annotations,
    )
    if (source_paths is None) != (source_fingerprint_bytes is None):
        raise ValueError("Stage-C production source binding is incomplete")
    return _create_stage_c_result(
        _StageCResultState(
            restricted_join_bytes=restricted_bytes,
            aggregate_bytes=aggregate_bytes,
            release=release,
            annotations=annotations,
            source_paths=source_paths,
            source_fingerprint_bytes=source_fingerprint_bytes,
        )
    )


def canonical_json(payload: Mapping[str, Any]) -> bytes:
    """Return the repository's byte-canonical JSON representation."""
    return canonical_json_bytes(payload)


def load_stage_a_release(stage_a_dir: str | Path) -> StageARelease:
    """Open Stage C only after the shared Stage-A gate and all file hashes pass."""
    root = _regular_directory(stage_a_dir, "Stage-A output")
    closure_path = root / STAGE_A_CLOSURE_FILENAME
    closure = _read_canonical_object(closure_path, "Stage-A closure")
    closure_file_sha256 = _file_sha256(closure_path)
    closure_payload_sha256 = _required_sha256(
        closure.get("closure_payload_sha256"), "Stage-A closure payload checksum"
    )
    validated = require_stage_b_release(
        root,
        closure_file_sha256=closure_file_sha256,
        closure_payload_sha256=closure_payload_sha256,
    )
    if validated != closure:
        raise ValueError("shared Stage-A release validator returned a different closure")

    if tuple(closure.get("learned_configurations", ())) != tuple(FIREWALLED_LEARNED_CONFIGURATIONS):
        raise ValueError("Stage-A learned-configuration registry drifted")
    if tuple(closure.get("frozen_seeds", ())) != tuple(FROZEN_FINAL_SEEDS):
        raise ValueError("Stage-A seed registry drifted")
    if tuple(closure.get("frozen_outer_folds", ())) != tuple(FROZEN_OUTER_ROTATIONS):
        raise ValueError("Stage-A fold registry drifted")

    learned_records = closure.get("learned_bundles")
    if not isinstance(learned_records, list):
        raise ValueError("Stage-A closure lacks the frozen learned-bundle registry")
    expected_learned = {
        (configuration, seed)
        for configuration in FIREWALLED_LEARNED_CONFIGURATIONS
        for seed in FROZEN_FINAL_SEEDS
    }
    if (
        len(learned_records) != len(expected_learned)
        or {
            (record.get("configuration"), record.get("seed"))
            for record in learned_records
            if isinstance(record, Mapping)
        }
        != expected_learned
    ):
        raise ValueError("Stage-A closure must bind exactly 21 learned prediction bundles")

    diagnostic_records = closure.get("diagnostic_floors")
    if not isinstance(diagnostic_records, list):
        raise ValueError("Stage-A closure lacks diagnostic-floor bundles")
    expected_floors = set(DIAGNOSTIC_FLOOR_CONFIGURATIONS)
    if {
        record.get("configuration") for record in diagnostic_records if isinstance(record, Mapping)
    } != expected_floors:
        raise ValueError("Stage-A diagnostic-floor registry drifted")

    learned_rows: dict[tuple[str, int], tuple[Mapping[str, Any], ...]] = {}
    floor_rows: dict[str, tuple[Mapping[str, Any], ...]] = {}
    manifests: dict[tuple[str, int | None], Mapping[str, Any]] = {}
    for record in learned_records:
        configuration = str(record["configuration"])
        seed = int(record["seed"])
        manifest, rows = _load_closed_bundle(root, record, configuration, seed)
        learned_rows[(configuration, seed)] = rows
        manifests[(configuration, seed)] = manifest
    for record in diagnostic_records:
        configuration = str(record["configuration"])
        manifest, rows = _load_closed_bundle(root, record, configuration, None)
        floor_rows[configuration] = rows
        manifests[(configuration, None)] = manifest

    first_key = (FIREWALLED_LEARNED_CONFIGURATIONS[0], FROZEN_FINAL_SEEDS[0])
    participant_ids, folds, session_keys = _paired_roster(learned_rows[first_key])
    reference = _row_identity_and_target(learned_rows[first_key])
    for key, rows in (*learned_rows.items(), *floor_rows.items()):
        if _row_identity_and_target(rows) != reference:
            raise ValueError(f"Stage-A bundle {key!r} does not share the exact OOF target roster")

    return StageARelease(
        root=root,
        closure=closure,
        closure_file_sha256=closure_file_sha256,
        closure_payload_sha256=closure_payload_sha256,
        learned_rows=learned_rows,
        floor_rows=floor_rows,
        lock_manifests=manifests,
        participant_ids=participant_ids,
        participant_folds=folds,
        session_keys=session_keys,
    )


def load_annotation_release(
    *,
    release: StageARelease,
    codebook_path: str | Path | None = None,
    consensus_path: str | Path | None = None,
    annotation_audit_path: str | Path | None = None,
    rater_annotations_path: str | Path | None = None,
    stimulus_manifest_path: str | Path | None = None,
    source_release_manifest_path: str | Path | None = None,
    allow_synthetic_codebook: bool = False,
) -> AnnotationRelease:
    """Validate every annotation file before constructing the one-way join."""
    legacy_paths = (
        codebook_path,
        consensus_path,
        annotation_audit_path,
        rater_annotations_path,
        stimulus_manifest_path,
    )
    if source_release_manifest_path is not None:
        if any(path is not None for path in legacy_paths) or allow_synthetic_codebook:
            raise ValueError("source and legacy annotation release inputs are mutually exclusive")
        source = load_source_annotation_release(
            source_release_manifest_path,
            expected_participant_ids=release.participant_ids,
        )
        return AnnotationRelease(
            codebook=source.codebook,
            participant_annotations=source.post_annotations,
            assessability={
                participant_id: "assessable" for participant_id in release.participant_ids
            },
            audit_report=source.audit_report,
            bindings=source.bindings,
            release_method=SOURCE_RELEASE_METHOD,
            session_annotations=source.session_annotations,
            designated_stage_b_field=source.codebook.designated_stage_b_field,
            stage_b_designation=source.codebook.stage_b_designation,
        )
    if any(path is None for path in legacy_paths):
        raise ValueError("legacy annotation release requires all five annotation paths")
    codebook_file = _regular_file(codebook_path, "annotation codebook")
    consensus_file = _regular_file(consensus_path, "consensus annotations")
    audit_file = _regular_file(annotation_audit_path, "annotation audit")
    rater_file = _regular_file(rater_annotations_path, "rater annotations")
    stimulus_file = _regular_file(stimulus_manifest_path, "stimulus manifest")
    codebook = load_annotation_codebook(
        codebook_file, require_approved=not allow_synthetic_codebook
    )
    rows = load_consensus_annotations(consensus_file, codebook=codebook)
    expected_consensus = build_consensus_annotations(
        load_rater_annotations(rater_file, codebook=codebook)
    )
    if rows != expected_consensus:
        raise ValueError(
            "consensus annotations do not equal deterministic aggregation of bound rater rows"
        )
    audit = validate_annotation_audit_report(
        audit_file,
        codebook_path=codebook_file,
        rater_annotations_path=rater_file,
        stimulus_manifest_path=stimulus_file,
    )
    stimulus = load_stimulus_manifest(stimulus_file)
    assessability = {
        str(row["participant_id"]): str(row["status"]) for row in stimulus["participants"]
    }
    if not set(release.participant_ids).issubset(assessability):
        raise ValueError("annotation stimulus roster does not cover every eligible participant")
    consensus = {row.participant_id: row for row in rows}
    if len(consensus) != len(rows):
        raise ValueError("consensus annotations contain duplicate participants")
    for participant_id in release.participant_ids:
        status = assessability[participant_id]
        if status == "assessable" and participant_id not in consensus:
            raise ValueError("assessable eligible participant lacks consensus annotation")
        if status == "not_assessable" and participant_id in consensus:
            raise ValueError("not-assessable participant unexpectedly has consensus annotation")

    audit_bindings = audit["bindings"]
    if (
        audit_bindings["codebook_payload_sha256"] != codebook.sha256
        or audit_bindings["codebook_file_sha256"] != _file_sha256(codebook_file)
        or audit_bindings["raw_rater_file_sha256"] != _file_sha256(rater_file)
        or audit_bindings["stimulus_manifest_file_sha256"] != _file_sha256(stimulus_file)
    ):
        raise ValueError("annotation audit bindings drifted after validation")
    bindings = {
        "codebook_file_sha256": _file_sha256(codebook_file),
        "codebook_payload_sha256": codebook.sha256,
        "consensus_file_sha256": _file_sha256(consensus_file),
        "participant_annotation_release_file_sha256": _file_sha256(consensus_file),
        "annotation_audit_file_sha256": _file_sha256(audit_file),
        "annotation_audit_payload_sha256": str(audit["report_payload_sha256"]),
        "rater_annotations_file_sha256": _file_sha256(rater_file),
        "stimulus_manifest_file_sha256": _file_sha256(stimulus_file),
    }
    return AnnotationRelease(
        codebook=codebook,
        participant_annotations=consensus,
        assessability=assessability,
        audit_report=audit,
        bindings=bindings,
        release_method=CONSENSUS_METHOD,
    )


def _freeze_stage_a_release(release: StageARelease) -> StageARelease:
    if not isinstance(release, StageARelease):
        raise TypeError("Stage-C release must be a StageARelease")
    return StageARelease(
        root=release.root,
        closure=release.closure,
        closure_file_sha256=release.closure_file_sha256,
        closure_payload_sha256=release.closure_payload_sha256,
        learned_rows=release.learned_rows,
        floor_rows=release.floor_rows,
        lock_manifests=release.lock_manifests,
        participant_ids=release.participant_ids,
        participant_folds=release.participant_folds,
        session_keys=release.session_keys,
    )


def _freeze_annotation_release(annotations: AnnotationRelease) -> AnnotationRelease:
    if not isinstance(annotations, AnnotationRelease):
        raise TypeError("Stage-C annotations must be an AnnotationRelease")
    return AnnotationRelease(
        codebook=annotations.codebook,
        participant_annotations=annotations.participant_annotations,
        assessability=annotations.assessability,
        audit_report=annotations.audit_report,
        bindings=annotations.bindings,
        release_method=annotations.release_method,
        session_annotations=annotations.session_annotations,
        designated_stage_b_field=annotations.designated_stage_b_field,
        stage_b_designation=annotations.stage_b_designation,
    )


def _load_production_snapshot(
    paths: StageCSourcePaths, *, oracle_field: str
) -> _ProductionSnapshot:
    """Open and validate one complete production snapshot from the exact supplied paths."""
    evaluator_provenance = _evaluator_provenance(require_clean=True)
    with _capture_protected_sources(paths) as protected_sources:
        release = load_stage_a_release(paths.stage_a_dir)
        annotations = load_annotation_release(
            release=release,
            codebook_path=paths.codebook,
            consensus_path=paths.consensus,
            annotation_audit_path=paths.annotation_audit,
            rater_annotations_path=paths.rater_annotations,
            stimulus_manifest_path=paths.stimulus_manifest,
            source_release_manifest_path=paths.source_annotation_release,
            allow_synthetic_codebook=False,
        )
        _validate_captured_annotation_bindings(protected_sources, annotations)
        oracle_eligibility = freeze_json(
            validate_exact_binary_oracle_eligibility(release, annotations, field=oracle_field)
        )
        oracle_disposition = validate_stage_b_oracle_lock(
            paths.stage_b_oracle_disposition,
            release=release,
            annotations=annotations,
            eligibility=oracle_eligibility,
        )
        if (
            oracle_disposition.file_sha256
            != protected_sources["stage_b_oracle_disposition"].record.sha256
        ):
            raise RuntimeError("Stage-B disposition changed during Stage-C validation")
        modality_masks = freeze_json(
            _evaluate_modality_masks(
                release,
                _paired_arrays(release),
                paths.modality_mask_predictions,
            )
        )
        if (
            modality_masks.get("file_sha256")
            != protected_sources["modality_mask_predictions"].record.sha256
        ):
            raise RuntimeError("modality-mask source changed during Stage-C validation")
        fingerprint = {
            "source_paths": {
                name: (
                    None
                    if getattr(paths, name) is None
                    else str(Path(getattr(paths, name)).absolute())
                )
                for name in paths.__dataclass_fields__
            },
            "protected_source_files": {
                name: {
                    "sha256": capture.record.sha256,
                    "size": capture.record.size,
                    "mode": capture.record.mode,
                    "device": capture.record.device,
                    "inode": capture.record.inode,
                }
                for name, capture in protected_sources.items()
            },
            "stage_a": {
                "closure_file_sha256": release.closure_file_sha256,
                "closure_payload_sha256": release.closure_payload_sha256,
            },
            "annotations": dict(annotations.bindings),
            "oracle_eligibility_payload_sha256": _payload_sha256(oracle_eligibility),
            "stage_b": {
                "state": oracle_disposition.state,
                "file_sha256": oracle_disposition.file_sha256,
                "payload_sha256": oracle_disposition.payload_sha256,
            },
            "modality_masks_payload_sha256": _payload_sha256(modality_masks),
            "evaluator_provenance": evaluator_provenance,
        }
    return _ProductionSnapshot(
        release=release,
        annotations=annotations,
        oracle_eligibility=oracle_eligibility,
        oracle_disposition=oracle_disposition,
        modality_masks=modality_masks,
        fingerprint_bytes=canonical_json(fingerprint),
    )


_LEGACY_PROTECTED_STAGE_C_SOURCE_NAMES = (
    "codebook",
    "consensus",
    "annotation_audit",
    "rater_annotations",
    "stimulus_manifest",
    "stage_b_oracle_disposition",
    "modality_mask_predictions",
)
# Backward-compatible test/introspection alias for the unchanged legacy branch.
_PROTECTED_STAGE_C_SOURCE_NAMES = _LEGACY_PROTECTED_STAGE_C_SOURCE_NAMES


def _protected_stage_c_source_names(paths: StageCSourcePaths) -> tuple[str, ...]:
    if paths.source_annotation_release is not None:
        return (
            "source_annotation_release",
            "stage_b_oracle_disposition",
            "modality_mask_predictions",
        )
    return _LEGACY_PROTECTED_STAGE_C_SOURCE_NAMES


@contextmanager
def _capture_protected_sources(
    paths: StageCSourcePaths,
) -> Iterator[Mapping[str, _CapturedProtectedSource]]:
    """Hold and revalidate every protected source around all pathname parsers."""
    with ExitStack() as stack:
        captures: dict[str, _CapturedProtectedSource] = {}
        anchors: dict[str, AnchoredDirectory] = {}
        source_names = _protected_stage_c_source_names(paths)
        for name in source_names:
            source = Path(getattr(paths, name))
            anchor = stack.enter_context(
                AnchoredDirectory.open(
                    source.parent,
                    create=False,
                    directory_mode=0o700,
                )
            )
            record = anchor.read_bytes(source.name, expected_mode=0o600)
            anchors[name] = anchor
            captures[name] = _CapturedProtectedSource(path=source, record=record)
        frozen_captures = MappingProxyType(captures)
        yield frozen_captures
        for name in source_names:
            capture = frozen_captures[name]
            anchor = anchors[name]
            anchor.verify_path_identity()
            reopened = anchor.read_bytes(capture.path.name, expected_mode=0o600)
            if reopened != capture.record:
                raise RuntimeError(f"protected Stage-C source {name} changed during validation")
            anchor.verify_path_identity()


def _validate_captured_annotation_bindings(
    captures: Mapping[str, _CapturedProtectedSource],
    annotations: AnnotationRelease,
) -> None:
    if "source_annotation_release" in captures:
        if (
            annotations.bindings.get("manifest_file_sha256")
            != captures["source_annotation_release"].record.sha256
        ):
            raise RuntimeError("protected Stage-C source release manifest changed during parsing")
        return
    expected = {
        "codebook": "codebook_file_sha256",
        "consensus": "consensus_file_sha256",
        "annotation_audit": "annotation_audit_file_sha256",
        "rater_annotations": "rater_annotations_file_sha256",
        "stimulus_manifest": "stimulus_manifest_file_sha256",
    }
    for source_name, binding_name in expected.items():
        if annotations.bindings.get(binding_name) != captures[source_name].record.sha256:
            raise RuntimeError(
                f"protected Stage-C source {source_name} changed during annotation parsing"
            )


def _revalidate_production_result(result: StageCEvaluationResult, *, oracle_field: str) -> None:
    """Reopen every production source and require the exact original snapshot."""
    state = _stage_c_result_state(result)
    if state.source_paths is None or state.source_fingerprint_bytes is None:
        raise ValueError("research Stage-C result lacks exact source paths")
    current = _load_production_snapshot(state.source_paths, oracle_field=oracle_field)
    if current.fingerprint_bytes != state.source_fingerprint_bytes:
        raise RuntimeError("Stage-C source evidence changed after validation")
    if (
        current.release.closure_file_sha256 != state.release.closure_file_sha256
        or current.release.closure_payload_sha256 != state.release.closure_payload_sha256
        or current.annotations.bindings != state.annotations.bindings
    ):
        raise RuntimeError("Stage-C frozen evidence differs from reopened sources")


def validate_exact_binary_oracle_eligibility(
    release: StageARelease,
    annotations: AnnotationRelease,
    *,
    field: str,
) -> dict[str, Any]:
    """Authorize all 15 oracle fits only for an exactly binary released analysis axis."""
    categories = _codebook_categories(
        annotations.codebook, field, release_method=annotations.release_method
    )
    rules = {
        "released_analysis_category_count_required": 2,
        "known_training_annotation_required": True,
        "minimum_training_participants_per_category": MINIMUM_REPORTABLE_PARTICIPANTS,
        "category_merge": "forbidden",
        "category_drop": "forbidden",
        "post_hoc_binarisation": "forbidden",
        "designated_source_release_field_required": annotations.designated_stage_b_field,
        "authorization": "all_15_or_none",
    }
    reasons: list[str] = []
    if (
        annotations.designated_stage_b_field is not None
        and field != annotations.designated_stage_b_field
    ):
        reasons.append("field_not_designated_for_stage_b")
    if len(categories) != 2 or len(set(categories)) != 2:
        reasons.append("released_analysis_axis_is_not_exactly_binary")
    rotations: list[dict[str, Any]] = []
    for rotation in FROZEN_OUTER_ROTATIONS:
        training_ids = tuple(
            participant_id
            for participant_id in release.participant_ids
            if release.participant_folds[participant_id]
            not in {rotation, (rotation + 1) % len(FROZEN_OUTER_ROTATIONS)}
        )
        values = [
            _annotation_value(annotations.participant_annotations.get(participant_id), field)
            for participant_id in training_ids
        ]
        unknown_count = sum(value is None for value in values)
        outside_codebook_count = sum(
            value is not None and value not in categories for value in values
        )
        counts = Counter(value for value in values if value is not None)
        rotation_reasons: list[str] = []
        if unknown_count:
            rotation_reasons.append("unknown_training_annotation")
        if outside_codebook_count:
            rotation_reasons.append("training_annotation_outside_codebook")
        if (
            len(categories) == 2
            and len(set(categories)) == 2
            and any(counts[category] < MINIMUM_REPORTABLE_PARTICIPANTS for category in categories)
        ):
            rotation_reasons.append("training_category_below_10")
        rotations.append(
            {
                "outer_rotation": rotation,
                "training_participant_count": len(training_ids),
                "known_training_participant_count": len(training_ids) - unknown_count,
                "unknown_training_participant_count": unknown_count,
                "outside_codebook_training_participant_count": outside_codebook_count,
                "category_counts": [
                    {
                        "category": category,
                        "count": counts[category],
                        "reportable": counts[category] >= MINIMUM_REPORTABLE_PARTICIPANTS,
                    }
                    for category in categories
                ],
                "eligible": (
                    not rotation_reasons and len(categories) == 2 and len(set(categories)) == 2
                ),
                "reasons": rotation_reasons,
            }
        )
        reasons.extend(rotation_reasons)
    eligible = len(categories) == 2 and len(set(categories)) == 2 and not reasons
    return {
        "field": field,
        "codebook_version": annotations.codebook.codebook_version,
        "codebook_categories": list(categories),
        "release_method": annotations.release_method,
        "stage_b_designation": annotations.stage_b_designation,
        "designated_stage_b_field": annotations.designated_stage_b_field,
        "rules": rules,
        "rotations": rotations,
        "status": "eligible_all_15" if eligible else "omitted_all_15",
        "authorized_fit_count": 15 if eligible else 0,
        "omitted_fit_count": 0 if eligible else 15,
        "reasons": sorted(set(reasons)),
    }


def _evaluate_stage_c_payloads(
    release: StageARelease,
    annotations: AnnotationRelease,
    *,
    oracle_field: str,
    stage_b_oracle_path: str | Path | None = None,
    modality_mask_path: str | Path | None = None,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    synthetic_contract_test: bool = False,
    validated_oracle_disposition: StageBOracleDisposition | None = None,
    validated_modality_masks: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the restricted row join and a non-identifying aggregate audit."""
    if isinstance(bootstrap_replicates, bool) or not isinstance(bootstrap_replicates, int):
        raise TypeError("bootstrap_replicates must be an integer")
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    if bootstrap_replicates != BOOTSTRAP_REPLICATES and not synthetic_contract_test:
        raise ValueError("research Stage C requires exactly 10,000 bootstrap replicates")
    accepted_codebook = (
        annotations.codebook.status == "approved_for_rating"
        and annotations.release_method == CONSENSUS_METHOD
    ) or (
        annotations.codebook.status == "retrospectively_documented_from_embedded_source_legend"
        and annotations.release_method == SOURCE_RELEASE_METHOD
    )
    if not synthetic_contract_test and not accepted_codebook:
        raise ValueError("research Stage C requires a validated annotation release codebook")
    if not synthetic_contract_test and (
        release.closure.get("state") != "closed"
        or release.closure.get("execution_mode") != "research"
        or release.closure.get("firewall_stage") != "A"
        or release.closure.get("protected_annotation_fields_accessed") != []
    ):
        raise ValueError("research Stage C requires a validated closed Stage-A release")
    if (
        not synthetic_contract_test
        and modality_mask_path is None
        and validated_modality_masks is None
    ):
        raise ValueError(
            "research Stage C requires the locked inference-only modality-mask predictions"
        )

    arrays = _paired_arrays(release)
    groups, group_specs = _annotation_arrays(release, annotations)
    source_release = annotations.release_method == SOURCE_RELEASE_METHOD
    oracle = validate_exact_binary_oracle_eligibility(release, annotations, field=oracle_field)
    oracle_disposition = validated_oracle_disposition
    if oracle_disposition is None:
        oracle_disposition = _stage_b_oracle_disposition(
            path=stage_b_oracle_path,
            release=release,
            annotations=annotations,
            eligibility=oracle,
            synthetic_contract_test=synthetic_contract_test,
        )
    if oracle_disposition is not None and oracle_disposition.state == "locked":
        _attach_oracle_predictions(arrays, release, oracle_disposition)
    contrast_specs = _active_contrasts(oracle_disposition)
    proxy_rows, proxy_diagnostics = _load_proxy_diagnostics(release, annotations, groups)
    masks = validated_modality_masks
    if masks is None:
        masks = (
            {"status": "not_supplied", "reason": "locked_mask_predictions_unavailable"}
            if modality_mask_path is None
            else _evaluate_modality_masks(release, arrays, modality_mask_path)
        )

    restricted_rows = _restricted_join_rows(
        release,
        annotations,
        proxy_rows=proxy_rows,
        oracle_disposition=oracle_disposition,
    )
    restricted_unsigned = {
        "schema_version": STAGE_C_RESTRICTED_JOIN_SCHEMA_VERSION,
        "classification": "restricted_participant_level_do_not_publish",
        "evaluator_version": STAGE_C_EVALUATOR_VERSION,
        "stage_a_closure_file_sha256": release.closure_file_sha256,
        "stage_a_closure_payload_sha256": release.closure_payload_sha256,
        "annotation_bindings": dict(annotations.bindings),
        "stage_b_oracle": _oracle_public_summary(oracle_disposition, oracle),
        "participant_count": release.participant_count,
        "rows": restricted_rows,
    }
    if source_release:
        _assert_no_source_consensus_terms(restricted_unsigned)
    restricted_join = _signed_payload(restricted_unsigned, "join_payload_sha256")

    utility = _utility_report(arrays)
    phase = _phase_report(arrays)
    paired_change = _paired_change_report(arrays)
    disparities = _disparity_report(
        arrays,
        groups,
        group_specs,
        fairness_phase="post" if source_release else None,
    )
    residuals = _group_error_report(
        arrays,
        groups,
        group_specs,
        fairness_phase="post" if source_release else None,
    )
    pre_sensitivity = _pre_session_sensitivity_report(
        arrays,
        release,
        annotations,
        group_specs,
        contrast_specs=contrast_specs,
        replicates=bootstrap_replicates,
        research_evidence=(
            not synthetic_contract_test and bootstrap_replicates == BOOTSTRAP_REPLICATES
        ),
    )
    contrasts = _contrast_report(utility, disparities, contrast_specs)
    intervals = _bootstrap_report(
        arrays,
        groups,
        group_specs,
        contrast_specs=contrast_specs,
        replicates=bootstrap_replicates,
        research_evidence=(
            not synthetic_contract_test and bootstrap_replicates == BOOTSTRAP_REPLICATES
        ),
        fairness_phase="post" if source_release else None,
    )
    aggregate_unsigned: dict[str, Any] = {
        "schema_version": STAGE_C_AGGREGATE_SCHEMA_VERSION,
        "scope": "aggregate_only_no_participant_session_or_rater_identifiers",
        "evaluator_version": STAGE_C_EVALUATOR_VERSION,
        "evaluator_provenance": _evaluator_provenance(require_clean=not synthetic_contract_test),
        "execution_mode": ("synthetic_contract_test" if synthetic_contract_test else "research"),
        "stage_a": {
            "closure_file_sha256": release.closure_file_sha256,
            "closure_payload_sha256": release.closure_payload_sha256,
            "eligible_participant_count": release.participant_count,
            "eligible_session_count": 2 * release.participant_count,
            "learned_bundle_count": len(release.learned_rows),
            "diagnostic_floor_bundle_count": len(release.floor_rows),
        },
        "annotations": {
            "bindings": dict(annotations.bindings),
            "codebook_version": annotations.codebook.codebook_version,
            "eligible_assessable_count": sum(
                annotations.assessability[participant_id] == "assessable"
                for participant_id in release.participant_ids
            ),
            "eligible_not_assessable_count": sum(
                annotations.assessability[participant_id] == "not_assessable"
                for participant_id in release.participant_ids
            ),
            "measurement_audit": _annotation_measurement_summary(
                annotations, synthetic_contract_test=synthetic_contract_test
            ),
            "release_method": annotations.release_method,
            "stage_b_designation": annotations.stage_b_designation,
        },
        "utility": utility,
        "three_seed_sensitivity_summary": _seed_sensitivity_report(utility),
        "phase_stratified_utility": phase,
        "paired_change": paired_change,
        "audited_disparity": disparities,
        "group_residual_and_mae": residuals,
        "pre_session_sensitivity": pre_sensitivity,
        "contrasts": contrasts,
        "participant_cluster_bootstrap": intervals,
        "proxy_diagnostics": proxy_diagnostics,
        "modality_masks": masks,
        "exact_binary_oracle_eligibility": oracle,
        "stage_b_oracle": _oracle_public_summary(oracle_disposition, oracle),
        "interpretation_contract": {
            "paired_change": "descriptive_noncausal",
            "proxy_audit_association": "association_not_identity",
            "fixed_seeds": "sensitivity_runs_not_an_ensemble_or_inferential_sample",
            "automatic_superiority_claims": "forbidden",
            "source_annotation_phase_alignment": (
                "post_primary_pre_available_case_sensitivity_no_label_carry_forward"
                if source_release
                else "participant_consensus_shared_across_sessions"
            ),
            "ordinary_model_rerun_after_annotation_join": "forbidden",
        },
    }
    if source_release:
        _assert_no_source_consensus_terms(aggregate_unsigned)
    _assert_public_aggregate(aggregate_unsigned)
    aggregate = _signed_payload(aggregate_unsigned, "aggregate_payload_sha256")
    return restricted_join, aggregate


def _evaluator_provenance(*, require_clean: bool = False) -> dict[str, Any]:
    """Bind the exact evaluator sources and numerical runtime used by this process."""
    current = _current_evaluator_provenance()
    if current != _IMPORTED_EVALUATOR_PROVENANCE:
        raise RuntimeError("Stage-C evaluator sources changed after module import")
    if require_clean and current["working_tree_dirty"]:
        raise ValueError("research Stage C requires clean committed evaluator sources")
    return {
        **current,
        "component_source_sha256": dict(current["component_source_sha256"]),
        "runtime_versions": dict(current["runtime_versions"]),
    }


def evaluate_stage_c(
    release: StageARelease,
    annotations: AnnotationRelease,
    *,
    oracle_field: str,
    stage_b_oracle_path: str | Path | None = None,
    modality_mask_path: str | Path | None = None,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    synthetic_contract_test: bool = False,
) -> StageCEvaluationResult:
    """Evaluate isolated synthetic fixtures only; this API cannot create research evidence."""
    if synthetic_contract_test is not True:
        raise ValueError(
            "in-memory Stage-C evaluation is synthetic-only; research requires exact source paths"
        )
    frozen_release = _freeze_stage_a_release(release)
    frozen_annotations = _freeze_annotation_release(annotations)
    restricted_join, aggregate = _evaluate_stage_c_payloads(
        frozen_release,
        frozen_annotations,
        oracle_field=oracle_field,
        stage_b_oracle_path=stage_b_oracle_path,
        modality_mask_path=modality_mask_path,
        bootstrap_replicates=bootstrap_replicates,
        synthetic_contract_test=True,
    )
    return _make_stage_c_result(
        restricted_join,
        aggregate,
        release=frozen_release,
        annotations=frozen_annotations,
        source_paths=None,
        source_fingerprint_bytes=None,
    )


def evaluate_stage_c_from_paths(
    *,
    source_paths: StageCSourcePaths,
    oracle_field: str,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> StageCEvaluationResult:
    """Build research evidence from exact paths, then reopen every source before return."""
    if not isinstance(source_paths, StageCSourcePaths):
        raise TypeError("production Stage C requires StageCSourcePaths")
    if bootstrap_replicates != BOOTSTRAP_REPLICATES:
        raise ValueError("research Stage C requires exactly 10,000 bootstrap replicates")
    snapshot = _load_production_snapshot(source_paths, oracle_field=oracle_field)
    restricted_join, aggregate = _evaluate_stage_c_payloads(
        snapshot.release,
        snapshot.annotations,
        oracle_field=oracle_field,
        bootstrap_replicates=bootstrap_replicates,
        synthetic_contract_test=False,
        validated_oracle_disposition=snapshot.oracle_disposition,
        validated_modality_masks=snapshot.modality_masks,
    )
    result = _make_stage_c_result(
        restricted_join,
        aggregate,
        release=snapshot.release,
        annotations=snapshot.annotations,
        source_paths=source_paths,
        source_fingerprint_bytes=snapshot.fingerprint_bytes,
    )
    _revalidate_production_result(result, oracle_field=oracle_field)
    return result


def _stage_b_oracle_disposition(
    *,
    path: str | Path | None,
    release: StageARelease,
    annotations: AnnotationRelease,
    eligibility: Mapping[str, Any],
    synthetic_contract_test: bool,
) -> StageBOracleDisposition | None:
    if path is None:
        if synthetic_contract_test:
            return None
        raise ValueError("research Stage C requires the immutable Stage-B oracle lock or omission")
    return validate_stage_b_oracle_lock(
        path,
        release=release,
        annotations=annotations,
        eligibility=eligibility,
    )


def _attach_oracle_predictions(
    arrays: dict[str, Any],
    release: StageARelease,
    disposition: StageBOracleDisposition,
) -> None:
    configuration = str(disposition.payload["configuration"])
    participant_index = {
        participant_id: index for index, participant_id in enumerate(release.participant_ids)
    }
    phase_index = {"pre": 0, "post": 1}
    if tuple(disposition.prediction_rows) != tuple(FROZEN_FINAL_SEEDS):
        raise ValueError("Stage-B oracle prediction registry is not in frozen seed order")
    for seed in FROZEN_FINAL_SEEDS:
        matrix = np.empty_like(arrays["target"])
        seen: set[tuple[str, str]] = set()
        for row in disposition.prediction_rows[seed]:
            participant_id = str(row["participant_id"])
            phase = str(row["session_phase"])
            key = (participant_id, phase)
            if key in seen:
                raise ValueError("Stage-B oracle predictions duplicate a participant phase")
            matrix[participant_index[participant_id], phase_index[phase]] = float(row["prediction"])
            seen.add(key)
        if len(seen) != 2 * release.participant_count:
            raise ValueError("Stage-B oracle predictions do not cover the paired roster")
        arrays["predictions"][(configuration, seed)] = matrix


def _active_contrasts(
    disposition: StageBOracleDisposition | None,
) -> tuple[dict[str, str], ...]:
    if disposition is None or disposition.state != "locked":
        return CONTRASTS
    return (
        *CONTRASTS,
        {
            "contrast_id": "protected_oracle_w1_total",
            "intervention": str(disposition.payload["configuration"]),
            "reference": "erm",
            "family": "conditional_exact_binary_oracle",
        },
    )


def _oracle_public_summary(
    disposition: StageBOracleDisposition | None,
    eligibility: Mapping[str, Any],
) -> dict[str, Any]:
    if disposition is not None:
        return disposition.public_summary
    return {
        "state": "not_supplied_synthetic_contract_test",
        "configuration": f"oracle_{eligibility['field']}",
        "oracle_field": eligibility["field"],
        "authorized_fit_count": 0,
        "completed_fit_count": 0,
        "completed_bundle_count": 0,
        "expected_oof_rows_per_seed": 0,
        "completed_oof_rows_per_seed": 0,
        "omission_reasons": ["synthetic_contract_test_did_not_supply_stage_b"],
    }


def build_claim_ledger(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    """Bind predeclared claim families to one immutable aggregate, without auto-claims."""
    _validate_signed_object(aggregate, "aggregate_payload_sha256", "Stage-C aggregate")
    if aggregate.get("schema_version") != STAGE_C_AGGREGATE_SCHEMA_VERSION:
        raise ValueError("claim ledger requires a Stage-C aggregate")
    oracle_pointers = [
        "/exact_binary_oracle_eligibility",
        "/stage_b_oracle",
    ]
    if aggregate["stage_b_oracle"]["state"] == "locked":
        oracle_pointers.extend(["/contrasts", "/participant_cluster_bootstrap/contrasts"])
    rows = [
        {
            "claim_id": "C-UTILITY",
            "scope": "seed_specific_oof_utility",
            "status": "evidence_computed",
            "aggregate_json_pointers": [
                "/utility",
                "/three_seed_sensitivity_summary",
                "/participant_cluster_bootstrap/utility",
            ],
            "automatic_conclusion": "forbidden",
        },
        {
            "claim_id": "C-PRIMARY-CONTRASTS",
            "scope": "two_total_effect_models_x_srcc_and_audited_disparity",
            "status": "effect_sizes_and_intervals_computed",
            "aggregate_json_pointers": [
                "/contrasts",
                "/participant_cluster_bootstrap/contrasts",
            ],
            "multiplicity_policy": (
                "fixed_estimand_family;pointwise_intervals;no_p_values;"
                "no_automatic_significance_claims;interpret_jointly_with_interval_width"
            ),
            "automatic_conclusion": "forbidden",
        },
        {
            "claim_id": "C-PAIRED-CHANGE",
            "scope": "descriptive_post_minus_pre_tracking",
            "status": "evidence_computed",
            "aggregate_json_pointers": [
                "/paired_change",
                "/participant_cluster_bootstrap/paired_change",
            ],
            "causal_language": "forbidden",
        },
        {
            "claim_id": "C-ORACLE",
            "scope": "exact_binary_only_conditional_oracle",
            "status": aggregate["stage_b_oracle"]["state"],
            "aggregate_json_pointers": oracle_pointers,
            "automatic_conclusion": "forbidden",
        },
        {
            "claim_id": "C-MODALITY-MASKS",
            "scope": "locked_erm_inference_only_modality_masks",
            "status": aggregate["modality_masks"]["status"],
            "aggregate_json_pointers": ["/modality_masks"],
            "automatic_conclusion": "forbidden",
        },
    ]
    unsigned = {
        "schema_version": STAGE_C_CLAIM_LEDGER_SCHEMA_VERSION,
        "aggregate_payload_sha256": aggregate["aggregate_payload_sha256"],
        "claim_rows": rows,
        "claim_language_review_required": True,
    }
    return _signed_payload(unsigned, "claim_ledger_payload_sha256")


def build_no_rerun_manifest(
    *,
    release: StageARelease,
    annotations: AnnotationRelease,
    restricted_join: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    claim_ledger: Mapping[str, Any],
    restricted_join_file_sha256: str,
    aggregate_file_sha256: str,
    claim_ledger_file_sha256: str,
) -> dict[str, Any]:
    """Close Stage C and make the no-feedback rule machine-readable."""
    _validate_signed_object(restricted_join, "join_payload_sha256", "restricted join")
    _validate_signed_object(aggregate, "aggregate_payload_sha256", "aggregate")
    _validate_signed_object(claim_ledger, "claim_ledger_payload_sha256", "claim ledger")
    unsigned = {
        "schema_version": STAGE_C_NO_RERUN_SCHEMA_VERSION,
        "state": "closed",
        "firewall_transition": "A_locked_then_conditional_B_then_single_C_join",
        "evaluator_version": STAGE_C_EVALUATOR_VERSION,
        "stage_a_closure_file_sha256": release.closure_file_sha256,
        "stage_a_closure_payload_sha256": release.closure_payload_sha256,
        "annotation_bindings": dict(annotations.bindings),
        "stage_b_oracle": dict(aggregate["stage_b_oracle"]),
        "restricted_join": {
            "file_sha256": _required_sha256(
                restricted_join_file_sha256, "restricted join file checksum"
            ),
            "payload_sha256": restricted_join["join_payload_sha256"],
            "classification": "restricted_not_publishable",
        },
        "aggregate": {
            "file_sha256": _required_sha256(aggregate_file_sha256, "aggregate file checksum"),
            "payload_sha256": aggregate["aggregate_payload_sha256"],
        },
        "claim_ledger": {
            "file_sha256": _required_sha256(claim_ledger_file_sha256, "claim ledger file checksum"),
            "payload_sha256": claim_ledger["claim_ledger_payload_sha256"],
        },
        "ordinary_model_rerun_after_audit": "forbidden",
        "checkpoint_or_proxy_selection_after_audit": "forbidden",
        "new_model_seed_fold_or_sweep_after_audit": "forbidden",
        "permitted_follow_up": "aggregate_reporting_and_thesis_writing_only",
    }
    return _signed_payload(unsigned, "no_rerun_payload_sha256")


def _validate_claim_ledger_exact(payload: Mapping[str, Any], aggregate: Mapping[str, Any]) -> None:
    _validate_signed_object(payload, "claim_ledger_payload_sha256", "claim ledger")
    if (
        set(payload) != _CLAIM_LEDGER_FIELDS
        or payload.get("schema_version") != STAGE_C_CLAIM_LEDGER_SCHEMA_VERSION
    ):
        raise ValueError("claim ledger schema drifted")
    _assert_public_aggregate(payload)
    if payload != build_claim_ledger(aggregate):
        raise ValueError("claim ledger is not the exact deterministic aggregate projection")


def _validate_no_rerun_exact(
    payload: Mapping[str, Any],
    *,
    release: StageARelease,
    annotations: AnnotationRelease,
    restricted_join: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    claim_ledger: Mapping[str, Any],
    file_hashes: Mapping[str, str],
) -> None:
    _validate_signed_object(payload, "no_rerun_payload_sha256", "no-rerun manifest")
    if (
        set(payload) != _NO_RERUN_FIELDS
        or payload.get("schema_version") != STAGE_C_NO_RERUN_SCHEMA_VERSION
        or payload.get("state") != "closed"
    ):
        raise ValueError("no-rerun manifest schema or state drifted")
    _assert_public_aggregate(payload)
    expected = build_no_rerun_manifest(
        release=release,
        annotations=annotations,
        restricted_join=restricted_join,
        aggregate=aggregate,
        claim_ledger=claim_ledger,
        restricted_join_file_sha256=file_hashes["restricted_join"],
        aggregate_file_sha256=file_hashes["aggregate"],
        claim_ledger_file_sha256=file_hashes["claim_ledger"],
    )
    if payload != expected:
        raise ValueError("no-rerun manifest is not the exact final Stage-C commit")


def write_stage_c_outputs(
    *,
    restricted_join_path: str | Path,
    aggregate_path: str | Path,
    claim_ledger_path: str | Path,
    no_rerun_path: str | Path,
    result: StageCEvaluationResult,
) -> dict[str, Path]:
    """Publish one opaque result; the no-rerun manifest is the final commit marker."""
    state = _stage_c_result_state(result)
    restricted_join = _canonical_object_from_bytes(
        state.restricted_join_bytes, "Stage-C restricted result"
    )
    aggregate = _canonical_object_from_bytes(state.aggregate_bytes, "Stage-C aggregate result")
    release = state.release
    annotations = state.annotations
    _validate_stage_c_payload_pair(
        restricted_join,
        aggregate,
        release=release,
        annotations=annotations,
    )
    research = state.source_paths is not None
    if research != (aggregate["execution_mode"] == "research"):
        raise ValueError("Stage-C result mode and source binding disagree")
    oracle_field = str(aggregate["exact_binary_oracle_eligibility"]["field"])
    if research:
        _revalidate_production_result(result, oracle_field=oracle_field)

    paths = {
        "restricted_join": _lexical_output_path(restricted_join_path, "restricted join"),
        "aggregate": _lexical_output_path(aggregate_path, "aggregate"),
        "claim_ledger": _lexical_output_path(claim_ledger_path, "claim ledger"),
        "no_rerun": _lexical_output_path(no_rerun_path, "no-rerun manifest"),
    }
    if len(set(paths.values())) != len(paths):
        raise ValueError("Stage-C output files must be distinct")
    if any(release.root == path or release.root in path.parents for path in paths.values()):
        raise ValueError("Stage-C outputs may not be written inside immutable Stage-A output")

    claim = build_claim_ledger(aggregate)
    _validate_claim_ledger_exact(claim, aggregate)
    payloads = {
        "restricted_join": canonical_json(restricted_join),
        "aggregate": canonical_json(aggregate),
        "claim_ledger": canonical_json(claim),
    }
    modes = {
        "restricted_join": 0o600,
        "aggregate": 0o644,
        "claim_ledger": 0o644,
        "no_rerun": 0o644,
    }

    with ExitStack() as stack:
        anchors: dict[Path, AnchoredDirectory] = {}
        restricted_parent = paths["restricted_join"].parent
        for parent in dict.fromkeys(path.parent for path in paths.values()):
            anchors[parent] = stack.enter_context(
                AnchoredDirectory.open(
                    parent,
                    create=True,
                    directory_mode=0o700 if parent == restricted_parent else 0o755,
                )
            )
        restricted_anchor = anchors[paths["restricted_join"].parent]
        public_anchors = {
            anchors[paths[name].parent] for name in ("aggregate", "claim_ledger", "no_rerun")
        }
        if any(
            (restricted_anchor.device, restricted_anchor.inode) == (anchor.device, anchor.inode)
            for anchor in public_anchors
        ):
            raise ValueError("restricted join must be stored separately from aggregate outputs")
        for anchor in anchors.values():
            anchor.verify_path_identity()
        for name, path in paths.items():
            if path.name in anchors[path.parent].list_names():
                raise FileExistsError(f"refusing existing Stage-C {name} output")

        written = {}
        for name in ("restricted_join", "aggregate", "claim_ledger"):
            path = paths[name]
            written[name] = anchors[path.parent].publish_bytes(
                path.name,
                payloads[name],
                mode=modes[name],
                allow_existing=False,
            )
            if written[name].content != payloads[name]:
                raise RuntimeError(f"Stage-C {name} post-write bytes drifted")
        reopened_restricted = _canonical_object_from_bytes(
            written["restricted_join"].content, "published restricted join"
        )
        reopened_aggregate = _canonical_object_from_bytes(
            written["aggregate"].content, "published aggregate"
        )
        reopened_claim = _canonical_object_from_bytes(
            written["claim_ledger"].content, "published claim ledger"
        )
        _validate_stage_c_payload_pair(
            reopened_restricted,
            reopened_aggregate,
            release=release,
            annotations=annotations,
        )
        _validate_claim_ledger_exact(reopened_claim, reopened_aggregate)
        if research:
            _revalidate_production_result(result, oracle_field=oracle_field)

        no_rerun = build_no_rerun_manifest(
            release=release,
            annotations=annotations,
            restricted_join=reopened_restricted,
            aggregate=reopened_aggregate,
            claim_ledger=reopened_claim,
            restricted_join_file_sha256=written["restricted_join"].sha256,
            aggregate_file_sha256=written["aggregate"].sha256,
            claim_ledger_file_sha256=written["claim_ledger"].sha256,
        )
        _validate_no_rerun_exact(
            no_rerun,
            release=release,
            annotations=annotations,
            restricted_join=reopened_restricted,
            aggregate=reopened_aggregate,
            claim_ledger=reopened_claim,
            file_hashes={name: written[name].sha256 for name in written},
        )
        no_rerun_bytes = canonical_json(no_rerun)
        no_rerun_path_value = paths["no_rerun"]
        for anchor in anchors.values():
            anchor.verify_path_identity()
        if research:
            _revalidate_production_result(result, oracle_field=oracle_field)
        anchors[no_rerun_path_value.parent].publish_bytes(
            no_rerun_path_value.name,
            no_rerun_bytes,
            mode=modes["no_rerun"],
            allow_existing=False,
        )
    return paths


def _load_closed_bundle(
    root: Path,
    record: Mapping[str, Any],
    configuration: str,
    seed: int | None,
) -> tuple[dict[str, Any], tuple[Mapping[str, Any], ...]]:
    if not isinstance(record, Mapping):
        raise ValueError("Stage-A bundle record is invalid")
    lock_path = _bound_child(root, record.get("prediction_lock_filename"), "prediction lock")
    prediction_path = _bound_child(root, record.get("prediction_filename"), "predictions")
    if _file_sha256(lock_path) != record.get("prediction_lock_file_sha256"):
        raise ValueError("closed prediction-lock file checksum failed")
    if _file_sha256(prediction_path) != record.get("prediction_file_sha256"):
        raise ValueError("closed prediction file checksum failed")
    manifest = _read_canonical_object(lock_path, "prediction lock")
    _validate_signed_object(manifest, "manifest_payload_sha256", "prediction lock")
    if (
        manifest.get("schema_version") != PREDICTION_LOCK_SCHEMA_VERSION
        or manifest.get("configuration") != configuration
        or manifest.get("seed") != seed
        or manifest.get("execution_mode") != "research"
        or manifest.get("state") != "locked"
        or manifest.get("firewall_stage") != "A"
        or manifest.get("protected_annotation_fields_accessed") != []
        or manifest.get("manifest_payload_sha256") != record.get("prediction_lock_payload_sha256")
    ):
        raise ValueError("closed prediction-lock identity or firewall drifted")
    if record.get("prediction_lock_id") != manifest.get("prediction_lock_id"):
        raise ValueError("closure and prediction-lock IDs disagree")
    prediction_record = manifest.get("predictions")
    target_summary = manifest.get("target")
    if (
        not isinstance(prediction_record, Mapping)
        or prediction_record.get("filename") != prediction_path.name
        or prediction_record.get("sha256") != record.get("prediction_file_sha256")
        or prediction_record.get("row_count") != record.get("oof_row_count")
        or not isinstance(target_summary, Mapping)
        or target_summary.get("construct") != "crowd_perceived_hireability_not_actual_hiring"
        or target_summary.get("transform") != TARGET_TRANSFORM
        or not isinstance(target_summary.get("source"), Mapping)
        or target_summary["source"].get("field") != "RecommendHiring"
    ):
        raise ValueError("prediction or target-definition record drifted from its lock")
    payload = _read_canonical_object(prediction_path, "OOF predictions")
    rows = payload.get("rows")
    if (
        payload.get("schema_version") != PREDICTION_SCHEMA_VERSION
        or payload.get("prediction_lock_id") != manifest.get("prediction_lock_id")
        or not isinstance(rows, list)
        or len(rows) != record.get("oof_row_count")
    ):
        raise ValueError("OOF prediction payload drifted from its lock")
    seen_sessions: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != PREDICTION_ROW_FIELDS:
            raise ValueError("OOF prediction row schema drifted")
        prediction = row.get("prediction")
        target = row.get("target")
        if (
            row.get("configuration") != configuration
            or row.get("seed") != seed
            or row.get("corpus") != "mit_interview"
            or row.get("rotation_role") != "test"
            or row.get("fixed_fold") != row.get("outer_rotation")
            or row.get("firewall_stage") != "A"
            or row.get("target_transform") != TARGET_TRANSFORM
            or row.get("target_manifest_sha256") != manifest.get("target_manifest_sha256")
            or row.get("eligibility_manifest_sha256") != manifest.get("eligibility_manifest_sha256")
            or row.get("feature_contract_sha256") != manifest.get("feature_contract_sha256")
            or row.get("preprocessing_sha256") != manifest.get("preprocessing_sha256")
            or row.get("session_mapping") != manifest.get("session_mapping")
            or row.get("prediction_lock_id") != manifest.get("prediction_lock_id")
            or not isinstance(row.get("model_fit_sha256"), str)
            or _HEX64.fullmatch(row["model_fit_sha256"]) is None
            or row.get("session_id") in seen_sessions
            or isinstance(prediction, bool)
            or not isinstance(prediction, (int, float))
            or not math.isfinite(float(prediction))
            or isinstance(target, bool)
            or not isinstance(target, (int, float))
            or not math.isfinite(float(target))
            or not 0 <= float(target) <= 1
            or (seed is not None and not 0 <= float(prediction) <= 1)
        ):
            raise ValueError("OOF prediction row identity, value or lock linkage drifted")
        seen_sessions.add(str(row["session_id"]))
    return manifest, tuple(rows)


def _paired_roster(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, ...], dict[str, int], tuple[tuple[str, str, str], ...]]:
    by_participant: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        participant_id = row.get("participant_id")
        if not isinstance(participant_id, str) or _PARTICIPANT_ID.fullmatch(participant_id) is None:
            raise ValueError("OOF participant ID must be canonical decimal text")
        by_participant.setdefault(participant_id, []).append(row)
    if not by_participant:
        raise ValueError("OOF participant roster is empty")
    folds: dict[str, int] = {}
    session_keys: list[tuple[str, str, str]] = []
    for participant_id in sorted(by_participant, key=int):
        pair = by_participant[participant_id]
        phases = {row.get("session_phase") for row in pair}
        fold_values = {row.get("fixed_fold") for row in pair}
        if len(pair) != 2 or phases != {"pre", "post"} or len(fold_values) != 1:
            raise ValueError("each eligible participant must have one pre/post OOF pair")
        fold = next(iter(fold_values))
        if isinstance(fold, bool) or fold not in FROZEN_OUTER_ROTATIONS:
            raise ValueError("OOF participant fold is invalid")
        folds[participant_id] = int(fold)
        for row in sorted(pair, key=lambda item: item["session_phase"] == "post"):
            session_keys.append(
                (
                    participant_id,
                    str(row["session_phase"]),
                    str(row["session_id"]),
                )
            )
    return tuple(sorted(by_participant, key=int)), folds, tuple(session_keys)


def _row_identity_and_target(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, str, str, str, int, float], ...]:
    return tuple(
        sorted(
            (
                str(row["participant_id"]),
                str(row["session_phase"]),
                str(row["session_id"]),
                str(row["source_session_id"]),
                int(row["fixed_fold"]),
                float(row["target"]),
            )
            for row in rows
        )
    )


def _paired_arrays(release: StageARelease) -> dict[str, Any]:
    participant_index = {
        participant_id: index for index, participant_id in enumerate(release.participant_ids)
    }
    phase_index = {"pre": 0, "post": 1}
    target = np.empty((release.participant_count, 2), dtype=np.float64)
    predictions: dict[tuple[str, int | None], np.ndarray] = {}
    all_rows = {
        **release.learned_rows,
        **{(name, None): rows for name, rows in release.floor_rows.items()},
    }
    for key, rows in all_rows.items():
        matrix = np.empty_like(target)
        for row in rows:
            location = (
                participant_index[str(row["participant_id"])],
                phase_index[str(row["session_phase"])],
            )
            matrix[location] = float(row["prediction"])
            target[location] = float(row["target"])
        predictions[key] = matrix
    return {
        "participant_ids": release.participant_ids,
        "target": target,
        "predictions": predictions,
    }


def _annotation_arrays(
    release: StageARelease,
    annotations: AnnotationRelease,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[Any, ...]]]:
    specs = {
        field: _codebook_categories(
            annotations.codebook,
            field,
            release_method=annotations.release_method,
        )
        for field in PERCEIVED_FIELDS
    }
    groups: dict[str, np.ndarray] = {}
    for field in PERCEIVED_FIELDS:
        groups[field] = np.asarray(
            [
                _annotation_value(
                    annotations.participant_annotations.get(participant_id), field
                )
                for participant_id in release.participant_ids
            ],
            dtype=object,
        )
    return groups, specs


def _annotation_measurement_summary(
    annotations: AnnotationRelease, *, synthetic_contract_test: bool
) -> dict[str, Any]:
    audit = annotations.audit_report
    if annotations.release_method == SOURCE_RELEASE_METHOD:
        required = (
            "coverage",
            "post_primary_category_counts",
            "pre_sensitivity_category_counts",
            "paired_stability",
            "measurement_limitations",
            "analysis_contract",
        )
        if not isinstance(audit, Mapping) or any(field not in audit for field in required):
            raise ValueError("source release lacks its validated measurement audit")
        return {
            "status": "reported_single_rater_source_release",
            "coverage": audit["coverage"],
            "post_primary_category_counts": audit["post_primary_category_counts"],
            "pre_sensitivity_category_counts": audit["pre_sensitivity_category_counts"],
            "paired_stability": audit["paired_stability"],
            "measurement_limitations": audit["measurement_limitations"],
            "analysis_contract": audit["analysis_contract"],
        }
    required = ("coverage", "stimulus_quality", "fields", "consensus_method")
    if not isinstance(audit, Mapping) or any(field not in audit for field in required):
        if synthetic_contract_test:
            return {"status": "synthetic_fixture_not_supplied"}
        raise ValueError("research Stage C requires the validated annotation measurement audit")
    return {
        "status": "reported",
        "coverage": audit["coverage"],
        "stimulus_quality": audit["stimulus_quality"],
        "fields": audit["fields"],
        "consensus_method": audit["consensus_method"],
    }


def _pre_session_sensitivity_report(
    arrays: Mapping[str, Any],
    release: StageARelease,
    annotations: AnnotationRelease,
    specs: Mapping[str, tuple[Any, ...]],
    *,
    contrast_specs: Sequence[Mapping[str, str]],
    replicates: int,
    research_evidence: bool,
) -> dict[str, Any]:
    if annotations.release_method != SOURCE_RELEASE_METHOD:
        return {"status": "not_applicable_participant_consensus_shared_across_sessions"}
    groups = {
        field: np.asarray(
            [
                _annotation_value(
                    annotations.session_annotations.get((participant_id, "pre")),
                    field,
                )
                for participant_id in release.participant_ids
            ],
            dtype=object,
        )
        for field in PERCEIVED_FIELDS
    }
    participant_ids = release.participant_ids
    available_indices = np.asarray(
        [
            index
            for index, participant_id in enumerate(participant_ids)
            if (participant_id, "pre") in annotations.session_annotations
        ],
        dtype=np.int64,
    )
    available = sum(
        (participant_id, "pre") in annotations.session_annotations
        for participant_id in participant_ids
    )
    if available != EXPECTED_SOURCE_PRE_COUNT:
        raise ValueError("source pre-session sensitivity coverage drifted")
    available_arrays = {
        "target": arrays["target"][available_indices],
        "predictions": {
            key: values[available_indices] for key, values in arrays["predictions"].items()
        },
    }
    available_groups = {field: values[available_indices] for field, values in groups.items()}
    pre_utility = _phase_matched_utility_report(available_arrays, phase="pre")
    pre_disparity = _disparity_report(
        available_arrays,
        available_groups,
        specs,
        fairness_phase="pre",
    )
    return {
        "status": "reported_descriptive_available_case_sensitivity",
        "coverage": {
            "eligible_participant_count": len(participant_ids),
            "available_pre_annotation_count": available,
            "missing_pre_annotation_count": len(participant_ids) - available,
            "missingness_interpretation": "incomplete_missingness_not_assumed_random",
            "post_rating_carry_forward": "forbidden",
        },
        "phase_matched_utility": pre_utility,
        "audited_disparity": pre_disparity,
        "group_residual_and_mae": _group_error_report(
            available_arrays,
            available_groups,
            specs,
            fairness_phase="pre",
        ),
        "seed_matched_point_contrasts": _contrast_report(
            pre_utility,
            pre_disparity,
            contrast_specs,
        ),
        "participant_cluster_bootstrap": _pre_session_bootstrap_report(
            available_arrays,
            available_groups,
            specs,
            contrast_specs=contrast_specs,
            replicates=replicates,
            research_evidence=research_evidence,
        ),
        "interpretation": "sensitivity_only_not_confirmatory",
    }


def _phase_matched_utility_report(
    arrays: Mapping[str, Any], *, phase: str
) -> list[dict[str, Any]]:
    """Report utility on only the rows admitted to one phase-aligned audit."""
    if phase not in {"pre", "post"}:
        raise ValueError("phase-matched utility requires pre or post")
    column = 0 if phase == "pre" else 1
    target = np.asarray(arrays["target"], dtype=np.float64)[:, column]
    rows = [
        {
            "configuration": configuration,
            "seed": seed,
            "phase": phase,
            "participant_count": int(target.size),
            "metrics": utility_metrics(target, np.asarray(prediction)[:, column]),
        }
        for (configuration, seed), prediction in arrays["predictions"].items()
    ]
    return sorted(rows, key=_configuration_seed_sort)


def _pre_session_bootstrap_report(
    arrays: Mapping[str, Any],
    groups: Mapping[str, np.ndarray],
    specs: Mapping[str, tuple[Any, ...]],
    *,
    contrast_specs: Sequence[Mapping[str, str]],
    replicates: int,
    research_evidence: bool,
) -> dict[str, Any]:
    """Independently resample the participants with matching pre ratings and predictions."""
    target = np.asarray(arrays["target"], dtype=np.float64)[:, 0]
    predictions = {
        key: np.asarray(values, dtype=np.float64)[:, 0]
        for key, values in arrays["predictions"].items()
        if key[1] is not None
    }
    if target.shape != (EXPECTED_SOURCE_PRE_COUNT,) or any(
        values.shape != target.shape for values in predictions.values()
    ):
        raise ValueError("pre-session bootstrap requires exactly 57 aligned participants")
    reportable_categories = {
        field: _originally_reportable_categories(groups[field], specs[field])
        for field in PERCEIVED_FIELDS
    }
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0,
        EXPECTED_SOURCE_PRE_COUNT,
        size=(replicates, EXPECTED_SOURCE_PRE_COUNT),
    )
    target_disparity_samples = {
        field: np.full(replicates, np.nan, dtype=np.float64) for field in PERCEIVED_FIELDS
    }
    disparity_samples = {
        (configuration, seed, field): np.full(replicates, np.nan, dtype=np.float64)
        for configuration, seed in predictions
        for field in PERCEIVED_FIELDS
    }
    group_error_samples = {
        (configuration, seed, field, category, metric): np.full(
            replicates, np.nan, dtype=np.float64
        )
        for configuration, seed in predictions
        for field in PERCEIVED_FIELDS
        for category in reportable_categories[field]
        for metric in ("mean_residual", "mae")
    }
    contrast_samples: dict[tuple[str, int | str, str], np.ndarray] = {}
    contrast_metrics = (
        "srcc_change",
        "pearson_change",
        "mae_improvement",
        "rmse_improvement",
        *(f"disparity_gain:{field}" for field in PERCEIVED_FIELDS),
    )
    for contrast in contrast_specs:
        for seed in (*FROZEN_FINAL_SEEDS, "three_seed_mean"):
            for metric in contrast_metrics:
                contrast_samples[(contrast["contrast_id"], seed, metric)] = np.full(
                    replicates, np.nan, dtype=np.float64
                )

    for replicate_index, draw in enumerate(draws):
        sampled_target = target[draw]
        sampled_groups = {field: values[draw] for field, values in groups.items()}
        for field in PERCEIVED_FIELDS:
            target_disparity_samples[field][replicate_index] = _nan_if_none(
                _bootstrap_disparity_value(
                    sampled_target,
                    sampled_groups[field],
                    reportable_categories[field],
                    sessions_per_participant=1,
                )
            )
        replicate_utility: dict[tuple[str, int], dict[str, Any]] = {}
        replicate_disparity: dict[tuple[str, int, str], float | None] = {}
        for (configuration, seed), prediction in predictions.items():
            sampled_prediction = prediction[draw]
            replicate_utility[(configuration, seed)] = utility_metrics(
                sampled_target, sampled_prediction
            )
            errors = sampled_prediction - sampled_target
            for field in PERCEIVED_FIELDS:
                value = _bootstrap_disparity_value(
                    sampled_prediction,
                    sampled_groups[field],
                    reportable_categories[field],
                    sessions_per_participant=1,
                )
                replicate_disparity[(configuration, seed, field)] = value
                disparity_samples[(configuration, seed, field)][replicate_index] = _nan_if_none(
                    value
                )
                for category in reportable_categories[field]:
                    selected = sampled_groups[field] == category
                    if not np.any(selected):
                        continue
                    values = errors[selected]
                    group_error_samples[
                        (configuration, seed, field, category, "mean_residual")
                    ][replicate_index] = float(np.mean(values))
                    group_error_samples[(configuration, seed, field, category, "mae")][
                        replicate_index
                    ] = float(np.mean(np.abs(values)))

        for contrast in contrast_specs:
            per_seed = {metric: [] for metric in contrast_metrics}
            for seed in FROZEN_FINAL_SEEDS:
                intervention = replicate_utility[(contrast["intervention"], seed)]
                reference = replicate_utility[(contrast["reference"], seed)]
                delta = _utility_delta(intervention, reference)
                values: dict[str, float | None] = {
                    "srcc_change": delta["srcc_change"],
                    "pearson_change": delta["pearson_change"],
                    "mae_improvement": delta["mae_improvement"]["normalized"],
                    "rmse_improvement": delta["rmse_improvement"]["normalized"],
                }
                for field in PERCEIVED_FIELDS:
                    values[f"disparity_gain:{field}"] = _optional_subtract(
                        replicate_disparity[(contrast["reference"], seed, field)],
                        replicate_disparity[(contrast["intervention"], seed, field)],
                    )
                for metric, value in values.items():
                    contrast_samples[(contrast["contrast_id"], seed, metric)][
                        replicate_index
                    ] = _nan_if_none(value)
                    per_seed[metric].append(value)
            for metric, values in per_seed.items():
                complete = [float(value) for value in values if value is not None]
                mean = (
                    float(np.mean(complete))
                    if len(complete) == len(FROZEN_FINAL_SEEDS)
                    else None
                )
                contrast_samples[(contrast["contrast_id"], "three_seed_mean", metric)][
                    replicate_index
                ] = _nan_if_none(mean)

    return {
        "method": "participant_cluster_percentile",
        "resamples": replicates,
        "random_state": BOOTSTRAP_SEED,
        "confidence_level": CONFIDENCE_LEVEL,
        "percentile_quantile_method": "linear",
        "interval_scope": "pointwise_descriptive_not_simultaneous",
        "cluster_unit": "participant_with_matching_pre_prediction_and_pre_annotation",
        "available_case_participant_count": EXPECTED_SOURCE_PRE_COUNT,
        "fairness_annotation_phase": "pre",
        "post_rating_carry_forward": "forbidden",
        "audit_replicate_policy": (
            "originally_reportable_categories_fixed;absent_required_group_invalid;no_imputation"
        ),
        "seeds_are_sensitivity_runs_not_resampling_units": True,
        "research_evidence": research_evidence,
        "observed_target_disparity": _interval_rows(
            {(field,): values for field, values in target_disparity_samples.items()},
            original_units="all",
        ),
        "audited_disparity": _interval_rows(disparity_samples, original_units="all"),
        "group_residual_and_mae": _interval_rows(group_error_samples, original_units="all"),
        "contrasts": _interval_rows(contrast_samples, original_units="errors"),
    }


def _utility_report(arrays: Mapping[str, Any]) -> list[dict[str, Any]]:
    target = arrays["target"].reshape(-1)
    rows: list[dict[str, Any]] = []
    for (configuration, seed), prediction in arrays["predictions"].items():
        rows.append(
            {
                "configuration": configuration,
                "seed": seed,
                "session_count": int(target.size),
                "metrics": utility_metrics(target, prediction.reshape(-1)),
            }
        )
    return sorted(rows, key=_configuration_seed_sort)


def _phase_report(arrays: Mapping[str, Any]) -> list[dict[str, Any]]:
    target = arrays["target"]
    result: list[dict[str, Any]] = []
    for (configuration, seed), prediction in arrays["predictions"].items():
        for phase, column in (("pre", 0), ("post", 1)):
            result.append(
                {
                    "configuration": configuration,
                    "seed": seed,
                    "phase": phase,
                    "participant_count": int(target.shape[0]),
                    "metrics": utility_metrics(target[:, column], prediction[:, column]),
                }
            )
    return sorted(result, key=lambda row: (*_configuration_seed_sort(row), row["phase"]))


def _seed_sensitivity_report(
    utility: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Describe the three fixed seeds without treating them as an ensemble or sample."""
    index = {
        (row["configuration"], row["seed"]): row["metrics"]
        for row in utility
        if row["seed"] is not None
    }
    result = []
    configurations = list(FIREWALLED_LEARNED_CONFIGURATIONS)
    configurations.extend(
        sorted(
            {
                str(configuration)
                for configuration, seed in index
                if seed in FROZEN_FINAL_SEEDS
                and configuration not in FIREWALLED_LEARNED_CONFIGURATIONS
            }
        )
    )
    for configuration in configurations:
        rows = [index[(configuration, seed)] for seed in FROZEN_FINAL_SEEDS]
        metrics: dict[str, Any] = {}
        for metric in ("srcc", "pearson"):
            metrics[metric] = _mean_and_sample_sd([row[metric] for row in rows])
        for metric in ("mae", "rmse"):
            summary = _mean_and_sample_sd([row[metric]["normalized"] for row in rows])
            metrics[metric] = (
                None
                if summary is None
                else {
                    "normalized": summary,
                    "original_1_7": {
                        "mean": RAW_SCALE_WIDTH * summary["mean"],
                        "sample_sd": RAW_SCALE_WIDTH * summary["sample_sd"],
                    },
                }
            )
        result.append(
            {
                "configuration": configuration,
                "fixed_seed_count": len(FROZEN_FINAL_SEEDS),
                "interpretation": "descriptive_sensitivity_not_ensemble_or_inference",
                "metrics": metrics,
            }
        )
    return result


def _paired_change_report(arrays: Mapping[str, Any]) -> list[dict[str, Any]]:
    observed = arrays["target"][:, 1] - arrays["target"][:, 0]
    result = []
    for (configuration, seed), prediction in arrays["predictions"].items():
        predicted = prediction[:, 1] - prediction[:, 0]
        result.append(
            {
                "configuration": configuration,
                "seed": seed,
                "participant_count": int(observed.size),
                "descriptive_noncausal": True,
                "metrics": paired_change_metrics(observed, predicted),
            }
        )
    return sorted(result, key=_configuration_seed_sort)


def utility_metrics(target: Sequence[float], prediction: Sequence[float]) -> dict[str, Any]:
    """Return the four frozen session-level utility statistics."""
    target_array, prediction_array = _validated_pair(target, prediction)
    error = prediction_array - target_array
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    return {
        "srcc": _correlation(target_array, prediction_array, rank=True),
        "pearson": _correlation(target_array, prediction_array, rank=False),
        "mae": {"normalized": mae, "original_1_7": RAW_SCALE_WIDTH * mae},
        "rmse": {"normalized": rmse, "original_1_7": RAW_SCALE_WIDTH * rmse},
    }


def paired_change_metrics(
    observed_change: Sequence[float], predicted_change: Sequence[float]
) -> dict[str, Any]:
    observed, predicted = _validated_pair(observed_change, predicted_change)
    error = predicted - observed
    mae = float(np.mean(np.abs(error)))
    bias = float(np.mean(error))
    return {
        "delta_srcc": _correlation(observed, predicted, rank=True),
        "delta_mae": {"normalized": mae, "original_1_7": RAW_SCALE_WIDTH * mae},
        "signed_delta_bias": {
            "normalized": bias,
            "original_1_7": RAW_SCALE_WIDTH * bias,
        },
    }


def maximum_pairwise_disparity(
    scores: Sequence[float],
    participant_groups: Sequence[Any],
    categories: Sequence[Any],
    *,
    sessions_per_participant: int,
) -> dict[str, Any]:
    """Compute max-pairwise empirical W1 without merging sparse categories."""
    score = np.asarray(scores, dtype=np.float64)
    group = np.asarray(participant_groups, dtype=object)
    categories = tuple(categories)
    if score.ndim != 1 or not score.size or not np.isfinite(score).all():
        raise ValueError("disparity scores must be a non-empty finite vector")
    if group.ndim != 1 or group.size * sessions_per_participant != score.size:
        raise ValueError("participant groups do not align with repeated score rows")
    if sessions_per_participant <= 0:
        raise ValueError("sessions_per_participant must be positive")
    if not categories or len(set(categories)) != len(categories):
        raise ValueError("disparity categories must be non-empty and unique")
    unexpected = {
        value for value in group.tolist() if value is not None and value not in categories
    }
    if unexpected:
        raise ValueError("participant group is absent from the frozen codebook categories")
    expanded = np.repeat(group, sessions_per_participant)
    unknown_count = int(sum(value is None for value in group.tolist()))
    category_rows: list[dict[str, Any]] = []
    reportable: list[Any] = []
    for category in categories:
        participant_count = int(sum(value == category for value in group.tolist()))
        is_reportable = participant_count >= MINIMUM_REPORTABLE_PARTICIPANTS
        row: dict[str, Any] = {
            "category": category,
            "participant_count": participant_count,
            "session_count": sessions_per_participant * participant_count,
            "reportable": is_reportable,
            "suppression_reason": None if is_reportable else "fewer_than_10_participants",
        }
        if is_reportable:
            values = score[expanded == category]
            row["score_distribution"] = {
                "mean": _score_units(float(np.mean(values))),
                "median": _score_units(float(np.median(values))),
                "q25": _score_units(float(np.quantile(values, 0.25))),
                "q75": _score_units(float(np.quantile(values, 0.75))),
            }
            reportable.append(category)
        category_rows.append(row)
    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(reportable):
        for right in reportable[left_index + 1 :]:
            value = empirical_wasserstein1(score[expanded == left], score[expanded == right])
            pairs.append({"category_a": left, "category_b": right, "wasserstein1": _units(value)})
    if not pairs:
        maximum = None
        status = "suppressed_fewer_than_two_reportable_categories"
    else:
        maximum_value = max(float(row["wasserstein1"]["normalized"]) for row in pairs)
        attaining = next(
            row for row in pairs if float(row["wasserstein1"]["normalized"]) == maximum_value
        )
        maximum = dict(attaining)
        status = "reported"
    binary_special_case = None
    if len(categories) == 2 and len(reportable) == 2 and maximum is not None:
        left, right = categories
        binary_special_case = {
            "name": "spdd",
            "explicitly_binary_codebook_axis": True,
            "direction_mean_b_minus_a": _units(
                float(np.mean(score[expanded == right]) - np.mean(score[expanded == left]))
            ),
            "wasserstein1": maximum["wasserstein1"],
        }
    return {
        "status": status,
        "minimum_reportable_participants": MINIMUM_REPORTABLE_PARTICIPANTS,
        "attaining_pair_tie_break": "first_pair_in_frozen_codebook_order",
        "unknown_participant_count": unknown_count,
        "unknown_session_count": sessions_per_participant * unknown_count,
        "categories": category_rows,
        "pairwise": pairs,
        "maximum_pairwise": maximum,
        "binary_special_case": binary_special_case,
    }


def _disparity_report(
    arrays: Mapping[str, Any],
    groups: Mapping[str, np.ndarray],
    specs: Mapping[str, tuple[Any, ...]],
    *,
    fairness_phase: str | None = None,
) -> dict[str, Any]:
    target, sessions_per_participant = _fairness_score_vector(arrays["target"], fairness_phase)
    observed = [
        {
            "field": field,
            "disparity": maximum_pairwise_disparity(
                target,
                groups[field],
                specs[field],
                sessions_per_participant=sessions_per_participant,
            ),
        }
        for field in PERCEIVED_FIELDS
    ]
    predictions: list[dict[str, Any]] = []
    for (configuration, seed), prediction in arrays["predictions"].items():
        for field in PERCEIVED_FIELDS:
            predictions.append(
                {
                    "configuration": configuration,
                    "seed": seed,
                    "field": field,
                    "disparity": maximum_pairwise_disparity(
                        _fairness_score_vector(prediction, fairness_phase)[0],
                        groups[field],
                        specs[field],
                        sessions_per_participant=sessions_per_participant,
                    ),
                }
            )
    predictions.sort(key=lambda row: (*_configuration_seed_sort(row), row["field"]))
    return {
        "analysis_phase": fairness_phase or "both_sessions_shared_participant_annotation",
        "observed_target": observed,
        "predictions": predictions,
    }


def _group_error_report(
    arrays: Mapping[str, Any],
    groups: Mapping[str, np.ndarray],
    specs: Mapping[str, tuple[Any, ...]],
    *,
    fairness_phase: str | None = None,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for (configuration, seed), prediction in arrays["predictions"].items():
        prediction_values, sessions_per_participant = _fairness_score_vector(
            prediction, fairness_phase
        )
        target_values, _ = _fairness_score_vector(arrays["target"], fairness_phase)
        error = prediction_values - target_values
        for field in PERCEIVED_FIELDS:
            expanded = np.repeat(groups[field], sessions_per_participant)
            categories: list[dict[str, Any]] = []
            for category in specs[field]:
                participant_count = int(sum(value == category for value in groups[field].tolist()))
                if participant_count < MINIMUM_REPORTABLE_PARTICIPANTS:
                    categories.append(
                        {
                            "category": category,
                            "participant_count": participant_count,
                            "status": "suppressed_fewer_than_10_participants",
                        }
                    )
                    continue
                values = error[expanded == category]
                categories.append(
                    {
                        "category": category,
                        "participant_count": participant_count,
                        "status": "reported",
                        "mean_residual": _units(float(np.mean(values))),
                        "mae": _units(float(np.mean(np.abs(values)))),
                    }
                )
            result.append(
                {
                    "configuration": configuration,
                    "seed": seed,
                    "field": field,
                    "analysis_phase": (
                        fairness_phase or "both_sessions_shared_participant_annotation"
                    ),
                    "residual_definition": "prediction_minus_target",
                    "unknown_participant_count": int(
                        sum(value is None for value in groups[field].tolist())
                    ),
                    "categories": categories,
                }
            )
    return sorted(result, key=lambda row: (*_configuration_seed_sort(row), row["field"]))


def _fairness_score_vector(
    values: np.ndarray, fairness_phase: str | None
) -> tuple[np.ndarray, int]:
    if fairness_phase is None:
        return values.reshape(-1), 2
    if fairness_phase not in {"pre", "post"}:
        raise ValueError("fairness phase must be pre, post or shared participant annotation")
    column = 0 if fairness_phase == "pre" else 1
    return values[:, column], 1


def _contrast_report(
    utility: Sequence[Mapping[str, Any]],
    disparities: Mapping[str, Any],
    contrast_specs: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    utility_index = {
        (row["configuration"], row["seed"]): row["metrics"]
        for row in utility
        if row["seed"] is not None
    }
    disparity_index = {
        (row["configuration"], row["seed"], row["field"]): row["disparity"]
        for row in disparities["predictions"]
        if row["seed"] is not None
    }
    result: list[dict[str, Any]] = []
    for contrast in contrast_specs:
        seed_rows: list[dict[str, Any]] = []
        for seed in FROZEN_FINAL_SEEDS:
            intervention = utility_index[(contrast["intervention"], seed)]
            reference = utility_index[(contrast["reference"], seed)]
            disparity_rows = []
            for field in PERCEIVED_FIELDS:
                inter_disparity = disparity_index[(contrast["intervention"], seed, field)]
                ref_disparity = disparity_index[(contrast["reference"], seed, field)]
                disparity_rows.append(
                    {
                        "field": field,
                        "gain_reference_minus_intervention": _difference_disparity(
                            ref_disparity, inter_disparity
                        ),
                    }
                )
            seed_rows.append(
                {
                    "seed": seed,
                    "utility": _utility_delta(intervention, reference),
                    "audited_disparity": disparity_rows,
                }
            )
        result.append(
            {
                **contrast,
                "sign_contract": {
                    "correlation_change": "intervention_minus_reference",
                    "error_improvement": "reference_minus_intervention",
                    "disparity_gain": "reference_minus_intervention",
                    "positive_is_improvement": True,
                },
                "seed_specific": seed_rows,
                "three_seed_arithmetic_mean": _mean_seed_contrast(seed_rows),
            }
        )
    return result


def _bootstrap_report(
    arrays: Mapping[str, Any],
    groups: Mapping[str, np.ndarray],
    specs: Mapping[str, tuple[Any, ...]],
    *,
    contrast_specs: Sequence[Mapping[str, str]],
    replicates: int,
    research_evidence: bool,
    fairness_phase: str | None = None,
) -> dict[str, Any]:
    """Recompute metrics after participant resampling; never resample summaries."""
    target = arrays["target"]
    learned = {key: value for key, value in arrays["predictions"].items() if key[1] is not None}
    reportable_categories = {
        field: _originally_reportable_categories(groups[field], specs[field])
        for field in PERCEIVED_FIELDS
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, target.shape[0], size=(replicates, target.shape[0]))

    utility_samples: dict[tuple[str, int, str], np.ndarray] = {
        (configuration, seed, metric): np.full(replicates, np.nan, dtype=np.float64)
        for configuration, seed in learned
        for metric in ("srcc", "pearson", "mae", "rmse")
    }
    change_samples: dict[tuple[str, int, str], np.ndarray] = {
        (configuration, seed, metric): np.full(replicates, np.nan, dtype=np.float64)
        for configuration, seed in learned
        for metric in ("delta_srcc", "delta_mae", "signed_delta_bias")
    }
    disparity_samples: dict[tuple[str, int, str], np.ndarray] = {
        (configuration, seed, field): np.full(replicates, np.nan, dtype=np.float64)
        for configuration, seed in learned
        for field in PERCEIVED_FIELDS
    }
    target_disparity_samples = {
        field: np.full(replicates, np.nan, dtype=np.float64) for field in PERCEIVED_FIELDS
    }
    phase_samples: dict[tuple[str, int, str, str], np.ndarray] = {
        (configuration, seed, phase, metric): np.full(replicates, np.nan, dtype=np.float64)
        for configuration, seed in learned
        for phase in ("pre", "post")
        for metric in ("srcc", "pearson", "mae", "rmse")
    }
    group_error_samples: dict[tuple[str, int, str, Any, str], np.ndarray] = {
        (configuration, seed, field, category, metric): np.full(
            replicates, np.nan, dtype=np.float64
        )
        for configuration, seed in learned
        for field in PERCEIVED_FIELDS
        for category in reportable_categories[field]
        for metric in ("mean_residual", "mae")
    }
    contrast_samples: dict[tuple[str, int | str, str], np.ndarray] = {}
    for contrast in contrast_specs:
        for seed in (*FROZEN_FINAL_SEEDS, "three_seed_mean"):
            for metric in (
                "srcc_change",
                "pearson_change",
                "mae_improvement",
                "rmse_improvement",
            ):
                contrast_samples[(contrast["contrast_id"], seed, metric)] = np.full(
                    replicates, np.nan, dtype=np.float64
                )
            for field in PERCEIVED_FIELDS:
                contrast_samples[(contrast["contrast_id"], seed, f"disparity_gain:{field}")] = (
                    np.full(replicates, np.nan, dtype=np.float64)
                )

    for replicate_index, draw in enumerate(draws):
        sampled_target = target[draw]
        sampled_groups = {field: values[draw] for field, values in groups.items()}
        for field in PERCEIVED_FIELDS:
            target_disparity_samples[field][replicate_index] = _nan_if_none(
                _bootstrap_disparity_value(
                    _fairness_score_vector(sampled_target, fairness_phase)[0],
                    sampled_groups[field],
                    reportable_categories[field],
                    sessions_per_participant=(1 if fairness_phase is not None else 2),
                )
            )
        replicate_utility: dict[tuple[str, int], dict[str, Any]] = {}
        replicate_disparity: dict[tuple[str, int, str], float | None] = {}
        for (configuration, seed), prediction in learned.items():
            sampled_prediction = prediction[draw]
            metrics = utility_metrics(sampled_target.reshape(-1), sampled_prediction.reshape(-1))
            replicate_utility[(configuration, seed)] = metrics
            for metric in ("srcc", "pearson"):
                utility_samples[(configuration, seed, metric)][replicate_index] = _nan_if_none(
                    metrics[metric]
                )
            for metric in ("mae", "rmse"):
                utility_samples[(configuration, seed, metric)][replicate_index] = metrics[metric][
                    "normalized"
                ]
            for phase, column in (("pre", 0), ("post", 1)):
                phase_metrics = utility_metrics(
                    sampled_target[:, column], sampled_prediction[:, column]
                )
                for metric in ("srcc", "pearson"):
                    phase_samples[(configuration, seed, phase, metric)][replicate_index] = (
                        _nan_if_none(phase_metrics[metric])
                    )
                for metric in ("mae", "rmse"):
                    phase_samples[(configuration, seed, phase, metric)][replicate_index] = (
                        phase_metrics[metric]["normalized"]
                    )
            change = paired_change_metrics(
                sampled_target[:, 1] - sampled_target[:, 0],
                sampled_prediction[:, 1] - sampled_prediction[:, 0],
            )
            change_samples[(configuration, seed, "delta_srcc")][replicate_index] = _nan_if_none(
                change["delta_srcc"]
            )
            change_samples[(configuration, seed, "delta_mae")][replicate_index] = change[
                "delta_mae"
            ]["normalized"]
            change_samples[(configuration, seed, "signed_delta_bias")][replicate_index] = change[
                "signed_delta_bias"
            ]["normalized"]
            for field in PERCEIVED_FIELDS:
                value = _bootstrap_disparity_value(
                    _fairness_score_vector(sampled_prediction, fairness_phase)[0],
                    sampled_groups[field],
                    reportable_categories[field],
                    sessions_per_participant=(1 if fairness_phase is not None else 2),
                )
                replicate_disparity[(configuration, seed, field)] = value
                disparity_samples[(configuration, seed, field)][replicate_index] = _nan_if_none(
                    value
                )
                for category in reportable_categories[field]:
                    selected = sampled_groups[field] == category
                    if not np.any(selected):
                        continue
                    if fairness_phase is None:
                        errors = (sampled_prediction[selected] - sampled_target[selected]).reshape(
                            -1
                        )
                    else:
                        column = 0 if fairness_phase == "pre" else 1
                        errors = (
                            sampled_prediction[selected, column] - sampled_target[selected, column]
                        )
                    group_error_samples[(configuration, seed, field, category, "mean_residual")][
                        replicate_index
                    ] = float(np.mean(errors))
                    group_error_samples[(configuration, seed, field, category, "mae")][
                        replicate_index
                    ] = float(np.mean(np.abs(errors)))

        for contrast in contrast_specs:
            per_seed: dict[str, list[float | None]] = {
                metric: []
                for metric in (
                    "srcc_change",
                    "pearson_change",
                    "mae_improvement",
                    "rmse_improvement",
                    *(f"disparity_gain:{field}" for field in PERCEIVED_FIELDS),
                )
            }
            for seed in FROZEN_FINAL_SEEDS:
                intervention = replicate_utility[(contrast["intervention"], seed)]
                reference = replicate_utility[(contrast["reference"], seed)]
                delta = _utility_delta(intervention, reference)
                scalar_delta = {
                    "srcc_change": delta["srcc_change"],
                    "pearson_change": delta["pearson_change"],
                    "mae_improvement": delta["mae_improvement"]["normalized"],
                    "rmse_improvement": delta["rmse_improvement"]["normalized"],
                }
                for metric, value in scalar_delta.items():
                    contrast_samples[(contrast["contrast_id"], seed, metric)][replicate_index] = (
                        _nan_if_none(value)
                    )
                    per_seed[metric].append(value)
                for field in PERCEIVED_FIELDS:
                    reference_value = replicate_disparity[(contrast["reference"], seed, field)]
                    intervention_value = replicate_disparity[
                        (contrast["intervention"], seed, field)
                    ]
                    value = _optional_subtract(reference_value, intervention_value)
                    key = f"disparity_gain:{field}"
                    contrast_samples[(contrast["contrast_id"], seed, key)][replicate_index] = (
                        _nan_if_none(value)
                    )
                    per_seed[key].append(value)
            for metric, values in per_seed.items():
                valid = [float(value) for value in values if value is not None]
                mean = float(np.mean(valid)) if len(valid) == len(FROZEN_FINAL_SEEDS) else None
                contrast_samples[(contrast["contrast_id"], "three_seed_mean", metric)][
                    replicate_index
                ] = _nan_if_none(mean)

    return {
        "method": "participant_cluster_percentile",
        "resamples": replicates,
        "random_state": BOOTSTRAP_SEED,
        "confidence_level": CONFIDENCE_LEVEL,
        "percentile_quantile_method": "linear",
        "interval_scope": "pointwise_descriptive_not_simultaneous",
        "multiplicity_policy": (
            "no_p_values;no_automatic_significance_claims;interpret_fixed_estimand_family_jointly"
        ),
        "cluster_unit": "participant_with_both_sessions_and_all_seed_model_predictions",
        "fairness_annotation_phase": (
            fairness_phase or "both_sessions_shared_participant_annotation"
        ),
        "audit_replicate_policy": (
            "originally_reportable_categories_fixed;absent_required_group_invalid;no_imputation"
        ),
        "seeds_are_sensitivity_runs_not_resampling_units": True,
        "research_evidence": research_evidence,
        "utility": _interval_rows(utility_samples, original_units="errors"),
        "phase_stratified_utility": _interval_rows(phase_samples, original_units="errors"),
        "paired_change": _interval_rows(change_samples, original_units="errors"),
        "observed_target_disparity": _interval_rows(
            {(field,): values for field, values in target_disparity_samples.items()},
            original_units="all",
        ),
        "audited_disparity": _interval_rows(disparity_samples, original_units="all"),
        "group_residual_and_mae": _interval_rows(group_error_samples, original_units="all"),
        "contrasts": _interval_rows(contrast_samples, original_units="errors"),
    }


def _interval_rows(
    samples: Mapping[tuple[Any, ...], Sequence[float | None] | np.ndarray],
    *,
    original_units: str = "none",
) -> list[dict[str, Any]]:
    if original_units not in {"none", "errors", "all"}:
        raise ValueError("unknown interval unit policy")
    rows: list[dict[str, Any]] = []
    for key, values in samples.items():
        raw_values = np.asarray(values, dtype=np.float64)
        valid = raw_values[np.isfinite(raw_values)]
        row = {
            "key": list(key),
            "valid_replicate_count": int(valid.size),
            "invalid_replicate_count": int(raw_values.size - valid.size),
            "valid_replicate_fraction": float(valid.size / raw_values.size),
            "percentile_95": None,
        }
        if valid.size:
            lower, upper = np.quantile(valid, (0.025, 0.975))
            metric_name = str(key[-1])
            scale = original_units == "all" or (
                original_units == "errors"
                and any(token in metric_name for token in ("mae", "rmse", "bias", "disparity"))
            )
            interval = {"lower": float(lower), "upper": float(upper)}
            row["percentile_95"] = (
                {
                    "normalized": interval,
                    "original_1_7": {
                        "lower": RAW_SCALE_WIDTH * float(lower),
                        "upper": RAW_SCALE_WIDTH * float(upper),
                    },
                }
                if scale
                else interval
            )
        rows.append(row)
    return sorted(rows, key=lambda row: tuple(str(value) for value in row["key"]))


def _load_proxy_diagnostics(
    release: StageARelease,
    annotations: AnnotationRelease,
    groups: Mapping[str, np.ndarray],
) -> tuple[dict[tuple[str, int, str], int], dict[str, Any]]:
    source_configs = {
        "visual": "face_w1",
        "interview_audio": "audio_session_w1",
        "random": "random_proxy_w1",
    }
    assignments: dict[tuple[str, int, str], int] = {}
    artifact_hashes: list[dict[str, Any]] = []
    centre_separation: list[dict[str, Any]] = []
    for source, configuration in source_configs.items():
        for fold in FROZEN_OUTER_ROTATIONS:
            filename = f"{source}.fold-{fold}.proxy.json"
            manifest = release.lock_manifests[(configuration, FROZEN_FINAL_SEEDS[0])]
            inventory = {
                row.get("filename"): row
                for row in manifest.get("artifacts", [])
                if isinstance(row, Mapping)
            }
            record = inventory.get(filename)
            path = _bound_child(release.root, filename, "proxy manifest")
            if not isinstance(record, Mapping) or _file_sha256(path) != record.get("sha256"):
                raise ValueError("Stage-A proxy artifact is absent or checksum-invalid")
            payload = _read_canonical_object(path, "proxy manifest")
            _validate_signed_object(payload, "manifest_payload_sha256", "proxy manifest")
            if (
                payload.get("schema_version") != PROXY_ARTIFACT_SCHEMA_VERSION
                or payload.get("proxy_source") != source
                or payload.get("outer_rotation") != fold
                or payload.get("outcome_fields_accessed") != []
                or payload.get("protected_annotation_fields_accessed") != []
                or payload.get("eligibility_manifest_sha256")
                != release.closure["context"]["eligibility_manifest_sha256"]
            ):
                raise ValueError("Stage-A proxy artifact identity or firewall drifted")
            rows = payload.get("participants")
            if not isinstance(rows, list) or len(rows) != release.participant_count:
                raise ValueError("Stage-A proxy artifact participant coverage drifted")
            seen: set[str] = set()
            for row in rows:
                participant_id = row.get("participant_id") if isinstance(row, Mapping) else None
                proxy_group = row.get("proxy_group") if isinstance(row, Mapping) else None
                if (
                    participant_id not in release.participant_ids
                    or participant_id in seen
                    or isinstance(proxy_group, bool)
                    or proxy_group not in {0, 1}
                    or row.get("fixed_fold") != release.participant_folds[participant_id]
                ):
                    raise ValueError("Stage-A proxy assignment coverage or fold drifted")
                seen.add(participant_id)
                assignments[(source, fold, participant_id)] = int(proxy_group)
            artifact_hashes.append(
                {
                    "source": source,
                    "outer_rotation": fold,
                    "file_sha256": str(record["sha256"]),
                    "payload_sha256": str(payload["manifest_payload_sha256"]),
                }
            )
            if source != "random":
                estimator = payload.get("estimator")
                kmeans = estimator.get("kmeans") if isinstance(estimator, Mapping) else None
                centres = kmeans.get("centres_canonical") if isinstance(kmeans, Mapping) else None
                try:
                    centre_array = np.asarray(centres, dtype=np.float64)
                except (TypeError, ValueError) as error:
                    raise ValueError("learned proxy lacks finite canonical centres") from error
                if (
                    centre_array.ndim != 2
                    or centre_array.shape[0] != 2
                    or centre_array.shape[1] == 0
                    or not np.isfinite(centre_array).all()
                ):
                    raise ValueError("learned proxy lacks finite canonical centres")
                centre_separation.append(
                    {
                        "source": source,
                        "outer_rotation": fold,
                        "standardised_euclidean_centre_separation": float(
                            np.linalg.norm(centre_array[0] - centre_array[1])
                        ),
                        "dimension": int(centre_array.shape[1]),
                    }
                )

            for seed in FROZEN_FINAL_SEEDS:
                for peer_configuration in _proxy_peer_configurations(source):
                    peer_manifest = release.lock_manifests[(peer_configuration, seed)]
                    peer_inventory = {
                        row.get("filename"): row
                        for row in peer_manifest.get("artifacts", [])
                        if isinstance(row, Mapping)
                    }
                    peer_record = peer_inventory.get(filename)
                    if not isinstance(peer_record, Mapping) or peer_record.get(
                        "sha256"
                    ) != record.get("sha256"):
                        raise ValueError(
                            "proxy artifact changed across Stage-A configurations or seeds"
                        )

    stability = []
    for source in source_configs:
        for left in FROZEN_OUTER_ROTATIONS:
            for right in FROZEN_OUTER_ROTATIONS:
                if right <= left:
                    continue
                first = [
                    assignments[(source, left, participant)]
                    for participant in release.participant_ids
                ]
                second = [
                    assignments[(source, right, participant)]
                    for participant in release.participant_ids
                ]
                stability.append(
                    {
                        "source": source,
                        "rotation_a": left,
                        "rotation_b": right,
                        "adjusted_rand_index": _adjusted_rand(first, second),
                        "normalised_mutual_information": _normalised_mutual_information(
                            first, second
                        ),
                    }
                )
    cross_proxy = []
    for fold in FROZEN_OUTER_ROTATIONS:
        first = [
            assignments[("visual", fold, participant)] for participant in release.participant_ids
        ]
        second = [
            assignments[("interview_audio", fold, participant)]
            for participant in release.participant_ids
        ]
        cross_proxy.append(
            {
                "outer_rotation": fold,
                "adjusted_rand_index": _adjusted_rand(first, second),
                "normalised_mutual_information": _normalised_mutual_information(first, second),
            }
        )
    association = []
    for source in source_configs:
        for fold in FROZEN_OUTER_ROTATIONS:
            proxy = np.asarray(
                [
                    assignments[(source, fold, participant)]
                    for participant in release.participant_ids
                ],
                dtype=object,
            )
            for field in PERCEIVED_FIELDS:
                known = np.asarray([value is not None for value in groups[field]], dtype=bool)
                if not known.any():
                    association.append(
                        {
                            "source": source,
                            "outer_rotation": fold,
                            "field": field,
                            "known_participant_count": 0,
                            "status": "undefined_no_known_annotation",
                        }
                    )
                    continue
                association.append(
                    {
                        "source": source,
                        "outer_rotation": fold,
                        "field": field,
                        "known_participant_count": int(known.sum()),
                        "status": "reported_association_not_identity",
                        "adjusted_rand_index": _adjusted_rand(
                            proxy[known].tolist(), groups[field][known].tolist()
                        ),
                        "normalised_mutual_information": _normalised_mutual_information(
                            proxy[known].tolist(), groups[field][known].tolist()
                        ),
                        "contingency": _suppressed_contingency(
                            proxy[known].tolist(), groups[field][known].tolist()
                        ),
                    }
                )
    oof_assignment = {
        source: np.asarray(
            [
                assignments[(source, release.participant_folds[participant], participant)]
                for participant in release.participant_ids
            ],
            dtype=object,
        )
        for source in source_configs
    }
    oof_association = []
    for source, proxy in oof_assignment.items():
        for field in PERCEIVED_FIELDS:
            known = np.asarray([value is not None for value in groups[field]], dtype=bool)
            if not known.any():
                oof_association.append(
                    {
                        "source": source,
                        "field": field,
                        "known_participant_count": 0,
                        "status": "undefined_no_known_annotation",
                    }
                )
                continue
            oof_association.append(
                {
                    "source": source,
                    "field": field,
                    "known_participant_count": int(known.sum()),
                    "status": "reported_association_not_identity",
                    "adjusted_rand_index": _adjusted_rand(
                        proxy[known].tolist(), groups[field][known].tolist()
                    ),
                    "normalised_mutual_information": _normalised_mutual_information(
                        proxy[known].tolist(), groups[field][known].tolist()
                    ),
                    "contingency": _suppressed_contingency(
                        proxy[known].tolist(), groups[field][known].tolist()
                    ),
                }
            )
    return assignments, {
        "status": "reported",
        "artifact_bindings": artifact_hashes,
        "assignment_counts": [
            {
                "source": source,
                "outer_rotation": fold,
                "all_participants": _proxy_group_counts(
                    release, assignments, source=source, rotation=fold, role=None
                ),
                "roles": [
                    {
                        "role": role,
                        **_proxy_group_counts(
                            release,
                            assignments,
                            source=source,
                            rotation=fold,
                            role=role,
                        ),
                    }
                    for role in ("train", "validation", "test")
                ],
            }
            for source in source_configs
            for fold in FROZEN_OUTER_ROTATIONS
        ],
        "standardised_centre_separation": centre_separation,
        "cross_rotation_stability": stability,
        "visual_audio_agreement": cross_proxy,
        "oof_visual_audio_agreement": {
            "adjusted_rand_index": _adjusted_rand(
                oof_assignment["visual"].tolist(),
                oof_assignment["interview_audio"].tolist(),
            ),
            "normalised_mutual_information": _normalised_mutual_information(
                oof_assignment["visual"].tolist(),
                oof_assignment["interview_audio"].tolist(),
            ),
        },
        "post_firewall_proxy_audit_association": association,
        "oof_post_firewall_proxy_audit_association": oof_association,
        "interpretation": "association_not_identity_or_protected_attribute_inference",
    }


def _evaluate_modality_masks(
    release: StageARelease,
    arrays: Mapping[str, Any],
    path: str | Path,
) -> dict[str, Any]:
    source = _regular_file(path, "modality-mask predictions")
    payload = _read_canonical_object(source, "modality-mask predictions")
    _validate_signed_object(payload, "mask_payload_sha256", "modality-mask predictions")
    expected_top = {
        "schema_version",
        "stage_a_closure_file_sha256",
        "stage_a_closure_payload_sha256",
        "configuration",
        "intervention",
        "refit_or_retuning",
        "firewall_stage",
        "target_fields_accessed",
        "protected_annotation_fields_accessed",
        "rows",
        "mask_payload_sha256",
    }
    if (
        set(payload) != expected_top
        or payload["schema_version"] != STAGE_C_MASK_SCHEMA_VERSION
        or payload["stage_a_closure_file_sha256"] != release.closure_file_sha256
        or payload["stage_a_closure_payload_sha256"] != release.closure_payload_sha256
        or payload["configuration"] != "erm"
        or payload["intervention"]
        != "set_one_training_standardised_modality_summary_to_zero_at_inference"
        or payload["refit_or_retuning"] != "forbidden"
        or payload["firewall_stage"] != "A_locked_inference_only"
        or payload["target_fields_accessed"] != []
        or payload["protected_annotation_fields_accessed"] != []
        or not isinstance(payload["rows"], list)
    ):
        raise ValueError("modality-mask input contract drifted")
    expected_keys = {
        (seed, participant_id, phase, modality)
        for seed in FROZEN_FINAL_SEEDS
        for participant_id in release.participant_ids
        for phase in ("pre", "post")
        for modality in MASKED_MODALITIES
    }
    observed: dict[tuple[int, str, str, str], float] = {}
    base_rows = {
        (int(row["seed"]), str(row["participant_id"]), str(row["session_phase"])): row
        for seed in FROZEN_FINAL_SEEDS
        for row in release.learned_rows[("erm", seed)]
    }
    expected_row_fields = {
        "participant_id",
        "session_id",
        "source_session_id",
        "session_phase",
        "fixed_fold",
        "seed",
        "masked_modality",
        "prediction",
        "base_prediction_lock_id",
        "model_fit_sha256",
    }
    for row in payload["rows"]:
        if not isinstance(row, Mapping) or set(row) != expected_row_fields:
            raise ValueError("modality-mask row schema drifted")
        key = (
            row["seed"],
            row["participant_id"],
            row["session_phase"],
            row["masked_modality"],
        )
        base = base_rows.get(key[:3])
        prediction = row["prediction"]
        if (
            isinstance(row["seed"], bool)
            or not isinstance(row["seed"], int)
            or key not in expected_keys
            or key in observed
            or base is None
            or row["session_id"] != base["session_id"]
            or row["source_session_id"] != base["source_session_id"]
            or row["fixed_fold"] != base["fixed_fold"]
            or row["base_prediction_lock_id"] != base["prediction_lock_id"]
            or row["model_fit_sha256"] != base["model_fit_sha256"]
            or isinstance(prediction, bool)
            or not isinstance(prediction, (int, float))
            or not math.isfinite(float(prediction))
            or not 0 <= float(prediction) <= 1
        ):
            raise ValueError("modality-mask row identity, model or value drifted")
        observed[key] = float(prediction)
    if set(observed) != expected_keys:
        raise ValueError("modality-mask rows do not exactly cover ERM seeds, sessions and masks")
    target = arrays["target"]
    participant_index = {
        participant_id: index for index, participant_id in enumerate(release.participant_ids)
    }
    results = []
    for seed in FROZEN_FINAL_SEEDS:
        base_metrics = utility_metrics(
            target.reshape(-1), arrays["predictions"][("erm", seed)].reshape(-1)
        )
        for modality in MASKED_MODALITIES:
            matrix = np.empty_like(target)
            for participant_id in release.participant_ids:
                for phase, column in (("pre", 0), ("post", 1)):
                    matrix[participant_index[participant_id], column] = observed[
                        (seed, participant_id, phase, modality)
                    ]
            metrics = utility_metrics(target.reshape(-1), matrix.reshape(-1))
            results.append(
                {
                    "seed": seed,
                    "masked_modality": modality,
                    "metrics": metrics,
                    "change_from_locked_erm": _utility_delta(metrics, base_metrics),
                }
            )
    return {
        "status": "reported_locked_inference_only_no_refit",
        "file_sha256": _file_sha256(source),
        "payload_sha256": payload["mask_payload_sha256"],
        "results": results,
    }


def _restricted_join_rows(
    release: StageARelease,
    annotations: AnnotationRelease,
    *,
    proxy_rows: Mapping[tuple[str, int, str], int],
    oracle_disposition: StageBOracleDisposition | None,
) -> list[dict[str, Any]]:
    predictions: dict[tuple[str, str], dict[str, float]] = {}
    targets: dict[tuple[str, str], float] = {}
    sessions: dict[tuple[str, str], tuple[str, str, int]] = {}
    for (configuration, seed), rows in release.learned_rows.items():
        for row in rows:
            key = (str(row["participant_id"]), str(row["session_phase"]))
            predictions.setdefault(key, {})[f"{configuration}:seed-{seed}"] = float(
                row["prediction"]
            )
            targets[key] = float(row["target"])
            sessions[key] = (
                str(row["session_id"]),
                str(row["source_session_id"]),
                int(row["fixed_fold"]),
            )
    for configuration, rows in release.floor_rows.items():
        for row in rows:
            key = (str(row["participant_id"]), str(row["session_phase"]))
            predictions[key][configuration] = float(row["prediction"])
    if oracle_disposition is not None and oracle_disposition.state == "locked":
        configuration = str(oracle_disposition.payload["configuration"])
        for seed in FROZEN_FINAL_SEEDS:
            for row in oracle_disposition.prediction_rows[seed]:
                key = (str(row["participant_id"]), str(row["session_phase"]))
                predictions[key][f"{configuration}:seed-{seed}"] = float(row["prediction"])
    result = []
    for participant_id in release.participant_ids:
        proxy = {
            f"{source}:fold-{fold}": proxy_rows[(source, fold, participant_id)]
            for source in ("visual", "interview_audio", "random")
            for fold in FROZEN_OUTER_ROTATIONS
        }
        for phase in ("pre", "post"):
            key = (participant_id, phase)
            session_id, source_session_id, fold = sessions[key]
            annotation = _annotation_for_session(annotations, participant_id, phase)
            annotations_row = {
                field: _annotation_value(annotation, field) for field in PERCEIVED_FIELDS
            }
            result.append(
                {
                    "participant_id": participant_id,
                    "session_id": session_id,
                    "source_session_id": source_session_id,
                    "session_phase": phase,
                    "fixed_fold": fold,
                    "target": targets[key],
                    "predictions": dict(sorted(predictions[key].items())),
                    (
                        "perceived_attributes"
                        if annotations.release_method == SOURCE_RELEASE_METHOD
                        else "consensus"
                    ): annotations_row,
                    "assessability": _annotation_status(annotations, participant_id, phase),
                    "proxy_assignments": proxy,
                }
            )
    return result


def _codebook_categories(
    codebook: AnnotationCodebook | SourceAnnotationCodebook,
    field: str,
    *,
    release_method: str = CONSENSUS_METHOD,
) -> tuple[Any, ...]:
    if field == "perceived_gender_presentation":
        return tuple(codebook.gender_categories)
    if field == "perceived_racialised_appearance":
        return tuple(codebook.racialised_appearance_categories)
    if field == "perceived_skin_tone":
        return (
            tuple(codebook.skin_tone_values)
            if release_method == SOURCE_RELEASE_METHOD
            else SKIN_TONE_CONSENSUS_VALUES
        )
    raise ValueError(f"unsupported protected audit field {field!r}")


def _annotation_value(row: ConsensusAnnotation | SourceSessionAnnotation | None, field: str) -> Any:
    return None if row is None else getattr(row, field)


def _annotation_for_session(
    annotations: AnnotationRelease, participant_id: str, phase: str
) -> ConsensusAnnotation | SourceSessionAnnotation | None:
    if annotations.release_method == SOURCE_RELEASE_METHOD:
        return annotations.session_annotations.get((participant_id, phase))
    return annotations.participant_annotations.get(participant_id)


def _annotation_status(annotations: AnnotationRelease, participant_id: str, phase: str) -> str:
    if annotations.release_method == SOURCE_RELEASE_METHOD:
        return (
            "assessable"
            if (participant_id, phase) in annotations.session_annotations
            else "not_annotated_source_session"
        )
    return annotations.assessability[participant_id]


def _utility_delta(intervention: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "srcc_change": _optional_subtract(intervention["srcc"], reference["srcc"]),
        "pearson_change": _optional_subtract(intervention["pearson"], reference["pearson"]),
        "mae_improvement": _units(
            float(reference["mae"]["normalized"] - intervention["mae"]["normalized"])
        ),
        "rmse_improvement": _units(
            float(reference["rmse"]["normalized"] - intervention["rmse"]["normalized"])
        ),
    }


def _difference_disparity(
    reference: Mapping[str, Any], intervention: Mapping[str, Any]
) -> dict[str, Any] | None:
    left = _maximum_value(reference)
    right = _maximum_value(intervention)
    return None if left is None or right is None else _units(left - right)


def _maximum_value(disparity: Mapping[str, Any]) -> float | None:
    maximum = disparity.get("maximum_pairwise")
    if not isinstance(maximum, Mapping):
        return None
    value = maximum.get("wasserstein1")
    return float(value["normalized"]) if isinstance(value, Mapping) else None


def _mean_seed_contrast(seed_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    utility = {}
    for metric in ("srcc_change", "pearson_change"):
        values = [row["utility"][metric] for row in seed_rows]
        utility[metric] = _complete_mean(values)
    for metric in ("mae_improvement", "rmse_improvement"):
        values = [row["utility"][metric]["normalized"] for row in seed_rows]
        mean = _complete_mean(values)
        utility[metric] = None if mean is None else _units(mean)
    disparity = []
    for field in PERCEIVED_FIELDS:
        values = [
            next(
                item["gain_reference_minus_intervention"]
                for item in row["audited_disparity"]
                if item["field"] == field
            )
            for row in seed_rows
        ]
        normalized = [None if value is None else value["normalized"] for value in values]
        mean = _complete_mean(normalized)
        disparity.append({"field": field, "gain": None if mean is None else _units(mean)})
    return {"utility": utility, "audited_disparity": disparity}


def _originally_reportable_categories(
    groups: np.ndarray, categories: Sequence[Any]
) -> tuple[Any, ...]:
    return tuple(
        category
        for category in categories
        if int(sum(value == category for value in groups.tolist()))
        >= MINIMUM_REPORTABLE_PARTICIPANTS
    )


def _bootstrap_disparity_value(
    scores: np.ndarray,
    sampled_groups: np.ndarray,
    frozen_reportable_categories: Sequence[Any],
    *,
    sessions_per_participant: int = 2,
) -> float | None:
    """Recompute max W1; only an absent originally reportable group invalidates a draw."""
    categories = tuple(frozen_reportable_categories)
    if len(categories) < 2:
        return None
    expanded = np.repeat(sampled_groups, sessions_per_participant)
    if any(not np.any(expanded == category) for category in categories):
        return None
    values = [
        empirical_wasserstein1(scores[expanded == left], scores[expanded == right])
        for index, left in enumerate(categories)
        for right in categories[index + 1 :]
    ]
    return float(max(values))


def _proxy_peer_configurations(source: str) -> tuple[str, ...]:
    if source == "visual":
        return "face_sampling_only", "face_w1"
    if source == "interview_audio":
        return "audio_session_sampling_only", "audio_session_w1"
    if source == "random":
        return ("random_proxy_w1",)
    raise ValueError("unknown proxy source")


def _proxy_group_counts(
    release: StageARelease,
    assignments: Mapping[tuple[str, int, str], int],
    *,
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


def _correlation(left: np.ndarray, right: np.ndarray, *, rank: bool) -> float | None:
    if left.size < 2:
        return None
    if rank:
        left = _average_ranks(left)
        right = _average_ranks(right)
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = float(np.sqrt(np.sum(np.square(left)) * np.sum(np.square(right))))
    if denominator == 0:
        return None
    return float(np.sum(left * right) / denominator)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def _validated_pair(left: Sequence[float], right: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if (
        first.ndim != 1
        or not first.size
        or second.shape != first.shape
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
    ):
        raise ValueError("metric inputs must be aligned non-empty finite vectors")
    return first, second


def _adjusted_rand(left: Sequence[Any], right: Sequence[Any]) -> float:
    table, row_counts, column_counts = _contingency_matrix(left, right)
    n = int(table.sum())
    if n < 2:
        return 1.0

    def combinations(values: np.ndarray) -> float:
        return float(np.sum(values * (values - 1) / 2))

    observed = combinations(table)
    row_sum = combinations(row_counts)
    column_sum = combinations(column_counts)
    total = n * (n - 1) / 2
    expected = row_sum * column_sum / total
    maximum = 0.5 * (row_sum + column_sum)
    return 1.0 if maximum == expected else float((observed - expected) / (maximum - expected))


def _normalised_mutual_information(left: Sequence[Any], right: Sequence[Any]) -> float:
    table, row_counts, column_counts = _contingency_matrix(left, right)
    n = float(table.sum())
    probabilities = table / n
    row_probabilities = row_counts / n
    column_probabilities = column_counts / n
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
    return 1.0 if denominator == 0 else float(mutual / denominator)


def _contingency_matrix(
    left: Sequence[Any], right: Sequence[Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(left) != len(right) or not left:
        raise ValueError("association vectors must be non-empty and aligned")
    left_values = sorted(set(left), key=_category_sort)
    right_values = sorted(set(right), key=_category_sort)
    left_index = {value: index for index, value in enumerate(left_values)}
    right_index = {value: index for index, value in enumerate(right_values)}
    table = np.zeros((len(left_values), len(right_values)), dtype=np.float64)
    for first, second in zip(left, right, strict=True):
        table[left_index[first], right_index[second]] += 1
    return table, table.sum(axis=1), table.sum(axis=0)


def _suppressed_contingency(left: Sequence[Any], right: Sequence[Any]) -> list[dict[str, Any]]:
    counts = Counter(zip(left, right, strict=True))
    rows = []
    for (proxy_group, annotation_category), count in sorted(
        counts.items(), key=lambda item: (item[0][0], _category_sort(item[0][1]))
    ):
        rows.append(
            {
                "proxy_group": proxy_group,
                "annotation_category": annotation_category,
                "count": count if count >= MINIMUM_REPORTABLE_PARTICIPANTS else None,
                "suppressed": count < MINIMUM_REPORTABLE_PARTICIPANTS,
            }
        )
    return rows


def _assert_public_aggregate(payload: Mapping[str, Any]) -> None:
    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            forbidden = set(value) & _PROHIBITED_PUBLIC_KEYS
            if forbidden:
                raise ValueError(f"aggregate output contains restricted keys: {sorted(forbidden)}")
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)


def _assert_no_source_consensus_terms(payload: Mapping[str, Any]) -> None:
    """Keep multi-rater aggregation terminology out of single-rater evidence."""

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if "consensus" in str(key).lower():
                    raise ValueError("single-rater source evidence contains consensus terminology")
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
        elif isinstance(value, str) and "consensus" in value.lower():
            raise ValueError("single-rater source evidence contains consensus terminology")

    visit(payload)


def _validate_stage_c_payload_pair(
    restricted_join: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    *,
    release: StageARelease,
    annotations: AnnotationRelease,
) -> None:
    """Revalidate exact schemas, privacy and cross-bindings at publication boundaries."""
    _validate_signed_object(restricted_join, "join_payload_sha256", "restricted join")
    _validate_signed_object(aggregate, "aggregate_payload_sha256", "Stage-C aggregate")
    recorded_evaluator_provenance = validate_stage_c_evaluator_provenance(
        aggregate.get("evaluator_provenance")
    )
    execution_mode = aggregate.get("execution_mode")
    if execution_mode == "research" and recorded_evaluator_provenance["working_tree_dirty"]:
        raise ValueError("research Stage-C aggregate records dirty evaluator sources")
    if (
        set(restricted_join) != _RESTRICTED_JOIN_FIELDS
        or restricted_join.get("schema_version") != STAGE_C_RESTRICTED_JOIN_SCHEMA_VERSION
        or restricted_join.get("classification") != "restricted_participant_level_do_not_publish"
        or restricted_join.get("evaluator_version") != STAGE_C_EVALUATOR_VERSION
    ):
        raise ValueError("Stage-C restricted-join schema or classification drifted")
    if (
        set(aggregate) != _AGGREGATE_FIELDS
        or aggregate.get("schema_version") != STAGE_C_AGGREGATE_SCHEMA_VERSION
        or aggregate.get("scope") != "aggregate_only_no_participant_session_or_rater_identifiers"
        or aggregate.get("evaluator_version") != STAGE_C_EVALUATOR_VERSION
        or execution_mode not in {"research", "synthetic_contract_test"}
        or aggregate.get("evaluator_provenance") != recorded_evaluator_provenance
    ):
        raise ValueError("Stage-C aggregate schema, mode or provenance drifted")
    _assert_public_aggregate(aggregate)

    stage_a = aggregate.get("stage_a")
    annotation_summary = aggregate.get("annotations")
    if (
        restricted_join.get("stage_a_closure_file_sha256") != release.closure_file_sha256
        or restricted_join.get("stage_a_closure_payload_sha256") != release.closure_payload_sha256
        or not isinstance(stage_a, Mapping)
        or stage_a.get("closure_file_sha256") != release.closure_file_sha256
        or stage_a.get("closure_payload_sha256") != release.closure_payload_sha256
        or stage_a.get("eligible_participant_count") != release.participant_count
        or stage_a.get("eligible_session_count") != 2 * release.participant_count
        or restricted_join.get("participant_count") != release.participant_count
        or restricted_join.get("annotation_bindings") != annotations.bindings
        or not isinstance(annotation_summary, Mapping)
        or annotation_summary.get("bindings") != annotations.bindings
        or restricted_join.get("stage_b_oracle") != aggregate.get("stage_b_oracle")
    ):
        raise ValueError("Stage-C restricted and aggregate source bindings drifted")

    rows = restricted_join.get("rows")
    if not isinstance(rows, list) or len(rows) != 2 * release.participant_count:
        raise ValueError("Stage-C restricted join does not contain the exact paired roster")
    observed: set[tuple[str, str, str]] = set()
    source_annotations = annotations.release_method == SOURCE_RELEASE_METHOD
    if source_annotations:
        _assert_no_source_consensus_terms(restricted_join)
        _assert_no_source_consensus_terms(aggregate)
    expected_row_fields = (
        _SOURCE_RESTRICTED_ROW_FIELDS
        if source_annotations
        else _LEGACY_RESTRICTED_ROW_FIELDS
    )
    annotation_key = "perceived_attributes" if source_annotations else "consensus"
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != expected_row_fields:
            raise ValueError("Stage-C restricted row schema drifted")
        identity = (
            str(row["participant_id"]),
            str(row["session_phase"]),
            str(row["session_id"]),
        )
        participant_id = identity[0]
        if (
            identity in observed
            or identity not in release.session_keys
            or row["assessability"]
            != _annotation_status(annotations, participant_id, str(row["session_phase"]))
            or not isinstance(row["predictions"], Mapping)
            or not isinstance(row[annotation_key], Mapping)
            or not isinstance(row["proxy_assignments"], Mapping)
        ):
            raise ValueError("Stage-C restricted row identity or annotation binding drifted")
        expected_annotation = _annotation_for_session(
            annotations, participant_id, str(row["session_phase"])
        )
        if row[annotation_key] != {
            field: _annotation_value(expected_annotation, field)
            for field in PERCEIVED_FIELDS
        }:
            raise ValueError("Stage-C restricted annotation row drifted")
        observed.add(identity)
    if observed != set(release.session_keys):
        raise ValueError("Stage-C restricted row coverage drifted")


def _units(value: float) -> dict[str, float]:
    return {"normalized": float(value), "original_1_7": RAW_SCALE_WIDTH * float(value)}


def _score_units(value: float) -> dict[str, float]:
    return {
        "normalized": float(value),
        "original_1_7": 1.0 + RAW_SCALE_WIDTH * float(value),
    }


def _optional_subtract(left: Any, right: Any) -> float | None:
    return None if left is None or right is None else float(left) - float(right)


def _nan_if_none(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def _complete_mean(values: Sequence[Any]) -> float | None:
    return None if any(value is None for value in values) else float(np.mean(values))


def _mean_and_sample_sd(values: Sequence[float | None]) -> dict[str, float] | None:
    if any(value is None for value in values):
        return None
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(array)), "sample_sd": float(np.std(array, ddof=1))}


def _configuration_seed_sort(row: Mapping[str, Any]) -> tuple[str, int]:
    seed = row.get("seed")
    return str(row["configuration"]), -1 if seed is None else int(seed)


def _category_sort(value: Any) -> tuple[str, str]:
    return type(value).__name__, str(value)


def _signed_payload(unsigned: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {**unsigned, field: _payload_sha256(unsigned)}


def _validate_signed_object(payload: Mapping[str, Any], field: str, label: str) -> None:
    observed = _required_sha256(payload.get(field), f"{label} checksum")
    unsigned = {key: value for key, value in payload.items() if key != field}
    if observed != _payload_sha256(unsigned):
        raise ValueError(f"{label} payload checksum failed")


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def _required_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_canonical_object(path: Path, label: str) -> dict[str, Any]:
    source = _regular_file(path, label)
    return _canonical_object_from_bytes(source.read_bytes(), label)


def _canonical_object_from_bytes(raw: bytes, label: str) -> dict[str, Any]:
    """Parse one exact canonical JSON object from already-bound bytes."""
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
    if not isinstance(payload, dict) or canonical_json(payload) != raw:
        raise ValueError(f"{label} JSON is not byte-canonical")
    return payload


def _regular_directory(path: str | Path, label: str) -> Path:
    source = Path(path)
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"{label} must be a regular non-symlink directory")
    return source.resolve()


def _regular_file(path: str | Path, label: str) -> Path:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return source.resolve()


def _bound_child(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError(f"{label} filename must be a safe basename")
    path = root / value
    if path.resolve(strict=False).parent != root:
        raise ValueError(f"{label} escapes the Stage-A directory")
    return _regular_file(path, label)


def _lexical_output_path(path: str | Path, label: str) -> Path:
    destination = Path(path).expanduser()
    if destination.name in {"", ".", ".."}:
        raise ValueError(f"{label} output filename is invalid")
    if any(component in {".", ".."} for component in destination.parts):
        raise ValueError(f"{label} output path contains unsafe dot components")
    if not destination.is_absolute():
        destination = Path.cwd() / destination
    return destination
