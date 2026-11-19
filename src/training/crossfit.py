"""Outcome-blind eligibility and deterministic participant-grouped training.

Definitions are described in ``docs/research-overview.md``. This module consumes only
checksum-verified frozen feature shards when it constructs the complete-pair eligibility
manifest.  Outcomes are bound afterwards, and protected annotations are not part of any API in
this module.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import random
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.data import (
    CORPUS_ID,
    EXPECTED_PARTICIPANTS,
    EXPECTED_SESSIONS,
    SESSION_MAPPING_PROVENANCE_FIELDS,
    TRANSCRIPT_PARSER_REVISION,
    MITInterviewDataset,
    SessionMappingContext,
    assign_participant_folds,
    canonical_participant_id,
    load_mit_interview_dataset,
    parse_session_id,
)
from src.data.mit_session_mapping import MIT_SESSION_MAPPING_SCHEMA_VERSION
from src.features import (
    FEATURE_EXTRACTION_CLOSURE_SCHEMA_VERSION,
    FeatureExtractionClosure,
    FeatureRecordArtifacts,
)
from src.features.extraction import EXTRACTOR_VERSION, SCHEMA_VERSION, SHARD_KEYS, safe_clip_stem
from src.models.fusion import LowCapacityMultimodalScorer
from src.secure_io import AnchoredDirectory, AnchoredFile

Condition = Literal["pre", "post"]

ELIGIBILITY_SCHEMA_VERSION = "fair-avi-mit-complete-pair-eligibility-3"
TARGET_SCHEMA_VERSION = "fair-avi-mit-target-set-2"
PREDICTION_SCHEMA_VERSION = "fair-avi-mit-oof-predictions-2"
PREDICTION_LOCK_SCHEMA_VERSION = "fair-avi-mit-prediction-lock-4"
FIT_ARTIFACT_SCHEMA_VERSION = "fair-avi-mit-fit-artifact-1"
TARGET_TRANSFORM = "(raw-1)/6"
FIREWALL_STAGE = "A"
PROXY_LEARNED_CONFIGURATIONS = (
    "face_sampling_only",
    "audio_session_sampling_only",
    "random_proxy_w1",
    "face_w1",
    "audio_session_w1",
)
FIREWALLED_LEARNED_CONFIGURATIONS = (
    "erm",
    *PROXY_LEARNED_CONFIGURATIONS,
    "arl",
)
RIDGE_DIAGNOSTIC_MODALITIES = {
    "ridge_video": ("video",),
    "ridge_audio": ("audio",),
    "ridge_text": ("text",),
    "ridge_video_audio": ("video", "audio"),
    "ridge_video_text": ("video", "text"),
    "ridge_audio_text": ("audio", "text"),
    "ridge": ("video", "audio", "text"),
}
DIAGNOSTIC_FLOOR_CONFIGURATIONS = (
    "mean",
    "phase_mean",
    *RIDGE_DIAGNOSTIC_MODALITIES,
)
ARL_PRETRAINING_UPDATES = 6
ARL_WEIGHT_DENOMINATOR_GUARD = 1e-4
ARL_DIAGNOSTIC_TIMING_VERSION = "arl_diagnostics_v1"
ARL_DIAGNOSTIC_TIMING = (
    "weighting_and_learner_metrics_pre_learner_step_"
    "adversary_objective_and_gradient_post_learner_step_pre_adversary_step"
)
CANONICAL_FOLD_MANIFEST_SHA256 = "d8af01ba696cbf44960177ed2f1d513c672dc4a83c9f5635cd371dcd16ee8a93"
TARGET_TABLE_REFERENCE = "Labels/turker_scores_full_interview.csv"
PREDICTION_ROW_FIELDS = {
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
    "encoder_revisions",
    "feature_contract_sha256",
    "preprocessing_sha256",
    "proxy_source",
    "sampling_mode",
    "coefficients",
    "checkpoint_epoch",
    "firewall_stage",
    "eligibility_manifest_sha256",
    "session_mapping",
    "prediction_lock_id",
}

SUMMARY_DIMENSIONS = {"video": 1024, "audio": 1536, "text": 768}
EXPECTED_PREPROCESSING = {
    "video": {
        "input": "preselected_principal_face_crops",
        "sampling_fps": 1,
        "maximum_samples": 128,
        "source_index_set": "candidate_valid_1hz_within_active_interview_interval",
        "selection": "endpoint_rank_over_candidate_valid_indices_when_capped",
        "processor": "pinned_artifact_rgb_bicubic_shortest_256_center_crop_224_imagenet_norm",
    },
    "audio": {
        "mono": True,
        "sample_rate": 16_000,
        "waveform_normalization": False,
        "context": "active_interview_audio_not_candidate_voice",
        "boundary_source": "candidate_face_active_interval",
        "window_samples": 320_000,
        "maximum_windows": 16,
        "selection": "floor(j*(N-1)/(maximum-1))_when_capped",
        "pooling": "bounded_window_final_hidden_center_time_half_open_seconds_mean_then_max",
        "frame_centre_formula": "(j*320+199.5)/16000",
        "partial_last_second": "retain_if_nonempty",
        "end_alignment": "allow_end_of_stream_shortfall_below_one_source_audio_tick",
    },
    "text": {
        "source": "corpus_supplied_only",
        "chunking": "tokenizer_ids_nonoverlapping_no_specials_then_add_specials",
        "max_sequence_tokens_including_specials": 384,
        "pooling": "attention_mask_mean_then_l2_normalize_per_chunk",
    },
}
ENCODER_CONTRACT = {
    "video": {
        "name": "facebook/dinov2-large",
        "revision": "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
        "output_dim": 1024,
        "weight_sha256": "399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23",
    },
    "audio": {
        "name": "microsoft/wavlm-base-plus",
        "revision": "4c66d4806a428f2e922ccfa1a962776e232d487b",
        "output_dim": 768,
        "weight_sha256": "3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51",
    },
    "text": {
        "name": "sentence-transformers/all-mpnet-base-v2",
        "revision": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
        "output_dim": 768,
        "weight_sha256": "78c0197b6159d92658e319bc1d72e4c73a9a03dd03815e70e555c5ef05615658",
    },
}


class FeatureArtifactError(ValueError):
    """A stable outcome-blind reason why one feature artifact is unusable."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        excludable: bool = False,
        artifact_sha256: str | None = None,
        extraction_fingerprint: str | None = None,
        run_contract_sha256: str | None = None,
        config_sha256: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.excludable = excludable
        self.artifact_sha256 = artifact_sha256
        self.extraction_fingerprint = extraction_fingerprint
        self.run_contract_sha256 = run_contract_sha256
        self.config_sha256 = config_sha256


@dataclass(frozen=True, slots=True)
class SessionIdentity:
    """Outcome-free identity and fixed fold for one interview session."""

    session_id: str
    participant_id: str
    condition: Condition
    fold: int
    source_session_id: str | None = None


@dataclass(frozen=True, slots=True)
class FeatureSummary:
    """Validated frozen session summaries plus non-path provenance."""

    identity: SessionIdentity
    video: np.ndarray
    audio: np.ndarray
    text: np.ndarray
    sidecar_sha256: str
    shard_sha256: str
    extraction_fingerprint: str
    feature_contract_sha256: str
    preprocessing_sha256: str
    run_contract_sha256: str
    config_sha256: str


@dataclass(frozen=True, slots=True)
class EligibilityResult:
    """One outcome-blind complete-pair analysis set and its canonical manifest."""

    features: tuple[FeatureSummary, ...]
    manifest: Mapping[str, Any]
    manifest_bytes: bytes
    manifest_sha256: str
    fold_manifest_bytes: bytes
    fold_manifest_sha256: str
    feature_contract_sha256: str
    preprocessing_sha256: str
    feature_config_sha256: str
    production_ready: bool

    def __post_init__(self) -> None:
        _validate_eligibility_result(self)

    @property
    def participant_ids(self) -> tuple[str, ...]:
        """Return eligible canonical participant IDs in numeric order."""
        return tuple(sorted({row.identity.participant_id for row in self.features}, key=int))


@dataclass(frozen=True, slots=True)
class EligibilityLock:
    """Durable outcome-blind lock written before target construction."""

    output_dir: Path
    eligibility_filename: str
    eligibility_file_sha256: str
    fold_filename: str
    fold_file_sha256: str
    eligibility_manifest_sha256: str
    output_device: int
    output_inode: int

    def __post_init__(self) -> None:
        if not isinstance(self.output_dir, Path) or not self.output_dir.is_absolute():
            raise ValueError("eligibility lock output_dir must be an absolute Path")
        for filename in (self.eligibility_filename, self.fold_filename):
            if not filename or Path(filename).name != filename:
                raise ValueError("eligibility lock filenames must be safe basenames")
        _require_hex(self.eligibility_file_sha256, 64, "eligibility_file_sha256")
        _require_hex(self.fold_file_sha256, 64, "fold_file_sha256")
        _require_hex(self.eligibility_manifest_sha256, 64, "eligibility_manifest_sha256")
        if (
            isinstance(self.output_device, bool)
            or not isinstance(self.output_device, int)
            or self.output_device < 0
            or isinstance(self.output_inode, bool)
            or not isinstance(self.output_inode, int)
            or self.output_inode <= 0
        ):
            raise ValueError("eligibility lock output identity is invalid")


@dataclass(frozen=True, slots=True)
class TargetSet:
    """Outcome values with a canonical, checksum-bound source manifest."""

    values: tuple[tuple[str, float], ...]
    manifest: Mapping[str, Any]
    manifest_bytes: bytes
    manifest_sha256: str
    production_ready: bool
    eligibility_lock: EligibilityLock | None = None

    def __post_init__(self) -> None:
        _validate_target_set(self)

    def as_mapping(self) -> dict[str, float]:
        """Return a fresh session-to-target mapping."""
        return dict(self.values)


@dataclass(frozen=True, slots=True)
class RunProvenance:
    """Code/config identity supplied by the eventual run entry point."""

    config_sha256: str
    code_commit_sha: str
    code_source_tree_sha256: str
    source_date_epoch: int = 0

    def __post_init__(self) -> None:
        _require_hex(self.config_sha256, 64, "config_sha256")
        _require_hex(self.code_commit_sha, 40, "code_commit_sha")
        _require_hex(self.code_source_tree_sha256, 64, "code_source_tree_sha256")
        if (
            isinstance(self.source_date_epoch, bool)
            or not isinstance(self.source_date_epoch, int)
            or self.source_date_epoch < 0
        ):
            raise ValueError("source_date_epoch must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ArtifactPayload:
    """One immutable fit artifact to write beside a prediction lock."""

    filename: str
    content: bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.content).hexdigest()


@dataclass(frozen=True, slots=True)
class PredictionBundle:
    """Canonical OOF predictions, fit artifacts and their self-checking lock."""

    configuration: str
    seed: int | None
    prediction_filename: str
    manifest_filename: str
    predictions: tuple[Mapping[str, Any], ...]
    prediction_bytes: bytes
    manifest: Mapping[str, Any]
    manifest_bytes: bytes
    manifest_sha256: str
    artifacts: tuple[ArtifactPayload, ...]
    execution_mode: str
    eligibility_lock: EligibilityLock | None

    def __post_init__(self) -> None:
        _validate_prediction_bundle(self)


class ARLAdversary(torch.nn.Module):
    """Training-only affine adversary for the frozen ARL adaptation.

    Formal ARL terminology and the project-specific departures are recorded in
    ``docs/research-overview.md``. The adversary consumes only a detached copy of the
    common scorer's 24-dimensional fused representation and the normalized continuous target.
    """

    EXPECTED_TRAINABLE_PARAMETERS = 26

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(25, 1)
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)
        if self.trainable_parameter_count != self.EXPECTED_TRAINABLE_PARAMETERS:
            raise RuntimeError("ARL adversary parameter contract drifted")

    @property
    def trainable_parameter_count(self) -> int:
        """Return the exact number of training-only adversary parameters."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(
        self, fused_representation: torch.Tensor, normalized_target: torch.Tensor
    ) -> torch.Tensor:
        """Return one logit per row from detached representation and target inputs."""
        if fused_representation.ndim != 2 or fused_representation.shape[1] != 24:
            raise ValueError("fused_representation must have shape (batch,24)")
        if normalized_target.ndim != 1:
            raise ValueError("normalized_target must have shape (batch,)")
        if fused_representation.shape[0] != normalized_target.shape[0]:
            raise ValueError("ARL adversary inputs must have the same batch size")
        if fused_representation.shape[0] == 0:
            raise ValueError("ARL adversary inputs may not be empty")
        value = torch.cat(
            (fused_representation.detach(), normalized_target.detach().unsqueeze(1)), dim=1
        )
        return self.linear(value).squeeze(1)


def arl_example_weights(adversary_logits: torch.Tensor) -> torch.Tensor:
    """Apply the frozen guarded current-minibatch Lahoti weighting formula."""
    weights, _scores, _denominator = _arl_weight_components(adversary_logits)
    return weights


def _arl_weight_components(
    adversary_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return weights, sigmoid scores and the exact guarded batch denominator."""
    if adversary_logits.ndim != 1 or adversary_logits.numel() == 0:
        raise ValueError("adversary_logits must be one non-empty vector")
    if not torch.is_floating_point(adversary_logits) or not bool(
        torch.isfinite(adversary_logits).all()
    ):
        raise ValueError("adversary_logits must be finite floating-point values")
    scores = torch.sigmoid(adversary_logits)
    denominator = torch.clamp(scores.mean(), min=ARL_WEIGHT_DENOMINATOR_GUARD)
    return 1.0 + scores / denominator, scores, denominator


def session_identities(dataset: MITInterviewDataset) -> tuple[SessionIdentity, ...]:
    """Project the validated corpus adapter to an outcome-free session roster."""
    return tuple(
        SessionIdentity(
            session_id=row.session_id,
            participant_id=row.participant_id,
            condition=row.condition,
            fold=row.fold,
            source_session_id=getattr(row, "source_session_id", None),
        )
        for row in dataset.records
    )


def build_target_set(
    dataset: MITInterviewDataset,
    dataset_root: str | Path,
    *,
    eligibility: EligibilityResult,
    eligibility_lock: EligibilityLock,
    session_mapping: SessionMappingContext,
) -> TargetSet:
    """Bind targets to the checksum-verified MIT aggregate-rating table.

    Outcomes are deliberately constructed through a separate boundary after eligibility.  The
    supplied dataset must equal a fresh strict load from ``dataset_root``; this prevents a caller
    from presenting an arbitrary target mapping as the frozen AGGR RecommendHiring construct.
    """
    if not isinstance(dataset, MITInterviewDataset):
        raise TypeError("dataset must be a validated MITInterviewDataset")
    _validate_eligibility_result(eligibility)
    if not eligibility.production_ready:
        raise ValueError("research targets require full-corpus eligibility")
    if not isinstance(session_mapping, SessionMappingContext):
        raise TypeError("research targets require a verified session mapping")
    if eligibility.manifest.get("session_mapping") != session_mapping.provenance:
        raise ValueError("target construction mapping disagrees with frozen eligibility")
    session_mapping.verify_current_files()
    _validate_durable_eligibility_lock(eligibility_lock, eligibility)
    supplied_root = Path(dataset_root)
    if supplied_root.is_symlink() or not supplied_root.is_dir():
        raise ValueError("dataset_root must be a regular non-symlink directory")
    root = supplied_root.resolve()
    rating_path = root / TARGET_TABLE_REFERENCE
    if (
        rating_path.is_symlink()
        or not rating_path.is_file()
        or not rating_path.resolve().is_relative_to(root)
    ):
        raise ValueError("aggregate rating table is unavailable or unsafe")
    source_sha256 = _file_sha256(rating_path)
    validated = load_mit_interview_dataset(root, session_mapping=session_mapping)
    if _file_sha256(rating_path) != source_sha256:
        raise ValueError("aggregate rating table changed during target validation")
    if dataset != validated:
        raise ValueError("dataset does not match a fresh validation of dataset_root")
    _validated_roster(
        session_identities(validated),
        require_full_corpus=True,
        session_mapping=session_mapping,
    )

    values: list[tuple[str, float]] = []
    rows: list[dict[str, float | str]] = []
    for record in validated.records:
        raw = float(record.outcome.recommend_hiring_raw)
        target = float(record.outcome.recommend_hiring)
        if not math.isclose(target, (raw - 1.0) / 6.0, rel_tol=0.0, abs_tol=0.0):
            raise ValueError("MIT target normalisation drifted from the frozen transform")
        values.append((record.session_id, target))
        rows.append(
            {
                "session_id": record.session_id,
                "source_session_id": record.source_session_id,
                "recommend_hiring_raw": raw,
                "target": target,
            }
        )
    unsigned = {
        "schema_version": TARGET_SCHEMA_VERSION,
        "corpus": CORPUS_ID,
        "construction_mode": "verified_mit_aggregate_rating_table",
        "construct": "crowd_perceived_hireability_not_actual_hiring",
        "source": {
            "reference": TARGET_TABLE_REFERENCE,
            "sha256": source_sha256,
            "worker": "AGGR",
            "field": "RecommendHiring",
        },
        "raw_scale": [1.0, 7.0],
        "transform": TARGET_TRANSFORM,
        "session_mapping": session_mapping.provenance,
        "eligibility_lock": {
            "eligibility_manifest_sha256": eligibility.manifest_sha256,
            "eligibility_filename": eligibility_lock.eligibility_filename,
            "eligibility_file_sha256": eligibility_lock.eligibility_file_sha256,
            "fold_filename": eligibility_lock.fold_filename,
            "fold_file_sha256": eligibility_lock.fold_file_sha256,
        },
        "protected_annotation_fields_accessed": [],
        "rows": rows,
    }
    manifest_sha256 = _payload_sha256(unsigned)
    manifest = {**unsigned, "manifest_payload_sha256": manifest_sha256}
    if _file_sha256(rating_path) != source_sha256:
        raise ValueError("aggregate rating table changed during target construction")
    session_mapping.verify_current_files()
    return TargetSet(
        values=tuple(values),
        manifest=manifest,
        manifest_bytes=canonical_json_bytes(manifest),
        manifest_sha256=manifest_sha256,
        production_ready=True,
        eligibility_lock=eligibility_lock,
    )


def synthetic_target_set(targets: Mapping[str, float]) -> TargetSet:
    """Create an explicitly non-research target set for bounded synthetic tests."""
    if not isinstance(targets, Mapping) or not targets:
        raise TypeError("synthetic targets must be a non-empty session-ID mapping")
    values: list[tuple[str, float]] = []
    for supplied_session_id, supplied_value in targets.items():
        session_id, _, _ = parse_session_id(supplied_session_id)
        if supplied_session_id != session_id:
            raise ValueError("synthetic target session IDs must already be canonical")
        if isinstance(supplied_value, bool):
            raise TypeError("synthetic targets must be finite numeric values")
        try:
            target = float(supplied_value)
        except (TypeError, ValueError) as error:
            raise TypeError("synthetic targets must be finite numeric values") from error
        if not math.isfinite(target) or not 0.0 <= target <= 1.0:
            raise ValueError("synthetic targets must be finite and in [0,1]")
        values.append((session_id, target))
    values.sort(key=lambda row: (int(parse_session_id(row[0])[1]), row[0].startswith("PP")))
    unsigned = {
        "schema_version": TARGET_SCHEMA_VERSION,
        "corpus": CORPUS_ID,
        "construction_mode": "synthetic_contract_test",
        "source": "caller_supplied_synthetic_values",
        "transform": "already_normalised_synthetic_value",
        "protected_annotation_fields_accessed": [],
        "rows": [{"session_id": session_id, "target": target} for session_id, target in values],
    }
    manifest_sha256 = _payload_sha256(unsigned)
    manifest = {**unsigned, "manifest_payload_sha256": manifest_sha256}
    return TargetSet(
        values=tuple(values),
        manifest=manifest,
        manifest_bytes=canonical_json_bytes(manifest),
        manifest_sha256=manifest_sha256,
        production_ready=False,
        eligibility_lock=None,
    )


def build_complete_pair_eligibility(
    sessions: Sequence[SessionIdentity],
    feature_dir: str | Path | None = None,
    *,
    feature_closure: FeatureExtractionClosure | None = None,
    session_mapping: SessionMappingContext | None = None,
    require_full_corpus: bool = True,
) -> EligibilityResult:
    """Build the common complete-pair manifest without outcomes or annotations.

    The production path requires all 69 participant identities.  ``require_full_corpus=False``
    exists only for bounded synthetic contract tests and is recorded visibly in the manifest.
    """
    if require_full_corpus:
        if feature_dir is not None:
            raise ValueError(
                "production eligibility consumes only a feature-extraction closure capability"
            )
        if not isinstance(feature_closure, FeatureExtractionClosure):
            raise TypeError(
                "production eligibility requires a validated feature-extraction closure"
            )
        if not isinstance(session_mapping, SessionMappingContext):
            raise TypeError("production eligibility requires an approved session mapping")
    else:
        if feature_closure is not None:
            raise ValueError("synthetic eligibility may not consume a research feature closure")
        if feature_dir is None:
            raise ValueError("synthetic eligibility requires an explicit feature directory")

    roster = _validated_roster(
        sessions,
        require_full_corpus=require_full_corpus,
        session_mapping=session_mapping,
    )
    successes: dict[str, FeatureSummary] = {}
    failures: dict[str, dict[str, Any]] = {}
    closure_manifest: Mapping[str, Any] | None = None
    bound_closure_hashes: tuple[str, str] | None = None
    if feature_closure is not None:
        bound_closure_hashes = (
            feature_closure.closure_file_sha256,
            feature_closure.closure_payload_sha256,
        )
        closure_manifest = _validated_feature_closure_manifest(
            feature_closure, roster=roster, session_mapping=session_mapping
        )
        for visit_index, identity in enumerate(roster):
            artifacts = feature_closure.read_record_artifacts(visit_index)
            expected_record = closure_manifest["records"][visit_index]
            if dict(artifacts.record) != expected_record:
                raise ValueError("feature closure record changed after manifest validation")
            _collect_feature_artifact(
                successes,
                failures,
                identity,
                lambda artifacts=artifacts, identity=identity: _load_feature_summary_from_closure(
                    artifacts, identity, session_mapping=session_mapping
                ),
            )
        feature_closure.verify_current_files()
        if bound_closure_hashes != (
            feature_closure.closure_file_sha256,
            feature_closure.closure_payload_sha256,
        ):
            raise RuntimeError("feature extraction closure binding changed during eligibility")
        session_mapping.verify_current_files()
    else:
        try:
            with AnchoredDirectory.open(feature_dir, directory_mode=None) as feature_root:
                for identity in roster:
                    _collect_feature_artifact(
                        successes,
                        failures,
                        identity,
                        lambda identity=identity: _load_feature_summary(
                            feature_root, identity, session_mapping=session_mapping
                        ),
                    )
                feature_root.verify_path_identity()
        except (OSError, RuntimeError, ValueError) as error:
            if isinstance(error, ValueError) and str(error).startswith("feature integrity failed"):
                raise
            raise ValueError("feature_dir must be an anchored non-symlink directory") from error

    contracts = {row.feature_contract_sha256 for row in successes.values()}
    preprocessing = {row.preprocessing_sha256 for row in successes.values()}
    run_contracts = {row.run_contract_sha256 for row in successes.values()}
    run_contracts.update(
        failure["run_contract_sha256"]
        for failure in failures.values()
        if failure["run_contract_sha256"] is not None
    )
    if len(contracts) > 1 or len(preprocessing) > 1:
        raise ValueError("successful feature shards do not share one frozen extraction contract")
    if len(run_contracts) > 1:
        raise ValueError("feature successes and failures do not share one extraction run contract")
    feature_contract_sha256 = next(iter(contracts), "0" * 64)
    preprocessing_sha256 = next(iter(preprocessing), "0" * 64)
    run_contract_sha256 = next(iter(run_contracts), "0" * 64)
    config_values = {row.config_sha256 for row in successes.values()}
    config_values.update(
        failure["config_sha256"]
        for failure in failures.values()
        if failure["config_sha256"] is not None
    )
    if len(config_values) > 1:
        raise ValueError("feature artifacts do not share one configuration checksum")
    feature_config_sha256 = next(iter(config_values), "0" * 64)
    closure_file_sha256: str | None = None
    closure_payload_sha256: str | None = None
    if closure_manifest is not None and bound_closure_hashes is not None:
        observed_contract_sets = {
            "feature_contract_sha256": contracts,
            "preprocessing_sha256": preprocessing,
            "feature_run_contract_sha256": run_contracts,
            "config_sha256": config_values,
        }
        for name, observed in observed_contract_sets.items():
            if observed and observed != {closure_manifest[name]}:
                raise ValueError(f"feature artifacts disagree with closure field {name}")
        feature_contract_sha256 = closure_manifest["feature_contract_sha256"]
        preprocessing_sha256 = closure_manifest["preprocessing_sha256"]
        run_contract_sha256 = closure_manifest["feature_run_contract_sha256"]
        feature_config_sha256 = closure_manifest["config_sha256"]
        closure_file_sha256, closure_payload_sha256 = bound_closure_hashes

    eligible: list[FeatureSummary] = []
    eligible_manifest_rows: list[dict[str, Any]] = []
    excluded_manifest_rows: list[dict[str, Any]] = []
    by_participant: dict[str, list[SessionIdentity]] = {}
    for identity in roster:
        by_participant.setdefault(identity.participant_id, []).append(identity)
    for participant_id in sorted(by_participant, key=int):
        pair = sorted(by_participant[participant_id], key=lambda row: row.condition == "post")
        pair_failures = [
            {
                "session_id": row.session_id,
                "source_session_id": row.source_session_id,
                **failures[row.session_id],
            }
            for row in pair
            if row.session_id in failures
        ]
        if pair_failures:
            excluded_manifest_rows.append(
                {
                    "participant_id": participant_id,
                    "fold": pair[0].fold,
                    "session_failures": pair_failures,
                }
            )
            continue
        feature_pair = [successes[row.session_id] for row in pair]
        eligible.extend(feature_pair)
        eligible_manifest_rows.append(
            {
                "participant_id": participant_id,
                "fold": pair[0].fold,
                "sessions": [
                    {
                        "session_id": row.identity.session_id,
                        "source_session_id": row.identity.source_session_id,
                        "condition": row.identity.condition,
                        "sidecar_sha256": row.sidecar_sha256,
                        "shard_sha256": row.shard_sha256,
                        "extraction_fingerprint": row.extraction_fingerprint,
                        "summary_sha256": {
                            modality: _summary_sha256(getattr(row, modality))
                            for modality in SUMMARY_DIMENSIONS
                        },
                    }
                    for row in feature_pair
                ],
            }
        )

    fold_manifest = _fold_manifest_bytes(roster)
    fold_sha256 = hashlib.sha256(fold_manifest).hexdigest()
    fold_counts = {
        str(fold): sum(row["fold"] == fold for row in eligible_manifest_rows) for fold in range(5)
    }
    unsigned = {
        "schema_version": ELIGIBILITY_SCHEMA_VERSION,
        "corpus": CORPUS_ID,
        "construction_mode": "full_corpus" if require_full_corpus else "synthetic_contract_test",
        "session_mapping": session_mapping.provenance if session_mapping is not None else None,
        "outcome_fields_accessed": [],
        "protected_annotation_fields_accessed": [],
        "required_modalities": ["video", "interview_audio", "candidate_text"],
        "complete_pair_rule": "both_sessions_have_all_three_valid_frozen_summaries",
        "imputation": "forbidden",
        "input_sessions": [
            {
                "session_id": row.session_id,
                "source_session_id": row.source_session_id,
                "participant_id": row.participant_id,
                "condition": row.condition,
                "fold": row.fold,
            }
            for row in roster
        ],
        "input_participant_count": len(by_participant),
        "input_session_count": len(roster),
        "eligible_participant_count": len(eligible_manifest_rows),
        "eligible_session_count": len(eligible),
        "eligible_participants_per_fold": fold_counts,
        "fold_manifest_sha256": fold_sha256,
        "feature_contract_sha256": feature_contract_sha256,
        "preprocessing_sha256": preprocessing_sha256,
        "feature_run_contract_sha256": run_contract_sha256,
        "feature_config_sha256": feature_config_sha256,
        "feature_extraction_closure_file_sha256": closure_file_sha256,
        "feature_extraction_closure_payload_sha256": closure_payload_sha256,
        "eligible_participants": eligible_manifest_rows,
        "exclusions": excluded_manifest_rows,
    }
    manifest_sha256 = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    manifest = {**unsigned, "manifest_payload_sha256": manifest_sha256}
    manifest_bytes = canonical_json_bytes(manifest)
    if session_mapping is not None:
        session_mapping.verify_current_files()
    return EligibilityResult(
        features=tuple(eligible),
        manifest=manifest,
        manifest_bytes=manifest_bytes,
        manifest_sha256=manifest_sha256,
        fold_manifest_bytes=fold_manifest,
        fold_manifest_sha256=fold_sha256,
        feature_contract_sha256=feature_contract_sha256,
        preprocessing_sha256=preprocessing_sha256,
        feature_config_sha256=feature_config_sha256,
        production_ready=require_full_corpus,
    )


def write_eligibility_lock(
    output_dir: str | Path | AnchoredDirectory, eligibility: EligibilityResult
) -> EligibilityLock:
    """Durably lock outcome-blind eligibility and folds before outcomes are bound."""
    _validate_eligibility_result(eligibility)
    if isinstance(output_dir, AnchoredDirectory):
        return _write_eligibility_lock(output_dir, eligibility)
    try:
        output = AnchoredDirectory.open(output_dir, directory_mode=None)
    except ValueError:
        output = AnchoredDirectory.open(
            output_dir, create=True, directory_mode=0o700
        )
    with output:
        return _write_eligibility_lock(output, eligibility)


def _write_eligibility_lock(
    output: AnchoredDirectory, eligibility: EligibilityResult
) -> EligibilityLock:
    output.verify_path_identity()
    eligibility_filename = "complete-pair-eligibility.json"
    fold_filename = "participant-folds.csv"
    output.publish_bytes(
        eligibility_filename, eligibility.manifest_bytes, mode=0o600
    )
    output.publish_bytes(
        fold_filename, eligibility.fold_manifest_bytes, mode=0o600
    )
    lock = EligibilityLock(
        output_dir=output.path,
        eligibility_filename=eligibility_filename,
        eligibility_file_sha256=hashlib.sha256(eligibility.manifest_bytes).hexdigest(),
        fold_filename=fold_filename,
        fold_file_sha256=hashlib.sha256(eligibility.fold_manifest_bytes).hexdigest(),
        eligibility_manifest_sha256=eligibility.manifest_sha256,
        output_device=output.device,
        output_inode=output.inode,
    )
    _validate_durable_eligibility_lock(lock, eligibility, output=output)
    return lock


def load_eligibility_lock(
    output_dir: str | Path | AnchoredDirectory, eligibility: EligibilityResult
) -> EligibilityLock:
    """Open an existing outcome-blind lock without creating or replacing files."""
    _validate_eligibility_result(eligibility)
    if isinstance(output_dir, AnchoredDirectory):
        return _load_eligibility_lock(output_dir, eligibility)
    try:
        with AnchoredDirectory.open(output_dir, directory_mode=None) as output:
            return _load_eligibility_lock(output, eligibility)
    except (RuntimeError, ValueError) as error:
        raise ValueError("eligibility lock directory is unavailable or unsafe") from error


def _load_eligibility_lock(
    output: AnchoredDirectory, eligibility: EligibilityResult
) -> EligibilityLock:
    output.verify_path_identity()
    eligibility_filename = "complete-pair-eligibility.json"
    fold_filename = "participant-folds.csv"
    eligibility_file = output.read_bytes(eligibility_filename, expected_mode=None)
    fold_file = output.read_bytes(fold_filename, expected_mode=None)
    lock = EligibilityLock(
        output_dir=output.path,
        eligibility_filename=eligibility_filename,
        eligibility_file_sha256=eligibility_file.sha256,
        fold_filename=fold_filename,
        fold_file_sha256=fold_file.sha256,
        eligibility_manifest_sha256=eligibility.manifest_sha256,
        output_device=output.device,
        output_inode=output.inode,
    )
    _validate_durable_eligibility_lock(lock, eligibility, output=output)
    return lock


def run_training_mean(
    eligibility: EligibilityResult,
    targets: TargetSet,
    *,
    provenance: RunProvenance,
    synthetic_smoke: bool = False,
) -> PredictionBundle:
    """Cross-fit the training-session-row arithmetic-mean floor."""
    execution_mode = _execution_mode(
        eligibility, targets, provenance, synthetic_requested=synthetic_smoke
    )
    examples = _bind_targets(eligibility, targets)
    model_config = {
        "configuration": "mean",
        "fit_scope": "training_session_rows_per_rotation",
        "estimator": "arithmetic_mean_of_normalised_training_targets",
        "validation_refit": False,
        "prediction_postprocessing": "none",
    }
    model_config_sha256 = _payload_sha256(model_config)
    rows: list[dict[str, Any]] = []
    rotations: list[dict[str, Any]] = []
    artifacts: list[ArtifactPayload] = []
    for fold in range(5):
        train, validation, test = _rotation_examples(examples, fold)
        training_mean = float(np.mean([row.target for row in train], dtype=np.float64))
        fit = ArtifactPayload(
            f"mean.fold-{fold}.fit.npz",
            deterministic_npz_bytes(
                {
                    "schema_version_utf8": np.frombuffer(
                        FIT_ARTIFACT_SCHEMA_VERSION.encode("utf-8"), dtype=np.uint8
                    ),
                    "training_mean": np.asarray(training_mean, dtype="<f8"),
                }
            ),
        )
        artifacts.append(fit)
        validation_prediction = np.full(len(validation), training_mean, dtype=np.float64)
        test_prediction = np.full(len(test), training_mean, dtype=np.float64)
        rotations.append(
            _rotation_manifest(
                fold,
                train,
                validation,
                test,
                fit,
                validation_prediction,
                training_mean=training_mean,
            )
        )
        rows.extend(
            _prediction_rows(
                test,
                test_prediction,
                configuration="mean",
                seed=None,
                model_config_sha256=model_config_sha256,
                model_fit_sha256=fit.sha256,
                eligibility=eligibility,
                targets=targets,
                checkpoint_epoch=None,
                proxy_source="none",
                sampling_mode="not_applicable_deterministic_floor",
                coefficients={},
            )
        )
    return _lock_bundle(
        rows,
        rotations,
        artifacts,
        configuration="mean",
        seed=None,
        model_config=model_config,
        model_config_sha256=model_config_sha256,
        eligibility=eligibility,
        targets=targets,
        provenance=provenance,
        execution_mode=execution_mode,
    )


def run_phase_mean(
    eligibility: EligibilityResult,
    targets: TargetSet,
    *,
    provenance: RunProvenance,
    synthetic_smoke: bool = False,
) -> PredictionBundle:
    """Cross-fit separate pre/post training means as an untuned phase diagnostic."""
    execution_mode = _execution_mode(
        eligibility, targets, provenance, synthetic_requested=synthetic_smoke
    )
    examples = _bind_targets(eligibility, targets)
    model_config = {
        "configuration": "phase_mean",
        "fit_scope": "training_session_rows_separately_by_phase_per_rotation",
        "estimator": "arithmetic_mean_of_normalised_training_targets_within_phase",
        "phase_field": "canonical_session_phase",
        "validation_refit": False,
        "tuning": False,
        "prediction_postprocessing": "none",
    }
    model_config_sha256 = _payload_sha256(model_config)
    rows: list[dict[str, Any]] = []
    rotations: list[dict[str, Any]] = []
    artifacts: list[ArtifactPayload] = []
    for fold in range(5):
        train, validation, test = _rotation_examples(examples, fold)
        phase_means = {
            condition: float(
                np.mean(
                    [
                        row.target
                        for row in train
                        if row.feature.identity.condition == condition
                    ],
                    dtype=np.float64,
                )
            )
            for condition in ("pre", "post")
        }
        validation_prediction = np.asarray(
            [phase_means[row.feature.identity.condition] for row in validation],
            dtype=np.float64,
        )
        test_prediction = np.asarray(
            [phase_means[row.feature.identity.condition] for row in test], dtype=np.float64
        )
        fit = ArtifactPayload(
            f"phase_mean.fold-{fold}.fit.npz",
            deterministic_npz_bytes(
                {
                    "schema_version_utf8": np.frombuffer(
                        FIT_ARTIFACT_SCHEMA_VERSION.encode("utf-8"), dtype=np.uint8
                    ),
                    "training_pre_mean": np.asarray(phase_means["pre"], dtype="<f8"),
                    "training_post_mean": np.asarray(phase_means["post"], dtype="<f8"),
                }
            ),
        )
        artifacts.append(fit)
        rotations.append(
            _rotation_manifest(
                fold,
                train,
                validation,
                test,
                fit,
                validation_prediction,
                training_phase_means=phase_means,
            )
        )
        rows.extend(
            _prediction_rows(
                test,
                test_prediction,
                configuration="phase_mean",
                seed=None,
                model_config_sha256=model_config_sha256,
                model_fit_sha256=fit.sha256,
                eligibility=eligibility,
                targets=targets,
                checkpoint_epoch=None,
                proxy_source="none",
                sampling_mode="not_applicable_deterministic_floor",
                coefficients={
                    "training_pre_mean": phase_means["pre"],
                    "training_post_mean": phase_means["post"],
                },
            )
        )
    return _lock_bundle(
        rows,
        rotations,
        artifacts,
        configuration="phase_mean",
        seed=None,
        model_config=model_config,
        model_config_sha256=model_config_sha256,
        eligibility=eligibility,
        targets=targets,
        provenance=provenance,
        execution_mode=execution_mode,
    )


def run_fixed_ridge(
    eligibility: EligibilityResult,
    targets: TargetSet,
    *,
    provenance: RunProvenance,
    configuration: str = "ridge",
    synthetic_smoke: bool = False,
) -> PredictionBundle:
    """Cross-fit the exact train-scaled, untuned Ridge floor."""
    try:
        modalities = RIDGE_DIAGNOSTIC_MODALITIES[configuration]
    except KeyError as error:
        raise ValueError("unknown fixed ridge diagnostic configuration") from error
    execution_mode = _execution_mode(
        eligibility, targets, provenance, synthetic_requested=synthetic_smoke
    )
    examples = _bind_targets(eligibility, targets)
    model_config = {
        "configuration": configuration,
        "modalities": list(modalities),
        "input_order": [
            {
                "video": "video_1024",
                "audio": "interview_audio_1536",
                "text": "candidate_text_768",
            }[modality]
            for modality in modalities
        ],
        "input_dtype": "float64",
        "preprocessing": "training_session_rows_standard_scaler",
        "estimator": "sklearn.linear_model.Ridge",
        "alpha": 1.0,
        "fit_intercept": True,
        "solver": "lsqr",
        "tol": 1e-6,
        "max_iter": 10_000,
        "tuning": False,
        "validation_refit": False,
        "prediction_postprocessing": "none",
    }
    model_config_sha256 = _payload_sha256(model_config)
    rows: list[dict[str, Any]] = []
    rotations: list[dict[str, Any]] = []
    artifacts: list[ArtifactPayload] = []
    for fold in range(5):
        train, validation, test = _rotation_examples(examples, fold)
        train_x = _concatenated_features(train, dtype=np.float64, modalities=modalities)
        validation_x = _concatenated_features(
            validation, dtype=np.float64, modalities=modalities
        )
        test_x = _concatenated_features(test, dtype=np.float64, modalities=modalities)
        train_y = np.asarray([row.target for row in train], dtype=np.float64)
        scaler = StandardScaler().fit(train_x)
        model = Ridge(
            alpha=1.0,
            fit_intercept=True,
            solver="lsqr",
            tol=1e-6,
            max_iter=10_000,
        ).fit(scaler.transform(train_x), train_y)
        validation_prediction = np.asarray(
            model.predict(scaler.transform(validation_x)), dtype=np.float64
        )
        test_prediction = np.asarray(model.predict(scaler.transform(test_x)), dtype=np.float64)
        fit = ArtifactPayload(
            f"{configuration}.fold-{fold}.fit.npz",
            deterministic_npz_bytes(
                {
                    "schema_version_utf8": np.frombuffer(
                        FIT_ARTIFACT_SCHEMA_VERSION.encode("utf-8"), dtype=np.uint8
                    ),
                    "scaler_mean": np.asarray(scaler.mean_, dtype="<f8"),
                    "scaler_scale": np.asarray(scaler.scale_, dtype="<f8"),
                    "scaler_var": np.asarray(scaler.var_, dtype="<f8"),
                    "scaler_n_samples_seen": np.asarray(scaler.n_samples_seen_, dtype="<i8"),
                    "ridge_coef": np.asarray(model.coef_, dtype="<f8"),
                    "ridge_intercept": np.asarray(model.intercept_, dtype="<f8"),
                    "ridge_n_iter": np.asarray(model.n_iter_, dtype="<i8"),
                }
            ),
        )
        artifacts.append(fit)
        rotations.append(
            _rotation_manifest(
                fold,
                train,
                validation,
                test,
                fit,
                validation_prediction,
                scaler_fit_session_count=len(train),
                estimator_parameters=model_config,
            )
        )
        rows.extend(
            _prediction_rows(
                test,
                test_prediction,
                configuration=configuration,
                seed=None,
                model_config_sha256=model_config_sha256,
                model_fit_sha256=fit.sha256,
                eligibility=eligibility,
                targets=targets,
                checkpoint_epoch=None,
                proxy_source="none",
                sampling_mode="not_applicable_deterministic_floor",
                coefficients={"ridge_alpha": 1.0},
            )
        )
    return _lock_bundle(
        rows,
        rotations,
        artifacts,
        configuration=configuration,
        seed=None,
        model_config=model_config,
        model_config_sha256=model_config_sha256,
        eligibility=eligibility,
        targets=targets,
        provenance=provenance,
        execution_mode=execution_mode,
    )


def run_ordinary_erm(
    eligibility: EligibilityResult,
    targets: TargetSet,
    *,
    seed: int,
    provenance: RunProvenance,
    synthetic_smoke_epoch_cap: int | None = None,
) -> PredictionBundle:
    """Run the frozen five-rotation ordinary-ERM scorer on CPU.

    A reduced epoch cap is accepted only as an explicitly labelled synthetic smoke.  It cannot
    produce a research-mode lock.
    """
    if isinstance(seed, bool) or seed not in {1, 11, 21}:
        raise ValueError("seed must be one of the frozen final seeds 1, 11 or 21")
    if synthetic_smoke_epoch_cap is not None and (
        isinstance(synthetic_smoke_epoch_cap, bool) or not 1 <= synthetic_smoke_epoch_cap <= 200
    ):
        raise ValueError("synthetic_smoke_epoch_cap must be an integer from 1 to 200")
    maximum_epochs = synthetic_smoke_epoch_cap or 200
    execution_mode = _execution_mode(
        eligibility,
        targets,
        provenance,
        synthetic_requested=synthetic_smoke_epoch_cap is not None,
    )
    examples = _bind_targets(eligibility, targets)
    model_config = {
        "configuration": "erm",
        "architecture": "low_capacity_multimodal_scorer_26721_parameters",
        "trainable_parameter_count": 26_721,
        "modality_scalers": "separate_training_session_row_standard_scalers",
        "linear_initialisation": "xavier_uniform_weight_zero_bias",
        "layer_norm_initialisation": "one_weight_zero_bias",
        "participant_batch_size": 16,
        "both_sessions_enter_batch_together": True,
        "last_batch_policy": "retain_smaller_complete_participant_batch",
        "epoch_order": "numeric_participant_ids_then_seeded_numpy_permutation",
        "task_loss": "mse_on_normalised_recommend_hiring",
        "optimizer": {
            "name": "torch.optim.AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-2,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "amsgrad": False,
            "foreach": False,
        },
        "maximum_epochs": 200,
        "early_stopping_patience": 25,
        "checkpoint_metric": "validation_srcc",
        "checkpoint_direction": "maximize",
        "exact_tie_break": "earliest_epoch",
        "gradient_norm_clip": 1.0,
        "prediction_postprocessing": "none_sigmoid_is_model_output",
        "device": "cpu",
    }
    model_config_sha256 = _payload_sha256(model_config)
    rows: list[dict[str, Any]] = []
    rotations: list[dict[str, Any]] = []
    artifacts: list[ArtifactPayload] = []
    for fold in range(5):
        train, validation, test = _rotation_examples(examples, fold)
        result = _fit_erm_rotation(
            train,
            validation,
            test,
            seed=seed,
            maximum_epochs=maximum_epochs,
        )
        fit = ArtifactPayload(f"erm.seed-{seed}.fold-{fold}.fit.npz", result.artifact_bytes)
        artifacts.append(fit)
        rotations.append(
            _rotation_manifest(
                fold,
                train,
                validation,
                test,
                fit,
                result.validation_prediction,
                checkpoint_epoch=result.checkpoint_epoch,
                checkpoint_validation_srcc=result.checkpoint_validation_srcc,
                epochs_completed=result.epochs_completed,
                stopped_early=result.stopped_early,
                training_history=list(result.training_history),
                participant_batch_sizes=list(result.participant_batch_sizes),
                scaler_fit_session_count=len(train),
            )
        )
        rows.extend(
            _prediction_rows(
                test,
                result.test_prediction,
                configuration="erm",
                seed=seed,
                model_config_sha256=model_config_sha256,
                model_fit_sha256=fit.sha256,
                eligibility=eligibility,
                targets=targets,
                checkpoint_epoch=result.checkpoint_epoch,
                proxy_source="none",
                sampling_mode="ordinary_participant_paired",
                coefficients={"lambda_w": 0.0},
            )
        )
    return _lock_bundle(
        rows,
        rotations,
        artifacts,
        configuration="erm",
        seed=seed,
        model_config=model_config,
        model_config_sha256=model_config_sha256,
        eligibility=eligibility,
        targets=targets,
        provenance=provenance,
        execution_mode=execution_mode,
        executed_maximum_epochs=maximum_epochs,
    )


def run_adversarially_reweighted_learning(
    eligibility: EligibilityResult,
    targets: TargetSet,
    *,
    seed: int,
    provenance: RunProvenance,
    synthetic_smoke_epoch_cap: int | None = None,
) -> PredictionBundle:
    """Run the frozen continuous-regression ARL adaptation on CPU.

    A reduced epoch cap is available only for labelled synthetic contract tests and cannot
    produce a research-mode lock.  The production contract always retains the common 200-epoch
    cap and validation-SRCC checkpoint rule.
    """
    if isinstance(seed, bool) or seed not in {1, 11, 21}:
        raise ValueError("seed must be one of the frozen final seeds 1, 11 or 21")
    if synthetic_smoke_epoch_cap is not None and (
        isinstance(synthetic_smoke_epoch_cap, bool) or not 1 <= synthetic_smoke_epoch_cap <= 200
    ):
        raise ValueError("synthetic_smoke_epoch_cap must be an integer from 1 to 200")
    maximum_epochs = synthetic_smoke_epoch_cap or 200
    execution_mode = _execution_mode(
        eligibility,
        targets,
        provenance,
        synthetic_requested=synthetic_smoke_epoch_cap is not None,
    )
    examples = _bind_targets(eligibility, targets)
    model_config = {
        "configuration": "arl",
        "method_status": "transparent_project_adaptation_not_exact_reproduction",
        "architecture": "low_capacity_multimodal_scorer_26721_parameters",
        "learner_trainable_parameter_count": 26_721,
        "adversary": {
            "architecture": "xavier_initialised_linear_25_to_1_zero_bias_then_sigmoid",
            "input": [
                "detached_common_fused_representation_24",
                "detached_normalised_continuous_target_1",
            ],
            "trainable_parameter_count": 26,
            "forbidden_inputs": [
                "prediction",
                "per_example_loss",
                "proxy_group",
                "protected_annotation",
            ],
        },
        "complete_training_system_parameter_count": 26_747,
        "modality_scalers": "separate_training_session_row_standard_scalers",
        "linear_initialisation": "xavier_uniform_weight_zero_bias",
        "layer_norm_initialisation": "one_weight_zero_bias",
        "participant_batch_size": 16,
        "session_rows_per_full_batch": 32,
        "both_sessions_enter_batch_together": True,
        "last_batch_policy": "retain_smaller_complete_participant_batch",
        "epoch_order": "numeric_participant_ids_then_seeded_numpy_permutation",
        "task_loss": "mse_on_normalised_recommend_hiring",
        "weighting": {
            "scope": "current_session_row_minibatch",
            "formula": "1+sigmoid(logit)/max(mean(sigmoid(logit)),1e-4)",
            "denominator_guard": ARL_WEIGHT_DENOMINATOR_GUARD,
            "learner_weight_detached": True,
            "reduction": "mean_of_weight_times_per_row_mse_not_weight_sum",
        },
        "learner_only_minibatch_updates": ARL_PRETRAINING_UPDATES,
        "alternating_update_order": (
            "learner_first_then_recompute_detached_current_representation_and_error_"
            "then_adversary"
        ),
        "optimizer": {
            "learner": "separate_torch.optim.AdamW",
            "adversary": "separate_torch.optim.AdamW",
            "learning_rate": 1e-3,
            "weight_decay": 1e-2,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "amsgrad": False,
            "foreach": False,
        },
        "maximum_epochs": 200,
        "early_stopping_patience": 25,
        "checkpoint_metric": "unweighted_validation_srcc",
        "checkpoint_direction": "maximize",
        "exact_tie_break": "earliest_epoch",
        "gradient_norm_clip_separate": {"learner": 1.0, "adversary": 1.0},
        "diagnostics": {
            "timing_version": ARL_DIAGNOSTIC_TIMING_VERSION,
            "timing": ARL_DIAGNOSTIC_TIMING,
            "minibatch": [
                "learner_and_adversary_objectives",
                "adversary_logit_and_sigmoid_score_mean_min_max",
                "guarded_weight_denominator",
                "weight_mean_min_max",
                "weight_effective_sample_size",
                "weight_error_pearson_or_null",
                "mean_absolute_within_pair_weight_difference",
                "learner_and_adversary_preclip_gradient_norms",
                "phase_and_update_counts",
            ],
            "epoch": ["aggregated_minibatch_diagnostics", "validation_srcc", "validation_mse"],
            "undefined_value": None,
            "serialization": "canonical_json_utf8_in_fit_artifact_and_prediction_manifest",
        },
        "prediction_postprocessing": "none_sigmoid_is_model_output",
        "proxy_fields_accessed": [],
        "protected_annotation_fields_accessed": [],
        "device": "cpu",
    }
    model_config_sha256 = _payload_sha256(model_config)
    rows: list[dict[str, Any]] = []
    rotations: list[dict[str, Any]] = []
    artifacts: list[ArtifactPayload] = []
    for fold in range(5):
        train, validation, test = _rotation_examples(examples, fold)
        result = _fit_arl_rotation(
            train,
            validation,
            test,
            seed=seed,
            maximum_epochs=maximum_epochs,
        )
        fit = ArtifactPayload(f"arl.seed-{seed}.fold-{fold}.fit.npz", result.artifact_bytes)
        artifacts.append(fit)
        rotations.append(
            _rotation_manifest(
                fold,
                train,
                validation,
                test,
                fit,
                result.validation_prediction,
                checkpoint_epoch=result.checkpoint_epoch,
                checkpoint_validation_srcc=result.checkpoint_validation_srcc,
                checkpoint_phase=result.checkpoint_phase,
                checkpoint_learner_updates=result.checkpoint_learner_updates,
                checkpoint_adversary_updates=result.checkpoint_adversary_updates,
                epochs_completed=result.epochs_completed,
                stopped_early=result.stopped_early,
                training_history=list(result.training_history),
                minibatch_diagnostics=list(result.minibatch_diagnostics),
                participant_batch_sizes=list(result.participant_batch_sizes),
                scaler_fit_session_count=len(train),
                learner_trainable_parameter_count=26_721,
                adversary_trainable_parameter_count=26,
                learner_only_minibatch_updates=ARL_PRETRAINING_UPDATES,
                proxy_fields_accessed=[],
                protected_annotation_fields_accessed=[],
            )
        )
        rows.extend(
            _prediction_rows(
                test,
                result.test_prediction,
                configuration="arl",
                seed=seed,
                model_config_sha256=model_config_sha256,
                model_fit_sha256=fit.sha256,
                eligibility=eligibility,
                targets=targets,
                checkpoint_epoch=result.checkpoint_epoch,
                proxy_source="none",
                sampling_mode="ordinary_participant_paired",
                coefficients={
                    "lambda_w": 0.0,
                    "arl_weight_denominator_guard": ARL_WEIGHT_DENOMINATOR_GUARD,
                },
            )
        )
    return _lock_bundle(
        rows,
        rotations,
        artifacts,
        configuration="arl",
        seed=seed,
        model_config=model_config,
        model_config_sha256=model_config_sha256,
        eligibility=eligibility,
        targets=targets,
        provenance=provenance,
        execution_mode=execution_mode,
        executed_maximum_epochs=maximum_epochs,
    )


def write_prediction_bundle(
    output_dir: str | Path | AnchoredDirectory, bundle: PredictionBundle
) -> tuple[Path, ...]:
    """Write a prediction lock immutably, accepting only byte-identical resume."""
    if not isinstance(bundle, PredictionBundle):
        raise TypeError("bundle must be a PredictionBundle")
    _validate_prediction_bundle(bundle)
    if isinstance(output_dir, AnchoredDirectory):
        return _write_prediction_bundle(output_dir, bundle)
    try:
        output = AnchoredDirectory.open(output_dir, directory_mode=None)
    except ValueError:
        output = AnchoredDirectory.open(
            output_dir, create=True, directory_mode=0o700
        )
    with output:
        return _write_prediction_bundle(output, bundle)


def _write_prediction_bundle(
    output: AnchoredDirectory, bundle: PredictionBundle
) -> tuple[Path, ...]:
    output.verify_path_identity()
    if bundle.execution_mode == "research":
        if bundle.eligibility_lock is None:
            raise ValueError("research prediction bundle lacks its prior eligibility lock")
        _validate_lock_files(bundle.eligibility_lock, output=output)
        if (
            output.path != bundle.eligibility_lock.output_dir
            or output.device != bundle.eligibility_lock.output_device
            or output.inode != bundle.eligibility_lock.output_inode
        ):
            raise ValueError(
                "research predictions must be written beside the prior eligibility lock"
            )
    elif bundle.execution_mode != "synthetic_smoke":
        raise ValueError("prediction bundle execution mode is invalid")
    payloads = [*bundle.artifacts]
    payloads.append(ArtifactPayload(bundle.prediction_filename, bundle.prediction_bytes))
    payloads.append(ArtifactPayload(bundle.manifest_filename, bundle.manifest_bytes))
    destinations: list[ArtifactPayload] = []
    seen_destinations: set[str] = set()
    for artifact in payloads:
        if (
            Path(artifact.filename).name != artifact.filename
            or not artifact.filename
            or artifact.filename in seen_destinations
        ):
            raise ValueError("prediction artifact filenames must be safe unique basenames")
        seen_destinations.add(artifact.filename)
        if artifact.filename in output.list_names():
            existing = output.read_bytes(artifact.filename, expected_mode=None)
            if existing.content != artifact.content:
                raise FileExistsError(
                    f"refusing to replace existing prediction artifact {artifact.filename}"
                )
        destinations.append(artifact)
    written: list[Path] = []
    for artifact in destinations:
        output.publish_bytes(artifact.filename, artifact.content, mode=0o600)
        written.append(output.path / artifact.filename)
    output.verify_path_identity()
    return tuple(written)


@dataclass(frozen=True, slots=True)
class _Example:
    feature: FeatureSummary
    target: float


@dataclass(frozen=True, slots=True)
class _ErmFit:
    validation_prediction: np.ndarray
    test_prediction: np.ndarray
    checkpoint_epoch: int
    checkpoint_validation_srcc: float
    epochs_completed: int
    stopped_early: bool
    training_history: tuple[dict[str, float | int], ...]
    participant_batch_sizes: tuple[int, ...]
    artifact_bytes: bytes


@dataclass(frozen=True, slots=True)
class _ArlFit:
    validation_prediction: np.ndarray
    test_prediction: np.ndarray
    checkpoint_epoch: int
    checkpoint_validation_srcc: float
    checkpoint_phase: str
    checkpoint_learner_updates: int
    checkpoint_adversary_updates: int
    epochs_completed: int
    stopped_early: bool
    training_history: tuple[dict[str, float | int | None | str], ...]
    minibatch_diagnostics: tuple[dict[str, Any], ...]
    participant_batch_sizes: tuple[int, ...]
    artifact_bytes: bytes


@dataclass(frozen=True, slots=True)
class _ArlStepDiagnostics:
    phase: str
    learner_loss: float
    unweighted_mse: float
    adversary_loss: float | None
    weight_mean: float
    weight_min: float
    weight_max: float
    adversary_logit_mean: float | None
    adversary_logit_min: float | None
    adversary_logit_max: float | None
    adversary_score_mean: float | None
    adversary_score_min: float | None
    adversary_score_max: float | None
    weight_denominator: float | None
    effective_sample_size: float
    weight_error_pearson: float | None
    mean_absolute_within_pair_weight_difference: float
    learner_gradient_norm: float
    adversary_gradient_norm: float | None


def _fit_erm_rotation(
    train: Sequence[_Example],
    validation: Sequence[_Example],
    test: Sequence[_Example],
    *,
    seed: int,
    maximum_epochs: int,
) -> _ErmFit:
    _set_training_determinism(seed)
    scalers = {
        modality: StandardScaler().fit(_modality_features(train, modality, dtype=np.float64))
        for modality in SUMMARY_DIMENSIONS
    }
    train_values = _scaled_modalities(train, scalers)
    validation_values = _scaled_modalities(validation, scalers)
    test_values = _scaled_modalities(test, scalers)
    train_targets = torch.from_numpy(np.asarray([row.target for row in train], dtype=np.float32))
    validation_targets = np.asarray([row.target for row in validation], dtype=np.float64)

    model = LowCapacityMultimodalScorer().to("cpu")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-2,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
        foreach=False,
    )
    participant_rows = _participant_row_indices(train)
    ordered_participants = np.asarray(sorted(participant_rows, key=int), dtype=object)
    participant_batch_sizes = tuple(
        min(16, int(ordered_participants.size) - start)
        for start in range(0, int(ordered_participants.size), 16)
    )
    generator = np.random.default_rng(seed)
    best_score = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_optimizer: dict[str, Any] | None = None
    without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, maximum_epochs + 1):
        model.train()
        permutation = ordered_participants[generator.permutation(ordered_participants.size)]
        loss_sum = 0.0
        row_count = 0
        for start in range(0, permutation.size, 16):
            batch_participants = permutation[start : start + 16]
            indices = np.asarray(
                [index for pid in batch_participants for index in participant_rows[str(pid)]],
                dtype=np.int64,
            )
            if indices.size != 2 * batch_participants.size:
                raise RuntimeError("participant-paired batch lost one or more sessions")
            prediction = model(
                train_values["video"][indices],
                train_values["audio"][indices],
                train_values["text"][indices],
            ).squeeze(1)
            loss = torch.nn.functional.mse_loss(prediction, train_targets[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * int(indices.size)
            row_count += int(indices.size)

        validation_prediction = _torch_predict(model, validation_values)
        score = _required_correlation(validation_targets, validation_prediction, rank=True)
        history.append(
            {
                "epoch": epoch,
                "training_mse": loss_sum / row_count,
                "validation_srcc": score,
            }
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
            best_optimizer = copy.deepcopy(optimizer.state_dict())
            without_improvement = 0
        else:
            without_improvement += 1
        if without_improvement >= 25:
            break

    if best_state is None or best_optimizer is None:
        raise RuntimeError("ERM training produced no selectable validation checkpoint")
    model.load_state_dict(best_state)
    validation_prediction = _torch_predict(model, validation_values)
    test_prediction = _torch_predict(model, test_values)
    arrays: dict[str, np.ndarray] = {
        "schema_version_utf8": np.frombuffer(
            FIT_ARTIFACT_SCHEMA_VERSION.encode("utf-8"), dtype=np.uint8
        ),
        "checkpoint_epoch": np.asarray(best_epoch, dtype="<i8"),
        "checkpoint_validation_srcc": np.asarray(best_score, dtype="<f8"),
    }
    for modality, scaler in scalers.items():
        arrays[f"scaler.{modality}.mean"] = np.asarray(scaler.mean_, dtype="<f8")
        arrays[f"scaler.{modality}.scale"] = np.asarray(scaler.scale_, dtype="<f8")
        arrays[f"scaler.{modality}.var"] = np.asarray(scaler.var_, dtype="<f8")
        arrays[f"scaler.{modality}.n_samples_seen"] = np.asarray(
            scaler.n_samples_seen_, dtype="<i8"
        )
    for name, value in sorted(best_state.items()):
        arrays[f"model.{name}"] = value.detach().cpu().numpy()
    arrays.update(_optimizer_arrays(best_optimizer))
    return _ErmFit(
        validation_prediction=validation_prediction,
        test_prediction=test_prediction,
        checkpoint_epoch=best_epoch,
        checkpoint_validation_srcc=best_score,
        epochs_completed=len(history),
        stopped_early=len(history) < maximum_epochs,
        training_history=tuple(history),
        participant_batch_sizes=participant_batch_sizes,
        artifact_bytes=deterministic_npz_bytes(arrays),
    )


def _fit_arl_rotation(
    train: Sequence[_Example],
    validation: Sequence[_Example],
    test: Sequence[_Example],
    *,
    seed: int,
    maximum_epochs: int,
) -> _ArlFit:
    """Fit one frozen ARL rotation with participant-paired complete-remainder batches."""
    _set_training_determinism(seed)
    scalers = {
        modality: StandardScaler().fit(_modality_features(train, modality, dtype=np.float64))
        for modality in SUMMARY_DIMENSIONS
    }
    train_values = _scaled_modalities(train, scalers)
    validation_values = _scaled_modalities(validation, scalers)
    test_values = _scaled_modalities(test, scalers)
    train_targets = torch.from_numpy(np.asarray([row.target for row in train], dtype=np.float32))
    validation_targets = np.asarray([row.target for row in validation], dtype=np.float64)

    # Instantiation order is binding: a common seed must give ARL and ERM the same scorer start.
    learner = LowCapacityMultimodalScorer().to("cpu")
    adversary = ARLAdversary().to("cpu")
    learner_optimizer = _adamw(learner.parameters())
    adversary_optimizer = _adamw(adversary.parameters())
    participant_rows = _participant_row_indices(train)
    ordered_participants = np.asarray(sorted(participant_rows, key=int), dtype=object)
    participant_batch_sizes = tuple(
        min(16, int(ordered_participants.size) - start)
        for start in range(0, int(ordered_participants.size), 16)
    )
    generator = np.random.default_rng(seed)
    best_score = -math.inf
    best_epoch = 0
    best_phase = "pretraining"
    best_learner_updates = 0
    best_adversary_updates = 0
    best_learner_state: dict[str, torch.Tensor] | None = None
    best_adversary_state: dict[str, torch.Tensor] | None = None
    best_learner_optimizer: dict[str, Any] | None = None
    best_adversary_optimizer: dict[str, Any] | None = None
    without_improvement = 0
    learner_updates = 0
    adversary_updates = 0
    history: list[dict[str, float | int | None | str]] = []
    minibatch_history: list[dict[str, Any]] = []

    for epoch in range(1, maximum_epochs + 1):
        learner.train()
        adversary.train()
        permutation = ordered_participants[generator.permutation(ordered_participants.size)]
        learner_loss_sum = 0.0
        unweighted_mse_sum = 0.0
        weight_sum = 0.0
        minimum_weight = math.inf
        maximum_weight = -math.inf
        adversary_loss_sum = 0.0
        adversary_batch_count = 0
        pretraining_batch_count = 0
        alternating_batch_count = 0
        row_count = 0
        epoch_diagnostics: list[dict[str, Any]] = []
        for start in range(0, permutation.size, 16):
            batch_participants = permutation[start : start + 16]
            indices = np.asarray(
                [index for pid in batch_participants for index in participant_rows[str(pid)]],
                dtype=np.int64,
            )
            if indices.size != 2 * batch_participants.size:
                raise RuntimeError("participant-paired ARL batch lost one or more sessions")
            learner_updates += 1
            diagnostics = _arl_minibatch_update(
                learner,
                adversary,
                learner_optimizer,
                adversary_optimizer,
                {modality: values[indices] for modality, values in train_values.items()},
                train_targets[indices],
                learner_update=learner_updates,
            )
            rows_in_batch = int(indices.size)
            learner_loss_sum += diagnostics.learner_loss * rows_in_batch
            unweighted_mse_sum += diagnostics.unweighted_mse * rows_in_batch
            weight_sum += diagnostics.weight_mean * rows_in_batch
            minimum_weight = min(minimum_weight, diagnostics.weight_min)
            maximum_weight = max(maximum_weight, diagnostics.weight_max)
            row_count += rows_in_batch
            if diagnostics.phase == "pretraining":
                pretraining_batch_count += 1
            else:
                alternating_batch_count += 1
                adversary_updates += 1
                if diagnostics.adversary_loss is None:
                    raise RuntimeError("alternating ARL update lacks adversary diagnostics")
                adversary_loss_sum += diagnostics.adversary_loss
                adversary_batch_count += 1
            minibatch_record: dict[str, Any] = {
                "diagnostic_timing_version": ARL_DIAGNOSTIC_TIMING_VERSION,
                "diagnostic_timing": ARL_DIAGNOSTIC_TIMING,
                "epoch": epoch,
                "minibatch": len(epoch_diagnostics) + 1,
                "participant_ids": [str(participant_id) for participant_id in batch_participants],
                "participant_count": int(batch_participants.size),
                "session_row_count": rows_in_batch,
                "phase": diagnostics.phase,
                "learner_objective": diagnostics.learner_loss,
                "unweighted_mse": diagnostics.unweighted_mse,
                "adversary_objective": diagnostics.adversary_loss,
                "example_weight_mean": diagnostics.weight_mean,
                "example_weight_minimum": diagnostics.weight_min,
                "example_weight_maximum": diagnostics.weight_max,
                "adversary_logit_mean": diagnostics.adversary_logit_mean,
                "adversary_logit_minimum": diagnostics.adversary_logit_min,
                "adversary_logit_maximum": diagnostics.adversary_logit_max,
                "adversary_sigmoid_score_mean": diagnostics.adversary_score_mean,
                "adversary_sigmoid_score_minimum": diagnostics.adversary_score_min,
                "adversary_sigmoid_score_maximum": diagnostics.adversary_score_max,
                "guarded_weight_denominator": diagnostics.weight_denominator,
                "weight_effective_sample_size": diagnostics.effective_sample_size,
                "weight_error_pearson": diagnostics.weight_error_pearson,
                "mean_absolute_within_pair_weight_difference": (
                    diagnostics.mean_absolute_within_pair_weight_difference
                ),
                "learner_preclip_gradient_norm": diagnostics.learner_gradient_norm,
                "adversary_preclip_gradient_norm": diagnostics.adversary_gradient_norm,
                "learner_updates_total": learner_updates,
                "adversary_updates_total": adversary_updates,
            }
            epoch_diagnostics.append(minibatch_record)
            minibatch_history.append(minibatch_record)

        validation_prediction = _torch_predict(learner, validation_values)
        score = _required_correlation(validation_targets, validation_prediction, rank=True)
        validation_mse = float(
            np.mean(np.square(validation_prediction - validation_targets), dtype=np.float64)
        )
        checkpoint_phase = (
            "pretraining" if learner_updates <= ARL_PRETRAINING_UPDATES else "alternating"
        )
        history.append(
            {
                "epoch": epoch,
                "phase_at_epoch_end": checkpoint_phase,
                "learner_weighted_objective": learner_loss_sum / row_count,
                "unweighted_training_mse": unweighted_mse_sum / row_count,
                "mean_example_weight": weight_sum / row_count,
                "minimum_example_weight": minimum_weight,
                "maximum_example_weight": maximum_weight,
                "mean_adversary_objective": (
                    adversary_loss_sum / adversary_batch_count
                    if adversary_batch_count
                    else None
                ),
                "pretraining_minibatches": pretraining_batch_count,
                "alternating_minibatches": alternating_batch_count,
                "learner_updates_total": learner_updates,
                "adversary_updates_total": adversary_updates,
                "validation_srcc": score,
                "validation_mse": validation_mse,
                "adversary_logit_mean": _weighted_diagnostic_mean(
                    epoch_diagnostics, "adversary_logit_mean", "session_row_count"
                ),
                "adversary_logit_minimum": _defined_diagnostic_extreme(
                    epoch_diagnostics, "adversary_logit_minimum", minimum=True
                ),
                "adversary_logit_maximum": _defined_diagnostic_extreme(
                    epoch_diagnostics, "adversary_logit_maximum", minimum=False
                ),
                "adversary_sigmoid_score_mean": _weighted_diagnostic_mean(
                    epoch_diagnostics, "adversary_sigmoid_score_mean", "session_row_count"
                ),
                "adversary_sigmoid_score_minimum": _defined_diagnostic_extreme(
                    epoch_diagnostics, "adversary_sigmoid_score_minimum", minimum=True
                ),
                "adversary_sigmoid_score_maximum": _defined_diagnostic_extreme(
                    epoch_diagnostics, "adversary_sigmoid_score_maximum", minimum=False
                ),
                "guarded_weight_denominator_mean": _defined_diagnostic_mean(
                    epoch_diagnostics, "guarded_weight_denominator"
                ),
                "guarded_weight_denominator_minimum": _defined_diagnostic_extreme(
                    epoch_diagnostics, "guarded_weight_denominator", minimum=True
                ),
                "guarded_weight_denominator_maximum": _defined_diagnostic_extreme(
                    epoch_diagnostics, "guarded_weight_denominator", minimum=False
                ),
                "weight_effective_sample_size_mean": _defined_diagnostic_mean(
                    epoch_diagnostics, "weight_effective_sample_size"
                ),
                "weight_effective_sample_size_minimum": _defined_diagnostic_extreme(
                    epoch_diagnostics, "weight_effective_sample_size", minimum=True
                ),
                "weight_effective_sample_size_maximum": _defined_diagnostic_extreme(
                    epoch_diagnostics, "weight_effective_sample_size", minimum=False
                ),
                "weight_error_pearson_mean": _defined_diagnostic_mean(
                    epoch_diagnostics, "weight_error_pearson"
                ),
                "weight_error_pearson_defined_minibatches": sum(
                    row["weight_error_pearson"] is not None for row in epoch_diagnostics
                ),
                "mean_absolute_within_pair_weight_difference": _weighted_diagnostic_mean(
                    epoch_diagnostics,
                    "mean_absolute_within_pair_weight_difference",
                    "participant_count",
                ),
                "learner_preclip_gradient_norm_mean": _defined_diagnostic_mean(
                    epoch_diagnostics, "learner_preclip_gradient_norm"
                ),
                "learner_preclip_gradient_norm_maximum": _defined_diagnostic_extreme(
                    epoch_diagnostics, "learner_preclip_gradient_norm", minimum=False
                ),
                "adversary_preclip_gradient_norm_mean": _defined_diagnostic_mean(
                    epoch_diagnostics, "adversary_preclip_gradient_norm"
                ),
                "adversary_preclip_gradient_norm_maximum": _defined_diagnostic_extreme(
                    epoch_diagnostics, "adversary_preclip_gradient_norm", minimum=False
                ),
            }
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_phase = checkpoint_phase
            best_learner_updates = learner_updates
            best_adversary_updates = adversary_updates
            best_learner_state = {
                name: value.detach().clone() for name, value in learner.state_dict().items()
            }
            best_adversary_state = {
                name: value.detach().clone() for name, value in adversary.state_dict().items()
            }
            best_learner_optimizer = copy.deepcopy(learner_optimizer.state_dict())
            best_adversary_optimizer = copy.deepcopy(adversary_optimizer.state_dict())
            without_improvement = 0
        else:
            without_improvement += 1
        if without_improvement >= 25:
            break

    if (
        best_learner_state is None
        or best_adversary_state is None
        or best_learner_optimizer is None
        or best_adversary_optimizer is None
    ):
        raise RuntimeError("ARL training produced no selectable validation checkpoint")
    learner.load_state_dict(best_learner_state)
    adversary.load_state_dict(best_adversary_state)
    learner_optimizer.load_state_dict(best_learner_optimizer)
    adversary_optimizer.load_state_dict(best_adversary_optimizer)
    validation_prediction = _torch_predict(learner, validation_values)
    test_prediction = _torch_predict(learner, test_values)
    arrays: dict[str, np.ndarray] = {
        "schema_version_utf8": np.frombuffer(
            FIT_ARTIFACT_SCHEMA_VERSION.encode("utf-8"), dtype=np.uint8
        ),
        "configuration_utf8": np.frombuffer(b"arl", dtype=np.uint8),
        "checkpoint_epoch": np.asarray(best_epoch, dtype="<i8"),
        "checkpoint_validation_srcc": np.asarray(best_score, dtype="<f8"),
        "checkpoint_phase_utf8": np.frombuffer(best_phase.encode("utf-8"), dtype=np.uint8),
        "checkpoint_learner_updates": np.asarray(best_learner_updates, dtype="<i8"),
        "checkpoint_adversary_updates": np.asarray(best_adversary_updates, dtype="<i8"),
        "learner_only_minibatch_updates": np.asarray(ARL_PRETRAINING_UPDATES, dtype="<i8"),
        "learner_trainable_parameter_count": np.asarray(26_721, dtype="<i8"),
        "adversary_trainable_parameter_count": np.asarray(26, dtype="<i8"),
        "protected_annotation_field_count": np.asarray(0, dtype="<i8"),
        "proxy_field_count": np.asarray(0, dtype="<i8"),
        "diagnostic_timing_version_utf8": np.frombuffer(
            ARL_DIAGNOSTIC_TIMING_VERSION.encode("utf-8"), dtype=np.uint8
        ),
        "diagnostic_timing_utf8": np.frombuffer(
            ARL_DIAGNOSTIC_TIMING.encode("utf-8"), dtype=np.uint8
        ),
        "training_history_json_utf8": np.frombuffer(
            canonical_json_bytes({"training_history": history}), dtype=np.uint8
        ),
        "minibatch_diagnostics_json_utf8": np.frombuffer(
            canonical_json_bytes({"minibatch_diagnostics": minibatch_history}), dtype=np.uint8
        ),
    }
    for modality, scaler in scalers.items():
        arrays[f"scaler.{modality}.mean"] = np.asarray(scaler.mean_, dtype="<f8")
        arrays[f"scaler.{modality}.scale"] = np.asarray(scaler.scale_, dtype="<f8")
        arrays[f"scaler.{modality}.var"] = np.asarray(scaler.var_, dtype="<f8")
        arrays[f"scaler.{modality}.n_samples_seen"] = np.asarray(
            scaler.n_samples_seen_, dtype="<i8"
        )
    for name, value in sorted(best_learner_state.items()):
        arrays[f"learner.{name}"] = value.detach().cpu().numpy()
    for name, value in sorted(best_adversary_state.items()):
        arrays[f"adversary.{name}"] = value.detach().cpu().numpy()
    arrays.update(_optimizer_arrays(best_learner_optimizer, prefix="learner_optimizer"))
    arrays.update(_optimizer_arrays(best_adversary_optimizer, prefix="adversary_optimizer"))
    return _ArlFit(
        validation_prediction=validation_prediction,
        test_prediction=test_prediction,
        checkpoint_epoch=best_epoch,
        checkpoint_validation_srcc=best_score,
        checkpoint_phase=best_phase,
        checkpoint_learner_updates=best_learner_updates,
        checkpoint_adversary_updates=best_adversary_updates,
        epochs_completed=len(history),
        stopped_early=len(history) < maximum_epochs,
        training_history=tuple(history),
        minibatch_diagnostics=tuple(minibatch_history),
        participant_batch_sizes=participant_batch_sizes,
        artifact_bytes=deterministic_npz_bytes(arrays),
    )


def _defined_diagnostic_mean(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row[field] is not None]
    return math.fsum(values) / len(values) if values else None


def _weighted_diagnostic_mean(
    rows: Sequence[Mapping[str, Any]], field: str, weight_field: str
) -> float | None:
    values = [
        (float(row[field]), float(row[weight_field]))
        for row in rows
        if row[field] is not None
    ]
    if not values:
        return None
    weight_sum = math.fsum(weight for _value, weight in values)
    return math.fsum(value * weight for value, weight in values) / weight_sum


def _defined_diagnostic_extreme(
    rows: Sequence[Mapping[str, Any]], field: str, *, minimum: bool
) -> float | None:
    values = [float(row[field]) for row in rows if row[field] is not None]
    if not values:
        return None
    return min(values) if minimum else max(values)


def _arl_minibatch_update(
    learner: LowCapacityMultimodalScorer,
    adversary: ARLAdversary,
    learner_optimizer: torch.optim.Optimizer,
    adversary_optimizer: torch.optim.Optimizer,
    values: Mapping[str, torch.Tensor],
    targets: torch.Tensor,
    *,
    learner_update: int,
) -> _ArlStepDiagnostics:
    """Apply one learner update and, after pretraining, one isolated adversary update."""
    if (
        isinstance(learner_update, bool)
        or not isinstance(learner_update, int)
        or learner_update < 1
    ):
        raise ValueError("learner_update must be a positive integer")
    if targets.ndim != 1 or targets.numel() == 0 or not bool(torch.isfinite(targets).all()):
        raise ValueError("targets must be one non-empty finite vector")
    if targets.numel() % 2:
        raise ValueError("ARL minibatches must contain complete two-session participant pairs")
    learner.train()
    adversary.train()
    learner_optimizer.zero_grad(set_to_none=True)
    adversary_optimizer.zero_grad(set_to_none=True)
    fused = learner.fused_representation(values["video"], values["audio"], values["text"])
    prediction = torch.sigmoid(learner.head(fused)).squeeze(1)
    per_row_error = torch.square(prediction - targets)
    if learner_update <= ARL_PRETRAINING_UPDATES:
        phase = "pretraining"
        weights = torch.ones_like(per_row_error)
        adversary_logits: torch.Tensor | None = None
        adversary_scores: torch.Tensor | None = None
        weight_denominator: torch.Tensor | None = None
        learner_loss = per_row_error.mean()
    else:
        phase = "alternating"
        with torch.no_grad():
            adversary_logits = adversary(fused.detach(), targets)
            weights, adversary_scores, weight_denominator = _arl_weight_components(
                adversary_logits
            )
            weights = weights.detach()
        learner_loss = torch.mean(weights * per_row_error)
    detached_weights = weights.detach()
    detached_error = per_row_error.detach()
    weight_sum = detached_weights.sum()
    effective_sample_size = float(
        torch.square(weight_sum) / torch.square(detached_weights).sum()
    )
    weight_error_pearson = _optional_correlation(
        detached_weights.cpu().numpy().astype(np.float64),
        detached_error.cpu().numpy().astype(np.float64),
        rank=False,
    )
    within_pair_difference = float(
        torch.abs(detached_weights[0::2] - detached_weights[1::2]).mean()
    )
    learner_loss.backward()
    learner_gradient_norm = float(torch.nn.utils.clip_grad_norm_(learner.parameters(), 1.0))
    learner_optimizer.step()
    learner_optimizer.zero_grad(set_to_none=True)

    adversary_loss_value: float | None = None
    adversary_gradient_norm: float | None = None
    if phase == "alternating":
        with torch.no_grad():
            current_fused = learner.fused_representation(
                values["video"], values["audio"], values["text"]
            )
            current_prediction = torch.sigmoid(learner.head(current_fused)).squeeze(1)
            current_error = torch.square(current_prediction - targets).detach()
        adversary_weights, _scores, _denominator = _arl_weight_components(
            adversary(current_fused.detach(), targets)
        )
        adversary_loss = -torch.mean(adversary_weights * current_error)
        adversary_loss.backward()
        adversary_gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(adversary.parameters(), 1.0)
        )
        adversary_optimizer.step()
        adversary_loss_value = float(adversary_loss.detach())
        adversary_optimizer.zero_grad(set_to_none=True)

    return _ArlStepDiagnostics(
        phase=phase,
        learner_loss=float(learner_loss.detach()),
        unweighted_mse=float(per_row_error.detach().mean()),
        adversary_loss=adversary_loss_value,
        weight_mean=float(weights.mean()),
        weight_min=float(weights.min()),
        weight_max=float(weights.max()),
        adversary_logit_mean=(
            float(adversary_logits.mean()) if adversary_logits is not None else None
        ),
        adversary_logit_min=(
            float(adversary_logits.min()) if adversary_logits is not None else None
        ),
        adversary_logit_max=(
            float(adversary_logits.max()) if adversary_logits is not None else None
        ),
        adversary_score_mean=(
            float(adversary_scores.mean()) if adversary_scores is not None else None
        ),
        adversary_score_min=(
            float(adversary_scores.min()) if adversary_scores is not None else None
        ),
        adversary_score_max=(
            float(adversary_scores.max()) if adversary_scores is not None else None
        ),
        weight_denominator=(
            float(weight_denominator) if weight_denominator is not None else None
        ),
        effective_sample_size=effective_sample_size,
        weight_error_pearson=weight_error_pearson,
        mean_absolute_within_pair_weight_difference=within_pair_difference,
        learner_gradient_norm=learner_gradient_norm,
        adversary_gradient_norm=adversary_gradient_norm,
    )


def _adamw(parameters: Any) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        parameters,
        lr=1e-3,
        weight_decay=1e-2,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
        foreach=False,
    )


def _optimizer_arrays(
    state_dict: Mapping[str, Any], *, prefix: str = "optimizer"
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    states = state_dict.get("state")
    if not isinstance(states, Mapping):
        raise RuntimeError("optimizer state is unavailable")
    parameter_groups = state_dict.get("param_groups")
    if not isinstance(parameter_groups, list):
        raise RuntimeError("optimizer parameter groups are unavailable")
    arrays[f"{prefix}.state_count"] = np.asarray(len(states), dtype="<i8")
    arrays[f"{prefix}.param_groups_json_utf8"] = np.frombuffer(
        canonical_json_bytes({"param_groups": parameter_groups}), dtype=np.uint8
    )
    for parameter_index in sorted(states):
        state = states[parameter_index]
        if not isinstance(state, Mapping):
            raise RuntimeError("optimizer parameter state is malformed")
        for name in sorted(state):
            value = state[name]
            key = f"{prefix}.parameter-{int(parameter_index):04d}.{name}"
            if isinstance(value, torch.Tensor):
                arrays[key] = value.detach().cpu().numpy()
            elif isinstance(value, (int, float)):
                arrays[key] = np.asarray(value)
            else:
                raise RuntimeError("optimizer state contains an unsupported value")
    return arrays


def _set_training_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _torch_predict(
    model: LowCapacityMultimodalScorer, values: Mapping[str, torch.Tensor]
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        result = model(values["video"], values["audio"], values["text"]).squeeze(1)
    output = result.detach().cpu().numpy().astype(np.float64)
    if output.ndim != 1 or not np.isfinite(output).all() or np.any((output < 0) | (output > 1)):
        raise RuntimeError("ERM produced invalid sigmoid predictions")
    return output


def _scaled_modalities(
    rows: Sequence[_Example], scalers: Mapping[str, StandardScaler]
) -> dict[str, torch.Tensor]:
    return {
        modality: torch.from_numpy(
            np.asarray(
                scalers[modality].transform(_modality_features(rows, modality, dtype=np.float64)),
                dtype=np.float32,
            )
        )
        for modality in SUMMARY_DIMENSIONS
    }


def _participant_row_indices(rows: Sequence[_Example]) -> dict[str, tuple[int, int]]:
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(row.feature.identity.participant_id, []).append(index)
    result: dict[str, tuple[int, int]] = {}
    for participant_id, indices in grouped.items():
        if len(indices) != 2:
            raise ValueError("every training participant must contribute exactly two sessions")
        conditions = {rows[index].feature.identity.condition for index in indices}
        if conditions != {"pre", "post"}:
            raise ValueError("every training batch pair must contain pre and post sessions")
        result[participant_id] = (indices[0], indices[1])
    return result


def _prediction_rows(
    rows: Sequence[_Example],
    predictions: np.ndarray,
    *,
    configuration: str,
    seed: int | None,
    model_config_sha256: str,
    model_fit_sha256: str,
    eligibility: EligibilityResult,
    targets: TargetSet,
    checkpoint_epoch: int | None,
    proxy_source: str,
    sampling_mode: str,
    coefficients: Mapping[str, float],
) -> list[dict[str, Any]]:
    values = np.asarray(predictions, dtype=np.float64)
    if values.shape != (len(rows),) or not np.isfinite(values).all():
        raise ValueError("predictions must be one finite value per test session")
    return [
        {
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "corpus": CORPUS_ID,
            "participant_id": row.feature.identity.participant_id,
            "session_id": row.feature.identity.session_id,
            "source_session_id": row.feature.identity.source_session_id,
            "session_phase": row.feature.identity.condition,
            "fixed_fold": row.feature.identity.fold,
            "outer_rotation": row.feature.identity.fold,
            "rotation_role": "test",
            "seed": seed,
            "configuration": configuration,
            "model_config_sha256": model_config_sha256,
            "model_fit_sha256": model_fit_sha256,
            "target_transform": targets.manifest["transform"],
            "target_manifest_sha256": targets.manifest_sha256,
            "target": row.target,
            "prediction": float(prediction),
            "encoder_revisions": {
                modality: contract["revision"] for modality, contract in ENCODER_CONTRACT.items()
            },
            "feature_contract_sha256": eligibility.feature_contract_sha256,
            "preprocessing_sha256": eligibility.preprocessing_sha256,
            "proxy_source": proxy_source,
            "sampling_mode": sampling_mode,
            "coefficients": dict(coefficients),
            "checkpoint_epoch": checkpoint_epoch,
            "firewall_stage": FIREWALL_STAGE,
            "eligibility_manifest_sha256": eligibility.manifest_sha256,
            "session_mapping": eligibility.manifest["session_mapping"],
        }
        for row, prediction in zip(rows, values, strict=True)
    ]


def _lock_bundle(
    rows: list[dict[str, Any]],
    rotations: list[dict[str, Any]],
    artifacts: list[ArtifactPayload],
    *,
    configuration: str,
    seed: int | None,
    model_config: Mapping[str, Any],
    model_config_sha256: str,
    eligibility: EligibilityResult,
    targets: TargetSet,
    provenance: RunProvenance,
    execution_mode: str,
    executed_maximum_epochs: int | None = None,
) -> PredictionBundle:
    expected = 2 * len(eligibility.participant_ids)
    if len(rows) != expected:
        raise RuntimeError(f"OOF row count must equal 2E={expected}; observed {len(rows)}")
    keys = [(row["seed"], row["session_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("OOF predictions contain a duplicate session/seed row")
    expected_folds = {
        feature.identity.session_id: feature.identity.fold for feature in eligibility.features
    }
    if {row["session_id"] for row in rows} != set(expected_folds) or any(
        row["fixed_fold"] != expected_folds[row["session_id"]]
        or row["outer_rotation"] != row["fixed_fold"]
        for row in rows
    ):
        raise RuntimeError("OOF rows do not cover each eligible test session exactly once")

    support_artifacts = [
        ArtifactPayload("complete-pair-eligibility.json", eligibility.manifest_bytes),
        ArtifactPayload("participant-folds.csv", eligibility.fold_manifest_bytes),
        ArtifactPayload("target-manifest.json", targets.manifest_bytes),
    ]
    all_artifacts = [*support_artifacts, *artifacts]
    artifact_records = [
        {"filename": artifact.filename, "sha256": artifact.sha256, "bytes": len(artifact.content)}
        for artifact in all_artifacts
    ]
    lock_identity = {
        "schema_version": PREDICTION_LOCK_SCHEMA_VERSION,
        "corpus": CORPUS_ID,
        "configuration": configuration,
        "seed": seed,
        "execution_mode": execution_mode,
        "eligibility_manifest_sha256": eligibility.manifest_sha256,
        "fold_manifest_sha256": eligibility.fold_manifest_sha256,
        "target_manifest_sha256": targets.manifest_sha256,
        "session_mapping": eligibility.manifest["session_mapping"],
        "feature_contract_sha256": eligibility.feature_contract_sha256,
        "preprocessing_sha256": eligibility.preprocessing_sha256,
        "feature_run_contract_sha256": eligibility.manifest[
            "feature_run_contract_sha256"
        ],
        "feature_config_sha256": eligibility.feature_config_sha256,
        "feature_extraction_closure_file_sha256": eligibility.manifest[
            "feature_extraction_closure_file_sha256"
        ],
        "feature_extraction_closure_payload_sha256": eligibility.manifest[
            "feature_extraction_closure_payload_sha256"
        ],
        "config_sha256": provenance.config_sha256,
        "code_commit_sha": provenance.code_commit_sha,
        "code_source_tree_sha256": provenance.code_source_tree_sha256,
        "model_config_sha256": model_config_sha256,
        "artifacts": artifact_records,
    }
    lock_id = _payload_sha256(lock_identity)
    locked_rows = tuple({**row, "prediction_lock_id": lock_id} for row in rows)
    prediction_filename = (
        f"{configuration}.predictions.json"
        if seed is None
        else f"{configuration}.seed-{seed}.predictions.json"
    )
    manifest_filename = (
        f"{configuration}.prediction-lock.json"
        if seed is None
        else f"{configuration}.seed-{seed}.prediction-lock.json"
    )
    prediction_payload = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "prediction_lock_id": lock_id,
        "rows": list(locked_rows),
    }
    prediction_bytes = canonical_json_bytes(prediction_payload)
    predictions_sha256 = hashlib.sha256(prediction_bytes).hexdigest()
    metric_payload = utility_metrics(
        np.asarray([row["target"] for row in locked_rows], dtype=np.float64),
        np.asarray([row["prediction"] for row in locked_rows], dtype=np.float64),
    )
    unsigned_manifest = {
        **lock_identity,
        "prediction_lock_id": lock_id,
        "state": "locked",
        "firewall_stage": FIREWALL_STAGE,
        "protected_annotation_fields_accessed": [],
        "source_date_epoch": provenance.source_date_epoch,
        "target": {
            "manifest_sha256": targets.manifest_sha256,
            "construction_mode": targets.manifest["construction_mode"],
            "construct": targets.manifest.get("construct", "synthetic_contract_test"),
            "source": targets.manifest["source"],
            "transform": targets.manifest["transform"],
            "session_mapping": targets.manifest.get("session_mapping"),
        },
        "encoder_contract": ENCODER_CONTRACT,
        "model_config": dict(model_config),
        "executed_maximum_epochs": executed_maximum_epochs,
        "eligible_participant_count": len(eligibility.participant_ids),
        "expected_oof_session_count": expected,
        "observed_oof_session_count": len(locked_rows),
        "rotations": rotations,
        "utility": metric_payload,
        "runtime": _runtime_versions(
            include_torch=configuration in FIREWALLED_LEARNED_CONFIGURATIONS
        ),
        "predictions": {
            "filename": prediction_filename,
            "sha256": predictions_sha256,
            "row_count": len(locked_rows),
        },
    }
    manifest_sha256 = hashlib.sha256(canonical_json_bytes(unsigned_manifest)).hexdigest()
    manifest = {**unsigned_manifest, "manifest_payload_sha256": manifest_sha256}
    return PredictionBundle(
        configuration=configuration,
        seed=seed,
        prediction_filename=prediction_filename,
        manifest_filename=manifest_filename,
        predictions=locked_rows,
        prediction_bytes=prediction_bytes,
        manifest=manifest,
        manifest_bytes=canonical_json_bytes(manifest),
        manifest_sha256=manifest_sha256,
        artifacts=tuple(all_artifacts),
        execution_mode=execution_mode,
        eligibility_lock=targets.eligibility_lock if execution_mode == "research" else None,
    )


def _validate_prediction_bundle(bundle: PredictionBundle) -> None:
    if bundle.configuration not in {
        *DIAGNOSTIC_FLOOR_CONFIGURATIONS,
        *FIREWALLED_LEARNED_CONFIGURATIONS,
    }:
        raise ValueError("prediction bundle configuration is invalid")
    if bundle.execution_mode not in {"research", "synthetic_smoke"}:
        raise ValueError("prediction bundle execution mode is invalid")
    if bundle.configuration in FIREWALLED_LEARNED_CONFIGURATIONS:
        if isinstance(bundle.seed, bool) or bundle.seed not in {1, 11, 21}:
            raise ValueError("learned prediction bundle seed is invalid")
    elif bundle.seed is not None:
        raise ValueError("deterministic floor prediction bundles may not have a seed")
    expected_prediction_filename = (
        f"{bundle.configuration}.predictions.json"
        if bundle.seed is None
        else f"{bundle.configuration}.seed-{bundle.seed}.predictions.json"
    )
    expected_manifest_filename = (
        f"{bundle.configuration}.prediction-lock.json"
        if bundle.seed is None
        else f"{bundle.configuration}.seed-{bundle.seed}.prediction-lock.json"
    )
    if (
        bundle.prediction_filename != expected_prediction_filename
        or bundle.manifest_filename != expected_manifest_filename
    ):
        raise ValueError("prediction bundle filenames disagree with configuration and seed")
    _validate_signed_manifest(
        bundle.manifest,
        bundle.manifest_bytes,
        bundle.manifest_sha256,
        label="prediction lock",
    )
    manifest = bundle.manifest
    if (
        manifest.get("configuration") != bundle.configuration
        or manifest.get("seed") != bundle.seed
        or manifest.get("execution_mode") != bundle.execution_mode
        or manifest.get("state") != "locked"
        or manifest.get("protected_annotation_fields_accessed") != []
    ):
        raise ValueError("prediction bundle fields disagree with its lock manifest")
    mapping_provenance = _validate_mapping_provenance(
        manifest.get("session_mapping"), required=bundle.execution_mode == "research"
    )
    for name in (
        "feature_extraction_closure_file_sha256",
        "feature_extraction_closure_payload_sha256",
    ):
        value = manifest.get(name)
        if bundle.execution_mode == "research":
            _require_hex(value, 64, name)
        elif value is not None:
            raise ValueError("synthetic prediction lock may not claim a feature closure")

    try:
        prediction_payload = _loads_unique_json(bundle.prediction_bytes)
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("prediction payload cannot be read") from error
    if canonical_json_bytes(prediction_payload) != bundle.prediction_bytes:
        raise ValueError("prediction payload is not canonical")
    payload_rows = prediction_payload.get("rows")
    if (
        prediction_payload.get("schema_version") != PREDICTION_SCHEMA_VERSION
        or prediction_payload.get("prediction_lock_id") != manifest.get("prediction_lock_id")
        or not isinstance(payload_rows, list)
        or tuple(payload_rows) != bundle.predictions
    ):
        raise ValueError("prediction bytes disagree with the in-memory rows or lock")
    prediction_record = manifest.get("predictions")
    if (
        not isinstance(prediction_record, Mapping)
        or prediction_record.get("filename") != bundle.prediction_filename
        or prediction_record.get("sha256") != hashlib.sha256(bundle.prediction_bytes).hexdigest()
        or prediction_record.get("row_count") != len(bundle.predictions)
        or any(
            not isinstance(row, Mapping)
            or row.get("prediction_lock_id") != manifest.get("prediction_lock_id")
            for row in bundle.predictions
        )
    ):
        raise ValueError("prediction payload checksum, count or lock linkage failed")

    if not isinstance(bundle.artifacts, tuple) or not bundle.artifacts:
        raise ValueError("prediction bundle artifacts are unavailable")
    filenames: set[str] = set()
    observed_artifacts: list[dict[str, Any]] = []
    artifacts_by_name: dict[str, ArtifactPayload] = {}
    for artifact in bundle.artifacts:
        if (
            not isinstance(artifact, ArtifactPayload)
            or not artifact.filename
            or Path(artifact.filename).name != artifact.filename
            or artifact.filename in filenames
            or not isinstance(artifact.content, bytes)
        ):
            raise ValueError("prediction bundle artifact is invalid or duplicated")
        filenames.add(artifact.filename)
        artifacts_by_name[artifact.filename] = artifact
        observed_artifacts.append(
            {
                "filename": artifact.filename,
                "sha256": artifact.sha256,
                "bytes": len(artifact.content),
            }
        )
    if manifest.get("artifacts") != observed_artifacts:
        raise ValueError("prediction bundle artifacts disagree with the lock manifest")
    required_support = {
        "complete-pair-eligibility.json",
        "participant-folds.csv",
        "target-manifest.json",
    }
    if not required_support.issubset(artifacts_by_name):
        raise ValueError("prediction bundle lacks required support artifacts")
    eligibility_artifact = artifacts_by_name["complete-pair-eligibility.json"]
    target_artifact = artifacts_by_name["target-manifest.json"]
    try:
        eligibility_manifest = _loads_unique_json(eligibility_artifact.content)
        target_manifest = _loads_unique_json(target_artifact.content)
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("prediction support manifest cannot be read") from error
    _validate_signed_manifest(
        eligibility_manifest,
        eligibility_artifact.content,
        manifest.get("eligibility_manifest_sha256"),
        label="bundled eligibility",
    )
    _validate_signed_manifest(
        target_manifest,
        target_artifact.content,
        manifest.get("target_manifest_sha256"),
        label="bundled target",
    )
    fold_artifact = artifacts_by_name["participant-folds.csv"]
    fold_sha256 = hashlib.sha256(fold_artifact.content).hexdigest()
    if fold_sha256 != manifest.get("fold_manifest_sha256"):
        raise ValueError("bundled fold manifest checksum failed")
    if (
        manifest.get("corpus") != CORPUS_ID
        or eligibility_manifest.get("schema_version") != ELIGIBILITY_SCHEMA_VERSION
        or eligibility_manifest.get("corpus") != CORPUS_ID
        or eligibility_manifest.get("fold_manifest_sha256") != fold_sha256
        or eligibility_manifest.get("feature_contract_sha256")
        != manifest.get("feature_contract_sha256")
        or eligibility_manifest.get("preprocessing_sha256") != manifest.get("preprocessing_sha256")
        or eligibility_manifest.get("feature_run_contract_sha256")
        != manifest.get("feature_run_contract_sha256")
        or eligibility_manifest.get("feature_config_sha256")
        != manifest.get("feature_config_sha256")
        or eligibility_manifest.get("feature_extraction_closure_file_sha256")
        != manifest.get("feature_extraction_closure_file_sha256")
        or eligibility_manifest.get("feature_extraction_closure_payload_sha256")
        != manifest.get("feature_extraction_closure_payload_sha256")
        or eligibility_manifest.get("outcome_fields_accessed") != []
        or eligibility_manifest.get("protected_annotation_fields_accessed") != []
        or eligibility_manifest.get("session_mapping") != mapping_provenance
    ):
        raise ValueError("bundled eligibility provenance does not match the prediction lock")
    target_mapping = target_manifest.get("session_mapping")
    if target_mapping != mapping_provenance:
        raise ValueError("bundled target and eligibility session mappings disagree")
    expected_target_summary = {
        "manifest_sha256": manifest.get("target_manifest_sha256"),
        "construction_mode": target_manifest.get("construction_mode"),
        "construct": target_manifest.get("construct", "synthetic_contract_test"),
        "source": target_manifest.get("source"),
        "transform": target_manifest.get("transform"),
        "session_mapping": target_mapping,
    }
    if (
        target_manifest.get("schema_version") != TARGET_SCHEMA_VERSION
        or target_manifest.get("corpus") != CORPUS_ID
        or target_manifest.get("protected_annotation_fields_accessed") != []
        or manifest.get("target") != expected_target_summary
    ):
        raise ValueError("bundled target provenance does not match the prediction lock")

    lock_identity_fields = (
        "schema_version",
        "corpus",
        "configuration",
        "seed",
        "execution_mode",
        "eligibility_manifest_sha256",
        "fold_manifest_sha256",
        "target_manifest_sha256",
        "session_mapping",
        "feature_contract_sha256",
        "preprocessing_sha256",
        "feature_run_contract_sha256",
        "feature_config_sha256",
        "feature_extraction_closure_file_sha256",
        "feature_extraction_closure_payload_sha256",
        "config_sha256",
        "code_commit_sha",
        "code_source_tree_sha256",
        "model_config_sha256",
        "artifacts",
    )
    if any(field not in manifest for field in lock_identity_fields):
        raise ValueError("prediction lock identity fields are incomplete")
    lock_identity = {field: manifest[field] for field in lock_identity_fields}
    if _payload_sha256(lock_identity) != manifest.get("prediction_lock_id"):
        raise ValueError("prediction lock identity checksum failed")

    input_sessions = eligibility_manifest.get("input_sessions")
    if not isinstance(input_sessions, list):
        raise ValueError("bundled eligibility lacks its canonical-to-source roster")
    source_by_session = {
        row.get("session_id"): row.get("source_session_id")
        for row in input_sessions
        if isinstance(row, Mapping)
    }
    if len(source_by_session) != len(input_sessions):
        raise ValueError("bundled eligibility canonical-to-source roster is invalid")
    eligible_sessions = {
        session.get("session_id"): {
            "source_session_id": session.get("source_session_id"),
            "participant_id": participant.get("participant_id"),
            "session_phase": session.get("condition"),
            "fixed_fold": participant.get("fold"),
        }
        for participant in eligibility_manifest.get("eligible_participants", [])
        if isinstance(participant, Mapping)
        for session in participant.get("sessions", [])
        if isinstance(session, Mapping)
    }
    target_rows = target_manifest.get("rows")
    if not isinstance(target_rows, list):
        raise ValueError("bundled target rows are unavailable")
    target_by_session = {
        row.get("session_id"): row for row in target_rows if isinstance(row, Mapping)
    }
    if len(target_by_session) != len(target_rows):
        raise ValueError("bundled target rows contain invalid or duplicate sessions")
    rotations = manifest.get("rotations")
    if not isinstance(rotations, list) or len(rotations) != 5:
        raise ValueError("prediction lock must contain five ordered rotations")
    fit_by_fold: dict[int, tuple[str, int | None]] = {}
    for fold, rotation in enumerate(rotations):
        fit = rotation.get("fit_artifact") if isinstance(rotation, Mapping) else None
        if (
            not isinstance(rotation, Mapping)
            or rotation.get("outer_rotation") != fold
            or not isinstance(fit, Mapping)
            or not isinstance(fit.get("sha256"), str)
        ):
            raise ValueError("prediction rotation fit provenance is invalid")
        fit_by_fold[fold] = (fit["sha256"], rotation.get("checkpoint_epoch"))
    expected_encoder_revisions = {
        modality: contract["revision"] for modality, contract in ENCODER_CONTRACT.items()
    }
    seen_prediction_sessions: set[str] = set()
    for row in bundle.predictions:
        session_id = row.get("session_id")
        identity = eligible_sessions.get(session_id)
        target_row = target_by_session.get(session_id)
        fold = identity.get("fixed_fold") if isinstance(identity, Mapping) else None
        prediction = row.get("prediction")
        if (
            set(row) != PREDICTION_ROW_FIELDS
            or identity is None
            or target_row is None
            or session_id in seen_prediction_sessions
            or row.get("schema_version") != PREDICTION_SCHEMA_VERSION
            or row.get("corpus") != CORPUS_ID
            or row.get("participant_id") != identity["participant_id"]
            or row.get("source_session_id") != identity["source_session_id"]
            or row.get("source_session_id") != source_by_session.get(session_id)
            or target_row.get("source_session_id") != identity["source_session_id"]
            or row.get("session_phase") != identity["session_phase"]
            or row.get("fixed_fold") != fold
            or row.get("outer_rotation") != fold
            or row.get("rotation_role") != "test"
            or row.get("seed") != bundle.seed
            or row.get("configuration") != bundle.configuration
            or row.get("model_config_sha256") != manifest.get("model_config_sha256")
            or row.get("model_fit_sha256") != fit_by_fold.get(fold, (None, None))[0]
            or row.get("target_transform") != target_manifest.get("transform")
            or row.get("target_manifest_sha256") != manifest.get("target_manifest_sha256")
            or row.get("target") != target_row.get("target")
            or isinstance(prediction, bool)
            or not isinstance(prediction, (int, float))
            or not math.isfinite(float(prediction))
            or row.get("encoder_revisions") != expected_encoder_revisions
            or row.get("feature_contract_sha256") != manifest.get("feature_contract_sha256")
            or row.get("preprocessing_sha256") != manifest.get("preprocessing_sha256")
            or row.get("checkpoint_epoch") != fit_by_fold.get(fold, (None, None))[1]
            or row.get("firewall_stage") != FIREWALL_STAGE
            or row.get("eligibility_manifest_sha256")
            != manifest.get("eligibility_manifest_sha256")
            or row.get("session_mapping") != mapping_provenance
            or row.get("prediction_lock_id") != manifest.get("prediction_lock_id")
        ):
            raise ValueError("prediction row contract or source-session provenance drifted")
        seen_prediction_sessions.add(session_id)
    if seen_prediction_sessions != set(eligible_sessions):
        raise ValueError("prediction rows do not exactly cover eligible canonical sessions")
    if bundle.execution_mode == "research":
        if bundle.eligibility_lock is None:
            raise ValueError("research prediction bundle lacks its prior eligibility lock")
        lock = bundle.eligibility_lock
        _validate_lock_files(lock)
        expected_target_lock = {
            "eligibility_manifest_sha256": lock.eligibility_manifest_sha256,
            "eligibility_filename": lock.eligibility_filename,
            "eligibility_file_sha256": lock.eligibility_file_sha256,
            "fold_filename": lock.fold_filename,
            "fold_file_sha256": lock.fold_file_sha256,
        }
        if (
            eligibility_manifest.get("construction_mode") != "full_corpus"
            or target_manifest.get("construction_mode") != "verified_mit_aggregate_rating_table"
            or lock.eligibility_filename != eligibility_artifact.filename
            or lock.fold_filename != fold_artifact.filename
            or lock.eligibility_manifest_sha256 != manifest.get("eligibility_manifest_sha256")
            or lock.eligibility_file_sha256 != eligibility_artifact.sha256
            or lock.fold_file_sha256 != fold_sha256
            or (lock.output_dir / lock.eligibility_filename).read_bytes()
            != eligibility_artifact.content
            or (lock.output_dir / lock.fold_filename).read_bytes() != fold_artifact.content
            or target_manifest.get("eligibility_lock") != expected_target_lock
        ):
            raise ValueError(
                "research eligibility lock does not match the bundled support artifacts"
            )
    elif bundle.eligibility_lock is not None:
        raise ValueError("synthetic prediction bundle may not carry an eligibility lock")
    if bundle.configuration in PROXY_LEARNED_CONFIGURATIONS:
        # Imported lazily because proxy training depends on the common cross-fit primitives.
        from .proxy_training import _validate_proxy_prediction_bundle

        _validate_proxy_prediction_bundle(bundle)


def utility_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    """Return the fixed session-level utility package without clipping predictions."""
    y = _finite_vector(target, "target")
    pred = _finite_vector(prediction, "prediction")
    if y.shape != pred.shape:
        raise ValueError("target and prediction must have equal shapes")
    residual = pred - y
    return {
        "srcc": _optional_correlation(y, pred, rank=True),
        "pearson": _optional_correlation(y, pred, rank=False),
        "mae_normalised": float(np.mean(np.abs(residual), dtype=np.float64)),
        "mae_raw_scale": float(6.0 * np.mean(np.abs(residual), dtype=np.float64)),
        "rmse": float(np.sqrt(np.mean(np.square(residual), dtype=np.float64))),
    }


def deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Serialize named numeric arrays as a byte-stable uncompressed NPZ."""
    if not arrays:
        raise ValueError("fit artifact must contain at least one array")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for key in sorted(arrays):
            if not isinstance(key, str) or not key or "/" in key or "\\" in key:
                raise ValueError("fit artifact array names must be safe non-empty strings")
            array = np.asarray(arrays[key])
            if array.dtype.hasobject:
                raise ValueError("fit artifact arrays may not use object dtype")
            value = io.BytesIO()
            np.lib.format.write_array(value, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, value.getvalue())
    return output.getvalue()


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return canonical UTF-8 JSON with no non-finite extension values."""
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


def validate_eligibility_result(eligibility: EligibilityResult) -> None:
    """Revalidate a possibly persisted or externally retained eligibility object."""
    if not isinstance(eligibility, EligibilityResult):
        raise TypeError("eligibility must be an EligibilityResult")
    _validate_eligibility_result(eligibility)


def _validated_roster(
    sessions: Sequence[SessionIdentity],
    *,
    require_full_corpus: bool,
    session_mapping: SessionMappingContext | None = None,
) -> tuple[SessionIdentity, ...]:
    if not isinstance(sessions, Sequence) or not sessions:
        raise ValueError("sessions must be a non-empty outcome-free roster")
    seen: set[str] = set()
    seen_sources: set[str] = set()
    grouped: dict[str, list[SessionIdentity]] = {}
    if require_full_corpus:
        if not isinstance(session_mapping, SessionMappingContext):
            raise ValueError("production eligibility requires the approved session mapping")
        session_mapping.verify_current_files()
        expected_sources = session_mapping.lookup
    else:
        if session_mapping is not None:
            raise ValueError("synthetic eligibility must remain mapping-free")
        expected_sources = {}
    for supplied in sessions:
        if not isinstance(supplied, SessionIdentity):
            raise TypeError("sessions must contain SessionIdentity rows")
        session_id, participant_id, condition = parse_session_id(supplied.session_id)
        if supplied.session_id != session_id:
            raise ValueError("session IDs must already be canonical")
        if supplied.participant_id != canonical_participant_id(supplied.participant_id):
            raise ValueError("participant IDs must already be canonical")
        if supplied.participant_id != participant_id or supplied.condition != condition:
            raise ValueError("session identity fields disagree")
        if isinstance(supplied.fold, bool) or supplied.fold not in range(5):
            raise ValueError("session fold must be an integer from 0 to 4")
        if session_id in seen:
            raise ValueError("session roster contains duplicate IDs")
        if require_full_corpus:
            source_session_id = supplied.source_session_id
            if (
                not isinstance(source_session_id, str)
                or expected_sources.get(session_id) != source_session_id
                or source_session_id in seen_sources
                or parse_session_id(source_session_id)[2] != condition
            ):
                raise ValueError(
                    "session roster disagrees with the approved canonical-to-source mapping"
                )
            seen_sources.add(source_session_id)
        elif supplied.source_session_id is not None:
            raise ValueError("synthetic session identities must not claim a production source join")
        seen.add(session_id)
        grouped.setdefault(participant_id, []).append(supplied)
    for participant_id, pair in grouped.items():
        if len(pair) != 2 or {row.condition for row in pair} != {"pre", "post"}:
            raise ValueError(f"participant {participant_id} does not have one complete pair")
        if len({row.fold for row in pair}) != 1:
            raise ValueError(f"participant {participant_id} sessions do not share a fold")
    if require_full_corpus:
        if len(grouped) != EXPECTED_PARTICIPANTS or len(sessions) != EXPECTED_SESSIONS:
            raise ValueError("production eligibility requires the full 69-participant roster")
        expected_folds = assign_participant_folds(list(grouped))
        if any(
            pair[0].fold != expected_folds[participant_id]
            for participant_id, pair in grouped.items()
        ):
            raise ValueError(
                "session roster disagrees with the immutable participant fold manifest"
            )
    canonical = tuple(
        sorted(
            sessions,
            key=lambda row: (int(row.participant_id), row.condition == "post"),
        )
    )
    if require_full_corpus and hashlib.sha256(_fold_manifest_bytes(canonical)).hexdigest() != (
        CANONICAL_FOLD_MANIFEST_SHA256
    ):
        raise ValueError("session roster does not match the audited MIT participant manifest")
    if require_full_corpus:
        if set(seen) != set(expected_sources) or seen_sources != set(expected_sources.values()):
            raise ValueError("session roster does not exactly cover the approved session mapping")
    return canonical


def _collect_feature_artifact(
    successes: dict[str, FeatureSummary],
    failures: dict[str, dict[str, Any]],
    identity: SessionIdentity,
    loader: Any,
) -> None:
    try:
        successes[identity.session_id] = loader()
    except FeatureArtifactError as error:
        if not error.excludable:
            raise ValueError(
                f"feature integrity failed for {identity.session_id}: {error.code}"
            ) from error
        failures[identity.session_id] = {
            "code": error.code,
            "message": str(error),
            "artifact_sha256": error.artifact_sha256,
            "extraction_fingerprint": error.extraction_fingerprint,
            "run_contract_sha256": error.run_contract_sha256,
            "config_sha256": error.config_sha256,
        }


def _validated_feature_closure_manifest(
    closure: FeatureExtractionClosure,
    *,
    roster: Sequence[SessionIdentity],
    session_mapping: SessionMappingContext,
) -> dict[str, Any]:
    closure.verify_current_files()
    session_mapping.verify_current_files()
    try:
        manifest = _loads_unique_json(closure.manifest_bytes)
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("feature extraction closure manifest cannot be decoded") from error
    if canonical_json_bytes(manifest) != closure.manifest_bytes:
        raise ValueError("feature extraction closure manifest is not canonical")
    if hashlib.sha256(closure.manifest_bytes).hexdigest() != closure.closure_file_sha256:
        raise ValueError("feature extraction closure file checksum changed")
    unsigned = dict(manifest)
    signed = unsigned.pop("closure_payload_sha256", None)
    if signed != closure.closure_payload_sha256 or _payload_sha256(unsigned) != signed:
        raise ValueError("feature extraction closure payload checksum changed")
    if (
        manifest.get("schema_version") != FEATURE_EXTRACTION_CLOSURE_SCHEMA_VERSION
        or manifest.get("state") != "closed"
        or manifest.get("execution_mode") != "research"
        or manifest.get("corpus") != CORPUS_ID
        or manifest.get("feature_schema_version") != SCHEMA_VERSION
        or manifest.get("extractor_version") != EXTRACTOR_VERSION
        or manifest.get("session_mapping") != session_mapping.provenance
        or manifest.get("outcome_fields_accessed") != []
        or manifest.get("protected_annotation_fields_accessed") != []
        or manifest.get("selected_session_count") != EXPECTED_SESSIONS
        or manifest.get("completed_session_count") != EXPECTED_SESSIONS
    ):
        raise ValueError("feature extraction closure research contract drifted")
    for name in (
        "feature_run_contract_sha256",
        "feature_contract_sha256",
        "preprocessing_sha256",
        "config_sha256",
    ):
        _require_hex(manifest.get(name), 64, name)
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != len(roster):
        raise ValueError("feature extraction closure does not cover the eligibility roster")
    expected_identities = [
        {
            "visit_index": index,
            "clip_id": identity.session_id,
            "session_id": identity.session_id,
            "source_session_id": identity.source_session_id,
            "participant_id": identity.participant_id,
            "condition": identity.condition,
            "fold": identity.fold,
        }
        for index, identity in enumerate(roster)
    ]
    expected_record_fields = {
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
    observed_identities = []
    for record in records:
        if not isinstance(record, Mapping) or set(record) != expected_record_fields:
            raise ValueError("feature extraction closure record is invalid")
        observed_identities.append(
            {name: record.get(name) for name in expected_identities[0]}
        )
    if observed_identities != expected_identities:
        raise ValueError("feature extraction closure roster differs from eligibility")
    closure.verify_current_files()
    session_mapping.verify_current_files()
    return manifest


def _load_feature_summary(
    root: AnchoredDirectory,
    identity: SessionIdentity,
    *,
    session_mapping: SessionMappingContext | None,
) -> FeatureSummary:
    stem = safe_clip_stem(identity.session_id)
    sidecar_filename = f"{stem}.json"
    shard_filename = f"{stem}.npz"
    failure_filename = f"{stem}.failure.json"
    names = set(root.list_names())
    if failure_filename in names:
        if sidecar_filename in names or shard_filename in names:
            raise FeatureArtifactError(
                "feature_artifact_ambiguous", "success and failure artifacts coexist"
            )
        return _load_feature_summary_bytes(
            identity,
            metadata_file=root.read_bytes(failure_filename, expected_mode=None),
            shard_file=None,
            expected_state="failure",
            session_mapping=session_mapping,
        )
    if sidecar_filename not in names and shard_filename not in names:
        raise FeatureArtifactError("feature_artifact_missing", "feature artifact is unavailable")
    if sidecar_filename not in names or shard_filename not in names:
        raise FeatureArtifactError("feature_artifact_incomplete", "feature artifact is incomplete")

    return _load_feature_summary_bytes(
        identity,
        metadata_file=root.read_bytes(sidecar_filename, expected_mode=None),
        shard_file=root.read_bytes(shard_filename, expected_mode=None),
        expected_state="success",
        session_mapping=session_mapping,
    )


def _load_feature_summary_from_closure(
    artifacts: FeatureRecordArtifacts,
    identity: SessionIdentity,
    *,
    session_mapping: SessionMappingContext,
) -> FeatureSummary:
    record = artifacts.record
    if record.get("session_id") != identity.session_id:
        raise FeatureArtifactError(
            "feature_identity_mismatch", "feature closure record does not match roster"
        )
    return _load_feature_summary_bytes(
        identity,
        metadata_file=artifacts.metadata,
        shard_file=artifacts.shard,
        expected_state=record["state"],
        session_mapping=session_mapping,
    )


def _load_feature_summary_bytes(
    identity: SessionIdentity,
    *,
    metadata_file: AnchoredFile,
    shard_file: AnchoredFile | None,
    expected_state: str,
    session_mapping: SessionMappingContext | None,
) -> FeatureSummary:
    try:
        metadata_raw = metadata_file.content
        payload = _loads_unique_json(metadata_raw)
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        code = (
            "feature_failure_record_invalid"
            if expected_state == "failure"
            else "feature_sidecar_unreadable"
        )
        raise FeatureArtifactError(code, "feature metadata cannot be read") from error
    if canonical_json_bytes(payload) != metadata_raw:
        code = (
            "feature_failure_record_invalid"
            if expected_state == "failure"
            else "feature_sidecar_noncanonical"
        )
        raise FeatureArtifactError(code, "feature metadata is not canonical")
    if payload.get("state") != expected_state:
        raise FeatureArtifactError(
            "feature_identity_mismatch", "feature metadata state does not match its closure"
        )
    signed_field = (
        "failure_payload_sha256"
        if expected_state == "failure" and "failure_payload_sha256" in payload
        else "sidecar_payload_sha256"
    )
    if expected_state == "failure" and signed_field not in payload:
        signed_sha = None
    else:
        signed_sha = payload.get(signed_field)
        unsigned = dict(payload)
        unsigned.pop(signed_field, None)
        if not isinstance(signed_sha, str) or _payload_sha256(unsigned) != signed_sha:
            code = (
                "feature_failure_record_invalid"
                if expected_state == "failure"
                else "feature_sidecar_checksum_invalid"
            )
            raise FeatureArtifactError(code, "feature metadata checksum failed")

    required_identity = {
        "schema_version": SCHEMA_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "state": expected_state,
        "clip_id": identity.session_id,
        "participant_id": identity.participant_id,
        "condition": identity.condition,
        "fold": identity.fold,
    }
    if any(payload.get(key) != value for key, value in required_identity.items()):
        raise FeatureArtifactError(
            "feature_identity_mismatch", "feature metadata does not match roster"
        )
    if session_mapping is not None and payload.get("session_id") != identity.session_id:
        raise FeatureArtifactError(
            "feature_identity_mismatch", "feature canonical session ID drifted"
        )
    if payload.get("source_session_id") != identity.source_session_id:
        raise FeatureArtifactError(
            "feature_identity_mismatch", "feature source session ID drifted"
        )
    _validate_feature_mapping_binding(payload, session_mapping)
    run_contract_sha256, config_sha256 = _validate_feature_run_contract(
        payload, success=expected_state == "success"
    )
    extraction_fingerprint = payload.get("extraction_fingerprint")
    if extraction_fingerprint is not None:
        _require_hex_error(extraction_fingerprint, 64, "feature_fingerprint_invalid")

    if expected_state == "failure":
        if shard_file is not None:
            raise FeatureArtifactError(
                "feature_artifact_ambiguous", "failure record unexpectedly has a shard"
            )
        details = payload.get("failure")
        code = details.get("code") if isinstance(details, Mapping) else None
        if not isinstance(code, str) or not code:
            raise FeatureArtifactError(
                "feature_failure_record_invalid", "feature failure code is unavailable"
            )
        raise FeatureArtifactError(
            code,
            "frozen feature extraction reported a session failure",
            excludable=True,
            artifact_sha256=metadata_file.sha256,
            extraction_fingerprint=extraction_fingerprint,
            run_contract_sha256=run_contract_sha256,
            config_sha256=config_sha256,
        )

    sidecar = payload
    sidecar_raw = metadata_raw
    shard_filename = f"{safe_clip_stem(identity.session_id)}.npz"
    if shard_file is None:
        raise FeatureArtifactError(
            "feature_artifact_incomplete", "feature success record lacks its shard"
        )
    _require_hex_error(extraction_fingerprint, 64, "feature_fingerprint_invalid")
    preprocessing = sidecar.get("preprocessing")
    if preprocessing != EXPECTED_PREPROCESSING:
        raise FeatureArtifactError(
            "feature_preprocessing_invalid", "feature preprocessing contract drifted"
        )
    preprocessing_sha256 = _payload_sha256(preprocessing)
    contract_payload = {
        "schema_version": sidecar["schema_version"],
        "extractor_version": sidecar["extractor_version"],
        "config_sha256": sidecar.get("config_sha256"),
        "code_source_tree_sha256": (
            sidecar.get("code", {}).get("source_tree_sha256")
            if isinstance(sidecar.get("code"), Mapping)
            else None
        ),
        "encoders": sidecar["encoders"],
        "preprocessing": preprocessing,
        "runtime": sidecar.get("runtime"),
        "session_mapping": sidecar.get("session_mapping"),
    }
    feature_contract_sha256 = _payload_sha256(contract_payload)

    shard = sidecar.get("shard")
    if not isinstance(shard, Mapping):
        raise FeatureArtifactError(
            "feature_shard_metadata_invalid", "feature shard metadata is invalid"
        )
    if shard.get("filename") != shard_filename or shard.get("keys") != list(SHARD_KEYS):
        raise FeatureArtifactError(
            "feature_shard_metadata_invalid", "feature shard metadata drifted"
        )
    expected_shard_sha = shard.get("sha256")
    _require_hex_error(expected_shard_sha, 64, "feature_shard_checksum_invalid")
    actual_shard_sha = shard_file.sha256
    if actual_shard_sha != expected_shard_sha:
        raise FeatureArtifactError(
            "feature_shard_checksum_invalid", "feature shard checksum failed"
        )

    try:
        with np.load(io.BytesIO(shard_file.content), allow_pickle=False) as archive:
            if tuple(archive.files) != SHARD_KEYS:
                raise FeatureArtifactError(
                    "feature_shard_schema_invalid", "feature shard keys drifted"
                )
            arrays = {key: archive[key] for key in SHARD_KEYS}
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise FeatureArtifactError(
            "feature_shard_unreadable", "feature shard cannot be read"
        ) from error
    metadata = sidecar.get("arrays")
    if not isinstance(metadata, Mapping):
        raise FeatureArtifactError("feature_shard_schema_invalid", "array metadata is unavailable")
    for key, array in arrays.items():
        entry = metadata.get(key)
        if (
            not isinstance(entry, Mapping)
            or entry.get("shape") != list(array.shape)
            or entry.get("dtype") != str(array.dtype)
        ):
            raise FeatureArtifactError(
                "feature_shard_schema_invalid", "array metadata does not match payload"
            )
    summaries: dict[str, np.ndarray] = {}
    for modality, width in SUMMARY_DIMENSIONS.items():
        matrix = np.asarray(arrays[modality])
        mask = np.asarray(arrays[f"{modality}_mask"])
        summary = np.asarray(arrays[f"{modality}_mean"])
        if (
            matrix.dtype != np.float32
            or matrix.ndim != 2
            or matrix.shape[0] == 0
            or matrix.shape[1] != width
            or not np.isfinite(matrix).all()
            or mask.dtype != np.bool_
            or mask.shape != (matrix.shape[0],)
            or not mask.all()
            or summary.dtype != np.float32
            or summary.shape != (width,)
            or not np.isfinite(summary).all()
            or not np.array_equal(summary, matrix.mean(axis=0, dtype=np.float32))
        ):
            raise FeatureArtifactError(
                "feature_summary_invalid", f"{modality} summary violates the frozen feature schema"
            )
        summaries[modality] = _readonly_copy(summary)
    return FeatureSummary(
        identity=identity,
        video=summaries["video"],
        audio=summaries["audio"],
        text=summaries["text"],
        sidecar_sha256=hashlib.sha256(sidecar_raw).hexdigest(),
        shard_sha256=actual_shard_sha,
        extraction_fingerprint=extraction_fingerprint,
        feature_contract_sha256=feature_contract_sha256,
        preprocessing_sha256=preprocessing_sha256,
        run_contract_sha256=run_contract_sha256,
        config_sha256=config_sha256,
    )


def _validate_feature_mapping_binding(
    payload: Mapping[str, Any], session_mapping: SessionMappingContext | None
) -> None:
    observed = payload.get("session_mapping")
    if session_mapping is None:
        if observed is not None:
            raise FeatureArtifactError(
                "feature_mapping_invalid",
                "synthetic feature artifact may not claim a production session mapping",
            )
        return
    if not isinstance(observed, Mapping) or set(observed) != set(
        SESSION_MAPPING_PROVENANCE_FIELDS
    ):
        raise FeatureArtifactError(
            "feature_mapping_invalid", "feature session-mapping provenance schema drifted"
        )
    if dict(observed) != session_mapping.provenance:
        raise FeatureArtifactError(
            "feature_mapping_invalid",
            "feature artifact is not bound to the approved session mapping",
        )


def _validate_feature_run_contract(payload: Mapping[str, Any], *, success: bool) -> tuple[str, str]:
    config_sha256 = payload.get("config_sha256")
    _require_hex_error(config_sha256, 64, "feature_config_invalid")
    code = payload.get("code")
    if not isinstance(code, Mapping):
        raise FeatureArtifactError("feature_code_invalid", "feature code provenance is unavailable")
    _require_hex_error(code.get("commit_sha"), 40, "feature_code_invalid")
    _require_hex_error(code.get("source_tree_sha256"), 64, "feature_code_invalid")
    if not isinstance(code.get("dirty"), bool):
        raise FeatureArtifactError(
            "feature_code_invalid", "feature dirty-state provenance is invalid"
        )
    source_date_epoch = payload.get("source_date_epoch")
    if (
        isinstance(source_date_epoch, bool)
        or not isinstance(source_date_epoch, int)
        or source_date_epoch < 0
    ):
        raise FeatureArtifactError(
            "feature_runtime_invalid", "feature source-date epoch is invalid"
        )
    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping) or not runtime:
        raise FeatureArtifactError(
            "feature_runtime_invalid", "feature runtime provenance is unavailable"
        )
    _validate_encoder_contract(payload.get("encoders"))
    inputs = payload.get("input")
    if not isinstance(inputs, Mapping):
        raise FeatureArtifactError(
            "feature_input_invalid", "feature input provenance is unavailable"
        )
    production_contract = isinstance(payload.get("session_mapping"), Mapping)
    if success or production_contract:
        if payload.get("script") != "scripts/extract_features.py" or payload.get("seed") != 2026:
            raise FeatureArtifactError("feature_run_invalid", "feature run contract drifted")
        expected_input_fields = {
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
        if set(inputs) != expected_input_fields:
            raise FeatureArtifactError(
                "feature_input_invalid", "feature input provenance schema drifted"
            )
        source_session_id = payload.get("source_session_id", payload.get("clip_id"))
        if (
            inputs.get("video_reference") != f"Videos/{source_session_id}.avi"
            or inputs.get("audio_reference") != f"Audio/{source_session_id}.wav"
        ):
            raise FeatureArtifactError(
                "feature_input_invalid", "feature media references do not match the MIT session"
            )
        for field in ("source_dialogue_sha256", "face_crop_fingerprint"):
            _require_hex_error(inputs.get(field), 64, "feature_input_invalid")
        for field in ("video_sha256", "audio_sha256", "candidate_transcript_sha256"):
            value = inputs.get(field)
            if value is None and not success:
                continue
            _require_hex_error(value, 64, "feature_input_invalid")
        if inputs.get("transcript_parser_revision") != TRANSCRIPT_PARSER_REVISION:
            raise FeatureArtifactError(
                "feature_input_invalid", "feature transcript parser provenance drifted"
            )
        parser_audit = inputs.get("transcript_parser_audit")
        expected_audit_fields = {
            "explicit_candidate_segments",
            "retained_candidate_continuations",
            "retained_candidate_continuation_characters",
            "excluded_colonless_fragments",
            "excluded_colonless_characters",
            "excluded_unclassified_markers",
            "excluded_interviewer_segments",
        }
        if not isinstance(parser_audit, Mapping) or set(parser_audit) != expected_audit_fields:
            raise FeatureArtifactError(
                "feature_input_invalid", "feature transcript parser audit drifted"
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in parser_audit.values()
        ):
            raise FeatureArtifactError(
                "feature_input_invalid", "feature transcript parser audit is invalid"
            )
    else:
        required_failure_inputs = {
            "video_reference",
            "video_sha256",
            "audio_reference",
            "audio_sha256",
            "transcript_sha256",
            "face_crop_fingerprint",
        }
        if set(inputs) != required_failure_inputs:
            raise FeatureArtifactError(
                "feature_failure_record_invalid", "feature failure input provenance drifted"
            )
        source_session_id = payload.get("source_session_id", payload.get("clip_id"))
        if (
            inputs.get("video_reference") != f"Videos/{source_session_id}.avi"
            or inputs.get("audio_reference") != f"Audio/{source_session_id}.wav"
        ):
            raise FeatureArtifactError(
                "feature_failure_record_invalid",
                "feature failure media references do not match the MIT session",
            )
        for field in (
            "video_sha256",
            "audio_sha256",
            "transcript_sha256",
            "face_crop_fingerprint",
        ):
            value = inputs.get(field)
            if value is not None:
                _require_hex_error(value, 64, "feature_failure_record_invalid")
    run_contract = {
        "schema_version": payload.get("schema_version"),
        "extractor_version": payload.get("extractor_version"),
        "config_sha256": config_sha256,
        "code": dict(code),
        "source_date_epoch": source_date_epoch,
        "runtime": runtime,
        "encoders": payload.get("encoders"),
        "session_mapping": payload.get("session_mapping"),
    }
    if production_contract:
        preprocessing = payload.get("preprocessing")
        if preprocessing != EXPECTED_PREPROCESSING:
            raise FeatureArtifactError(
                "feature_preprocessing_invalid", "feature preprocessing contract drifted"
            )
        run_contract = {
            "schema_version": payload.get("schema_version"),
            "extractor_version": payload.get("extractor_version"),
            "script": payload.get("script"),
            "seed": payload.get("seed"),
            "source_date_epoch": source_date_epoch,
            "config_sha256": config_sha256,
            "code": dict(code),
            "session_mapping": payload.get("session_mapping"),
            "runtime": runtime,
            "encoders": payload.get("encoders"),
            "preprocessing": preprocessing,
        }
    return _payload_sha256(run_contract), config_sha256


def _validate_encoder_contract(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != set(ENCODER_CONTRACT):
        raise FeatureArtifactError("feature_encoder_invalid", "feature encoder metadata is invalid")
    for modality, expected in ENCODER_CONTRACT.items():
        observed = value.get(modality)
        if not isinstance(observed, Mapping) or any(
            observed.get(field) != expected_value for field, expected_value in expected.items()
        ):
            raise FeatureArtifactError(
                "feature_encoder_invalid", f"{modality} encoder does not match the frozen artifact"
            )


def _validate_mapping_provenance(value: object, *, required: bool) -> dict[str, str] | None:
    if value is None:
        if required:
            raise ValueError("production artifact lacks approved session-mapping provenance")
        return None
    if not required:
        raise ValueError("synthetic artifact may not claim production session-mapping provenance")
    if not isinstance(value, Mapping) or set(value) != set(SESSION_MAPPING_PROVENANCE_FIELDS):
        raise ValueError("session-mapping provenance schema drifted")
    result = dict(value)
    if result["schema_version"] != MIT_SESSION_MAPPING_SCHEMA_VERSION:
        raise ValueError("session-mapping provenance version drifted")
    for name in SESSION_MAPPING_PROVENANCE_FIELDS[1:]:
        _require_hex(result[name], 64, name)
    return result


def _validate_eligibility_result(eligibility: EligibilityResult) -> None:
    if not isinstance(eligibility.manifest, Mapping):
        raise TypeError("eligibility manifest must be a mapping")
    _validate_signed_manifest(
        eligibility.manifest,
        eligibility.manifest_bytes,
        eligibility.manifest_sha256,
        label="eligibility",
    )
    _require_hex(eligibility.fold_manifest_sha256, 64, "fold_manifest_sha256")
    if hashlib.sha256(eligibility.fold_manifest_bytes).hexdigest() != (
        eligibility.fold_manifest_sha256
    ):
        raise ValueError("eligibility fold manifest checksum failed")
    fold_by_participant = _parse_fold_manifest(eligibility.fold_manifest_bytes)
    manifest = eligibility.manifest
    if manifest.get("schema_version") != ELIGIBILITY_SCHEMA_VERSION:
        raise ValueError("eligibility schema version drifted")
    duplicated_fields = {
        "fold_manifest_sha256": eligibility.fold_manifest_sha256,
        "feature_contract_sha256": eligibility.feature_contract_sha256,
        "preprocessing_sha256": eligibility.preprocessing_sha256,
        "feature_config_sha256": eligibility.feature_config_sha256,
    }
    if any(manifest.get(key) != value for key, value in duplicated_fields.items()):
        raise ValueError("eligibility duplicated provenance fields disagree")
    for name, value in (
        ("feature_contract_sha256", eligibility.feature_contract_sha256),
        ("preprocessing_sha256", eligibility.preprocessing_sha256),
        ("feature_config_sha256", eligibility.feature_config_sha256),
    ):
        _require_hex(value, 64, name)
    if not isinstance(eligibility.production_ready, bool):
        raise TypeError("eligibility production_ready must be boolean")
    expected_mode = "full_corpus" if eligibility.production_ready else "synthetic_contract_test"
    if manifest.get("construction_mode") != expected_mode:
        raise ValueError("eligibility construction mode disagrees with production readiness")
    _validate_mapping_provenance(
        manifest.get("session_mapping"), required=eligibility.production_ready
    )
    closure_hashes = (
        manifest.get("feature_extraction_closure_file_sha256"),
        manifest.get("feature_extraction_closure_payload_sha256"),
    )
    if eligibility.production_ready:
        _require_hex(closure_hashes[0], 64, "feature_extraction_closure_file_sha256")
        _require_hex(
            closure_hashes[1], 64, "feature_extraction_closure_payload_sha256"
        )
    elif closure_hashes != (None, None):
        raise ValueError("synthetic eligibility may not claim a research feature closure")
    if manifest.get("outcome_fields_accessed") != []:
        raise ValueError("eligibility may not record outcome access")
    if manifest.get("protected_annotation_fields_accessed") != []:
        raise ValueError("eligibility may not record protected-annotation access")
    if eligibility.production_ready:
        if eligibility.fold_manifest_sha256 != CANONICAL_FOLD_MANIFEST_SHA256:
            raise ValueError("production eligibility does not use the audited MIT fold manifest")
        if len(fold_by_participant) != EXPECTED_PARTICIPANTS:
            raise ValueError("production eligibility fold roster is incomplete")

    input_rows = manifest.get("input_sessions")
    if not isinstance(input_rows, list) or len(input_rows) != 2 * len(fold_by_participant):
        raise ValueError("eligibility input-session roster is invalid")
    input_by_session: dict[str, Mapping[str, Any]] = {}
    source_ids: set[str] = set()
    nonidentity_mapping: dict[str, str] = {}
    for row in input_rows:
        if not isinstance(row, Mapping) or set(row) != {
            "session_id",
            "source_session_id",
            "participant_id",
            "condition",
            "fold",
        }:
            raise ValueError("eligibility input-session row schema drifted")
        session_id = row["session_id"]
        canonical, participant_id, condition = parse_session_id(session_id)
        if (
            canonical != session_id
            or session_id in input_by_session
            or row["participant_id"] != participant_id
            or row["condition"] != condition
            or fold_by_participant.get(participant_id) != row["fold"]
        ):
            raise ValueError("eligibility input-session identity fields disagree")
        source_session_id = row["source_session_id"]
        if eligibility.production_ready:
            if not isinstance(source_session_id, str):
                raise ValueError("production eligibility lacks a source-session ID")
            parsed_source, _, source_condition = parse_session_id(source_session_id)
            if (
                parsed_source != source_session_id
                or source_condition != condition
                or source_session_id in source_ids
            ):
                raise ValueError("eligibility input mapping is not a phase-preserving bijection")
            source_ids.add(source_session_id)
            if source_session_id != session_id:
                nonidentity_mapping[session_id] = source_session_id
        elif source_session_id is not None:
            raise ValueError("synthetic eligibility input may not claim a source-session join")
        input_by_session[session_id] = row
    if eligibility.production_ready:
        if len(source_ids) != EXPECTED_SESSIONS or nonidentity_mapping != {
            "PP32": "PP33",
            "PP33": "PP32",
        }:
            raise ValueError("eligibility input roster does not implement the approved MIT mapping")

    eligible_rows = manifest.get("eligible_participants")
    exclusions = manifest.get("exclusions")
    if not isinstance(eligible_rows, list) or not isinstance(exclusions, list):
        raise ValueError("eligibility participant rows are invalid")
    manifest_sessions: dict[str, Mapping[str, Any]] = {}
    manifest_participants: set[str] = set()
    for participant in eligible_rows:
        if not isinstance(participant, Mapping):
            raise ValueError("eligible participant row is invalid")
        participant_id = participant.get("participant_id")
        fold = participant.get("fold")
        sessions = participant.get("sessions")
        if (
            not isinstance(participant_id, str)
            or participant_id in manifest_participants
            or fold_by_participant.get(participant_id) != fold
            or not isinstance(sessions, list)
            or len(sessions) != 2
        ):
            raise ValueError("eligible participant identity or fold is invalid")
        manifest_participants.add(participant_id)
        if {row.get("condition") for row in sessions if isinstance(row, Mapping)} != {
            "pre",
            "post",
        }:
            raise ValueError("eligible participant does not contain one pre/post pair")
        for row in sessions:
            if not isinstance(row, Mapping):
                raise ValueError("eligible session row is invalid")
            session_id = row.get("session_id")
            if not isinstance(session_id, str) or session_id in manifest_sessions:
                raise ValueError("eligible session ID is invalid or duplicated")
            canonical, parsed_participant, condition = parse_session_id(session_id)
            input_row = input_by_session.get(session_id)
            if (
                canonical != session_id
                or parsed_participant != participant_id
                or row.get("condition") != condition
                or input_row is None
                or row.get("source_session_id") != input_row.get("source_session_id")
            ):
                raise ValueError("eligible session identity fields disagree")
            manifest_sessions[session_id] = row

    feature_sessions: set[str] = set()
    for feature in eligibility.features:
        if not isinstance(feature, FeatureSummary):
            raise TypeError("eligibility features must contain FeatureSummary rows")
        identity = feature.identity
        row = manifest_sessions.get(identity.session_id)
        if row is None or identity.session_id in feature_sessions:
            raise ValueError("eligibility features disagree with eligible manifest sessions")
        feature_sessions.add(identity.session_id)
        if (
            identity.participant_id not in manifest_participants
            or fold_by_participant.get(identity.participant_id) != identity.fold
            or row.get("condition") != identity.condition
            or row.get("source_session_id") != identity.source_session_id
            or row.get("sidecar_sha256") != feature.sidecar_sha256
            or row.get("shard_sha256") != feature.shard_sha256
            or row.get("extraction_fingerprint") != feature.extraction_fingerprint
        ):
            raise ValueError("eligibility feature provenance disagrees with its manifest row")
        for name, value in (
            ("sidecar_sha256", feature.sidecar_sha256),
            ("shard_sha256", feature.shard_sha256),
            ("extraction_fingerprint", feature.extraction_fingerprint),
            ("feature_contract_sha256", feature.feature_contract_sha256),
            ("preprocessing_sha256", feature.preprocessing_sha256),
            ("run_contract_sha256", feature.run_contract_sha256),
            ("config_sha256", feature.config_sha256),
        ):
            _require_hex(value, 64, name)
        if (
            feature.feature_contract_sha256 != eligibility.feature_contract_sha256
            or feature.preprocessing_sha256 != eligibility.preprocessing_sha256
            or feature.config_sha256 != eligibility.feature_config_sha256
            or feature.run_contract_sha256 != manifest.get("feature_run_contract_sha256")
        ):
            raise ValueError("eligibility feature contracts disagree with global provenance")
        summary_digests = row.get("summary_sha256")
        if not isinstance(summary_digests, Mapping):
            raise ValueError("eligibility summary digests are unavailable")
        for modality, dimension in SUMMARY_DIMENSIONS.items():
            summary = np.asarray(getattr(feature, modality))
            if (
                summary.dtype != np.float32
                or summary.shape != (dimension,)
                or not np.isfinite(summary).all()
                or summary_digests.get(modality) != _summary_sha256(summary)
            ):
                raise ValueError("eligibility feature summary disagrees with its manifest digest")
    if feature_sessions != set(manifest_sessions):
        raise ValueError("eligibility feature rows do not exactly cover the manifest sessions")

    excluded_participants: set[str] = set()
    for exclusion in exclusions:
        if not isinstance(exclusion, Mapping):
            raise ValueError("eligibility exclusion row is invalid")
        participant_id = exclusion.get("participant_id")
        if (
            not isinstance(participant_id, str)
            or participant_id in manifest_participants
            or participant_id in excluded_participants
            or fold_by_participant.get(participant_id) != exclusion.get("fold")
        ):
            raise ValueError("eligibility exclusion identity or fold is invalid")
        excluded_participants.add(participant_id)
        failures = exclusion.get("session_failures")
        if not isinstance(failures, list) or not failures:
            raise ValueError("eligibility exclusion lacks session failure evidence")
        for failure in failures:
            if not isinstance(failure, Mapping):
                raise ValueError("eligibility exclusion session failure is invalid")
            session_id = failure.get("session_id")
            input_row = input_by_session.get(session_id)
            if (
                input_row is None
                or failure.get("source_session_id") != input_row.get("source_session_id")
            ):
                raise ValueError("eligibility exclusion source-session provenance drifted")
    if manifest_participants | excluded_participants != set(fold_by_participant):
        raise ValueError("eligibility participants do not cover the fold manifest")
    if (
        manifest.get("input_participant_count") != len(fold_by_participant)
        or manifest.get("input_session_count") != 2 * len(fold_by_participant)
        or manifest.get("eligible_participant_count") != len(manifest_participants)
        or manifest.get("eligible_session_count") != len(manifest_sessions)
    ):
        raise ValueError("eligibility manifest counts disagree with its rows")


def _validate_target_set(targets: TargetSet) -> None:
    if not isinstance(targets.manifest, Mapping):
        raise TypeError("target manifest must be a mapping")
    _validate_signed_manifest(
        targets.manifest,
        targets.manifest_bytes,
        targets.manifest_sha256,
        label="target",
    )
    if not isinstance(targets.production_ready, bool):
        raise TypeError("target production_ready must be boolean")
    mode = targets.manifest.get("construction_mode")
    expected_mode = (
        "verified_mit_aggregate_rating_table"
        if targets.production_ready
        else "synthetic_contract_test"
    )
    if mode != expected_mode:
        raise ValueError("target construction mode disagrees with production readiness")
    _validate_mapping_provenance(
        targets.manifest.get("session_mapping"), required=targets.production_ready
    )
    if targets.manifest.get("protected_annotation_fields_accessed") != []:
        raise ValueError("target set may not record protected-annotation access")
    rows = targets.manifest.get("rows")
    if not isinstance(rows, list) or not rows or len(rows) != len(targets.values):
        raise ValueError("target rows are unavailable or incomplete")
    observed: list[tuple[str, float]] = []
    seen: set[str] = set()
    seen_sources: set[str] = set()
    source_overrides: dict[str, str] = {}
    for row, supplied in zip(rows, targets.values, strict=True):
        if not isinstance(row, Mapping) or not isinstance(supplied, tuple) or len(supplied) != 2:
            raise ValueError("target row is invalid")
        session_id, supplied_target = supplied
        canonical, _, _ = parse_session_id(session_id)
        if canonical != session_id or session_id in seen or row.get("session_id") != session_id:
            raise ValueError("target session IDs are invalid, duplicated or reordered")
        seen.add(session_id)
        if isinstance(supplied_target, bool):
            raise TypeError("target values must be finite numeric values")
        target = float(supplied_target)
        if not math.isfinite(target) or not 0.0 <= target <= 1.0 or row.get("target") != target:
            raise ValueError("in-memory target values disagree with the target manifest")
        if targets.production_ready:
            source_session_id = row.get("source_session_id")
            if not isinstance(source_session_id, str):
                raise ValueError("production target row lacks its raw source-session ID")
            parsed_source, _, source_condition = parse_session_id(source_session_id)
            if (
                parsed_source != source_session_id
                or source_condition != parse_session_id(session_id)[2]
                or source_session_id in seen_sources
            ):
                raise ValueError(
                    "production target rows are not a phase-preserving source bijection"
                )
            seen_sources.add(source_session_id)
            if source_session_id != session_id:
                source_overrides[session_id] = source_session_id
            raw = row.get("recommend_hiring_raw")
            if (
                isinstance(raw, bool)
                or not isinstance(raw, (int, float))
                or not math.isfinite(float(raw))
                or not 1.0 <= float(raw) <= 7.0
                or not math.isclose(target, (float(raw) - 1.0) / 6.0, rel_tol=0.0, abs_tol=0.0)
            ):
                raise ValueError("production target row violates the frozen normalisation")
        observed.append((session_id, target))
    expected_order = sorted(
        observed, key=lambda row: (int(parse_session_id(row[0])[1]), row[0].startswith("PP"))
    )
    if observed != expected_order:
        raise ValueError("target rows are not in canonical participant/session order")
    if targets.production_ready:
        if len(seen_sources) != EXPECTED_SESSIONS or source_overrides != {
            "PP32": "PP33",
            "PP33": "PP32",
        }:
            raise ValueError("production targets do not implement the approved MIT source mapping")
        if targets.eligibility_lock is None:
            raise ValueError("production target set requires a durable eligibility lock")
        _validate_lock_files(targets.eligibility_lock)
        lock_manifest = targets.manifest.get("eligibility_lock")
        expected_lock_manifest = {
            "eligibility_manifest_sha256": targets.eligibility_lock.eligibility_manifest_sha256,
            "eligibility_filename": targets.eligibility_lock.eligibility_filename,
            "eligibility_file_sha256": targets.eligibility_lock.eligibility_file_sha256,
            "fold_filename": targets.eligibility_lock.fold_filename,
            "fold_file_sha256": targets.eligibility_lock.fold_file_sha256,
        }
        if lock_manifest != expected_lock_manifest:
            raise ValueError("target manifest disagrees with its prior eligibility lock")
        source = targets.manifest.get("source")
        if (
            not isinstance(source, Mapping)
            or source.get("reference") != TARGET_TABLE_REFERENCE
            or source.get("worker") != "AGGR"
            or source.get("field") != "RecommendHiring"
        ):
            raise ValueError("production target source contract drifted")
        _require_hex(source.get("sha256"), 64, "target source sha256")
        if targets.manifest.get("transform") != TARGET_TRANSFORM:
            raise ValueError("production target transform drifted")
    elif targets.eligibility_lock is not None:
        raise ValueError("synthetic target set may not carry a research eligibility lock")


def _validate_signed_manifest(
    manifest: Mapping[str, Any], raw: bytes, manifest_sha256: str, *, label: str
) -> None:
    _require_hex(manifest_sha256, 64, f"{label}_manifest_sha256")
    if canonical_json_bytes(manifest) != raw:
        raise ValueError(f"{label} manifest bytes do not match the supplied manifest")
    unsigned = dict(manifest)
    signed_sha = unsigned.pop("manifest_payload_sha256", None)
    if signed_sha != manifest_sha256 or _payload_sha256(unsigned) != manifest_sha256:
        raise ValueError(f"{label} manifest integrity check failed")


def _validate_durable_eligibility_lock(
    lock: EligibilityLock,
    eligibility: EligibilityResult,
    *,
    output: AnchoredDirectory | None = None,
) -> None:
    if not isinstance(lock, EligibilityLock):
        raise TypeError("eligibility_lock must be an EligibilityLock")
    if output is None:
        with AnchoredDirectory.open(
            lock.output_dir, directory_mode=None
        ) as anchored:
            _validate_durable_eligibility_lock(
                lock, eligibility, output=anchored
            )
        return
    _validate_lock_files(lock, output=output)
    eligibility_file = output.read_bytes(
        lock.eligibility_filename, expected_mode=None
    )
    fold_file = output.read_bytes(lock.fold_filename, expected_mode=None)
    if (
        lock.eligibility_manifest_sha256 != eligibility.manifest_sha256
        or lock.eligibility_file_sha256 != hashlib.sha256(eligibility.manifest_bytes).hexdigest()
        or lock.fold_file_sha256 != hashlib.sha256(eligibility.fold_manifest_bytes).hexdigest()
        or eligibility_file.content != eligibility.manifest_bytes
        or fold_file.content != eligibility.fold_manifest_bytes
    ):
        raise ValueError("durable eligibility lock does not match eligibility")


def _validate_lock_files(
    lock: EligibilityLock, *, output: AnchoredDirectory | None = None
) -> None:
    if not isinstance(lock, EligibilityLock):
        raise TypeError("eligibility_lock must be an EligibilityLock")
    if output is None:
        try:
            with AnchoredDirectory.open(
                lock.output_dir, directory_mode=None
            ) as anchored:
                _validate_lock_files(lock, output=anchored)
            return
        except (RuntimeError, ValueError) as error:
            raise ValueError(
                "eligibility lock directory is unavailable or unsafe"
            ) from error
    output.verify_path_identity()
    if (
        output.path != lock.output_dir
        or output.device != lock.output_device
        or output.inode != lock.output_inode
    ):
        raise ValueError("eligibility lock directory identity changed")
    for filename, expected_sha256 in (
        (lock.eligibility_filename, lock.eligibility_file_sha256),
        (lock.fold_filename, lock.fold_file_sha256),
    ):
        artifact = output.read_bytes(filename, expected_mode=None)
        if artifact.sha256 != expected_sha256:
            raise ValueError("eligibility lock artifact checksum failed")


def _execution_mode(
    eligibility: EligibilityResult,
    targets: TargetSet,
    provenance: RunProvenance,
    *,
    synthetic_requested: bool,
) -> str:
    if not isinstance(synthetic_requested, bool):
        raise TypeError("synthetic smoke mode must be a boolean")
    if not isinstance(eligibility, EligibilityResult):
        raise TypeError("eligibility must be an EligibilityResult")
    if not isinstance(targets, TargetSet):
        raise TypeError("targets must be a TargetSet")
    _validate_eligibility_result(eligibility)
    _validate_target_set(targets)
    if eligibility.manifest.get("session_mapping") != targets.manifest.get("session_mapping"):
        raise ValueError("eligibility and targets do not share one session-mapping provenance")
    production_pair = eligibility.production_ready and targets.production_ready
    if synthetic_requested:
        if production_pair:
            raise ValueError(
                "synthetic smoke caps are forbidden for production-ready eligibility and targets"
            )
        if eligibility.production_ready != targets.production_ready:
            raise ValueError(
                "synthetic smoke execution requires both eligibility and targets to be synthetic"
            )
        return "synthetic_smoke"
    if not eligibility.production_ready or not targets.production_ready:
        raise ValueError(
            "research execution requires full-corpus eligibility and a verified MIT target set"
        )
    if eligibility.fold_manifest_sha256 != CANONICAL_FOLD_MANIFEST_SHA256:
        raise ValueError("research eligibility does not use the audited MIT fold manifest")
    if eligibility.manifest.get("construction_mode") != "full_corpus":
        raise ValueError("research eligibility construction mode is invalid")
    if targets.manifest.get("construction_mode") != "verified_mit_aggregate_rating_table":
        raise ValueError("research target construction mode is invalid")
    if targets.eligibility_lock is None:
        raise ValueError("research target set lacks the prior durable eligibility lock")
    _validate_durable_eligibility_lock(targets.eligibility_lock, eligibility)
    target_lock = targets.manifest.get("eligibility_lock")
    if (
        not isinstance(target_lock, Mapping)
        or target_lock.get("eligibility_manifest_sha256") != eligibility.manifest_sha256
    ):
        raise ValueError("target set is not bound to this eligibility lock")
    if eligibility.feature_config_sha256 != provenance.config_sha256:
        raise ValueError("training config checksum does not match frozen feature extraction")
    return "research"


def _bind_targets(eligibility: EligibilityResult, targets: TargetSet) -> tuple[_Example, ...]:
    if not isinstance(targets, TargetSet):
        raise TypeError("targets must be a TargetSet")
    target_values = targets.as_mapping()
    if len(target_values) != len(targets.values):
        raise ValueError("target set contains duplicate session IDs")
    examples = []
    for feature in eligibility.features:
        if feature.identity.session_id not in target_values:
            raise ValueError(
                f"target is missing for eligible session {feature.identity.session_id}"
            )
        value = target_values[feature.identity.session_id]
        if isinstance(value, bool):
            raise TypeError("targets must be finite numeric values")
        try:
            target = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError("targets must be finite numeric values") from error
        if not math.isfinite(target) or not 0 <= target <= 1:
            raise ValueError("normalised targets must be finite and in [0,1]")
        examples.append(_Example(feature, target))
    return tuple(examples)


def _rotation_examples(
    rows: Sequence[_Example], test_fold: int
) -> tuple[tuple[_Example, ...], tuple[_Example, ...], tuple[_Example, ...]]:
    validation_fold = (test_fold + 1) % 5
    test = tuple(row for row in rows if row.feature.identity.fold == test_fold)
    validation = tuple(row for row in rows if row.feature.identity.fold == validation_fold)
    train = tuple(
        row for row in rows if row.feature.identity.fold not in {test_fold, validation_fold}
    )
    if not train or not validation or not test:
        raise ValueError("every rotation requires non-empty train, validation and test roles")
    participant_roles = [
        {row.feature.identity.participant_id for row in role} for role in (train, validation, test)
    ]
    if (
        not participant_roles[0].isdisjoint(participant_roles[1])
        or not participant_roles[0].isdisjoint(participant_roles[2])
        or not participant_roles[1].isdisjoint(participant_roles[2])
    ):
        raise RuntimeError("participant isolation failed")
    return train, validation, test


def _rotation_manifest(
    fold: int,
    train: Sequence[_Example],
    validation: Sequence[_Example],
    test: Sequence[_Example],
    fit: ArtifactPayload,
    validation_prediction: np.ndarray,
    **details: Any,
) -> dict[str, Any]:
    return {
        "outer_rotation": fold,
        "train_participant_ids": sorted(
            {row.feature.identity.participant_id for row in train}, key=int
        ),
        "validation_participant_ids": sorted(
            {row.feature.identity.participant_id for row in validation}, key=int
        ),
        "test_participant_ids": sorted(
            {row.feature.identity.participant_id for row in test}, key=int
        ),
        "train_session_count": len(train),
        "validation_session_count": len(validation),
        "test_session_count": len(test),
        "fit_artifact": {"filename": fit.filename, "sha256": fit.sha256},
        "validation_utility": utility_metrics(
            np.asarray([row.target for row in validation], dtype=np.float64),
            validation_prediction,
        ),
        **details,
    }


def _concatenated_features(
    rows: Sequence[_Example],
    *,
    dtype: np.dtype[Any],
    modalities: Sequence[str] = tuple(SUMMARY_DIMENSIONS),
) -> np.ndarray:
    if not modalities or any(modality not in SUMMARY_DIMENSIONS for modality in modalities):
        raise ValueError("concatenated feature modalities are invalid")
    return np.concatenate(
        [_modality_features(rows, modality, dtype=dtype) for modality in modalities],
        axis=1,
    )


def _modality_features(
    rows: Sequence[_Example], modality: str, *, dtype: np.dtype[Any]
) -> np.ndarray:
    return np.asarray(
        np.stack([getattr(row.feature, modality) for row in rows]),
        dtype=dtype,
    )


def _optional_correlation(left: np.ndarray, right: np.ndarray, *, rank: bool) -> float | None:
    if left.size < 2:
        return None
    x = _average_ranks(left) if rank else left
    y = _average_ranks(right) if rank else right
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if math.isfinite(value) else None


def _required_correlation(left: np.ndarray, right: np.ndarray, *, rank: bool) -> float:
    value = _optional_correlation(left, right, rank=rank)
    if value is None or not math.isfinite(value):
        raise ValueError("validation SRCC is undefined for this rotation")
    return value


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


def _finite_vector(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
    return result


def _fold_manifest_bytes(rows: Sequence[SessionIdentity]) -> bytes:
    folds = {row.participant_id: row.fold for row in rows}
    lines = ["participant_id,fold"]
    lines.extend(
        f"{participant_id},{folds[participant_id]}" for participant_id in sorted(folds, key=int)
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _parse_fold_manifest(raw: bytes) -> dict[str, int]:
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeError as error:
        raise ValueError("fold manifest is not valid UTF-8") from error
    if not lines or lines[0] != "participant_id,fold":
        raise ValueError("fold manifest header is invalid")
    result: dict[str, int] = {}
    for line in lines[1:]:
        fields = line.split(",")
        if len(fields) != 2:
            raise ValueError("fold manifest row is invalid")
        participant_id = canonical_participant_id(fields[0])
        if participant_id != fields[0] or participant_id in result:
            raise ValueError("fold manifest participant ID is invalid or duplicated")
        try:
            fold = int(fields[1])
        except ValueError as error:
            raise ValueError("fold manifest fold is invalid") from error
        if str(fold) != fields[1] or fold not in range(5):
            raise ValueError("fold manifest fold is invalid")
        result[participant_id] = fold
    canonical_rows = tuple(
        SessionIdentity(f"P{participant_id}", participant_id, "pre", fold)
        for participant_id, fold in result.items()
    )
    if not result or _fold_manifest_bytes(canonical_rows) != raw:
        raise ValueError("fold manifest rows are not canonical")
    return result


def _summary_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.dtype != np.float32 or array.ndim != 1:
        raise ValueError("feature summary digest requires a float32 vector")
    return hashlib.sha256(
        np.ascontiguousarray(array.astype("<f4", copy=False)).tobytes()
    ).hexdigest()


def _runtime_versions(*, include_torch: bool) -> dict[str, str | None]:
    packages = ["numpy", "scipy", "scikit-learn"]
    if include_torch:
        packages.append("torch")
    return {package: _package_version(package) for package in packages}


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _read_canonical_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        payload = _loads_unique_json(raw)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise FeatureArtifactError(
            "feature_failure_record_invalid", "feature failure record cannot be read"
        ) from error
    if canonical_json_bytes(payload) != raw:
        raise FeatureArtifactError(
            "feature_failure_record_invalid", "feature failure record is not canonical"
        )
    return payload


def _loads_unique_json(raw: bytes) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    payload = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=unique_object,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"invalid JSON constant {value}")
        ),
    )
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _readonly_copy(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=np.float32, copy=True)
    result.setflags(write=False)
    return result


def _require_hex(value: str, length: int, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a {length}-character lowercase hexadecimal value")


def _require_hex_error(value: Any, length: int, code: str) -> None:
    try:
        _require_hex(value, length, "value")
    except ValueError as error:
        raise FeatureArtifactError(code, "feature provenance digest is invalid") from error
