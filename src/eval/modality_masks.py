"""Deterministic inference-only modality masks for the locked Stage-A ERM scorer.

The production entry point reopens the accepted feature release and the closed
Stage-A release.  It restores only the persisted ERM scorer and train-only
scalers, verifies the ordinary locked predictions, and then masks one
training-standardised modality summary at a time.  It has no fitting,
retuning, outcome-loading or protected-annotation interface.
"""

from __future__ import annotations

import hashlib
import io
import math
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from src.data import SessionMappingContext
from src.eval.stage_c import (
    MASKED_MODALITIES,
    STAGE_C_MASK_SCHEMA_VERSION,
    StageARelease,
    load_stage_a_release,
)
from src.features import FeatureExtractionClosure
from src.frozen_json import freeze_json
from src.models.fusion import LowCapacityMultimodalScorer
from src.secure_io import AnchoredDirectory, AnchoredFile
from src.training.crossfit import (
    FIT_ARTIFACT_SCHEMA_VERSION,
    SUMMARY_DIMENSIONS,
    FeatureSummary,
    SessionIdentity,
    build_complete_pair_eligibility,
    canonical_json_bytes,
)
from src.training.proxy_training import FROZEN_FINAL_SEEDS, FROZEN_OUTER_ROTATIONS

MODALITY_MASK_INTERVENTION = "set_one_training_standardised_modality_summary_to_zero_at_inference"
UNMASKED_REPRODUCTION_ABSOLUTE_TOLERANCE = 1e-7
EXPECTED_SESSION_COUNT = 138
EXPECTED_PARTICIPANT_COUNT = 69
EXPECTED_ROW_COUNT = len(FROZEN_FINAL_SEEDS) * EXPECTED_SESSION_COUNT * len(MASKED_MODALITIES)

_MODEL_MODALITY_BY_MASK = {
    "video": "video",
    "interview_audio": "audio",
    "candidate_text": "text",
}
_MASK_TOP_FIELDS = {
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
_MASK_ROW_FIELDS = {
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


@dataclass(frozen=True, slots=True)
class ModalityMaskRelease:
    """Canonical Stage-C-compatible mask payload plus reproduction evidence."""

    payload: Mapping[str, Any]
    content: bytes
    bound_stage_a_release: StageARelease
    reproduced_prediction_count: int
    maximum_abs_reproduction_error: float
    fit_artifact_sha256s: Mapping[tuple[int, int], str]

    def __post_init__(self) -> None:
        if not isinstance(self.bound_stage_a_release, StageARelease):
            raise TypeError("modality-mask release requires its bound Stage-A release")
        validate_modality_mask_payload(
            self.payload,
            content=self.content,
            release=self.bound_stage_a_release,
        )
        if self.reproduced_prediction_count != len(FROZEN_FINAL_SEEDS) * EXPECTED_SESSION_COUNT:
            raise ValueError("modality-mask release did not reproduce all locked ERM rows")
        if (
            isinstance(self.maximum_abs_reproduction_error, bool)
            or not isinstance(self.maximum_abs_reproduction_error, (int, float))
            or not math.isfinite(float(self.maximum_abs_reproduction_error))
            or not 0
            <= float(self.maximum_abs_reproduction_error)
            <= UNMASKED_REPRODUCTION_ABSOLUTE_TOLERANCE
        ):
            raise ValueError("modality-mask release has invalid reproduction evidence")
        expected_fits = {
            (seed, fold) for seed in FROZEN_FINAL_SEEDS for fold in FROZEN_OUTER_ROTATIONS
        }
        if set(self.fit_artifact_sha256s) != expected_fits:
            raise ValueError("modality-mask release must bind all 15 ERM fit artifacts")
        for value in self.fit_artifact_sha256s.values():
            _require_sha256(value, "fit artifact checksum")
        object.__setattr__(self, "payload", freeze_json(self.payload))
        object.__setattr__(
            self,
            "fit_artifact_sha256s",
            MappingProxyType(dict(self.fit_artifact_sha256s)),
        )


@dataclass(frozen=True, slots=True)
class ModalityMaskPublication:
    """Verified immutable publication result."""

    path: Path
    state: str
    file_sha256: str
    payload_sha256: str
    row_count: int


def build_modality_mask_release(
    release: StageARelease,
    features: Sequence[FeatureSummary],
    fit_artifacts: Mapping[tuple[int, int], bytes],
) -> ModalityMaskRelease:
    """Restore all 15 locked ERM fits and build the exact mask payload.

    ``release`` and ``features`` must already come from the production closure
    validators.  This bounded function remains public so the inference and
    serialization contract can be tested without research data.
    """
    _validate_release_context(release)
    feature_by_session = _validated_feature_roster(release, features)
    fit_bindings = _expected_fit_bindings(release)
    if set(fit_artifacts) != set(fit_bindings):
        raise ValueError("modality masks require exactly the 15 locked ERM fit artifacts")

    base_by_key = _validated_base_rows(release, fit_bindings)
    masked_predictions: dict[tuple[int, str, str], float] = {}
    reproduced_count = 0
    maximum_error = 0.0

    for seed in FROZEN_FINAL_SEEDS:
        for fold in FROZEN_OUTER_ROTATIONS:
            expected_fit = fit_bindings[(seed, fold)]
            fit_bytes = fit_artifacts[(seed, fold)]
            if not isinstance(fit_bytes, bytes):
                raise TypeError("ERM fit artifacts must be immutable bytes")
            observed_sha256 = hashlib.sha256(fit_bytes).hexdigest()
            if observed_sha256 != expected_fit["sha256"]:
                raise ValueError("ERM fit artifact checksum disagrees with its prediction lock")
            model, scalers = _restore_erm_fit(
                fit_bytes,
                expected_checkpoint_epoch=expected_fit["checkpoint_epoch"],
                expected_checkpoint_validation_srcc=expected_fit["checkpoint_validation_srcc"],
                expected_scaler_session_count=expected_fit["train_session_count"],
            )
            fold_rows = [
                base_by_key[(seed, participant_id, phase)]
                for participant_id in release.participant_ids
                if release.participant_folds[participant_id] == fold
                for phase in ("pre", "post")
            ]
            values = _standardised_feature_values(
                fold_rows, feature_by_session=feature_by_session, scalers=scalers
            )
            unmasked = _predict(model, values)
            locked = np.asarray([float(row["prediction"]) for row in fold_rows], dtype=np.float64)
            errors = np.abs(unmasked - locked)
            fold_maximum = float(np.max(errors)) if errors.size else math.inf
            if fold_maximum > UNMASKED_REPRODUCTION_ABSOLUTE_TOLERANCE:
                raise ValueError(
                    "restored ERM fit did not reproduce its locked OOF predictions "
                    f"within {UNMASKED_REPRODUCTION_ABSOLUTE_TOLERANCE:g}"
                )
            maximum_error = max(maximum_error, fold_maximum)
            reproduced_count += len(fold_rows)

            for masked_modality in MASKED_MODALITIES:
                model_modality = _MODEL_MODALITY_BY_MASK[masked_modality]
                masked_values = {
                    name: (torch.zeros_like(value) if name == model_modality else value)
                    for name, value in values.items()
                }
                predictions = _predict(model, masked_values)
                for row, prediction in zip(fold_rows, predictions, strict=True):
                    key = (seed, str(row["session_id"]), masked_modality)
                    if key in masked_predictions:
                        raise RuntimeError("modality-mask inference produced a duplicate row")
                    masked_predictions[key] = float(prediction)

    rows = []
    for seed in FROZEN_FINAL_SEEDS:
        for participant_id in release.participant_ids:
            for phase in ("pre", "post"):
                base = base_by_key[(seed, participant_id, phase)]
                for modality in MASKED_MODALITIES:
                    rows.append(
                        {
                            "participant_id": participant_id,
                            "session_id": base["session_id"],
                            "source_session_id": base["source_session_id"],
                            "session_phase": phase,
                            "fixed_fold": base["fixed_fold"],
                            "seed": seed,
                            "masked_modality": modality,
                            "prediction": masked_predictions[
                                (seed, str(base["session_id"]), modality)
                            ],
                            "base_prediction_lock_id": base["prediction_lock_id"],
                            "model_fit_sha256": base["model_fit_sha256"],
                        }
                    )
    unsigned = {
        "schema_version": STAGE_C_MASK_SCHEMA_VERSION,
        "stage_a_closure_file_sha256": release.closure_file_sha256,
        "stage_a_closure_payload_sha256": release.closure_payload_sha256,
        "configuration": "erm",
        "intervention": MODALITY_MASK_INTERVENTION,
        "refit_or_retuning": "forbidden",
        "firewall_stage": "A_locked_inference_only",
        "target_fields_accessed": [],
        "protected_annotation_fields_accessed": [],
        "rows": rows,
    }
    payload = {**unsigned, "mask_payload_sha256": _payload_sha256(unsigned)}
    content = canonical_json_bytes(payload)
    result = ModalityMaskRelease(
        payload=payload,
        content=content,
        bound_stage_a_release=release,
        reproduced_prediction_count=reproduced_count,
        maximum_abs_reproduction_error=maximum_error,
        fit_artifact_sha256s={key: str(binding["sha256"]) for key, binding in fit_bindings.items()},
    )
    validate_modality_mask_payload(
        result.payload,
        content=result.content,
        release=release,
    )
    return result


def build_research_modality_mask_release(
    *,
    stage_a_dir: str | Path,
    stage_a_closure_file_sha256: str,
    stage_a_closure_payload_sha256: str,
    feature_closure: FeatureExtractionClosure,
    session_mapping: SessionMappingContext,
) -> ModalityMaskRelease:
    """Reopen exact research releases and generate inference-only masks."""
    _require_sha256(stage_a_closure_file_sha256, "Stage-A closure file checksum")
    _require_sha256(stage_a_closure_payload_sha256, "Stage-A closure payload checksum")
    if not isinstance(feature_closure, FeatureExtractionClosure):
        raise TypeError("research modality masks require a verified feature closure")
    if not isinstance(session_mapping, SessionMappingContext):
        raise TypeError("research modality masks require a verified session mapping")

    release = load_stage_a_release(stage_a_dir)
    if (
        release.closure_file_sha256 != stage_a_closure_file_sha256
        or release.closure_payload_sha256 != stage_a_closure_payload_sha256
    ):
        raise ValueError("Stage-A closure checksum does not match the requested release")
    context = _validate_release_context(release)
    if context["session_mapping"] != session_mapping.provenance:
        raise ValueError("Stage-A and feature session-mapping provenance disagree")
    if (
        context["feature_extraction_closure_file_sha256"] != feature_closure.closure_file_sha256
        or context["feature_extraction_closure_payload_sha256"]
        != feature_closure.closure_payload_sha256
    ):
        raise ValueError("Stage-A does not bind the supplied feature release")

    identities = _closure_identities(feature_closure)
    eligibility = build_complete_pair_eligibility(
        identities,
        None,
        feature_closure=feature_closure,
        session_mapping=session_mapping,
        require_full_corpus=True,
    )
    _validate_eligibility_binding(eligibility, context)
    feature_closure.verify_current_files()

    observations: dict[str, AnchoredFile]
    with AnchoredDirectory.open(stage_a_dir, directory_mode=0o700) as stage_a:
        fit_artifacts, observations = _read_bound_erm_artifacts(stage_a, release)
        result = build_modality_mask_release(
            release,
            eligibility.features,
            fit_artifacts,
        )
        for filename, observed in observations.items():
            stage_a.verify_file(filename, observed, expected_mode=0o600)
        stage_a.verify_path_identity()
    feature_closure.verify_current_files()
    session_mapping.verify_current_files()

    reopened = load_stage_a_release(stage_a_dir)
    if (
        reopened.closure_file_sha256 != release.closure_file_sha256
        or reopened.closure_payload_sha256 != release.closure_payload_sha256
        or reopened.learned_rows[("erm", 1)] != release.learned_rows[("erm", 1)]
        or reopened.learned_rows[("erm", 11)] != release.learned_rows[("erm", 11)]
        or reopened.learned_rows[("erm", 21)] != release.learned_rows[("erm", 21)]
    ):
        raise RuntimeError("Stage-A release changed during modality-mask generation")
    return result


def write_modality_mask_release(
    output_path: str | Path,
    release: ModalityMaskRelease,
) -> ModalityMaskPublication:
    """Atomically publish mode-0600 bytes, accepting only identical resume."""
    if not isinstance(release, ModalityMaskRelease):
        raise TypeError("release must be a ModalityMaskRelease")
    validate_modality_mask_payload(
        release.payload,
        content=release.content,
        release=release.bound_stage_a_release,
    )
    supplied = Path(output_path).expanduser()
    if not supplied.is_absolute():
        supplied = Path.cwd() / supplied
    if supplied.name in {"", ".", ".."} or any(part in {".", ".."} for part in supplied.parts[1:]):
        raise ValueError("modality-mask output path is unsafe")
    with AnchoredDirectory.open(supplied.parent, create=True, directory_mode=0o700) as output:
        with output.exclusive_lock():
            existed = supplied.name in output.list_names()
            written = output.publish_bytes(
                supplied.name,
                release.content,
                mode=0o600,
                allow_existing=True,
            )
            validate_modality_mask_payload(
                _canonical_object(written.content),
                content=written.content,
                release=release.bound_stage_a_release,
            )
            output.verify_file(supplied.name, written, expected_mode=0o600)
    return ModalityMaskPublication(
        path=supplied,
        state="resumed" if existed else "published",
        file_sha256=written.sha256,
        payload_sha256=str(release.payload["mask_payload_sha256"]),
        row_count=len(release.payload["rows"]),
    )


def validate_modality_mask_payload(
    payload: Mapping[str, Any],
    *,
    content: bytes | None = None,
    release: StageARelease | None = None,
) -> None:
    """Validate the exact self-checking contract consumed by Stage C."""
    if not isinstance(payload, Mapping) or set(payload) != _MASK_TOP_FIELDS:
        raise ValueError("modality-mask payload top-level schema drifted")
    signed = payload.get("mask_payload_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "mask_payload_sha256"}
    if (
        payload.get("schema_version") != STAGE_C_MASK_SCHEMA_VERSION
        or payload.get("configuration") != "erm"
        or payload.get("intervention") != MODALITY_MASK_INTERVENTION
        or payload.get("refit_or_retuning") != "forbidden"
        or payload.get("firewall_stage") != "A_locked_inference_only"
        or payload.get("target_fields_accessed") != []
        or payload.get("protected_annotation_fields_accessed") != []
        or not isinstance(payload.get("rows"), list)
        or signed != _payload_sha256(unsigned)
    ):
        raise ValueError("modality-mask payload fixed contract or signature drifted")
    _require_sha256(payload.get("stage_a_closure_file_sha256"), "Stage-A closure file checksum")
    _require_sha256(
        payload.get("stage_a_closure_payload_sha256"),
        "Stage-A closure payload checksum",
    )
    if content is not None and (
        not isinstance(content, bytes) or canonical_json_bytes(payload) != content
    ):
        raise ValueError("modality-mask payload is not canonical JSON")
    if len(payload["rows"]) != EXPECTED_ROW_COUNT:
        raise ValueError("modality-mask payload must contain exactly 1,242 rows")

    bound_rows = _bound_release_rows(release) if release is not None else None
    observed: set[tuple[int, str, str, str]] = set()
    for row in payload["rows"]:
        if not isinstance(row, Mapping) or set(row) != _MASK_ROW_FIELDS:
            raise ValueError("modality-mask row schema drifted")
        seed = row["seed"]
        participant_id = row["participant_id"]
        phase = row["session_phase"]
        modality = row["masked_modality"]
        key = (seed, participant_id, phase, modality)
        prediction = row["prediction"]
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed not in FROZEN_FINAL_SEEDS
            or modality not in MASKED_MODALITIES
            or phase not in {"pre", "post"}
            or isinstance(row["fixed_fold"], bool)
            or row["fixed_fold"] not in FROZEN_OUTER_ROTATIONS
            or not all(
                isinstance(row[field], str) and row[field]
                for field in (
                    "participant_id",
                    "session_id",
                    "source_session_id",
                )
            )
            or isinstance(prediction, bool)
            or not isinstance(prediction, (int, float))
            or not math.isfinite(float(prediction))
            or not 0 <= float(prediction) <= 1
            or key in observed
        ):
            raise ValueError("modality-mask row identity or prediction drifted")
        _require_sha256(row["base_prediction_lock_id"], "base prediction lock ID")
        _require_sha256(row["model_fit_sha256"], "model fit checksum")
        if bound_rows is not None:
            base = bound_rows.get((seed, participant_id, phase))
            if (
                base is None
                or row["session_id"] != base["session_id"]
                or row["source_session_id"] != base["source_session_id"]
                or row["fixed_fold"] != base["fixed_fold"]
                or row["base_prediction_lock_id"] != base["prediction_lock_id"]
                or row["model_fit_sha256"] != base["model_fit_sha256"]
            ):
                raise ValueError("modality-mask row does not match its bound locked ERM prediction")
        observed.add(key)
    if release is not None:
        if (
            payload["stage_a_closure_file_sha256"] != release.closure_file_sha256
            or payload["stage_a_closure_payload_sha256"] != release.closure_payload_sha256
        ):
            raise ValueError("modality-mask payload does not bind the supplied Stage-A release")
        expected = {
            (seed, participant_id, phase, modality)
            for seed in FROZEN_FINAL_SEEDS
            for participant_id in release.participant_ids
            for phase in ("pre", "post")
            for modality in MASKED_MODALITIES
        }
        if observed != expected:
            raise ValueError("modality-mask rows do not exactly cover the Stage-A roster")


def _bound_release_rows(
    release: StageARelease,
) -> dict[tuple[int, str, str], Mapping[str, Any]]:
    """Return exact locked ERM row bindings for publication validation."""
    fit_bindings = _expected_fit_bindings(release)
    result: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for seed in FROZEN_FINAL_SEEDS:
        lock = release.lock_manifests.get(("erm", seed))
        rows = release.learned_rows.get(("erm", seed))
        if not isinstance(lock, Mapping) or not isinstance(rows, tuple):
            raise ValueError("bound Stage-A release lacks a locked ERM seed")
        prediction_lock_id = lock.get("prediction_lock_id")
        _require_sha256(prediction_lock_id, "bound ERM prediction lock ID")
        for row in rows:
            participant_id = row.get("participant_id")
            phase = row.get("session_phase")
            fold = row.get("fixed_fold")
            key = (seed, participant_id, phase)
            fit = fit_bindings.get((seed, fold))
            if (
                not isinstance(participant_id, str)
                or phase not in {"pre", "post"}
                or key in result
                or fit is None
                or release.participant_folds.get(participant_id) != fold
                or row.get("prediction_lock_id") != prediction_lock_id
                or row.get("model_fit_sha256") != fit["sha256"]
            ):
                raise ValueError("bound Stage-A ERM row or fit binding drifted")
            result[key] = row
    expected = {
        (seed, participant_id, phase)
        for seed in FROZEN_FINAL_SEEDS
        for participant_id in release.participant_ids
        for phase in ("pre", "post")
    }
    if set(result) != expected:
        raise ValueError("bound Stage-A ERM rows do not exactly cover the paired roster")
    return result


def _validate_release_context(release: StageARelease) -> Mapping[str, Any]:
    if not isinstance(release, StageARelease):
        raise TypeError("release must be a validated StageARelease")
    _require_sha256(release.closure_file_sha256, "Stage-A closure file checksum")
    _require_sha256(release.closure_payload_sha256, "Stage-A closure payload checksum")
    if release.participant_count != EXPECTED_PARTICIPANT_COUNT:
        raise ValueError("modality masks require the exact 69-participant Stage-A roster")
    if len(release.session_keys) != EXPECTED_SESSION_COUNT:
        raise ValueError("modality masks require the exact 138-session Stage-A roster")
    if set(release.participant_folds) != set(release.participant_ids) or set(
        release.participant_folds.values()
    ) != set(FROZEN_OUTER_ROTATIONS):
        raise ValueError("Stage-A participant-fold registry is incomplete")
    closure = release.closure
    context = closure.get("context")
    if (
        closure.get("execution_mode") != "research"
        or closure.get("firewall_stage") != "A"
        or closure.get("protected_annotation_fields_accessed") != []
        or not isinstance(context, Mapping)
    ):
        raise ValueError("modality masks require a closed annotation-blind Stage-A release")
    for field in (
        "config_sha256",
        "code_source_tree_sha256",
        "feature_contract_sha256",
        "preprocessing_sha256",
        "feature_run_contract_sha256",
        "feature_config_sha256",
        "feature_extraction_closure_file_sha256",
        "feature_extraction_closure_payload_sha256",
        "eligibility_manifest_sha256",
        "fold_manifest_sha256",
    ):
        _require_sha256(context.get(field), f"Stage-A context {field}")
    commit = context.get("code_commit_sha")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValueError("Stage-A context code commit is invalid")
    if (
        context.get("eligible_participant_count") != EXPECTED_PARTICIPANT_COUNT
        or context.get("expected_oof_row_count") != EXPECTED_SESSION_COUNT
    ):
        raise ValueError("Stage-A context roster counts drifted")
    return context


def _validated_feature_roster(
    release: StageARelease,
    features: Sequence[FeatureSummary],
) -> dict[str, FeatureSummary]:
    context = release.closure["context"]
    expected_sessions = {
        str(row["session_id"]): (
            str(row["participant_id"]),
            str(row["source_session_id"]),
            str(row["session_phase"]),
            int(row["fixed_fold"]),
        )
        for row in release.learned_rows[("erm", FROZEN_FINAL_SEEDS[0])]
    }
    result: dict[str, FeatureSummary] = {}
    for feature in features:
        if not isinstance(feature, FeatureSummary):
            raise TypeError("modality-mask features must be validated FeatureSummary values")
        identity = feature.identity
        expected = expected_sessions.get(identity.session_id)
        if (
            expected is None
            or identity.session_id in result
            or expected
            != (
                identity.participant_id,
                identity.source_session_id,
                identity.condition,
                identity.fold,
            )
            or release.participant_folds.get(identity.participant_id) != identity.fold
            or feature.feature_contract_sha256 != context["feature_contract_sha256"]
            or feature.preprocessing_sha256 != context["preprocessing_sha256"]
            or feature.run_contract_sha256 != context["feature_run_contract_sha256"]
            or feature.config_sha256 != context["feature_config_sha256"]
        ):
            raise ValueError("feature summary does not match the locked Stage-A context")
        for modality, width in SUMMARY_DIMENSIONS.items():
            value = np.asarray(getattr(feature, modality))
            if value.shape != (width,) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError("feature summary is invalid for modality-mask inference")
        result[identity.session_id] = feature
    if set(result) != set(expected_sessions):
        raise ValueError("feature summaries do not exactly cover the Stage-A session roster")
    return result


def _expected_fit_bindings(
    release: StageARelease,
) -> dict[tuple[int, int], dict[str, Any]]:
    result: dict[tuple[int, int], dict[str, Any]] = {}
    context = release.closure["context"]
    provenance_fields = (
        "config_sha256",
        "code_commit_sha",
        "code_source_tree_sha256",
        "source_date_epoch",
        "feature_contract_sha256",
        "preprocessing_sha256",
        "feature_run_contract_sha256",
        "feature_config_sha256",
        "feature_extraction_closure_file_sha256",
        "feature_extraction_closure_payload_sha256",
    )
    for seed in FROZEN_FINAL_SEEDS:
        lock = release.lock_manifests.get(("erm", seed))
        if (
            not isinstance(lock, Mapping)
            or lock.get("configuration") != "erm"
            or lock.get("seed") != seed
            or lock.get("execution_mode") != "research"
            or lock.get("firewall_stage") != "A"
            or lock.get("protected_annotation_fields_accessed") != []
            or any(lock.get(field) != context[field] for field in provenance_fields)
        ):
            raise ValueError("ERM prediction lock provenance disagrees with Stage A")
        model_config = lock.get("model_config")
        if (
            not isinstance(model_config, Mapping)
            or model_config.get("configuration") != "erm"
            or model_config.get("architecture") != "low_capacity_multimodal_scorer_26721_parameters"
            or model_config.get("trainable_parameter_count") != 26_721
            or model_config.get("modality_scalers")
            != "separate_training_session_row_standard_scalers"
            or model_config.get("prediction_postprocessing") != "none_sigmoid_is_model_output"
            or model_config.get("device") != "cpu"
        ):
            raise ValueError("ERM prediction lock model configuration drifted")
        rotations = lock.get("rotations")
        if not isinstance(rotations, list) or len(rotations) != len(FROZEN_OUTER_ROTATIONS):
            raise ValueError("ERM prediction lock rotation registry is incomplete")
        for fold, rotation in zip(FROZEN_OUTER_ROTATIONS, rotations, strict=True):
            fit = rotation.get("fit_artifact") if isinstance(rotation, Mapping) else None
            if (
                not isinstance(rotation, Mapping)
                or rotation.get("outer_rotation") != fold
                or not isinstance(fit, Mapping)
                or set(fit) != {"filename", "sha256"}
                or fit.get("filename") != f"erm.seed-{seed}.fold-{fold}.fit.npz"
            ):
                raise ValueError("ERM fit registry disagrees with the locked rotation")
            _require_sha256(fit.get("sha256"), "ERM fit checksum")
            train_count = rotation.get("train_session_count")
            checkpoint_epoch = rotation.get("checkpoint_epoch")
            checkpoint_score = rotation.get("checkpoint_validation_srcc")
            if (
                isinstance(train_count, bool)
                or not isinstance(train_count, int)
                or train_count <= 0
                or isinstance(checkpoint_epoch, bool)
                or not isinstance(checkpoint_epoch, int)
                or checkpoint_epoch <= 0
                or isinstance(checkpoint_score, bool)
                or not isinstance(checkpoint_score, (int, float))
                or not math.isfinite(float(checkpoint_score))
                or not -1 <= float(checkpoint_score) <= 1
            ):
                raise ValueError("ERM locked rotation checkpoint or scaler count is invalid")
            result[(seed, fold)] = {
                "filename": fit["filename"],
                "sha256": fit["sha256"],
                "train_session_count": train_count,
                "checkpoint_epoch": checkpoint_epoch,
                "checkpoint_validation_srcc": float(checkpoint_score),
            }
    return result


def _validated_base_rows(
    release: StageARelease,
    fit_bindings: Mapping[tuple[int, int], Mapping[str, Any]],
) -> dict[tuple[int, str, str], Mapping[str, Any]]:
    first_seed_rows = release.learned_rows.get(("erm", FROZEN_FINAL_SEEDS[0]))
    if not isinstance(first_seed_rows, tuple):
        raise ValueError("Stage-A release lacks the first locked ERM seed")
    expected_sessions = {
        (str(row["participant_id"]), str(row["session_id"])): (
            str(row["source_session_id"]),
            str(row["session_phase"]),
            int(row["fixed_fold"]),
        )
        for row in first_seed_rows
    }
    if set(release.session_keys) != {
        (participant_id, phase, session_id)
        for (participant_id, session_id), (
            _source_session_id,
            phase,
            _fold,
        ) in expected_sessions.items()
    }:
        raise ValueError("Stage-A paired session-key registry drifted")
    result: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for seed in FROZEN_FINAL_SEEDS:
        rows = release.learned_rows.get(("erm", seed))
        lock = release.lock_manifests[("erm", seed)]
        if not isinstance(rows, tuple) or len(rows) != EXPECTED_SESSION_COUNT:
            raise ValueError("ERM locked rows do not cover all 138 sessions")
        for row in rows:
            participant_id = str(row.get("participant_id"))
            session_id = str(row.get("session_id"))
            phase = row.get("session_phase")
            fold = row.get("fixed_fold")
            prediction = row.get("prediction")
            key = (seed, participant_id, str(phase))
            fit = fit_bindings.get((seed, fold))
            if (
                phase not in {"pre", "post"}
                or key in result
                or fit is None
                or expected_sessions.get((participant_id, session_id))
                != (row.get("source_session_id"), phase, fold)
                or release.participant_folds.get(participant_id) != fold
                or row.get("prediction_lock_id") != lock.get("prediction_lock_id")
                or row.get("model_fit_sha256") != fit["sha256"]
                or row.get("model_config_sha256") != lock.get("model_config_sha256")
                or isinstance(prediction, bool)
                or not isinstance(prediction, (int, float))
                or not math.isfinite(float(prediction))
                or not 0 <= float(prediction) <= 1
            ):
                raise ValueError("locked ERM prediction row identity or fit binding drifted")
            result[key] = row
    expected = {
        (seed, participant_id, phase)
        for seed in FROZEN_FINAL_SEEDS
        for participant_id in release.participant_ids
        for phase in ("pre", "post")
    }
    if set(result) != expected:
        raise ValueError("locked ERM predictions do not exactly cover every participant phase")
    return result


def _restore_erm_fit(
    content: bytes,
    *,
    expected_checkpoint_epoch: int,
    expected_checkpoint_validation_srcc: float,
    expected_scaler_session_count: int,
) -> tuple[LowCapacityMultimodalScorer, dict[str, tuple[np.ndarray, np.ndarray]]]:
    try:
        with np.load(io.BytesIO(content), allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)):
                raise ValueError("ERM fit archive contains duplicate arrays")
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError("ERM fit artifact cannot be read safely") from error
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("ERM fit artifact may not contain object arrays")
    schema = arrays.get("schema_version_utf8")
    checkpoint_epoch = arrays.get("checkpoint_epoch")
    checkpoint_score = arrays.get("checkpoint_validation_srcc")
    if (
        schema is None
        or schema.dtype != np.uint8
        or schema.tobytes().decode("utf-8") != FIT_ARTIFACT_SCHEMA_VERSION
        or checkpoint_epoch is None
        or checkpoint_epoch.dtype != np.int64
        or checkpoint_epoch.shape != ()
        or int(checkpoint_epoch) != expected_checkpoint_epoch
        or checkpoint_score is None
        or checkpoint_score.dtype != np.float64
        or checkpoint_score.shape != ()
        or not math.isclose(
            float(checkpoint_score),
            expected_checkpoint_validation_srcc,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise ValueError("ERM fit checkpoint metadata disagrees with its prediction lock")

    scalers: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    expected_scaler_keys: set[str] = set()
    for modality, width in SUMMARY_DIMENSIONS.items():
        keys = {
            field: f"scaler.{modality}.{field}"
            for field in ("mean", "scale", "var", "n_samples_seen")
        }
        expected_scaler_keys.update(keys.values())
        mean = arrays.get(keys["mean"])
        scale = arrays.get(keys["scale"])
        variance = arrays.get(keys["var"])
        samples = arrays.get(keys["n_samples_seen"])
        if (
            mean is None
            or scale is None
            or variance is None
            or samples is None
            or mean.dtype != np.float64
            or scale.dtype != np.float64
            or variance.dtype != np.float64
            or samples.dtype != np.int64
            or mean.shape != (width,)
            or scale.shape != (width,)
            or variance.shape != (width,)
            or samples.shape != ()
            or int(samples) != expected_scaler_session_count
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or not np.isfinite(variance).all()
            or not np.all(scale > 0)
            or not np.all(variance >= 0)
            or not np.array_equal(scale, np.where(variance == 0, 1.0, np.sqrt(variance)))
        ):
            raise ValueError("ERM train-only scaler state is invalid")
        scalers[modality] = (mean, scale)
    observed_scaler_keys = {name for name in arrays if name.startswith("scaler.")}
    if observed_scaler_keys != expected_scaler_keys:
        raise ValueError("ERM fit artifact scaler registry drifted")

    model = LowCapacityMultimodalScorer().to("cpu")
    expected_state = model.state_dict()
    observed_model_keys = {name for name in arrays if name.startswith("model.")}
    expected_model_keys = {f"model.{name}" for name in expected_state}
    if observed_model_keys != expected_model_keys:
        raise ValueError("ERM fit artifact model-state registry drifted")
    restored: dict[str, torch.Tensor] = {}
    for name, expected in expected_state.items():
        value = arrays[f"model.{name}"]
        if (
            value.dtype != np.float32
            or value.shape != tuple(expected.shape)
            or not np.isfinite(value).all()
        ):
            raise ValueError("ERM fit artifact contains an invalid model tensor")
        restored[name] = torch.from_numpy(value)
    model.load_state_dict(restored, strict=True)
    model.eval()
    return model, scalers


def _standardised_feature_values(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_by_session: Mapping[str, FeatureSummary],
    scalers: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, torch.Tensor]:
    result = {}
    for modality in SUMMARY_DIMENSIONS:
        matrix = np.stack(
            [getattr(feature_by_session[str(row["session_id"])], modality) for row in rows]
        ).astype(np.float64, copy=False)
        mean, scale = scalers[modality]
        standardised = np.asarray((matrix - mean) / scale, dtype=np.float32)
        if not np.isfinite(standardised).all():
            raise ValueError("standardised modality summaries are non-finite")
        result[modality] = torch.from_numpy(standardised)
    return result


def _predict(
    model: LowCapacityMultimodalScorer,
    values: Mapping[str, torch.Tensor],
) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        prediction = model(values["video"], values["audio"], values["text"])
    output = prediction.squeeze(1).detach().cpu().numpy().astype(np.float64)
    if output.ndim != 1 or not np.isfinite(output).all() or np.any((output < 0) | (output > 1)):
        raise RuntimeError("restored ERM scorer produced invalid predictions")
    return output


def _closure_identities(
    feature_closure: FeatureExtractionClosure,
) -> tuple[SessionIdentity, ...]:
    records = feature_closure.manifest.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_SESSION_COUNT:
        raise ValueError("feature closure does not contain the exact 138-session roster")
    identities = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("feature closure record is invalid")
        identities.append(
            SessionIdentity(
                session_id=str(record["session_id"]),
                participant_id=str(record["participant_id"]),
                condition=str(record["condition"]),  # type: ignore[arg-type]
                fold=int(record["fold"]),
                source_session_id=str(record["source_session_id"]),
            )
        )
    return tuple(identities)


def _validate_eligibility_binding(
    eligibility: Any,
    context: Mapping[str, Any],
) -> None:
    expected = {
        "eligibility_manifest_sha256": eligibility.manifest_sha256,
        "fold_manifest_sha256": eligibility.fold_manifest_sha256,
        "feature_contract_sha256": eligibility.feature_contract_sha256,
        "preprocessing_sha256": eligibility.preprocessing_sha256,
        "feature_run_contract_sha256": eligibility.manifest["feature_run_contract_sha256"],
        "feature_config_sha256": eligibility.feature_config_sha256,
        "feature_extraction_closure_file_sha256": eligibility.manifest[
            "feature_extraction_closure_file_sha256"
        ],
        "feature_extraction_closure_payload_sha256": eligibility.manifest[
            "feature_extraction_closure_payload_sha256"
        ],
        "eligible_participant_count": len(eligibility.participant_ids),
        "expected_oof_row_count": len(eligibility.features),
    }
    if any(context.get(field) != value for field, value in expected.items()):
        raise ValueError("reopened feature eligibility disagrees with Stage-A context")


def _read_bound_erm_artifacts(
    stage_a: AnchoredDirectory,
    release: StageARelease,
) -> tuple[dict[tuple[int, int], bytes], dict[str, AnchoredFile]]:
    observations: dict[str, AnchoredFile] = {}
    closure = stage_a.read_bytes("stage-a-closure.json", expected_mode=0o600)
    if closure.sha256 != release.closure_file_sha256:
        raise ValueError("anchored Stage-A closure checksum failed")
    observations["stage-a-closure.json"] = closure
    records = {
        (record["configuration"], record["seed"]): record
        for record in release.closure["learned_bundles"]
    }
    fit_artifacts: dict[tuple[int, int], bytes] = {}
    for seed in FROZEN_FINAL_SEEDS:
        record = records.get(("erm", seed))
        if not isinstance(record, Mapping):
            raise ValueError("Stage-A closure lacks a locked ERM seed bundle")
        for filename_field, checksum_field in (
            ("prediction_lock_filename", "prediction_lock_file_sha256"),
            ("prediction_filename", "prediction_file_sha256"),
        ):
            filename = record[filename_field]
            observed = stage_a.read_bytes(filename, expected_mode=0o600)
            if observed.sha256 != record[checksum_field]:
                raise ValueError("anchored ERM prediction bundle checksum failed")
            observations[filename] = observed
        fits = record.get("fit_artifacts")
        if not isinstance(fits, list) or len(fits) != len(FROZEN_OUTER_ROTATIONS):
            raise ValueError("Stage-A closure ERM fit inventory is incomplete")
        for fold, fit in zip(FROZEN_OUTER_ROTATIONS, fits, strict=True):
            if not isinstance(fit, Mapping) or fit.get("outer_fold") != fold:
                raise ValueError("Stage-A closure ERM fit order drifted")
            filename = fit["filename"]
            observed = stage_a.read_bytes(filename, expected_mode=0o600)
            if observed.sha256 != fit["sha256"]:
                raise ValueError("anchored ERM fit checksum failed")
            observations[filename] = observed
            fit_artifacts[(seed, fold)] = observed.content
    stage_a.verify_path_identity()
    return fit_artifacts, observations


def _canonical_object(content: bytes) -> dict[str, Any]:
    import json

    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("modality-mask JSON contains duplicate keys")
            result[key] = value
        return result

    try:
        payload = json.loads(content.decode("utf-8"), object_pairs_hook=reject_pairs)
    except (UnicodeError, ValueError) as error:
        raise ValueError("modality-mask JSON cannot be read") from error
    if not isinstance(payload, dict) or canonical_json_bytes(payload) != content:
        raise ValueError("modality-mask JSON is not canonical")
    return payload


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _require_sha256(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
