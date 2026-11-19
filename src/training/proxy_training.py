"""Frozen proxy-balanced Stage-A training on the common multimodal scorer."""

from __future__ import annotations

import copy
import io
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.features.pseudo_attributes import BalancedParticipantSampler
from src.losses.fairness import total_loss
from src.models.fusion import LowCapacityMultimodalScorer

from .crossfit import (
    FIREWALLED_LEARNED_CONFIGURATIONS,
    FIT_ARTIFACT_SCHEMA_VERSION,
    PROXY_LEARNED_CONFIGURATIONS,
    SUMMARY_DIMENSIONS,
    ArtifactPayload,
    EligibilityResult,
    PredictionBundle,
    RunProvenance,
    TargetSet,
    _adamw,
    _bind_targets,
    _execution_mode,
    _lock_bundle,
    _modality_features,
    _optimizer_arrays,
    _participant_row_indices,
    _payload_sha256,
    _prediction_rows,
    _required_correlation,
    _rotation_examples,
    _rotation_manifest,
    _scaled_modalities,
    _set_training_determinism,
    _torch_predict,
    canonical_json_bytes,
    deterministic_npz_bytes,
)
from .proxy_artifacts import (
    PROXY_ARTIFACT_SCHEMA_VERSION,
    ProxyArtifact,
    ProxySource,
    _UnvalidatedProxyArtifact,
    _validate_proxy_artifact_structure,
    build_fold_proxy_artifact,
    build_random_proxy_artifact,
    validate_proxy_artifact,
)

ProxyTrainingMode = Literal["sampling_only", "wasserstein1"]

PROXY_TRAINING_ARTIFACT_SCHEMA_VERSION = "fair-avi-mit-proxy-training-fit-1"
PROXY_SCHEDULE_SCHEMA_VERSION = "fair-avi-mit-proxy-training-schedule-1"
PROXY_DIAGNOSTICS_SCHEMA_VERSION = "fair-avi-mit-proxy-training-diagnostics-1"
FROZEN_FINAL_SEEDS = (1, 11, 21)
FROZEN_OUTER_ROTATIONS = (0, 1, 2, 3, 4)
PROXY_PARTICIPANTS_PER_GROUP = 8
PROXY_PARTICIPANTS_PER_BATCH = 16
PROXY_SESSION_ROWS_PER_BATCH = 32
WASSERSTEIN_COEFFICIENT = 0.02

_SCHEDULE_FIELDS = {
    "schema_version",
    "configuration",
    "seed",
    "outer_rotation",
    "proxy_source",
    "eligibility_manifest_sha256",
    "proxy_manifest",
    "proxy_fit",
    "sampler",
    "optimizer_step_budget_formula",
    "optimizer_steps_per_epoch",
    "epochs_completed",
    "draw_count",
    "checkpoint_draws_consumed",
    "checkpoint_resume",
    "draws",
    "protected_annotation_fields_accessed",
}
_DIAGNOSTICS_FIELDS = {
    "schema_version",
    "configuration",
    "seed",
    "outer_rotation",
    "proxy_source",
    "sampling_mode",
    "lambda_w",
    "model_config_sha256",
    "proxy_manifest_sha256",
    "proxy_fit_sha256",
    "schedule_sha256",
    "checkpoint",
    "epochs_completed",
    "stopped_early",
    "training_history",
    "minibatches",
    "validation_sampling",
    "test_sampling",
    "protected_annotation_fields_accessed",
}
_MINIBATCH_DIAGNOSTIC_FIELDS = {
    "epoch",
    "step_in_epoch",
    "optimizer_step",
    "schedule_index",
    "participant_count",
    "session_row_count",
    "participant_group_counts",
    "session_row_group_counts",
    "unique_participant_count",
    "repeated_draw_count",
    "unique_participant_rate",
    "duplication_rate",
    "group_evidence",
    "task_mse",
    "wasserstein1",
    "lambda_w",
    "total_objective",
    "objective_received_proxy_groups",
    "preclip_gradient_norm",
}
_TRAINING_HISTORY_FIELDS = {
    "epoch",
    "optimizer_steps",
    "cumulative_optimizer_steps",
    "sampler_draws_consumed",
    "training_task_mse_mean",
    "training_wasserstein1_mean",
    "training_total_objective_mean",
    "preclip_gradient_norm_mean",
    "validation_srcc",
    "validation_mse",
    "validation_sampling",
}
_MODEL_STATE_SHAPES = {
    "video.linear.weight": (8, 1024),
    "video.linear.bias": (8,),
    "video.norm.weight": (8,),
    "video.norm.bias": (8,),
    "audio.linear.weight": (8, 1536),
    "audio.linear.bias": (8,),
    "audio.norm.weight": (8,),
    "audio.norm.bias": (8,),
    "text.linear.weight": (8, 768),
    "text.linear.bias": (8,),
    "text.norm.weight": (8,),
    "text.norm.bias": (8,),
    "head.weight": (1, 24),
    "head.bias": (1,),
}
_PROXY_PREDICTION_FIELDS = {
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


@dataclass(frozen=True, slots=True)
class ProxyConfigurationSpec:
    """One exact source/objective pairing in the frozen Stage-A registry."""

    source: ProxySource
    training_mode: ProxyTrainingMode
    lambda_w: float


PROXY_CONFIGURATION_SPECS: Mapping[str, ProxyConfigurationSpec] = MappingProxyType(
    {
        "face_sampling_only": ProxyConfigurationSpec("visual", "sampling_only", 0.0),
        "audio_session_sampling_only": ProxyConfigurationSpec(
            "interview_audio", "sampling_only", 0.0
        ),
        "random_proxy_w1": ProxyConfigurationSpec(
            "random", "wasserstein1", WASSERSTEIN_COEFFICIENT
        ),
        "face_w1": ProxyConfigurationSpec(
            "visual", "wasserstein1", WASSERSTEIN_COEFFICIENT
        ),
        "audio_session_w1": ProxyConfigurationSpec(
            "interview_audio", "wasserstein1", WASSERSTEIN_COEFFICIENT
        ),
    }
)

if tuple(PROXY_CONFIGURATION_SPECS) != PROXY_LEARNED_CONFIGURATIONS:
    raise RuntimeError("proxy configuration registry disagrees with the cross-fit registry")


@dataclass(frozen=True, slots=True)
class _ProxyFit:
    validation_prediction: np.ndarray
    test_prediction: np.ndarray
    checkpoint_epoch: int
    checkpoint_validation_srcc: float
    checkpoint_optimizer_step: int
    checkpoint_sampler_draws_consumed: int
    epochs_completed: int
    stopped_early: bool
    optimizer_steps_per_epoch: int
    training_history: tuple[dict[str, Any], ...]
    minibatch_diagnostics: tuple[dict[str, Any], ...]
    used_schedule: tuple[dict[str, Any], ...]
    artifact_arrays: Mapping[str, np.ndarray]


def matched_optimizer_steps_per_epoch(training_participant_count: int) -> int:
    """Return the exact ADR-041 update budget for one completed proxy epoch."""
    if (
        isinstance(training_participant_count, bool)
        or not isinstance(training_participant_count, int)
        or training_participant_count <= 0
    ):
        raise ValueError("training_participant_count must be a positive integer")
    return math.ceil(training_participant_count / PROXY_PARTICIPANTS_PER_BATCH)


def frozen_stage_a_fit_registry() -> tuple[tuple[str, int, int], ...]:
    """Return the exact seven-configuration, five-rotation, three-seed fit registry."""
    return tuple(
        (configuration, fold, seed)
        for configuration in FIREWALLED_LEARNED_CONFIGURATIONS
        for fold in FROZEN_OUTER_ROTATIONS
        for seed in FROZEN_FINAL_SEEDS
    )


def run_proxy_configuration(
    eligibility: EligibilityResult,
    targets: TargetSet,
    *,
    configuration: str,
    seed: int,
    provenance: RunProvenance,
    synthetic_smoke_epoch_cap: int | None = None,
    global_participant_ids: Sequence[str] | np.ndarray | None = None,
) -> PredictionBundle:
    """Run one of the five frozen proxy-balanced configurations on CPU."""
    if configuration not in PROXY_CONFIGURATION_SPECS:
        raise ValueError("configuration must be one of the five frozen proxy configurations")
    if isinstance(seed, bool) or seed not in FROZEN_FINAL_SEEDS:
        raise ValueError("seed must be one of the frozen final seeds 1, 11 or 21")
    if synthetic_smoke_epoch_cap is not None and (
        isinstance(synthetic_smoke_epoch_cap, bool)
        or not 1 <= synthetic_smoke_epoch_cap <= 200
    ):
        raise ValueError("synthetic_smoke_epoch_cap must be an integer from 1 to 200")
    spec = PROXY_CONFIGURATION_SPECS[configuration]
    if spec.source != "random" and global_participant_ids is not None:
        raise ValueError("a global participant roster is valid only for the random proxy")
    maximum_epochs = synthetic_smoke_epoch_cap or 200
    execution_mode = _execution_mode(
        eligibility,
        targets,
        provenance,
        synthetic_requested=synthetic_smoke_epoch_cap is not None,
    )
    examples = _bind_targets(eligibility, targets)
    model_config = _model_config(configuration, spec)
    model_config_sha256 = _payload_sha256(model_config)
    rows: list[dict[str, Any]] = []
    rotations: list[dict[str, Any]] = []
    artifacts: list[ArtifactPayload] = []

    for fold in FROZEN_OUTER_ROTATIONS:
        train, validation, test = _rotation_examples(examples, fold)
        proxy = _build_validated_proxy(
            eligibility,
            spec=spec,
            outer_rotation=fold,
            provenance=provenance,
            global_participant_ids=global_participant_ids,
        )
        proxy_manifest = ArtifactPayload(
            f"{spec.source}.fold-{fold}.proxy.json", proxy.manifest_bytes
        )
        proxy_fit = proxy.fit_artifact
        result = _fit_proxy_rotation(
            train,
            validation,
            test,
            proxy=proxy,
            configuration=configuration,
            spec=spec,
            seed=seed,
            maximum_epochs=maximum_epochs,
        )
        schedule = _schedule_artifact(
            result,
            configuration=configuration,
            spec=spec,
            seed=seed,
            fold=fold,
            eligibility=eligibility,
            proxy_manifest=proxy_manifest,
            proxy_fit=proxy_fit,
        )
        diagnostics = _diagnostics_artifact(
            result,
            configuration=configuration,
            spec=spec,
            seed=seed,
            fold=fold,
            model_config_sha256=model_config_sha256,
            schedule=schedule,
            proxy_manifest=proxy_manifest,
            proxy_fit=proxy_fit,
        )
        fit_arrays = dict(result.artifact_arrays)
        fit_arrays.update(
            {
                "model_config_sha256_utf8": _utf8_array(model_config_sha256),
                "proxy_manifest_sha256_utf8": _utf8_array(proxy_manifest.sha256),
                "proxy_fit_sha256_utf8": _utf8_array(proxy_fit.sha256),
                "schedule_sha256_utf8": _utf8_array(schedule.sha256),
                "diagnostics_sha256_utf8": _utf8_array(diagnostics.sha256),
            }
        )
        fit = ArtifactPayload(
            f"{configuration}.seed-{seed}.fold-{fold}.fit.npz",
            deterministic_npz_bytes(fit_arrays),
        )
        artifacts.extend((proxy_manifest, proxy_fit, schedule, diagnostics, fit))
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
                checkpoint_optimizer_step=result.checkpoint_optimizer_step,
                checkpoint_sampler_draws_consumed=(
                    result.checkpoint_sampler_draws_consumed
                ),
                epochs_completed=result.epochs_completed,
                stopped_early=result.stopped_early,
                optimizer_steps_per_epoch=result.optimizer_steps_per_epoch,
                optimizer_step_budget_formula="ceil(training_participants/16)",
                total_optimizer_steps=(
                    result.epochs_completed * result.optimizer_steps_per_epoch
                ),
                training_history=list(result.training_history),
                participant_batch_sizes=[PROXY_PARTICIPANTS_PER_BATCH]
                * result.optimizer_steps_per_epoch,
                scaler_fit_session_count=len(train),
                validation_sampling="complete_natural_role_distribution",
                test_sampling="complete_natural_role_distribution",
                proxy_source=spec.source,
                sampling_mode=spec.training_mode,
                lambda_w=spec.lambda_w,
                proxy_artifact={
                    "manifest": {
                        "filename": proxy_manifest.filename,
                        "sha256": proxy_manifest.sha256,
                    },
                    "fit": {"filename": proxy_fit.filename, "sha256": proxy_fit.sha256},
                },
                schedule_artifact={
                    "filename": schedule.filename,
                    "sha256": schedule.sha256,
                    "draw_count": len(result.used_schedule),
                },
                diagnostics_artifact={
                    "filename": diagnostics.filename,
                    "sha256": diagnostics.sha256,
                    "minibatch_count": len(result.minibatch_diagnostics),
                },
                protected_annotation_fields_accessed=[],
            )
        )
        rows.extend(
            _prediction_rows(
                test,
                result.test_prediction,
                configuration=configuration,
                seed=seed,
                model_config_sha256=model_config_sha256,
                model_fit_sha256=fit.sha256,
                eligibility=eligibility,
                targets=targets,
                checkpoint_epoch=result.checkpoint_epoch,
                proxy_source=spec.source,
                sampling_mode=spec.training_mode,
                coefficients={"lambda_w": spec.lambda_w},
            )
        )

    return _lock_bundle(
        rows,
        rotations,
        artifacts,
        configuration=configuration,
        seed=seed,
        model_config=model_config,
        model_config_sha256=model_config_sha256,
        eligibility=eligibility,
        targets=targets,
        provenance=provenance,
        execution_mode=execution_mode,
        executed_maximum_epochs=maximum_epochs,
    )


def _build_validated_proxy(
    eligibility: EligibilityResult,
    *,
    spec: ProxyConfigurationSpec,
    outer_rotation: int,
    provenance: RunProvenance,
    global_participant_ids: Sequence[str] | np.ndarray | None,
) -> ProxyArtifact:
    if spec.source == "random":
        artifact = build_random_proxy_artifact(
            eligibility,
            outer_rotation=outer_rotation,
            provenance=provenance,
            global_participant_ids=global_participant_ids,
        )
    else:
        artifact = build_fold_proxy_artifact(
            eligibility,
            source=spec.source,
            outer_rotation=outer_rotation,
            provenance=provenance,
        )
    return validate_proxy_artifact(
        artifact,
        eligibility,
        provenance=provenance,
        global_participant_ids=global_participant_ids,
    )


def _model_config(
    configuration: str, spec: ProxyConfigurationSpec
) -> dict[str, Any]:
    return {
        "configuration": configuration,
        "architecture": "low_capacity_multimodal_scorer_26721_parameters",
        "trainable_parameter_count": 26_721,
        "modality_scalers": "separate_training_session_row_standard_scalers",
        "linear_initialisation": "xavier_uniform_weight_zero_bias",
        "layer_norm_initialisation": "one_weight_zero_bias",
        "proxy_source": spec.source,
        "proxy_fit_scope": (
            "training_participants_only_per_outer_rotation"
            if spec.source != "random"
            else "global_all_69_participants_once_then_subset_without_reranking"
        ),
        "sampling_mode": spec.training_mode,
        "participant_batch_size": PROXY_PARTICIPANTS_PER_BATCH,
        "participants_per_proxy_group": PROXY_PARTICIPANTS_PER_GROUP,
        "session_rows_per_batch": PROXY_SESSION_ROWS_PER_BATCH,
        "both_sessions_enter_batch_together": True,
        "optimizer_steps_per_epoch": "ceil(training_participant_count/16)",
        "sampler_state": "single_seeded_cyclic_state_continuous_across_epochs",
        "checkpoint_resume": (
            "restore_model_optimizer_and_scalers_then_regenerate_seeded_schedule_"
            "and_advance_to_checkpoint_draw_count"
        ),
        "validation_sampling": "complete_natural_role_distribution",
        "test_sampling": "complete_natural_role_distribution",
        "task_loss": "unweighted_mse_on_normalised_recommend_hiring",
        "wasserstein": {
            "coefficient": spec.lambda_w,
            "application": (
                "disabled_and_group_not_passed_to_objective"
                if spec.lambda_w == 0.0
                else "exact_balanced_empirical_w1_from_update_one_applied_once"
            ),
        },
        "adversarial_loss": "forbidden",
        "protected_annotation_fields_accessed": [],
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
        "checkpoint_metric": "unweighted_validation_srcc",
        "checkpoint_direction": "maximize",
        "exact_tie_break": "earliest_epoch",
        "gradient_norm_clip": 1.0,
        "prediction_postprocessing": "none_sigmoid_is_model_output",
        "device": "cpu",
    }


def _fit_proxy_rotation(
    train: Sequence[Any],
    validation: Sequence[Any],
    test: Sequence[Any],
    *,
    proxy: ProxyArtifact,
    configuration: str,
    spec: ProxyConfigurationSpec,
    seed: int,
    maximum_epochs: int,
) -> _ProxyFit:
    """Fit one proxy-balanced rotation using a continuous cyclic schedule."""
    if proxy.source != spec.source:
        raise ValueError("proxy source does not match the requested configuration")
    _set_training_determinism(seed)
    scalers = {
        modality: StandardScaler().fit(_modality_features(train, modality, dtype=np.float64))
        for modality in SUMMARY_DIMENSIONS
    }
    train_values = _scaled_modalities(train, scalers)
    validation_values = _scaled_modalities(validation, scalers)
    test_values = _scaled_modalities(test, scalers)
    train_targets = torch.from_numpy(
        np.asarray([row.target for row in train], dtype=np.float32)
    ).unsqueeze(1)
    validation_targets = np.asarray([row.target for row in validation], dtype=np.float64)
    participant_rows = _participant_row_indices(train)
    session_rows = {
        row.feature.identity.session_id: index for index, row in enumerate(train)
    }
    training_participant_count = len(participant_rows)
    steps_per_epoch = matched_optimizer_steps_per_epoch(training_participant_count)
    complete_schedule = proxy.paired_training_schedule(
        seed=seed, batch_count=steps_per_epoch * maximum_epochs
    )
    training_groups = proxy.participant_groups("train")
    if set(training_groups) != set(participant_rows):
        raise ValueError("proxy training assignments do not exactly match training participants")

    model = LowCapacityMultimodalScorer().to("cpu")
    optimizer = _adamw(model.parameters())
    best_score = -math.inf
    best_epoch = 0
    best_optimizer_step = 0
    best_sampler_draws = 0
    best_model_state: dict[str, torch.Tensor] | None = None
    best_optimizer_state: dict[str, Any] | None = None
    without_improvement = 0
    history: list[dict[str, Any]] = []
    minibatch_diagnostics: list[dict[str, Any]] = []
    optimizer_step = 0

    for epoch in range(1, maximum_epochs + 1):
        model.train()
        epoch_task_sum = 0.0
        epoch_w1_sum = 0.0
        epoch_total_sum = 0.0
        epoch_gradient_sum = 0.0
        for step_in_epoch in range(steps_per_epoch):
            schedule_index = (epoch - 1) * steps_per_epoch + step_in_epoch
            draw = complete_schedule[schedule_index]
            indices, row_groups = _validated_draw_indices(
                draw,
                participant_rows=participant_rows,
                session_rows=session_rows,
                participant_groups=training_groups,
                expected_batch_index=schedule_index,
            )
            prediction = model(
                train_values["video"][indices],
                train_values["audio"][indices],
                train_values["text"][indices],
            )
            objective_groups = (
                None
                if spec.training_mode == "sampling_only"
                else torch.from_numpy(row_groups)
            )
            objective, components = total_loss(
                prediction,
                train_targets[indices],
                None,
                objective_groups,
                spec.lambda_w,
                0.0,
                task_kind="mse",
            )
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimizer.step()
            optimizer_step += 1
            task_value = float(components["task"].detach())
            w1_value = float(components["w1"].detach())
            total_value = float(objective.detach())
            epoch_task_sum += task_value
            epoch_w1_sum += w1_value
            epoch_total_sum += total_value
            epoch_gradient_sum += gradient_norm
            minibatch_diagnostics.append(
                {
                    "epoch": epoch,
                    "step_in_epoch": step_in_epoch + 1,
                    "optimizer_step": optimizer_step,
                    "schedule_index": schedule_index,
                    "participant_count": PROXY_PARTICIPANTS_PER_BATCH,
                    "session_row_count": PROXY_SESSION_ROWS_PER_BATCH,
                    "participant_group_counts": [8, 8],
                    "session_row_group_counts": [16, 16],
                    "unique_participant_count": draw["unique_participant_count"],
                    "repeated_draw_count": draw["repeated_draw_count"],
                    "unique_participant_rate": draw["unique_participant_rate"],
                    "duplication_rate": draw["duplication_rate"],
                    "group_evidence": draw["groups"],
                    "task_mse": task_value,
                    "wasserstein1": w1_value,
                    "lambda_w": spec.lambda_w,
                    "total_objective": total_value,
                    "objective_received_proxy_groups": objective_groups is not None,
                    "preclip_gradient_norm": gradient_norm,
                }
            )

        validation_prediction = _torch_predict(model, validation_values)
        validation_srcc = _required_correlation(
            validation_targets, validation_prediction, rank=True
        )
        validation_mse = float(
            np.mean(np.square(validation_prediction - validation_targets), dtype=np.float64)
        )
        history.append(
            {
                "epoch": epoch,
                "optimizer_steps": steps_per_epoch,
                "cumulative_optimizer_steps": optimizer_step,
                "sampler_draws_consumed": optimizer_step,
                "training_task_mse_mean": epoch_task_sum / steps_per_epoch,
                "training_wasserstein1_mean": epoch_w1_sum / steps_per_epoch,
                "training_total_objective_mean": epoch_total_sum / steps_per_epoch,
                "preclip_gradient_norm_mean": epoch_gradient_sum / steps_per_epoch,
                "validation_srcc": validation_srcc,
                "validation_mse": validation_mse,
                "validation_sampling": "complete_natural_role_distribution",
            }
        )
        if validation_srcc > best_score:
            best_score = validation_srcc
            best_epoch = epoch
            best_optimizer_step = optimizer_step
            best_sampler_draws = optimizer_step
            best_model_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            without_improvement = 0
        else:
            without_improvement += 1
        if without_improvement >= 25:
            break

    if best_model_state is None or best_optimizer_state is None:
        raise RuntimeError("proxy training produced no selectable validation checkpoint")
    model.load_state_dict(best_model_state)
    validation_prediction = _torch_predict(model, validation_values)
    test_prediction = _torch_predict(model, test_values)
    used_draws = len(history) * steps_per_epoch
    arrays: dict[str, np.ndarray] = {
        "schema_version_utf8": _utf8_array(PROXY_TRAINING_ARTIFACT_SCHEMA_VERSION),
        "shared_fit_schema_version_utf8": _utf8_array(FIT_ARTIFACT_SCHEMA_VERSION),
        "configuration_utf8": _utf8_array(configuration),
        "proxy_source_utf8": _utf8_array(spec.source),
        "sampling_mode_utf8": _utf8_array(spec.training_mode),
        "lambda_w": np.asarray(spec.lambda_w, dtype="<f8"),
        "checkpoint_epoch": np.asarray(best_epoch, dtype="<i8"),
        "checkpoint_validation_srcc": np.asarray(best_score, dtype="<f8"),
        "checkpoint_optimizer_step": np.asarray(best_optimizer_step, dtype="<i8"),
        "checkpoint_sampler_draws_consumed": np.asarray(best_sampler_draws, dtype="<i8"),
        "epochs_completed": np.asarray(len(history), dtype="<i8"),
        "optimizer_steps_per_epoch": np.asarray(steps_per_epoch, dtype="<i8"),
        "optimizer_steps_completed": np.asarray(used_draws, dtype="<i8"),
        "training_participant_count": np.asarray(training_participant_count, dtype="<i8"),
        "participant_batch_size": np.asarray(PROXY_PARTICIPANTS_PER_BATCH, dtype="<i8"),
        "session_rows_per_batch": np.asarray(PROXY_SESSION_ROWS_PER_BATCH, dtype="<i8"),
        "protected_annotation_field_count": np.asarray(0, dtype="<i8"),
        "adversarial_loss_component_count": np.asarray(0, dtype="<i8"),
    }
    for modality, scaler in scalers.items():
        arrays[f"scaler.{modality}.mean"] = np.asarray(scaler.mean_, dtype="<f8")
        arrays[f"scaler.{modality}.scale"] = np.asarray(scaler.scale_, dtype="<f8")
        arrays[f"scaler.{modality}.var"] = np.asarray(scaler.var_, dtype="<f8")
        arrays[f"scaler.{modality}.n_samples_seen"] = np.asarray(
            scaler.n_samples_seen_, dtype="<i8"
        )
    for name, value in sorted(best_model_state.items()):
        arrays[f"model.{name}"] = value.detach().cpu().numpy()
    arrays.update(_optimizer_arrays(best_optimizer_state))
    return _ProxyFit(
        validation_prediction=validation_prediction,
        test_prediction=test_prediction,
        checkpoint_epoch=best_epoch,
        checkpoint_validation_srcc=best_score,
        checkpoint_optimizer_step=best_optimizer_step,
        checkpoint_sampler_draws_consumed=best_sampler_draws,
        epochs_completed=len(history),
        stopped_early=len(history) < maximum_epochs,
        optimizer_steps_per_epoch=steps_per_epoch,
        training_history=tuple(history),
        minibatch_diagnostics=tuple(minibatch_diagnostics),
        used_schedule=tuple(complete_schedule[:used_draws]),
        artifact_arrays=MappingProxyType(arrays),
    )


def _validated_draw_indices(
    draw: Mapping[str, Any],
    *,
    participant_rows: Mapping[str, tuple[int, int]],
    session_rows: Mapping[str, int],
    participant_groups: Mapping[str, int],
    expected_batch_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    participant_ids = draw.get("participant_ids")
    proxy_groups = draw.get("proxy_groups")
    session_ids = draw.get("session_ids")
    if (
        draw.get("batch_index") != expected_batch_index
        or not isinstance(participant_ids, list)
        or len(participant_ids) != PROXY_PARTICIPANTS_PER_BATCH
        or not isinstance(proxy_groups, list)
        or len(proxy_groups) != PROXY_PARTICIPANTS_PER_BATCH
        or proxy_groups.count(0) != PROXY_PARTICIPANTS_PER_GROUP
        or proxy_groups.count(1) != PROXY_PARTICIPANTS_PER_GROUP
        or draw.get("group_counts") != [8, 8]
        or not isinstance(session_ids, list)
        or len(session_ids) != PROXY_SESSION_ROWS_PER_BATCH
        or draw.get("session_count") != PROXY_SESSION_ROWS_PER_BATCH
    ):
        raise ValueError("proxy schedule batch is not the exact paired 8+8 contract")
    if any(participant_id not in participant_rows for participant_id in participant_ids):
        raise ValueError("proxy schedule contains a held-out participant")
    if any(
        participant_groups[participant_id] != group
        for participant_id, group in zip(participant_ids, proxy_groups, strict=True)
    ):
        raise ValueError("proxy schedule group labels disagree with the validated assignment")
    expected_sessions = [
        session_index
        for participant_id in participant_ids
        for session_index in participant_rows[participant_id]
    ]
    observed_sessions: list[int] = []
    for session_id in session_ids:
        if session_id not in session_rows:
            raise ValueError("proxy schedule contains a held-out session")
        observed_sessions.append(session_rows[session_id])
    if observed_sessions != expected_sessions:
        raise ValueError("proxy schedule session order does not match both participant sessions")
    row_groups = np.repeat(np.asarray(proxy_groups, dtype=np.int64), 2)
    if np.bincount(row_groups, minlength=2).tolist() != [16, 16]:
        raise RuntimeError("proxy schedule did not produce 16 session rows per group")
    return np.asarray(observed_sessions, dtype=np.int64), row_groups


def _schedule_artifact(
    result: _ProxyFit,
    *,
    configuration: str,
    spec: ProxyConfigurationSpec,
    seed: int,
    fold: int,
    eligibility: EligibilityResult,
    proxy_manifest: ArtifactPayload,
    proxy_fit: ArtifactPayload,
) -> ArtifactPayload:
    payload = {
        "schema_version": PROXY_SCHEDULE_SCHEMA_VERSION,
        "configuration": configuration,
        "seed": seed,
        "outer_rotation": fold,
        "proxy_source": spec.source,
        "eligibility_manifest_sha256": eligibility.manifest_sha256,
        "proxy_manifest": {
            "filename": proxy_manifest.filename,
            "sha256": proxy_manifest.sha256,
        },
        "proxy_fit": {"filename": proxy_fit.filename, "sha256": proxy_fit.sha256},
        "sampler": "deterministic_seeded_cyclic_8_plus_8_continuous_across_epochs",
        "optimizer_step_budget_formula": "ceil(training_participants/16)",
        "optimizer_steps_per_epoch": result.optimizer_steps_per_epoch,
        "epochs_completed": result.epochs_completed,
        "draw_count": len(result.used_schedule),
        "checkpoint_draws_consumed": result.checkpoint_sampler_draws_consumed,
        "checkpoint_resume": (
            "regenerate_from_validated_proxy_and_seed_then_advance_to_checkpoint_draws_consumed"
        ),
        "draws": list(result.used_schedule),
        "protected_annotation_fields_accessed": [],
    }
    return ArtifactPayload(
        f"{configuration}.seed-{seed}.fold-{fold}.schedule.json",
        canonical_json_bytes(payload),
    )


def _diagnostics_artifact(
    result: _ProxyFit,
    *,
    configuration: str,
    spec: ProxyConfigurationSpec,
    seed: int,
    fold: int,
    model_config_sha256: str,
    schedule: ArtifactPayload,
    proxy_manifest: ArtifactPayload,
    proxy_fit: ArtifactPayload,
) -> ArtifactPayload:
    payload = {
        "schema_version": PROXY_DIAGNOSTICS_SCHEMA_VERSION,
        "configuration": configuration,
        "seed": seed,
        "outer_rotation": fold,
        "proxy_source": spec.source,
        "sampling_mode": spec.training_mode,
        "lambda_w": spec.lambda_w,
        "model_config_sha256": model_config_sha256,
        "proxy_manifest_sha256": proxy_manifest.sha256,
        "proxy_fit_sha256": proxy_fit.sha256,
        "schedule_sha256": schedule.sha256,
        "checkpoint": {
            "epoch": result.checkpoint_epoch,
            "validation_srcc": result.checkpoint_validation_srcc,
            "optimizer_step": result.checkpoint_optimizer_step,
            "sampler_draws_consumed": result.checkpoint_sampler_draws_consumed,
        },
        "epochs_completed": result.epochs_completed,
        "stopped_early": result.stopped_early,
        "training_history": list(result.training_history),
        "minibatches": list(result.minibatch_diagnostics),
        "validation_sampling": "complete_natural_role_distribution",
        "test_sampling": "complete_natural_role_distribution",
        "protected_annotation_fields_accessed": [],
    }
    return ArtifactPayload(
        f"{configuration}.seed-{seed}.fold-{fold}.diagnostics.json",
        canonical_json_bytes(payload),
    )


def _validate_proxy_prediction_bundle(bundle: PredictionBundle) -> None:
    """Validate proxy-specific semantics and every checksum cross-link in a bundle."""
    spec = PROXY_CONFIGURATION_SPECS.get(bundle.configuration)
    if spec is None:
        raise ValueError("proxy bundle configuration is invalid")
    model_config = bundle.manifest.get("model_config")
    if not isinstance(model_config, Mapping) or dict(model_config) != _model_config(
        bundle.configuration, spec
    ):
        raise ValueError("proxy model configuration drifted")
    artifacts = {artifact.filename: artifact for artifact in bundle.artifacts}
    eligibility_payload = _canonical_payload(
        artifacts["complete-pair-eligibility.json"].content,
        "proxy bundled eligibility",
    )
    target_payload = _canonical_payload(
        artifacts["target-manifest.json"].content,
        "proxy bundled target",
    )
    eligible_sessions = {
        session["session_id"]: {
            "participant_id": participant["participant_id"],
            "fixed_fold": participant["fold"],
            "session_phase": session["condition"],
            "source_session_id": session["source_session_id"],
        }
        for participant in eligibility_payload.get("eligible_participants", [])
        for session in participant.get("sessions", [])
    }
    target_by_session = {
        row["session_id"]: row["target"] for row in target_payload.get("rows", [])
    }
    if set(eligible_sessions) != set(target_by_session):
        raise ValueError("proxy support artifacts disagree on eligible target sessions")
    rotations = bundle.manifest.get("rotations")
    if not isinstance(rotations, list) or len(rotations) != 5:
        raise ValueError("proxy bundle must contain exactly five rotations")
    executed_maximum_epochs = bundle.manifest.get("executed_maximum_epochs")
    if (
        isinstance(executed_maximum_epochs, bool)
        or not isinstance(executed_maximum_epochs, int)
        or not 1 <= executed_maximum_epochs <= 200
        or (bundle.execution_mode == "research" and executed_maximum_epochs != 200)
    ):
        raise ValueError("proxy executed epoch cap is invalid")
    expected_artifact_names = {
        "complete-pair-eligibility.json",
        "participant-folds.csv",
        "target-manifest.json",
    }
    expected_fit_names: set[str] = set()
    expected_fit_sha256: set[str] = set()
    expected_fit_sha256_by_rotation: dict[int, str] = {}
    for fold, rotation in enumerate(rotations):
        if not isinstance(rotation, Mapping) or rotation.get("outer_rotation") != fold:
            raise ValueError("proxy rotation order is invalid")
        training_count = len(rotation.get("train_participant_ids", []))
        expected_steps = matched_optimizer_steps_per_epoch(training_count)
        expected_common = {
            "proxy_source": spec.source,
            "sampling_mode": spec.training_mode,
            "lambda_w": spec.lambda_w,
            "optimizer_steps_per_epoch": expected_steps,
            "optimizer_step_budget_formula": "ceil(training_participants/16)",
            "validation_sampling": "complete_natural_role_distribution",
            "test_sampling": "complete_natural_role_distribution",
            "protected_annotation_fields_accessed": [],
        }
        if any(rotation.get(key) != value for key, value in expected_common.items()):
            raise ValueError("proxy rotation contract drifted")
        proxy_record = rotation.get("proxy_artifact")
        schedule_record = rotation.get("schedule_artifact")
        diagnostics_record = rotation.get("diagnostics_artifact")
        fit_record = rotation.get("fit_artifact")
        references = (
            proxy_record.get("manifest") if isinstance(proxy_record, Mapping) else None,
            proxy_record.get("fit") if isinstance(proxy_record, Mapping) else None,
            schedule_record,
            diagnostics_record,
            fit_record,
        )
        expected_reference_names = (
            f"{spec.source}.fold-{fold}.proxy.json",
            f"{spec.source}.fold-{fold}.proxy-fit.npz",
            f"{bundle.configuration}.seed-{bundle.seed}.fold-{fold}.schedule.json",
            f"{bundle.configuration}.seed-{bundle.seed}.fold-{fold}.diagnostics.json",
            f"{bundle.configuration}.seed-{bundle.seed}.fold-{fold}.fit.npz",
        )
        for reference in references:
            if not isinstance(reference, Mapping):
                raise ValueError("proxy rotation artifact reference is missing")
            filename = reference.get("filename")
            if filename not in artifacts or artifacts[filename].sha256 != reference.get("sha256"):
                raise ValueError("proxy rotation artifact checksum cross-link failed")
        if tuple(reference["filename"] for reference in references) != expected_reference_names:
            raise ValueError("proxy rotation artifact filenames are not canonical")
        expected_artifact_names.update(expected_reference_names)
        proxy_manifest_artifact = artifacts[references[0]["filename"]]
        proxy_fit_artifact = artifacts[references[1]["filename"]]
        schedule_artifact = artifacts[references[2]["filename"]]
        diagnostics_artifact = artifacts[references[3]["filename"]]
        fit_artifact = artifacts[references[4]["filename"]]
        expected_fit_names.add(fit_artifact.filename)
        expected_fit_sha256.add(fit_artifact.sha256)
        expected_fit_sha256_by_rotation[fold] = fit_artifact.sha256
        proxy_payload = _canonical_payload(proxy_manifest_artifact.content, "proxy manifest")
        schedule_payload = _canonical_payload(schedule_artifact.content, "proxy schedule")
        diagnostics_payload = _canonical_payload(
            diagnostics_artifact.content, "proxy diagnostics"
        )
        _validate_proxy_artifact_structure(
            _UnvalidatedProxyArtifact(
                source=spec.source,
                outer_rotation=fold,
                manifest=proxy_payload,
                manifest_bytes=proxy_manifest_artifact.content,
                manifest_sha256=proxy_payload.get("manifest_payload_sha256"),
                fit_artifact=proxy_fit_artifact,
            )
        )
        if (
            proxy_payload.get("schema_version") != PROXY_ARTIFACT_SCHEMA_VERSION
            or proxy_payload.get("proxy_source") != spec.source
            or proxy_payload.get("outer_rotation") != fold
            or proxy_payload.get("eligibility_manifest_sha256")
            != bundle.manifest.get("eligibility_manifest_sha256")
            or proxy_payload.get("fold_manifest_sha256")
            != bundle.manifest.get("fold_manifest_sha256")
            or proxy_payload.get("feature_contract_sha256")
            != bundle.manifest.get("feature_contract_sha256")
            or proxy_payload.get("preprocessing_sha256")
            != bundle.manifest.get("preprocessing_sha256")
            or proxy_payload.get("session_mapping") != bundle.manifest.get("session_mapping")
            or proxy_payload.get("provenance", {}).get("config_sha256")
            != bundle.manifest.get("config_sha256")
            or proxy_payload.get("provenance", {}).get("code_commit_sha")
            != bundle.manifest.get("code_commit_sha")
            or proxy_payload.get("provenance", {}).get("code_source_tree_sha256")
            != bundle.manifest.get("code_source_tree_sha256")
            or proxy_payload.get("provenance", {}).get("source_date_epoch")
            != bundle.manifest.get("source_date_epoch")
            or proxy_payload.get("protected_annotation_fields_accessed") != []
            or proxy_payload.get("outcome_fields_accessed") != []
            or proxy_payload.get("fit_artifact", {}).get("filename")
            != proxy_fit_artifact.filename
            or proxy_payload.get("fit_artifact", {}).get("sha256") != proxy_fit_artifact.sha256
        ):
            raise ValueError("proxy manifest disagrees with its training rotation")
        proxy_roles = proxy_payload.get("roles")
        if not isinstance(proxy_roles, Mapping) or any(
            proxy_roles.get(role) != rotation.get(f"{role}_participant_ids")
            for role in ("train", "validation", "test")
        ):
            raise ValueError("proxy manifest roles disagree with the training rotation")
        proxy_participants = proxy_payload.get("participants")
        if not isinstance(proxy_participants, list):
            raise ValueError("proxy manifest participant assignments are unavailable")
        training_proxy_rows = {
            row["participant_id"]: row
            for row in proxy_participants
            if isinstance(row, Mapping) and row.get("rotation_role") == "train"
        }
        if set(training_proxy_rows) != set(rotation["train_participant_ids"]):
            raise ValueError("proxy training assignments do not match the training role")
        epochs_completed = schedule_payload.get("epochs_completed")
        draw_count = schedule_payload.get("draw_count")
        checkpoint_draws = schedule_payload.get("checkpoint_draws_consumed")
        if (
            set(schedule_payload) != _SCHEDULE_FIELDS
            or schedule_payload.get("schema_version") != PROXY_SCHEDULE_SCHEMA_VERSION
            or schedule_payload.get("configuration") != bundle.configuration
            or schedule_payload.get("seed") != bundle.seed
            or schedule_payload.get("outer_rotation") != fold
            or schedule_payload.get("proxy_source") != spec.source
            or schedule_payload.get("eligibility_manifest_sha256")
            != bundle.manifest.get("eligibility_manifest_sha256")
            or schedule_payload.get("sampler")
            != "deterministic_seeded_cyclic_8_plus_8_continuous_across_epochs"
            or schedule_payload.get("optimizer_step_budget_formula")
            != "ceil(training_participants/16)"
            or schedule_payload.get("optimizer_steps_per_epoch") != expected_steps
            or isinstance(epochs_completed, bool)
            or not isinstance(epochs_completed, int)
            or epochs_completed < 1
            or draw_count != epochs_completed * expected_steps
            or isinstance(checkpoint_draws, bool)
            or not isinstance(checkpoint_draws, int)
            or checkpoint_draws < 1
            or checkpoint_draws > draw_count
            or checkpoint_draws % expected_steps != 0
            or schedule_payload.get("checkpoint_resume")
            != (
                "regenerate_from_validated_proxy_and_seed_then_advance_to_"
                "checkpoint_draws_consumed"
            )
            or schedule_payload.get("protected_annotation_fields_accessed") != []
            or schedule_payload.get("proxy_manifest")
            != {
                "filename": proxy_manifest_artifact.filename,
                "sha256": proxy_manifest_artifact.sha256,
            }
            or schedule_payload.get("proxy_fit")
            != {"filename": proxy_fit_artifact.filename, "sha256": proxy_fit_artifact.sha256}
        ):
            raise ValueError("proxy schedule contract or cross-link failed")
        draws = schedule_payload.get("draws")
        if not isinstance(draws, list) or len(draws) != draw_count:
            raise ValueError("proxy schedule draw count is invalid")
        schedule_participant_ids = np.asarray(list(training_proxy_rows), dtype=object)
        schedule_groups = np.asarray(
            [
                training_proxy_rows[participant_id]["proxy_group"]
                for participant_id in training_proxy_rows
            ],
            dtype=np.int8,
        )
        session_pairs = {
            participant_id: tuple(
                session["session_id"]
                for session in training_proxy_rows[participant_id]["session_inputs"]
            )
            for participant_id in training_proxy_rows
        }
        expected_draws = list(
            BalancedParticipantSampler(
                schedule_participant_ids,
                schedule_groups,
                seed=bundle.seed,
            ).paired_schedule(session_pairs, batch_count=draw_count)
        )
        if draws != expected_draws:
            raise ValueError(
                "proxy schedule does not match the exact seeded continuous cyclic reconstruction"
            )
        minibatches = diagnostics_payload.get("minibatches")
        history = diagnostics_payload.get("training_history")
        checkpoint = diagnostics_payload.get("checkpoint")
        expected_checkpoint = {
            "epoch": rotation.get("checkpoint_epoch"),
            "validation_srcc": rotation.get("checkpoint_validation_srcc"),
            "optimizer_step": rotation.get("checkpoint_optimizer_step"),
            "sampler_draws_consumed": rotation.get("checkpoint_sampler_draws_consumed"),
        }
        if (
            not isinstance(checkpoint, Mapping)
            or set(checkpoint)
            != {"epoch", "validation_srcc", "optimizer_step", "sampler_draws_consumed"}
            or isinstance(expected_checkpoint["epoch"], bool)
            or not isinstance(expected_checkpoint["epoch"], int)
            or expected_checkpoint["epoch"] < 1
            or not math.isfinite(
                _finite_json_number(
                    expected_checkpoint["validation_srcc"], "checkpoint_validation_srcc"
                )
            )
            or isinstance(expected_checkpoint["optimizer_step"], bool)
            or not isinstance(expected_checkpoint["optimizer_step"], int)
            or isinstance(expected_checkpoint["sampler_draws_consumed"], bool)
            or not isinstance(expected_checkpoint["sampler_draws_consumed"], int)
        ):
            raise ValueError("proxy checkpoint diagnostics are invalid")
        if (
            set(diagnostics_payload) != _DIAGNOSTICS_FIELDS
            or diagnostics_payload.get("schema_version") != PROXY_DIAGNOSTICS_SCHEMA_VERSION
            or diagnostics_payload.get("configuration") != bundle.configuration
            or diagnostics_payload.get("seed") != bundle.seed
            or diagnostics_payload.get("outer_rotation") != fold
            or diagnostics_payload.get("proxy_source") != spec.source
            or diagnostics_payload.get("sampling_mode") != spec.training_mode
            or diagnostics_payload.get("lambda_w") != spec.lambda_w
            or diagnostics_payload.get("model_config_sha256")
            != bundle.manifest.get("model_config_sha256")
            or diagnostics_payload.get("schedule_sha256") != schedule_artifact.sha256
            or diagnostics_payload.get("proxy_manifest_sha256")
            != proxy_manifest_artifact.sha256
            or diagnostics_payload.get("proxy_fit_sha256") != proxy_fit_artifact.sha256
            or checkpoint != expected_checkpoint
            or expected_checkpoint["optimizer_step"] != checkpoint_draws
            or expected_checkpoint["sampler_draws_consumed"] != checkpoint_draws
            or expected_checkpoint["epoch"] * expected_steps != checkpoint_draws
            or diagnostics_payload.get("epochs_completed") != epochs_completed
            or diagnostics_payload.get("stopped_early") != rotation.get("stopped_early")
            or diagnostics_payload.get("validation_sampling")
            != "complete_natural_role_distribution"
            or diagnostics_payload.get("test_sampling")
            != "complete_natural_role_distribution"
            or diagnostics_payload.get("protected_annotation_fields_accessed") != []
            or not isinstance(minibatches, list)
            or len(minibatches) != len(draws)
            or not isinstance(history, list)
            or len(history) != epochs_completed
            or history != rotation.get("training_history")
            or rotation.get("epochs_completed") != epochs_completed
            or rotation.get("total_optimizer_steps") != draw_count
            or rotation.get("participant_batch_sizes") != [16] * expected_steps
            or schedule_record.get("draw_count") != draw_count
            or diagnostics_record.get("minibatch_count") != len(minibatches)
        ):
            raise ValueError("proxy diagnostics contract or cross-link failed")
        for schedule_index, (minibatch, draw) in enumerate(
            zip(minibatches, draws, strict=True)
        ):
            if (
                not isinstance(minibatch, Mapping)
                or set(minibatch) != _MINIBATCH_DIAGNOSTIC_FIELDS
            ):
                raise ValueError("proxy minibatch diagnostic fields are invalid")
            numeric_values = {
                name: _finite_json_number(minibatch.get(name), name)
                for name in (
                    "task_mse",
                    "wasserstein1",
                    "total_objective",
                    "preclip_gradient_norm",
                )
            }
            expected_total = numeric_values["task_mse"] + (
                spec.lambda_w * numeric_values["wasserstein1"]
            )
            if (
                minibatch.get("epoch") != schedule_index // expected_steps + 1
                or minibatch.get("step_in_epoch") != schedule_index % expected_steps + 1
                or minibatch.get("optimizer_step") != schedule_index + 1
                or minibatch.get("schedule_index") != schedule_index
                or minibatch.get("participant_count") != draw["participant_count"]
                or minibatch.get("session_row_count") != draw["session_count"]
                or minibatch.get("participant_group_counts") != draw["group_counts"]
                or minibatch.get("session_row_group_counts") != [16, 16]
                or minibatch.get("unique_participant_count")
                != draw["unique_participant_count"]
                or minibatch.get("repeated_draw_count") != draw["repeated_draw_count"]
                or minibatch.get("unique_participant_rate")
                != draw["unique_participant_rate"]
                or minibatch.get("duplication_rate") != draw["duplication_rate"]
                or minibatch.get("group_evidence") != draw["groups"]
                or minibatch.get("objective_received_proxy_groups")
                != (spec.lambda_w > 0.0)
                or minibatch.get("lambda_w") != spec.lambda_w
                or any(value < 0.0 for value in numeric_values.values())
                or (spec.lambda_w == 0.0 and numeric_values["wasserstein1"] != 0.0)
                or not math.isclose(
                    numeric_values["total_objective"],
                    expected_total,
                    rel_tol=1e-6,
                    abs_tol=1e-8,
                )
            ):
                raise ValueError("proxy minibatch objective or balance contract failed")
        for epoch_index, epoch_row in enumerate(history):
            epoch_minibatches = minibatches[
                epoch_index * expected_steps : (epoch_index + 1) * expected_steps
            ]
            expected_epoch = epoch_index + 1
            if (
                not isinstance(epoch_row, Mapping)
                or set(epoch_row) != _TRAINING_HISTORY_FIELDS
                or epoch_row.get("epoch") != expected_epoch
                or epoch_row.get("optimizer_steps") != expected_steps
                or epoch_row.get("cumulative_optimizer_steps")
                != expected_epoch * expected_steps
                or epoch_row.get("sampler_draws_consumed")
                != expected_epoch * expected_steps
                or epoch_row.get("validation_sampling")
                != "complete_natural_role_distribution"
                or _finite_json_number(epoch_row.get("validation_mse"), "validation_mse")
                < 0.0
                or not -1.0
                <= _finite_json_number(epoch_row.get("validation_srcc"), "validation_srcc")
                <= 1.0
                or not _mean_matches(
                    epoch_row.get("training_task_mse_mean"),
                    epoch_minibatches,
                    "task_mse",
                )
                or not _mean_matches(
                    epoch_row.get("training_wasserstein1_mean"),
                    epoch_minibatches,
                    "wasserstein1",
                )
                or not _mean_matches(
                    epoch_row.get("training_total_objective_mean"),
                    epoch_minibatches,
                    "total_objective",
                )
                or not _mean_matches(
                    epoch_row.get("preclip_gradient_norm_mean"),
                    epoch_minibatches,
                    "preclip_gradient_norm",
                )
            ):
                raise ValueError("proxy epoch diagnostics do not match their minibatches")
        history_scores = [
            _finite_json_number(row["validation_srcc"], "validation_srcc")
            for row in history
        ]
        best_score = max(history_scores)
        best_epoch = history_scores.index(best_score) + 1
        expected_epochs_completed = min(executed_maximum_epochs, best_epoch + 25)
        expected_stopped_early = expected_epochs_completed < executed_maximum_epochs
        if (
            expected_checkpoint["epoch"] != best_epoch
            or expected_checkpoint["validation_srcc"] != best_score
            or epochs_completed != expected_epochs_completed
            or rotation.get("stopped_early") is not expected_stopped_early
        ):
            raise ValueError("proxy checkpoint or patience-25 stopping history is invalid")
        with np.load(io.BytesIO(fit_artifact.content), allow_pickle=False) as fit:
            fit_arrays = {name: np.array(fit[name], copy=True) for name in fit.files}
            if (
                deterministic_npz_bytes(fit_arrays) != fit_artifact.content
                or set(fit_arrays) != _expected_proxy_fit_array_names()
                or _archive_utf8(fit, "schema_version_utf8")
                != PROXY_TRAINING_ARTIFACT_SCHEMA_VERSION
                or _archive_utf8(fit, "shared_fit_schema_version_utf8")
                != FIT_ARTIFACT_SCHEMA_VERSION
                or _archive_utf8(fit, "configuration_utf8") != bundle.configuration
                or _archive_utf8(fit, "proxy_source_utf8") != spec.source
                or _archive_utf8(fit, "sampling_mode_utf8") != spec.training_mode
                or float(fit["lambda_w"]) != spec.lambda_w
                or int(fit["checkpoint_epoch"]) != expected_checkpoint["epoch"]
                or float(fit["checkpoint_validation_srcc"])
                != expected_checkpoint["validation_srcc"]
                or int(fit["checkpoint_optimizer_step"])
                != expected_checkpoint["optimizer_step"]
                or int(fit["checkpoint_sampler_draws_consumed"])
                != expected_checkpoint["sampler_draws_consumed"]
                or int(fit["epochs_completed"]) != epochs_completed
                or int(fit["optimizer_steps_per_epoch"]) != expected_steps
                or int(fit["optimizer_steps_completed"]) != draw_count
                or int(fit["training_participant_count"]) != training_count
                or int(fit["participant_batch_size"]) != PROXY_PARTICIPANTS_PER_BATCH
                or int(fit["session_rows_per_batch"]) != PROXY_SESSION_ROWS_PER_BATCH
                or int(fit["protected_annotation_field_count"]) != 0
                or int(fit["adversarial_loss_component_count"]) != 0
                or int(fit["optimizer.state_count"]) != 14
                or _archive_utf8(fit, "proxy_manifest_sha256_utf8")
                != proxy_manifest_artifact.sha256
                or _archive_utf8(fit, "proxy_fit_sha256_utf8") != proxy_fit_artifact.sha256
                or _archive_utf8(fit, "schedule_sha256_utf8") != schedule_artifact.sha256
                or _archive_utf8(fit, "diagnostics_sha256_utf8")
                != diagnostics_artifact.sha256
                or _archive_utf8(fit, "model_config_sha256_utf8")
                != bundle.manifest.get("model_config_sha256")
            ):
                raise ValueError("proxy fit artifact contract or cross-link failed")
            _validate_proxy_fit_numeric_arrays(
                fit_arrays,
                training_count=training_count,
                checkpoint_optimizer_step=expected_checkpoint["optimizer_step"],
            )
    referenced_fit_sha256 = {
        row.get("model_fit_sha256") for row in bundle.predictions if isinstance(row, Mapping)
    }
    if referenced_fit_sha256 != expected_fit_sha256:
        raise ValueError("proxy predictions do not reference the exact rotation fit states")
    expected_encoder_revisions = {
        modality: contract["revision"]
        for modality, contract in bundle.manifest.get("encoder_contract", {}).items()
    }
    seen_prediction_sessions: set[str] = set()
    for row in bundle.predictions:
        session_id = row.get("session_id")
        identity = eligible_sessions.get(session_id)
        fold = row.get("outer_rotation")
        prediction = row.get("prediction")
        if (
            set(row) != _PROXY_PREDICTION_FIELDS
            or identity is None
            or session_id in seen_prediction_sessions
            or row.get("participant_id") != identity["participant_id"]
            or row.get("session_phase") != identity["session_phase"]
            or row.get("source_session_id") != identity["source_session_id"]
            or row.get("fixed_fold") != identity["fixed_fold"]
            or fold != identity["fixed_fold"]
            or row.get("rotation_role") != "test"
            or identity["participant_id"] not in rotations[fold]["test_participant_ids"]
            or row.get("seed") != bundle.seed
            or row.get("configuration") != bundle.configuration
            or row.get("model_config_sha256")
            != bundle.manifest.get("model_config_sha256")
            or row.get("target_manifest_sha256")
            != bundle.manifest.get("target_manifest_sha256")
            or row.get("target_transform") != target_payload.get("transform")
            or row.get("target") != target_by_session.get(session_id)
            or isinstance(prediction, bool)
            or not isinstance(prediction, (int, float))
            or not math.isfinite(float(prediction))
            or not 0.0 <= float(prediction) <= 1.0
            or row.get("encoder_revisions") != expected_encoder_revisions
            or row.get("feature_contract_sha256")
            != bundle.manifest.get("feature_contract_sha256")
            or row.get("preprocessing_sha256")
            != bundle.manifest.get("preprocessing_sha256")
            or row.get("proxy_source") != spec.source
            or row.get("sampling_mode") != spec.training_mode
            or row.get("coefficients") != {"lambda_w": spec.lambda_w}
            or row.get("checkpoint_epoch") != rotations[fold].get("checkpoint_epoch")
            or row.get("firewall_stage") != "A"
            or row.get("eligibility_manifest_sha256")
            != bundle.manifest.get("eligibility_manifest_sha256")
            or row.get("session_mapping") != bundle.manifest.get("session_mapping")
            or row.get("model_fit_sha256")
            != expected_fit_sha256_by_rotation.get(fold)
        ):
            raise ValueError("proxy prediction row contract drifted")
        seen_prediction_sessions.add(session_id)
    if seen_prediction_sessions != set(eligible_sessions):
        raise ValueError("proxy predictions do not exactly cover eligible sessions")
    if len(expected_fit_names) != 5:
        raise ValueError("proxy bundle fit-state filenames are not rotation-unique")
    if set(artifacts) != expected_artifact_names:
        raise ValueError("proxy bundle contains missing, extra or noncanonical artifacts")


def _finite_json_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"proxy diagnostic {name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"proxy diagnostic {name} must be a finite number")
    return result


def _mean_matches(value: Any, rows: Sequence[Mapping[str, Any]], field: str) -> bool:
    observed = _finite_json_number(value, field)
    expected = math.fsum(_finite_json_number(row.get(field), field) for row in rows) / len(rows)
    return math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12)


def _expected_proxy_fit_array_names() -> set[str]:
    fixed = {
        "schema_version_utf8",
        "shared_fit_schema_version_utf8",
        "configuration_utf8",
        "proxy_source_utf8",
        "sampling_mode_utf8",
        "lambda_w",
        "checkpoint_epoch",
        "checkpoint_validation_srcc",
        "checkpoint_optimizer_step",
        "checkpoint_sampler_draws_consumed",
        "epochs_completed",
        "optimizer_steps_per_epoch",
        "optimizer_steps_completed",
        "training_participant_count",
        "participant_batch_size",
        "session_rows_per_batch",
        "protected_annotation_field_count",
        "adversarial_loss_component_count",
        "model_config_sha256_utf8",
        "proxy_manifest_sha256_utf8",
        "proxy_fit_sha256_utf8",
        "schedule_sha256_utf8",
        "diagnostics_sha256_utf8",
        "optimizer.state_count",
        "optimizer.param_groups_json_utf8",
    }
    scaler = {
        f"scaler.{modality}.{field}"
        for modality in SUMMARY_DIMENSIONS
        for field in ("mean", "scale", "var", "n_samples_seen")
    }
    model = {f"model.{name}" for name in _MODEL_STATE_SHAPES}
    optimizer = {
        f"optimizer.parameter-{index:04d}.{field}"
        for index in range(len(_MODEL_STATE_SHAPES))
        for field in ("step", "exp_avg", "exp_avg_sq")
    }
    return fixed | scaler | model | optimizer


def _validate_proxy_fit_numeric_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    training_count: int,
    checkpoint_optimizer_step: int,
) -> None:
    for modality, dimension in SUMMARY_DIMENSIONS.items():
        for field in ("mean", "scale", "var"):
            value = np.asarray(arrays[f"scaler.{modality}.{field}"])
            if value.dtype != np.dtype("<f8") or value.shape != (dimension,):
                raise ValueError("proxy fit scaler state has an invalid dtype or shape")
            if not np.isfinite(value).all():
                raise ValueError("proxy fit scaler state is non-finite")
            if field == "scale" and np.any(value <= 0):
                raise ValueError("proxy fit scaler scale must be positive")
            if field == "var" and np.any(value < 0):
                raise ValueError("proxy fit scaler variance must be non-negative")
        samples = np.asarray(arrays[f"scaler.{modality}.n_samples_seen"])
        if (
            samples.dtype != np.dtype("<i8")
            or samples.shape != ()
            or int(samples) != 2 * training_count
        ):
            raise ValueError("proxy fit scaler sample count is invalid")

    for parameter_index, (name, shape) in enumerate(_MODEL_STATE_SHAPES.items()):
        model_value = np.asarray(arrays[f"model.{name}"])
        if model_value.dtype != np.float32 or model_value.shape != shape:
            raise ValueError("proxy fit model state has an invalid dtype or shape")
        if not np.isfinite(model_value).all():
            raise ValueError("proxy fit model state is non-finite")
        for field in ("exp_avg", "exp_avg_sq"):
            optimizer_value = np.asarray(
                arrays[f"optimizer.parameter-{parameter_index:04d}.{field}"]
            )
            if optimizer_value.dtype != np.float32 or optimizer_value.shape != shape:
                raise ValueError("proxy fit optimizer moment has an invalid dtype or shape")
            if not np.isfinite(optimizer_value).all():
                raise ValueError("proxy fit optimizer moment is non-finite")
            if field == "exp_avg_sq" and np.any(optimizer_value < 0):
                raise ValueError("proxy fit optimizer second moment must be non-negative")
        step = np.asarray(arrays[f"optimizer.parameter-{parameter_index:04d}.step"])
        if step.shape != () or not np.issubdtype(step.dtype, np.floating):
            raise ValueError("proxy fit optimizer step is invalid")
        if (
            not math.isfinite(float(step))
            or float(step) != checkpoint_optimizer_step
        ):
            raise ValueError("proxy fit optimizer step is invalid")

    parameter_groups = _canonical_payload(
        np.asarray(arrays["optimizer.param_groups_json_utf8"]).tobytes(),
        "proxy optimizer parameter groups",
    ).get("param_groups")
    if not isinstance(parameter_groups, list) or len(parameter_groups) != 1:
        raise ValueError("proxy optimizer parameter groups are invalid")
    group = parameter_groups[0]
    expected_group = {
        "lr": 1e-3,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 1e-2,
        "amsgrad": False,
        "maximize": False,
        "foreach": False,
        "capturable": False,
        "differentiable": False,
        "fused": None,

        "params": list(range(len(_MODEL_STATE_SHAPES))),
    }
    if not isinstance(group, Mapping):
        raise ValueError("proxy optimizer parameter groups drifted")
    actual_group = dict(group)
    if "decoupled_weight_decay" in actual_group:
        if actual_group.pop("decoupled_weight_decay") is not True:
            raise ValueError("proxy optimizer parameter groups drifted")
    if actual_group != expected_group:
        raise ValueError("proxy optimizer parameter groups drifted")


def _canonical_payload(raw: bytes, label: str) -> dict[str, Any]:
    try:
        import json

        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError) as error:
        raise ValueError(f"{label} cannot be read") from error
    if not isinstance(payload, dict) or canonical_json_bytes(payload) != raw:
        raise ValueError(f"{label} is not canonical")
    return payload


def _archive_utf8(archive: Any, name: str) -> str:
    value = np.asarray(archive[name])
    if value.dtype != np.uint8 or value.ndim != 1:
        raise ValueError(f"proxy fit string field {name} is invalid")
    try:
        return value.tobytes().decode("utf-8")
    except UnicodeError as error:
        raise ValueError(f"proxy fit string field {name} is invalid") from error


def _utf8_array(value: str) -> np.ndarray:
    return np.frombuffer(value.encode("utf-8"), dtype=np.uint8)
