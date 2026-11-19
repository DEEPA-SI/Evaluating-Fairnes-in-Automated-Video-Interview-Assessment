"""Segregated conditional Stage-B protected-annotation oracle runner.

The ordinary Stage-A model matrix never imports this module.  Stage B opens only
after an exact immutable Stage-A closure and either publishes all fifteen frozen
oracle fits or one immutable zero-fit omission.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any, cast

import numpy as np

from src.features.pseudo_attributes import BalancedParticipantSampler
from src.secure_io import AnchoredDirectory

from .crossfit import (
    FIT_ARTIFACT_SCHEMA_VERSION,
    SUMMARY_DIMENSIONS,
    ArtifactPayload,
    EligibilityResult,
    RunProvenance,
    TargetSet,
    _bind_targets,
    _payload_sha256,
    _rotation_examples,
    canonical_json_bytes,
    deterministic_npz_bytes,
    validate_eligibility_result,
)
from .proxy_training import (
    FROZEN_FINAL_SEEDS,
    FROZEN_OUTER_ROTATIONS,
    PROXY_PARTICIPANTS_PER_BATCH,
    PROXY_PARTICIPANTS_PER_GROUP,
    PROXY_SESSION_ROWS_PER_BATCH,
    WASSERSTEIN_COEFFICIENT,
    ProxyConfigurationSpec,
    _fit_proxy_rotation,
)

STAGE_B_ORACLE_FIT_SCHEMA_VERSION = "fair-avi-mit-stage-b-oracle-fit-3"
STAGE_B_ORACLE_GROUP_SOURCE = "protected_annotation_training_fold"
STAGE_B_ORACLE_SAMPLING_MODE = "wasserstein1"

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

_FIT_UTF8_FIELDS = {
    "schema_version_utf8",
    "shared_fit_schema_version_utf8",
    "configuration_utf8",
    "execution_mode_utf8",
    "group_source_utf8",
    "sampling_mode_utf8",
    "oracle_field_utf8",
    "category_0_json_utf8",
    "category_1_json_utf8",
    "stage_a_closure_file_sha256_utf8",
    "stage_a_closure_payload_sha256_utf8",
    "oracle_eligibility_payload_sha256_utf8",
    "access_log_payload_sha256_utf8",
    "access_log_entry_payload_sha256_utf8",
    "released_annotations_sha256_utf8",
    "target_manifest_sha256_utf8",
    "eligibility_manifest_sha256_utf8",
    "feature_contract_sha256_utf8",
    "preprocessing_sha256_utf8",
    "session_mapping_json_utf8",
    "config_sha256_utf8",
    "code_commit_sha_utf8",
    "code_source_tree_sha256_utf8",
    "stage_b_execution_provenance_json_utf8",
    "model_config_sha256_utf8",
    "training_history_json_utf8",
    "minibatch_diagnostics_json_utf8",
    "schedule_json_utf8",
}
_FIT_FLOAT_FIELDS = {"lambda_w", "checkpoint_validation_srcc"}
_FIT_INTEGER_FIELDS = {
    "seed",
    "outer_rotation",
    "source_date_epoch",
    "checkpoint_epoch",
    "checkpoint_optimizer_step",
    "checkpoint_sampler_draws_consumed",
    "epochs_completed",
    "executed_maximum_epochs",
    "optimizer_steps_per_epoch",
    "optimizer_steps_completed",
    "training_participant_count",
    "participant_batch_size",
    "session_rows_per_batch",
    "protected_annotation_field_count",
    "protected_validation_annotation_field_count",
    "protected_test_annotation_field_count",
    "adversarial_loss_component_count",
    "optimizer.state_count",
}


@dataclass(frozen=True, slots=True)
class StageBOracleSeedBundle:
    """One complete five-rotation Stage-B publication unit."""

    configuration: str
    seed: int
    access_log_filename: str
    access_log: Mapping[str, Any]
    access_log_bytes: bytes
    prediction_filename: str
    prediction_payload: Mapping[str, Any]
    prediction_bytes: bytes
    prediction_lock_filename: str
    prediction_lock: Mapping[str, Any]
    prediction_lock_bytes: bytes
    fit_artifacts: tuple[ArtifactPayload, ...]
    bundle_record: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.seed not in FROZEN_FINAL_SEEDS:
            raise ValueError("Stage-B oracle seed is not frozen")
        if self.configuration != self.prediction_lock.get("configuration"):
            raise ValueError("Stage-B oracle bundle configuration drifted")
        if len(self.fit_artifacts) != len(FROZEN_OUTER_ROTATIONS):
            raise ValueError("Stage-B oracle seed bundle must contain exactly five fits")
        for filename in (
            self.access_log_filename,
            self.prediction_filename,
            self.prediction_lock_filename,
            *(artifact.filename for artifact in self.fit_artifacts),
        ):
            if not filename or Path(filename).name != filename:
                raise ValueError("Stage-B oracle artifact filenames must be safe basenames")


@dataclass(frozen=True, slots=True)
class StageBOracleEvidence:
    """Reloadable Stage-B controller evidence with no full annotation table."""

    eligibility: EligibilityResult | None
    targets: TargetSet | None
    release: Any
    annotation_binding: Any
    training_releases: tuple[Any, ...]
    provenance: RunProvenance
    _initial_fingerprint_sha256: str = dataclass_field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "training_releases", tuple(self.training_releases))
        self._validate()
        object.__setattr__(
            self,
            "_initial_fingerprint_sha256",
            _payload_sha256(self._fingerprint_payload()),
        )

    @property
    def fingerprint_sha256(self) -> str:
        """Return the complete current controller-evidence fingerprint."""
        return _payload_sha256(self._fingerprint_payload())

    @property
    def oracle_eligibility(self) -> Mapping[str, Any]:
        return self.annotation_binding.eligibility

    def verify_integrity(self) -> None:
        """Revalidate retained evidence and reject any post-load mutation."""
        self._validate()
        if self.fingerprint_sha256 != self._initial_fingerprint_sha256:
            raise RuntimeError("Stage-B controller evidence changed after validation")

    def _validate(self) -> None:
        from src.eval.stage_b_oracle import (
            OracleAnnotationBinding,
            OracleTrainingAnnotationRelease,
        )

        if not isinstance(self.provenance, RunProvenance):
            raise TypeError("Stage-B evidence requires frozen run provenance")
        if not isinstance(self.annotation_binding, OracleAnnotationBinding):
            raise TypeError("Stage-B evidence requires a label-free annotation binding")
        if any(
            not isinstance(row, OracleTrainingAnnotationRelease) for row in self.training_releases
        ):
            raise TypeError("Stage-B evidence requires training-only annotation releases")
        status = self.oracle_eligibility.get("status")
        if status == "eligible_all_15":
            if not isinstance(self.eligibility, EligibilityResult) or not isinstance(
                self.targets, TargetSet
            ):
                raise ValueError("eligible Stage B requires features and targets")
            validate_eligibility_result(self.eligibility)
            _validate_runner_context(self.eligibility, self.targets, self.release, self.provenance)
            if len(self.training_releases) != len(FROZEN_OUTER_ROTATIONS):
                raise ValueError("eligible Stage B requires five training-only releases")
        elif status == "omitted_all_15":
            if self.eligibility is not None or self.targets is not None or self.training_releases:
                raise ValueError("omitted Stage B must retain zero fit inputs")
        else:
            raise ValueError("Stage-B evidence has no frozen disposition")

    def _fingerprint_payload(self) -> dict[str, Any]:
        eligibility = self.eligibility
        targets = self.targets
        return {
            "stage_a": {
                "root": str(Path(self.release.root).absolute()),
                "closure": self.release.closure,
                "closure_file_sha256": self.release.closure_file_sha256,
                "closure_payload_sha256": self.release.closure_payload_sha256,
            },
            "annotation_binding": {
                "annotation_bindings": self.annotation_binding.annotation_bindings,
                "eligibility": self.annotation_binding.eligibility,
                "eligibility_payload_sha256": self.annotation_binding.eligibility_payload_sha256,
            },
            "training_releases": [
                {
                    "outer_rotation": row.outer_rotation,
                    "field": row.field,
                    "rows": list(row.rows),
                    "released_annotations_sha256": row.released_annotations_sha256,
                    "category_counts": list(row.category_counts),
                }
                for row in self.training_releases
            ],
            "eligibility_result": (
                None
                if eligibility is None
                else {
                    "manifest": eligibility.manifest,
                    "manifest_file_sha256": hashlib.sha256(eligibility.manifest_bytes).hexdigest(),
                    "manifest_payload_sha256": eligibility.manifest_sha256,
                    "fold_file_sha256": hashlib.sha256(eligibility.fold_manifest_bytes).hexdigest(),
                    "feature_summary_sha256": [
                        {
                            "session_id": feature.identity.session_id,
                            **{
                                modality: hashlib.sha256(
                                    np.asarray(getattr(feature, modality)).tobytes(order="C")
                                ).hexdigest()
                                for modality in SUMMARY_DIMENSIONS
                            },
                        }
                        for feature in eligibility.features
                    ],
                }
            ),
            "targets": (
                None
                if targets is None
                else {
                    "manifest": targets.manifest,
                    "manifest_file_sha256": hashlib.sha256(targets.manifest_bytes).hexdigest(),
                    "manifest_payload_sha256": targets.manifest_sha256,
                    "values": [list(row) for row in targets.values],
                }
            ),
            "provenance": {
                "config_sha256": self.provenance.config_sha256,
                "code_commit_sha": self.provenance.code_commit_sha,
                "code_source_tree_sha256": self.provenance.code_source_tree_sha256,
                "source_date_epoch": self.provenance.source_date_epoch,
            },
        }


StageBEvidenceReloader = Callable[[str], StageBOracleEvidence]


@dataclass(frozen=True, slots=True)
class _OracleTrainingGroups:
    source: str
    groups: Mapping[str, int]
    session_pairs: Mapping[str, tuple[str, str]]

    def participant_groups(self, role: str) -> dict[str, int]:
        if role != "train":
            raise ValueError("Stage-B protected groups may be released only for training")
        return dict(self.groups)

    def paired_training_schedule(
        self, *, seed: int, batch_count: int
    ) -> tuple[dict[str, Any], ...]:
        participant_ids = np.asarray(sorted(self.groups, key=int), dtype=object)
        group_values = np.asarray(
            [self.groups[participant_id] for participant_id in participant_ids],
            dtype=np.int8,
        )
        return BalancedParticipantSampler(
            participant_ids,
            group_values,
            seed=seed,
        ).paired_schedule(self.session_pairs, batch_count=batch_count)


def stage_b_oracle_model_config(field: str, categories: Sequence[str | int]) -> dict[str, Any]:
    """Return the one frozen architecture-controlled Oracle-W1 configuration."""
    category_values = list(categories)
    if not isinstance(field, str) or not field or len(category_values) != 2:
        raise ValueError("Stage-B oracle requires one exact binary codebook field")
    if category_values[0] == category_values[1]:
        raise ValueError("Stage-B oracle codebook categories must be distinct")
    return {
        "configuration": f"oracle_{field}",
        "architecture": "low_capacity_multimodal_scorer_26721_parameters",
        "trainable_parameter_count": 26_721,
        "modality_scalers": "separate_training_session_row_standard_scalers",
        "linear_initialisation": "xavier_uniform_weight_zero_bias",
        "layer_norm_initialisation": "one_weight_zero_bias",
        "oracle_field": field,
        "codebook_categories_in_frozen_order": category_values,
        "group_encoding": [
            {"category": category, "group": index} for index, category in enumerate(category_values)
        ],
        "protected_annotation_scope": "outer_rotation_training_participants_only",
        "protected_validation_annotations_accessed": [],
        "protected_test_annotations_accessed": [],
        "category_merge": "forbidden",
        "category_drop": "forbidden",
        "post_hoc_binarisation": "forbidden",
        "participant_batch_size": PROXY_PARTICIPANTS_PER_BATCH,
        "participants_per_group": PROXY_PARTICIPANTS_PER_GROUP,
        "session_rows_per_batch": PROXY_SESSION_ROWS_PER_BATCH,
        "both_sessions_enter_batch_together": True,
        "optimizer_steps_per_epoch": "ceil(training_participant_count/16)",
        "sampler_state": "single_seeded_cyclic_state_continuous_across_epochs",
        "validation_sampling": "complete_natural_role_distribution",
        "test_sampling": "complete_natural_role_distribution",
        "task_loss": "unweighted_mse_on_normalised_recommend_hiring",
        "wasserstein": {
            "coefficient": WASSERSTEIN_COEFFICIENT,
            "application": "exact_balanced_empirical_w1_from_update_one_applied_once",
        },
        "adversarial_loss": "forbidden",
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
        "checkpoint_metric": "ordinary_validation_srcc_without_protected_annotations",
        "checkpoint_direction": "maximize",
        "exact_tie_break": "earliest_epoch",
        "gradient_norm_clip": 1.0,
        "prediction_postprocessing": "none_sigmoid_is_model_output",
        "device": "cpu",
    }


def run_stage_b_oracle_seed(
    eligibility: EligibilityResult,
    targets: TargetSet,
    *,
    release: Any,
    annotation_binding: Any,
    training_releases: Sequence[Any],
    seed: int,
    provenance: RunProvenance,
    synthetic_smoke_epoch_cap: int | None = None,
) -> StageBOracleSeedBundle:
    """Fit all five protected-training-only rotations for one frozen seed."""
    from src.eval.stage_b_oracle import (
        ORACLE_ACCESS_CHECKSUM_FIELD,
        ORACLE_PREDICTION_CHECKSUM_FIELD,
        ORACLE_PREDICTION_LOCK_CHECKSUM_FIELD,
        STAGE_B_ORACLE_PREDICTION_LOCK_SCHEMA_VERSION,
        STAGE_B_ORACLE_PREDICTION_SCHEMA_VERSION,
        OracleAnnotationBinding,
        OracleTrainingAnnotationRelease,
        build_oracle_access_log,
        build_stage_b_execution_provenance,
        oracle_eligibility_payload_sha256,
        oracle_prediction_lock_id,
    )

    if seed not in FROZEN_FINAL_SEEDS:
        raise ValueError("Stage-B oracle seed is not frozen")
    if not isinstance(annotation_binding, OracleAnnotationBinding):
        raise TypeError("Stage-B seed requires a label-free annotation binding")
    if any(not isinstance(row, OracleTrainingAnnotationRelease) for row in training_releases):
        raise TypeError("Stage-B seed requires training-only annotation releases")
    if synthetic_smoke_epoch_cap is not None and (
        isinstance(synthetic_smoke_epoch_cap, bool)
        or not isinstance(synthetic_smoke_epoch_cap, int)
        or not 1 <= synthetic_smoke_epoch_cap <= 200
    ):
        raise ValueError("synthetic Stage-B epoch cap must be an integer from 1 to 200")
    maximum_epochs = synthetic_smoke_epoch_cap or 200
    execution_mode = (
        "synthetic_contract_test" if synthetic_smoke_epoch_cap is not None else "research"
    )
    if execution_mode == "research" and (
        not eligibility.production_ready or not targets.production_ready
    ):
        raise ValueError("research Stage B requires research-ready eligibility and targets")
    if execution_mode == "synthetic_contract_test" and (
        eligibility.production_ready or targets.production_ready
    ):
        raise ValueError("synthetic Stage B may not consume research-ready inputs")
    if release.closure.get("execution_mode") != execution_mode:
        raise ValueError("Stage-B execution mode disagrees with the Stage-A release")
    _validate_runner_context(eligibility, targets, release, provenance)
    stage_b_execution_provenance = build_stage_b_execution_provenance(provenance)
    oracle_eligibility = annotation_binding.eligibility
    field, categories = _eligible_field_categories(oracle_eligibility)
    configuration = f"oracle_{field}"
    model_config = stage_b_oracle_model_config(field, categories)
    model_config_sha256 = _payload_sha256(model_config)
    examples = _bind_targets(eligibility, targets)
    access_log = build_oracle_access_log(
        release=release,
        annotation_binding=annotation_binding,
        training_releases=training_releases,
        seed=seed,
    )
    access_log_bytes = canonical_json_bytes(access_log)
    access_log_filename = f"{configuration}.seed-{seed}.access-log.json"
    access_log_file_sha256 = hashlib.sha256(access_log_bytes).hexdigest()
    access_entries = {int(row["outer_rotation"]): row for row in access_log["entries"]}
    spec = ProxyConfigurationSpec(
        source=cast(Any, STAGE_B_ORACLE_GROUP_SOURCE),
        training_mode=STAGE_B_ORACLE_SAMPLING_MODE,
        lambda_w=WASSERSTEIN_COEFFICIENT,
    )
    prediction_rows: list[dict[str, Any]] = []
    rotations: list[dict[str, Any]] = []
    fit_artifacts: list[ArtifactPayload] = []
    releases_by_fold = {row.outer_rotation: row for row in tuple(training_releases)}
    if set(releases_by_fold) != set(FROZEN_OUTER_ROTATIONS):
        raise ValueError("Stage-B seed requires five exact rotation releases")

    for fold in FROZEN_OUTER_ROTATIONS:
        train, validation, test = _rotation_examples(examples, fold)
        training_release = releases_by_fold[fold]
        if training_release.field != field:
            raise ValueError("Stage-B rotation release field drifted")
        release_rows = [dict(row) for row in training_release.rows]
        training_ids = sorted({row.feature.identity.participant_id for row in train}, key=int)
        if [row["participant_id"] for row in release_rows] != training_ids:
            raise ValueError("Stage-B annotation release does not exactly match training role")
        category_index = {category: index for index, category in enumerate(categories)}
        try:
            groups = {
                str(row["participant_id"]): category_index[row["value"]] for row in release_rows
            }
        except KeyError as error:
            raise ValueError("Stage-B release contains a non-codebook category") from error
        group_counts = [sum(value == group for value in groups.values()) for group in (0, 1)]
        if any(count < 10 for count in group_counts):
            raise ValueError("Stage-B training category fell below the frozen threshold")
        session_pairs = _training_session_pairs(train)
        training_groups = _OracleTrainingGroups(
            source=STAGE_B_ORACLE_GROUP_SOURCE,
            groups=groups,
            session_pairs=session_pairs,
        )
        result = _fit_proxy_rotation(
            train,
            validation,
            test,
            proxy=cast(Any, training_groups),
            configuration=configuration,
            spec=spec,
            seed=seed,
            maximum_epochs=maximum_epochs,
        )
        entry = access_entries[fold]
        fit = _build_fit_artifact(
            result,
            release=release,
            oracle_eligibility=oracle_eligibility,
            seed=seed,
            outer_rotation=fold,
            access_log=access_log,
            access_entry=entry,
            model_config_sha256=model_config_sha256,
            maximum_epochs=maximum_epochs,
            execution_mode=execution_mode,
            stage_b_execution_provenance=stage_b_execution_provenance,
        )
        fit_artifacts.append(fit)
        rotations.append(
            {
                "outer_rotation": fold,
                "training_annotation_count": len(training_ids),
                "access_log_entry_payload_sha256": entry["entry_payload_sha256"],
                "checkpoint_epoch": result.checkpoint_epoch,
                "fit_artifact": {"filename": fit.filename, "sha256": fit.sha256},
            }
        )
        prediction_rows.extend(
            _oracle_prediction_rows(
                test,
                result.test_prediction,
                release=release,
                field=field,
                configuration=configuration,
                seed=seed,
                model_config_sha256=model_config_sha256,
                model_fit_sha256=fit.sha256,
                checkpoint_epoch=result.checkpoint_epoch,
                access_log_payload_sha256=access_log[ORACLE_ACCESS_CHECKSUM_FIELD],
            )
        )

    context = release.closure["context"]
    prediction_filename = f"{configuration}.seed-{seed}.predictions.json"
    prediction_lock_filename = f"{configuration}.seed-{seed}.prediction-lock.json"
    annotation_bindings = dict(annotation_binding.annotation_bindings)
    lock_identity = {
        "schema_version": STAGE_B_ORACLE_PREDICTION_LOCK_SCHEMA_VERSION,
        "configuration": configuration,
        "seed": seed,
        "firewall_stage": "B",
        "stage_a_closure_file_sha256": release.closure_file_sha256,
        "stage_a_closure_payload_sha256": release.closure_payload_sha256,
        "oracle_eligibility_payload_sha256": oracle_eligibility_payload_sha256(oracle_eligibility),
        "stage_b_execution_provenance": stage_b_execution_provenance,
        "annotation_bindings": annotation_bindings,
        "annotation_access_log": {
            "filename": access_log_filename,
            "file_sha256": access_log_file_sha256,
            "payload_sha256": access_log[ORACLE_ACCESS_CHECKSUM_FIELD],
        },
        "target_manifest_sha256": context["target_manifest_sha256"],
        "eligibility_manifest_sha256": context["eligibility_manifest_sha256"],
        "feature_contract_sha256": context["feature_contract_sha256"],
        "preprocessing_sha256": context["preprocessing_sha256"],
        "feature_extraction_closure_file_sha256": context["feature_extraction_closure_file_sha256"],
        "feature_extraction_closure_payload_sha256": context[
            "feature_extraction_closure_payload_sha256"
        ],
        "session_mapping": context["session_mapping"],
        "config_sha256": stage_b_execution_provenance["config_sha256"],
        "code_commit_sha": stage_b_execution_provenance["code_commit_sha"],
        "code_source_tree_sha256": stage_b_execution_provenance["code_source_tree_sha256"],
        "source_date_epoch": stage_b_execution_provenance["source_date_epoch"],
        "model_config_sha256": model_config_sha256,
        "rotations": rotations,
    }
    prediction_lock_id = oracle_prediction_lock_id(lock_identity)
    locked_rows = [{**row, "prediction_lock_id": prediction_lock_id} for row in prediction_rows]
    prediction_unsigned = {
        "schema_version": STAGE_B_ORACLE_PREDICTION_SCHEMA_VERSION,
        "prediction_lock_id": prediction_lock_id,
        "rows": locked_rows,
    }
    prediction_payload = _signed(prediction_unsigned, ORACLE_PREDICTION_CHECKSUM_FIELD)
    prediction_bytes = canonical_json_bytes(prediction_payload)
    prediction_file_sha256 = hashlib.sha256(prediction_bytes).hexdigest()
    manifest_unsigned = {
        **lock_identity,
        "state": "locked",
        "protected_training_annotation_scope": ("outer_rotation_training_participants_only"),
        "protected_validation_annotation_fields_accessed": [],
        "protected_test_annotation_fields_accessed": [],
        "checkpoint_selection": ("ordinary_validation_srcc_without_protected_annotations"),
        "eligible_participant_count": release.participant_count,
        "expected_oof_session_count": 2 * release.participant_count,
        "observed_oof_session_count": len(locked_rows),
        "predictions": {
            "filename": prediction_filename,
            "file_sha256": prediction_file_sha256,
            "payload_sha256": prediction_payload[ORACLE_PREDICTION_CHECKSUM_FIELD],
            "row_count": len(locked_rows),
        },
        "prediction_lock_id": prediction_lock_id,
    }
    prediction_lock = _signed(manifest_unsigned, ORACLE_PREDICTION_LOCK_CHECKSUM_FIELD)
    prediction_lock_bytes = canonical_json_bytes(prediction_lock)
    prediction_lock_file_sha256 = hashlib.sha256(prediction_lock_bytes).hexdigest()
    bundle_record = {
        "configuration": configuration,
        "seed": seed,
        "prediction_lock_filename": prediction_lock_filename,
        "prediction_lock_file_sha256": prediction_lock_file_sha256,
        "prediction_lock_payload_sha256": prediction_lock[ORACLE_PREDICTION_LOCK_CHECKSUM_FIELD],
        "prediction_lock_id": prediction_lock_id,
        "prediction_filename": prediction_filename,
        "prediction_file_sha256": prediction_file_sha256,
        "prediction_payload_sha256": prediction_payload[ORACLE_PREDICTION_CHECKSUM_FIELD],
        "oof_row_count": len(locked_rows),
        "access_log_filename": access_log_filename,
        "access_log_file_sha256": access_log_file_sha256,
        "access_log_payload_sha256": access_log[ORACLE_ACCESS_CHECKSUM_FIELD],
        "fit_artifacts": [
            {"filename": artifact.filename, "sha256": artifact.sha256} for artifact in fit_artifacts
        ],
    }
    return StageBOracleSeedBundle(
        configuration=configuration,
        seed=seed,
        access_log_filename=access_log_filename,
        access_log=access_log,
        access_log_bytes=access_log_bytes,
        prediction_filename=prediction_filename,
        prediction_payload=prediction_payload,
        prediction_bytes=prediction_bytes,
        prediction_lock_filename=prediction_lock_filename,
        prediction_lock=prediction_lock,
        prediction_lock_bytes=prediction_lock_bytes,
        fit_artifacts=tuple(fit_artifacts),
        bundle_record=bundle_record,
    )


def run_conditional_stage_b_oracle(
    evidence: StageBOracleEvidence,
    *,
    output_dir: str | Path,
    evidence_reloader: StageBEvidenceReloader | None = None,
    synthetic_smoke_epoch_cap: int | None = None,
) -> Any:
    """Publish and validate either all fifteen oracle fits or the zero-fit omission."""
    from src.eval.stage_b_oracle import (
        STAGE_B_ORACLE_LOCK_FILENAME,
        build_stage_b_execution_provenance,
        build_stage_b_oracle_lock,
        build_stage_b_oracle_omission,
        validate_stage_b_oracle_lock_from_binding,
        write_stage_b_oracle_lock,
    )

    if not isinstance(evidence, StageBOracleEvidence):
        raise TypeError("Stage B requires a validated evidence capability")
    expected = evidence
    current = _reload_stage_b_evidence(expected, evidence_reloader, "before_eligibility")
    with AnchoredDirectory.open(output_dir, create=True, directory_mode=0o700) as output:
        if STAGE_B_ORACLE_LOCK_FILENAME in output.list_names():
            current = _reload_stage_b_evidence(expected, evidence_reloader, "readback")
            return validate_stage_b_oracle_lock_from_binding(
                output,
                release=current.release,
                annotation_binding=current.annotation_binding,
                training_releases=current.training_releases,
            )

        oracle_eligibility = current.oracle_eligibility
        status = oracle_eligibility.get("status")
        if status == "omitted_all_15":
            _require_clean_oracle_directory(output)
            current = _reload_stage_b_evidence(expected, evidence_reloader, "pre_lock")
            payload = build_stage_b_oracle_omission(
                release=current.release,
                annotation_binding=current.annotation_binding,
                stage_b_execution_provenance=build_stage_b_execution_provenance(current.provenance),
            )
            write_stage_b_oracle_lock(output, payload)
        elif status == "eligible_all_15":
            if not isinstance(current.eligibility, EligibilityResult) or not isinstance(
                current.targets, TargetSet
            ):
                raise ValueError("eligible Stage B requires frozen features and targets")
            bundle_records = []
            for seed in FROZEN_FINAL_SEEDS:
                current = _reload_stage_b_evidence(
                    expected, evidence_reloader, f"before_seed_{seed}"
                )
                if not isinstance(current.eligibility, EligibilityResult) or not isinstance(
                    current.targets, TargetSet
                ):
                    raise RuntimeError("Stage-B reloader dropped fit inputs")
                bundle = run_stage_b_oracle_seed(
                    current.eligibility,
                    current.targets,
                    release=current.release,
                    annotation_binding=current.annotation_binding,
                    training_releases=current.training_releases,
                    seed=seed,
                    provenance=current.provenance,
                    synthetic_smoke_epoch_cap=synthetic_smoke_epoch_cap,
                )
                _reload_stage_b_evidence(expected, evidence_reloader, f"after_seed_{seed}")
                write_stage_b_oracle_seed_bundle(output, bundle)
                bundle_records.append(bundle.bundle_record)
            current = _reload_stage_b_evidence(expected, evidence_reloader, "pre_lock")
            payload = build_stage_b_oracle_lock(
                release=current.release,
                annotation_binding=current.annotation_binding,
                stage_b_execution_provenance=build_stage_b_execution_provenance(current.provenance),
                bundles=bundle_records,
            )
            write_stage_b_oracle_lock(output, payload)
        else:
            raise ValueError("Stage-B oracle eligibility has no frozen disposition")
        output.verify_path_identity()
        current = _reload_stage_b_evidence(expected, evidence_reloader, "readback")
        return validate_stage_b_oracle_lock_from_binding(
            output,
            release=current.release,
            annotation_binding=current.annotation_binding,
            training_releases=current.training_releases,
        )


def _reload_stage_b_evidence(
    expected: StageBOracleEvidence,
    reloader: StageBEvidenceReloader | None,
    boundary: str,
) -> StageBOracleEvidence:
    expected.verify_integrity()
    research = expected.release.closure.get("execution_mode") == "research"
    if research and reloader is None:
        raise ValueError("research Stage B requires an exact evidence reloader")
    current = expected if reloader is None else reloader(boundary)
    if not isinstance(current, StageBOracleEvidence):
        raise TypeError("Stage-B evidence reloader returned an invalid capability")
    current.verify_integrity()
    if current.fingerprint_sha256 != expected.fingerprint_sha256:
        raise RuntimeError(f"Stage-B source evidence changed at boundary {boundary!r}")
    return current


def write_stage_b_oracle_seed_bundle(
    output_dir: str | Path | AnchoredDirectory, bundle: StageBOracleSeedBundle
) -> tuple[Path, ...]:
    """Publish one complete seed bundle without replacing non-identical bytes."""
    if not isinstance(bundle, StageBOracleSeedBundle):
        raise TypeError("bundle must be a StageBOracleSeedBundle")
    if isinstance(output_dir, AnchoredDirectory):
        return _write_stage_b_oracle_seed_bundle(output_dir, bundle)
    with AnchoredDirectory.open(output_dir, create=True, directory_mode=0o700) as anchored:
        return _write_stage_b_oracle_seed_bundle(anchored, bundle)


def _write_stage_b_oracle_seed_bundle(
    output: AnchoredDirectory, bundle: StageBOracleSeedBundle
) -> tuple[Path, ...]:
    output.verify_path_identity()
    artifacts = [
        *bundle.fit_artifacts,
        ArtifactPayload(bundle.access_log_filename, bundle.access_log_bytes),
        ArtifactPayload(bundle.prediction_filename, bundle.prediction_bytes),
        ArtifactPayload(bundle.prediction_lock_filename, bundle.prediction_lock_bytes),
    ]
    destinations: list[ArtifactPayload] = []
    seen: set[str] = set()
    existing_names = set(output.list_names())
    for artifact in artifacts:
        if artifact.filename in seen or Path(artifact.filename).name != artifact.filename:
            raise ValueError("Stage-B bundle artifact filename is unsafe or duplicated")
        seen.add(artifact.filename)
        if artifact.filename in existing_names:
            existing = output.read_bytes(artifact.filename, expected_mode=0o600)
            if existing.content != artifact.content:
                raise FileExistsError(f"refusing to replace Stage-B artifact {artifact.filename}")
        destinations.append(artifact)
    written: list[Path] = []
    for artifact in destinations:
        observed = output.publish_bytes(
            artifact.filename, artifact.content, mode=0o600, allow_existing=True
        )
        if observed.sha256 != artifact.sha256 or observed.size != len(artifact.content):
            raise RuntimeError("Stage-B artifact failed anchored readback validation")
        written.append(output.path / artifact.filename)
    output.verify_path_identity()
    return tuple(written)


def validate_stage_b_oracle_fit_artifact(
    path: str | Path,
    *,
    release: Any,
    annotation_binding: Any,
    training_release: Any,
    seed: int,
    outer_rotation: int,
    access_log: Mapping[str, Any],
    prediction_lock_manifest: Mapping[str, Any],
) -> None:
    """Fail closed on every deterministic Stage-B fit state and context binding."""
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("Stage-B oracle fit must be a regular non-symlink file")
    validate_stage_b_oracle_fit_bytes(
        source.read_bytes(),
        filename=source.name,
        release=release,
        annotation_binding=annotation_binding,
        training_release=training_release,
        seed=seed,
        outer_rotation=outer_rotation,
        access_log=access_log,
        prediction_lock_manifest=prediction_lock_manifest,
    )


def validate_stage_b_oracle_fit_bytes(
    content: bytes,
    *,
    filename: str,
    release: Any,
    annotation_binding: Any,
    training_release: Any,
    seed: int,
    outer_rotation: int,
    access_log: Mapping[str, Any],
    prediction_lock_manifest: Mapping[str, Any],
) -> None:
    """Validate fit bytes already read through an anchored directory handle."""
    if not isinstance(content, bytes):
        raise TypeError("Stage-B oracle fit content must be bytes")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise ValueError("Stage-B oracle fit filename must be a safe basename")
    _validate_fit_bytes(
        content,
        filename=filename,
        release=release,
        annotation_binding=annotation_binding,
        training_release=training_release,
        seed=seed,
        outer_rotation=outer_rotation,
        access_log=access_log,
        prediction_lock_manifest=prediction_lock_manifest,
    )


def _build_fit_artifact(
    result: Any,
    *,
    release: Any,
    oracle_eligibility: Mapping[str, Any],
    seed: int,
    outer_rotation: int,
    access_log: Mapping[str, Any],
    access_entry: Mapping[str, Any],
    model_config_sha256: str,
    maximum_epochs: int,
    execution_mode: str,
    stage_b_execution_provenance: Mapping[str, Any],
) -> ArtifactPayload:
    from src.eval.stage_b_oracle import oracle_eligibility_payload_sha256

    field, categories = _eligible_field_categories(oracle_eligibility)
    context = release.closure["context"]
    arrays = dict(result.artifact_arrays)
    arrays.pop("proxy_source_utf8")
    arrays["schema_version_utf8"] = _utf8_array(STAGE_B_ORACLE_FIT_SCHEMA_VERSION)
    arrays["configuration_utf8"] = _utf8_array(f"oracle_{field}")
    arrays["execution_mode_utf8"] = _utf8_array(execution_mode)
    arrays["group_source_utf8"] = _utf8_array(STAGE_B_ORACLE_GROUP_SOURCE)
    arrays["sampling_mode_utf8"] = _utf8_array(STAGE_B_ORACLE_SAMPLING_MODE)
    arrays["oracle_field_utf8"] = _utf8_array(field)
    arrays["category_0_json_utf8"] = _canonical_value_array(categories[0])
    arrays["category_1_json_utf8"] = _canonical_value_array(categories[1])
    arrays["stage_a_closure_file_sha256_utf8"] = _utf8_array(release.closure_file_sha256)
    arrays["stage_a_closure_payload_sha256_utf8"] = _utf8_array(release.closure_payload_sha256)
    arrays["oracle_eligibility_payload_sha256_utf8"] = _utf8_array(
        oracle_eligibility_payload_sha256(oracle_eligibility)
    )
    arrays["access_log_payload_sha256_utf8"] = _utf8_array(access_log["access_log_payload_sha256"])
    arrays["access_log_entry_payload_sha256_utf8"] = _utf8_array(
        access_entry["entry_payload_sha256"]
    )
    arrays["released_annotations_sha256_utf8"] = _utf8_array(
        access_entry["released_annotations_sha256"]
    )
    for name in (
        "target_manifest_sha256",
        "eligibility_manifest_sha256",
        "feature_contract_sha256",
        "preprocessing_sha256",
    ):
        arrays[f"{name}_utf8"] = _utf8_array(str(context[name]))
    for name in ("config_sha256", "code_commit_sha", "code_source_tree_sha256"):
        arrays[f"{name}_utf8"] = _utf8_array(str(stage_b_execution_provenance[name]))
    arrays["stage_b_execution_provenance_json_utf8"] = np.frombuffer(
        canonical_json_bytes(stage_b_execution_provenance), dtype=np.uint8
    )
    arrays["session_mapping_json_utf8"] = np.frombuffer(
        canonical_json_bytes(context["session_mapping"]), dtype=np.uint8
    )
    arrays["model_config_sha256_utf8"] = _utf8_array(model_config_sha256)
    arrays["training_history_json_utf8"] = np.frombuffer(
        canonical_json_bytes({"rows": list(result.training_history)}), dtype=np.uint8
    )
    arrays["minibatch_diagnostics_json_utf8"] = np.frombuffer(
        canonical_json_bytes({"rows": list(result.minibatch_diagnostics)}),
        dtype=np.uint8,
    )
    arrays["schedule_json_utf8"] = np.frombuffer(
        canonical_json_bytes({"rows": list(result.used_schedule)}), dtype=np.uint8
    )
    arrays["seed"] = np.asarray(seed, dtype="<i8")
    arrays["outer_rotation"] = np.asarray(outer_rotation, dtype="<i8")
    arrays["source_date_epoch"] = np.asarray(
        stage_b_execution_provenance["source_date_epoch"], dtype="<i8"
    )
    arrays["executed_maximum_epochs"] = np.asarray(maximum_epochs, dtype="<i8")
    arrays["protected_annotation_field_count"] = np.asarray(1, dtype="<i8")
    arrays["protected_validation_annotation_field_count"] = np.asarray(0, dtype="<i8")
    arrays["protected_test_annotation_field_count"] = np.asarray(0, dtype="<i8")
    filename = f"oracle_{field}.seed-{seed}.fold-{outer_rotation}.fit.npz"
    return ArtifactPayload(filename, deterministic_npz_bytes(arrays))


def _validate_fit_bytes(
    content: bytes,
    *,
    filename: str,
    release: Any,
    annotation_binding: Any,
    training_release: Any,
    seed: int,
    outer_rotation: int,
    access_log: Mapping[str, Any],
    prediction_lock_manifest: Mapping[str, Any],
) -> None:
    from src.eval.stage_b_oracle import (
        oracle_eligibility_payload_sha256,
        validate_stage_b_execution_provenance,
    )

    oracle_eligibility = annotation_binding.eligibility
    field, categories = _eligible_field_categories(oracle_eligibility)
    expected_filename = f"oracle_{field}.seed-{seed}.fold-{outer_rotation}.fit.npz"
    if filename != expected_filename:
        raise ValueError("Stage-B oracle fit filename is not canonical")
    try:
        with np.load(io.BytesIO(content), allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError) as error:
        raise ValueError("Stage-B oracle fit archive cannot be read") from error
    if deterministic_npz_bytes(arrays) != content:
        raise ValueError("Stage-B oracle fit is not byte-deterministic")
    if set(arrays) != _expected_fit_array_names():
        raise ValueError("Stage-B oracle fit array schema drifted")
    for name in _FIT_UTF8_FIELDS:
        _archive_utf8(arrays, name)
    for name in _FIT_FLOAT_FIELDS:
        value = np.asarray(arrays[name])
        if value.dtype != np.dtype("<f8") or value.shape != () or not math.isfinite(float(value)):
            raise ValueError(f"Stage-B oracle fit {name} is invalid")
    for name in _FIT_INTEGER_FIELDS:
        value = np.asarray(arrays[name])
        if value.dtype != np.dtype("<i8") or value.shape != ():
            raise ValueError(f"Stage-B oracle fit {name} is invalid")

    if seed not in FROZEN_FINAL_SEEDS or outer_rotation not in FROZEN_OUTER_ROTATIONS:
        raise ValueError("Stage-B oracle fit seed/fold is not frozen")
    context = release.closure["context"]
    access_entries = {int(row["outer_rotation"]): row for row in access_log["entries"]}
    entry = access_entries.get(outer_rotation)
    if not isinstance(entry, Mapping):
        raise ValueError("Stage-B oracle fit lacks its access-log entry")
    expected_model_config = stage_b_oracle_model_config(field, categories)
    stage_b_execution_provenance = validate_stage_b_execution_provenance(
        _canonical_object(
            arrays["stage_b_execution_provenance_json_utf8"],
            "execution provenance",
        )
    )
    if stage_b_execution_provenance != prediction_lock_manifest.get("stage_b_execution_provenance"):
        raise ValueError("Stage-B oracle fit and prediction lock provenance disagree")
    expected_strings = {
        "schema_version_utf8": STAGE_B_ORACLE_FIT_SCHEMA_VERSION,
        "shared_fit_schema_version_utf8": FIT_ARTIFACT_SCHEMA_VERSION,
        "configuration_utf8": f"oracle_{field}",
        "execution_mode_utf8": (
            "research"
            if release.closure.get("execution_mode") == "research"
            else "synthetic_contract_test"
        ),
        "group_source_utf8": STAGE_B_ORACLE_GROUP_SOURCE,
        "sampling_mode_utf8": STAGE_B_ORACLE_SAMPLING_MODE,
        "oracle_field_utf8": field,
        "stage_a_closure_file_sha256_utf8": release.closure_file_sha256,
        "stage_a_closure_payload_sha256_utf8": release.closure_payload_sha256,
        "oracle_eligibility_payload_sha256_utf8": oracle_eligibility_payload_sha256(
            oracle_eligibility
        ),
        "access_log_payload_sha256_utf8": access_log["access_log_payload_sha256"],
        "access_log_entry_payload_sha256_utf8": entry["entry_payload_sha256"],
        "released_annotations_sha256_utf8": entry["released_annotations_sha256"],
        "target_manifest_sha256_utf8": context["target_manifest_sha256"],
        "eligibility_manifest_sha256_utf8": context["eligibility_manifest_sha256"],
        "feature_contract_sha256_utf8": context["feature_contract_sha256"],
        "preprocessing_sha256_utf8": context["preprocessing_sha256"],
        "config_sha256_utf8": stage_b_execution_provenance["config_sha256"],
        "code_commit_sha_utf8": stage_b_execution_provenance["code_commit_sha"],
        "code_source_tree_sha256_utf8": stage_b_execution_provenance["code_source_tree_sha256"],
        "model_config_sha256_utf8": _payload_sha256(expected_model_config),
    }
    if any(_archive_utf8(arrays, name) != value for name, value in expected_strings.items()):
        raise ValueError("Stage-B oracle fit context string binding drifted")
    if (
        _canonical_value(arrays["category_0_json_utf8"]) != categories[0]
        or _canonical_value(arrays["category_1_json_utf8"]) != categories[1]
    ):
        raise ValueError("Stage-B oracle fit category encoding drifted")
    if (
        _canonical_json_value(arrays["session_mapping_json_utf8"], "session mapping")
        != (context["session_mapping"])
    ):
        raise ValueError("Stage-B oracle fit session mapping drifted")
    scalar_expectations = {
        "seed": seed,
        "outer_rotation": outer_rotation,
        "source_date_epoch": stage_b_execution_provenance["source_date_epoch"],
        "participant_batch_size": PROXY_PARTICIPANTS_PER_BATCH,
        "session_rows_per_batch": PROXY_SESSION_ROWS_PER_BATCH,
        "protected_annotation_field_count": 1,
        "protected_validation_annotation_field_count": 0,
        "protected_test_annotation_field_count": 0,
        "adversarial_loss_component_count": 0,
        "optimizer.state_count": len(_MODEL_STATE_SHAPES),
    }
    if any(int(arrays[name]) != value for name, value in scalar_expectations.items()):
        raise ValueError("Stage-B oracle fit scalar contract drifted")
    if float(arrays["lambda_w"]) != WASSERSTEIN_COEFFICIENT:
        raise ValueError("Stage-B oracle fit W1 coefficient drifted")
    checkpoint_epoch = int(arrays["checkpoint_epoch"])
    checkpoint_step = int(arrays["checkpoint_optimizer_step"])
    checkpoint_draws = int(arrays["checkpoint_sampler_draws_consumed"])
    epochs_completed = int(arrays["epochs_completed"])
    maximum_epochs = int(arrays["executed_maximum_epochs"])
    steps_per_epoch = int(arrays["optimizer_steps_per_epoch"])
    steps_completed = int(arrays["optimizer_steps_completed"])
    training_count = int(arrays["training_participant_count"])
    if (
        not 1 <= maximum_epochs <= 200
        or not 1 <= checkpoint_epoch <= epochs_completed <= maximum_epochs
        or steps_per_epoch != math.ceil(training_count / 16)
        or checkpoint_step != checkpoint_epoch * steps_per_epoch
        or checkpoint_draws != checkpoint_step
        or steps_completed != epochs_completed * steps_per_epoch
        or entry["released_participant_count"] != training_count
        or prediction_lock_manifest["rotations"][outer_rotation]["checkpoint_epoch"]
        != checkpoint_epoch
        or prediction_lock_manifest["model_config_sha256"] != _payload_sha256(expected_model_config)
    ):
        raise ValueError("Stage-B oracle fit checkpoint or count contract drifted")

    if training_release.outer_rotation != outer_rotation or training_release.field != field:
        raise ValueError("Stage-B fit received the wrong rotation-only annotation release")
    release_rows = [dict(row) for row in training_release.rows]
    if entry["released_annotations_sha256"] != training_release.released_annotations_sha256:
        raise ValueError("Stage-B fit annotation release checksum drifted")
    category_index = {category: index for index, category in enumerate(categories)}
    participant_ids = [str(row["participant_id"]) for row in release_rows]
    groups = np.asarray([category_index[row["value"]] for row in release_rows], dtype=np.int8)
    session_pairs = _release_training_session_pairs(release, participant_ids=participant_ids)
    expected_schedule = BalancedParticipantSampler(
        np.asarray(participant_ids, dtype=object), groups, seed=seed
    ).paired_schedule(session_pairs, batch_count=steps_completed)
    schedule = _canonical_rows(arrays["schedule_json_utf8"], "schedule")
    if schedule != list(expected_schedule):
        raise ValueError("Stage-B oracle fit schedule is not the exact training-only release")
    history = _canonical_rows(arrays["training_history_json_utf8"], "history")
    minibatches = _canonical_rows(
        arrays["minibatch_diagnostics_json_utf8"], "minibatch diagnostics"
    )
    _validate_training_history(
        history,
        minibatches,
        expected_schedule,
        epochs_completed=epochs_completed,
        steps_per_epoch=steps_per_epoch,
        checkpoint_epoch=checkpoint_epoch,
        checkpoint_validation_srcc=float(arrays["checkpoint_validation_srcc"]),
        maximum_epochs=maximum_epochs,
    )
    _validate_numeric_state(
        arrays,
        training_count=training_count,
        checkpoint_optimizer_step=checkpoint_step,
    )


def _validate_training_history(
    history: list[Any],
    minibatches: list[Any],
    schedule: Sequence[Mapping[str, Any]],
    *,
    epochs_completed: int,
    steps_per_epoch: int,
    checkpoint_epoch: int,
    checkpoint_validation_srcc: float,
    maximum_epochs: int,
) -> None:
    if len(history) != epochs_completed or len(minibatches) != len(schedule):
        raise ValueError("Stage-B oracle fit history length drifted")
    scores: list[float] = []
    for index, row in enumerate(history):
        if not isinstance(row, Mapping):
            raise ValueError("Stage-B oracle fit history row is invalid")
        score = row.get("validation_srcc")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("Stage-B oracle fit validation SRCC is invalid")
        score_value = float(score)
        if not math.isfinite(score_value) or not -1 <= score_value <= 1:
            raise ValueError("Stage-B oracle fit validation SRCC is invalid")
        if (
            row.get("epoch") != index + 1
            or row.get("optimizer_steps") != steps_per_epoch
            or row.get("cumulative_optimizer_steps") != (index + 1) * steps_per_epoch
            or row.get("sampler_draws_consumed") != (index + 1) * steps_per_epoch
            or row.get("validation_sampling") != "complete_natural_role_distribution"
        ):
            raise ValueError("Stage-B oracle fit epoch history drifted")
        scores.append(score_value)
    best_score = max(scores)
    best_epoch = scores.index(best_score) + 1
    expected_epochs = min(maximum_epochs, best_epoch + 25)
    if (
        checkpoint_epoch != best_epoch
        or checkpoint_validation_srcc != best_score
        or epochs_completed != expected_epochs
    ):
        raise ValueError("Stage-B oracle fit checkpoint selection drifted")
    for index, (row, draw) in enumerate(zip(minibatches, schedule, strict=True)):
        if not isinstance(row, Mapping):
            raise ValueError("Stage-B oracle minibatch diagnostic is invalid")
        objective = row.get("total_objective")
        task = row.get("task_mse")
        w1 = row.get("wasserstein1")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
            for value in (objective, task, w1)
        ):
            raise ValueError("Stage-B oracle minibatch objective is invalid")
        if (
            row.get("optimizer_step") != index + 1
            or row.get("schedule_index") != index
            or row.get("participant_group_counts") != [8, 8]
            or row.get("session_row_group_counts") != [16, 16]
            or row.get("objective_received_proxy_groups") is not True
            or row.get("lambda_w") != WASSERSTEIN_COEFFICIENT
            or row.get("group_evidence") != draw["groups"]
            or not math.isclose(
                float(objective),
                float(task) + WASSERSTEIN_COEFFICIENT * float(w1),
                rel_tol=1e-6,
                abs_tol=1e-8,
            )
        ):
            raise ValueError("Stage-B oracle minibatch balance/objective drifted")


def _validate_numeric_state(
    arrays: Mapping[str, np.ndarray],
    *,
    training_count: int,
    checkpoint_optimizer_step: int,
) -> None:
    for modality, dimension in SUMMARY_DIMENSIONS.items():
        for field in ("mean", "scale", "var"):
            value = np.asarray(arrays[f"scaler.{modality}.{field}"])
            if (
                value.dtype != np.dtype("<f8")
                or value.shape != (dimension,)
                or not np.isfinite(value).all()
                or (field == "scale" and np.any(value <= 0))
                or (field == "var" and np.any(value < 0))
            ):
                raise ValueError("Stage-B oracle scaler state is invalid")
        samples = np.asarray(arrays[f"scaler.{modality}.n_samples_seen"])
        if (
            samples.dtype != np.dtype("<i8")
            or samples.shape != ()
            or int(samples) != (2 * training_count)
        ):
            raise ValueError("Stage-B oracle scaler sample count is invalid")
    for parameter_index, (name, shape) in enumerate(_MODEL_STATE_SHAPES.items()):
        model = np.asarray(arrays[f"model.{name}"])
        if model.dtype != np.float32 or model.shape != shape or not np.isfinite(model).all():
            raise ValueError("Stage-B oracle model state is invalid")
        for moment in ("exp_avg", "exp_avg_sq"):
            value = np.asarray(arrays[f"optimizer.parameter-{parameter_index:04d}.{moment}"])
            if (
                value.dtype != np.float32
                or value.shape != shape
                or not np.isfinite(value).all()
                or (moment == "exp_avg_sq" and np.any(value < 0))
            ):
                raise ValueError("Stage-B oracle optimizer state is invalid")
        step = np.asarray(arrays[f"optimizer.parameter-{parameter_index:04d}.step"])
        if (
            step.shape != ()
            or not np.issubdtype(step.dtype, np.floating)
            or float(step) != checkpoint_optimizer_step
        ):
            raise ValueError("Stage-B oracle optimizer step is invalid")
    parameter_groups = _canonical_object(
        arrays["optimizer.param_groups_json_utf8"], "optimizer parameter groups"
    )
    groups = parameter_groups.get("param_groups")
    if not isinstance(groups, list) or len(groups) != 1:
        raise ValueError("Stage-B oracle optimizer parameter groups are invalid")
    expected = {
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
    if not isinstance(groups[0], Mapping):
        raise ValueError("Stage-B oracle optimizer configuration drifted")
    actual_group = dict(groups[0])
    if "decoupled_weight_decay" in actual_group:
        if actual_group.pop("decoupled_weight_decay") is not True:
            raise ValueError("Stage-B oracle optimizer configuration drifted")
    if actual_group != expected:
        raise ValueError("Stage-B oracle optimizer configuration drifted")


def _validate_runner_context(
    eligibility: EligibilityResult,
    targets: TargetSet,
    release: Any,
    provenance: RunProvenance,
) -> None:
    if not isinstance(eligibility, EligibilityResult) or not isinstance(targets, TargetSet):
        raise TypeError("Stage-B oracle requires validated eligibility and targets")
    if not isinstance(provenance, RunProvenance):
        raise TypeError("Stage-B oracle requires frozen run provenance")
    context = release.closure.get("context")
    if not isinstance(context, Mapping):
        raise ValueError("Stage-B oracle requires the exact Stage-A closure context")
    expected = {
        "eligibility_manifest_sha256": eligibility.manifest_sha256,
        "fold_manifest_sha256": eligibility.fold_manifest_sha256,
        "target_manifest_sha256": targets.manifest_sha256,
        "feature_contract_sha256": eligibility.feature_contract_sha256,
        "preprocessing_sha256": eligibility.preprocessing_sha256,
        "feature_extraction_closure_file_sha256": eligibility.manifest.get(
            "feature_extraction_closure_file_sha256"
        ),
        "feature_extraction_closure_payload_sha256": eligibility.manifest.get(
            "feature_extraction_closure_payload_sha256"
        ),
        "config_sha256": provenance.config_sha256,
        "source_date_epoch": provenance.source_date_epoch,
        "session_mapping": eligibility.manifest.get("session_mapping"),
        "eligible_participant_count": len(eligibility.participant_ids),
        "expected_oof_row_count": len(eligibility.features),
    }
    if any(context.get(name) != value for name, value in expected.items()):
        raise ValueError("Stage-B runner context does not match the immutable Stage-A closure")
    identities = {
        (
            row.feature.identity.participant_id,
            row.feature.identity.condition,
            row.feature.identity.session_id,
            row.feature.identity.source_session_id,
            row.feature.identity.fold,
            row.target,
        )
        for row in _bind_targets(eligibility, targets)
    }
    baseline_rows = release.learned_rows[("erm", FROZEN_FINAL_SEEDS[0])]
    baseline = {
        (
            row["participant_id"],
            row["session_phase"],
            row["session_id"],
            row["source_session_id"],
            row["fixed_fold"],
            row["target"],
        )
        for row in baseline_rows
    }
    if identities != baseline:
        raise ValueError("Stage-B runner roster/targets drifted from Stage-A OOF evidence")


def _eligible_field_categories(
    oracle_eligibility: Mapping[str, Any],
) -> tuple[str, tuple[str | int, str | int]]:
    field = oracle_eligibility.get("field")
    categories = oracle_eligibility.get("codebook_categories")
    if (
        oracle_eligibility.get("status") != "eligible_all_15"
        or oracle_eligibility.get("authorized_fit_count") != 15
        or oracle_eligibility.get("omitted_fit_count") != 0
        or oracle_eligibility.get("reasons") != []
        or not isinstance(field, str)
        or not isinstance(categories, list)
        or len(categories) != 2
        or categories[0] == categories[1]
    ):
        raise ValueError("Stage-B oracle eligibility does not authorize all fifteen fits")
    if any(isinstance(value, bool) or not isinstance(value, (str, int)) for value in categories):
        raise ValueError("Stage-B oracle categories have invalid types")
    return field, (categories[0], categories[1])


def _oracle_prediction_rows(
    rows: Sequence[Any],
    predictions: np.ndarray,
    *,
    release: Any,
    field: str,
    configuration: str,
    seed: int,
    model_config_sha256: str,
    model_fit_sha256: str,
    checkpoint_epoch: int,
    access_log_payload_sha256: str,
) -> list[dict[str, Any]]:
    from src.eval.stage_b_oracle import STAGE_B_ORACLE_PREDICTION_SCHEMA_VERSION

    baseline = {row["session_id"]: row for row in release.learned_rows[("erm", seed)]}
    values = np.asarray(predictions, dtype=np.float64)
    if values.shape != (len(rows),) or not np.isfinite(values).all():
        raise ValueError("Stage-B oracle predictions must be one finite value per test row")
    result = []
    for example, prediction in zip(rows, values, strict=True):
        identity = example.feature.identity
        base = baseline.get(identity.session_id)
        if (
            not isinstance(base, Mapping)
            or base["participant_id"] != identity.participant_id
            or base["source_session_id"] != identity.source_session_id
            or base["session_phase"] != identity.condition
            or base["fixed_fold"] != identity.fold
            or base["target"] != example.target
        ):
            raise ValueError("Stage-B test row drifted from the Stage-A OOF roster")
        result.append(
            {
                "schema_version": STAGE_B_ORACLE_PREDICTION_SCHEMA_VERSION,
                "corpus": "mit_interview",
                "participant_id": identity.participant_id,
                "session_id": identity.session_id,
                "source_session_id": identity.source_session_id,
                "session_phase": identity.condition,
                "fixed_fold": identity.fold,
                "outer_rotation": identity.fold,
                "rotation_role": "test",
                "seed": seed,
                "configuration": configuration,
                "model_config_sha256": model_config_sha256,
                "model_fit_sha256": model_fit_sha256,
                "target_transform": base["target_transform"],
                "target_manifest_sha256": base["target_manifest_sha256"],
                "target": example.target,
                "prediction": float(prediction),
                "feature_contract_sha256": base["feature_contract_sha256"],
                "preprocessing_sha256": base["preprocessing_sha256"],
                "checkpoint_epoch": checkpoint_epoch,
                "firewall_stage": "B",
                "eligibility_manifest_sha256": base["eligibility_manifest_sha256"],
                "session_mapping": base["session_mapping"],
                "oracle_field": field,
                "annotation_access_log_payload_sha256": (access_log_payload_sha256),
            }
        )
    return result


def _training_session_pairs(rows: Sequence[Any]) -> dict[str, tuple[str, str]]:
    grouped: dict[str, list[Any]] = {}
    for row in rows:
        grouped.setdefault(row.feature.identity.participant_id, []).append(row)
    result = {}
    for participant_id in sorted(grouped, key=int):
        pair = sorted(
            grouped[participant_id],
            key=lambda row: row.feature.identity.condition == "post",
        )
        if len(pair) != 2 or {row.feature.identity.condition for row in pair} != {
            "pre",
            "post",
        }:
            raise ValueError("Stage-B training participant lacks one canonical pair")
        result[participant_id] = tuple(row.feature.identity.session_id for row in pair)
    return cast(dict[str, tuple[str, str]], result)


def _release_training_session_pairs(
    release: Any, *, participant_ids: Sequence[str]
) -> dict[str, tuple[str, str]]:
    required = set(participant_ids)
    grouped: dict[str, list[tuple[str, str]]] = {value: [] for value in participant_ids}
    for participant_id, phase, session_id in release.session_keys:
        if participant_id in required:
            grouped[participant_id].append((phase, session_id))
    result = {}
    for participant_id in participant_ids:
        pair = sorted(grouped[participant_id], key=lambda row: row[0] == "post")
        if len(pair) != 2 or {phase for phase, _ in pair} != {"pre", "post"}:
            raise ValueError("Stage-B release lacks one canonical training pair")
        result[participant_id] = (pair[0][1], pair[1][1])
    return result


def _require_clean_oracle_directory(root: Path | AnchoredDirectory) -> None:
    names = (
        root.list_names()
        if isinstance(root, AnchoredDirectory)
        else tuple(path.name for path in root.iterdir())
    )
    stale = sorted(
        name for name in names if name.startswith("oracle_") or name == "stage-b-oracle-lock.json"
    )
    if stale:
        raise FileExistsError(
            "zero-fit Stage-B omission refuses existing oracle artifacts: " + ", ".join(stale)
        )


def _expected_fit_array_names() -> set[str]:
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
    return (
        _FIT_UTF8_FIELDS
        | _FIT_FLOAT_FIELDS
        | _FIT_INTEGER_FIELDS
        | scaler
        | model
        | optimizer
        | {"optimizer.param_groups_json_utf8"}
    )


def _signed(unsigned: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {**unsigned, field: _payload_sha256(unsigned)}


def _utf8_array(value: str) -> np.ndarray:
    return np.frombuffer(value.encode("utf-8"), dtype=np.uint8)


def _archive_utf8(arrays: Mapping[str, np.ndarray], name: str) -> str:
    value = np.asarray(arrays[name])
    if value.dtype != np.uint8 or value.ndim != 1:
        raise ValueError(f"Stage-B oracle fit string field {name} is invalid")
    try:
        return value.tobytes().decode("utf-8")
    except UnicodeError as error:
        raise ValueError(f"Stage-B oracle fit string field {name} is invalid") from error


def _canonical_value_array(value: str | int) -> np.ndarray:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return np.frombuffer(raw, dtype=np.uint8)


def _canonical_value(value: np.ndarray) -> Any:
    raw = np.asarray(value)
    if raw.dtype != np.uint8 or raw.ndim != 1:
        raise ValueError("Stage-B oracle category JSON is invalid")
    try:
        result = json.loads(raw.tobytes().decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Stage-B oracle category JSON is invalid") from error
    if _canonical_value_array(result).tobytes() != raw.tobytes():
        raise ValueError("Stage-B oracle category JSON is not canonical")
    return result


def _canonical_object(value: np.ndarray, label: str) -> dict[str, Any]:
    payload = _canonical_json_value(value, label)
    if not isinstance(payload, dict):
        raise ValueError(f"Stage-B oracle {label} JSON must be an object")
    return payload


def _canonical_json_value(value: np.ndarray, label: str) -> Any:
    raw = np.asarray(value)
    if raw.dtype != np.uint8 or raw.ndim != 1:
        raise ValueError(f"Stage-B oracle {label} JSON is invalid")
    try:
        payload = json.loads(raw.tobytes().decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Stage-B oracle {label} JSON is invalid") from error
    expected = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if expected != raw.tobytes():
        raise ValueError(f"Stage-B oracle {label} JSON is not canonical")
    return payload


def _canonical_rows(value: np.ndarray, label: str) -> list[Any]:
    payload = _canonical_object(value, label)
    if set(payload) != {"rows"} or not isinstance(payload["rows"], list):
        raise ValueError(f"Stage-B oracle {label} rows are invalid")
    return payload["rows"]
