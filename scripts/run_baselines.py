#!/usr/bin/env python3
"""Lock Stage-A eligibility, then run the implemented frozen MIT comparisons."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_mit_interview_dataset, load_mit_interview_roster  # noqa: E402
from src.features import load_verified_feature_extraction_closure  # noqa: E402
from src.features.extraction import code_provenance  # noqa: E402
from src.secure_io import AnchoredDirectory  # noqa: E402
from src.training import (  # noqa: E402
    DIAGNOSTIC_FLOOR_CONFIGURATIONS,
    FIREWALLED_LEARNED_CONFIGURATIONS,
    PROXY_CONFIGURATION_SPECS,
    RunProvenance,
    SessionIdentity,
    SessionMappingContext,
    StageARunLedger,
    build_complete_pair_eligibility,
    build_stage_a_execution_context,
    build_target_set,
    load_verified_session_mapping,
    run_adversarially_reweighted_learning,
    run_fixed_ridge,
    run_ordinary_erm,
    run_phase_mean,
    run_proxy_configuration,
    run_training_mean,
    session_identities,
    write_eligibility_lock,
    write_prediction_bundle,
)
from src.training.crossfit import canonical_json_bytes  # noqa: E402

FROZEN_SEEDS = (1, 11, 21)
REPORT_SCHEMA_VERSION = "fair-avi-mit-baseline-run-report-5"
DEFAULT_OUTPUT_DIR = Path("runs/stage-a/baselines")
OWNED_SOURCE_PATHS = (
    "configs/default.yaml",
    "pyproject.toml",
    "requirements.lock",
    "scripts/run_baselines.py",
    "scripts/run_stage_b_oracle.py",
    "src/__init__.py",
    "src/data/__init__.py",
    "src/data/annotation_audit.py",
    "src/data/annotation_stimuli.py",
    "src/data/mit_interview.py",
    "src/data/mit_session_mapping.py",
    "src/data/protected_annotations.py",
    "src/eval/stage_b_oracle.py",
    "src/eval/stage_c.py",
    "src/features/__init__.py",
    "src/features/closure.py",
    "src/features/extraction.py",
    "src/frozen_json.py",
    "src/models/__init__.py",
    "src/models/fusion.py",
    "src/secure_io.py",
    "src/training",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="restricted MIT corpus root; defaults to data.raw_dir in the configuration",
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        help="closed feature-extraction directory; defaults to data.feature_dir",
    )
    parser.add_argument(
        "--feature-closure-file-sha256",
        required=True,
        help="exact SHA-256 of feature-extraction-closure.json",
    )
    parser.add_argument(
        "--feature-closure-payload-sha256",
        required=True,
        help="payload SHA-256 inside feature-extraction-closure.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="ignored Stage-A lock directory (existing non-identical artefacts are never replaced)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="requested learned-comparison seeds; each must be frozen in the configuration",
    )
    parser.add_argument(
        "--source-date-epoch",
        type=int,
        help="non-negative reproducibility timestamp; defaults to SOURCE_DATE_EPOCH",
    )
    parser.add_argument(
        "--session-mapping",
        type=Path,
        default=Path("src/data/mit_source_session_map.json"),
        help="approved canonical-to-source session mapping",
    )
    parser.add_argument(
        "--identity-audit",
        type=Path,
        default=Path("data/restricted/mit_identity_join_audit.json"),
        help="restricted identity-join audit bound by the approved mapping",
    )
    parser.add_argument(
        "--face-crops",
        type=Path,
        default=Path("data/restricted/face_crops.json"),
        help="restricted face-crop manifest bound by the approved mapping",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="assert that Stage A may consume only pre-existing local artefacts",
    )
    return parser.parse_args(argv)


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("configuration must be a regular non-symlink file")
    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    return config, hashlib.sha256(raw).hexdigest()


def _resolved_paths(args: argparse.Namespace, config: dict[str, Any]) -> tuple[Path, Path, Path]:
    data = config.get("data")
    if not isinstance(data, dict):
        raise ValueError("configuration data section must be a mapping")
    raw_dir = data.get("raw_dir")
    feature_dir = data.get("feature_dir")
    if args.dataset_root is None and not isinstance(raw_dir, str):
        raise ValueError("configuration data.raw_dir must be a path string")
    if args.feature_dir is None and not isinstance(feature_dir, str):
        raise ValueError("configuration data.feature_dir must be a path string")
    return (
        args.dataset_root if args.dataset_root is not None else Path(raw_dir),
        args.feature_dir if args.feature_dir is not None else Path(feature_dir),
        args.output_dir,
    )


def _requested_seeds(config: dict[str, Any], supplied: list[int] | None) -> tuple[int, ...]:
    configured = config.get("seeds")
    if not isinstance(configured, list) or tuple(configured) != FROZEN_SEEDS:
        raise ValueError("configuration seeds must be exactly [1, 11, 21]")
    if supplied is None:
        return FROZEN_SEEDS
    if not supplied or len(supplied) != len(set(supplied)):
        raise ValueError("requested seeds must be non-empty and unique")
    unknown = sorted(set(supplied) - set(FROZEN_SEEDS))
    if unknown:
        raise ValueError(f"requested seeds are not frozen: {unknown}")
    return tuple(seed for seed in FROZEN_SEEDS if seed in supplied)


def _validate_arl_contract(config: dict[str, Any]) -> None:
    """Fail closed if production configuration drifts from the frozen ARL adaptation."""
    required: tuple[tuple[tuple[str, ...], Any], ...] = (
        (("common_scorer", "trainable_parameter_count"), 26_721),
        (("common_scorer", "preprocessing"), "train_only_standard_scaler_per_modality"),
        (("common_scorer", "fusion"), "concatenate_24_dimensions"),
        (
            ("common_scorer", "initialization"),
            {
                "linear_weight": "xavier_uniform",
                "linear_bias": "zero",
                "layer_norm_weight": "one",
                "layer_norm_bias": "zero",
            },
        ),
        (("training", "participant_batch_size"), 16),
        (("training", "session_rows_per_full_batch"), 32),
        (("training", "both_sessions_enter_batch_together"), True),
        (("training", "optimizer"), "adamw"),
        (("training", "learning_rate"), 1e-3),
        (("training", "weight_decay"), 1e-2),
        (("training", "maximum_epochs"), 200),
        (("training", "early_stopping_patience"), 25),
        (("training", "checkpoint_metric"), "validation_srcc"),
        (("training", "checkpoint_direction"), "maximize"),
        (("training", "exact_tie_break"), "earliest_epoch"),
        (("training", "gradient_norm_clip"), 1.0),
        (("arl", "method_status"), "transparent_project_adaptation_not_exact_reproduction"),
        (("arl", "learner", "fused_representation_dimensions"), 24),
        (("arl", "learner", "common_scorer_trainable_parameters"), 26_721),
        (
            ("arl", "adversary", "input"),
            [
                "detached_fused_representation_24",
                "detached_normalised_continuous_target_1",
            ],
        ),
        (("arl", "adversary", "architecture"), "linear_25_to_1_then_sigmoid"),
        (
            ("arl", "adversary", "initialization"),
            {"linear_weight": "xavier_uniform", "linear_bias": "zero"},
        ),
        (("arl", "adversary", "trainable_parameters"), 26),
        (
            ("arl", "adversary", "forbidden_inputs"),
            ["prediction", "per_example_loss", "proxy_group", "protected_annotation"],
        ),
        (("arl", "weighting", "scope"), "current_session_row_minibatch"),
        (
            ("arl", "weighting", "formula"),
            "1_plus_sigmoid_logit_divided_by_max_batch_mean_sigmoid_logit_and_1e_minus_4",
        ),
        (("arl", "weighting", "denominator_guard"), 1e-4),
        (("arl", "weighting", "learner_weight_detached"), True),
        (
            ("arl", "weighting", "reduction"),
            "mean_weight_times_per_row_mse_not_weight_sum",
        ),
        (("arl", "optimization", "learner_only_minibatch_updates"), 6),
        (
            ("arl", "optimization", "subsequent_order"),
            "learner_then_recompute_detached_current_state_then_adversary",
        ),
        (
            ("arl", "optimization", "sampler"),
            "ordinary_participant_paired_with_complete_remainder",
        ),
        (
            ("arl", "optimization", "learner_optimizer"),
            "separate_adamw_with_common_project_settings",
        ),
        (
            ("arl", "optimization", "adversary_optimizer"),
            "separate_adamw_with_common_project_settings",
        ),
        (
            ("arl", "optimization", "gradient_norm_clip_separate"),
            {"learner": 1.0, "adversary": 1.0},
        ),
        (("arl", "checkpoint", "metric"), "unweighted_validation_srcc"),
        (("arl", "checkpoint", "exact_tie_break"), "earliest_epoch"),
        (
            ("arl", "checkpoint", "save"),
            [
                "learner",
                "adversary",
                "learner_optimizer",
                "adversary_optimizer",
                "scalers",
                "phase",
                "diagnostics",
            ],
        ),
        (
            ("arl", "diagnostics", "minibatch"),
            [
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
        ),
        (("arl", "diagnostics", "timing_version"), "arl_diagnostics_v1"),
        (
            ("arl", "diagnostics", "timing"),
            "weighting_and_learner_metrics_pre_learner_step_"
            "adversary_objective_and_gradient_post_learner_step_pre_adversary_step",
        ),
        (
            ("arl", "diagnostics", "epoch"),
            ["aggregated_minibatch_diagnostics", "validation_srcc", "validation_mse"],
        ),
        (("arl", "diagnostics", "undefined_value"), None),
        (
            ("arl", "diagnostics", "serialization"),
            "canonical_json_utf8_in_fit_artifact_and_prediction_manifest",
        ),
        (("arl", "protected_annotation_access"), "forbidden"),
        (("arl", "proxy_access"), "forbidden"),
    )
    for path, expected in required:
        value: Any = config
        for key in path:
            if not isinstance(value, dict) or key not in value:
                raise ValueError(f"configuration is missing frozen ARL field {'.'.join(path)}")
            value = value[key]
        if value != expected:
            raise ValueError(f"configuration frozen ARL field drifted: {'.'.join(path)}")


def _validate_proxy_contract(config: dict[str, Any]) -> None:
    """Fail closed if proxy-balanced training drifts from ADR-041 and INV-036."""
    required: tuple[tuple[tuple[str, ...], Any], ...] = (
        (
            ("configurations", "firewalled"),
            list(FIREWALLED_LEARNED_CONFIGURATIONS),
        ),
        (
            ("configurations", "deterministic_floors"),
            list(DIAGNOSTIC_FLOOR_CONFIGURATIONS),
        ),
        (("run_budget", "firewalled", "configurations"), 7),
        (("run_budget", "firewalled", "folds"), 5),
        (("run_budget", "firewalled", "seeds"), 3),
        (("run_budget", "firewalled", "fits"), 105),
        (
            ("run_budget", "diagnostic_floors_outside_learned_matrix"),
            len(DIAGNOSTIC_FLOOR_CONFIGURATIONS),
        ),
        (("training", "participant_batch_size"), 16),
        (("training", "both_sessions_enter_batch_together"), True),
        (("training", "optimizer"), "adamw"),
        (("training", "learning_rate"), 1e-3),
        (("training", "weight_decay"), 1e-2),
        (("training", "maximum_epochs"), 200),
        (("training", "early_stopping_patience"), 25),
        (("training", "checkpoint_metric"), "validation_srcc"),
        (("training", "checkpoint_direction"), "maximize"),
        (("training", "exact_tie_break"), "earliest_epoch"),
        (("training", "gradient_norm_clip"), 1.0),
        (("pseudo_group", "fit_scope"), "training_participants_only_per_outer_fold"),
        (
            ("pseudo_group", "sources"),
            {
                "face": "mean_session_dinov2_summary_per_participant",
                "audio_session": "mean_session_wavlm_summary_per_participant",
            },
        ),
        (("pseudo_group", "preprocessing"), "train_only_standard_scaler"),
        (
            ("pseudo_group", "kmeans"),
            {
                "n_clusters": 2,
                "init": "k-means++",
                "n_init": 20,
                "max_iter": 300,
                "tol": 1e-4,
                "algorithm": "lloyd",
                "random_state": 0,
            },
        ),
        (("pseudo_group", "held_out_assignment"), "unchanged_scaler_and_nearest_frozen_centre"),
        (("pseudo_group", "both_sessions_share_assignment"), True),
        (("pseudo_group", "cluster_identity_claim"), "forbidden"),
        (("pseudo_group", "random_proxy", "salt"), "fair-avi-mit-random-proxy-v1"),
        (
            ("pseudo_group", "random_proxy", "assignment"),
            "global_hash_rank_first_34_group_0_remaining_35_group_1",
        ),
        (("pseudo_group", "random_proxy", "fold_role_or_seed_dependency"), "forbidden"),
        (("pseudo_group", "balanced_sampler", "participant_batch_size"), 16),
        (("pseudo_group", "balanced_sampler", "participants_per_group"), 8),
        (("pseudo_group", "balanced_sampler", "session_rows_per_batch"), 32),
        (("pseudo_group", "balanced_sampler", "session_rows_per_group"), 16),
        (
            ("pseudo_group", "balanced_sampler", "policy"),
            "deterministic_seeded_cyclic_shuffle",
        ),
        (
            ("pseudo_group", "balanced_sampler", "state"),
            "single_continuous_state_across_epoch_boundaries",
        ),
        (("pseudo_group", "balanced_sampler", "require_both_groups_nonempty"), True),
        (
            ("pseudo_group", "balanced_sampler", "no_repeat_before_group_exhaustion"),
            True,
        ),
        (
            (
                "pseudo_group",
                "balanced_sampler",
                "duplicate_within_half_batch_only_if_group_size_below_8",
            ),
            True,
        ),
        (
            ("pseudo_group", "balanced_sampler", "optimizer_steps_per_epoch"),
            "ceil_training_participant_count_divided_by_16",
        ),
        (
            ("pseudo_group", "balanced_sampler", "checkpoint_resume"),
            "restore_model_optimizer_scalers_regenerate_schedule_and_advance_saved_draw_count",
        ),
        (
            ("pseudo_group", "balanced_sampler", "validation_sampling"),
            "complete_natural_role_distribution",
        ),
        (
            ("pseudo_group", "balanced_sampler", "test_sampling"),
            "complete_natural_role_distribution",
        ),
        (
            ("pseudo_group", "balanced_sampler", "report"),
            ["unique_participant_rate", "duplication_rate"],
        ),
        (("loss", "task"), "mse_on_normalised_recommend_hiring"),
        (("loss", "lambda_w"), 0.02),
        (("loss", "lambda_sweep"), False),
        (("loss", "schedule"), "fixed_from_first_update"),
        (("loss", "w1_groups"), "binary_training_proxy_only"),
        (
            ("loss", "sampling_only"),
            "unweighted_task_mse_without_proxy_group_objective_access",
        ),
        (
            ("loss", "w1_objective"),
            "task_mse_plus_lambda_w_times_exact_balanced_empirical_w1",
        ),
        (("loss", "adversarial_component_for_proxy_configurations"), "forbidden"),
    )
    for path, expected in required:
        value: Any = config
        for key in path:
            if not isinstance(value, dict) or key not in value:
                raise ValueError(f"configuration is missing frozen proxy field {'.'.join(path)}")
            value = value[key]
        if value != expected:
            raise ValueError(f"configuration frozen proxy field drifted: {'.'.join(path)}")
    if tuple(PROXY_CONFIGURATION_SPECS) != FIREWALLED_LEARNED_CONFIGURATIONS[1:-1]:
        raise RuntimeError("executable proxy registry does not match the frozen Stage-A registry")


def _source_date_epoch(supplied: int | None) -> int:
    if supplied is None:
        raw = os.environ.get("SOURCE_DATE_EPOCH")
        if raw is None:
            raise ValueError("SOURCE_DATE_EPOCH (or --source-date-epoch) is required")
        try:
            value = int(raw)
        except ValueError as error:
            raise ValueError("SOURCE_DATE_EPOCH must be an integer") from error
    else:
        value = supplied
    if value < 0:
        raise ValueError("SOURCE_DATE_EPOCH must be non-negative")
    return value


def _outcome_blind_identities(
    dataset_root: Path, session_mapping: SessionMappingContext
) -> tuple[SessionIdentity, ...]:
    """Read only the validated Videos/Audio roster for the pre-outcome lock stage."""
    session_mapping.verify_current_files()
    return tuple(
        SessionIdentity(
            session_id=row.session_id,
            participant_id=row.participant_id,
            condition=row.condition,
            fold=row.fold,
            source_session_id=row.source_session_id,
        )
        for row in load_mit_interview_roster(dataset_root, session_mapping=session_mapping)
    )


def _run_diagnostic_floor(
    configuration: str,
    eligibility: Any,
    targets: Any,
    provenance: RunProvenance,
) -> Any:
    if configuration == "mean":
        return run_training_mean(eligibility, targets, provenance=provenance)
    if configuration == "phase_mean":
        return run_phase_mean(eligibility, targets, provenance=provenance)
    return run_fixed_ridge(
        eligibility,
        targets,
        configuration=configuration,
        provenance=provenance,
    )


def _run_learned_bundle(
    configuration: str,
    seed: int,
    eligibility: Any,
    targets: Any,
    provenance: RunProvenance,
) -> Any:
    if configuration == "erm":
        return run_ordinary_erm(eligibility, targets, seed=seed, provenance=provenance)
    if configuration == "arl":
        return run_adversarially_reweighted_learning(
            eligibility, targets, seed=seed, provenance=provenance
        )
    return run_proxy_configuration(
        eligibility,
        targets,
        configuration=configuration,
        seed=seed,
        provenance=provenance,
    )


def run_research_workflow(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the production-only Stage-A baseline workflow in firewall order."""
    config, config_sha256 = _load_config(args.config)
    _validate_arl_contract(config)
    _validate_proxy_contract(config)
    if args.local_files_only is not True:
        raise ValueError("production baselines require --local-files-only")
    dataset_root, feature_dir, output_dir = _resolved_paths(args, config)
    seeds = _requested_seeds(config, args.seeds)
    source_date_epoch = _source_date_epoch(args.source_date_epoch)
    code = code_provenance(ROOT, OWNED_SOURCE_PATHS)
    provenance = RunProvenance(
        config_sha256=config_sha256,
        code_commit_sha=code.commit_sha,
        code_source_tree_sha256=code.source_tree_sha256,
        source_date_epoch=source_date_epoch,
    )
    session_mapping = load_verified_session_mapping(
        args.session_mapping,
        identity_join_audit_report_path=args.identity_audit,
        face_crop_manifest_path=args.face_crops,
    )

    # This stage reads only media identities and frozen feature artefacts. The ratings table and
    # full dataset loader remain untouched until the eligibility/fold files exist durably.
    outcome_blind_roster = _outcome_blind_identities(dataset_root, session_mapping)
    with load_verified_feature_extraction_closure(
        feature_dir,
        closure_file_sha256=args.feature_closure_file_sha256,
        closure_payload_sha256=args.feature_closure_payload_sha256,
        session_mapping=session_mapping,
    ) as feature_closure:
        eligibility = build_complete_pair_eligibility(
            outcome_blind_roster,
            None,
            feature_closure=feature_closure,
            session_mapping=session_mapping,
            require_full_corpus=True,
        )
        feature_closure.verify_current_files()
    stage_a_output = AnchoredDirectory.open(output_dir, create=True, directory_mode=0o700)
    eligibility_lock = write_eligibility_lock(stage_a_output, eligibility)

    # Only the successful lock above opens the outcome boundary.
    session_mapping.verify_current_files()
    dataset = load_mit_interview_dataset(dataset_root, session_mapping=session_mapping)
    if session_identities(dataset) != outcome_blind_roster:
        raise RuntimeError("post-lock dataset identities disagree with the locked media roster")
    targets = build_target_set(
        dataset,
        dataset_root,
        eligibility=eligibility,
        eligibility_lock=eligibility_lock,
        session_mapping=session_mapping,
    )

    session_mapping.verify_current_files()
    stage_a_context = build_stage_a_execution_context(eligibility, targets, provenance)
    ledger = StageARunLedger(
        stage_a_output,
        context=stage_a_context,
        learned_configurations=FIREWALLED_LEARNED_CONFIGURATIONS,
        diagnostic_floor_configurations=DIAGNOSTIC_FLOOR_CONFIGURATIONS,
        requested_seeds=seeds,
    )
    bundle_records: list[dict[str, Any]] = []
    newly_completed_learned_cells = 0
    newly_completed_diagnostic_floors = 0
    for configuration in DIAGNOSTIC_FLOOR_CONFIGURATIONS:
        if ledger.has_completed_bundle(configuration, None):
            record = ledger.bundle_record(configuration, None)
            if record is not None:
                bundle_records.append({**record, "disposition": "resumed"})
            continue
        try:
            bundle = _run_diagnostic_floor(configuration, eligibility, targets, provenance)
            write_prediction_bundle(stage_a_output, bundle)
            record = ledger.record_written_bundle(bundle)
        except Exception as error:
            ledger.record_failure(configuration, None, error)
        else:
            newly_completed_diagnostic_floors += 1
            bundle_records.append({**record, "disposition": "written"})
    for seed in seeds:
        for configuration in FIREWALLED_LEARNED_CONFIGURATIONS:
            if ledger.has_completed_bundle(configuration, seed):
                record = ledger.bundle_record(configuration, seed)
                if record is not None:
                    bundle_records.append({**record, "disposition": "resumed"})
                continue
            try:
                bundle = _run_learned_bundle(configuration, seed, eligibility, targets, provenance)
                write_prediction_bundle(stage_a_output, bundle)
                record = ledger.record_written_bundle(bundle)
            except Exception as error:
                ledger.record_failure(configuration, seed, error)
            else:
                newly_completed_learned_cells += 5
                bundle_records.append({**record, "disposition": "written"})

    closure = ledger.finalize_if_complete()
    ledger_report = ledger.report()
    ledger_file = stage_a_output.read_bytes("stage-a-ledger.json", expected_mode=0o600)
    closure_record = (
        {
            "filename": "stage-a-closure.json",
            "file_sha256": stage_a_output.read_bytes(
                "stage-a-closure.json", expected_mode=0o600
            ).sha256,
            "payload_sha256": closure["closure_payload_sha256"],
        }
        if closure is not None
        else None
    )

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": ledger_report["status"],
        "execution_mode": "research",
        "firewall_stage": "A",
        "protected_annotation_fields_accessed": [],
        "config_sha256": config_sha256,
        "code_commit_sha": code.commit_sha,
        "code_source_tree_sha256": code.source_tree_sha256,
        "source_date_epoch": source_date_epoch,
        "local_files_only": True,
        "session_mapping": session_mapping.provenance,
        "feature_extraction_closure": {
            "file_sha256": eligibility.manifest["feature_extraction_closure_file_sha256"],
            "payload_sha256": eligibility.manifest["feature_extraction_closure_payload_sha256"],
        },
        "eligibility": {
            "manifest_sha256": eligibility.manifest_sha256,
            "participant_count": eligibility.manifest["eligible_participant_count"],
            "session_count": eligibility.manifest["eligible_session_count"],
            "participants_per_fold": eligibility.manifest["eligible_participants_per_fold"],
            "exclusion_count": len(eligibility.manifest["exclusions"]),
        },
        "target_manifest_sha256": targets.manifest_sha256,
        "requested_learned_seeds": list(seeds),
        "frozen_firewalled_configurations": list(FIREWALLED_LEARNED_CONFIGURATIONS),
        "full_frozen_learned_fit_count": 105,
        "completed_learned_fit_count": ledger_report["completed_learned_cell_count"],
        "failed_learned_fit_count": ledger_report["failed_learned_cell_count"],
        "pending_learned_fit_count": ledger_report["pending_learned_cell_count"],
        "newly_completed_learned_fit_count": newly_completed_learned_cells,
        "diagnostic_floor_configurations": list(DIAGNOSTIC_FLOOR_CONFIGURATIONS),
        "newly_completed_diagnostic_floor_count": newly_completed_diagnostic_floors,
        "failed_diagnostic_floor_count": ledger_report["diagnostic_floor_failed_count"],
        "stage_a_ledger": {
            "filename": "stage-a-ledger.json",
            "file_sha256": ledger_file.sha256,
            "payload_sha256": ledger_report["ledger_payload_sha256"],
            "recovery_contract": ledger_report["recovery_contract"],
        },
        "stage_a_closure": closure_record,
        "bundles": bundle_records,
    }
    stage_a_output.verify_path_identity()
    stage_a_output.close()
    return report


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = run_research_workflow(args)
    except Exception as error:
        print(f"baseline run failed: {error}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(report).decode("utf-8"), end="")
    if report["failed_learned_fit_count"] or report["failed_diagnostic_floor_count"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
