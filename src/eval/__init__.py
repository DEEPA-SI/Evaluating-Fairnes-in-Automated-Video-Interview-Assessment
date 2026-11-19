"""Evaluation modules with lazy loading of optional source-dependent workflows."""

from importlib import import_module

__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "MINIMUM_REPORTABLE_PARTICIPANTS",
    "ORACLE_ACCESS_CHECKSUM_FIELD",
    "ORACLE_EXPECTED_BUNDLE_COUNT",
    "ORACLE_EXPECTED_FIT_COUNT",
    "ORACLE_LOCK_CHECKSUM_FIELD",
    "ORACLE_PREDICTION_CHECKSUM_FIELD",
    "ORACLE_PREDICTION_LOCK_CHECKSUM_FIELD",
    "ORACLE_PREDICTION_ROW_FIELDS",
    "OracleAnnotationBinding",
    "OracleTrainingAnnotationRelease",
    "STAGE_B_ORACLE_ACCESS_LOG_SCHEMA_VERSION",
    "STAGE_B_ORACLE_LOCK_FILENAME",
    "STAGE_B_ORACLE_LOCK_SCHEMA_VERSION",
    "STAGE_B_ORACLE_PREDICTION_LOCK_SCHEMA_VERSION",
    "STAGE_B_ORACLE_PREDICTION_SCHEMA_VERSION",
    "AnnotationRelease",
    "StageARelease",
    "StageBOracleDisposition",
    "StageCEvaluationResult",
    "StageCSourcePaths",
    "build_oracle_access_log",
    "build_claim_ledger",
    "build_no_rerun_manifest",
    "build_stage_b_oracle_lock",
    "build_stage_b_oracle_omission",
    "evaluate_stage_c",
    "evaluate_stage_c_from_paths",
    "load_annotation_release",
    "load_stage_a_release",
    "maximum_pairwise_disparity",
    "oracle_annotation_bindings",
    "oracle_eligibility_payload_sha256",
    "oracle_prediction_lock_id",
    "oracle_training_release_rows",
    "prepare_oracle_annotation_binding",
    "prepare_oracle_training_releases",
    "paired_change_metrics",
    "utility_metrics",
    "validate_exact_binary_oracle_eligibility",
    "validate_stage_b_oracle_lock",
    "validate_stage_b_oracle_lock_from_binding",
    "write_stage_b_oracle_lock",
    "write_stage_c_outputs",
]


def __getattr__(name: str):
    """Load optional neural/source workflows only when their public names are requested."""
    if name not in __all__:
        raise AttributeError(name)
    for module_name in ("stage_b_oracle", "stage_c"):
        module = import_module(f".{module_name}", __name__)
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(name)
