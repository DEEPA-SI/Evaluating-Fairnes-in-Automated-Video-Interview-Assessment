"""Closure-bound, annotation-free secondary utility reporting for Stage A."""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.secure_io import AnchoredDirectory
from src.training import (
    DIAGNOSTIC_FLOOR_CONFIGURATIONS,
    FIREWALLED_LEARNED_CONFIGURATIONS,
    FROZEN_FINAL_SEEDS,
)
from src.training.crossfit import canonical_json_bytes

from .stage_c import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONFIDENCE_LEVEL,
    CONTRASTS,
    RAW_SCALE_WIDTH,
    StageARelease,
    load_stage_a_release,
    paired_change_metrics,
    utility_metrics,
)

JSON_FILENAME = "stage_a_secondary_utility.json"
MARKDOWN_FILENAME = "STAGE_A_SECONDARY_UTILITY.md"
REPORT_SCHEMA_VERSION = "fair-avi-mit-stage-a-secondary-utility-1"

_UTILITY_METRICS = ("srcc", "pearson", "mae", "rmse")
_CHANGE_METRICS = ("delta_srcc", "delta_mae", "signed_delta_bias")
_ERROR_METRICS = {"mae", "rmse", "delta_mae", "signed_delta_bias"}
_PROHIBITED_AGGREGATE_KEYS = {
    "participant_id",
    "session_id",
    "source_session_id",
    "prediction",
    "predictions",
    "target",
    "prediction_lock_id",
    "rater_id",
}
_SCIENTIFIC_TIMING = {
    "specification_status": "predeclared_secondary_descriptive",
    "execution_timing": "post_stage_a_prediction_lock",
    "evidentiary_role": "not_confirmatory_primary_evidence",
}
_PUBLICATION_ISSUER = object()


@dataclass(frozen=True, slots=True)
class _PublicationAuthorization:
    """In-process evidence that the research builder issued these exact report bytes."""

    issuer: object
    source_root: Path
    closure_file_sha256: str
    closure_payload_sha256: str
    participant_count: int
    session_count: int
    bundle_count: int
    json_sha256: str
    markdown_sha256: str


@dataclass(frozen=True, slots=True)
class StageASecondaryReport:
    """Deterministic aggregate evidence derived from one closed Stage-A release."""

    source_root: Path
    closure_file_sha256: str
    closure_payload_sha256: str
    participant_count: int
    session_count: int
    payload: Mapping[str, Any]
    json_bytes: bytes
    markdown_bytes: bytes
    _publication_authorization: _PublicationAuthorization | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True, slots=True)
class PublishedStageASecondaryReport:
    """Checksums observed after safe publication of both aggregate files."""

    json_sha256: str
    markdown_sha256: str
    bundle_count: int


def build_stage_a_secondary_report(
    stage_a_dir: str | Path,
    *,
    expected_closure_file_sha256: str,
    expected_closure_payload_sha256: str,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    synthetic_contract_test: bool = False,
) -> StageASecondaryReport:
    """Load the checksum-bound release and recompute all secondary utility evidence."""
    _require_sha256(expected_closure_file_sha256, "expected closure file SHA-256")
    _require_sha256(expected_closure_payload_sha256, "expected closure payload SHA-256")
    _validate_bootstrap_contract(bootstrap_replicates, synthetic_contract_test)
    release = load_stage_a_release(stage_a_dir)
    if (
        release.closure_file_sha256 != expected_closure_file_sha256
        or release.closure_payload_sha256 != expected_closure_payload_sha256
    ):
        raise ValueError("Stage-A closure does not match the externally recorded checksums")
    return _build_validated_report(
        release,
        bootstrap_replicates=bootstrap_replicates,
        synthetic_contract_test=synthetic_contract_test,
    )


def publish_stage_a_secondary_report(
    report: StageASecondaryReport, output_dir: str | Path
) -> PublishedStageASecondaryReport:
    """Publish distinct secondary files safely without replacing different bytes."""
    bundle_count = _validate_publishable_report(report)
    output = _validated_separate_output(report.source_root, output_dir)
    contents = {
        JSON_FILENAME: report.json_bytes,
        MARKDOWN_FILENAME: report.markdown_bytes,
    }
    with AnchoredDirectory.open(output, create=True, directory_mode=0o755) as anchor:
        with anchor.exclusive_lock():
            names = set(anchor.list_names())
            for filename, content in contents.items():
                if filename not in names:
                    continue
                existing = anchor.read_bytes(filename, expected_mode=0o644)
                if existing.content != content:
                    raise FileExistsError(
                        f"refusing to replace non-identical secondary report {filename!r}"
                    )
            written = {
                filename: anchor.publish_bytes(filename, content, mode=0o644)
                for filename, content in contents.items()
            }
            for filename, observed in written.items():
                anchor.verify_file(filename, observed, expected_mode=0o644)
    return PublishedStageASecondaryReport(
        json_sha256=written[JSON_FILENAME].sha256,
        markdown_sha256=written[MARKDOWN_FILENAME].sha256,
        bundle_count=bundle_count,
    )


def publication_summary(
    report: StageASecondaryReport, published: PublishedStageASecondaryReport
) -> bytes:
    """Return one canonical aggregate-only CLI receipt."""
    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "published",
        "execution_mode": report.payload["execution_mode"],
        "stage_a_closure": {
            "file_sha256": report.closure_file_sha256,
            "payload_sha256": report.closure_payload_sha256,
        },
        "participant_count": report.participant_count,
        "session_count": report.session_count,
        "bundle_count": published.bundle_count,
        "bootstrap_replicates": report.payload["participant_cluster_bootstrap"]["resamples"],
        "json": {"filename": JSON_FILENAME, "sha256": published.json_sha256},
        "markdown": {
            "filename": MARKDOWN_FILENAME,
            "sha256": published.markdown_sha256,
        },
        "protected_annotation_fields_accessed": [],
    }
    return canonical_json_bytes(payload) + b"\n"


def report_content_sha256(report: StageASecondaryReport) -> Mapping[str, str]:
    """Expose deterministic content checksums for independent verification."""
    return {
        JSON_FILENAME: hashlib.sha256(report.json_bytes).hexdigest(),
        MARKDOWN_FILENAME: hashlib.sha256(report.markdown_bytes).hexdigest(),
    }


def _validate_publishable_report(report: StageASecondaryReport) -> int:
    """Fail closed unless the research builder issued these exact aggregate bytes."""
    if not isinstance(report, StageASecondaryReport):
        raise TypeError("secondary publication requires a StageASecondaryReport")
    authorization = report._publication_authorization
    if authorization is None or authorization.issuer is not _PUBLICATION_ISSUER:
        raise ValueError("only builder-issued research secondary reports may be published")

    payload = _canonical_publication_payload(report)
    _assert_aggregate_only(payload)
    if (
        payload.get("schema_version") != REPORT_SCHEMA_VERSION
        or payload.get("scope")
        != "aggregate_only_no_participant_session_prediction_or_annotation_rows"
        or payload.get("execution_mode") != "research"
        or payload.get("protected_annotation_fields_accessed") != []
    ):
        raise ValueError("secondary publication scope or research execution contract drifted")
    if payload.get("scientific_timing") != _SCIENTIFIC_TIMING:
        raise ValueError("secondary publication scientific-timing contract drifted")

    bootstrap = payload.get("participant_cluster_bootstrap")
    expected_bootstrap = {
        "method": "participant_cluster_percentile",
        "resamples": BOOTSTRAP_REPLICATES,
        "random_state": BOOTSTRAP_SEED,
        "confidence_level": CONFIDENCE_LEVEL,
        "percentile_quantile_method": "linear",
        "cluster_unit": "participant_with_both_sessions_and_all_seed_model_predictions",
        "sessions_retained_per_sampled_participant": 2,
        "interval_scope": "pointwise_descriptive_not_simultaneous",
        "seeds_are_sensitivity_runs_not_resampling_units": True,
        "research_evidence": True,
    }
    if bootstrap != expected_bootstrap:
        raise ValueError("secondary publication requires the frozen research bootstrap contract")
    _validate_bootstrap_contract(bootstrap["resamples"], synthetic_contract_test=False)

    stage_a = _validated_publication_stage_binding(payload, report)
    bundle_count = _validated_publication_bundle_count(payload, report)
    if stage_a["learned_bundle_count"] + stage_a["diagnostic_floor_bundle_count"] != bundle_count:
        raise ValueError("secondary publication bundle counts disagree")
    if payload.get("evaluator_provenance") != _evaluator_provenance():
        raise ValueError("secondary publication evaluator provenance drifted")

    aggregate_payload_sha256 = _require_sha256(
        payload.get("aggregate_payload_sha256"), "aggregate payload SHA-256"
    )
    unsigned = dict(payload)
    del unsigned["aggregate_payload_sha256"]
    if hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != aggregate_payload_sha256:
        raise ValueError("secondary publication aggregate payload checksum failed")

    if not isinstance(report.markdown_bytes, bytes):
        raise TypeError("secondary publication Markdown content must be bytes")
    if report.markdown_bytes != _markdown_bytes(payload):
        raise ValueError("secondary publication Markdown does not match its signed JSON payload")

    observed_json_sha256 = hashlib.sha256(report.json_bytes).hexdigest()
    observed_markdown_sha256 = hashlib.sha256(report.markdown_bytes).hexdigest()
    if (
        authorization.source_root != Path(report.source_root).resolve(strict=False)
        or authorization.closure_file_sha256 != report.closure_file_sha256
        or authorization.closure_payload_sha256 != report.closure_payload_sha256
        or authorization.participant_count != report.participant_count
        or authorization.session_count != report.session_count
        or authorization.bundle_count != bundle_count
        or authorization.json_sha256 != observed_json_sha256
        or authorization.markdown_sha256 != observed_markdown_sha256
    ):
        raise ValueError("secondary publication no longer matches its builder authorization")
    _revalidate_publication_source(report, payload)
    return bundle_count


def _canonical_publication_payload(report: StageASecondaryReport) -> dict[str, Any]:
    if not isinstance(report.json_bytes, bytes):
        raise TypeError("secondary publication JSON content must be bytes")
    try:
        decoded = json.loads(report.json_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("secondary publication JSON is invalid") from error
    if not isinstance(decoded, dict):
        raise ValueError("secondary publication JSON root must be an object")
    if canonical_json_bytes(decoded) + b"\n" != report.json_bytes:
        raise ValueError("secondary publication JSON is not byte-canonical")
    if not isinstance(report.payload, Mapping):
        raise TypeError("secondary publication payload must be a mapping")
    try:
        payload_bytes = canonical_json_bytes(_plain_json(report.payload)) + b"\n"
    except (TypeError, ValueError) as error:
        raise ValueError("secondary publication payload is not canonical JSON data") from error
    if payload_bytes != report.json_bytes:
        raise ValueError("secondary publication payload and JSON bytes disagree")
    return decoded


def _validated_publication_stage_binding(
    payload: Mapping[str, Any], report: StageASecondaryReport
) -> Mapping[str, Any]:
    stage_a = payload.get("stage_a")
    if not isinstance(stage_a, Mapping):
        raise ValueError("secondary publication lacks its Stage-A binding")
    closure_file_sha256 = _require_sha256(
        report.closure_file_sha256, "report Stage-A closure file SHA-256"
    )
    closure_payload_sha256 = _require_sha256(
        report.closure_payload_sha256, "report Stage-A closure payload SHA-256"
    )
    if (
        stage_a.get("closure_file_sha256") != closure_file_sha256
        or stage_a.get("closure_payload_sha256") != closure_payload_sha256
    ):
        raise ValueError("secondary publication Stage-A closure binding drifted")
    context_payload_sha256 = _require_sha256(
        stage_a.get("context_payload_sha256"), "Stage-A context payload SHA-256"
    )
    provenance = stage_a.get("provenance")
    if not isinstance(provenance, Mapping) or (
        hashlib.sha256(canonical_json_bytes(_plain_json(provenance))).hexdigest()
        != context_payload_sha256
    ):
        raise ValueError("secondary publication Stage-A context checksum failed")

    participant_count = _positive_int(report.participant_count, "report participant count")
    session_count = _positive_int(report.session_count, "report session count")
    if session_count != 2 * participant_count:
        raise ValueError("secondary publication requires exactly two sessions per participant")
    expected_learned = len(FIREWALLED_LEARNED_CONFIGURATIONS) * len(FROZEN_FINAL_SEEDS)
    expected_floors = len(DIAGNOSTIC_FLOOR_CONFIGURATIONS)
    if (
        stage_a.get("eligible_participant_count") != participant_count
        or stage_a.get("eligible_session_count") != session_count
        or stage_a.get("learned_bundle_count") != expected_learned
        or stage_a.get("diagnostic_floor_bundle_count") != expected_floors
    ):
        raise ValueError("secondary publication Stage-A counts drifted")
    return stage_a


def _validated_publication_bundle_count(
    payload: Mapping[str, Any], report: StageASecondaryReport
) -> int:
    expected_keys = tuple(_bundle_keys())
    pooled = _publication_rows(payload.get("pooled_utility"), "pooled utility")
    phase = _publication_rows(payload.get("phase_stratified_utility"), "phase utility")
    change = _publication_rows(payload.get("paired_change"), "paired change")
    if tuple((row.get("configuration"), row.get("seed")) for row in pooled) != expected_keys:
        raise ValueError("secondary publication pooled bundle registry drifted")
    expected_phase = tuple(
        (*key, phase_name) for key in expected_keys for phase_name in ("pre", "post")
    )
    if (
        tuple((row.get("configuration"), row.get("seed"), row.get("phase")) for row in phase)
        != expected_phase
    ):
        raise ValueError("secondary publication phase bundle registry drifted")
    if tuple((row.get("configuration"), row.get("seed")) for row in change) != expected_keys:
        raise ValueError("secondary publication paired-change bundle registry drifted")

    for row, key in zip(pooled, expected_keys, strict=True):
        if (
            row.get("result_type") != _bundle_identity(key)["result_type"]
            or row.get("participant_count") != report.participant_count
            or row.get("session_count") != report.session_count
        ):
            raise ValueError("secondary publication pooled counts or identity drifted")
    for row, (configuration, seed, _phase) in zip(phase, expected_phase, strict=True):
        if (
            row.get("result_type") != _bundle_identity((configuration, seed))["result_type"]
            or row.get("participant_count") != report.participant_count
        ):
            raise ValueError("secondary publication phase counts or identity drifted")
    for row, key in zip(change, expected_keys, strict=True):
        if (
            row.get("result_type") != _bundle_identity(key)["result_type"]
            or row.get("participant_count") != report.participant_count
            or row.get("descriptive_noncausal") is not True
        ):
            raise ValueError("secondary publication paired-change counts or identity drifted")
    return len(pooled)


def _publication_rows(value: Any, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"secondary publication lacks {label}")
    rows = value.get("bundle_estimates")
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise ValueError(f"secondary publication {label} rows are invalid")
    return rows


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _revalidate_publication_source(
    report: StageASecondaryReport, payload: Mapping[str, Any]
) -> None:
    """Reopen the closed Stage-A release immediately before public file creation."""
    release = load_stage_a_release(report.source_root)
    _validate_release_contract(release, synthetic_contract_test=False)
    stage_a = payload["stage_a"]
    context = release.closure.get("context")
    if (
        release.root.resolve() != Path(report.source_root).resolve(strict=False)
        or release.closure_file_sha256 != report.closure_file_sha256
        or release.closure_payload_sha256 != report.closure_payload_sha256
        or release.participant_count != report.participant_count
        or len(release.session_keys) != report.session_count
        or not isinstance(context, Mapping)
        or _plain_json(context) != stage_a["provenance"]
        or release.closure.get("context_payload_sha256") != stage_a["context_payload_sha256"]
    ):
        raise ValueError("closed Stage-A source drifted after secondary report construction")


def utility_contrast(
    intervention: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the frozen positive-is-improvement sign convention to utility metrics."""
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


def _build_validated_report(
    release: StageARelease,
    *,
    bootstrap_replicates: int,
    synthetic_contract_test: bool,
) -> StageASecondaryReport:
    _validate_bootstrap_contract(bootstrap_replicates, synthetic_contract_test)
    _validate_release_contract(release, synthetic_contract_test=synthetic_contract_test)
    arrays = _paired_arrays(release)
    bundle_keys = _bundle_keys()

    pooled = _pooled_estimates(arrays, bundle_keys)
    phase = _phase_estimates(arrays, bundle_keys)
    change = _change_estimates(arrays, bundle_keys)
    pooled_summaries = _learned_seed_summaries(pooled, dimensions=())
    phase_summaries = _learned_seed_summaries(phase, dimensions=("phase",))
    change_summaries = _learned_seed_summaries(change, dimensions=())
    contrasts = _contrast_estimates(pooled)

    samples = _bootstrap_samples(
        arrays,
        bundle_keys,
        replicates=bootstrap_replicates,
    )
    _attach_bundle_intervals(pooled, samples["pooled"], dimensions=())
    _attach_bundle_intervals(phase, samples["phase"], dimensions=("phase",))
    _attach_bundle_intervals(change, samples["change"], dimensions=())
    _attach_summary_intervals(pooled_summaries, samples["pooled"], dimensions=())
    _attach_summary_intervals(phase_summaries, samples["phase"], dimensions=("phase",))
    _attach_summary_intervals(change_summaries, samples["change"], dimensions=())
    _attach_contrast_intervals(contrasts, samples["contrasts"])

    context = release.closure.get("context")
    assert isinstance(context, Mapping)
    unsigned: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "scope": "aggregate_only_no_participant_session_prediction_or_annotation_rows",
        "execution_mode": ("synthetic_contract_test" if synthetic_contract_test else "research"),
        "stage_a": {
            "closure_file_sha256": release.closure_file_sha256,
            "closure_payload_sha256": release.closure_payload_sha256,
            "context_payload_sha256": release.closure["context_payload_sha256"],
            "provenance": _plain_json(context),
            "eligible_participant_count": release.participant_count,
            "eligible_session_count": 2 * release.participant_count,
            "learned_bundle_count": 21,
            "diagnostic_floor_bundle_count": 9,
        },
        "evaluator_provenance": _evaluator_provenance(),
        "participant_cluster_bootstrap": {
            "method": "participant_cluster_percentile",
            "resamples": bootstrap_replicates,
            "random_state": BOOTSTRAP_SEED,
            "confidence_level": CONFIDENCE_LEVEL,
            "percentile_quantile_method": "linear",
            "cluster_unit": "participant_with_both_sessions_and_all_seed_model_predictions",
            "sessions_retained_per_sampled_participant": 2,
            "interval_scope": "pointwise_descriptive_not_simultaneous",
            "seeds_are_sensitivity_runs_not_resampling_units": True,
            "research_evidence": (
                not synthetic_contract_test and bootstrap_replicates == BOOTSTRAP_REPLICATES
            ),
        },
        "scientific_timing": dict(_SCIENTIFIC_TIMING),
        "pooled_utility": {
            "bundle_estimates": pooled,
            "learned_three_seed_summaries": pooled_summaries,
        },
        "phase_stratified_utility": {
            "bundle_estimates": phase,
            "learned_three_seed_summaries": phase_summaries,
        },
        "paired_change": {
            "interpretation": "descriptive_noncausal",
            "bundle_estimates": change,
            "learned_three_seed_summaries": change_summaries,
        },
        "utility_contrasts": contrasts,
        "interpretation_contract": {
            "fixed_seeds": "sensitivity_runs_not_an_ensemble_or_inferential_sample",
            "paired_change": "descriptive_noncausal",
            "utility_only": "no_fairness_or_disparity_inference",
            "construct": "crowd_perceived_hireability_not_actual_hiring_or_job_performance",
            "automatic_superiority_claims": "forbidden",
        },
        "protected_annotation_fields_accessed": [],
    }
    _assert_aggregate_only(unsigned)
    payload = {
        **unsigned,
        "aggregate_payload_sha256": hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest(),
    }
    json_bytes = canonical_json_bytes(payload) + b"\n"
    markdown_bytes = _markdown_bytes(payload)
    report = StageASecondaryReport(
        source_root=release.root,
        closure_file_sha256=release.closure_file_sha256,
        closure_payload_sha256=release.closure_payload_sha256,
        participant_count=release.participant_count,
        session_count=2 * release.participant_count,
        payload=payload,
        json_bytes=json_bytes,
        markdown_bytes=markdown_bytes,
    )
    if not synthetic_contract_test:
        object.__setattr__(
            report,
            "_publication_authorization",
            _PublicationAuthorization(
                issuer=_PUBLICATION_ISSUER,
                source_root=release.root.resolve(),
                closure_file_sha256=release.closure_file_sha256,
                closure_payload_sha256=release.closure_payload_sha256,
                participant_count=release.participant_count,
                session_count=2 * release.participant_count,
                bundle_count=len(pooled),
                json_sha256=hashlib.sha256(json_bytes).hexdigest(),
                markdown_sha256=hashlib.sha256(markdown_bytes).hexdigest(),
            ),
        )
    return report


def _validate_release_contract(release: StageARelease, *, synthetic_contract_test: bool) -> None:
    closure = release.closure
    if closure.get("protected_annotation_fields_accessed") != []:
        raise ValueError("Stage-A closure unexpectedly accessed protected annotations")
    if not synthetic_contract_test and (
        closure.get("state") != "closed"
        or closure.get("execution_mode") != "research"
        or closure.get("firewall_stage") != "A"
    ):
        raise ValueError("research secondary reporting requires a closed research Stage-A release")
    if tuple(closure.get("learned_configurations", ())) != tuple(FIREWALLED_LEARNED_CONFIGURATIONS):
        raise ValueError("Stage-A learned-configuration registry drifted")
    if tuple(closure.get("diagnostic_floor_configurations", ())) != tuple(
        DIAGNOSTIC_FLOOR_CONFIGURATIONS
    ):
        raise ValueError("Stage-A diagnostic-floor registry drifted")
    if tuple(closure.get("frozen_seeds", ())) != tuple(FROZEN_FINAL_SEEDS):
        raise ValueError("Stage-A seed registry drifted")
    expected_learned = {
        (configuration, seed)
        for configuration in FIREWALLED_LEARNED_CONFIGURATIONS
        for seed in FROZEN_FINAL_SEEDS
    }
    if set(release.learned_rows) != expected_learned:
        raise ValueError("Stage-A release does not contain exactly 21 learned bundles")
    if set(release.floor_rows) != set(DIAGNOSTIC_FLOOR_CONFIGURATIONS):
        raise ValueError("Stage-A release does not contain exactly nine diagnostic floors")
    if release.participant_count <= 0 or len(release.session_keys) != 2 * release.participant_count:
        raise ValueError("secondary reporting requires a complete paired OOF roster")
    for key, manifest in release.lock_manifests.items():
        if manifest.get("protected_annotation_fields_accessed") != []:
            raise ValueError(f"Stage-A bundle {key!r} accessed protected annotations")
    context = closure.get("context")
    if not isinstance(context, Mapping):
        raise ValueError("Stage-A closure lacks its provenance context")
    context_payload_sha256 = _require_sha256(
        closure.get("context_payload_sha256"), "Stage-A context payload SHA-256"
    )
    if (
        hashlib.sha256(canonical_json_bytes(_plain_json(context))).hexdigest()
        != context_payload_sha256
    ):
        raise ValueError("Stage-A context payload checksum failed")


def _validate_bootstrap_contract(replicates: int, synthetic_contract_test: bool) -> None:
    if isinstance(replicates, bool) or not isinstance(replicates, int):
        raise TypeError("bootstrap_replicates must be an integer")
    if replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    if not synthetic_contract_test and replicates != BOOTSTRAP_REPLICATES:
        raise ValueError("research secondary reporting requires exactly 10,000 replicates")


def _bundle_keys() -> tuple[tuple[str, int | None], ...]:
    return (
        *((configuration, None) for configuration in DIAGNOSTIC_FLOOR_CONFIGURATIONS),
        *(
            (configuration, seed)
            for configuration in FIREWALLED_LEARNED_CONFIGURATIONS
            for seed in FROZEN_FINAL_SEEDS
        ),
    )


def _paired_arrays(release: StageARelease) -> dict[str, Any]:
    participant_index = {
        participant_id: index for index, participant_id in enumerate(release.participant_ids)
    }
    phase_index = {"pre": 0, "post": 1}
    target = np.full((release.participant_count, 2), np.nan, dtype=np.float64)
    all_rows = {
        **release.learned_rows,
        **{(configuration, None): rows for configuration, rows in release.floor_rows.items()},
    }
    predictions: dict[tuple[str, int | None], np.ndarray] = {}
    for key in _bundle_keys():
        rows = all_rows[key]
        if len(rows) != 2 * release.participant_count:
            raise ValueError(f"Stage-A bundle {key!r} does not cover the paired roster")
        matrix = np.full_like(target, np.nan)
        seen: set[tuple[str, str]] = set()
        for row in rows:
            participant_id = str(row.get("participant_id"))
            phase = str(row.get("session_phase"))
            identity = (participant_id, phase)
            if (
                participant_id not in participant_index
                or phase not in phase_index
                or identity in seen
            ):
                raise ValueError(f"Stage-A bundle {key!r} has invalid paired-row identity")
            seen.add(identity)
            location = (participant_index[participant_id], phase_index[phase])
            prediction = float(row["prediction"])
            observed = float(row["target"])
            if not np.isfinite(prediction) or not np.isfinite(observed):
                raise ValueError(f"Stage-A bundle {key!r} contains a non-finite value")
            matrix[location] = prediction
            if np.isnan(target[location]):
                target[location] = observed
            elif target[location] != observed:
                raise ValueError("Stage-A target roster drifted between bundles")
        if np.isnan(matrix).any():
            raise ValueError(f"Stage-A bundle {key!r} does not contain complete pairs")
        predictions[key] = matrix
    if np.isnan(target).any():
        raise ValueError("Stage-A target roster is incomplete")
    return {"target": target, "predictions": predictions}


def _pooled_estimates(
    arrays: Mapping[str, Any], bundle_keys: Sequence[tuple[str, int | None]]
) -> list[dict[str, Any]]:
    target = arrays["target"].reshape(-1)
    return [
        {
            **_bundle_identity(key),
            "participant_count": int(arrays["target"].shape[0]),
            "session_count": int(target.size),
            "metrics": utility_metrics(target, arrays["predictions"][key].reshape(-1)),
        }
        for key in bundle_keys
    ]


def _phase_estimates(
    arrays: Mapping[str, Any], bundle_keys: Sequence[tuple[str, int | None]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in bundle_keys:
        for phase, column in (("pre", 0), ("post", 1)):
            rows.append(
                {
                    **_bundle_identity(key),
                    "phase": phase,
                    "participant_count": int(arrays["target"].shape[0]),
                    "metrics": utility_metrics(
                        arrays["target"][:, column],
                        arrays["predictions"][key][:, column],
                    ),
                }
            )
    return rows


def _change_estimates(
    arrays: Mapping[str, Any], bundle_keys: Sequence[tuple[str, int | None]]
) -> list[dict[str, Any]]:
    observed = arrays["target"][:, 1] - arrays["target"][:, 0]
    return [
        {
            **_bundle_identity(key),
            "participant_count": int(observed.size),
            "descriptive_noncausal": True,
            "metrics": paired_change_metrics(
                observed,
                arrays["predictions"][key][:, 1] - arrays["predictions"][key][:, 0],
            ),
        }
        for key in bundle_keys
    ]


def _bundle_identity(key: tuple[str, int | None]) -> dict[str, Any]:
    return {
        "result_type": "diagnostic_floor" if key[1] is None else "learned",
        "configuration": key[0],
        "seed": key[1],
    }


def _learned_seed_summaries(
    estimates: Sequence[Mapping[str, Any]], *, dimensions: tuple[str, ...]
) -> list[dict[str, Any]]:
    index = {
        (
            str(row["configuration"]),
            int(row["seed"]),
            *(row[dimension] for dimension in dimensions),
        ): row
        for row in estimates
        if row["seed"] is not None
    }
    dimension_values = (
        list(dict.fromkeys(tuple(row[dimension] for dimension in dimensions) for row in estimates))
        if dimensions
        else [()]
    )
    result: list[dict[str, Any]] = []
    metric_names = (
        _CHANGE_METRICS
        if estimates and "delta_srcc" in estimates[0]["metrics"]
        else _UTILITY_METRICS
    )
    for configuration in FIREWALLED_LEARNED_CONFIGURATIONS:
        for values in dimension_values:
            seed_rows = [index[(configuration, seed, *values)] for seed in FROZEN_FINAL_SEEDS]
            metrics = {
                metric: _seed_metric_summary(
                    [_metric_scalar(row["metrics"], metric) for row in seed_rows],
                    scaled=_is_scaled_metric(metric),
                )
                for metric in metric_names
            }
            result.append(
                {
                    "configuration": configuration,
                    **dict(zip(dimensions, values, strict=True)),
                    "fixed_seed_count": len(FROZEN_FINAL_SEEDS),
                    "interpretation": "descriptive_sensitivity_not_ensemble_or_inference",
                    "metrics": metrics,
                }
            )
    return result


def _seed_metric_summary(values: Sequence[float | None], *, scaled: bool) -> dict[str, Any] | None:
    if any(value is None for value in values):
        return None
    array = np.asarray(values, dtype=np.float64)
    normalized = {
        "mean": float(np.mean(array)),
        "sample_sd": float(np.std(array, ddof=1)),
    }
    if not scaled:
        return normalized
    return {
        "normalized": normalized,
        "original_1_7": {
            "mean": RAW_SCALE_WIDTH * normalized["mean"],
            "sample_sd": RAW_SCALE_WIDTH * normalized["sample_sd"],
        },
    }


def _contrast_estimates(pooled: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    utility_index = {
        (row["configuration"], row["seed"]): row["metrics"]
        for row in pooled
        if row["seed"] is not None
    }
    result: list[dict[str, Any]] = []
    for spec in CONTRASTS:
        seed_rows = [
            {
                "seed": seed,
                "metrics": utility_contrast(
                    utility_index[(spec["intervention"], seed)],
                    utility_index[(spec["reference"], seed)],
                ),
            }
            for seed in FROZEN_FINAL_SEEDS
        ]
        result.append(
            {
                **dict(spec),
                "sign_contract": {
                    "correlation_change": "intervention_minus_reference",
                    "error_improvement": "reference_minus_intervention",
                    "positive_is_improvement": True,
                },
                "seed_specific": seed_rows,
                "three_seed_arithmetic_mean": {
                    "fixed_seed_count": len(FROZEN_FINAL_SEEDS),
                    "interpretation": "descriptive_sensitivity_not_ensemble_or_inference",
                    "metrics": {
                        metric: _seed_metric_summary(
                            [_metric_scalar(row["metrics"], metric) for row in seed_rows],
                            scaled=_is_scaled_metric(metric),
                        )
                        for metric in (
                            "srcc_change",
                            "pearson_change",
                            "mae_improvement",
                            "rmse_improvement",
                        )
                    },
                },
            }
        )
    return result


def _bootstrap_samples(
    arrays: Mapping[str, Any],
    bundle_keys: Sequence[tuple[str, int | None]],
    *,
    replicates: int,
) -> dict[str, Any]:
    participant_count = int(arrays["target"].shape[0])
    draws = _participant_draws(participant_count, replicates)
    pooled = {
        (*key, metric): np.full(replicates, np.nan, dtype=np.float64)
        for key in bundle_keys
        for metric in _UTILITY_METRICS
    }
    phase = {
        (*key, phase_name, metric): np.full(replicates, np.nan, dtype=np.float64)
        for key in bundle_keys
        for phase_name in ("pre", "post")
        for metric in _UTILITY_METRICS
    }
    change = {
        (*key, metric): np.full(replicates, np.nan, dtype=np.float64)
        for key in bundle_keys
        for metric in _CHANGE_METRICS
    }
    contrasts = {
        (spec["contrast_id"], seed, metric): np.full(replicates, np.nan, dtype=np.float64)
        for spec in CONTRASTS
        for seed in FROZEN_FINAL_SEEDS
        for metric in (
            "srcc_change",
            "pearson_change",
            "mae_improvement",
            "rmse_improvement",
        )
    }

    for replicate_index, draw in enumerate(draws):
        sampled_target = arrays["target"][draw]
        replicate_utility: dict[tuple[str, int | None], Mapping[str, Any]] = {}
        for key in bundle_keys:
            sampled_prediction = arrays["predictions"][key][draw]
            metrics = utility_metrics(sampled_target.reshape(-1), sampled_prediction.reshape(-1))
            replicate_utility[key] = metrics
            for metric in _UTILITY_METRICS:
                pooled[(*key, metric)][replicate_index] = _nan_if_none(
                    _metric_scalar(metrics, metric)
                )
            for phase_name, column in (("pre", 0), ("post", 1)):
                phase_metrics = utility_metrics(
                    sampled_target[:, column], sampled_prediction[:, column]
                )
                for metric in _UTILITY_METRICS:
                    phase[(*key, phase_name, metric)][replicate_index] = _nan_if_none(
                        _metric_scalar(phase_metrics, metric)
                    )
            change_metrics = paired_change_metrics(
                sampled_target[:, 1] - sampled_target[:, 0],
                sampled_prediction[:, 1] - sampled_prediction[:, 0],
            )
            for metric in _CHANGE_METRICS:
                change[(*key, metric)][replicate_index] = _nan_if_none(
                    _metric_scalar(change_metrics, metric)
                )

        for spec in CONTRASTS:
            for seed in FROZEN_FINAL_SEEDS:
                metrics = utility_contrast(
                    replicate_utility[(spec["intervention"], seed)],
                    replicate_utility[(spec["reference"], seed)],
                )
                for metric in (
                    "srcc_change",
                    "pearson_change",
                    "mae_improvement",
                    "rmse_improvement",
                ):
                    contrasts[(spec["contrast_id"], seed, metric)][replicate_index] = _nan_if_none(
                        _metric_scalar(metrics, metric)
                    )
    return {
        "pooled": pooled,
        "phase": phase,
        "change": change,
        "contrasts": contrasts,
    }


def _participant_draws(participant_count: int, replicates: int) -> np.ndarray:
    """Return fixed-seed participant indices; callers retain both matrix columns."""
    if participant_count <= 0 or replicates <= 0:
        raise ValueError("participant bootstrap dimensions must be positive")
    return np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, participant_count, size=(replicates, participant_count)
    )


def _attach_bundle_intervals(
    estimates: Sequence[dict[str, Any]],
    samples: Mapping[tuple[Any, ...], np.ndarray],
    *,
    dimensions: tuple[str, ...],
) -> None:
    metric_names = (
        _CHANGE_METRICS
        if estimates and "delta_srcc" in estimates[0]["metrics"]
        else _UTILITY_METRICS
    )
    for row in estimates:
        prefix = (
            row["configuration"],
            row["seed"],
            *(row[dimension] for dimension in dimensions),
        )
        row["bootstrap_intervals"] = {
            metric: _bootstrap_interval(
                samples[(*prefix, metric)], scaled=_is_scaled_metric(metric)
            )
            for metric in metric_names
        }


def _attach_summary_intervals(
    summaries: Sequence[dict[str, Any]],
    samples: Mapping[tuple[Any, ...], np.ndarray],
    *,
    dimensions: tuple[str, ...],
) -> None:
    metric_names = (
        _CHANGE_METRICS
        if summaries and "delta_srcc" in summaries[0]["metrics"]
        else _UTILITY_METRICS
    )
    for row in summaries:
        dimension_values = tuple(row[dimension] for dimension in dimensions)
        row["bootstrap_intervals_of_arithmetic_mean"] = {}
        for metric in metric_names:
            seed_samples = np.stack(
                [
                    samples[(row["configuration"], seed, *dimension_values, metric)]
                    for seed in FROZEN_FINAL_SEEDS
                ]
            )
            complete = np.all(np.isfinite(seed_samples), axis=0)
            mean_samples = np.full(seed_samples.shape[1], np.nan, dtype=np.float64)
            mean_samples[complete] = np.mean(seed_samples[:, complete], axis=0)
            row["bootstrap_intervals_of_arithmetic_mean"][metric] = _bootstrap_interval(
                mean_samples, scaled=_is_scaled_metric(metric)
            )


def _attach_contrast_intervals(
    contrasts: Sequence[dict[str, Any]],
    samples: Mapping[tuple[Any, ...], np.ndarray],
) -> None:
    metric_names = (
        "srcc_change",
        "pearson_change",
        "mae_improvement",
        "rmse_improvement",
    )
    for contrast in contrasts:
        contrast_id = contrast["contrast_id"]
        for row in contrast["seed_specific"]:
            row["bootstrap_intervals"] = {
                metric: _bootstrap_interval(
                    samples[(contrast_id, row["seed"], metric)],
                    scaled=_is_scaled_metric(metric),
                )
                for metric in metric_names
            }
        summary = contrast["three_seed_arithmetic_mean"]
        summary["bootstrap_intervals_of_arithmetic_mean"] = {}
        for metric in metric_names:
            seed_samples = np.stack(
                [samples[(contrast_id, seed, metric)] for seed in FROZEN_FINAL_SEEDS]
            )
            complete = np.all(np.isfinite(seed_samples), axis=0)
            mean_samples = np.full(seed_samples.shape[1], np.nan, dtype=np.float64)
            mean_samples[complete] = np.mean(seed_samples[:, complete], axis=0)
            summary["bootstrap_intervals_of_arithmetic_mean"][metric] = _bootstrap_interval(
                mean_samples, scaled=_is_scaled_metric(metric)
            )


def _bootstrap_interval(values: Sequence[float], *, scaled: bool) -> dict[str, Any]:
    raw = np.asarray(values, dtype=np.float64)
    valid = raw[np.isfinite(raw)]
    result: dict[str, Any] = {
        "valid_replicate_count": int(valid.size),
        "invalid_replicate_count": int(raw.size - valid.size),
        "valid_replicate_fraction": float(valid.size / raw.size),
        "percentile_95": None,
    }
    if valid.size:
        lower, upper = np.quantile(valid, (0.025, 0.975), method="linear")
        interval = {"lower": float(lower), "upper": float(upper)}
        result["percentile_95"] = (
            {
                "normalized": interval,
                "original_1_7": {
                    "lower": RAW_SCALE_WIDTH * float(lower),
                    "upper": RAW_SCALE_WIDTH * float(upper),
                },
            }
            if scaled
            else interval
        )
    return result


def _metric_scalar(metrics: Mapping[str, Any], metric: str) -> float | None:
    value = metrics[metric]
    if _is_scaled_metric(metric):
        return float(value["normalized"])
    return None if value is None else float(value)


def _is_scaled_metric(metric: str) -> bool:
    return metric in _ERROR_METRICS or metric.endswith("_improvement")


def _units(value: float) -> dict[str, float]:
    return {"normalized": float(value), "original_1_7": RAW_SCALE_WIDTH * float(value)}


def _optional_subtract(left: Any, right: Any) -> float | None:
    return None if left is None or right is None else float(left) - float(right)


def _nan_if_none(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def _current_evaluator_provenance() -> dict[str, Any]:
    source_paths = {
        "stage_a_secondary.py": Path(__file__),
        "stage_c.py": Path(__file__).with_name("stage_c.py"),
    }
    return {
        "component_source_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in source_paths.items()
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


_IMPORTED_EVALUATOR_PROVENANCE = _current_evaluator_provenance()


def _evaluator_provenance() -> dict[str, Any]:
    current = _current_evaluator_provenance()
    if current != _IMPORTED_EVALUATOR_PROVENANCE:
        raise RuntimeError("Stage-A secondary evaluator sources changed after module import")
    return current


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


def _assert_aggregate_only(value: Any) -> None:
    if isinstance(value, Mapping):
        prohibited = set(value) & _PROHIBITED_AGGREGATE_KEYS
        if prohibited:
            raise ValueError(
                "secondary public report contains prohibited row-level keys: "
                + ", ".join(sorted(prohibited))
            )
        for item in value.values():
            _assert_aggregate_only(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _assert_aggregate_only(item)


def _markdown_bytes(payload: Mapping[str, Any]) -> bytes:
    stage_a = payload["stage_a"]
    bootstrap = payload["participant_cluster_bootstrap"]
    lines = [
        "# Stage A Secondary Utility Analysis",
        "",
        "This annotation-free report recomputes secondary utility analyses from the "
        "checksum-validated, closed Stage-A out-of-fold prediction release. It contains "
        "aggregate results only.",
        "",
        "These analyses were predeclared as secondary/descriptive and executed only after "
        "Stage-A predictions were locked. They are not confirmatory primary evidence.",
        "",
        "## Evidence binding",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Stage-A closure file SHA-256 | `{stage_a['closure_file_sha256']}` |",
        f"| Stage-A closure payload SHA-256 | `{stage_a['closure_payload_sha256']}` |",
        f"| Stage-A context payload SHA-256 | `{stage_a['context_payload_sha256']}` |",
        f"| Aggregate payload SHA-256 | `{payload['aggregate_payload_sha256']}` |",
        f"| Eligible participants | {stage_a['eligible_participant_count']} |",
        f"| Eligible sessions | {stage_a['eligible_session_count']} |",
        "| Closed prediction bundles | "
        f"{stage_a['learned_bundle_count'] + stage_a['diagnostic_floor_bundle_count']} |",
        "| Protected annotation fields accessed | none |",
        "",
        "## Participant-cluster bootstrap",
        "",
        f"Intervals use {bootstrap['resamples']:,} participant resamples with RNG seed "
        f"{bootstrap['random_state']}. Each sampled participant contributes both sessions and "
        "all same-seed/model predictions. Metrics are recomputed inside every replicate; "
        "the reported 95% percentile intervals are pointwise and descriptive.",
        "",
        "## Pooled utility by closed bundle",
        "",
        *_utility_table(payload["pooled_utility"]["bundle_estimates"]),
        "",
        "## Pooled learned three-seed sensitivity",
        "",
        *_utility_summary_table(payload["pooled_utility"]["learned_three_seed_summaries"]),
        "",
        "## Phase-stratified utility by closed bundle",
        "",
        *_utility_table(
            payload["phase_stratified_utility"]["bundle_estimates"], include_phase=True
        ),
        "",
        "## Phase-stratified learned three-seed sensitivity",
        "",
        *_utility_summary_table(
            payload["phase_stratified_utility"]["learned_three_seed_summaries"],
            include_phase=True,
        ),
        "",
        "## Descriptive paired pre/post change by closed bundle",
        "",
        *_change_table(payload["paired_change"]["bundle_estimates"]),
        "",
        "## Paired-change learned three-seed sensitivity",
        "",
        *_change_summary_table(payload["paired_change"]["learned_three_seed_summaries"]),
        "",
        "## Fixed learned-configuration utility contrasts",
        "",
        "Positive correlation changes are intervention minus reference. Positive error "
        "improvements are reference minus intervention.",
        "",
        *_contrast_table(payload["utility_contrasts"]),
        "",
        "## Interpretation boundaries",
        "",
        "- This is a predeclared secondary/descriptive analysis executed post-lock, not "
        "confirmatory primary evidence.",
        "- The three seeds are fixed sensitivity runs, not an ensemble or inferential sample.",
        "- The paired pre/post analysis is descriptive and cannot identify a coaching effect.",
        "- These utility analyses do not establish fairness, disparity reduction, actual hiring "
        "validity or job-performance validity.",
        "- MAE, RMSE, change error and error-improvement values use the normalized target; "
        "their intervals are also retained in original 1--7 scale-width units in the JSON file.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _utility_table(rows: Sequence[Mapping[str, Any]], *, include_phase: bool = False) -> list[str]:
    phase_header = " Phase |" if include_phase else ""
    separator = "---|" if include_phase else ""
    lines = [
        f"| Configuration | Seed |{phase_header} SRCC [95% CI] | Pearson [95% CI] | "
        "MAE norm. [95% CI] | RMSE norm. [95% CI] |",
        f"|---|---:|{separator}---:|---:|---:|---:|",
    ]
    for row in rows:
        phase = f" {row['phase']} |" if include_phase else ""
        intervals = row["bootstrap_intervals"]
        mae = _point_interval(row["metrics"]["mae"]["normalized"], intervals["mae"], scaled=True)
        rmse = _point_interval(row["metrics"]["rmse"]["normalized"], intervals["rmse"], scaled=True)
        lines.append(
            f"| `{row['configuration']}` | {_seed_display(row['seed'])} |{phase} "
            f"{_point_interval(row['metrics']['srcc'], intervals['srcc'])} | "
            f"{_point_interval(row['metrics']['pearson'], intervals['pearson'])} | "
            f"{mae} | {rmse} |"
        )
    return lines


def _utility_summary_table(
    rows: Sequence[Mapping[str, Any]], *, include_phase: bool = False
) -> list[str]:
    phase_header = " Phase |" if include_phase else ""
    separator = "---|" if include_phase else ""
    lines = [
        f"| Configuration |{phase_header} SRCC mean (SD) [95% CI] | "
        "Pearson mean (SD) [95% CI] | MAE norm. mean (SD) [95% CI] | "
        "RMSE norm. mean (SD) [95% CI] |",
        f"|---|{separator}---:|---:|---:|---:|",
    ]
    for row in rows:
        phase = f" {row['phase']} |" if include_phase else ""
        intervals = row["bootstrap_intervals_of_arithmetic_mean"]
        lines.append(
            f"| `{row['configuration']}` |{phase} "
            f"{_summary_interval(row['metrics']['srcc'], intervals['srcc'])} | "
            f"{_summary_interval(row['metrics']['pearson'], intervals['pearson'])} | "
            f"{_summary_interval(row['metrics']['mae'], intervals['mae'], scaled=True)} | "
            f"{_summary_interval(row['metrics']['rmse'], intervals['rmse'], scaled=True)} |"
        )
    return lines


def _change_table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| Configuration | Seed | Delta SRCC [95% CI] | Delta MAE norm. [95% CI] | "
        "Signed delta bias norm. [95% CI] |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row["metrics"]
        intervals = row["bootstrap_intervals"]
        delta_mae = _point_interval(
            metrics["delta_mae"]["normalized"], intervals["delta_mae"], scaled=True
        )
        delta_bias = _point_interval(
            metrics["signed_delta_bias"]["normalized"],
            intervals["signed_delta_bias"],
            scaled=True,
        )
        lines.append(
            f"| `{row['configuration']}` | {_seed_display(row['seed'])} | "
            f"{_point_interval(metrics['delta_srcc'], intervals['delta_srcc'])} | "
            f"{delta_mae} | {delta_bias} |"
        )
    return lines


def _change_summary_table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| Configuration | Delta SRCC mean (SD) [95% CI] | "
        "Delta MAE norm. mean (SD) [95% CI] | "
        "Signed delta bias norm. mean (SD) [95% CI] |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row["metrics"]
        intervals = row["bootstrap_intervals_of_arithmetic_mean"]
        delta_bias = _summary_interval(
            metrics["signed_delta_bias"], intervals["signed_delta_bias"], scaled=True
        )
        lines.append(
            f"| `{row['configuration']}` | "
            f"{_summary_interval(metrics['delta_srcc'], intervals['delta_srcc'])} | "
            f"{_summary_interval(metrics['delta_mae'], intervals['delta_mae'], scaled=True)} | "
            f"{delta_bias} |"
        )
    return lines


def _contrast_table(contrasts: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| Contrast | Seed | SRCC change [95% CI] | Pearson change [95% CI] | "
        "MAE improvement norm. [95% CI] | RMSE improvement norm. [95% CI] |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for contrast in contrasts:
        for row in contrast["seed_specific"]:
            metrics = row["metrics"]
            intervals = row["bootstrap_intervals"]
            mae = _point_interval(
                metrics["mae_improvement"]["normalized"],
                intervals["mae_improvement"],
                scaled=True,
            )
            rmse = _point_interval(
                metrics["rmse_improvement"]["normalized"],
                intervals["rmse_improvement"],
                scaled=True,
            )
            lines.append(
                f"| `{contrast['contrast_id']}` | {row['seed']} | "
                f"{_point_interval(metrics['srcc_change'], intervals['srcc_change'])} | "
                f"{_point_interval(metrics['pearson_change'], intervals['pearson_change'])} | "
                f"{mae} | {rmse} |"
            )
        summary = contrast["three_seed_arithmetic_mean"]
        metrics = summary["metrics"]
        intervals = summary["bootstrap_intervals_of_arithmetic_mean"]
        mae = _summary_interval(
            metrics["mae_improvement"], intervals["mae_improvement"], scaled=True
        )
        rmse = _summary_interval(
            metrics["rmse_improvement"], intervals["rmse_improvement"], scaled=True
        )
        lines.append(
            f"| `{contrast['contrast_id']}` | mean | "
            f"{_summary_interval(metrics['srcc_change'], intervals['srcc_change'])} | "
            f"{_summary_interval(metrics['pearson_change'], intervals['pearson_change'])} | "
            f"{mae} | {rmse} |"
        )
    return lines


def _seed_display(seed: Any) -> str:
    return "--" if seed is None else str(seed)


def _point_interval(value: Any, interval: Mapping[str, Any], *, scaled: bool = False) -> str:
    if value is None:
        point = "NA"
    else:
        point = f"{float(value):.6f}"
    bounds = interval["percentile_95"]
    if bounds is None:
        return f"{point} [NA]"
    if scaled:
        bounds = bounds["normalized"]
    return f"{point} [{bounds['lower']:.6f}, {bounds['upper']:.6f}]"


def _summary_interval(
    summary: Mapping[str, Any] | None,
    interval: Mapping[str, Any],
    *,
    scaled: bool = False,
) -> str:
    if summary is None:
        point = "NA"
    else:
        values = summary["normalized"] if scaled else summary
        point = f"{values['mean']:.6f} ({values['sample_sd']:.6f})"
    bounds = interval["percentile_95"]
    if bounds is None:
        return f"{point} [NA]"
    if scaled:
        bounds = bounds["normalized"]
    return f"{point} [{bounds['lower']:.6f}, {bounds['upper']:.6f}]"


def _validated_separate_output(source_root: Path, output_dir: str | Path) -> Path:
    output = Path(output_dir).expanduser()
    if any(part in {".", ".."} for part in output.parts):
        raise ValueError("secondary output path may not contain dot components")
    if not output.is_absolute():
        output = Path.cwd() / output
    resolved = output.resolve(strict=False)
    source = source_root.resolve()
    if resolved == source or resolved.is_relative_to(source) or source.is_relative_to(resolved):
        raise ValueError("secondary reports must be outside the closed Stage-A directory tree")
    return resolved


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value
