"""Immutable aggregate reporting for annotation-free Stage-A modality masks."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.frozen_json import freeze_json
from src.secure_io import AnchoredDirectory
from src.training import FROZEN_FINAL_SEEDS
from src.training.crossfit import canonical_json_bytes

from .modality_masks import (
    EXPECTED_PARTICIPANT_COUNT,
    EXPECTED_SESSION_COUNT,
    MODALITY_MASK_INTERVENTION,
    _canonical_object,
    validate_modality_mask_payload,
)
from .stage_c import (
    MASKED_MODALITIES,
    STAGE_C_MASK_SCHEMA_VERSION,
    StageARelease,
    _evaluate_modality_masks,
    _paired_arrays,
    load_stage_a_release,
)

JSON_FILENAME = "stage_a_modality_masks.json"
MARKDOWN_FILENAME = "STAGE_A_MODALITY_MASKS.md"
REPORT_SCHEMA_VERSION = "fair-avi-mit-stage-a-modality-mask-report-1"

_SCOPE = "aggregate_only_no_participant_session_prediction_or_annotation_rows"
_ANALYSIS_ROLE = "predeclared_annotation_free_failure_analysis"
_INTERVENTION_CONTRACT = {
    "configuration": "erm",
    "operation": MODALITY_MASK_INTERVENTION,
    "inference_only": True,
    "refit_or_retuning": "forbidden",
    "masked_one_modality_at_a_time": True,
}
_INTERPRETATION_CONTRACT = {
    "inference_only_no_refit": (
        "Masks were applied to locked ERM inference only; no model fitting, checkpoint "
        "selection or retuning was performed."
    ),
    "utility_only_no_fairness": (
        "This report contains utility failure-analysis results only; it does not estimate "
        "fairness, group disparity or protected-attribute effects."
    ),
    "change_sign_contract": (
        "Positive correlation change and positive MAE or RMSE improvement favour the masked "
        "prediction over locked ERM."
    ),
    "seed_summary": (
        "Arithmetic mean and sample SD across fixed seeds 1, 11 and 21; descriptive "
        "sensitivity only, not an ensemble or confidence interval."
    ),
}
_TOP_FIELDS = {
    "schema_version",
    "scope",
    "analysis_role",
    "execution_mode",
    "stage_a",
    "modality_mask",
    "intervention_contract",
    "interpretation_contract",
    "participant_count",
    "session_count",
    "point_estimates",
    "three_seed_summaries",
    "evaluator_provenance",
    "protected_annotation_fields_accessed",
    "report_payload_sha256",
}
_POINT_FIELDS = {
    "seed",
    "masked_modality",
    "participant_count",
    "session_count",
    "masked_utility",
    "change_from_locked_erm",
}
_SUMMARY_FIELDS = {
    "masked_modality",
    "fixed_seed_count",
    "summary_statistic",
    "masked_utility",
    "change_from_locked_erm",
}
_PROHIBITED_AGGREGATE_KEYS = {
    "participant_id",
    "session_id",
    "source_session_id",
    "prediction",
    "predictions",
    "prediction_lock_id",
    "base_prediction_lock_id",
    "model_fit_sha256",
    "rater_id",
    "annotation",
    "annotations",
}
_PUBLICATION_ISSUER = object()


def _current_evaluator_provenance() -> dict[str, Any]:
    source_root = Path(__file__).resolve().parent
    return {
        "reporting_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "component_source_sha256": {
            filename: hashlib.sha256((source_root / filename).read_bytes()).hexdigest()
            for filename in ("modality_masks.py", "stage_c.py")
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


_IMPORTED_EVALUATOR_PROVENANCE = freeze_json(_current_evaluator_provenance())


@dataclass(frozen=True, slots=True)
class _PublicationAuthorization:
    issuer: object
    source_root: Path
    mask_path: Path
    closure_file_sha256: str
    closure_payload_sha256: str
    mask_file_sha256: str
    mask_payload_sha256: str
    json_sha256: str
    markdown_sha256: str


@dataclass(frozen=True, slots=True)
class StageAMaskReport:
    """Canonical aggregate evidence derived from one closed mask release."""

    source_root: Path
    mask_path: Path
    closure_file_sha256: str
    closure_payload_sha256: str
    mask_file_sha256: str
    mask_payload_sha256: str
    payload: Mapping[str, Any]
    json_bytes: bytes
    markdown_bytes: bytes
    _publication_authorization: _PublicationAuthorization | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_root", Path(self.source_root).resolve(strict=False))
        object.__setattr__(self, "mask_path", Path(self.mask_path).resolve(strict=False))
        object.__setattr__(self, "payload", freeze_json(self.payload))


@dataclass(frozen=True, slots=True)
class PublishedStageAMaskReport:
    """Observed checksums after immutable public-file publication."""

    state: str
    json_sha256: str
    markdown_sha256: str
    point_row_count: int
    summary_row_count: int


def build_stage_a_mask_report(
    stage_a_dir: str | Path,
    modality_mask_path: str | Path,
    *,
    expected_stage_a_closure_file_sha256: str,
    expected_stage_a_closure_payload_sha256: str,
    expected_modality_mask_file_sha256: str,
    expected_modality_mask_payload_sha256: str,
) -> StageAMaskReport:
    """Reopen both externally bound releases and build aggregate utility evidence."""
    closure_file_sha256 = _require_sha256(
        expected_stage_a_closure_file_sha256,
        "expected Stage-A closure file SHA-256",
    )
    closure_payload_sha256 = _require_sha256(
        expected_stage_a_closure_payload_sha256,
        "expected Stage-A closure payload SHA-256",
    )
    mask_file_sha256 = _require_sha256(
        expected_modality_mask_file_sha256,
        "expected modality-mask file SHA-256",
    )
    mask_payload_sha256 = _require_sha256(
        expected_modality_mask_payload_sha256,
        "expected modality-mask payload SHA-256",
    )
    source = _lexical_absolute(stage_a_dir, "Stage-A source")
    release = load_stage_a_release(source)
    _validate_stage_a_release(
        release,
        source=source,
        closure_file_sha256=closure_file_sha256,
        closure_payload_sha256=closure_payload_sha256,
    )
    mask_path = _lexical_absolute(modality_mask_path, "modality-mask input")
    mask_payload, evaluation = _read_and_evaluate_mask(
        release,
        mask_path,
        expected_file_sha256=mask_file_sha256,
        expected_payload_sha256=mask_payload_sha256,
    )
    points = _point_estimates(evaluation, release)
    summaries = _three_seed_summaries(points)
    unsigned = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "scope": _SCOPE,
        "analysis_role": _ANALYSIS_ROLE,
        "execution_mode": "research",
        "stage_a": {
            "closure_file_sha256": closure_file_sha256,
            "closure_payload_sha256": closure_payload_sha256,
        },
        "modality_mask": {
            "file_sha256": mask_file_sha256,
            "payload_sha256": mask_payload_sha256,
            "schema_version": mask_payload["schema_version"],
            "configuration": mask_payload["configuration"],
            "masked_modalities": list(MASKED_MODALITIES),
            "fixed_seeds": list(FROZEN_FINAL_SEEDS),
        },
        "intervention_contract": _INTERVENTION_CONTRACT,
        "interpretation_contract": _INTERPRETATION_CONTRACT,
        "participant_count": release.participant_count,
        "session_count": len(release.session_keys),
        "point_estimates": points,
        "three_seed_summaries": summaries,
        "evaluator_provenance": _evaluator_provenance(),
        "protected_annotation_fields_accessed": [],
    }
    payload = {
        **unsigned,
        "report_payload_sha256": hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest(),
    }
    json_bytes = canonical_json_bytes(payload)
    markdown_bytes = _markdown_bytes(payload)
    report = StageAMaskReport(
        source_root=release.root,
        mask_path=mask_path,
        closure_file_sha256=closure_file_sha256,
        closure_payload_sha256=closure_payload_sha256,
        mask_file_sha256=mask_file_sha256,
        mask_payload_sha256=mask_payload_sha256,
        payload=payload,
        json_bytes=json_bytes,
        markdown_bytes=markdown_bytes,
    )
    _validate_report_payload(payload, report)
    authorization = _PublicationAuthorization(
        issuer=_PUBLICATION_ISSUER,
        source_root=report.source_root,
        mask_path=report.mask_path,
        closure_file_sha256=closure_file_sha256,
        closure_payload_sha256=closure_payload_sha256,
        mask_file_sha256=mask_file_sha256,
        mask_payload_sha256=mask_payload_sha256,
        json_sha256=hashlib.sha256(json_bytes).hexdigest(),
        markdown_sha256=hashlib.sha256(markdown_bytes).hexdigest(),
    )
    object.__setattr__(report, "_publication_authorization", authorization)
    return report


def publish_stage_a_mask_report(
    report: StageAMaskReport,
    output_dir: str | Path,
) -> PublishedStageAMaskReport:
    """Revalidate sources and publish the two fixed files without replacement."""
    payload = _validate_publishable_report(report)
    output = _validated_separate_output(report, output_dir)
    contents = {
        JSON_FILENAME: report.json_bytes,
        MARKDOWN_FILENAME: report.markdown_bytes,
    }
    with AnchoredDirectory.open(output, create=True, directory_mode=0o755) as anchor:
        with anchor.exclusive_lock():
            existing_names = set(anchor.list_names())
            for filename, content in contents.items():
                if filename not in existing_names:
                    continue
                existing = anchor.read_bytes(filename, expected_mode=0o644)
                if existing.content != content:
                    raise FileExistsError(
                        f"refusing to replace non-identical Stage-A mask report {filename!r}"
                    )
            written = {
                filename: anchor.publish_bytes(filename, content, mode=0o644)
                for filename, content in contents.items()
            }
            for filename, observed in written.items():
                anchor.verify_file(filename, observed, expected_mode=0o644)
    return PublishedStageAMaskReport(
        state="resumed" if set(contents).issubset(existing_names) else "published",
        json_sha256=written[JSON_FILENAME].sha256,
        markdown_sha256=written[MARKDOWN_FILENAME].sha256,
        point_row_count=len(payload["point_estimates"]),
        summary_row_count=len(payload["three_seed_summaries"]),
    )


def publication_summary(
    report: StageAMaskReport,
    published: PublishedStageAMaskReport,
) -> bytes:
    """Return a compact canonical aggregate-only publication receipt."""
    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": published.state,
        "scope": _SCOPE,
        "stage_a_closure": {
            "file_sha256": report.closure_file_sha256,
            "payload_sha256": report.closure_payload_sha256,
        },
        "modality_mask": {
            "file_sha256": report.mask_file_sha256,
            "payload_sha256": report.mask_payload_sha256,
        },
        "json": {
            "filename": JSON_FILENAME,
            "sha256": published.json_sha256,
            "point_row_count": published.point_row_count,
        },
        "markdown": {
            "filename": MARKDOWN_FILENAME,
            "sha256": published.markdown_sha256,
            "summary_row_count": published.summary_row_count,
        },
        "protected_annotation_fields_accessed": [],
    }
    return canonical_json_bytes(payload)


def report_content_sha256(report: StageAMaskReport) -> Mapping[str, str]:
    """Expose deterministic content hashes for independent verification."""
    return {
        JSON_FILENAME: hashlib.sha256(report.json_bytes).hexdigest(),
        MARKDOWN_FILENAME: hashlib.sha256(report.markdown_bytes).hexdigest(),
    }


def _validate_publishable_report(report: StageAMaskReport) -> dict[str, Any]:
    if not isinstance(report, StageAMaskReport):
        raise TypeError("Stage-A mask publication requires a StageAMaskReport")
    authorization = report._publication_authorization
    if authorization is None or authorization.issuer is not _PUBLICATION_ISSUER:
        raise ValueError("only builder-issued Stage-A mask reports may be published")
    payload = _canonical_report_object(report.json_bytes)
    if canonical_json_bytes(_plain_json(report.payload)) != report.json_bytes:
        raise ValueError("Stage-A mask report payload and JSON bytes disagree")
    _validate_report_payload(payload, report)
    if report.markdown_bytes != _markdown_bytes(payload):
        raise ValueError("Stage-A mask Markdown does not match its signed JSON payload")
    if (
        authorization.source_root != report.source_root
        or authorization.mask_path != report.mask_path
        or authorization.closure_file_sha256 != report.closure_file_sha256
        or authorization.closure_payload_sha256 != report.closure_payload_sha256
        or authorization.mask_file_sha256 != report.mask_file_sha256
        or authorization.mask_payload_sha256 != report.mask_payload_sha256
        or authorization.json_sha256 != hashlib.sha256(report.json_bytes).hexdigest()
        or authorization.markdown_sha256 != hashlib.sha256(report.markdown_bytes).hexdigest()
    ):
        raise ValueError("Stage-A mask report no longer matches its builder authorization")
    rebuilt = build_stage_a_mask_report(
        report.source_root,
        report.mask_path,
        expected_stage_a_closure_file_sha256=report.closure_file_sha256,
        expected_stage_a_closure_payload_sha256=report.closure_payload_sha256,
        expected_modality_mask_file_sha256=report.mask_file_sha256,
        expected_modality_mask_payload_sha256=report.mask_payload_sha256,
    )
    if rebuilt.json_bytes != report.json_bytes or rebuilt.markdown_bytes != report.markdown_bytes:
        raise ValueError("Stage-A or modality-mask source drifted after report construction")
    return payload


def _validate_stage_a_release(
    release: StageARelease,
    *,
    source: Path,
    closure_file_sha256: str,
    closure_payload_sha256: str,
) -> None:
    if not isinstance(release, StageARelease):
        raise TypeError("Stage-A loader did not return a StageARelease")
    closure = release.closure
    if (
        release.root.resolve() != source.resolve()
        or release.closure_file_sha256 != closure_file_sha256
        or release.closure_payload_sha256 != closure_payload_sha256
    ):
        raise ValueError("Stage-A release does not match the externally recorded checksums")
    if (
        closure.get("execution_mode") != "research"
        or closure.get("firewall_stage") != "A"
        or closure.get("protected_annotation_fields_accessed") != []
        or release.participant_count != EXPECTED_PARTICIPANT_COUNT
        or len(release.session_keys) != EXPECTED_SESSION_COUNT
        or tuple(closure.get("frozen_seeds", ())) != tuple(FROZEN_FINAL_SEEDS)
    ):
        raise ValueError("Stage-A modality-mask reporting requires the closed research release")


def _read_and_evaluate_mask(
    release: StageARelease,
    path: Path,
    *,
    expected_file_sha256: str,
    expected_payload_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if path.name in {"", ".", ".."}:
        raise ValueError("modality-mask input filename is unsafe")
    with AnchoredDirectory.open(path.parent, directory_mode=0o700) as anchor:
        with anchor.open_file(
            path.name,
            expected_mode=0o600,
            expected_sha256=expected_file_sha256,
        ) as opened:
            content = opened.read_bytes()
            payload = _canonical_object(content)
            validate_modality_mask_payload(payload, content=content, release=release)
            if payload["mask_payload_sha256"] != expected_payload_sha256:
                raise ValueError(
                    "modality-mask payload does not match the externally recorded checksum"
                )
            evaluation = _evaluate_exact_mask_bytes(release, content)
            opened.verify_current_file()
        anchor.verify_path_identity()
    if (
        evaluation.get("status") != "reported_locked_inference_only_no_refit"
        or evaluation.get("file_sha256") != expected_file_sha256
        or evaluation.get("payload_sha256") != expected_payload_sha256
    ):
        raise ValueError("cross-bound modality-mask evaluation provenance drifted")
    return payload, evaluation


def _evaluate_exact_mask_bytes(release: StageARelease, content: bytes) -> dict[str, Any]:
    temporary_root = Path(tempfile.gettempdir()).resolve()
    with tempfile.TemporaryDirectory(prefix="fair-avi-stage-a-mask-", dir=temporary_root) as value:
        directory = Path(value)
        os.chmod(directory, 0o700)
        filename = "bound-modality-mask.json"
        with AnchoredDirectory.open(directory, directory_mode=0o700) as anchor:
            observed = anchor.publish_bytes(
                filename,
                content,
                mode=0o600,
                allow_existing=False,
            )
            result = _evaluate_modality_masks(
                release,
                _paired_arrays(release),
                directory / filename,
            )
            anchor.verify_file(filename, observed, expected_mode=0o600)
    return result


def _point_estimates(evaluation: Mapping[str, Any], release: StageARelease) -> list[dict[str, Any]]:
    raw_rows = evaluation.get("results")
    if not isinstance(raw_rows, list) or len(raw_rows) != 9:
        raise ValueError("modality-mask evaluation must contain exactly nine point rows")
    expected = [(seed, modality) for seed in FROZEN_FINAL_SEEDS for modality in MASKED_MODALITIES]
    rows = []
    for raw, key in zip(raw_rows, expected, strict=True):
        if (
            not isinstance(raw, Mapping)
            or set(raw) != {"seed", "masked_modality", "metrics", "change_from_locked_erm"}
            or (raw["seed"], raw["masked_modality"]) != key
        ):
            raise ValueError("modality-mask seed/modality point registry drifted")
        utility = _plain_json(raw["metrics"])
        change = _plain_json(raw["change_from_locked_erm"])
        _validate_utility(utility)
        _validate_change(change)
        rows.append(
            {
                "seed": key[0],
                "masked_modality": key[1],
                "participant_count": release.participant_count,
                "session_count": len(release.session_keys),
                "masked_utility": utility,
                "change_from_locked_erm": change,
            }
        )
    return rows


def _three_seed_summaries(
    points: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {(row["seed"], row["masked_modality"]): row for row in points}
    expected = {(seed, modality) for seed in FROZEN_FINAL_SEEDS for modality in MASKED_MODALITIES}
    if set(by_key) != expected:
        raise ValueError("three-seed summary input registry drifted")
    summaries = []
    for modality in MASKED_MODALITIES:
        rows = [by_key[(seed, modality)] for seed in FROZEN_FINAL_SEEDS]
        summaries.append(
            {
                "masked_modality": modality,
                "fixed_seed_count": len(FROZEN_FINAL_SEEDS),
                "summary_statistic": "arithmetic_mean_and_sample_sd",
                "masked_utility": _summarise_utility(rows),
                "change_from_locked_erm": _summarise_change(rows),
            }
        )
    return summaries


def _summarise_utility(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = [row["masked_utility"] for row in rows]
    return {
        "srcc": _mean_and_sample_sd([value["srcc"] for value in metrics]),
        "pearson": _mean_and_sample_sd([value["pearson"] for value in metrics]),
        "mae": {
            unit: _mean_and_sample_sd([value["mae"][unit] for value in metrics])
            for unit in ("normalized", "original_1_7")
        },
        "rmse": {
            unit: _mean_and_sample_sd([value["rmse"][unit] for value in metrics])
            for unit in ("normalized", "original_1_7")
        },
    }


def _summarise_change(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = [row["change_from_locked_erm"] for row in rows]
    return {
        "srcc_change": _mean_and_sample_sd([value["srcc_change"] for value in metrics]),
        "pearson_change": _mean_and_sample_sd([value["pearson_change"] for value in metrics]),
        "mae_improvement": {
            unit: _mean_and_sample_sd([value["mae_improvement"][unit] for value in metrics])
            for unit in ("normalized", "original_1_7")
        },
        "rmse_improvement": {
            unit: _mean_and_sample_sd([value["rmse_improvement"][unit] for value in metrics])
            for unit in ("normalized", "original_1_7")
        },
    }


def _mean_and_sample_sd(values: Sequence[Any]) -> dict[str, float] | None:
    if len(values) != len(FROZEN_FINAL_SEEDS):
        raise ValueError("seed summaries require exactly three values")
    if any(value is None for value in values):
        return None
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(FROZEN_FINAL_SEEDS),) or not np.isfinite(array).all():
        raise ValueError("seed-summary values must be finite scalars or null")
    return {
        "mean": float(np.mean(array)),
        "sample_sd": float(np.std(array, ddof=1)),
    }


def _validate_report_payload(payload: Mapping[str, Any], report: StageAMaskReport) -> None:
    if not isinstance(payload, Mapping) or set(payload) != _TOP_FIELDS:
        raise ValueError("Stage-A mask report top-level schema drifted")
    unsigned = {key: value for key, value in payload.items() if key != "report_payload_sha256"}
    if (
        payload.get("schema_version") != REPORT_SCHEMA_VERSION
        or payload.get("scope") != _SCOPE
        or payload.get("analysis_role") != _ANALYSIS_ROLE
        or payload.get("execution_mode") != "research"
        or payload.get("intervention_contract") != _INTERVENTION_CONTRACT
        or payload.get("interpretation_contract") != _INTERPRETATION_CONTRACT
        or payload.get("participant_count") != EXPECTED_PARTICIPANT_COUNT
        or payload.get("session_count") != EXPECTED_SESSION_COUNT
        or payload.get("protected_annotation_fields_accessed") != []
        or payload.get("evaluator_provenance") != _evaluator_provenance()
        or payload.get("report_payload_sha256")
        != hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    ):
        raise ValueError("Stage-A mask report fixed contract or signature drifted")
    stage_a = payload.get("stage_a")
    mask = payload.get("modality_mask")
    if stage_a != {
        "closure_file_sha256": report.closure_file_sha256,
        "closure_payload_sha256": report.closure_payload_sha256,
    }:
        raise ValueError("Stage-A mask report closure binding drifted")
    if mask != {
        "file_sha256": report.mask_file_sha256,
        "payload_sha256": report.mask_payload_sha256,
        "schema_version": STAGE_C_MASK_SCHEMA_VERSION,
        "configuration": "erm",
        "masked_modalities": list(MASKED_MODALITIES),
        "fixed_seeds": list(FROZEN_FINAL_SEEDS),
    }:
        raise ValueError("Stage-A mask report input binding drifted")
    points = payload.get("point_estimates")
    if not isinstance(points, list) or len(points) != 9:
        raise ValueError("Stage-A mask report must contain exactly nine point rows")
    expected = [(seed, modality) for seed in FROZEN_FINAL_SEEDS for modality in MASKED_MODALITIES]
    for row, key in zip(points, expected, strict=True):
        if (
            not isinstance(row, Mapping)
            or set(row) != _POINT_FIELDS
            or (row.get("seed"), row.get("masked_modality")) != key
            or row.get("participant_count") != EXPECTED_PARTICIPANT_COUNT
            or row.get("session_count") != EXPECTED_SESSION_COUNT
        ):
            raise ValueError("Stage-A mask point-row registry drifted")
        _validate_utility(row["masked_utility"])
        _validate_change(row["change_from_locked_erm"])
    summaries = payload.get("three_seed_summaries")
    expected_summaries = _three_seed_summaries(points)
    if (
        not isinstance(summaries, list)
        or len(summaries) != len(MASKED_MODALITIES)
        or any(not isinstance(row, Mapping) or set(row) != _SUMMARY_FIELDS for row in summaries)
        or canonical_json_bytes({"rows": summaries})
        != canonical_json_bytes({"rows": expected_summaries})
    ):
        raise ValueError("Stage-A mask three-seed summaries drifted")
    _assert_aggregate_only(payload)


def _validate_utility(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != {"srcc", "pearson", "mae", "rmse"}:
        raise ValueError("masked utility schema drifted")
    _optional_finite(value["srcc"], "masked utility SRCC")
    _optional_finite(value["pearson"], "masked utility Pearson")
    _validate_units(value["mae"], "masked utility MAE")
    _validate_units(value["rmse"], "masked utility RMSE")


def _validate_change(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "srcc_change",
        "pearson_change",
        "mae_improvement",
        "rmse_improvement",
    }:
        raise ValueError("locked-ERM utility-change schema drifted")
    _optional_finite(value["srcc_change"], "SRCC change")
    _optional_finite(value["pearson_change"], "Pearson change")
    _validate_units(value["mae_improvement"], "MAE improvement", allow_negative=True)
    _validate_units(value["rmse_improvement"], "RMSE improvement", allow_negative=True)


def _validate_units(value: Any, label: str, *, allow_negative: bool = False) -> None:
    if not isinstance(value, Mapping) or set(value) != {"normalized", "original_1_7"}:
        raise ValueError(f"{label} unit schema drifted")
    normalized = _finite(value["normalized"], f"{label} normalized")
    original = _finite(value["original_1_7"], f"{label} original scale")
    if (not allow_negative and normalized < 0) or not math.isclose(
        original,
        6.0 * normalized,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError(f"{label} units drifted")


def _optional_finite(value: Any, label: str) -> float | None:
    return None if value is None else _finite(value, label)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _assert_aggregate_only(payload: Mapping[str, Any]) -> None:
    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            forbidden = set(value) & _PROHIBITED_AGGREGATE_KEYS
            if forbidden:
                raise ValueError(f"aggregate report contains restricted keys: {sorted(forbidden)}")
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(payload)


def _markdown_bytes(payload: Mapping[str, Any]) -> bytes:
    points = payload["point_estimates"]
    summaries = payload["three_seed_summaries"]
    lines = [
        "# Stage A Modality-Mask Failure Analysis",
        "",
        "These are predeclared, annotation-free, inference-only failure-analysis results for "
        "the locked ERM model. One training-standardised modality summary was set to zero at "
        "inference. No model was refitted, selected or retuned.",
        "",
        "This is a utility-only report. It does not estimate fairness, group disparity or "
        "protected-attribute effects, and no protected annotations were accessed.",
        "",
        "## Bound inputs",
        "",
        "| Binding | SHA-256 |",
        "|---|---|",
        f"| Stage-A closure file | `{payload['stage_a']['closure_file_sha256']}` |",
        f"| Stage-A closure payload | `{payload['stage_a']['closure_payload_sha256']}` |",
        f"| Modality-mask file | `{payload['modality_mask']['file_sha256']}` |",
        f"| Modality-mask payload | `{payload['modality_mask']['payload_sha256']}` |",
        f"| Aggregate report payload | `{payload['report_payload_sha256']}` |",
        "",
        "## Seed-specific point estimates",
        "",
        "Positive correlation changes and positive MAE/RMSE improvements favour the masked "
        "prediction over locked ERM.",
        "",
        "| Seed | Masked modality | SRCC | Pearson | MAE norm. | RMSE norm. | SRCC change | "
        "Pearson change | MAE improvement | RMSE improvement |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in points:
        utility = row["masked_utility"]
        change = row["change_from_locked_erm"]
        lines.append(
            f"| {row['seed']} | `{row['masked_modality']}` | "
            f"{_display_number(utility['srcc'])} | {_display_number(utility['pearson'])} | "
            f"{_display_number(utility['mae']['normalized'])} | "
            f"{_display_number(utility['rmse']['normalized'])} | "
            f"{_display_number(change['srcc_change'])} | "
            f"{_display_number(change['pearson_change'])} | "
            f"{_display_number(change['mae_improvement']['normalized'])} | "
            f"{_display_number(change['rmse_improvement']['normalized'])} |"
        )
    lines.extend(
        [
            "",
            "## Three-seed descriptive sensitivity",
            "",
            "Each entry is the arithmetic mean and sample SD across fixed seeds 1, 11 and 21. "
            "It is not an ensemble estimate, confidence interval or inferential standard error.",
            "",
            "| Masked modality | SRCC mean (SD) | Pearson mean (SD) | MAE norm. mean (SD) | "
            "RMSE norm. mean (SD) | SRCC change mean (SD) | Pearson change mean (SD) | "
            "MAE improvement mean (SD) | RMSE improvement mean (SD) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        utility = row["masked_utility"]
        change = row["change_from_locked_erm"]
        lines.append(
            f"| `{row['masked_modality']}` | {_display_summary(utility['srcc'])} | "
            f"{_display_summary(utility['pearson'])} | "
            f"{_display_summary(utility['mae']['normalized'])} | "
            f"{_display_summary(utility['rmse']['normalized'])} | "
            f"{_display_summary(change['srcc_change'])} | "
            f"{_display_summary(change['pearson_change'])} | "
            f"{_display_summary(change['mae_improvement']['normalized'])} | "
            f"{_display_summary(change['rmse_improvement']['normalized'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The mask intervention is an inference-only diagnostic of model dependence on each "
            "modality summary. It is not a causal ablation, a retrained modality-specific model, "
            "or evidence of fairness improvement.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _display_number(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.6f}"


def _display_summary(value: Mapping[str, Any] | None) -> str:
    if value is None:
        return "NA"
    return f"{float(value['mean']):.6f} ({float(value['sample_sd']):.6f})"


def _canonical_report_object(raw: bytes) -> dict[str, Any]:
    if not isinstance(raw, bytes):
        raise TypeError("Stage-A mask report JSON must be bytes")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate Stage-A mask report JSON key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("Stage-A mask report JSON cannot be read") from error
    if not isinstance(payload, dict) or canonical_json_bytes(payload) != raw:
        raise ValueError("Stage-A mask report JSON is not byte-canonical")
    return payload


def _validated_separate_output(report: StageAMaskReport, output_dir: str | Path) -> Path:
    output = _lexical_absolute(output_dir, "Stage-A mask report output")
    resolved = output.resolve(strict=False)
    source = report.source_root.resolve()
    mask_parent = report.mask_path.parent.resolve()
    if (
        resolved == source
        or resolved.is_relative_to(source)
        or source.is_relative_to(resolved)
        or resolved == mask_parent
        or resolved.is_relative_to(mask_parent)
        or mask_parent.is_relative_to(resolved)
    ):
        raise ValueError("public mask reports must be separate from restricted input trees")
    return resolved


def _lexical_absolute(value: str | Path, label: str) -> Path:
    supplied = Path(value).expanduser()
    if any(part in {".", ".."} for part in supplied.parts):
        raise ValueError(f"{label} path may not contain dot components")
    return supplied if supplied.is_absolute() else Path.cwd() / supplied


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    return value


def _evaluator_provenance() -> dict[str, Any]:
    return _plain_json(_IMPORTED_EVALUATOR_PROVENANCE)
