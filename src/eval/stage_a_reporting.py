"""Aggregate-only public reporting for a checksum-validated Stage-A release."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.secure_io import AnchoredDirectory
from src.training import (
    DIAGNOSTIC_FLOOR_CONFIGURATIONS,
    FIREWALLED_LEARNED_CONFIGURATIONS,
    FROZEN_FINAL_SEEDS,
    FROZEN_OUTER_ROTATIONS,
    utility_metrics,
)
from src.training.crossfit import canonical_json_bytes

from .stage_c import StageARelease, load_stage_a_release

CSV_FILENAME = "stage_a_utility.csv"
MARKDOWN_FILENAME = "STAGE_A_RESULTS.md"
REPORT_SCHEMA_VERSION = "fair-avi-mit-stage-a-public-report-1"

_LOCK_UTILITY_FIELDS = {
    "srcc",
    "pearson",
    "mae_normalised",
    "mae_raw_scale",
    "rmse",
}
_BUNDLE_HASH_FIELDS = (
    "prediction_lock_id",
    "prediction_lock_file_sha256",
    "prediction_lock_payload_sha256",
    "prediction_file_sha256",
)
_CSV_FIELDS = (
    "result_type",
    "configuration",
    "seed",
    "participant_count",
    "session_count",
    "srcc",
    "pearson",
    "mae_normalised",
    "mae_raw_scale",
    "rmse_normalised",
    "rmse_raw_scale",
    *_BUNDLE_HASH_FIELDS,
    "stage_a_closure_file_sha256",
    "stage_a_closure_payload_sha256",
)


@dataclass(frozen=True, slots=True)
class StageAPublicReport:
    """Deterministic aggregate report with no participant-level rows."""

    source_root: Path
    closure_file_sha256: str
    closure_payload_sha256: str
    participant_count: int
    session_count: int
    fold_participant_counts: tuple[int, ...]
    metric_rows: tuple[Mapping[str, Any], ...]
    learned_summaries: tuple[Mapping[str, Any], ...]
    csv_bytes: bytes
    markdown_bytes: bytes


@dataclass(frozen=True, slots=True)
class PublishedStageAReport:
    """Checksums for the two public files after safe publication."""

    csv_sha256: str
    markdown_sha256: str
    row_count: int


def build_stage_a_public_report(
    stage_a_dir: str | Path,
    *,
    expected_closure_file_sha256: str,
    expected_closure_payload_sha256: str,
) -> StageAPublicReport:
    """Validate the closed release, recompute utility, and build aggregate-only bytes."""
    _require_sha256(expected_closure_file_sha256, "expected closure file SHA-256")
    _require_sha256(expected_closure_payload_sha256, "expected closure payload SHA-256")
    release = load_stage_a_release(stage_a_dir)
    if (
        release.closure_file_sha256 != expected_closure_file_sha256
        or release.closure_payload_sha256 != expected_closure_payload_sha256
    ):
        raise ValueError("Stage-A closure does not match the externally recorded checksums")
    return _build_validated_report(release)


def publish_stage_a_public_report(
    report: StageAPublicReport, output_dir: str | Path
) -> PublishedStageAReport:
    """Publish both public files atomically per file without replacing different content."""
    output = _validated_separate_output(report.source_root, output_dir)
    contents = {
        CSV_FILENAME: report.csv_bytes,
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
                        f"refusing to replace non-identical public report {filename!r}"
                    )
            written = {
                filename: anchor.publish_bytes(filename, content, mode=0o644)
                for filename, content in contents.items()
            }
            for filename, observed in written.items():
                anchor.verify_file(filename, observed, expected_mode=0o644)
    return PublishedStageAReport(
        csv_sha256=written[CSV_FILENAME].sha256,
        markdown_sha256=written[MARKDOWN_FILENAME].sha256,
        row_count=len(report.metric_rows),
    )


def _build_validated_report(release: StageARelease) -> StageAPublicReport:
    closure = release.closure
    if closure.get("protected_annotation_fields_accessed") != []:
        raise ValueError("Stage-A closure unexpectedly accessed protected annotations")
    if tuple(closure.get("learned_configurations", ())) != tuple(FIREWALLED_LEARNED_CONFIGURATIONS):
        raise ValueError("Stage-A learned-configuration registry drifted")
    if tuple(closure.get("diagnostic_floor_configurations", ())) != tuple(
        DIAGNOSTIC_FLOOR_CONFIGURATIONS
    ):
        raise ValueError("Stage-A diagnostic-floor registry drifted")
    if tuple(closure.get("frozen_seeds", ())) != tuple(FROZEN_FINAL_SEEDS):
        raise ValueError("Stage-A seed registry drifted")
    if tuple(closure.get("frozen_outer_folds", ())) != tuple(FROZEN_OUTER_ROTATIONS):
        raise ValueError("Stage-A fold registry drifted")

    expected_learned = {
        (configuration, seed)
        for configuration in FIREWALLED_LEARNED_CONFIGURATIONS
        for seed in FROZEN_FINAL_SEEDS
    }
    expected_floors = {(configuration, None) for configuration in DIAGNOSTIC_FLOOR_CONFIGURATIONS}
    if set(release.learned_rows) != expected_learned:
        raise ValueError("Stage-A release does not contain exactly 21 learned bundles")
    if {(configuration, None) for configuration in release.floor_rows} != expected_floors:
        raise ValueError("Stage-A release does not contain exactly nine diagnostic floors")
    if set(release.lock_manifests) != expected_learned | expected_floors:
        raise ValueError("Stage-A lock-manifest registry is incomplete or unexpected")

    participant_count = release.participant_count
    session_count = len(release.session_keys)
    if participant_count <= 0 or session_count != 2 * participant_count:
        raise ValueError("Stage-A public reporting requires a complete paired OOF roster")
    fold_counts = Counter(release.participant_folds.values())
    if set(fold_counts) - set(FROZEN_OUTER_ROTATIONS):
        raise ValueError("Stage-A participant fold registry is invalid")
    fold_participant_counts = tuple(fold_counts[fold] for fold in FROZEN_OUTER_ROTATIONS)
    if sum(fold_participant_counts) != participant_count:
        raise ValueError("Stage-A participant fold counts do not cover the release")

    bundle_records = _closure_bundle_records(closure)
    rows: list[dict[str, Any]] = []
    for configuration in DIAGNOSTIC_FLOOR_CONFIGURATIONS:
        key = (configuration, None)
        rows.append(
            _aggregate_row(
                release,
                key=key,
                result_type="deterministic_floor",
                prediction_rows=release.floor_rows[configuration],
                bundle_record=bundle_records[key],
                participant_count=participant_count,
                session_count=session_count,
            )
        )
    for configuration in FIREWALLED_LEARNED_CONFIGURATIONS:
        for seed in FROZEN_FINAL_SEEDS:
            key = (configuration, seed)
            rows.append(
                _aggregate_row(
                    release,
                    key=key,
                    result_type="learned",
                    prediction_rows=release.learned_rows[key],
                    bundle_record=bundle_records[key],
                    participant_count=participant_count,
                    session_count=session_count,
                )
            )
    if len(rows) != 30:
        raise RuntimeError("Stage-A public utility table must contain exactly 30 rows")

    summaries = _learned_seed_summaries(rows)
    csv_bytes = _csv_bytes(rows)
    markdown_bytes = _markdown_bytes(
        release,
        rows,
        summaries,
        participant_count=participant_count,
        session_count=session_count,
        fold_participant_counts=fold_participant_counts,
    )
    return StageAPublicReport(
        source_root=release.root,
        closure_file_sha256=release.closure_file_sha256,
        closure_payload_sha256=release.closure_payload_sha256,
        participant_count=participant_count,
        session_count=session_count,
        fold_participant_counts=fold_participant_counts,
        metric_rows=tuple(rows),
        learned_summaries=tuple(summaries),
        csv_bytes=csv_bytes,
        markdown_bytes=markdown_bytes,
    )


def _closure_bundle_records(
    closure: Mapping[str, Any],
) -> dict[tuple[str, int | None], Mapping[str, Any]]:
    learned = closure.get("learned_bundles")
    floors = closure.get("diagnostic_floors")
    if not isinstance(learned, Sequence) or isinstance(learned, (str, bytes)):
        raise ValueError("Stage-A closure lacks learned bundle records")
    if not isinstance(floors, Sequence) or isinstance(floors, (str, bytes)):
        raise ValueError("Stage-A closure lacks diagnostic-floor records")
    result: dict[tuple[str, int | None], Mapping[str, Any]] = {}
    for record in (*learned, *floors):
        if not isinstance(record, Mapping):
            raise ValueError("Stage-A closure bundle record is invalid")
        configuration = record.get("configuration")
        seed = record.get("seed")
        if not isinstance(configuration, str) or (
            seed is not None and (isinstance(seed, bool) or not isinstance(seed, int))
        ):
            raise ValueError("Stage-A closure bundle identity is invalid")
        key = (configuration, seed)
        if key in result:
            raise ValueError("Stage-A closure bundle identity is duplicated")
        for field in _BUNDLE_HASH_FIELDS:
            _require_sha256(record.get(field), f"Stage-A bundle {field}")
        result[key] = record
    expected = {
        *((configuration, None) for configuration in DIAGNOSTIC_FLOOR_CONFIGURATIONS),
        *(
            (configuration, seed)
            for configuration in FIREWALLED_LEARNED_CONFIGURATIONS
            for seed in FROZEN_FINAL_SEEDS
        ),
    }
    if set(result) != expected:
        raise ValueError("Stage-A closure bundle registry is incomplete or unexpected")
    return result


def _aggregate_row(
    release: StageARelease,
    *,
    key: tuple[str, int | None],
    result_type: str,
    prediction_rows: Sequence[Mapping[str, Any]],
    bundle_record: Mapping[str, Any],
    participant_count: int,
    session_count: int,
) -> dict[str, Any]:
    if len(prediction_rows) != session_count:
        raise ValueError(f"Stage-A bundle {key!r} does not cover the common OOF roster")
    target = np.asarray([row["target"] for row in prediction_rows], dtype=np.float64)
    prediction = np.asarray([row["prediction"] for row in prediction_rows], dtype=np.float64)
    recomputed = utility_metrics(target, prediction)
    manifest = release.lock_manifests[key]
    if manifest.get("protected_annotation_fields_accessed") != []:
        raise ValueError(f"Stage-A bundle {key!r} accessed protected annotations")
    locked = manifest.get("utility")
    if not isinstance(locked, Mapping) or set(locked) != _LOCK_UTILITY_FIELDS:
        raise ValueError(f"Stage-A bundle {key!r} has an invalid locked utility package")
    if canonical_json_bytes(dict(locked)) != canonical_json_bytes(recomputed):
        raise ValueError(
            f"Stage-A bundle {key!r} locked utility disagrees with recomputed pooled OOF metrics"
        )
    if bundle_record.get("oof_row_count") != session_count:
        raise ValueError(f"Stage-A bundle {key!r} closure row count drifted")
    return {
        "result_type": result_type,
        "configuration": key[0],
        "seed": key[1],
        "participant_count": participant_count,
        "session_count": session_count,
        "srcc": recomputed["srcc"],
        "pearson": recomputed["pearson"],
        "mae_normalised": recomputed["mae_normalised"],
        "mae_raw_scale": recomputed["mae_raw_scale"],
        "rmse_normalised": recomputed["rmse"],
        "rmse_raw_scale": 6.0 * float(recomputed["rmse"]),
        **{field: bundle_record[field] for field in _BUNDLE_HASH_FIELDS},
        "stage_a_closure_file_sha256": release.closure_file_sha256,
        "stage_a_closure_payload_sha256": release.closure_payload_sha256,
    }


def _learned_seed_summaries(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {
        (row["configuration"], row["seed"]): row for row in rows if row["result_type"] == "learned"
    }
    summaries = []
    for configuration in FIREWALLED_LEARNED_CONFIGURATIONS:
        seed_rows = [by_key[(configuration, seed)] for seed in FROZEN_FINAL_SEEDS]
        metrics: dict[str, Mapping[str, float] | None] = {}
        for field in ("srcc", "pearson", "mae_normalised", "rmse_normalised"):
            metrics[field] = _mean_and_sample_sd([row[field] for row in seed_rows])
        mae = metrics["mae_normalised"]
        rmse = metrics["rmse_normalised"]
        metrics["mae_raw_scale"] = _scaled_summary(mae, 6.0)
        metrics["rmse_raw_scale"] = _scaled_summary(rmse, 6.0)
        summaries.append(
            {
                "configuration": configuration,
                "fixed_seed_count": len(FROZEN_FINAL_SEEDS),
                "interpretation": "descriptive_sensitivity_not_ensemble_or_inference",
                "metrics": metrics,
            }
        )
    return summaries


def _mean_and_sample_sd(values: Sequence[Any]) -> dict[str, float] | None:
    if any(value is None for value in values):
        return None
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(FROZEN_FINAL_SEEDS),) or not np.all(np.isfinite(array)):
        raise ValueError("Stage-A seed metrics must be three finite scalar values or null")
    return {
        "mean": float(np.mean(array)),
        "sample_sd": float(np.std(array, ddof=1)),
    }


def _scaled_summary(summary: Mapping[str, float] | None, scale: float) -> dict[str, float] | None:
    if summary is None:
        return None
    return {
        "mean": scale * float(summary["mean"]),
        "sample_sd": scale * float(summary["sample_sd"]),
    }


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=_CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_value(row[field]) for field in _CSV_FIELDS})
    content = output.getvalue().encode("utf-8")
    if len(list(csv.DictReader(io.StringIO(content.decode("utf-8"))))) != 30:
        raise RuntimeError("public Stage-A CSV must contain exactly 30 data rows")
    return content


def _csv_value(value: Any) -> str | int:
    if value is None:
        return ""
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("public Stage-A CSV cannot contain non-finite values")
        return repr(value)
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        return value
    raise TypeError("public Stage-A CSV contains an unsupported aggregate value")


def _markdown_bytes(
    release: StageARelease,
    rows: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    *,
    participant_count: int,
    session_count: int,
    fold_participant_counts: Sequence[int],
) -> bytes:
    context = release.closure.get("context")
    if not isinstance(context, Mapping):
        raise ValueError("Stage-A closure lacks its reporting context")
    provenance_fields = (
        ("Stage-A closure file SHA-256", release.closure_file_sha256),
        ("Stage-A closure payload SHA-256", release.closure_payload_sha256),
        ("Stage-A context payload SHA-256", release.closure.get("context_payload_sha256")),
        ("Eligibility manifest SHA-256", context.get("eligibility_manifest_sha256")),
        ("Target manifest SHA-256", context.get("target_manifest_sha256")),
        (
            "Feature closure file SHA-256",
            context.get("feature_extraction_closure_file_sha256"),
        ),
        (
            "Feature closure payload SHA-256",
            context.get("feature_extraction_closure_payload_sha256"),
        ),
        ("Configuration SHA-256", context.get("config_sha256")),
        ("Code source-tree SHA-256", context.get("code_source_tree_sha256")),
        ("Code commit", context.get("code_commit_sha")),
    )
    for label, value in provenance_fields:
        if label == "Code commit":
            if not isinstance(value, str) or len(value) != 40:
                raise ValueError("Stage-A reporting context has an invalid code commit")
        else:
            _require_sha256(value, label)

    floors = [row for row in rows if row["result_type"] == "deterministic_floor"]
    learned = [row for row in rows if row["result_type"] == "learned"]
    lines = [
        "# Stage A Utility Results",
        "",
        "This report was generated from checksum-validated, locked out-of-fold prediction "
        "bundles. It contains aggregate utility only; no participant-level predictions or "
        "identifiers are included.",
        "",
        "The target is crowd-perceived hireability (`RecommendHiring`), not actual hiring or "
        "job performance. Metrics pool all held-out session rows on the common participant-"
        "grouped eligibility set; fold metrics are not averaged and predictions are not clipped.",
        "",
        "## Release and evaluation set",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Eligible participants | {participant_count} |",
        f"| Eligible sessions | {session_count} |",
        "| Participant counts in folds 0--4 | "
        + "/".join(str(value) for value in fold_participant_counts)
        + " |",
        f"| Learned configurations | {len(FIREWALLED_LEARNED_CONFIGURATIONS)} |",
        f"| Learned seed bundles | {len(learned)} |",
        f"| Learned fits | {len(learned) * len(FROZEN_OUTER_ROTATIONS)} |",
        f"| Deterministic diagnostic floors | {len(floors)} |",
        "",
        "## Provenance",
        "",
        "| Binding | Value |",
        "|---|---|",
        *(f"| {label} | `{value}` |" for label, value in provenance_fields),
        "",
        "## Deterministic diagnostic floors",
        "",
        *_metric_table(floors),
        "",
        "## Learned configurations by seed",
        "",
        *_metric_table(learned),
        "",
        "## Learned three-seed descriptive sensitivity",
        "",
        "Each entry is the arithmetic mean and sample SD across fixed seeds 1, 11 and 21. "
        "The SD is a descriptive sensitivity summary, not an ensemble result, confidence "
        "interval or inferential standard error.",
        "",
        *_summary_table(summaries),
        "",
        "## Interpretation boundaries",
        "",
        "- SRCC and Pearson are unitless. MAE and RMSE are shown on the normalized 0--1 "
        "target and in original 1--7 scale-width units.",
        "- The repeated prediction entries across configurations and seeds do not increase "
        "the independent sample beyond the eligible paired participants and sessions above.",
        "- These Stage-A utility results do not establish fairness, disparity reduction, "
        "causal coaching effects, hiring validity or job-performance validity.",
        "- Protected annotations were not accessed by Stage A. Annotation-dependent Stage B "
        "and Stage C results must be reported separately after their release gates.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _metric_table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| Configuration | Seed | Sessions | SRCC | Pearson | MAE norm. | MAE 1--7 | "
        "RMSE norm. | RMSE 1--7 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        seed = "--" if row["seed"] is None else str(row["seed"])
        lines.append(
            f"| `{row['configuration']}` | {seed} | {row['session_count']} | "
            f"{_display_number(row['srcc'])} | {_display_number(row['pearson'])} | "
            f"{_display_number(row['mae_normalised'])} | "
            f"{_display_number(row['mae_raw_scale'])} | "
            f"{_display_number(row['rmse_normalised'])} | "
            f"{_display_number(row['rmse_raw_scale'])} |"
        )
    return lines


def _summary_table(summaries: Sequence[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| Configuration | SRCC mean (SD) | Pearson mean (SD) | MAE norm. mean (SD) | "
        "MAE 1--7 mean (SD) | RMSE norm. mean (SD) | RMSE 1--7 mean (SD) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        metrics = summary["metrics"]
        lines.append(
            f"| `{summary['configuration']}` | {_display_summary(metrics['srcc'])} | "
            f"{_display_summary(metrics['pearson'])} | "
            f"{_display_summary(metrics['mae_normalised'])} | "
            f"{_display_summary(metrics['mae_raw_scale'])} | "
            f"{_display_summary(metrics['rmse_normalised'])} | "
            f"{_display_summary(metrics['rmse_raw_scale'])} |"
        )
    return lines


def _display_number(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.6f}"


def _display_summary(value: Mapping[str, Any] | None) -> str:
    if value is None:
        return "NA"
    return f"{float(value['mean']):.6f} ({float(value['sample_sd']):.6f})"


def _validated_separate_output(source_root: Path, output_dir: str | Path) -> Path:
    output = Path(output_dir).expanduser()
    if any(part in {".", ".."} for part in output.parts):
        raise ValueError("public Stage-A output path may not contain dot components")
    if not output.is_absolute():
        output = Path.cwd() / output
    resolved = output.resolve(strict=False)
    source = source_root.resolve()
    if resolved == source or resolved.is_relative_to(source) or source.is_relative_to(resolved):
        raise ValueError("public reports must be outside the closed Stage-A directory tree")
    return resolved


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def publication_summary(report: StageAPublicReport, published: PublishedStageAReport) -> bytes:
    """Return a canonical aggregate-only CLI receipt."""
    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "published",
        "stage_a_closure": {
            "file_sha256": report.closure_file_sha256,
            "payload_sha256": report.closure_payload_sha256,
        },
        "participant_count": report.participant_count,
        "session_count": report.session_count,
        "csv": {
            "filename": CSV_FILENAME,
            "sha256": published.csv_sha256,
            "aggregate_row_count": published.row_count,
        },
        "markdown": {
            "filename": MARKDOWN_FILENAME,
            "sha256": published.markdown_sha256,
        },
        "protected_annotation_fields_accessed": [],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def report_content_sha256(report: StageAPublicReport) -> Mapping[str, str]:
    """Expose deterministic content checksums for validation and tests."""
    return {
        CSV_FILENAME: hashlib.sha256(report.csv_bytes).hexdigest(),
        MARKDOWN_FILENAME: hashlib.sha256(report.markdown_bytes).hexdigest(),
    }
