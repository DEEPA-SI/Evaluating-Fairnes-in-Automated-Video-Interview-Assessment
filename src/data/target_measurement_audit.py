"""Deterministic, privacy-safe measurement audit for MIT ``RecommendHiring``.

The versioned report contains only aggregate coverage and measurement summaries. Worker,
participant and session identifiers, row-level ratings, and raw CSV text are used transiently
for validation and are never included in the returned payload.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import stat
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from .annotation_audit import krippendorff_alpha
from .mit_interview import EXPECTED_PARTICIPANTS, EXPECTED_SESSIONS, RATING_FIELDS, parse_session_id

TARGET_MEASUREMENT_AUDIT_SCHEMA_VERSION = "mit-recommend-hiring-measurement-audit-1"
TARGET_MEASUREMENT_METHOD_VERSION = "fair-avi-mit-recommend-hiring-measurement-1"
TARGET_FIELD = "RecommendHiring"
AGGREGATE_WORKER_TOKEN = "AGGR"
EXPECTED_INDIVIDUAL_RATINGS = 1_240
EXPECTED_WORKERS = 9
EXPECTED_RATERS_PER_SESSION = (8, 9)
SCORE_HEADER = ("Participant", "Worker", *RATING_FIELDS)

_DECIMAL_PATTERN = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", re.ASCII)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}", re.ASCII)
_MAX_SCORE_FILE_BYTES = 32 * 1024 * 1024
_NUMERIC_DECIMAL_PLACES = 12
_TOP_FIELDS = {
    "schema_version",
    "status",
    "purpose",
    "privacy",
    "bindings",
    "method",
    "coverage",
    "ordinal_krippendorff_alpha",
    "session_dispersion",
    "anonymized_worker_severity",
    "em_aggregate_vs_unweighted_individual_mean",
    "raw_pre_post_descriptive",
    "report_payload_sha256",
}
_PRIVACY_CONTRACT = {
    "contains_worker_identifiers": False,
    "contains_participant_or_session_identifiers": False,
    "contains_row_level_ratings": False,
    "contains_raw_csv_rows_or_text": False,
    "worker_reporting_scope": "anonymized_aggregate_distribution_only",
}


@dataclass(frozen=True, slots=True)
class _SessionRatings:
    """Validated values retained only while aggregate summaries are computed."""

    condition: Literal["pre", "post"]
    aggregate: Fraction
    individual: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class _ParsedScores:
    sessions: tuple[_SessionRatings, ...]
    source_session_ids: tuple[str, ...]
    worker_count: int
    total_rows: int
    individual_rows: int
    aggregate_rows: int


def audit_target_measurement_file(
    path: str | Path,
    *,
    expected_source_session_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Read one stable regular scores CSV and return its aggregate-only audit report."""
    raw = _read_stable_regular_file(Path(path), "MIT scores CSV", _MAX_SCORE_FILE_BYTES)
    return build_target_measurement_audit(
        raw,
        expected_source_session_ids=expected_source_session_ids,
    )


def build_target_measurement_audit(
    scores_csv_bytes: bytes,
    *,
    expected_source_session_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build the checksum-bound report from exact scores CSV bytes."""
    if not isinstance(scores_csv_bytes, bytes):
        raise TypeError("scores_csv_bytes must be exact bytes")
    if not scores_csv_bytes:
        raise ValueError("MIT scores CSV is empty")
    if len(scores_csv_bytes) > _MAX_SCORE_FILE_BYTES:
        raise ValueError("MIT scores CSV exceeds the fixed size limit")
    parsed = _parse_scores_csv(scores_csv_bytes)
    if expected_source_session_ids is None:
        bound_roster = parsed.source_session_ids
        roster_binding_scope = "scores_csv_observed_source_roster_only"
    else:
        bound_roster = validate_source_session_roster(expected_source_session_ids)
        roster_binding_scope = "explicit_expected_source_roster"
        if bound_roster != parsed.source_session_ids:
            raise ValueError("scores CSV source-session coverage does not match expected roster")
    sessions = parsed.sessions

    units = [[value for _, value in row.individual] for row in sessions]
    alpha = krippendorff_alpha(units, categories=tuple(range(1, 8)), metric="ordinal")

    standard_deviations: list[float] = []
    interquartile_ranges: list[Fraction] = []
    value_ranges: list[Fraction] = []
    mean_absolute_deviations: list[Fraction] = []
    residuals_by_worker: dict[str, list[Fraction]] = defaultdict(list)
    worker_rating_counts: Counter[str] = Counter()
    aggregate_values: list[Fraction] = []
    individual_means: list[Fraction] = []

    for row in sessions:
        individual = tuple(Fraction(value) for _, value in row.individual)
        mean = sum(individual, Fraction()) / len(individual)
        individual_means.append(mean)
        aggregate_values.append(row.aggregate)
        variance = sum((value - mean) ** 2 for value in individual) / len(individual)
        standard_deviations.append(_canonical_float(math.sqrt(float(variance))))
        interquartile_ranges.append(
            _quantile(individual, Fraction(3, 4)) - _quantile(individual, Fraction(1, 4))
        )
        value_ranges.append(max(individual) - min(individual))
        mean_absolute_deviations.append(
            sum(abs(value - mean) for value in individual) / len(individual)
        )
        for worker, value in row.individual:
            residuals_by_worker[worker].append(Fraction(value) - mean)
            worker_rating_counts[worker] += 1

    severity = tuple(
        sum(values, Fraction()) / len(values) for values in residuals_by_worker.values()
    )
    if len(severity) != EXPECTED_WORKERS:
        raise ValueError("worker-severity calculation did not cover exactly nine workers")

    pre_values = [row.aggregate for row in sessions if row.condition == "pre"]
    post_values = [row.aggregate for row in sessions if row.condition == "post"]
    unsigned: dict[str, Any] = {
        "schema_version": TARGET_MEASUREMENT_AUDIT_SCHEMA_VERSION,
        "status": "validated_aggregate_measurement_audit",
        "purpose": "restricted_recommend_hiring_target_measurement_audit",
        "privacy": dict(_PRIVACY_CONTRACT),
        "bindings": {
            "scores_csv": {
                "size_bytes": len(scores_csv_bytes),
                "file_sha256": hashlib.sha256(scores_csv_bytes).hexdigest(),
            },
            "parser_contract_sha256": _canonical_sha256(_parser_contract()),
            "source_session_roster": {
                "binding_scope": roster_binding_scope,
                "source_session_count": EXPECTED_SESSIONS,
                "source_numeric_pair_count": EXPECTED_PARTICIPANTS,
                "roster_sha256": _source_roster_sha256(bound_roster),
            },
            "implementation_file_sha256": _implementation_file_hashes(),
        },
        "method": _method_contract(),
        "coverage": {
            "source_numeric_pair_count": EXPECTED_PARTICIPANTS,
            "session_count": parsed.aggregate_rows,
            "pre_session_count": len(pre_values),
            "post_session_count": len(post_values),
            "aggregate_rating_count": parsed.aggregate_rows,
            "individual_rating_count": parsed.individual_rows,
            "total_csv_row_count": parsed.total_rows,
            "distinct_individual_worker_count": parsed.worker_count,
            "individual_ratings_per_session": {"8": 2, "9": 136},
            "complete_pre_post_pairing_verified": True,
            "complete_aggregate_and_individual_session_coverage_verified": True,
        },
        "ordinal_krippendorff_alpha": alpha,
        "session_dispersion": {
            "session_count": len(sessions),
            "population_standard_deviation": _distribution_summary(standard_deviations),
            "interquartile_range": _distribution_summary(interquartile_ranges),
            "range": _distribution_summary(value_ranges),
            "mean_absolute_deviation_from_session_mean": _distribution_summary(
                mean_absolute_deviations
            ),
        },
        "anonymized_worker_severity": {
            "worker_count": len(severity),
            "ratings_per_worker_distribution": _distribution_summary(
                tuple(Fraction(value) for value in worker_rating_counts.values())
            ),
            "session_centered_mean_residual_distribution": {
                **_distribution_summary(severity),
                "range": _fraction_float(max(severity) - min(severity)),
                "negative_count": sum(value < 0 for value in severity),
                "zero_count": sum(value == 0 for value in severity),
                "positive_count": sum(value > 0 for value in severity),
            },
        },
        "em_aggregate_vs_unweighted_individual_mean": {
            "session_count": len(sessions),
            "pearson_correlation": _correlation(aggregate_values, individual_means),
            "spearman_correlation": _correlation(
                _average_ranks(aggregate_values), _average_ranks(individual_means)
            ),
            "mean_absolute_error": _fraction_float(
                sum(
                    (
                        abs(aggregate - mean)
                        for aggregate, mean in zip(aggregate_values, individual_means, strict=True)
                    ),
                    Fraction(),
                )
                / len(sessions)
            ),
            "maximum_absolute_difference": _fraction_float(
                max(
                    abs(aggregate - mean)
                    for aggregate, mean in zip(aggregate_values, individual_means, strict=True)
                )
            ),
        },
        "raw_pre_post_descriptive": {
            "interpretation": "unadjusted_descriptive_only_noncausal",
            "value_source": "raw_EM_aggregate_RecommendHiring_on_1_to_7_scale",
            "pre": {"count": len(pre_values), "mean": _fraction_float(_mean(pre_values))},
            "post": {"count": len(post_values), "mean": _fraction_float(_mean(post_values))},
            "post_minus_pre_mean": _fraction_float(_mean(post_values) - _mean(pre_values)),
        },
    }
    report = {**unsigned, "report_payload_sha256": _canonical_sha256(unsigned)}
    validate_target_measurement_audit_report(report)
    return report


def canonical_target_measurement_audit_json(payload: Mapping[str, Any]) -> str:
    """Return stable human-readable JSON after complete schema validation."""
    validate_target_measurement_audit_report(payload)
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"


def validate_source_session_roster(values: Sequence[str]) -> tuple[str, ...]:
    """Validate 69 non-assumed numeric source pairs and return canonical source IDs."""
    if isinstance(values, (str, bytes)):
        raise TypeError("source-session roster must be a sequence of identifiers")
    canonical: list[tuple[str, str, Literal["pre", "post"]]] = []
    for value in values:
        try:
            canonical.append(parse_session_id(value))
        except ValueError:
            raise ValueError("source-session roster contains an invalid identifier") from None
    session_ids = [session_id for session_id, _, _ in canonical]
    if len(session_ids) != EXPECTED_SESSIONS or len(set(session_ids)) != EXPECTED_SESSIONS:
        raise ValueError("source-session roster must contain exactly 138 unique source sessions")
    phases_by_number: dict[str, set[str]] = defaultdict(set)
    for _, numeric_id, phase in canonical:
        phases_by_number[numeric_id].add(phase)
    if len(phases_by_number) != EXPECTED_PARTICIPANTS or any(
        phases != {"pre", "post"} for phases in phases_by_number.values()
    ):
        raise ValueError("source-session roster must contain 69 complete numeric P/PP pairs")
    return tuple(
        sorted(
            session_ids,
            key=lambda value: (
                int(parse_session_id(value)[1]),
                0 if parse_session_id(value)[2] == "pre" else 1,
            ),
        )
    )


def source_session_roster_from_video_directory(path: str | Path) -> tuple[str, ...]:
    """Return the exact canonical source roster represented by safe ``*.avi`` filenames."""
    directory = Path(path)
    if directory.is_symlink():
        raise ValueError("source video roster must be a real non-symlink directory")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError as error:
        raise ValueError("cannot open a safe source video roster directory") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISDIR(before.st_mode):
            raise ValueError("source video roster must be a real directory")
        session_ids: list[str] = []
        with os.scandir(descriptor) as entries:
            for entry in entries:
                suffix = Path(entry.name).suffix
                if suffix.lower() != ".avi":
                    continue
                if (
                    suffix != ".avi"
                    or entry.is_symlink()
                    or not entry.is_file(follow_symlinks=False)
                ):
                    raise ValueError("source video roster contains an unsafe AVI entry")
                try:
                    session_id, _, _ = parse_session_id(Path(entry.name).stem)
                except ValueError:
                    raise ValueError(
                        "source video roster contains an invalid AVI filename"
                    ) from None
                if entry.name != f"{session_id}.avi":
                    raise ValueError("source video roster contains a noncanonical AVI filename")
                session_ids.append(session_id)
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_identity != after_identity:
            raise ValueError("source video roster changed while it was being read")
    finally:
        os.close(descriptor)
    return validate_source_session_roster(session_ids)


def load_target_measurement_audit_report(path: str | Path) -> dict[str, Any]:
    """Load a canonical, duplicate-key-safe, checksum-bound audit report."""
    source = Path(path)
    raw = _read_stable_regular_file(
        source, "target-measurement audit report", _MAX_SCORE_FILE_BYTES
    )
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("cannot read a valid target-measurement audit report") from error
    validate_target_measurement_audit_report(payload)
    canonical = canonical_target_measurement_audit_json(payload).encode("utf-8")
    if raw != canonical:
        raise ValueError("target-measurement audit report is not canonically serialized")
    return dict(payload)


def validate_target_measurement_audit_against_source(
    report_path: str | Path,
    scores_path: str | Path,
    *,
    expected_source_session_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Rebuild from the bound source and reject report or implementation drift."""
    observed = load_target_measurement_audit_report(report_path)
    expected = audit_target_measurement_file(
        scores_path,
        expected_source_session_ids=expected_source_session_ids,
    )
    if canonical_target_measurement_audit_json(observed) != canonical_target_measurement_audit_json(
        expected
    ):
        raise ValueError("target-measurement audit does not match its bound source and code")
    return observed


def validate_target_measurement_audit_report(payload: Mapping[str, Any]) -> None:
    """Fail closed on schema, privacy, coverage, method, numeric, or checksum drift."""
    if not isinstance(payload, Mapping) or set(payload) != _TOP_FIELDS:
        raise ValueError("target-measurement audit has an unexpected top-level schema")
    if payload["schema_version"] != TARGET_MEASUREMENT_AUDIT_SCHEMA_VERSION:
        raise ValueError("unsupported target-measurement audit schema_version")
    if (
        payload["status"] != "validated_aggregate_measurement_audit"
        or payload["purpose"] != "restricted_recommend_hiring_target_measurement_audit"
        or payload["privacy"] != _PRIVACY_CONTRACT
    ):
        raise ValueError("target-measurement audit purpose or privacy contract drifted")
    if payload["method"] != _method_contract():
        raise ValueError("target-measurement audit method contract drifted")
    _validate_bindings(payload["bindings"])
    _validate_coverage(payload["coverage"])
    _validate_alpha(payload["ordinal_krippendorff_alpha"])
    _validate_session_dispersion(payload["session_dispersion"])
    _validate_worker_severity(payload["anonymized_worker_severity"])
    _validate_aggregate_comparison(payload["em_aggregate_vs_unweighted_individual_mean"])
    _validate_pre_post(payload["raw_pre_post_descriptive"])
    checksum = payload["report_payload_sha256"]
    _validated_sha256(checksum, "report_payload_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "report_payload_sha256"}
    if checksum != _canonical_sha256(unsigned):
        raise ValueError("target-measurement audit payload checksum does not match")


def _parse_scores_csv(raw: bytes) -> _ParsedScores:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("MIT scores CSV must be strict UTF-8") from error
    if "\0" in text:
        raise ValueError("MIT scores CSV may not contain NUL bytes")
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        header = next(reader)
    except (StopIteration, csv.Error) as error:
        raise ValueError("MIT scores CSV is empty or malformed") from error
    if tuple(header) != SCORE_HEADER or len(set(header)) != len(header):
        raise ValueError("MIT scores CSV header does not match the exact published schema")

    aggregate: dict[str, Fraction] = {}
    individual: dict[str, list[tuple[str, int]]] = defaultdict(list)
    session_metadata: dict[str, tuple[str, Literal["pre", "post"]]] = {}
    seen_keys: set[tuple[str, str]] = set()
    workers: set[str] = set()
    total_rows = individual_rows = aggregate_rows = 0
    try:
        for row_number, raw_row in enumerate(reader, start=2):
            if len(raw_row) != len(SCORE_HEADER):
                raise ValueError(
                    f"MIT scores CSV row {row_number} must have exactly {len(SCORE_HEADER)} fields"
                )
            if any("\n" in value or "\r" in value or "\0" in value for value in raw_row):
                raise ValueError(f"MIT scores CSV row {row_number} contains an unsafe field")
            row = dict(zip(SCORE_HEADER, raw_row, strict=True))
            try:
                session_id, participant_id, condition = parse_session_id(row["Participant"])
            except ValueError:
                raise ValueError(
                    f"MIT scores CSV row {row_number} has an invalid session key"
                ) from None
            worker = row["Worker"]
            if (
                not worker
                or worker != worker.strip()
                or len(worker) > 256
                or any(ord(character) < 32 or ord(character) == 127 for character in worker)
            ):
                raise ValueError(f"MIT scores CSV row {row_number} has an invalid worker key")
            if (
                worker.casefold() == AGGREGATE_WORKER_TOKEN.casefold()
                and worker != AGGREGATE_WORKER_TOKEN
            ):
                raise ValueError("MIT scores CSV contains a noncanonical aggregate worker token")
            key = (session_id, worker)
            if key in seen_keys:
                raise ValueError(f"duplicate session-worker rating key at row {row_number}")
            seen_keys.add(key)

            values = {
                field: _decimal_rating(row[field], field, row_number) for field in RATING_FIELDS
            }
            target = values[TARGET_FIELD]
            if not Decimal(1) <= target <= Decimal(7):
                raise ValueError(
                    f"{TARGET_FIELD} at row {row_number} is outside the required 1--7 scale"
                )
            for field in RATING_FIELDS[:-1]:
                if not Decimal(0) <= values[field] <= Decimal(7):
                    raise ValueError(
                        f"{field} at row {row_number} is outside the accepted 0--7 range"
                    )
            expected_total = sum((values[field] for field in RATING_FIELDS[:-1]), Decimal())
            if abs(values["Total"] - expected_total) > Decimal("0.000001"):
                raise ValueError(f"Total at row {row_number} does not equal the rating-field sum")

            total_rows += 1
            session_metadata[session_id] = (participant_id, condition)
            if worker == AGGREGATE_WORKER_TOKEN:
                aggregate_rows += 1
                if session_id in aggregate:
                    raise ValueError(f"duplicate aggregate rating at row {row_number}")
                aggregate[session_id] = Fraction(target)
            else:
                individual_rows += 1
                if target != target.to_integral_value():
                    raise ValueError(
                        f"individual {TARGET_FIELD} at row {row_number} must be an integer category"
                    )
                individual[session_id].append((worker, int(target)))
                workers.add(worker)
    except csv.Error as error:
        raise ValueError("MIT scores CSV contains malformed CSV structure") from error

    if total_rows != EXPECTED_SESSIONS + EXPECTED_INDIVIDUAL_RATINGS:
        raise ValueError("MIT scores CSV total row-count contract drifted")
    if aggregate_rows != EXPECTED_SESSIONS or individual_rows != EXPECTED_INDIVIDUAL_RATINGS:
        raise ValueError("MIT scores CSV aggregate or individual row-count contract drifted")
    if set(aggregate) != set(individual) or len(aggregate) != EXPECTED_SESSIONS:
        raise ValueError("aggregate and individual target session coverage does not match")
    counts = Counter(len(values) for values in individual.values())
    if counts != Counter({9: 136, 8: 2}):
        raise ValueError("individual target coverage must contain 8 or 9 ratings per session")
    if len(workers) != EXPECTED_WORKERS:
        raise ValueError("MIT scores CSV must contain exactly nine individual workers")

    source_session_ids = validate_source_session_roster(tuple(aggregate))

    sessions = tuple(
        _SessionRatings(
            condition=session_metadata[session_id][1],
            aggregate=aggregate[session_id],
            individual=tuple(individual[session_id]),
        )
        for session_id in sorted(
            aggregate,
            key=lambda value: (
                int(session_metadata[value][0]),
                0 if session_metadata[value][1] == "pre" else 1,
            ),
        )
    )
    return _ParsedScores(
        sessions=sessions,
        source_session_ids=source_session_ids,
        worker_count=len(workers),
        total_rows=total_rows,
        individual_rows=individual_rows,
        aggregate_rows=aggregate_rows,
    )


def _parser_contract() -> dict[str, Any]:
    return {
        "header": list(SCORE_HEADER),
        "encoding": "utf-8-sig_strict",
        "csv_reader": "python_csv_reader_strict_exact_row_shape",
        "session_key": "canonical_uppercase_P_or_PP_plus_positive_ascii_decimal",
        "duplicate_key": "canonical_session_id_plus_exact_worker_token",
        "aggregate_token": AGGREGATE_WORKER_TOKEN,
        "target_field": TARGET_FIELD,
        "individual_target_categories": list(range(1, 8)),
        "aggregate_target_range": [1, 7],
        "non_target_rating_range": [0, 7],
        "total_tolerance": 0.000001,
        "expected_sessions": EXPECTED_SESSIONS,
        "expected_source_numeric_pair_count": EXPECTED_PARTICIPANTS,
        "source_numeric_id_contiguity": "not_assumed",
        "expected_individual_values": EXPECTED_INDIVIDUAL_RATINGS,
        "expected_ratings_per_session": list(EXPECTED_RATERS_PER_SESSION),
        "expected_worker_count": EXPECTED_WORKERS,
    }


def _method_contract() -> dict[str, Any]:
    return {
        "method_version": TARGET_MEASUREMENT_METHOD_VERSION,
        "identity_interpretation": "none_source_filename_pairing_only",
        "target": "RecommendHiring_raw_1_to_7",
        "aggregate_source": "Worker_equals_AGGR_expectation_maximization_aggregate",
        "individual_source": "all_non_AGGR_rows",
        "ordinal_reliability": {
            "implementation": "src.data.annotation_audit.krippendorff_alpha",
            "metric": "ordinal",
            "categories": list(range(1, 8)),
            "definition": "coincidence_matrix_marginal_frequency_cumulative_distance",
            "missing_value_rule": "pairwise_exclusion_no_missing_values_expected",
        },
        "session_dispersion": {
            "standard_deviation": "population_denominator_n",
            "interquartile_range": "linear_type_7_q3_minus_q1",
            "range": "maximum_minus_minimum",
            "mean_absolute_deviation": "mean_absolute_distance_from_unweighted_session_mean",
            "reported_scope": "distribution_across_sessions_no_session_identifiers",
        },
        "worker_severity": {
            "definition": "mean_of_rating_minus_unweighted_session_mean",
            "reported_scope": "distribution_across_workers_no_worker_identifiers",
        },
        "aggregate_comparison": {
            "reference": "unweighted_arithmetic_mean_of_individual_ratings_per_session",
            "pearson": "product_moment_with_explicit_zero_variance_status",
            "spearman": "pearson_on_one_based_average_tie_ranks",
            "errors": ["mean_absolute_error", "maximum_absolute_difference"],
        },
        "pre_post": {
            "source": "raw_AGGR_RecommendHiring",
            "contrast": "post_mean_minus_pre_mean",
            "interpretation": "unadjusted_descriptive_only_noncausal",
        },
        "derived_summary_decimal_places": _NUMERIC_DECIMAL_PLACES,
        "canonical_payload_hash": "sha256_compact_sorted_utf8_json_with_terminal_newline",
    }


def _implementation_file_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = {
        "target_measurement_audit_module": Path(__file__),
        "target_measurement_audit_cli": root / "scripts" / "audit_target_measurement.py",
        "mit_interview_parser_module": root / "src" / "data" / "mit_interview.py",
        "krippendorff_alpha_module": root / "src" / "data" / "annotation_audit.py",
    }
    return {
        label: hashlib.sha256(
            _read_stable_regular_file(path, "implementation source", _MAX_SCORE_FILE_BYTES)
        ).hexdigest()
        for label, path in paths.items()
    }


def _decimal_rating(value: str, field: str, row_number: int) -> Decimal:
    if not isinstance(value, str) or _DECIMAL_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} at row {row_number} is not a canonical decimal number")
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"{field} at row {row_number} is not numeric") from error
    if not result.is_finite():
        raise ValueError(f"{field} at row {row_number} must be finite")
    return result


def _mean(values: Sequence[Fraction]) -> Fraction:
    if not values:
        raise ValueError("cannot summarize an empty measurement sequence")
    return sum(values, Fraction()) / len(values)


def _quantile(values: Sequence[Fraction], probability: Fraction) -> Fraction:
    if not values or not 0 <= probability <= 1:
        raise ValueError("quantile requires values and a probability in [0,1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = position.numerator // position.denominator
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _distribution_summary(values: Sequence[Fraction | float]) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty measurement distribution")
    exact = tuple(
        value if isinstance(value, Fraction) else Fraction(str(_canonical_float(value)))
        for value in values
    )
    return {
        "count": len(exact),
        "minimum": _fraction_float(min(exact)),
        "q1": _fraction_float(_quantile(exact, Fraction(1, 4))),
        "median": _fraction_float(_quantile(exact, Fraction(1, 2))),
        "q3": _fraction_float(_quantile(exact, Fraction(3, 4))),
        "maximum": _fraction_float(max(exact)),
        "mean": _fraction_float(_mean(exact)),
    }


def _average_ranks(values: Sequence[Fraction]) -> tuple[Fraction, ...]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [Fraction() for _ in values]
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = Fraction((start + 1) + end, 2)
        for index in order[start:end]:
            ranks[index] = average_rank
        start = end
    return tuple(ranks)


def _correlation(left: Sequence[Fraction], right: Sequence[Fraction]) -> dict[str, Any]:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation requires equal sequences containing at least two values")
    left_mean = _mean(left)
    right_mean = _mean(right)
    left_centered = tuple(value - left_mean for value in left)
    right_centered = tuple(value - right_mean for value in right)
    left_sum_squares = sum((value**2 for value in left_centered), Fraction())
    right_sum_squares = sum((value**2 for value in right_centered), Fraction())
    if left_sum_squares == 0 or right_sum_squares == 0:
        return {"status": "undefined_zero_variance", "value": None}
    numerator = sum((a * b for a, b in zip(left_centered, right_centered, strict=True)), Fraction())
    denominator = math.sqrt(float(left_sum_squares * right_sum_squares))
    value = max(-1.0, min(1.0, float(numerator) / denominator))
    return {"status": "defined", "value": _canonical_float(value)}


def _fraction_float(value: Fraction) -> float:
    return _canonical_float(float(value))


def _canonical_float(value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("measurement summary produced a non-finite value")
    rounded = round(result, _NUMERIC_DECIMAL_PLACES)
    return 0.0 if rounded == 0 else rounded


def _validate_bindings(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "scores_csv",
        "parser_contract_sha256",
        "source_session_roster",
        "implementation_file_sha256",
    }:
        raise ValueError("target-measurement audit bindings schema drifted")
    source = value["scores_csv"]
    if not isinstance(source, Mapping) or set(source) != {"size_bytes", "file_sha256"}:
        raise ValueError("target-measurement source binding schema drifted")
    size = source["size_bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= _MAX_SCORE_FILE_BYTES:
        raise ValueError("target-measurement source size binding is invalid")
    _validated_sha256(source["file_sha256"], "scores CSV SHA-256")
    _validated_sha256(value["parser_contract_sha256"], "parser contract SHA-256")
    if value["parser_contract_sha256"] != _canonical_sha256(_parser_contract()):
        raise ValueError("target-measurement parser contract hash drifted")
    roster = value["source_session_roster"]
    if not isinstance(roster, Mapping) or set(roster) != {
        "binding_scope",
        "source_session_count",
        "source_numeric_pair_count",
        "roster_sha256",
    }:
        raise ValueError("target-measurement source-roster binding schema drifted")
    if (
        roster["binding_scope"]
        not in {"scores_csv_observed_source_roster_only", "explicit_expected_source_roster"}
        or roster["source_session_count"] != EXPECTED_SESSIONS
        or roster["source_numeric_pair_count"] != EXPECTED_PARTICIPANTS
    ):
        raise ValueError("target-measurement source-roster binding contract drifted")
    _validated_sha256(roster["roster_sha256"], "source roster SHA-256")
    implementation = value["implementation_file_sha256"]
    expected_keys = {
        "target_measurement_audit_module",
        "target_measurement_audit_cli",
        "mit_interview_parser_module",
        "krippendorff_alpha_module",
    }
    if not isinstance(implementation, Mapping) or set(implementation) != expected_keys:
        raise ValueError("target-measurement implementation hash schema drifted")
    for label, checksum in implementation.items():
        _validated_sha256(checksum, f"{label} SHA-256")


def _validate_coverage(value: object) -> None:
    expected = {
        "source_numeric_pair_count": EXPECTED_PARTICIPANTS,
        "session_count": EXPECTED_SESSIONS,
        "pre_session_count": EXPECTED_PARTICIPANTS,
        "post_session_count": EXPECTED_PARTICIPANTS,
        "aggregate_rating_count": EXPECTED_SESSIONS,
        "individual_rating_count": EXPECTED_INDIVIDUAL_RATINGS,
        "total_csv_row_count": EXPECTED_SESSIONS + EXPECTED_INDIVIDUAL_RATINGS,
        "distinct_individual_worker_count": EXPECTED_WORKERS,
        "individual_ratings_per_session": {"8": 2, "9": 136},
        "complete_pre_post_pairing_verified": True,
        "complete_aggregate_and_individual_session_coverage_verified": True,
    }
    if value != expected:
        raise ValueError("target-measurement coverage contract drifted")


def _validate_alpha(value: object) -> None:
    fields = {
        "metric",
        "missing_values_excluded_pairwise",
        "pairable_unit_count",
        "pairable_value_count",
        "observed_disagreement",
        "expected_disagreement",
        "status",
        "value",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("target-measurement alpha schema drifted")
    if (
        value["metric"] != "ordinal"
        or value["missing_values_excluded_pairwise"] is not True
        or value["pairable_unit_count"] != EXPECTED_SESSIONS
        or value["pairable_value_count"] != EXPECTED_INDIVIDUAL_RATINGS
    ):
        raise ValueError("target-measurement alpha coverage or method drifted")
    if value["status"] == "defined":
        for field in ("observed_disagreement", "expected_disagreement", "value"):
            _finite_json_number(value[field], f"alpha {field}", enforce_precision=False)
        if value["observed_disagreement"] < 0 or value["expected_disagreement"] <= 0:
            raise ValueError("target-measurement alpha disagreements are invalid")
    elif value["status"] == "undefined_zero_expected_disagreement":
        if value["value"] is not None or value["expected_disagreement"] != 0.0:
            raise ValueError("target-measurement undefined alpha fields are inconsistent")
        _finite_json_number(
            value["observed_disagreement"],
            "alpha observed_disagreement",
            enforce_precision=False,
        )
    else:
        raise ValueError("target-measurement alpha status is invalid")


def _validate_session_dispersion(value: object) -> None:
    fields = {
        "session_count",
        "population_standard_deviation",
        "interquartile_range",
        "range",
        "mean_absolute_deviation_from_session_mean",
    }
    if not isinstance(value, Mapping) or set(value) != fields or value["session_count"] != 138:
        raise ValueError("target-measurement session-dispersion schema drifted")
    for field in fields - {"session_count"}:
        _validate_distribution(value[field], expected_count=138, label=f"session {field}")
        if value[field]["minimum"] < 0:
            raise ValueError("target-measurement session dispersion must be nonnegative")
    if value["population_standard_deviation"]["maximum"] > 3:
        raise ValueError("target-measurement session standard deviation exceeds its scale bound")
    for field in ("interquartile_range", "range", "mean_absolute_deviation_from_session_mean"):
        if value[field]["maximum"] > 6:
            raise ValueError("target-measurement session dispersion exceeds its scale bound")


def _validate_worker_severity(value: object) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {
            "worker_count",
            "ratings_per_worker_distribution",
            "session_centered_mean_residual_distribution",
        }
        or value["worker_count"] != EXPECTED_WORKERS
    ):
        raise ValueError("target-measurement worker-severity schema drifted")
    _validate_distribution(
        value["ratings_per_worker_distribution"],
        expected_count=EXPECTED_WORKERS,
        label="ratings per worker",
    )
    worker_counts = value["ratings_per_worker_distribution"]
    if (
        worker_counts["minimum"] < 1
        or worker_counts["maximum"] > EXPECTED_SESSIONS
        or worker_counts["mean"]
        != _fraction_float(Fraction(EXPECTED_INDIVIDUAL_RATINGS, EXPECTED_WORKERS))
    ):
        raise ValueError("target-measurement ratings-per-worker distribution is invalid")
    severity = value["session_centered_mean_residual_distribution"]
    if not isinstance(severity, Mapping) or set(severity) != {
        "count",
        "minimum",
        "q1",
        "median",
        "q3",
        "maximum",
        "mean",
        "range",
        "negative_count",
        "zero_count",
        "positive_count",
    }:
        raise ValueError("target-measurement worker-severity distribution schema drifted")
    _validate_distribution(
        {
            key: severity[key]
            for key in ("count", "minimum", "q1", "median", "q3", "maximum", "mean")
        },
        expected_count=EXPECTED_WORKERS,
        label="worker severity",
    )
    _finite_json_number(severity["range"], "worker severity range")
    if (
        not -6 <= severity["minimum"] <= severity["maximum"] <= 6
        or not 0 <= severity["range"] <= 6
        or not math.isclose(
            severity["range"],
            severity["maximum"] - severity["minimum"],
            rel_tol=0.0,
            abs_tol=2 * 10**-_NUMERIC_DECIMAL_PLACES,
        )
    ):
        raise ValueError("target-measurement worker-severity range is invalid")
    sign_counts = [severity[field] for field in ("negative_count", "zero_count", "positive_count")]
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in sign_counts
    ):
        raise ValueError("target-measurement worker-severity sign counts are invalid")
    if sum(sign_counts) != EXPECTED_WORKERS:
        raise ValueError("target-measurement worker-severity sign counts do not cover workers")


def _validate_aggregate_comparison(value: object) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {
            "session_count",
            "pearson_correlation",
            "spearman_correlation",
            "mean_absolute_error",
            "maximum_absolute_difference",
        }
        or value["session_count"] != EXPECTED_SESSIONS
    ):
        raise ValueError("target-measurement aggregate-comparison schema drifted")
    _validate_correlation(value["pearson_correlation"], "Pearson")
    _validate_correlation(value["spearman_correlation"], "Spearman")
    _finite_json_number(value["mean_absolute_error"], "mean absolute error")
    _finite_json_number(value["maximum_absolute_difference"], "maximum absolute difference")
    if value["mean_absolute_error"] < 0 or value["maximum_absolute_difference"] < 0:
        raise ValueError("target-measurement aggregate-comparison errors must be nonnegative")
    if value["mean_absolute_error"] > 6 or value["maximum_absolute_difference"] > 6:
        raise ValueError("target-measurement aggregate-comparison errors exceed the scale")


def _validate_pre_post(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "interpretation",
        "value_source",
        "pre",
        "post",
        "post_minus_pre_mean",
    }:
        raise ValueError("target-measurement pre/post schema drifted")
    if (
        value["interpretation"] != "unadjusted_descriptive_only_noncausal"
        or value["value_source"] != "raw_EM_aggregate_RecommendHiring_on_1_to_7_scale"
    ):
        raise ValueError("target-measurement pre/post noncausal boundary drifted")
    for phase in ("pre", "post"):
        phase_value = value[phase]
        if not isinstance(phase_value, Mapping) or set(phase_value) != {"count", "mean"}:
            raise ValueError("target-measurement pre/post phase schema drifted")
        if phase_value["count"] != EXPECTED_PARTICIPANTS:
            raise ValueError("target-measurement pre/post phase coverage drifted")
        mean = _finite_json_number(phase_value["mean"], f"{phase} mean")
        if not 1 <= mean <= 7:
            raise ValueError("target-measurement pre/post mean is outside 1--7")
    difference = _finite_json_number(value["post_minus_pre_mean"], "post-minus-pre mean")
    if not -6 <= difference <= 6:
        raise ValueError("target-measurement post-minus-pre mean exceeds the scale")


def _validate_distribution(value: object, *, expected_count: int, label: str) -> None:
    fields = {"count", "minimum", "q1", "median", "q3", "maximum", "mean"}
    if not isinstance(value, Mapping) or set(value) != fields or value["count"] != expected_count:
        raise ValueError(f"{label} distribution schema or count drifted")
    numbers = [
        _finite_json_number(value[field], f"{label} {field}") for field in fields - {"count"}
    ]
    minimum = value["minimum"]
    q1 = value["q1"]
    median = value["median"]
    q3 = value["q3"]
    maximum = value["maximum"]
    if not minimum <= q1 <= median <= q3 <= maximum:
        raise ValueError(f"{label} distribution order is invalid")
    if not numbers:
        raise ValueError(f"{label} distribution is empty")


def _validate_correlation(value: object, label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != {"status", "value"}:
        raise ValueError(f"target-measurement {label} schema drifted")
    if value["status"] == "defined":
        number = _finite_json_number(value["value"], f"{label} correlation")
        if not -1 <= number <= 1:
            raise ValueError(f"target-measurement {label} correlation is outside [-1,1]")
    elif value != {"status": "undefined_zero_variance", "value": None}:
        raise ValueError(f"target-measurement {label} status is invalid")


def _finite_json_number(value: object, label: str, *, enforce_precision: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a canonical finite JSON float")
    if enforce_precision and _canonical_float(value) != value:
        raise ValueError(f"{label} exceeds the frozen decimal precision")
    return value


def _validated_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be 64 lowercase hexadecimal digits")
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_stable_regular_file(path: Path, label: str, maximum_bytes: int) -> bytes:
    if not isinstance(path, Path):
        path = Path(path)
    if path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cannot open a safe {label}") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > maximum_bytes
        ):
            raise ValueError(f"{label} must be a non-empty regular file within the size limit")
        chunks: list[bytes] = []
        total = 0
        while block := os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - total)):
            chunks.append(block)
            total += len(block)
            if total > maximum_bytes:
                raise ValueError(f"{label} exceeds the fixed size limit")
        after = os.fstat(descriptor)
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if identity != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError(f"{label} changed while it was being read")
    finally:
        os.close(descriptor)
    try:
        final = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"{label} changed while it was being read") from error
    if stat.S_ISLNK(final.st_mode) or (final.st_dev, final.st_ino) != (
        before.st_dev,
        before.st_ino,
    ):
        raise ValueError(f"{label} changed while it was being read")
    return b"".join(chunks)


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
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


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _source_roster_sha256(source_session_ids: Sequence[str]) -> str:
    return _canonical_sha256({"source_session_ids": list(source_session_ids)})
