"""Deterministic aggregate agreement audit for restricted perception annotations.

The report produced here contains counts and reliability summaries only.  Participant IDs,
rater IDs, session IDs, crop references, row-level ratings, outcomes, predictions and proxy
assignments never enter the returned payload.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from .annotation_stimuli import STIMULUS_SCHEMA_VERSION, load_stimulus_manifest
from .protected_annotations import (
    ANNOTATION_SCHEMA_VERSION,
    CODEBOOK_SCHEMA_VERSION,
    CONSENSUS_METHOD,
    PERCEIVED_FIELDS,
    SKIN_TONE_CONSENSUS_VALUES,
    RaterAnnotation,
    build_consensus_annotations,
    load_annotation_codebook,
    load_rater_annotations,
    require_exact_participant_coverage,
    require_rater_participant_coverage,
)

AUDIT_REPORT_SCHEMA_VERSION = "mit-annotation-audit-report-2"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$", re.ASCII)
_PROHIBITED_INPUTS = [
    "outcomes",
    "predictions",
    "transcripts",
    "phase_labels",
    "proxy_assignments",
]
_TOP_FIELDS = {
    "schema_version",
    "aggregation_scope",
    "bindings",
    "coverage",
    "stimulus_quality",
    "fields",
    "consensus_method",
    "prohibited_inputs",
    "report_payload_sha256",
}


def krippendorff_alpha(
    units: Iterable[Iterable[str | int | None]],
    *,
    categories: Sequence[str | int],
    metric: Literal["nominal", "ordinal"],
) -> dict[str, Any]:
    """Compute coincidence-matrix Krippendorff alpha with pairwise missing exclusion.

    Ordinal distances use Krippendorff's marginal-frequency cumulative-distance definition.
    Units containing fewer than two non-missing observations cannot form coincidences and are
    excluded.  Undefined cases are returned explicitly with a JSON-safe ``None`` value.
    """
    ordered_categories = _validated_categories(categories)
    if metric not in {"nominal", "ordinal"}:
        raise ValueError("Krippendorff metric must be nominal or ordinal")
    category_index = {category: index for index, category in enumerate(ordered_categories)}
    pairable_units: list[tuple[str | int, ...]] = []
    for raw_unit in units:
        if isinstance(raw_unit, (str, bytes)):
            raise TypeError("each reliability unit must be an iterable of observations")
        values = tuple(value for value in raw_unit if value is not None)
        if any(
            isinstance(value, bool) or not isinstance(value, (str, int)) for value in values
        ):
            raise ValueError("reliability observation has an invalid type")
        unexpected = [value for value in values if value not in category_index]
        if unexpected:
            raise ValueError("reliability observation is not present in the frozen categories")
        if len(values) >= 2:
            pairable_units.append(values)

    marginal = Counter(value for unit in pairable_units for value in unit)
    pairable_value_count = sum(marginal.values())
    if pairable_value_count < 2:
        return _undefined_alpha(
            metric,
            pairable_units=len(pairable_units),
            pairable_values=pairable_value_count,
            reason="undefined_no_pairable_values",
        )

    observed: dict[tuple[str | int, str | int], Fraction] = defaultdict(Fraction)
    for unit in pairable_units:
        counts = Counter(unit)
        denominator = len(unit) - 1
        for left_index, left in enumerate(ordered_categories):
            for right in ordered_categories[left_index + 1 :]:
                observed[(left, right)] += Fraction(
                    counts[left] * counts[right], denominator
                )

    observed_weighted = Fraction()
    expected_weighted = Fraction()
    for left_index, left in enumerate(ordered_categories):
        for right_index in range(left_index + 1, len(ordered_categories)):
            right = ordered_categories[right_index]
            distance = _krippendorff_distance_squared(
                ordered_categories,
                marginal,
                left_index,
                right_index,
                metric,
            )
            observed_weighted += observed[(left, right)] * distance
            expected_weighted += marginal[left] * marginal[right] * distance

    if expected_weighted == 0:
        return _undefined_alpha(
            metric,
            pairable_units=len(pairable_units),
            pairable_values=pairable_value_count,
            reason="undefined_zero_expected_disagreement",
            observed_disagreement=_fraction_float(
                Fraction(2, pairable_value_count) * observed_weighted
            ),
            expected_disagreement=0.0,
        )

    observed_disagreement = Fraction(2, pairable_value_count) * observed_weighted
    expected_disagreement = Fraction(
        2, pairable_value_count * (pairable_value_count - 1)
    ) * expected_weighted
    alpha = Fraction(1) - observed_disagreement / expected_disagreement
    return {
        "metric": metric,
        "missing_values_excluded_pairwise": True,
        "pairable_unit_count": len(pairable_units),
        "pairable_value_count": pairable_value_count,
        "observed_disagreement": _fraction_float(observed_disagreement),
        "expected_disagreement": _fraction_float(expected_disagreement),
        "status": "defined",
        "value": _fraction_float(alpha),
    }


def build_annotation_audit_report(
    *,
    codebook_path: str | Path,
    rater_annotations_path: str | Path,
    stimulus_manifest_path: str | Path,
) -> dict[str, Any]:
    """Build a checksum-bound aggregate audit from approved, restricted inputs."""
    codebook_file = _regular_file(codebook_path, "annotation codebook")
    rater_file = _regular_file(rater_annotations_path, "raw rater annotation")
    stimulus_file = _regular_file(stimulus_manifest_path, "stimulus manifest")

    codebook = load_annotation_codebook(codebook_file, require_approved=True)
    annotations = load_rater_annotations(rater_file, codebook=codebook)
    stimulus = load_stimulus_manifest(stimulus_file)

    assessable_ids = tuple(
        row["participant_id"]
        for row in stimulus["participants"]
        if row["status"] == "assessable"
    )
    not_assessable_ids = {
        row["participant_id"]
        for row in stimulus["participants"]
        if row["status"] == "not_assessable"
    }
    if not assessable_ids:
        raise ValueError("annotation audit requires at least one assessable stimulus participant")
    if any(row.participant_id in not_assessable_ids for row in annotations):
        raise ValueError("not-assessable participants must not have rater annotations")
    require_rater_participant_coverage(assessable_ids, annotations, minimum_raters=3)

    consensus = build_consensus_annotations(annotations, minimum_raters=3)
    require_exact_participant_coverage(assessable_ids, consensus)
    annotations_by_participant: dict[str, list[RaterAnnotation]] = defaultdict(list)
    for row in annotations:
        annotations_by_participant[row.participant_id].append(row)
    rater_counts = [len(annotations_by_participant[value]) for value in assessable_ids]

    field_specs: tuple[
        tuple[str, Literal["nominal", "ordinal"], tuple[str | int, ...]], ...
    ] = (
        (
            "perceived_gender_presentation",
            "nominal",
            tuple(codebook.gender_categories),
        ),
        (
            "perceived_racialised_appearance",
            "nominal",
            tuple(codebook.racialised_appearance_categories),
        ),
        ("perceived_skin_tone", "ordinal", tuple(codebook.skin_tone_values)),
    )
    fields = [
        _field_report(
            field,
            metric,
            categories,
            annotations_by_participant,
            consensus,
        )
        for field, metric, categories in field_specs
    ]

    offsets, sensitivity_count, failure_counts = _stimulus_summary(stimulus)
    unsigned: dict[str, Any] = {
        "schema_version": AUDIT_REPORT_SCHEMA_VERSION,
        "aggregation_scope": "aggregate_only_no_participant_rater_session_or_crop_identifiers",
        "bindings": {
            "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
            "codebook_schema_version": CODEBOOK_SCHEMA_VERSION,
            "codebook_version": codebook.codebook_version,
            "codebook_payload_sha256": codebook.sha256,
            "codebook_file_sha256": _file_sha256(codebook_file),
            "raw_rater_file_sha256": _file_sha256(rater_file),
            "stimulus_manifest_schema_version": STIMULUS_SCHEMA_VERSION,
            "stimulus_manifest_payload_sha256": _canonical_sha256(stimulus),
            "stimulus_manifest_file_sha256": _file_sha256(stimulus_file),
        },
        "coverage": {
            "stimulus_participant_count": len(stimulus["participants"]),
            "assessable_participant_count": len(assessable_ids),
            "not_assessable_participant_count": len(not_assessable_ids),
            "raw_rater_row_count": len(annotations),
            "distinct_rater_count": len({row.rater_id for row in annotations}),
            "minimum_independent_raters_required": 3,
            "minimum_raters_per_assessable_participant": min(rater_counts),
            "maximum_raters_per_assessable_participant": max(rater_counts),
            "exact_assessable_coverage_verified": True,
            "not_assessable_rating_row_count": 0,
        },
        "stimulus_quality": {
            "assessable_crop_offset_count": len(offsets),
            "absolute_offset_seconds": _offset_distribution(offsets),
            "participants_with_all_six_offsets_at_most_30_seconds": sensitivity_count,
            "not_assessable_failure_reason_counts": failure_counts,
        },
        "fields": fields,
        "consensus_method": CONSENSUS_METHOD,
        "prohibited_inputs": list(_PROHIBITED_INPUTS),
    }
    result = {
        **unsigned,
        "report_payload_sha256": _canonical_sha256(unsigned),
    }
    _validate_audit_report_payload(result)
    return result


def canonical_annotation_audit_json(payload: Mapping[str, Any]) -> str:
    """Return stable human-readable JSON after exact report validation."""
    _validate_audit_report_payload(payload)
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def load_annotation_audit_report(path: str | Path) -> dict[str, Any]:
    """Load a duplicate-key-safe report and validate its embedded payload checksum."""
    source = _regular_file(path, "annotation audit report")
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("cannot read a valid annotation audit report") from error
    _validate_audit_report_payload(payload)
    return dict(payload)


def validate_annotation_audit_report(
    report_path: str | Path,
    *,
    codebook_path: str | Path,
    rater_annotations_path: str | Path,
    stimulus_manifest_path: str | Path,
) -> dict[str, Any]:
    """Rebuild a report from its bound inputs and reject any contextual drift."""
    observed = load_annotation_audit_report(report_path)
    expected = build_annotation_audit_report(
        codebook_path=codebook_path,
        rater_annotations_path=rater_annotations_path,
        stimulus_manifest_path=stimulus_manifest_path,
    )
    if canonical_annotation_audit_json(observed) != canonical_annotation_audit_json(expected):
        raise ValueError("annotation audit report does not match its checksum-bound inputs")
    return observed


def _field_report(
    field: str,
    metric: Literal["nominal", "ordinal"],
    categories: tuple[str | int, ...],
    annotations_by_participant: Mapping[str, Sequence[RaterAnnotation]],
    consensus: Sequence[Any],
) -> dict[str, Any]:
    units = [
        [getattr(row, field) for row in rows]
        for _, rows in sorted(annotations_by_participant.items(), key=lambda item: int(item[0]))
    ]
    flat = [value for unit in units for value in unit]
    valid = [value for value in flat if value is not None]
    raw_counts = Counter(valid)
    agreeing_pairs = 0
    total_pairs = 0
    disagreements: Counter[tuple[str | int, str | int]] = Counter()
    order = {value: index for index, value in enumerate(categories)}
    for unit in units:
        values = [value for value in unit if value is not None]
        for left_index, left in enumerate(values):
            for right in values[left_index + 1 :]:
                total_pairs += 1
                if left == right:
                    agreeing_pairs += 1
                else:
                    first, second = sorted((left, right), key=order.__getitem__)
                    disagreements[(first, second)] += 1

    consensus_values = [getattr(row, field) for row in consensus]
    if field == "perceived_skin_tone":
        consensus_categories: tuple[str | int | float, ...] = SKIN_TONE_CONSENSUS_VALUES
    else:
        consensus_categories = categories
    consensus_counts = Counter(value for value in consensus_values if value is not None)
    return {
        "field": field,
        "measurement_level": metric,
        "categories": list(categories),
        "observations": {
            "total_count": len(flat),
            "valid_count": len(valid),
            "missing_count": len(flat) - len(valid),
            "category_counts": _ordered_counts(categories, raw_counts),
        },
        "pairwise_exact_agreement": {
            "agreeing_pair_count": agreeing_pairs,
            "total_pair_count": total_pairs,
            "rate": agreeing_pairs / total_pairs if total_pairs else None,
        },
        "unordered_disagreement_counts": [
            {
                "category_a": left,
                "category_b": right,
                "pair_count": disagreements[(left, right)],
            }
            for left_index, left in enumerate(categories)
            for right in categories[left_index + 1 :]
        ],
        "krippendorff_alpha": krippendorff_alpha(
            units,
            categories=categories,
            metric=metric,
        ),
        "consensus": {
            "known_count": len(consensus_values)
            - sum(value is None for value in consensus_values),
            "unknown_count": sum(value is None for value in consensus_values),
            "category_counts": _ordered_counts(consensus_categories, consensus_counts),
        },
    }


def _stimulus_summary(
    stimulus: Mapping[str, Any],
) -> tuple[list[Fraction], int, list[dict[str, Any]]]:
    offsets: list[Fraction] = []
    sensitivity_count = 0
    failure_counts: Counter[str] = Counter()
    for packet in stimulus["participants"]:
        if packet["status"] == "assessable":
            packet_offsets = [
                Fraction(row["absolute_offset_seconds"]) for row in packet["stimuli"]
            ]
            if len(packet_offsets) != 6:
                raise ValueError("assessable annotation packet must contain six offsets")
            offsets.extend(packet_offsets)
            if all(value <= 30 for value in packet_offsets):
                sensitivity_count += 1
            continue
        for reason in packet["assessment_reasons"]:
            parts = reason.split(":")
            if (
                len(parts) != 2
                or parts[0] not in {f"P{packet['participant_id']}", f"PP{packet['participant_id']}"}
                or re.fullmatch(r"[a-z0-9_]+", parts[1], re.ASCII) is None
            ):
                raise ValueError("stimulus assessment reason is not an aggregate-safe failure code")
            failure_counts[parts[1]] += 1
    if not offsets:
        raise ValueError("annotation audit requires assessable crop offsets")
    return (
        offsets,
        sensitivity_count,
        [
            {"reason_code": reason, "count": failure_counts[reason]}
            for reason in sorted(failure_counts)
        ],
    )


def _offset_distribution(values: Sequence[Fraction]) -> dict[str, str]:
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    return {
        "minimum": _fraction_text(ordered[0]),
        "median": _fraction_text(median),
        "maximum": _fraction_text(ordered[-1]),
    }


def _krippendorff_distance_squared(
    categories: tuple[str | int, ...],
    marginal: Counter[str | int],
    left_index: int,
    right_index: int,
    metric: Literal["nominal", "ordinal"],
) -> Fraction:
    if metric == "nominal":
        return Fraction(1)
    left = categories[left_index]
    right = categories[right_index]
    cumulative = sum(marginal[value] for value in categories[left_index : right_index + 1])
    midpoint_endpoints = Fraction(marginal[left] + marginal[right], 2)
    return (Fraction(cumulative) - midpoint_endpoints) ** 2


def _undefined_alpha(
    metric: str,
    *,
    pairable_units: int,
    pairable_values: int,
    reason: str,
    observed_disagreement: float | None = None,
    expected_disagreement: float | None = None,
) -> dict[str, Any]:
    return {
        "metric": metric,
        "missing_values_excluded_pairwise": True,
        "pairable_unit_count": pairable_units,
        "pairable_value_count": pairable_values,
        "observed_disagreement": observed_disagreement,
        "expected_disagreement": expected_disagreement,
        "status": reason,
        "value": None,
    }


def _validated_categories(categories: Sequence[str | int]) -> tuple[str | int, ...]:
    if isinstance(categories, (str, bytes)):
        raise TypeError("Krippendorff categories must be an ordered sequence")
    values = tuple(categories)
    if len(values) < 2 or len(set(values)) != len(values):
        raise ValueError("Krippendorff categories must be unique and contain at least two values")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (str, int))
        or (isinstance(value, str) and (not value or value != value.strip() or "\0" in value))
        for value in values
    ):
        raise ValueError("Krippendorff categories contain an invalid value")
    return values


def _ordered_counts(
    categories: Sequence[str | int | float], counts: Counter[Any]
) -> list[dict[str, Any]]:
    return [{"category": value, "count": counts[value]} for value in categories]


def _validate_audit_report_payload(payload: object) -> None:
    if not isinstance(payload, Mapping) or set(payload) != _TOP_FIELDS:
        raise ValueError("annotation audit report must have the exact versioned top-level schema")
    if payload["schema_version"] != AUDIT_REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported annotation audit report schema_version")
    if (
        payload["aggregation_scope"]
        != "aggregate_only_no_participant_rater_session_or_crop_identifiers"
        or payload["consensus_method"] != CONSENSUS_METHOD
        or payload["prohibited_inputs"] != _PROHIBITED_INPUTS
    ):
        raise ValueError("annotation audit aggregate/firewall contract drifted")
    checksum = payload["report_payload_sha256"]
    if not isinstance(checksum, str) or _SHA256_PATTERN.fullmatch(checksum) is None:
        raise ValueError("annotation audit report checksum is invalid")
    unsigned = {key: value for key, value in payload.items() if key != "report_payload_sha256"}
    if _canonical_sha256(unsigned) != checksum:
        raise ValueError("annotation audit report payload checksum failed")

    bindings = payload["bindings"]
    expected_bindings = {
        "annotation_schema_version",
        "codebook_schema_version",
        "codebook_version",
        "codebook_payload_sha256",
        "codebook_file_sha256",
        "raw_rater_file_sha256",
        "stimulus_manifest_schema_version",
        "stimulus_manifest_payload_sha256",
        "stimulus_manifest_file_sha256",
    }
    if not isinstance(bindings, Mapping) or set(bindings) != expected_bindings:
        raise ValueError("annotation audit report bindings are invalid")
    if (
        bindings["annotation_schema_version"] != ANNOTATION_SCHEMA_VERSION
        or bindings["codebook_schema_version"] != CODEBOOK_SCHEMA_VERSION
        or bindings["stimulus_manifest_schema_version"] != STIMULUS_SCHEMA_VERSION
    ):
        raise ValueError("annotation audit report schema binding drifted")
    for field in (
        "codebook_payload_sha256",
        "codebook_file_sha256",
        "raw_rater_file_sha256",
        "stimulus_manifest_payload_sha256",
        "stimulus_manifest_file_sha256",
    ):
        if (
            not isinstance(bindings[field], str)
            or _SHA256_PATTERN.fullmatch(bindings[field]) is None
        ):
            raise ValueError("annotation audit report input checksum is invalid")
    if (
        not isinstance(bindings["codebook_version"], str)
        or not bindings["codebook_version"]
        or bindings["codebook_version"] != bindings["codebook_version"].strip()
        or "\0" in bindings["codebook_version"]
    ):
        raise ValueError("annotation audit codebook-version binding is invalid")

    coverage = payload["coverage"]
    expected_coverage = {
        "stimulus_participant_count",
        "assessable_participant_count",
        "not_assessable_participant_count",
        "raw_rater_row_count",
        "distinct_rater_count",
        "minimum_independent_raters_required",
        "minimum_raters_per_assessable_participant",
        "maximum_raters_per_assessable_participant",
        "exact_assessable_coverage_verified",
        "not_assessable_rating_row_count",
    }
    if not isinstance(coverage, Mapping) or set(coverage) != expected_coverage:
        raise ValueError("annotation audit report coverage schema is invalid")
    integer_fields = expected_coverage - {"exact_assessable_coverage_verified"}
    if any(
        isinstance(coverage[field], bool)
        or not isinstance(coverage[field], int)
        or coverage[field] < 0
        for field in integer_fields
    ):
        raise ValueError("annotation audit report coverage values are invalid")
    if (
        coverage["minimum_independent_raters_required"] != 3
        or coverage["minimum_raters_per_assessable_participant"] < 3
        or coverage["maximum_raters_per_assessable_participant"]
        < coverage["minimum_raters_per_assessable_participant"]
        or coverage["stimulus_participant_count"]
        != coverage["assessable_participant_count"]
        + coverage["not_assessable_participant_count"]
        or coverage["exact_assessable_coverage_verified"] is not True
        or coverage["not_assessable_rating_row_count"] != 0
    ):
        raise ValueError("annotation audit report coverage contract failed")

    _validate_stimulus_quality(payload["stimulus_quality"], coverage)
    fields = payload["fields"]
    if (
        not isinstance(fields, list)
        or any(not isinstance(row, Mapping) for row in fields)
        or [row.get("field") for row in fields] != list(PERCEIVED_FIELDS)
    ):
        raise ValueError("annotation audit fields must follow the frozen construct order")
    for row in fields:
        _validate_field_report(row, coverage["assessable_participant_count"])
    _reject_identifier_rows(payload)


def _validate_stimulus_quality(value: object, coverage: Mapping[str, int]) -> None:
    expected = {
        "assessable_crop_offset_count",
        "absolute_offset_seconds",
        "participants_with_all_six_offsets_at_most_30_seconds",
        "not_assessable_failure_reason_counts",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("annotation audit stimulus-quality schema is invalid")
    if value["assessable_crop_offset_count"] != 6 * coverage["assessable_participant_count"]:
        raise ValueError("annotation audit crop-offset coverage is invalid")
    sensitivity = value["participants_with_all_six_offsets_at_most_30_seconds"]
    if (
        isinstance(sensitivity, bool)
        or not isinstance(sensitivity, int)
        or not 0 <= sensitivity <= coverage["assessable_participant_count"]
    ):
        raise ValueError("annotation audit offset-sensitivity count is invalid")
    distribution = value["absolute_offset_seconds"]
    if not isinstance(distribution, Mapping) or set(distribution) != {
        "minimum",
        "median",
        "maximum",
    }:
        raise ValueError("annotation audit offset distribution is invalid")
    parsed_offsets = [
        _nonnegative_fraction(distribution[field])
        for field in ("minimum", "median", "maximum")
    ]
    if not parsed_offsets[0] <= parsed_offsets[1] <= parsed_offsets[2]:
        raise ValueError("annotation audit offset order is invalid")
    reasons = value["not_assessable_failure_reason_counts"]
    if not isinstance(reasons, list):
        raise ValueError("annotation audit failure counts must be a list")
    observed_reason_codes: list[str] = []
    for row in reasons:
        if not isinstance(row, Mapping) or set(row) != {"reason_code", "count"}:
            raise ValueError("annotation audit failure count row is invalid")
        if (
            not isinstance(row["reason_code"], str)
            or re.fullmatch(r"[a-z0-9_]+", row["reason_code"], re.ASCII) is None
            or isinstance(row["count"], bool)
            or not isinstance(row["count"], int)
            or row["count"] <= 0
        ):
            raise ValueError("annotation audit failure count value is invalid")
        observed_reason_codes.append(row["reason_code"])
    if observed_reason_codes != sorted(set(observed_reason_codes)):
        raise ValueError("annotation audit failure reason ordering is invalid")


def _validate_field_report(value: object, participant_count: int) -> None:
    expected = {
        "field",
        "measurement_level",
        "categories",
        "observations",
        "pairwise_exact_agreement",
        "unordered_disagreement_counts",
        "krippendorff_alpha",
        "consensus",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("annotation audit field report schema is invalid")
    categories = _validated_categories(value["categories"])
    metric = value["measurement_level"]
    if metric not in {"nominal", "ordinal"}:
        raise ValueError("annotation audit measurement level is invalid")
    if (value["field"] == "perceived_skin_tone") != (metric == "ordinal"):
        raise ValueError("annotation audit field metric drifted")

    observations = value["observations"]
    if not isinstance(observations, Mapping) or set(observations) != {
        "total_count",
        "valid_count",
        "missing_count",
        "category_counts",
    }:
        raise ValueError("annotation audit observation schema is invalid")
    _validate_nonnegative_counts(observations, ("total_count", "valid_count", "missing_count"))
    if observations["total_count"] != observations["valid_count"] + observations["missing_count"]:
        raise ValueError("annotation audit observation totals are inconsistent")
    raw_counts = _validate_ordered_counts(observations["category_counts"], categories)
    if sum(raw_counts) != observations["valid_count"]:
        raise ValueError("annotation audit category counts do not equal valid observations")

    agreement = value["pairwise_exact_agreement"]
    if not isinstance(agreement, Mapping) or set(agreement) != {
        "agreeing_pair_count",
        "total_pair_count",
        "rate",
    }:
        raise ValueError("annotation audit pairwise-agreement schema is invalid")
    _validate_nonnegative_counts(agreement, ("agreeing_pair_count", "total_pair_count"))
    if agreement["agreeing_pair_count"] > agreement["total_pair_count"]:
        raise ValueError("annotation audit agreeing-pair count is invalid")
    expected_rate = (
        agreement["agreeing_pair_count"] / agreement["total_pair_count"]
        if agreement["total_pair_count"]
        else None
    )
    if agreement["rate"] != expected_rate:
        raise ValueError("annotation audit pairwise-agreement rate is invalid")

    disagreements = value["unordered_disagreement_counts"]
    expected_pairs = [
        (left, right)
        for index, left in enumerate(categories)
        for right in categories[index + 1 :]
    ]
    if not isinstance(disagreements, list) or len(disagreements) != len(expected_pairs):
        raise ValueError("annotation audit disagreement matrix is incomplete")
    disagreement_total = 0
    for row, pair in zip(disagreements, expected_pairs, strict=True):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"category_a", "category_b", "pair_count"}
            or (row["category_a"], row["category_b"]) != pair
            or isinstance(row["pair_count"], bool)
            or not isinstance(row["pair_count"], int)
            or row["pair_count"] < 0
        ):
            raise ValueError("annotation audit disagreement row is invalid")
        disagreement_total += row["pair_count"]
    if disagreement_total + agreement["agreeing_pair_count"] != agreement["total_pair_count"]:
        raise ValueError("annotation audit pairwise counts are inconsistent")
    _validate_alpha(value["krippendorff_alpha"], metric)

    consensus = value["consensus"]
    if not isinstance(consensus, Mapping) or set(consensus) != {
        "known_count",
        "unknown_count",
        "category_counts",
    }:
        raise ValueError("annotation audit consensus schema is invalid")
    _validate_nonnegative_counts(consensus, ("known_count", "unknown_count"))
    if consensus["known_count"] + consensus["unknown_count"] != participant_count:
        raise ValueError("annotation audit consensus coverage is invalid")
    consensus_counts = consensus["category_counts"]
    expected_consensus_categories: Sequence[str | int | float]
    if value["field"] == "perceived_skin_tone":
        expected_consensus_categories = SKIN_TONE_CONSENSUS_VALUES
    else:
        expected_consensus_categories = categories
    validated_consensus_counts = _validate_ordered_counts(
        consensus_counts, expected_consensus_categories
    )
    if sum(validated_consensus_counts) != consensus["known_count"]:
        raise ValueError("annotation audit consensus counts do not equal known count")


def _validate_alpha(value: object, metric: str) -> None:
    expected = {
        "metric",
        "missing_values_excluded_pairwise",
        "pairable_unit_count",
        "pairable_value_count",
        "observed_disagreement",
        "expected_disagreement",
        "status",
        "value",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("annotation audit Krippendorff-alpha schema is invalid")
    if value["metric"] != metric or value["missing_values_excluded_pairwise"] is not True:
        raise ValueError("annotation audit Krippendorff-alpha metric contract drifted")
    _validate_nonnegative_counts(value, ("pairable_unit_count", "pairable_value_count"))
    status = value["status"]
    if status not in {
        "defined",
        "undefined_no_pairable_values",
        "undefined_zero_expected_disagreement",
    }:
        raise ValueError("annotation audit Krippendorff-alpha status is invalid")
    for field in ("observed_disagreement", "expected_disagreement", "value"):
        numeric = value[field]
        if numeric is not None and (
            isinstance(numeric, bool)
            or not isinstance(numeric, (int, float))
            or not math.isfinite(float(numeric))
        ):
            raise ValueError("annotation audit Krippendorff-alpha number is invalid")
    if (status == "defined") != (value["value"] is not None):
        raise ValueError("annotation audit Krippendorff-alpha defined status is inconsistent")


def _validate_ordered_counts(value: object, categories: Sequence[Any]) -> list[int]:
    if not isinstance(value, list) or len(value) != len(categories):
        raise ValueError("annotation audit ordered category counts are invalid")
    result: list[int] = []
    for row, category in zip(value, categories, strict=True):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"category", "count"}
            or row["category"] != category
            or isinstance(row["count"], bool)
            or not isinstance(row["count"], int)
            or row["count"] < 0
        ):
            raise ValueError("annotation audit category-count row is invalid")
        result.append(row["count"])
    return result


def _validate_nonnegative_counts(value: Mapping[str, Any], fields: Sequence[str]) -> None:
    if any(
        isinstance(value[field], bool)
        or not isinstance(value[field], int)
        or value[field] < 0
        for field in fields
    ):
        raise ValueError("annotation audit count is invalid")


def _reject_identifier_rows(payload: Mapping[str, Any]) -> None:
    forbidden_keys = {
        "participant_id",
        "rater_id",
        "source_session_id",
        "crop_reference",
        "packet_id",
        "stimulus_id",
        "row",
        "rating",
    }

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if key in forbidden_keys:
                    raise ValueError("annotation audit report contains a row-level identifier")
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(payload)


def _regular_file(value: str | Path, label: str) -> Path:
    path = Path(value)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _nonnegative_fraction(value: object) -> Fraction:
    if not isinstance(value, str):
        raise ValueError("annotation audit decimal must be canonical text")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("annotation audit decimal is invalid") from error
    if result < 0 or _fraction_text(result) != value:
        raise ValueError("annotation audit decimal must be non-negative canonical text")
    return result


def _fraction_text(value: Fraction) -> str:
    denominator = value.denominator
    power_two = 0
    power_five = 0
    while denominator % 2 == 0:
        denominator //= 2
        power_two += 1
    while denominator % 5 == 0:
        denominator //= 5
        power_five += 1
    if denominator != 1:
        return format(float(value), ".15g")
    places = max(power_two, power_five)
    scaled = value.numerator * (2 ** (places - power_two)) * (5 ** (places - power_five))
    sign = "-" if scaled < 0 else ""
    digits = str(abs(scaled)).rjust(places + 1, "0")
    if places == 0:
        return sign + digits
    result = f"{sign}{digits[:-places]}.{digits[-places:]}".rstrip("0").rstrip(".")
    return result or "0"


def _fraction_float(value: Fraction) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Krippendorff calculation produced a non-finite value")
    return result
