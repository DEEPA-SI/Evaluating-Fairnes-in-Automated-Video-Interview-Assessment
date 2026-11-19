"""Strict participant-level perception annotations kept behind the audit firewall."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .mit_interview import canonical_participant_id

ANNOTATION_SCHEMA_VERSION = "mit-participant-audit-2"
CODEBOOK_SCHEMA_VERSION = "mit-annotation-codebook-2"
SYNTHETIC_CODEBOOK_VERSION = "synthetic-v1"
SKIN_TONE_VALUES = tuple(range(1, 6))
SKIN_TONE_CONSENSUS_VALUES = tuple(value / 2 for value in range(2, 11))
CONSENSUS_METHOD = "nominal_strict_majority;skin_median_range_le_1"

RATER_COLUMNS = (
    "participant_id",
    "rater_id",
    "perceived_gender_presentation",
    "perceived_racialised_appearance",
    "perceived_skin_tone",
    "cannot_determine",
    "gender_presentation_confidence",
    "racialised_appearance_confidence",
    "skin_tone_confidence",
    "codebook_version",
)
RATER_RESPONSE_COLUMNS = (
    "rater_manifest_id",
    "packet_id",
    "perceived_gender_presentation",
    "perceived_racialised_appearance",
    "perceived_skin_tone",
    "cannot_determine",
    "gender_presentation_confidence",
    "racialised_appearance_confidence",
    "skin_tone_confidence",
    "codebook_version",
)
CONSENSUS_COLUMNS = (
    "participant_id",
    "perceived_gender_presentation",
    "perceived_racialised_appearance",
    "perceived_skin_tone",
    "cannot_determine",
    "rater_count",
    "consensus_method",
    "codebook_version",
)
SELF_REPORTED_COLUMNS = (
    "participant_id",
    "self_reported_gender",
    "self_reported_ethnicity",
    "source_provenance",
)
PERCEIVED_FIELDS = (
    "perceived_gender_presentation",
    "perceived_racialised_appearance",
    "perceived_skin_tone",
)
PROHIBITED_INFERENCES = (
    "ethnicity",
    "religion",
    "nationality",
    "citizenship",
    "disability",
    "sexual_orientation",
)

# These tokens are checked only on unexpected columns.  Exact versioned schemas remain the
# primary boundary; the list makes a failure involving leakage or identity-inference fields clear.
PROHIBITED_COLUMN_TOKENS = frozenset(
    {
        "age",
        "ancestry",
        "condition",
        "disability",
        "ethnicity",
        "gender",
        "hiring",
        "identity",
        "label",
        "nationality",
        "outcome",
        "phase",
        "participant",
        "post",
        "pre",
        "prediction",
        "proxy",
        "race",
        "religion",
        "score",
        "session",
        "sex",
        "sexual",
        "target",
        "transcript",
    }
)
_CODEBOOK_FIELDS = {
    "schema_version",
    "codebook_version",
    "status",
    "approval_reference",
    "constructs",
    "perceived_gender_presentation_categories",
    "perceived_racialised_appearance_categories",
    "perceived_skin_tone_values",
    "confidence_values",
    "unknown_representation",
    "prohibited_inferences",
}
_CATEGORY_FIELDS = {"value", "definition"}
_VERSION_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$", re.ASCII)
_RATER_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$", re.ASCII)
_OPAQUE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$", re.ASCII)
_SYNTHETIC_CODEBOOK_PATH = (
    Path(__file__).parent / "templates" / "annotation_codebook.synthetic-v1.json"
)
_SYNTHETIC_RATER_PATH = Path(__file__).parent / "templates" / "rater_annotations_template.csv"
_SYNTHETIC_CONSENSUS_PATH = (
    Path(__file__).parent / "templates" / "consensus_annotations_template.csv"
)


@dataclass(frozen=True, slots=True)
class AnnotationCodebook:
    """One exact, versioned category vocabulary for perception annotation."""

    codebook_version: str
    status: str
    approval_reference: str
    gender_categories: tuple[str, ...]
    racialised_appearance_categories: tuple[str, ...]
    skin_tone_values: tuple[int, ...]
    confidence_values: tuple[int, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class RaterAnnotation:
    """One pseudonymous rater's perceptions for one participant."""

    participant_id: str
    rater_id: str
    perceived_gender_presentation: str | None
    perceived_racialised_appearance: str | None
    perceived_skin_tone: int | None
    cannot_determine: bool
    gender_presentation_confidence: int | None
    racialised_appearance_confidence: int | None
    skin_tone_confidence: int | None
    codebook_version: str


@dataclass(frozen=True, slots=True)
class RaterResponse:
    """One blinded response keyed only by opaque rater-manifest and packet IDs."""

    rater_manifest_id: str
    packet_id: str
    perceived_gender_presentation: str | None
    perceived_racialised_appearance: str | None
    perceived_skin_tone: int | None
    cannot_determine: bool
    gender_presentation_confidence: int | None
    racialised_appearance_confidence: int | None
    skin_tone_confidence: int | None
    codebook_version: str


@dataclass(frozen=True, slots=True)
class ConsensusAnnotation:
    """Fixed participant-level consensus kept distinct from raw rater judgements."""

    participant_id: str
    perceived_gender_presentation: str | None
    perceived_racialised_appearance: str | None
    perceived_skin_tone: float | None
    cannot_determine: bool
    rater_count: int
    consensus_method: str
    codebook_version: str


@dataclass(frozen=True, slots=True)
class SelfReportedAnnotation:
    """Participant self-report with explicit provenance; never merged with perceptions."""

    participant_id: str
    self_reported_gender: str | None
    self_reported_ethnicity: str | None
    source_provenance: str


def load_annotation_codebook(
    path: str | Path, *, require_approved: bool = True
) -> AnnotationCodebook:
    """Load an exact codebook JSON and optionally require pre-rating approval status."""
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("annotation codebook must be a regular non-symlink file")
    try:
        raw = source.read_bytes()
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_json_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("cannot read a valid annotation codebook") from error
    if not isinstance(payload, dict) or set(payload) != _CODEBOOK_FIELDS:
        raise ValueError("annotation codebook must have the exact versioned schema")
    if payload["schema_version"] != CODEBOOK_SCHEMA_VERSION:
        raise ValueError("unsupported annotation codebook schema_version")
    version = payload["codebook_version"]
    if not isinstance(version, str) or _VERSION_PATTERN.fullmatch(version) is None:
        raise ValueError("annotation codebook version is invalid")
    status = payload["status"]
    if status not in {"approved_for_rating", "synthetic_example_only"}:
        raise ValueError("annotation codebook status is invalid")
    if require_approved and status != "approved_for_rating":
        raise ValueError("real annotation requires an approved_for_rating codebook")
    if (status == "synthetic_example_only") != (version == SYNTHETIC_CODEBOOK_VERSION):
        raise ValueError("synthetic codebook status and version must agree")
    approval_reference = _json_text(payload["approval_reference"], "approval_reference")
    if payload["constructs"] != list(PERCEIVED_FIELDS):
        raise ValueError("codebook must define exactly the three approved perceived constructs")
    if payload["perceived_skin_tone_values"] != list(SKIN_TONE_VALUES):
        raise ValueError("perceived skin tone codebook must use ordinal values 1 through 5")
    if payload["confidence_values"] != list(range(1, 6)):
        raise ValueError("field confidence codebook must use values 1 through 5")
    if payload["unknown_representation"] != "blank_field;cannot_determine=1_when_all_blank":
        raise ValueError("codebook unknown representation is invalid")
    if payload["prohibited_inferences"] != list(PROHIBITED_INFERENCES):
        raise ValueError("codebook prohibited-inference boundary is invalid")
    gender = _category_values(
        payload["perceived_gender_presentation_categories"],
        "perceived_gender_presentation_categories",
    )
    racialised = _category_values(
        payload["perceived_racialised_appearance_categories"],
        "perceived_racialised_appearance_categories",
    )
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return AnnotationCodebook(
        codebook_version=version,
        status=status,
        approval_reference=approval_reference,
        gender_categories=gender,
        racialised_appearance_categories=racialised,
        skin_tone_values=SKIN_TONE_VALUES,
        confidence_values=tuple(range(1, 6)),
        sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def load_rater_annotations(
    path: str | Path, *, codebook: AnnotationCodebook | str | Path | None = None
) -> tuple[RaterAnnotation, ...]:
    """Load strict participant/rater rows and validate every value against one codebook."""
    source = Path(path)
    rows = _read_exact_csv(source, RATER_COLUMNS, "rater annotation")
    resolved_codebook = _resolve_codebook(source, rows, codebook, table="rater")
    indexed: set[tuple[str, str]] = set()
    annotations: list[RaterAnnotation] = []
    for row_number, row in enumerate(rows, start=2):
        participant_id = _annotation_participant_id(row["participant_id"], row_number)
        rater_id = _pseudonymous_rater_id(row["rater_id"], row_number)
        key = (participant_id, rater_id)
        if key in indexed:
            raise ValueError(f"duplicate participant/rater annotation {key!r}")
        indexed.add(key)
        cannot_determine = _parse_bool(row["cannot_determine"], row_number)
        gender = _optional_text(
            row["perceived_gender_presentation"],
            "perceived_gender_presentation",
            row_number,
        )
        racialised = _optional_text(
            row["perceived_racialised_appearance"],
            "perceived_racialised_appearance",
            row_number,
        )
        skin_text = _optional_text(row["perceived_skin_tone"], "perceived_skin_tone", row_number)
        skin = _rater_skin_tone(skin_text, row_number)
        _require_category(gender, resolved_codebook.gender_categories, "gender presentation")
        _require_category(
            racialised,
            resolved_codebook.racialised_appearance_categories,
            "racialised appearance",
        )
        if cannot_determine and any(value is not None for value in (gender, racialised, skin)):
            raise ValueError("cannot_determine=1 requires every perceived field to be blank")
        if not cannot_determine and all(value is None for value in (gender, racialised, skin)):
            raise ValueError("a determinate row must contain at least one perceived field")
        annotations.append(
            RaterAnnotation(
                participant_id=participant_id,
                rater_id=rater_id,
                perceived_gender_presentation=gender,
                perceived_racialised_appearance=racialised,
                perceived_skin_tone=skin,
                cannot_determine=cannot_determine,
                gender_presentation_confidence=_field_confidence(
                    row["gender_presentation_confidence"],
                    "gender_presentation_confidence",
                    gender,
                    row_number,
                ),
                racialised_appearance_confidence=_field_confidence(
                    row["racialised_appearance_confidence"],
                    "racialised_appearance_confidence",
                    racialised,
                    row_number,
                ),
                skin_tone_confidence=_field_confidence(
                    row["skin_tone_confidence"],
                    "skin_tone_confidence",
                    skin_text,
                    row_number,
                ),
                codebook_version=resolved_codebook.codebook_version,
            )
        )
    return tuple(sorted(annotations, key=lambda item: (int(item.participant_id), item.rater_id)))


def load_rater_responses(
    path: str | Path, *, codebook: AnnotationCodebook | str | Path
) -> tuple[RaterResponse, ...]:
    """Load a rater-facing response table containing no participant or source identifiers."""
    source = Path(path)
    rows = _read_exact_csv(source, RATER_RESPONSE_COLUMNS, "blinded rater response")
    resolved_codebook = _resolve_codebook(source, rows, codebook, table="response")
    result: list[RaterResponse] = []
    seen: set[tuple[str, str]] = set()
    manifest_ids: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        manifest_id = _opaque_id(row["rater_manifest_id"], "rater_manifest_id", row_number)
        packet_id = _opaque_id(row["packet_id"], "packet_id", row_number)
        key = (manifest_id, packet_id)
        if key in seen:
            raise ValueError(f"duplicate blinded rater response {key!r}")
        seen.add(key)
        manifest_ids.add(manifest_id)
        gender, racialised, skin, cannot_determine, confidences = _parse_perceived_values(
            row, row_number, resolved_codebook
        )
        result.append(
            RaterResponse(
                rater_manifest_id=manifest_id,
                packet_id=packet_id,
                perceived_gender_presentation=gender,
                perceived_racialised_appearance=racialised,
                perceived_skin_tone=skin,
                cannot_determine=cannot_determine,
                gender_presentation_confidence=confidences[0],
                racialised_appearance_confidence=confidences[1],
                skin_tone_confidence=confidences[2],
                codebook_version=resolved_codebook.codebook_version,
            )
        )
    if len(manifest_ids) != 1:
        raise ValueError("one blinded response table must use exactly one rater_manifest_id")
    return tuple(result)


def blank_rater_response_csv(display_manifest: Mapping[str, Any], *, codebook_version: str) -> str:
    """Create an ordered blank response form keyed only by opaque display identifiers."""
    if (
        not isinstance(codebook_version, str)
        or _VERSION_PATTERN.fullmatch(codebook_version) is None
    ):
        raise ValueError("codebook_version is invalid")
    if display_manifest.get("schema_version") != "mit-annotation-rater-display-1":
        raise ValueError("unsupported rater display manifest")
    manifest_id = _opaque_id(display_manifest.get("rater_manifest_id"), "rater_manifest_id", 0)
    packets = display_manifest.get("packets")
    if not isinstance(packets, list):
        raise ValueError("rater display packets must be a list")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=RATER_RESPONSE_COLUMNS, lineterminator="\n")
    writer.writeheader()
    seen: set[str] = set()
    for packet in packets:
        if not isinstance(packet, dict):
            raise ValueError("rater display packet is invalid")
        packet_id = _opaque_id(packet.get("packet_id"), "packet_id", 0)
        if packet_id in seen:
            raise ValueError("rater display packet IDs must be unique")
        seen.add(packet_id)
        writer.writerow(
            {
                "rater_manifest_id": manifest_id,
                "packet_id": packet_id,
                "codebook_version": codebook_version,
            }
        )
    return output.getvalue()


def link_rater_responses(
    responses: Iterable[RaterResponse], linkage_manifest: Mapping[str, Any]
) -> tuple[RaterAnnotation, ...]:
    """Convert blinded responses to the restricted participant-keyed raw table."""
    from .annotation_stimuli import validate_rater_linkage_manifest

    validate_rater_linkage_manifest(linkage_manifest)
    response_rows = tuple(responses)
    if not response_rows:
        raise ValueError("at least one blinded response is required")
    expected_manifest_id = linkage_manifest["rater_manifest_id"]
    expected_rater_id = linkage_manifest["rater_id"]
    participant_by_packet = {
        row["packet_id"]: row["participant_id"] for row in linkage_manifest["packets"]
    }
    observed_ids = {row.packet_id for row in response_rows}
    if any(row.rater_manifest_id != expected_manifest_id for row in response_rows):
        raise ValueError("response rater_manifest_id does not match restricted linkage")
    missing = sorted(set(participant_by_packet) - observed_ids)
    unexpected = sorted(observed_ids - set(participant_by_packet))
    if missing or unexpected or len(response_rows) != len(observed_ids):
        raise ValueError(
            f"blinded response coverage mismatch: missing={missing!r}; unexpected={unexpected!r}"
        )
    return tuple(
        sorted(
            (
                RaterAnnotation(
                    participant_id=participant_by_packet[row.packet_id],
                    rater_id=expected_rater_id,
                    perceived_gender_presentation=row.perceived_gender_presentation,
                    perceived_racialised_appearance=row.perceived_racialised_appearance,
                    perceived_skin_tone=row.perceived_skin_tone,
                    cannot_determine=row.cannot_determine,
                    gender_presentation_confidence=row.gender_presentation_confidence,
                    racialised_appearance_confidence=row.racialised_appearance_confidence,
                    skin_tone_confidence=row.skin_tone_confidence,
                    codebook_version=row.codebook_version,
                )
                for row in response_rows
            ),
            key=lambda item: int(item.participant_id),
        )
    )


def rater_annotations_csv(annotations: Iterable[RaterAnnotation]) -> str:
    """Serialise linked raw rater rows with the internal participant-only schema."""
    rows = sorted(annotations, key=lambda item: (int(item.participant_id), item.rater_id))
    if not rows:
        raise ValueError("at least one linked rater annotation is required")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=RATER_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "participant_id": row.participant_id,
                "rater_id": row.rater_id,
                "perceived_gender_presentation": row.perceived_gender_presentation or "",
                "perceived_racialised_appearance": row.perceived_racialised_appearance or "",
                "perceived_skin_tone": _number_text(row.perceived_skin_tone),
                "cannot_determine": int(row.cannot_determine),
                "gender_presentation_confidence": row.gender_presentation_confidence or "",
                "racialised_appearance_confidence": (row.racialised_appearance_confidence or ""),
                "skin_tone_confidence": row.skin_tone_confidence or "",
                "codebook_version": row.codebook_version,
            }
        )
    return output.getvalue()


def build_consensus_annotations(
    annotations: Iterable[RaterAnnotation],
    *,
    minimum_raters: int = 3,
) -> tuple[ConsensusAnnotation, ...]:
    """Apply the frozen field-wise consensus rules to raw participant/rater rows."""
    if isinstance(minimum_raters, bool) or minimum_raters < 3:
        raise ValueError("minimum_raters must be at least three")
    grouped: dict[str, list[RaterAnnotation]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for annotation in annotations:
        participant_id = _annotation_participant_id(annotation.participant_id, 0)
        key = (participant_id, annotation.rater_id)
        if key in seen:
            raise ValueError(f"duplicate participant/rater annotation {key!r}")
        seen.add(key)
        grouped[participant_id].append(annotation)
    if not grouped:
        raise ValueError("at least one rater annotation is required")
    result: list[ConsensusAnnotation] = []
    for participant_id in sorted(grouped, key=int):
        rows = grouped[participant_id]
        if len(rows) < minimum_raters:
            raise ValueError(
                f"participant {participant_id} has fewer than {minimum_raters} independent raters"
            )
        versions = {row.codebook_version for row in rows}
        if len(versions) != 1:
            raise ValueError(f"participant {participant_id} mixes codebook versions")
        gender = _nominal_consensus(row.perceived_gender_presentation for row in rows)
        racialised = _nominal_consensus(row.perceived_racialised_appearance for row in rows)
        skin = _skin_tone_consensus(row.perceived_skin_tone for row in rows)
        result.append(
            ConsensusAnnotation(
                participant_id=participant_id,
                perceived_gender_presentation=gender,
                perceived_racialised_appearance=racialised,
                perceived_skin_tone=skin,
                cannot_determine=all(value is None for value in (gender, racialised, skin)),
                rater_count=len(rows),
                consensus_method=CONSENSUS_METHOD,
                codebook_version=next(iter(versions)),
            )
        )
    return tuple(result)


def consensus_annotations_csv(annotations: Iterable[ConsensusAnnotation]) -> str:
    """Serialise consensus rows with the exact participant-only schema."""
    rows = sorted(annotations, key=lambda item: int(item.participant_id))
    if not rows:
        raise ValueError("at least one consensus annotation is required")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CONSENSUS_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "participant_id": row.participant_id,
                "perceived_gender_presentation": row.perceived_gender_presentation or "",
                "perceived_racialised_appearance": (row.perceived_racialised_appearance or ""),
                "perceived_skin_tone": _number_text(row.perceived_skin_tone),
                "cannot_determine": int(row.cannot_determine),
                "rater_count": row.rater_count,
                "consensus_method": row.consensus_method,
                "codebook_version": row.codebook_version,
            }
        )
    return output.getvalue()


def load_consensus_annotations(
    path: str | Path, *, codebook: AnnotationCodebook | str | Path | None = None
) -> tuple[ConsensusAnnotation, ...]:
    """Load exactly one validated consensus row per participant."""
    source = Path(path)
    rows = _read_exact_csv(source, CONSENSUS_COLUMNS, "consensus annotation")
    resolved_codebook = _resolve_codebook(source, rows, codebook, table="consensus")
    indexed: set[str] = set()
    result: list[ConsensusAnnotation] = []
    for row_number, row in enumerate(rows, start=2):
        participant_id = _annotation_participant_id(row["participant_id"], row_number)
        if participant_id in indexed:
            raise ValueError(f"duplicate consensus participant {participant_id}")
        indexed.add(participant_id)
        cannot_determine = _parse_bool(row["cannot_determine"], row_number)
        gender = _optional_text(
            row["perceived_gender_presentation"],
            "perceived_gender_presentation",
            row_number,
        )
        racialised = _optional_text(
            row["perceived_racialised_appearance"],
            "perceived_racialised_appearance",
            row_number,
        )
        skin = _consensus_skin_tone(
            _optional_text(row["perceived_skin_tone"], "perceived_skin_tone", row_number),
            row_number,
        )
        _require_category(gender, resolved_codebook.gender_categories, "gender presentation")
        _require_category(
            racialised,
            resolved_codebook.racialised_appearance_categories,
            "racialised appearance",
        )
        if cannot_determine != all(value is None for value in (gender, racialised, skin)):
            raise ValueError(
                "consensus cannot_determine must be 1 exactly when all perceived fields are blank"
            )
        method = _required_text(row["consensus_method"], "consensus_method", row_number)
        if method != CONSENSUS_METHOD:
            raise ValueError("consensus_method does not match the frozen consensus rule")
        result.append(
            ConsensusAnnotation(
                participant_id=participant_id,
                perceived_gender_presentation=gender,
                perceived_racialised_appearance=racialised,
                perceived_skin_tone=skin,
                cannot_determine=cannot_determine,
                rater_count=_bounded_integer(row["rater_count"], "rater_count", 3, 100, row_number),
                consensus_method=method,
                codebook_version=resolved_codebook.codebook_version,
            )
        )
    return tuple(sorted(result, key=lambda item: int(item.participant_id)))


def load_self_reported_annotations(path: str | Path) -> tuple[SelfReportedAnnotation, ...]:
    """Load separately sourced self-report; provenance is mandatory for every row."""
    rows = _read_exact_csv(Path(path), SELF_REPORTED_COLUMNS, "self-reported annotation")
    indexed: set[str] = set()
    result = []
    for row_number, row in enumerate(rows, start=2):
        participant_id = _annotation_participant_id(row["participant_id"], row_number)
        if participant_id in indexed:
            raise ValueError(f"duplicate self-reported participant {participant_id}")
        indexed.add(participant_id)
        gender = _optional_text(row["self_reported_gender"], "self_reported_gender", row_number)
        ethnicity = _optional_text(
            row["self_reported_ethnicity"], "self_reported_ethnicity", row_number
        )
        if gender is None and ethnicity is None:
            raise ValueError("self-reported row must contain gender or ethnicity")
        result.append(
            SelfReportedAnnotation(
                participant_id=participant_id,
                self_reported_gender=gender,
                self_reported_ethnicity=ethnicity,
                source_provenance=_required_text(
                    row["source_provenance"], "source_provenance", row_number
                ),
            )
        )
    return tuple(sorted(result, key=lambda item: int(item.participant_id)))


def require_exact_participant_coverage(
    participant_ids: Iterable[str], annotations: Iterable[object]
) -> None:
    """Require exactly one participant-keyed annotation object per expected participant."""
    expected = {_annotation_participant_id(value, 0) for value in participant_ids}
    observed = [_annotation_participant_id(item.participant_id, 0) for item in annotations]
    if len(observed) != len(set(observed)):
        raise ValueError("participant annotation join must have one row per participant")
    _require_set_coverage(expected, set(observed))


def require_rater_participant_coverage(
    participant_ids: Iterable[str],
    annotations: Iterable[RaterAnnotation],
    *,
    minimum_raters: int = 3,
) -> None:
    """Require only expected participants and enough independent pseudonymous raters."""
    if isinstance(minimum_raters, bool) or minimum_raters < 3:
        raise ValueError("minimum_raters must be at least three")
    expected = {_annotation_participant_id(value, 0) for value in participant_ids}
    grouped: dict[str, set[str]] = defaultdict(set)
    for annotation in annotations:
        participant_id = _annotation_participant_id(annotation.participant_id, 0)
        grouped[participant_id].add(annotation.rater_id)
    _require_set_coverage(expected, set(grouped))
    insufficient = sorted(
        (
            participant_id
            for participant_id, raters in grouped.items()
            if len(raters) < minimum_raters
        ),
        key=int,
    )
    if insufficient:
        raise ValueError(f"participants with fewer than {minimum_raters} raters: {insufficient!r}")


def blank_rater_form_csv(
    participant_ids: Iterable[str], *, rater_id: str = "", codebook_version: str = ""
) -> str:
    """Generate an internal synthetic/testing form, never a rater-facing collection form.

    Real collection uses :func:`blank_rater_response_csv`, which exposes opaque packet IDs only.
    """
    canonical = sorted({_annotation_participant_id(value, 0) for value in participant_ids}, key=int)
    if rater_id:
        _pseudonymous_rater_id(rater_id, 0)
    if codebook_version and _VERSION_PATTERN.fullmatch(codebook_version) is None:
        raise ValueError("codebook_version is invalid")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=RATER_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for participant_id in canonical:
        writer.writerow(
            {
                "participant_id": participant_id,
                "rater_id": rater_id,
                "codebook_version": codebook_version,
            }
        )
    return output.getvalue()


def _read_exact_csv(path: Path, expected: tuple[str, ...], label: str) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or tuple(reader.fieldnames) != expected:
                observed = tuple(reader.fieldnames or ())
                forbidden = [value for value in observed if _forbidden_column(value)]
                if forbidden:
                    raise ValueError(f"{label} contains prohibited columns: {', '.join(forbidden)}")
                raise ValueError(f"{label} must have the exact versioned schema")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError(f"cannot read {label} table") from error
    if not rows:
        raise ValueError(f"{label} table must contain at least one row")
    if any(
        None in row
        or any(not isinstance(key, str) for key in row)
        or any(not isinstance(value, str) for value in row.values())
        for row in rows
    ):
        raise ValueError(f"{label} row length does not match the exact schema")
    return rows


def _resolve_codebook(
    source: Path,
    rows: Sequence[Mapping[str, str]],
    codebook: AnnotationCodebook | str | Path | None,
    *,
    table: str,
) -> AnnotationCodebook:
    versions = {
        _required_text(row.get("codebook_version"), "codebook_version", number)
        for number, row in enumerate(rows, start=2)
    }
    if len(versions) != 1:
        raise ValueError(f"{table} table must use exactly one codebook_version")
    version = next(iter(versions))
    if codebook is None:
        allowed_template = _SYNTHETIC_RATER_PATH if table == "rater" else _SYNTHETIC_CONSENSUS_PATH
        if source.resolve() != allowed_template.resolve() or version != SYNTHETIC_CODEBOOK_VERSION:
            raise ValueError("an external approved annotation codebook is required")
        resolved = load_annotation_codebook(_SYNTHETIC_CODEBOOK_PATH, require_approved=False)
    elif isinstance(codebook, AnnotationCodebook):
        resolved = codebook
    else:
        resolved = load_annotation_codebook(codebook, require_approved=True)
    if resolved.codebook_version != version:
        raise ValueError("annotation row codebook_version does not match the supplied codebook")
    return resolved


def _forbidden_column(value: object) -> bool:
    if not isinstance(value, str):
        return True
    token = re.sub(r"[^a-z0-9]+", "_", value.strip().casefold()).strip("_")
    if token in {
        "perceived_gender_presentation",
        "perceived_racialised_appearance",
        "perceived_skin_tone",
        "gender_presentation_confidence",
        "racialised_appearance_confidence",
        "skin_tone_confidence",
        "self_reported_gender",
        "self_reported_ethnicity",
    }:
        return False
    parts = set(token.split("_"))
    return bool(parts & PROHIBITED_COLUMN_TOKENS) or "recommendhiring" in token


def _annotation_participant_id(value: object, row_number: int) -> str:
    try:
        canonical = canonical_participant_id(value)
    except ValueError as error:
        raise ValueError(
            f"participant_id at row {row_number} must be canonical ASCII decimal"
        ) from error
    if value != canonical:
        raise ValueError(f"participant_id at row {row_number} must be canonical ASCII decimal")
    return canonical


def _pseudonymous_rater_id(value: object, row_number: int) -> str:
    parsed = _required_text(value, "rater_id", row_number)
    if _RATER_ID_PATTERN.fullmatch(parsed) is None:
        raise ValueError(f"rater_id at row {row_number} must be a pseudonymous token")
    return parsed


def _opaque_id(value: object, field: str, row_number: int) -> str:
    if not isinstance(value, str) or _OPAQUE_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} at row {row_number} must be 32 lowercase hexadecimal digits")
    return value


def _parse_perceived_values(
    row: Mapping[str, str], row_number: int, codebook: AnnotationCodebook
) -> tuple[str | None, str | None, int | None, bool, tuple[int | None, int | None, int | None]]:
    gender = _optional_text(
        row["perceived_gender_presentation"],
        "perceived_gender_presentation",
        row_number,
    )
    racialised = _optional_text(
        row["perceived_racialised_appearance"],
        "perceived_racialised_appearance",
        row_number,
    )
    skin_text = _optional_text(row["perceived_skin_tone"], "perceived_skin_tone", row_number)
    skin = _rater_skin_tone(skin_text, row_number)
    _require_category(gender, codebook.gender_categories, "gender presentation")
    _require_category(
        racialised, codebook.racialised_appearance_categories, "racialised appearance"
    )
    cannot_determine = _parse_bool(row["cannot_determine"], row_number)
    if cannot_determine and any(value is not None for value in (gender, racialised, skin)):
        raise ValueError("cannot_determine=1 requires every perceived field to be blank")
    if not cannot_determine and all(value is None for value in (gender, racialised, skin)):
        raise ValueError("a determinate row must contain at least one perceived field")
    confidences = (
        _field_confidence(
            row["gender_presentation_confidence"],
            "gender_presentation_confidence",
            gender,
            row_number,
        ),
        _field_confidence(
            row["racialised_appearance_confidence"],
            "racialised_appearance_confidence",
            racialised,
            row_number,
        ),
        _field_confidence(
            row["skin_tone_confidence"],
            "skin_tone_confidence",
            skin_text,
            row_number,
        ),
    )
    return gender, racialised, skin, cannot_determine, confidences


def _required_text(value: object, field: str, row_number: int) -> str:
    parsed = _optional_text(value, field, row_number)
    if parsed is None:
        raise ValueError(f"{field} at row {row_number} is required")
    return parsed


def _optional_text(value: object, field: str, row_number: int) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or value != value.strip() or "\0" in value:
        raise ValueError(f"{field} at row {row_number} must be trimmed and NUL-free")
    return value


def _parse_bool(value: object, row_number: int) -> bool:
    if value not in {"0", "1"}:
        raise ValueError(f"cannot_determine at row {row_number} must be 0 or 1")
    return value == "1"


def _bounded_integer(value: object, field: str, low: int, high: int, row_number: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} at row {row_number} must be an integer") from error
    if str(result) != value or not low <= result <= high:
        raise ValueError(f"{field} at row {row_number} must be in [{low},{high}]")
    return result


def _rater_skin_tone(value: str | None, row_number: int) -> int | None:
    if value is None:
        return None
    return _bounded_integer(value, "perceived_skin_tone", 1, 5, row_number)


def _consensus_skin_tone(value: str | None, row_number: int) -> float | None:
    if value is None:
        return None
    try:
        number = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"perceived_skin_tone at row {row_number} is invalid") from error
    if number < 1 or number > 5 or number * 2 != (number * 2).to_integral_value():
        raise ValueError("consensus perceived_skin_tone must be in half-step range [1,5]")
    if _number_text(float(number)) != value:
        raise ValueError("consensus perceived_skin_tone must use canonical decimal notation")
    return float(number)


def _field_confidence(
    value: object, field: str, perceived_value: str | None, row_number: int
) -> int | None:
    if perceived_value is None:
        if value not in {None, ""}:
            raise ValueError(f"{field} must be blank when its perceived field is blank")
        return None
    return _bounded_integer(value, field, 1, 5, row_number)


def _nominal_consensus(values: Iterable[str | None]) -> str | None:
    valid = [value for value in values if value is not None]
    if len(valid) < 2:
        return None
    counts = Counter(valid)
    winner, count = min(counts.items(), key=lambda item: (-item[1], item[0].encode("utf-8")))
    return winner if count * 2 > len(valid) else None


def _skin_tone_consensus(values: Iterable[int | None]) -> float | None:
    valid = sorted(value for value in values if value is not None)
    if len(valid) < 2 or valid[-1] - valid[0] > 1:
        return None
    return float(statistics.median(valid))


def _category_values(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(f"{field} must contain at least two defined categories")
    result: list[str] = []
    for row in value:
        if not isinstance(row, dict) or set(row) != _CATEGORY_FIELDS:
            raise ValueError(f"{field} category schema is invalid")
        category = _json_text(row["value"], f"{field}.value")
        _json_text(row["definition"], f"{field}.definition")
        if category.casefold() in {"unknown", "cannot_determine", "cannot-determine"}:
            raise ValueError("unknown must be represented by a blank field, not a category")
        result.append(category)
    if len(result) != len(set(result)):
        raise ValueError(f"{field} categories must be unique")
    return tuple(result)


def _require_category(value: str | None, allowed: tuple[str, ...], label: str) -> None:
    if value is not None and value not in allowed:
        raise ValueError(f"{label} value is not defined by the supplied codebook")


def _json_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or "\0" in value:
        raise ValueError(f"annotation codebook {field} must be trimmed, non-empty text")
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _require_set_coverage(expected: set[str], observed: set[str]) -> None:
    missing = sorted(expected - observed, key=int)
    unexpected = sorted(observed - expected, key=int)
    if missing or unexpected:
        raise ValueError(
            "participant annotation coverage mismatch: "
            f"missing={missing!r}; unexpected={unexpected!r}"
        )


def _number_text(value: float | None) -> str:
    if value is None:
        return ""
    return format(value, ".15g")
