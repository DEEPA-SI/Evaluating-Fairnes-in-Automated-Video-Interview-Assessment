"""Deterministic intake for Deepa's provisional MIT session annotations.

This module deliberately preserves the received workbook's source-specific semantics.  Its
normalized CSV is a reconciliation artefact, not the participant/rater table accepted by the
Stage-B/C annotation contract.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import posixpath
import re
import stat
import zipfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any
from xml.etree import ElementTree

from src.data.mit_session_mapping import (
    canonical_mit_session_mapping_json,
    canonical_to_source_lookup,
    load_verified_session_mapping,
)
from src.secure_io import AnchoredDirectory, AnchoredFile

INTAKE_SCHEMA_VERSION = "mit-perceived-session-annotation-intake-1"
RECEIPT_SCHEMA_VERSION = "mit-perceived-session-annotation-receipt-2"
SOURCE_RECEIPT_SCHEMA_VERSION = "mit-annotation-source-receipt-1"
EXPECTED_HEADERS = ("Video", "Gender", "Skin Tone", "Ethnicity", "Age")
EXPECTED_LEGEND = """Gender (Perceived Gender Presentation)

Male-presenting (M)
Female-presenting (F)
Androgynous / Ambiguous (A)
Cannot Determine (CD)

Skin Tone

1 (Lightest)
2
3
4
5 (Darkest)

Ethnicity → Perceived Racialised Appearance

White (W)
Black (B)
East Asian (EA)
South Asian (SA)
Southeast Asian (SEA)
Hispanic/Latino-appearing (H)
Middle Eastern/North African-appearing (MENA)
Ambiguous/Multiple (AM)
Cannot Determine (CD)

Age (bracket estimate)

18–25 (A1)
26–35 (A2)
36–50 (A3)
50+ (A4)
Cannot Determine (CD)"""
NORMALIZED_COLUMNS = (
    "source_session_id",
    "canonical_session_id",
    "participant_id",
    "phase",
    "source_row_number",
    "perceived_gender_presentation_source_code",
    "perceived_gender_presentation_source_label",
    "perceived_skin_tone_source_value",
    "perceived_skin_tone_source_scale",
    "perceived_racialised_appearance_source_code",
    "perceived_racialised_appearance_source_label",
    "perceived_age_band_source_code",
    "perceived_age_band_source_label",
)

GENDER_LABELS = {
    "M": "masculine_presentation",
    "F": "feminine_presentation",
    "A": "androgynous_or_ambiguous_presentation",
    "CD": "cannot_determine",
}
RACIALISED_APPEARANCE_LABELS = {
    "W": "white_appearance_source_category",
    "B": "black_appearance_source_category",
    "EA": "east_asian_appearance_source_category",
    "SA": "south_asian_appearance_source_category",
    "SEA": "southeast_asian_appearance_source_category",
    "H": "hispanic_or_latino_appearance_source_category",
    "MENA": "middle_eastern_or_north_african_appearance_source_category",
    "AM": "ambiguous_or_multiple_appearance_source_category",
    "CD": "cannot_determine",
}
AGE_LABELS = {
    "A1": "18_to_25_source_band",
    "A2": "26_to_35_source_band",
    "A3": "36_to_50_source_band",
    "A4": "50_plus_source_band",
    "CD": "cannot_determine",
}
SKIN_TONE_SCALE = "source_1_to_5_lightest_to_darkest"

_SESSION_ID = re.compile(r"^(PP|P)([1-9][0-9]*)$", re.ASCII)
_CELL_REFERENCE = re.compile(r"^([A-Z]+)([1-9][0-9]*)$", re.ASCII)
_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_DC_NS = "http://purl.org/dc/elements/1.1/"
_DCTERMS_NS = "http://purl.org/dc/terms/"
_CP_NS = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
_MAX_ZIP_ENTRIES = 200
_MAX_ENTRY_BYTES = 16 * 1024 * 1024
_MAX_TOTAL_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class SourceAnnotation:
    """One validated source-workbook row before canonical mapping."""

    source_session_id: str
    source_row_number: int
    gender_code: str
    skin_tone_value: int
    racialised_appearance_code: str
    age_code: str


@dataclass(frozen=True, slots=True)
class WorkbookAudit:
    """Validated workbook contents and structural metadata."""

    rows: tuple[SourceAnnotation, ...]
    sheet_name: str
    worksheet_dimension: str
    workbook_created_utc: str | None
    workbook_modified_utc: str | None
    workbook_creator_present: bool
    workbook_last_modified_by_present: bool
    zip_entry_count: int
    uncompressed_bytes: int
    formula_count: int
    merged_range_count: int
    hidden_sheet_count: int
    legend_row_number: int | None


@dataclass(frozen=True, slots=True)
class IntakePublication:
    """Published restricted paths and their verified checksums."""

    source_workbook: Path
    source_receipt: Path
    normalized_csv: Path
    conversion_receipt: Path
    source_sha256: str
    normalized_csv_sha256: str


def audit_annotation_workbook(content: bytes) -> WorkbookAudit:
    """Validate and parse one bounded, formula-free OOXML workbook."""
    if not isinstance(content, bytes) or not content:
        raise ValueError("annotation workbook must contain non-empty bytes")
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except (zipfile.BadZipFile, OSError) as error:
        raise ValueError("annotation workbook is not a valid OOXML ZIP archive") from error

    with archive:
        entries = _validate_archive(archive)
        _reject_external_relationships(archive, entries)
        workbook = _xml_root(_required_entry(archive, entries, "xl/workbook.xml"))
        relationships = _xml_root(_required_entry(archive, entries, "xl/_rels/workbook.xml.rels"))
        sheets = workbook.find(f"{{{_MAIN_NS}}}sheets")
        if sheets is None:
            raise ValueError("annotation workbook has no worksheet declaration")
        sheet_rows = list(sheets)
        visible = [row for row in sheet_rows if row.attrib.get("state", "visible") == "visible"]
        hidden_count = len(sheet_rows) - len(visible)
        if len(sheet_rows) != 1 or len(visible) != 1:
            raise ValueError("annotation workbook must contain exactly one visible worksheet")
        sheet = visible[0]
        relationship_id = sheet.attrib.get(f"{{{_OFFICE_REL_NS}}}id")
        target = _worksheet_target(relationships, relationship_id)
        worksheet_path = posixpath.normpath(posixpath.join("xl", target))
        if worksheet_path.startswith("../") or worksheet_path not in entries:
            raise ValueError("annotation worksheet relationship is unsafe or missing")
        worksheet = _xml_root(archive.read(worksheet_path))
        shared_strings = _shared_strings(archive, entries)
        parsed_rows, formula_count = _worksheet_values(worksheet, shared_strings)

        dimension_node = worksheet.find(f"{{{_MAIN_NS}}}dimension")
        dimension = dimension_node.attrib.get("ref", "") if dimension_node is not None else ""
        if not dimension:
            raise ValueError("annotation worksheet has no declared dimension")
        merged = worksheet.find(f"{{{_MAIN_NS}}}mergeCells")
        merged_count = len(list(merged)) if merged is not None else 0
        if formula_count:
            raise ValueError("annotation workbook formulas are not accepted")
        if merged_count:
            raise ValueError("annotation workbook merged cells are not accepted")

        rows, legend_row = _annotation_rows(parsed_rows)
        core = _core_properties(archive, entries)
        return WorkbookAudit(
            rows=tuple(rows),
            sheet_name=sheet.attrib.get("name", ""),
            worksheet_dimension=dimension,
            workbook_created_utc=core["created"],
            workbook_modified_utc=core["modified"],
            workbook_creator_present=bool(core["creator"]),
            workbook_last_modified_by_present=bool(core["last_modified_by"]),
            zip_entry_count=len(entries),
            uncompressed_bytes=sum(entry.file_size for entry in archive.infolist()),
            formula_count=formula_count,
            merged_range_count=merged_count,
            hidden_sheet_count=hidden_count,
            legend_row_number=legend_row,
        )


def build_normalized_intake(
    audit: WorkbookAudit, mapping_payload: dict[str, Any]
) -> tuple[bytes, dict[str, Any]]:
    """Map source sessions canonically and return deterministic CSV plus aggregate validation."""
    canonical_to_source = canonical_to_source_lookup(mapping_payload)
    source_to_canonical = {source: canonical for canonical, source in canonical_to_source.items()}
    if len(source_to_canonical) != len(canonical_to_source):
        raise ValueError("canonical mapping is not one-to-one")

    by_source: dict[str, SourceAnnotation] = {}
    unexpected: list[str] = []
    for row in audit.rows:
        if row.source_session_id in by_source:
            raise ValueError(f"duplicate source session {row.source_session_id!r}")
        by_source[row.source_session_id] = row
        if row.source_session_id not in source_to_canonical:
            unexpected.append(row.source_session_id)
    if unexpected:
        raise ValueError(f"unexpected source sessions: {', '.join(sorted(unexpected))}")

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=NORMALIZED_COLUMNS, lineterminator="\n")
    writer.writeheader()
    canonical_rows: list[dict[str, str | int]] = []
    for canonical, source in canonical_to_source.items():
        if source not in by_source:
            continue
        raw = by_source[source]
        match = _SESSION_ID.fullmatch(canonical)
        if match is None:
            raise ValueError("canonical mapping contains an invalid session ID")
        prefix, participant_id = match.groups()
        normalized: dict[str, str | int] = {
            "source_session_id": source,
            "canonical_session_id": canonical,
            "participant_id": participant_id,
            "phase": "post" if prefix == "PP" else "pre",
            "source_row_number": raw.source_row_number,
            "perceived_gender_presentation_source_code": raw.gender_code,
            "perceived_gender_presentation_source_label": GENDER_LABELS[raw.gender_code],
            "perceived_skin_tone_source_value": raw.skin_tone_value,
            "perceived_skin_tone_source_scale": SKIN_TONE_SCALE,
            "perceived_racialised_appearance_source_code": raw.racialised_appearance_code,
            "perceived_racialised_appearance_source_label": RACIALISED_APPEARANCE_LABELS[
                raw.racialised_appearance_code
            ],
            "perceived_age_band_source_code": raw.age_code,
            "perceived_age_band_source_label": AGE_LABELS[raw.age_code],
        }
        writer.writerow(normalized)
        canonical_rows.append(normalized)

    csv_bytes = output.getvalue().encode("utf-8")
    validation = _aggregate_validation(
        audit.rows,
        canonical_rows,
        expected_source_ids=set(source_to_canonical),
        observed_source_ids=set(by_source),
    )
    return csv_bytes, validation


def publish_annotation_intake(
    *,
    source_path: str | Path,
    mapping_path: str | Path,
    identity_join_audit_path: str | Path,
    face_crop_manifest_path: str | Path,
    restricted_root: str | Path,
    received_date: str,
) -> IntakePublication:
    """Validate and immutably publish a source workbook, normalized CSV and receipts."""
    _validate_iso_date(received_date)
    source = _read_external_file(Path(source_path), "source annotation workbook")
    audit = audit_annotation_workbook(source.content)
    mapping_bytes = _read_external_file(Path(mapping_path), "session mapping")
    mapping = _mapping_from_bytes(mapping_bytes.content)
    verified_mapping = load_verified_session_mapping(
        mapping_path,
        identity_join_audit_report_path=identity_join_audit_path,
        face_crop_manifest_path=face_crop_manifest_path,
    )
    if (
        verified_mapping.artifact_file_sha256 != mapping_bytes.sha256
        or verified_mapping.lookup != canonical_to_source_lookup(mapping)
    ):
        raise ValueError("session mapping changed across evidence-bound validation")
    csv_bytes, validation = build_normalized_intake(audit, mapping)
    repository_root = Path(__file__).resolve().parents[2]
    implementation_module = _read_external_file(
        Path(__file__).resolve(), "annotation intake implementation"
    )
    implementation_cli = _read_external_file(
        repository_root / "scripts" / "import_annotation_workbook.py",
        "annotation intake command",
    )

    root = Path(restricted_root)
    source_directory = root / "annotations" / "source" / received_date
    derived_directory = root / "annotations" / "derived" / received_date
    source_filename = Path(source_path).name
    if source_filename != "MIT_INTERVIEW_Percieved_Data.xlsx":
        raise ValueError("source workbook filename does not match the recorded delivery")
    source_receipt_name = "MIT_INTERVIEW_Percieved_Data.receipt.json"
    csv_name = "mit_perceived_session_annotations.csv"
    conversion_receipt_name = "mit_perceived_session_annotations.receipt.json"

    source_receipt = {
        "schema_version": SOURCE_RECEIPT_SCHEMA_VERSION,
        "status": "source_preserved_and_verified",
        "received_date": received_date,
        "transfer_provenance": "email_attachment_reported_by_supervisor",
        "source_filename": source_filename,
        "source_file_sha256": source.sha256,
        "source_file_bytes": source.size,
        "stored_filename": source_filename,
        "byte_for_byte_copy_verified": True,
        "verification_scope": {
            "verified": [
                "file_integrity_and_ooxml_structure",
                "worksheet_headers_embedded_legend_and_source_code_vocabulary",
            ],
            "not_verified": [
                "subjective_label_correctness_against_video_content",
                "rater_identity_independence_or_collection_blinding",
            ],
        },
        "workbook": _workbook_metadata(audit),
    }
    source_receipt_bytes = _canonical_json_bytes(source_receipt)

    with AnchoredDirectory.open(
        source_directory, create=True, directory_mode=0o700
    ) as source_anchor:
        with source_anchor.exclusive_lock():
            copied = source_anchor.publish_bytes(source_filename, source.content, mode=0o600)
            source_receipt_file = source_anchor.publish_bytes(
                source_receipt_name, source_receipt_bytes, mode=0o600
            )
            source_anchor.verify_file(source_filename, copied, expected_mode=0o600)
            source_anchor.verify_file(source_receipt_name, source_receipt_file, expected_mode=0o600)

    csv_sha256 = hashlib.sha256(csv_bytes).hexdigest()
    conversion_receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "intake_schema_version": INTAKE_SCHEMA_VERSION,
        "status": "verified_provisional_session_level_intake_not_stage_bc_ready",
        "received_date": received_date,
        "source": {
            "filename": source_filename,
            "file_sha256": source.sha256,
            "file_bytes": source.size,
            "source_receipt_sha256": hashlib.sha256(source_receipt_bytes).hexdigest(),
        },
        "mapping": {
            "schema_version": mapping["schema_version"],
            "file_sha256": mapping_bytes.sha256,
            "payload_sha256": mapping["artifact_payload_sha256"],
            "canonical_mapping_applied": True,
            "live_evidence_bindings_verified": True,
            "identity_join_audit_file_sha256": (verified_mapping.identity_join_audit_file_sha256),
            "identity_join_audit_report_payload_sha256": (
                verified_mapping.identity_join_audit_report_payload_sha256
            ),
            "face_crop_manifest_file_sha256": (verified_mapping.face_crop_manifest_file_sha256),
            "face_crop_manifest_payload_sha256": (
                verified_mapping.face_crop_manifest_payload_sha256
            ),
        },
        "conversion": {
            "implementation": {
                "module": "src.data.annotation_intake",
                "module_file_sha256": implementation_module.sha256,
                "command": "scripts/import_annotation_workbook.py",
                "command_file_sha256": implementation_cli.sha256,
                "invocation_contract": [
                    "python",
                    "scripts/import_annotation_workbook.py",
                    "SOURCE_WORKBOOK",
                    "--session-mapping",
                    "SESSION_MAPPING",
                    "--identity-join-audit",
                    "IDENTITY_JOIN_AUDIT",
                    "--face-crops",
                    "FACE_CROPS",
                    "--restricted-root",
                    "RESTRICTED_ROOT",
                    "--received-date",
                    "YYYY-MM-DD",
                ],
            },
            "worksheet": audit.sheet_name,
            "expected_headers": list(EXPECTED_HEADERS),
            "normalized_columns": list(NORMALIZED_COLUMNS),
            "source_skin_tone_scale": SKIN_TONE_SCALE,
            "source_skin_tone_numeric_scale_matches_project_contract": True,
            "source_age_retained_for_reconciliation_only": True,
            "no_category_collapsing_or_skin_tone_rescaling": True,
        },
        "workbook": _workbook_metadata(audit),
        "verification_scope": {
            "verified": [
                "source_file_integrity_and_ooxml_structure",
                "headers_legend_source_vocabulary_completeness_and_duplicates",
                "evidence_bound_canonical_session_join",
                "deterministic_normalization_and_aggregate_session_consistency",
            ],
            "not_verified": [
                "subjective_label_correctness_against_video_content",
                "rater_identity_count_independence_or_collection_blinding",
                "participant_level_consensus_or_stage_bc_validity",
            ],
        },
        "validation": validation,
        "output": {
            "filename": csv_name,
            "file_sha256": csv_sha256,
            "file_bytes": len(csv_bytes),
            "row_count": len(audit.rows),
        },
        "stage_bc_acceptance": {
            "ready": False,
            "reasons": [
                "session_level_rows_not_blinded_participant_level_packets",
                "expected_source_session_rows_are_missing",
                "rater_identity_count_and_independence_are_not_established",
                "field_confidence_values_are_absent",
                "approved_codebook_and_matching_version_are_absent",
                "source_racialised_appearance_categories_are_not_the_approved_audit_vocabulary",
                "source_age_field_is_outside_the_frozen_three_construct_contract",
            ],
            "accepted_rater_table_written": False,
            "accepted_consensus_written": False,
        },
    }
    conversion_receipt_bytes = _canonical_json_bytes(conversion_receipt)
    with AnchoredDirectory.open(
        derived_directory, create=True, directory_mode=0o700
    ) as derived_anchor:
        with derived_anchor.exclusive_lock():
            csv_file = derived_anchor.publish_bytes(csv_name, csv_bytes, mode=0o600)
            receipt_file = derived_anchor.publish_bytes(
                conversion_receipt_name, conversion_receipt_bytes, mode=0o600
            )
            derived_anchor.verify_file(csv_name, csv_file, expected_mode=0o600)
            derived_anchor.verify_file(conversion_receipt_name, receipt_file, expected_mode=0o600)

    return IntakePublication(
        source_workbook=source_directory / source_filename,
        source_receipt=source_directory / source_receipt_name,
        normalized_csv=derived_directory / csv_name,
        conversion_receipt=derived_directory / conversion_receipt_name,
        source_sha256=source.sha256,
        normalized_csv_sha256=csv_sha256,
    )


def _validate_archive(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    if not infos or len(infos) > _MAX_ZIP_ENTRIES:
        raise ValueError("annotation workbook ZIP entry count is outside the accepted bound")
    total = 0
    entries: dict[str, zipfile.ZipInfo] = {}
    for info in infos:
        path = PurePosixPath(info.filename)
        unix_mode = (info.external_attr >> 16) & 0xFFFF
        if (
            info.filename.startswith("/")
            or "\\" in info.filename
            or any(part in {"", ".", ".."} for part in path.parts)
            or info.flag_bits & 0x1
            or stat.S_IFMT(unix_mode) == stat.S_IFLNK
        ):
            raise ValueError("annotation workbook contains an unsafe ZIP entry")
        if info.filename in entries:
            raise ValueError("annotation workbook contains duplicate ZIP entries")
        if info.file_size > _MAX_ENTRY_BYTES:
            raise ValueError("annotation workbook ZIP entry is too large")
        total += info.file_size
        entries[info.filename] = info
    if total > _MAX_TOTAL_BYTES:
        raise ValueError("annotation workbook expands beyond the accepted size bound")
    lower_names = {name.lower() for name in entries}
    if any(
        "vbaproject" in name
        or name.startswith("xl/externallinks/")
        or name.startswith("xl/embeddings/")
        for name in lower_names
    ):
        raise ValueError("annotation workbook contains executable or external-linked content")
    return entries


def _required_entry(
    archive: zipfile.ZipFile, entries: dict[str, zipfile.ZipInfo], name: str
) -> bytes:
    if name not in entries:
        raise ValueError(f"annotation workbook is missing {name!r}")
    return archive.read(name)


def _reject_external_relationships(
    archive: zipfile.ZipFile, entries: dict[str, zipfile.ZipInfo]
) -> None:
    for name in entries:
        if not name.endswith(".rels"):
            continue
        root = _xml_root(archive.read(name))
        for relation in root.findall(f"{{{_REL_NS}}}Relationship"):
            if relation.attrib.get("TargetMode", "Internal") != "Internal":
                raise ValueError("annotation workbook contains an external relationship")


def _xml_root(content: bytes) -> ElementTree.Element:
    if b"<!DOCTYPE" in content.upper() or b"<!ENTITY" in content.upper():
        raise ValueError("annotation workbook XML declarations are unsafe")
    try:
        return ElementTree.fromstring(content)
    except ElementTree.ParseError as error:
        raise ValueError("annotation workbook contains invalid XML") from error


def _worksheet_target(root: ElementTree.Element, relationship_id: str | None) -> str:
    if not relationship_id:
        raise ValueError("annotation worksheet has no relationship identifier")
    for relation in root.findall(f"{{{_REL_NS}}}Relationship"):
        if relation.attrib.get("Id") != relationship_id:
            continue
        relation_type = relation.attrib.get("Type", "")
        target = relation.attrib.get("Target", "")
        if not relation_type.endswith("/worksheet") or not target:
            break
        if relation.attrib.get("TargetMode", "Internal") != "Internal":
            break
        return target
    raise ValueError("annotation worksheet relationship is invalid")


def _shared_strings(
    archive: zipfile.ZipFile, entries: dict[str, zipfile.ZipInfo]
) -> tuple[str, ...]:
    if "xl/sharedStrings.xml" not in entries:
        return ()
    root = _xml_root(archive.read("xl/sharedStrings.xml"))
    values: list[str] = []
    for item in root.findall(f"{{{_MAIN_NS}}}si"):
        values.append("".join(node.text or "" for node in item.iter(f"{{{_MAIN_NS}}}t")))
    return tuple(values)


def _worksheet_values(
    worksheet: ElementTree.Element, shared_strings: tuple[str, ...]
) -> tuple[dict[int, dict[int, str]], int]:
    parsed: dict[int, dict[int, str]] = {}
    formulas = 0
    for cell in worksheet.iter(f"{{{_MAIN_NS}}}c"):
        reference = cell.attrib.get("r", "")
        match = _CELL_REFERENCE.fullmatch(reference)
        if match is None:
            raise ValueError("annotation workbook contains an invalid cell reference")
        letters, row_text = match.groups()
        column = _column_number(letters)
        row_number = int(row_text)
        if column in parsed.setdefault(row_number, {}):
            raise ValueError("annotation workbook repeats a cell reference")
        if cell.find(f"{{{_MAIN_NS}}}f") is not None:
            formulas += 1
        value = _cell_value(cell, shared_strings)
        parsed[row_number][column] = value
    return parsed, formulas


def _cell_value(cell: ElementTree.Element, shared_strings: tuple[str, ...]) -> str:
    cell_type = cell.attrib.get("t", "n")
    if cell_type == "inlineStr":
        inline = cell.find(f"{{{_MAIN_NS}}}is")
        if inline is None:
            return ""
        return "".join(node.text or "" for node in inline.iter(f"{{{_MAIN_NS}}}t"))
    value = cell.find(f"{{{_MAIN_NS}}}v")
    text = value.text if value is not None and value.text is not None else ""
    if cell_type == "s":
        try:
            index = int(text)
            if index < 0:
                raise IndexError
            return shared_strings[index]
        except (ValueError, IndexError) as error:
            raise ValueError("annotation workbook has an invalid shared-string index") from error
    if cell_type in {"n", "str"}:
        return text
    raise ValueError(f"annotation workbook uses unsupported cell type {cell_type!r}")


def _column_number(letters: str) -> int:
    result = 0
    for character in letters:
        result = result * 26 + ord(character) - ord("A") + 1
    return result


def _annotation_rows(
    worksheet_rows: dict[int, dict[int, str]],
) -> tuple[list[SourceAnnotation], int | None]:
    header = tuple(worksheet_rows.get(1, {}).get(column, "").strip() for column in range(1, 6))
    if header != EXPECTED_HEADERS:
        raise ValueError("annotation workbook headers do not match the expected five fields")
    rows: list[SourceAnnotation] = []
    seen: set[str] = set()
    legend_row: int | None = None
    legend_text: str | None = None
    for row_number in sorted(number for number in worksheet_rows if number > 1):
        values = [worksheet_rows[row_number].get(column, "").strip() for column in range(1, 6)]
        session_id = values[0]
        if not any(values):
            continue
        if _SESSION_ID.fullmatch(session_id) is None:
            if legend_row is not None:
                raise ValueError("annotation workbook contains multiple non-data content rows")
            if any(values[1:]) or any(
                value.strip()
                for column, value in worksheet_rows[row_number].items()
                if column > len(EXPECTED_HEADERS)
            ):
                raise ValueError("annotation workbook contains an unexpected non-data row")
            legend_row = row_number
            legend_text = session_id
            continue
        if legend_row is not None:
            raise ValueError("annotation workbook contains data after its legend")
        if any(
            value.strip()
            for column, value in worksheet_rows[row_number].items()
            if column > len(EXPECTED_HEADERS)
        ):
            raise ValueError(f"annotation row {row_number} has unexpected extra fields")
        if any(value == "" for value in values):
            raise ValueError(f"annotation row {row_number} has a blank required field")
        if session_id in seen:
            raise ValueError(f"annotation workbook repeats source session {session_id!r}")
        seen.add(session_id)
        gender, skin_text, racialised, age = values[1:]
        if gender not in GENDER_LABELS:
            raise ValueError(f"annotation row {row_number} has an unknown gender code")
        if racialised not in RACIALISED_APPEARANCE_LABELS:
            raise ValueError(f"annotation row {row_number} has an unknown appearance code")
        if age not in AGE_LABELS:
            raise ValueError(f"annotation row {row_number} has an unknown age code")
        try:
            skin = int(skin_text)
        except ValueError as error:
            raise ValueError(f"annotation row {row_number} has a non-integer skin value") from error
        if str(skin) != skin_text or skin not in range(1, 6):
            raise ValueError(f"annotation row {row_number} has skin value outside source scale 1-5")
        rows.append(
            SourceAnnotation(
                source_session_id=session_id,
                source_row_number=row_number,
                gender_code=gender,
                skin_tone_value=skin,
                racialised_appearance_code=racialised,
                age_code=age,
            )
        )
    if not rows:
        raise ValueError("annotation workbook contains no recognized session rows")
    if legend_text != EXPECTED_LEGEND:
        raise ValueError("annotation workbook legend does not match the source code definitions")
    return rows, legend_row


def _aggregate_validation(
    source_rows: tuple[SourceAnnotation, ...],
    canonical_rows: list[dict[str, str | int]],
    *,
    expected_source_ids: set[str],
    observed_source_ids: set[str],
) -> dict[str, Any]:
    missing = sorted(
        expected_source_ids - observed_source_ids,
        key=lambda value: (int(value.removeprefix("PP").removeprefix("P")), value.startswith("PP")),
    )
    participants: dict[str, dict[str, dict[str, str | int]]] = defaultdict(dict)
    for row in canonical_rows:
        participants[str(row["participant_id"])][str(row["phase"])] = row
    complete = [rows for rows in participants.values() if set(rows) == {"pre", "post"}]
    post_only = sum(set(rows) == {"post"} for rows in participants.values())
    pre_only = sum(set(rows) == {"pre"} for rows in participants.values())
    gender_matches = sum(
        rows["pre"]["perceived_gender_presentation_source_code"]
        == rows["post"]["perceived_gender_presentation_source_code"]
        for rows in complete
    )
    age_matches = sum(
        rows["pre"]["perceived_age_band_source_code"]
        == rows["post"]["perceived_age_band_source_code"]
        for rows in complete
    )
    racialised_matches = sum(
        rows["pre"]["perceived_racialised_appearance_source_code"]
        == rows["post"]["perceived_racialised_appearance_source_code"]
        for rows in complete
    )
    skin_differences = Counter(
        abs(
            int(rows["pre"]["perceived_skin_tone_source_value"])
            - int(rows["post"]["perceived_skin_tone_source_value"])
        )
        for rows in complete
    )
    source_counts = {
        "gender": dict(sorted(Counter(row.gender_code for row in source_rows).items())),
        "skin_tone": {
            str(key): value
            for key, value in sorted(Counter(row.skin_tone_value for row in source_rows).items())
        },
        "racialised_appearance": dict(
            sorted(Counter(row.racialised_appearance_code for row in source_rows).items())
        ),
        "age": dict(sorted(Counter(row.age_code for row in source_rows).items())),
    }
    return {
        "expected_source_session_count": len(expected_source_ids),
        "observed_source_session_count": len(observed_source_ids),
        "recognized_source_session_count": len(source_rows),
        "unexpected_source_session_count": 0,
        "duplicate_source_session_count": 0,
        "blank_required_value_count": 0,
        "missing_expected_source_session_count": len(missing),
        "missing_expected_source_session_ids": missing,
        "pre_session_count": sum(
            row.source_session_id.startswith("P") and not row.source_session_id.startswith("PP")
            for row in source_rows
        ),
        "post_session_count": sum(row.source_session_id.startswith("PP") for row in source_rows),
        "canonical_participant_count": len(participants),
        "complete_pair_count": len(complete),
        "post_only_participant_count": post_only,
        "pre_only_participant_count": pre_only,
        "source_category_counts": source_counts,
        "within_complete_pair_consistency": {
            "pair_count": len(complete),
            "gender_exact_match_count": gender_matches,
            "age_exact_match_count": age_matches,
            "racialised_appearance_exact_match_count": racialised_matches,
            "racialised_appearance_conflict_count": len(complete) - racialised_matches,
            "skin_tone_absolute_difference_counts": {
                str(key): value for key, value in sorted(skin_differences.items())
            },
        },
    }


def _core_properties(
    archive: zipfile.ZipFile, entries: dict[str, zipfile.ZipInfo]
) -> dict[str, str | None]:
    if "docProps/core.xml" not in entries:
        return {"created": None, "modified": None, "creator": None, "last_modified_by": None}
    root = _xml_root(archive.read("docProps/core.xml"))

    def text(namespace: str, name: str) -> str | None:
        node = root.find(f"{{{namespace}}}{name}")
        value = (node.text or "").strip() if node is not None else ""
        return value or None

    return {
        "created": text(_DCTERMS_NS, "created"),
        "modified": text(_DCTERMS_NS, "modified"),
        "creator": text(_DC_NS, "creator"),
        "last_modified_by": text(_CP_NS, "lastModifiedBy"),
    }


def _workbook_metadata(audit: WorkbookAudit) -> dict[str, Any]:
    value = asdict(audit)
    value.pop("rows")
    value["data_row_count"] = len(audit.rows)
    value["headers"] = list(EXPECTED_HEADERS)
    value["structurally_valid_ooxml"] = True
    return value


def _read_external_file(path: Path, label: str) -> AnchoredFile:
    absolute = path.expanduser()
    if not absolute.is_absolute():
        absolute = Path.cwd() / absolute
    if absolute.name in {"", ".", ".."}:
        raise ValueError(f"{label} must name a regular file")
    try:
        with AnchoredDirectory.open(absolute.parent, directory_mode=None) as anchor:
            return anchor.read_bytes(absolute.name, expected_mode=None)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"cannot safely read {label}") from error


def _mapping_from_bytes(content: bytes) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"session mapping repeats JSON key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("session mapping bytes are not valid canonical JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("session mapping must contain one JSON object")
    canonical = canonical_mit_session_mapping_json(payload).encode("utf-8")
    if canonical != content:
        raise ValueError("session mapping bytes are not canonically serialized")
    return payload


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _validate_iso_date(value: str) -> None:
    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value, re.ASCII) is None:
        raise ValueError("received date must use YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError("received date must be a valid calendar date") from error
    if parsed.isoformat() != value:
        raise ValueError("received date must use canonical YYYY-MM-DD")
