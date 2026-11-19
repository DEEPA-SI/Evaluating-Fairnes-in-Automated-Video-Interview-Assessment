"""Read-only, source-faithful validation of the pinned AVI-Personality annotations.

The schema follows the released ``CODEBOOK.md``, not the stale dataset-card filenames.
``aggregate`` contains no participant IDs or media filenames. The other three outputs
are restricted: ``participants[id]`` contains only the official split and outcomes,
``media[id]`` only question-to-manifest references, and ``audit_attributes[id]`` the
separate self-reported/background fields. They are not an authorised experiment or
an audit join to predictions (see docs/research-overview.md).

Media joins here concern manifest references only, not local availability, decoding,
embedded-audio quality or transcript content. Invalid inputs yield no normalized
records. Documented source blanks remain ``None``; nothing is imputed or rescaled.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import re
from collections import Counter
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from src.secure_io import AnchoredDirectory

ANNOTATION_SCHEMA_VERSION = "avi-personality-annotations-v1"
SPLIT_FILES = {"train": "Training.csv", "validation": "Validation.csv", "test": "Testing.csv"}
EXPECTED_SPLIT_COUNTS = {"train": 452, "validation": 64, "test": 130}
QUESTION_NAMES = (
    "q1_generic",
    "q2_generic",
    "q3_personality",
    "q4_personality",
    "q5_personality",
    "q6_personality",
)
AUDIT_COLUMNS = (
    "gender",
    "age",
    "work_experience",
    "job_applications",
    "video_interviews",
    "education",
    "ethnicity",
    "english_proficiency",
    "accent",
    "accent_strength",
)
PERSONALITY_OBSERVER_COLUMNS = tuple(f"p_{trait}_observer" for trait in "HXAC")
GENERIC_OBSERVER_COLUMNS = tuple(f"g_{trait}_observer" for trait in "HEXACO")
SELF_REPORT_COLUMNS = tuple(f"{trait}_self" for trait in "HEXACO")
COMPETENCY_COLUMNS = (
    "mean_rating_Integr",
    "mean_rating_Colleg",
    "mean_rating_Soc_vers",
    "mean_rating_dev_orient",
    "mean_rating_hirea",
)
SCORE_COLUMNS = (
    PERSONALITY_OBSERVER_COLUMNS
    + GENERIC_OBSERVER_COLUMNS
    + SELF_REPORT_COLUMNS
    + COMPETENCY_COLUMNS
)
OUTCOME_COLUMNS = SCORE_COLUMNS + ("g_level",)
EXPECTED_COLUMNS = ("id",) + AUDIT_COLUMNS + OUTCOME_COLUMNS
PRIMARY_TARGET = "mean_rating_hirea"
_AUDIT_NAMES = {
    "gender": "self_reported_gender_code",
    "age": "age_years",
    "work_experience": "work_experience_years",
    "job_applications": "job_applications_last_two_years",
    "video_interviews": "video_interviews_last_two_years",
    "education": "education_code",
    "ethnicity": "self_reported_ethnicity_code",
    "english_proficiency": "self_reported_english_proficiency_code",
    "accent": "self_reported_accent_code",
    "accent_strength": "self_reported_accent_strength",
}
_CATEGORY_DOMAINS = {
    "gender": frozenset({1, 2, 3, 4}),
    "education": frozenset(range(1, 8)),
    "ethnicity": frozenset({1, 2, 3, 4, 6, 7, 8, 9}),
    "english_proficiency": frozenset(range(1, 7)),
}
_INTEGER_COLUMNS = frozenset(_CATEGORY_DOMAINS) | {
    "accent",
    "job_applications",
    "video_interviews",
}
_OPTIONAL_COLUMNS = frozenset(GENERIC_OBSERVER_COLUMNS + COMPETENCY_COLUMNS)
_NONVIDEO_FILES = frozenset(
    {
        ".gitattributes",
        "CODEBOOK.md",
        "CODEBOOK_zh.md",
        "README.md",
        "user_agreement.pdf",
        *SPLIT_FILES.values(),
    }
)
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def validate_annotations(source_root: str | Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate exact columns, numeric domains, official splits and six media joins.

    ``manifest`` is the pinned source manifest, with ``repo_id``, ``revision`` and
    ``files`` entries containing ``path``, ``size``, ``blob_id`` and nullable
    ``lfs_sha256``. The three CSVs and codebook are parsed from stable no-symlink
    reads, and their consumed bytes are checked against these source identities.
    Invalid manifest structure raises ``ValueError`` without echoing source values.
    Content failures instead set ``aggregate.valid`` false and withhold all three
    restricted maps. Diagnostics contain only field/split names and counts.

    The primary-target counts mean available valid source labels, not an experiment
    eligibility decision. Category counts are marginal support counts only. No
    target-by-group analysis, correlations, prediction fitting or audit is performed.
    """
    entries = _manifest_entries(manifest)
    issues: Counter[tuple[str, str, str]] = Counter()
    source_checksums: dict[str, str] = {}
    rows_by_split: dict[str, list[dict[str, str]]] = {}
    required_files = ("CODEBOOK.md", *SPLIT_FILES.values())
    for name in required_files:
        if name not in entries:
            issues[("missing_manifest_source", "", name)] += 1
    for path in entries:
        if not path.lower().endswith(".mp4") and path not in _NONVIDEO_FILES:
            issues[("unexpected_manifest_source", "", "")] += 1

    try:
        with AnchoredDirectory.open(source_root, directory_mode=None) as anchor:
            for name in required_files:
                if name not in entries:
                    continue
                try:
                    observed = anchor.read_bytes(name, expected_mode=None)
                except (OSError, ValueError, RuntimeError):
                    issues[("unavailable_or_unsafe_source", "", name)] += 1
                    continue
                content = observed.content
                source_checksums[name] = observed.sha256
                entry = entries[name]
                if len(content) != entry["size"]:
                    issues[("source_size_mismatch", "", name)] += 1
                if entry["lfs_sha256"] is not None:
                    matches = observed.sha256 == entry["lfs_sha256"]
                else:
                    digest = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content)
                    matches = digest.hexdigest() == entry["blob_id"]
                if not matches:
                    issues[("source_hash_mismatch", "", name)] += 1
                if name == "CODEBOOK.md":
                    if not content.strip():
                        issues[("empty_codebook", "", name)] += 1
                    continue
                split = next(key for key, value in SPLIT_FILES.items() if value == name)
                rows_by_split[split] = _parse_csv(content, split, issues)
            anchor.verify_path_identity()
    except (OSError, ValueError, RuntimeError):
        issues[("unavailable_or_unsafe_source_root", "", "")] += 1

    stats = {column: _empty_stats() for column in AUDIT_COLUMNS + OUTCOME_COLUMNS}
    split_stats = {
        split: {
            "rows": 0,
            "unique_participants": 0,
            "primary_target_available": 0,
            "primary_target_missing": 0,
            "self_reported_gender_code_counts": Counter(),
            "self_reported_ethnicity_code_counts": Counter(),
        }
        for split in SPLIT_FILES
    }
    missing_blocks = {split: {"generic_observer": 0, "competency": 0} for split in SPLIT_FILES}
    participants: dict[str, dict[str, Any]] = {}
    audit_attributes: dict[str, dict[str, Any]] = {}
    expected_media: dict[str, dict[str, str]] = {}
    expected_paths: set[str] = set()
    seen_ids: dict[str, str] = {}
    for split, filename in SPLIT_FILES.items():
        rows = rows_by_split.get(split, [])
        split_stats[split]["rows"] = len(rows)
        if len(rows) != EXPECTED_SPLIT_COUNTS[split]:
            issues[("official_split_count_mismatch", split, "")] += 1
        split_ids: set[str] = set()
        available_target_ids: set[str] = set()
        missing_target_ids: set[str] = set()
        for row in rows:
            participant_id = row["id"]
            valid_id = (
                bool(participant_id)
                and participant_id == participant_id.strip()
                and participant_id not in {".", ".."}
                and not any(char in "/\\" or ord(char) < 32 for char in participant_id)
            )
            if not valid_id:
                issues[("invalid_participant_id", split, "id")] += 1
            elif participant_id in seen_ids:
                code = (
                    "duplicate_participant_id"
                    if seen_ids[participant_id] == split
                    else ("participant_split_overlap")
                )
                issues[(code, split, "id")] += 1
            else:
                seen_ids[participant_id] = split
            if valid_id:
                split_ids.add(participant_id)
            parsed: dict[str, Any] = {}
            for column in AUDIT_COLUMNS + OUTCOME_COLUMNS:
                raw = row[column]
                if raw == "":
                    value, error = None, None
                    stats[column]["missing"] += 1
                    if split != "train" or column not in _OPTIONAL_COLUMNS:
                        issues[("unexpected_missing_value", split, column)] += 1
                elif column == "g_level":
                    value = raw if raw in {"low", "normal", "high"} else None
                    error = None if value is not None else "invalid_category"
                else:
                    value, error = _numeric_value(column, raw)
                parsed[column] = value
                if error is not None:
                    stats[column]["invalid"] += 1
                    issues[(error, split, column)] += 1
                elif value is not None:
                    _observe(stats[column], column, value)
            for block, columns in (
                ("generic_observer", GENERIC_OBSERVER_COLUMNS),
                ("competency", COMPETENCY_COLUMNS),
            ):
                count = sum(row[column] == "" for column in columns)
                if count == len(columns):
                    missing_blocks[split][block] += 1
                elif count:
                    issues[("partial_missing_score_block", split, block)] += 1
            if valid_id:
                if row[PRIMARY_TARGET] == "":
                    missing_target_ids.add(participant_id)
                elif parsed[PRIMARY_TARGET] is not None:
                    available_target_ids.add(participant_id)
                for column, key in (
                    ("gender", "self_reported_gender_code_counts"),
                    ("ethnicity", "self_reported_ethnicity_code_counts"),
                ):
                    if parsed[column] is not None:
                        split_stats[split][key][str(parsed[column])] += 1
                participants[participant_id] = {
                    "split": split,
                    "outcomes": {column: parsed[column] for column in OUTCOME_COLUMNS},
                }
                audit_attributes[participant_id] = {
                    _AUDIT_NAMES[column]: parsed[column] for column in AUDIT_COLUMNS
                }
                refs = {
                    question: filename[:-4] + "/" + participant_id + "_" + question + ".mp4"
                    for question in QUESTION_NAMES
                }
                expected_media[participant_id] = refs
                expected_paths.update(refs.values())
        split_stats[split]["unique_participants"] = len(split_ids)
        split_stats[split]["primary_target_available"] = len(available_target_ids)
        split_stats[split]["primary_target_missing"] = len(missing_target_ids)
        for key in ("self_reported_gender_code_counts", "self_reported_ethnicity_code_counts"):
            split_stats[split][key] = dict(sorted(split_stats[split][key].items()))

    for split in SPLIT_FILES:
        for block, expected in (("generic_observer", 1), ("competency", 2)):
            expected = expected if split == "train" else 0
            if missing_blocks[split][block] != expected:
                issues[("documented_missingness_mismatch", split, block)] += 1
    actual_paths = {path for path in entries if path.lower().endswith(".mp4")}
    missing_paths = expected_paths - actual_paths
    extra_paths = actual_paths - expected_paths
    if missing_paths:
        issues[("missing_media_reference", "", "")] += len(missing_paths)
    if extra_paths:
        issues[("unmatched_media_reference", "", "")] += len(extra_paths)
    for state in stats.values():
        state["category_counts"] = dict(sorted(state["category_counts"].items()))
        state["total"] = sum(state[key] for key in ("present", "missing", "invalid"))
    valid = not issues
    aggregate = {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "repo_id": manifest["repo_id"],
        "revision": manifest["revision"],
        "valid": valid,
        "participant_count": len(seen_ids),
        "split_counts": split_stats,
        "expected_split_counts": dict(EXPECTED_SPLIT_COUNTS),
        "primary_target": PRIMARY_TARGET,
        "primary_target_semantics": "Recruiter-rated interview performance / hireability, BARS 1–5",
        "column_statistics": stats,
        "documented_missingness": missing_blocks,
        "media_join": {
            "scope": "manifest_references_only",
            "expected": len(expected_paths),
            "manifest_videos": len(actual_paths),
            "matched": len(expected_paths & actual_paths),
            "missing": len(missing_paths),
            "unmatched": len(extra_paths),
            "questions_per_participant": len(QUESTION_NAMES),
        },
        "source_sha256": dict(sorted(source_checksums.items())),
        "field_roles": {"outcomes": list(OUTCOME_COLUMNS), "audit_only": dict(_AUDIT_NAMES)},
        "unavailable_fields": ["skin_tone", "transcript", "g_total", "item_level_ICAR_answers"],
        "caveats": [
            "Self-reported gender codes 3 (non-binary / third gender) and 4 (prefer not to say) "
            "remain distinct; no category is merged or visually inferred.",
            "Ethnicity and accent remain self-reported numeric source codes; full label maps "
            "were not published. Accent is checked as a positive integer, not assigned labels.",
            "Age/work experience have no published upper bound; observed min/max are descriptive.",
            "Blank scores are documented source missingness, not zero; no imputation is performed.",
            "This adapter validates source labels and manifest joins only, not media quality, "
            "a prediction result, group disparities or permission to begin an experiment.",
        ],
        "errors": [
            {"code": code, "split": split or None, "column": column or None, "count": count}
            for (code, split, column), count in sorted(issues.items())
        ],
    }
    restricted = (participants, expected_media, audit_attributes) if valid else ({}, {}, {})
    return {
        "aggregate": aggregate,
        "participants": restricted[0],
        "media": restricted[1],
        "audit_attributes": restricted[2],
    }


def _manifest_entries(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(manifest, Mapping) or manifest.get("repo_id") != "APAL-SEU/AVI-Personality":
        raise ValueError("source manifest must identify APAL-SEU/AVI-Personality")
    if not isinstance(manifest.get("revision"), str) or not _HEX40.fullmatch(manifest["revision"]):
        raise ValueError("source manifest must specify a lowercase 40-character pinned revision")
    if not isinstance(manifest.get("files"), list) or not manifest["files"]:
        raise ValueError("source manifest files must be a nonempty list")
    entries: dict[str, dict[str, Any]] = {}
    for entry in manifest["files"]:
        if not isinstance(entry, Mapping):
            raise ValueError("source manifest entries must be mappings")
        path = entry.get("path")
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or "\\" in path
            or any(ord(char) < 32 for char in path)
            or any(part in {"", ".", ".."} for part in path.split("/"))
        ):
            raise ValueError("source manifest paths must be canonical relative POSIX paths")
        if path in entries:
            raise ValueError("source manifest has duplicate paths")
        size = entry.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("source manifest file size must be a nonnegative integer")
        if not isinstance(entry.get("blob_id"), str) or not _HEX40.fullmatch(entry["blob_id"]):
            raise ValueError("source manifest blob_id must be lowercase Git SHA-1")
        if "lfs_sha256" not in entry or (
            entry["lfs_sha256"] is not None
            and (
                not isinstance(entry["lfs_sha256"], str)
                or not _HEX64.fullmatch(entry["lfs_sha256"])
            )
        ):
            raise ValueError("source manifest must specify nullable lowercase lfs_sha256")
        entries[path] = dict(entry)
    return entries


def _parse_csv(
    content: bytes, split: str, issues: Counter[tuple[str, str, str]]
) -> list[dict[str, str]]:
    try:
        text = content.decode("utf-8")
        if "\0" in text:
            raise ValueError("NUL byte")
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        if tuple(next(reader, ())) != EXPECTED_COLUMNS:
            issues[("annotation_header_mismatch", split, "")] += 1
            return []
        rows = []
        for values in reader:
            if len(values) != len(EXPECTED_COLUMNS):
                issues[("annotation_row_length_mismatch", split, "")] += 1
            else:
                rows.append(dict(zip(EXPECTED_COLUMNS, values, strict=True)))
        return rows
    except (UnicodeError, csv.Error, ValueError):
        issues[("annotation_csv_parse_error", split, "")] += 1
        return []


def _numeric_value(column: str, raw: str) -> tuple[int | float | None, str | None]:
    try:
        number = Decimal(raw)
    except InvalidOperation:
        return None, "invalid_numeric"
    if not number.is_finite():
        return None, "nonfinite_numeric"
    if column in _INTEGER_COLUMNS and number != number.to_integral_value():
        return None, "noninteger_numeric"
    if column in _CATEGORY_DOMAINS:
        if number not in _CATEGORY_DOMAINS[column]:
            return None, "numeric_out_of_domain"
    elif column == "accent":
        if number < 1:
            return None, "numeric_out_of_domain"
    elif column in SCORE_COLUMNS or column == "accent_strength":
        if not 1 <= number <= 5:
            return None, "numeric_out_of_domain"
    elif number < 0:
        return None, "numeric_out_of_domain"
    value = float(number)
    if not math.isfinite(value):
        return None, "nonfinite_numeric"
    if column in _INTEGER_COLUMNS:
        return int(number), None
    return value, None


def _empty_stats() -> dict[str, Any]:
    return {
        "present": 0,
        "missing": 0,
        "invalid": 0,
        "min": None,
        "max": None,
        "category_counts": Counter(),
    }


def _observe(state: dict[str, Any], column: str, value: int | float | str) -> None:
    state["present"] += 1
    if column == "g_level" or column in _CATEGORY_DOMAINS or column == "accent":
        state["category_counts"][str(value)] += 1
    if not isinstance(value, str):
        state["min"] = value if state["min"] is None else min(value, state["min"])
        state["max"] = value if state["max"] is None else max(value, state["max"])
