"""Restricted, nonvisual supporting diagnostics for MIT source-session joins.

The report produced here is deliberately not an identity result and not an approved mapping.
It compares the supplied candidate-only transcripts and supplied pitch-related prosody fields
across the source ``P`` and ``PP`` session rosters.  It never loads outcomes, protected
annotations, face crops, media payloads, transcript text into the report, or worker identifiers.
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
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

NONVISUAL_JOIN_DIAGNOSTIC_SCHEMA_VERSION = "mit-nonvisual-join-diagnostic-1"
NONVISUAL_JOIN_DIAGNOSTIC_METHOD_VERSION = "fair-avi-mit-nonvisual-join-diagnostic-1"
SUPPORTING_ONLY = "supporting_join_diagnostic_only"
TRANSCRIPT_PARSER_REVISION = "mit-candidate-transcript-parser-2"

CANDIDATE_MARKER_ALIASES = (
    "Interviewee",
    "INterviewee",
    "Intervieweee",
    "Interveiwee",
    "Inerviewee",
)
INTERVIEWER_MARKER_ALIASES = ("Interviewer", "Intrviewer")
UNCLASSIFIED_MARKER_ALIASES = ("r",)
PITCH_FIELDS = (
    "min_pitch",
    "max_pitch",
    "mean_pitch",
    "pitch_sd",
    "pitch_abs",
    "pitch_quant",
    "pitchUvsVRatio",
    "diffPitchMaxMin",
    "diffPitchMaxMean",
    "diffPitchMaxMode",
)
EXPECTED_QUESTIONS = (1, 2, 3, 4, 5)

_SESSION_PATTERN = re.compile(r"^(PP|P)([1-9][0-9]*)$", re.ASCII)
_PROSODY_ROW_PATTERN = re.compile(r"^(PP|P)([1-9][0-9]*)Q([1-9][0-9]*)$", re.ASCII)
_SPEAKER_LIKE_PATTERN = re.compile(r"^[A-Za-z]*interview[A-Za-z]*$", re.IGNORECASE)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$", re.ASCII)

_TOP_FIELDS = {
    "schema_version",
    "status",
    "purpose",
    "mapping_approved",
    "training_input_authorized",
    "interpretation",
    "bindings",
    "method",
    "coverage",
    "diagnostics",
    "report_payload_sha256",
}

_INTERPRETATION = {
    "identity_inference": "none",
    "mapping_boundary": "separate_approved_mapping_required_before_training_use",
    "prosody_warning": ("prosody_is_not_globally_discriminative_and_must_not_define_a_mapping"),
    "transcript_warning": "lexical_similarity_is_supporting_evidence_not_identity_truth",
}

_METHOD = {
    "method_version": NONVISUAL_JOIN_DIAGNOSTIC_METHOD_VERSION,
    "axis_order": "ascending_numeric_source_participant_id",
    "session_id_contract": "canonical_uppercase_P_or_PP_plus_positive_ascii_decimal",
    "transcript": {
        "parser_revision": TRANSCRIPT_PARSER_REVISION,
        "candidate_marker_aliases": list(CANDIDATE_MARKER_ALIASES),
        "interviewer_marker_aliases": list(INTERVIEWER_MARKER_ALIASES),
        "unclassified_marker_aliases": list(UNCLASSIFIED_MARKER_ALIASES),
        "continuation_rule": "colonless_fragment_inherits_most_recent_explicit_recognised_role",
        "input_scope": "candidate_only",
        "vectorizer": {
            "implementation": "sklearn.feature_extraction.text.TfidfVectorizer",
            "analyzer": "word",
            "lowercase": True,
            "strip_accents": "unicode",
            "token_pattern": r"(?u)\b\w\w+\b",
            "ngram_range": [1, 2],
            "min_df": 1,
            "max_df": 1.0,
            "max_features": None,
            "stop_words": None,
            "binary": False,
            "sublinear_tf": True,
            "use_idf": True,
            "smooth_idf": True,
            "norm": "l2",
            "dtype": "float64",
            "fit_scope": "all_source_sessions_pre_then_post_on_numeric_axes",
        },
        "similarity": "pre_to_post_cosine",
    },
    "prosody": {
        "row_id_pattern": r"^(PP|P)([1-9][0-9]*)Q([1-9][0-9]*)$",
        "selected_fields": list(PITCH_FIELDS),
        "expected_questions": list(EXPECTED_QUESTIONS),
        "finite_values_required": True,
        "session_aggregation": "arithmetic_mean_across_Q1_to_Q5_per_selected_field",
        "standardization": (
            "sklearn_StandardScaler_population_fit_over_all_source_sessions_pre_then_post"
        ),
        "similarity": "pre_to_post_cosine",
        "global_discrimination_use": "prohibited",
    },
    "ranking": {
        "nominal_rank": "one_based_count_strictly_greater_plus_one",
        "row_order": "descending_cosine_with_stable_numeric_axis_tie_order",
        "margin": "row_best_minus_row_second_best",
    },
    "assignment": {
        "algorithm": "scipy_linear_sum_assignment_maximum_cosine",
        "objective": "maximum_total_original_cosine",
        "axes": "numeric_source_pre_rows_and_post_columns",
        "interpretation": "supporting_diagnostic_assignment_not_identity_mapping",
    },
    "matrix_storage": {
        "dtype": "float64",
        "byte_order": "little_endian",
        "layout": "C",
        "hash": "sha256",
    },
}


def parse_source_session_id(value: object) -> tuple[str, str, str]:
    """Return canonical source-session ID, numeric component and phase."""
    if not isinstance(value, str):
        raise ValueError("source session ID must be a string")
    token = value.strip().upper()
    match = _SESSION_PATTERN.fullmatch(token)
    if match is None:
        raise ValueError(f"invalid source session ID {value!r}")
    prefix, number = match.groups()
    participant_number = str(int(number))
    return (
        f"{prefix}{participant_number}",
        participant_number,
        "post" if prefix == "PP" else "pre",
    )


def parse_candidate_only_transcript(full_dialogue: str) -> tuple[str, dict[str, int]]:
    """Parse candidate text with the frozen exact-role and continuation contract."""
    if not isinstance(full_dialogue, str) or not full_dialogue.strip() or "\0" in full_dialogue:
        raise ValueError("full transcript must be non-empty and NUL-free")
    candidate_aliases = set(CANDIDATE_MARKER_ALIASES)
    interviewer_aliases = set(INTERVIEWER_MARKER_ALIASES)
    unclassified_aliases = set(UNCLASSIFIED_MARKER_ALIASES)
    utterances: list[str] = []
    counts: Counter[str] = Counter()
    preceding_role: str | None = None
    for raw_segment in full_dialogue.split("|"):
        segment = raw_segment.strip()
        if not segment:
            continue
        if ":" not in segment:
            cleaned = " ".join(segment.split())
            if preceding_role == "candidate" and cleaned:
                utterances.append(cleaned)
                counts["retained_candidate_continuation_count"] += 1
                counts["retained_candidate_continuation_characters"] += len(segment)
            else:
                counts["excluded_colonless_fragment_count"] += 1
                counts["excluded_colonless_characters"] += len(segment)
            continue
        marker, content = segment.split(":", maxsplit=1)
        exact_marker = marker.strip()
        if exact_marker in candidate_aliases:
            preceding_role = "candidate"
            cleaned = " ".join(content.split())
            if cleaned:
                utterances.append(cleaned)
                counts["explicit_candidate_segment_count"] += 1
        elif exact_marker in interviewer_aliases:
            preceding_role = "interviewer"
            counts["excluded_interviewer_segment_count"] += 1
        elif exact_marker in unclassified_aliases:
            preceding_role = "unclassified"
            counts["excluded_unclassified_marker_count"] += 1
        elif _SPEAKER_LIKE_PATTERN.fullmatch(exact_marker):
            raise ValueError(f"unknown speaker marker {exact_marker!r}")
        else:
            raise ValueError(f"unknown colon-labelled marker {exact_marker!r}")
    if not utterances:
        raise ValueError("transcript contains no recognised candidate utterance")
    complete_counts = {
        field: int(counts[field])
        for field in (
            "explicit_candidate_segment_count",
            "retained_candidate_continuation_count",
            "retained_candidate_continuation_characters",
            "excluded_colonless_fragment_count",
            "excluded_colonless_characters",
            "excluded_interviewer_segment_count",
            "excluded_unclassified_marker_count",
        )
    }
    return " ".join(utterances), complete_counts


def analyse_join_similarity(
    pre_source_session_ids: Sequence[str],
    post_source_session_ids: Sequence[str],
    similarities: np.ndarray,
) -> dict[str, Any]:
    """Return deterministic row evidence and a maximum-cosine Hungarian assignment."""
    pre_ids = tuple(pre_source_session_ids)
    post_ids = tuple(post_source_session_ids)
    _validate_axes(pre_ids, post_ids)
    raw_matrix = np.asarray(similarities)
    if raw_matrix.dtype.kind not in {"f", "i"}:
        raise ValueError("join similarity matrix must contain only numeric values")
    matrix = np.asarray(raw_matrix, dtype=np.float64)
    if matrix.shape != (len(pre_ids), len(post_ids)) or not np.isfinite(matrix).all():
        raise ValueError("join similarity matrix has invalid shape or non-finite values")
    if np.any(matrix < -1.0) or np.any(matrix > 1.0):
        raise ValueError("join cosine similarities must be in [-1,1]")
    if len(pre_ids) < 2:
        raise ValueError("join diagnostic requires at least two participants")

    row_evidence: list[dict[str, Any]] = []
    nominal_ranks: list[int] = []
    row_top_nominal_count = 0
    for row_index, pre_id in enumerate(pre_ids):
        order = np.argsort(-matrix[row_index], kind="stable")
        nominal_post_id = f"PP{_session_number(pre_id)}"
        nominal_column = post_ids.index(nominal_post_id)
        best_column = int(order[0])
        second_column = int(order[1])
        nominal_rank = 1 + int(np.sum(matrix[row_index] > matrix[row_index, nominal_column]))
        best_margin = float(matrix[row_index, best_column] - matrix[row_index, second_column])
        if best_column == nominal_column:
            row_top_nominal_count += 1
        nominal_ranks.append(nominal_rank)
        row_evidence.append(
            {
                "pre_source_session_id": pre_id,
                "nominal_post_source_session_id": nominal_post_id,
                "nominal_similarity": _python_float(matrix[row_index, nominal_column]),
                "nominal_rank": nominal_rank,
                "best_post_source_session_id": post_ids[best_column],
                "best_similarity": _python_float(matrix[row_index, best_column]),
                "best_margin": _python_float(best_margin),
            }
        )

    row_indices, assigned_columns = linear_sum_assignment(-matrix)
    if not np.array_equal(row_indices, np.arange(len(pre_ids))):
        raise ValueError("Hungarian assignment did not cover every pre-session row")
    if len({int(value) for value in assigned_columns}) != len(post_ids):
        raise ValueError("Hungarian assignment did not cover every post-session column")

    pairs: list[dict[str, Any]] = []
    changed: list[dict[str, str]] = []
    nominal_count = 0
    for row_index, assigned_value in zip(row_indices, assigned_columns, strict=True):
        assigned_column = int(assigned_value)
        pre_id = pre_ids[int(row_index)]
        nominal_post_id = f"PP{_session_number(pre_id)}"
        nominal_column = post_ids.index(nominal_post_id)
        order = np.argsort(-matrix[int(row_index)], kind="stable")
        assigned_rank = 1 + int(
            np.sum(matrix[int(row_index)] > matrix[int(row_index), assigned_column])
        )
        assigned_post_id = post_ids[assigned_column]
        equals_nominal = assigned_column == nominal_column
        if equals_nominal:
            nominal_count += 1
        else:
            changed.append(
                {
                    "pre_source_session_id": pre_id,
                    "nominal_post_source_session_id": nominal_post_id,
                    "assigned_post_source_session_id": assigned_post_id,
                }
            )
        pairs.append(
            {
                "pre_source_session_id": pre_id,
                "post_source_session_id": assigned_post_id,
                "cosine_similarity": _python_float(matrix[int(row_index), assigned_column]),
                "assigned_rank": assigned_rank,
                "equals_row_best": assigned_column == int(order[0]),
                "equals_nominal": equals_nominal,
            }
        )

    rank_counts = Counter(nominal_ranks)
    nominal_total = float(np.trace(matrix))
    assigned_total = float(matrix[row_indices, assigned_columns].sum())
    return {
        "row_evidence": row_evidence,
        "summary": {
            "participant_count": len(pre_ids),
            "row_top_nominal_count": row_top_nominal_count,
            "row_top_non_nominal_count": len(pre_ids) - row_top_nominal_count,
            "nominal_rank_counts": [
                {"rank": rank, "count": rank_counts[rank]} for rank in sorted(rank_counts)
            ],
        },
        "hungarian": {
            "algorithm": "scipy_linear_sum_assignment_maximum_cosine",
            "one_to_one": True,
            "pair_count": len(pairs),
            "nominal_pair_count": nominal_count,
            "changed_nominal_pair_count": len(changed),
            "nominal_total_similarity": _python_float(nominal_total),
            "assigned_total_similarity": _python_float(assigned_total),
            "objective_gain_vs_nominal": _python_float(assigned_total - nominal_total),
            "pairs": pairs,
        },
        "changed_nominal_pairs": changed,
    }


def build_nonvisual_join_diagnostic(
    *,
    video_source_session_ids: Sequence[str],
    audio_source_session_ids: Sequence[str],
    transcript_csv_bytes: bytes,
    prosody_csv_bytes: bytes,
    expected_participant_count: int,
) -> dict[str, Any]:
    """Build a self-checking report from source rosters and the two permitted CSV inputs."""
    if (
        isinstance(expected_participant_count, bool)
        or not isinstance(expected_participant_count, int)
        or expected_participant_count < 2
    ):
        raise ValueError("expected participant count must be an integer of at least two")
    if not isinstance(transcript_csv_bytes, bytes) or not isinstance(prosody_csv_bytes, bytes):
        raise TypeError("CSV inputs must be exact bytes")

    video_ids = _canonical_source_roster(video_source_session_ids, "video")
    audio_ids = _canonical_source_roster(audio_source_session_ids, "audio")
    if video_ids != audio_ids:
        raise ValueError("video and audio source-session rosters differ")
    pre_ids, post_ids = _partition_roster(video_ids, expected_participant_count)

    transcript_texts, transcript_counts = _load_candidate_transcripts(transcript_csv_bytes)
    if set(transcript_texts) != set(video_ids):
        raise ValueError(_coverage_message("transcript", set(video_ids), set(transcript_texts)))
    prosody, prosody_row_count = _load_pitch_prosody(prosody_csv_bytes)
    if set(prosody) != set(video_ids):
        raise ValueError(_coverage_message("prosody", set(video_ids), set(prosody)))

    ordered_session_ids = (*pre_ids, *post_ids)
    candidate_documents = [transcript_texts[value] for value in ordered_session_ids]
    candidate_text_binding = _canonical_sha256(
        {
            "candidate_text_sha256_by_source_session": [
                {
                    "source_session_id": session_id,
                    "sha256": hashlib.sha256(
                        transcript_texts[session_id].encode("utf-8")
                    ).hexdigest(),
                }
                for session_id in ordered_session_ids
            ]
        }
    )
    vectorizer = TfidfVectorizer(
        analyzer="word",
        lowercase=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 2),
        min_df=1,
        max_df=1.0,
        max_features=None,
        stop_words=None,
        binary=False,
        sublinear_tf=True,
        use_idf=True,
        smooth_idf=True,
        norm="l2",
        dtype=np.float64,
    )
    try:
        transcript_vectors = vectorizer.fit_transform(candidate_documents)
    except ValueError as error:
        raise ValueError("candidate transcript TF-IDF fitting failed") from error
    if transcript_vectors.shape[0] != 2 * expected_participant_count:
        raise ValueError("candidate transcript vector coverage drifted")
    transcript_matrix = cosine_similarity(
        transcript_vectors[:expected_participant_count],
        transcript_vectors[expected_participant_count:],
        dense_output=True,
    )

    signatures = np.asarray(
        [
            np.mean(
                np.stack([prosody[session_id][question] for question in EXPECTED_QUESTIONS]),
                axis=0,
            )
            for session_id in ordered_session_ids
        ],
        dtype=np.float64,
    )
    if signatures.shape != (2 * expected_participant_count, len(PITCH_FIELDS)):
        raise ValueError("prosody session-signature coverage drifted")
    if not np.isfinite(signatures).all():
        raise ValueError("prosody session signatures contain non-finite values")
    standardizer = StandardScaler(with_mean=True, with_std=True).fit(signatures)
    if (
        standardizer.var_ is None
        or standardizer.scale_ is None
        or not np.isfinite(standardizer.var_).all()
        or np.any(standardizer.var_ <= 0)
    ):
        raise ValueError("selected prosody fields must each have positive finite variance")
    standardized = standardizer.transform(signatures)
    if not np.isfinite(standardized).all() or np.any(
        np.linalg.norm(standardized, axis=1) <= np.finfo(np.float64).eps
    ):
        raise ValueError("standardized prosody signatures must have positive finite norms")
    prosody_matrix = cosine_similarity(
        standardized[:expected_participant_count],
        standardized[expected_participant_count:],
        dense_output=True,
    )

    transcript_diagnostic = _diagnostic_payload(
        pre_ids,
        post_ids,
        transcript_matrix,
        extra_summary={"tfidf_feature_count": len(vectorizer.vocabulary_)},
    )
    prosody_diagnostic = _diagnostic_payload(pre_ids, post_ids, prosody_matrix)
    roster_binding = _roster_sha256(pre_ids, post_ids)
    bindings = {
        "transcript_csv_sha256": hashlib.sha256(transcript_csv_bytes).hexdigest(),
        "prosody_csv_sha256": hashlib.sha256(prosody_csv_bytes).hexdigest(),
        "source_session_roster_sha256": roster_binding,
        "candidate_text_set_sha256": candidate_text_binding,
        "prosody_session_signature_float64_little_endian_sha256": _matrix_sha256(signatures),
        "prosody_standardizer_mean_float64_little_endian_sha256": _matrix_sha256(
            np.asarray(standardizer.mean_, dtype=np.float64)
        ),
        "prosody_standardizer_scale_float64_little_endian_sha256": _matrix_sha256(
            np.asarray(standardizer.scale_, dtype=np.float64)
        ),
        "runtime": {
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "scikit_learn_version": sklearn.__version__,
        },
    }
    coverage = {
        "expected_participant_count": expected_participant_count,
        "source_session_count": 2 * expected_participant_count,
        "pre_source_session_count": expected_participant_count,
        "post_source_session_count": expected_participant_count,
        "video_roster_session_count": len(video_ids),
        "audio_roster_session_count": len(audio_ids),
        "transcript_session_count": len(transcript_texts),
        "prosody_session_count": len(prosody),
        "prosody_question_row_count": prosody_row_count,
        "questions_per_session": len(EXPECTED_QUESTIONS),
        "selected_prosody_field_count": len(PITCH_FIELDS),
        "selected_prosody_finite_value_count": prosody_row_count * len(PITCH_FIELDS),
        **transcript_counts,
        "complete_source_roster_coverage_verified": True,
        "complete_transcript_coverage_verified": True,
        "complete_prosody_coverage_verified": True,
    }
    unsigned: dict[str, Any] = {
        "schema_version": NONVISUAL_JOIN_DIAGNOSTIC_SCHEMA_VERSION,
        "status": SUPPORTING_ONLY,
        "purpose": SUPPORTING_ONLY,
        "mapping_approved": False,
        "training_input_authorized": False,
        "interpretation": dict(_INTERPRETATION),
        "bindings": bindings,
        "method": json.loads(json.dumps(_METHOD)),
        "coverage": coverage,
        "diagnostics": {
            "candidate_transcript_tfidf": transcript_diagnostic,
            "pitch_prosody": prosody_diagnostic,
        },
    }
    report = {**unsigned, "report_payload_sha256": _canonical_sha256(unsigned)}
    validate_nonvisual_join_diagnostic(report)
    return report


def audit_nonvisual_join_dataset(
    dataset_root: str | Path, *, expected_participant_count: int = 69
) -> dict[str, Any]:
    """Read only source rosters, candidate transcripts and prosody and build the report."""
    root = _regular_dataset_root(dataset_root)
    video_ids = _source_session_files(root / "Videos", ".avi", root)
    audio_ids = _source_session_files(root / "Audio", ".wav", root)
    transcript_bytes = _read_exact_regular_file(
        root / "Labels" / "interview_transcripts_by_turkers.csv", root, "transcript CSV"
    )
    prosody_bytes = _read_exact_regular_file(
        root / "Prosody" / "prosodic_features.csv", root, "prosody CSV"
    )
    return build_nonvisual_join_diagnostic(
        video_source_session_ids=video_ids,
        audio_source_session_ids=audio_ids,
        transcript_csv_bytes=transcript_bytes,
        prosody_csv_bytes=prosody_bytes,
        expected_participant_count=expected_participant_count,
    )


def canonical_nonvisual_join_diagnostic_json(payload: Mapping[str, Any]) -> str:
    """Return stable human-readable JSON after complete semantic validation."""
    validate_nonvisual_join_diagnostic(payload)
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"


def load_nonvisual_join_diagnostic(path: str | Path) -> dict[str, Any]:
    """Load a duplicate-key-safe, checksum-bound supporting diagnostic."""
    source = Path(path)
    raw = _read_stable_regular_file(source, "nonvisual join diagnostic")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {value}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("cannot read a valid nonvisual join diagnostic") from error
    validate_nonvisual_join_diagnostic(payload)
    canonical = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if raw != canonical:
        raise ValueError("nonvisual join diagnostic does not use canonical serialization")
    return dict(payload)


def validate_nonvisual_join_diagnostic(payload: Mapping[str, Any]) -> None:
    """Fail closed on schema, types, checksums, coverage or derived diagnostic drift."""
    if not isinstance(payload, Mapping) or set(payload) != _TOP_FIELDS:
        raise ValueError("nonvisual join diagnostic has an unexpected top-level schema")
    if payload["schema_version"] != NONVISUAL_JOIN_DIAGNOSTIC_SCHEMA_VERSION:
        raise ValueError("unsupported nonvisual join diagnostic schema version")
    if (
        payload["status"] != SUPPORTING_ONLY
        or payload["purpose"] != SUPPORTING_ONLY
        or payload["mapping_approved"] is not False
        or payload["training_input_authorized"] is not False
    ):
        raise ValueError("nonvisual supporting-only boundary was weakened")
    _require_exact_value(payload["interpretation"], _INTERPRETATION, "interpretation")
    _require_exact_value(payload["method"], _METHOD, "method")

    bindings = payload["bindings"]
    binding_fields = {
        "transcript_csv_sha256",
        "prosody_csv_sha256",
        "source_session_roster_sha256",
        "candidate_text_set_sha256",
        "prosody_session_signature_float64_little_endian_sha256",
        "prosody_standardizer_mean_float64_little_endian_sha256",
        "prosody_standardizer_scale_float64_little_endian_sha256",
        "runtime",
    }
    if not isinstance(bindings, Mapping) or set(bindings) != binding_fields:
        raise ValueError("nonvisual join diagnostic bindings schema drifted")
    for field in binding_fields - {"runtime"}:
        _validated_sha256(bindings[field], f"bindings.{field}")
    runtime = bindings["runtime"]
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "numpy_version",
        "scipy_version",
        "scikit_learn_version",
    }:
        raise ValueError("nonvisual join diagnostic runtime schema drifted")
    if any(
        not isinstance(runtime[field], str) or not runtime[field] or "\0" in runtime[field]
        for field in runtime
    ):
        raise ValueError("nonvisual join diagnostic runtime versions must be strings")

    coverage = payload["coverage"]
    count_fields = {
        "expected_participant_count",
        "source_session_count",
        "pre_source_session_count",
        "post_source_session_count",
        "video_roster_session_count",
        "audio_roster_session_count",
        "transcript_session_count",
        "prosody_session_count",
        "prosody_question_row_count",
        "questions_per_session",
        "selected_prosody_field_count",
        "selected_prosody_finite_value_count",
        "explicit_candidate_segment_count",
        "retained_candidate_continuation_count",
        "retained_candidate_continuation_characters",
        "excluded_colonless_fragment_count",
        "excluded_colonless_characters",
        "excluded_interviewer_segment_count",
        "excluded_unclassified_marker_count",
    }
    boolean_fields = {
        "complete_source_roster_coverage_verified",
        "complete_transcript_coverage_verified",
        "complete_prosody_coverage_verified",
    }
    if not isinstance(coverage, Mapping) or set(coverage) != count_fields | boolean_fields:
        raise ValueError("nonvisual join diagnostic coverage schema drifted")
    if any(type(coverage[field]) is not int or coverage[field] < 0 for field in count_fields):
        raise ValueError("nonvisual join diagnostic coverage counts must be nonnegative integers")
    if any(coverage[field] is not True for field in boolean_fields):
        raise ValueError("nonvisual join diagnostic does not prove complete coverage")
    participant_count = coverage["expected_participant_count"]
    session_count = 2 * participant_count
    if participant_count < 2:
        raise ValueError("nonvisual join diagnostic participant coverage is too small")
    expected_counts = {
        "source_session_count": session_count,
        "pre_source_session_count": participant_count,
        "post_source_session_count": participant_count,
        "video_roster_session_count": session_count,
        "audio_roster_session_count": session_count,
        "transcript_session_count": session_count,
        "prosody_session_count": session_count,
        "prosody_question_row_count": session_count * len(EXPECTED_QUESTIONS),
        "questions_per_session": len(EXPECTED_QUESTIONS),
        "selected_prosody_field_count": len(PITCH_FIELDS),
        "selected_prosody_finite_value_count": (
            session_count * len(EXPECTED_QUESTIONS) * len(PITCH_FIELDS)
        ),
    }
    if any(coverage[field] != value for field, value in expected_counts.items()):
        raise ValueError("nonvisual join diagnostic coverage counts are inconsistent")
    if coverage["explicit_candidate_segment_count"] < session_count:
        raise ValueError("every transcript must contain explicit candidate speech")

    diagnostics = payload["diagnostics"]
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != {
        "candidate_transcript_tfidf",
        "pitch_prosody",
    }:
        raise ValueError("nonvisual join diagnostic modality schema drifted")
    transcript_axes = _validate_diagnostic_payload(
        diagnostics["candidate_transcript_tfidf"],
        participant_count,
        allow_extra_summary_field=True,
    )
    prosody_axes = _validate_diagnostic_payload(
        diagnostics["pitch_prosody"],
        participant_count,
        allow_extra_summary_field=False,
    )
    if transcript_axes != prosody_axes:
        raise ValueError("nonvisual join diagnostic modality axes differ")
    pre_ids, post_ids = transcript_axes
    if _roster_sha256(pre_ids, post_ids) != bindings["source_session_roster_sha256"]:
        raise ValueError("nonvisual join diagnostic source roster checksum does not match")

    checksum = payload["report_payload_sha256"]
    _validated_sha256(checksum, "report_payload_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "report_payload_sha256"}
    if _canonical_sha256(unsigned) != checksum:
        raise ValueError("nonvisual join diagnostic payload checksum does not match")


def _diagnostic_payload(
    pre_ids: Sequence[str],
    post_ids: Sequence[str],
    matrix: np.ndarray,
    *,
    extra_summary: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (len(pre_ids), len(post_ids)) or not np.isfinite(values).all():
        raise ValueError("diagnostic matrix has invalid shape or values")
    values[values == 0] = 0.0
    analysis = analyse_join_similarity(pre_ids, post_ids, values)
    if extra_summary:
        for key, value in extra_summary.items():
            if type(value) is not int or value < 1:
                raise ValueError("diagnostic extra summary values must be positive integers")
            analysis["summary"][key] = value
    return {
        "matrix": {
            "pre_source_session_ids": list(pre_ids),
            "post_source_session_ids": list(post_ids),
            "shape": [len(pre_ids), len(post_ids)],
            "values": values.tolist(),
            "float64_little_endian_sha256": _matrix_sha256(values),
        },
        **analysis,
    }


def _validate_diagnostic_payload(
    payload: object,
    participant_count: int,
    *,
    allow_extra_summary_field: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not isinstance(payload, Mapping) or set(payload) != {
        "matrix",
        "row_evidence",
        "summary",
        "hungarian",
        "changed_nominal_pairs",
    }:
        raise ValueError("nonvisual join diagnostic payload schema drifted")
    matrix_payload = payload["matrix"]
    if not isinstance(matrix_payload, Mapping) or set(matrix_payload) != {
        "pre_source_session_ids",
        "post_source_session_ids",
        "shape",
        "values",
        "float64_little_endian_sha256",
    }:
        raise ValueError("nonvisual join diagnostic matrix schema drifted")
    pre_value = matrix_payload["pre_source_session_ids"]
    post_value = matrix_payload["post_source_session_ids"]
    if not isinstance(pre_value, list) or not isinstance(post_value, list):
        raise ValueError("nonvisual join diagnostic matrix axes must be arrays")
    pre_ids = tuple(pre_value)
    post_ids = tuple(post_value)
    _validate_axes(pre_ids, post_ids)
    if len(pre_ids) != participant_count:
        raise ValueError("nonvisual join diagnostic matrix axes have wrong coverage")
    shape = matrix_payload["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(value) is not int for value in shape)
        or shape != [participant_count, participant_count]
    ):
        raise ValueError("nonvisual join diagnostic matrix shape field drifted")
    matrix = _json_float_matrix(matrix_payload["values"])
    if matrix.shape != (participant_count, participant_count):
        raise ValueError("nonvisual join diagnostic matrix values have wrong shape")
    if not np.isfinite(matrix).all() or np.any(matrix < -1.0) or np.any(matrix > 1.0):
        raise ValueError("nonvisual join diagnostic matrix values are invalid")
    matrix_sha = _validated_sha256(
        matrix_payload["float64_little_endian_sha256"], "diagnostic matrix checksum"
    )
    if _matrix_sha256(matrix) != matrix_sha:
        raise ValueError("nonvisual join diagnostic matrix checksum does not match")
    expected = analyse_join_similarity(pre_ids, post_ids, matrix)
    summary = payload["summary"]
    if allow_extra_summary_field:
        if not isinstance(summary, Mapping) or set(summary) != set(expected["summary"]) | {
            "tfidf_feature_count"
        }:
            raise ValueError("transcript diagnostic summary schema drifted")
        if type(summary["tfidf_feature_count"]) is not int or summary["tfidf_feature_count"] < 1:
            raise ValueError("TF-IDF feature count must be a positive integer")
        expected["summary"]["tfidf_feature_count"] = summary["tfidf_feature_count"]
    if not _same_json_value(payload["row_evidence"], expected["row_evidence"]):
        raise ValueError("nonvisual join row ranks or margins are inconsistent")
    if not _same_json_value(summary, expected["summary"]):
        raise ValueError("nonvisual join diagnostic summary is inconsistent")
    if not _same_json_value(payload["hungarian"], expected["hungarian"]):
        raise ValueError("nonvisual join Hungarian result is inconsistent")
    if not _same_json_value(payload["changed_nominal_pairs"], expected["changed_nominal_pairs"]):
        raise ValueError("nonvisual join changed-pair list is inconsistent")
    return pre_ids, post_ids


def _load_candidate_transcripts(content: bytes) -> tuple[dict[str, str], dict[str, int]]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("cannot decode supplied transcript CSV as UTF-8") from error
    indexed: dict[str, str] = {}
    aggregate: Counter[str] = Counter()
    try:
        reader = csv.reader(io.StringIO(text, newline=""))
        for row_number, row in enumerate(reader, start=1):
            if len(row) != 2:
                raise ValueError(f"transcript row {row_number} must have exactly two fields")
            session_id, _, _ = parse_source_session_id(row[0])
            if session_id in indexed:
                raise ValueError(f"duplicate transcript source session {session_id}")
            candidate_text, counts = parse_candidate_only_transcript(row[1])
            indexed[session_id] = candidate_text
            aggregate.update(counts)
    except csv.Error as error:
        raise ValueError("cannot parse supplied transcript CSV") from error
    if not indexed:
        raise ValueError("supplied transcript CSV is empty")
    fields = (
        "explicit_candidate_segment_count",
        "retained_candidate_continuation_count",
        "retained_candidate_continuation_characters",
        "excluded_colonless_fragment_count",
        "excluded_colonless_characters",
        "excluded_interviewer_segment_count",
        "excluded_unclassified_marker_count",
    )
    return indexed, {field: int(aggregate[field]) for field in fields}


def _load_pitch_prosody(
    content: bytes,
) -> tuple[dict[str, dict[int, np.ndarray]], int]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("cannot decode supplied prosody CSV as UTF-8") from error
    indexed: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""))
        if (
            reader.fieldnames is None
            or any(not isinstance(field, str) or not field for field in reader.fieldnames)
            or len(reader.fieldnames) != len(set(reader.fieldnames))
        ):
            raise ValueError("prosody CSV must have a unique non-empty header")
        required = {"participant&question", *PITCH_FIELDS}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"prosody CSV is missing required columns: {sorted(missing)}")
        row_count = 0
        for row_number, row in enumerate(reader, start=2):
            if None in row or set(row) != set(reader.fieldnames):
                raise ValueError(f"prosody row {row_number} has an invalid width")
            raw_id = row["participant&question"]
            if not isinstance(raw_id, str):
                raise ValueError(f"prosody row {row_number} has no source-question ID")
            match = _PROSODY_ROW_PATTERN.fullmatch(raw_id.strip().upper())
            if match is None:
                raise ValueError(f"invalid prosody source-question ID {raw_id!r}")
            prefix, number, question_value = match.groups()
            session_id = f"{prefix}{int(number)}"
            question = int(question_value)
            if question not in EXPECTED_QUESTIONS:
                raise ValueError("prosody questions must be exactly Q1 through Q5")
            if question in indexed[session_id]:
                raise ValueError(f"duplicate prosody row for {session_id}Q{question}")
            values: list[float] = []
            for field in PITCH_FIELDS:
                raw_value = row[field]
                if not isinstance(raw_value, str) or not raw_value.strip():
                    raise ValueError(f"prosody row {row_number} field {field} is missing")
                try:
                    value = float(raw_value)
                except ValueError as error:
                    raise ValueError(
                        f"prosody row {row_number} field {field} is not numeric"
                    ) from error
                if not math.isfinite(value):
                    raise ValueError(f"prosody row {row_number} field {field} is not finite")
                values.append(value)
            indexed[session_id][question] = np.asarray(values, dtype=np.float64)
            row_count += 1
    except csv.Error as error:
        raise ValueError("cannot parse supplied prosody CSV") from error
    if not indexed:
        raise ValueError("supplied prosody CSV is empty")
    for session_id, questions in indexed.items():
        if set(questions) != set(EXPECTED_QUESTIONS):
            raise ValueError(f"prosody session {session_id} must contain exactly Q1 through Q5")
    return dict(indexed), row_count


def _canonical_source_roster(values: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} source-session roster must be a sequence of IDs")
    canonical: list[str] = []
    for value in values:
        session_id, _, _ = parse_source_session_id(value)
        if value != session_id:
            raise ValueError(f"{label} source-session roster IDs must be canonical")
        canonical.append(session_id)
    if not canonical or len(canonical) != len(set(canonical)):
        raise ValueError(f"{label} source-session roster must be non-empty and unique")
    return tuple(sorted(canonical, key=_source_session_sort_key))


def _partition_roster(
    source_session_ids: Sequence[str], expected_participant_count: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    pre = tuple(value for value in source_session_ids if not value.startswith("PP"))
    post = tuple(value for value in source_session_ids if value.startswith("PP"))
    _validate_axes(pre, post)
    if len(pre) != expected_participant_count:
        raise ValueError("source-session roster does not match expected participant count")
    return pre, post


def _validate_axes(pre_ids: Sequence[str], post_ids: Sequence[str]) -> None:
    if not pre_ids or len(pre_ids) != len(post_ids):
        raise ValueError("pre/post source-session axes must be non-empty and equally sized")
    for values, prefix in ((pre_ids, "P"), (post_ids, "PP")):
        if not all(isinstance(value, str) for value in values):
            raise ValueError("source-session axes must contain strings")
        canonical = tuple(parse_source_session_id(value)[0] for value in values)
        if canonical != tuple(values):
            raise ValueError("source-session axes must use canonical IDs")
        expected_phase = "post" if prefix == "PP" else "pre"
        if any(parse_source_session_id(value)[2] != expected_phase for value in values):
            raise ValueError(f"source-session axis must contain only {prefix} IDs")
        if tuple(values) != tuple(sorted(values, key=_source_session_sort_key)):
            raise ValueError("source-session axes must use ascending numeric order")
        if len(values) != len(set(values)):
            raise ValueError("source-session axes must be unique")
    if tuple(_session_number(value) for value in pre_ids) != tuple(
        _session_number(value) for value in post_ids
    ):
        raise ValueError("pre/post source-session axes must have the same numeric coverage")


def _source_session_sort_key(value: str) -> tuple[int, int]:
    session_id, number, phase = parse_source_session_id(value)
    del session_id
    return int(number), 1 if phase == "post" else 0


def _session_number(value: str) -> str:
    return parse_source_session_id(value)[1]


def _source_session_files(directory: Path, suffix: str, root: Path) -> tuple[str, ...]:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"source roster directory {directory.name} must be regular")
    resolved_directory = directory.resolve()
    if not resolved_directory.is_relative_to(root):
        raise ValueError("source roster directory escaped dataset root")
    values: list[str] = []
    try:
        for child in directory.iterdir():
            if child.suffix.lower() != suffix:
                continue
            if child.is_symlink() or not child.is_file():
                raise ValueError("source roster entry must be a regular non-symlink file")
            values.append(parse_source_session_id(child.stem)[0])
    except OSError as error:
        raise ValueError("cannot inspect source-session roster") from error
    return _canonical_source_roster(values, directory.name)


def _regular_dataset_root(value: str | Path) -> Path:
    root = Path(value)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("dataset root must be a regular non-symlink directory")
    return root.resolve()


def _read_exact_regular_file(path: Path, root: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file")
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} escaped dataset root")
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} path may not contain symlink components")
    return _read_stable_regular_file(path, label)


def _read_stable_regular_file(path: Path, label: str) -> bytes:
    """Read one regular inode without following a final symlink or path replacement."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} must be a regular non-symlink file") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular non-symlink file")
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        after = os.fstat(descriptor)
    except OSError as error:
        raise ValueError(f"cannot read {label}") from error
    finally:
        os.close(descriptor)
    signature_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    signature_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    content = b"".join(chunks)
    if signature_before != signature_after or len(content) != before.st_size:
        raise ValueError(f"{label} changed while being read")
    try:
        final = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"{label} changed while being read") from error
    if not stat.S_ISREG(final.st_mode) or (final.st_dev, final.st_ino) != (
        before.st_dev,
        before.st_ino,
    ):
        raise ValueError(f"{label} changed while being read")
    return content


def _roster_sha256(pre_ids: Sequence[str], post_ids: Sequence[str]) -> str:
    return _canonical_sha256(
        {
            "video_source_session_ids": [*pre_ids, *post_ids],
            "audio_source_session_ids": [*pre_ids, *post_ids],
        }
    )


def _matrix_sha256(values: np.ndarray) -> str:
    matrix = np.asarray(values, dtype="<f8", order="C")
    return hashlib.sha256(matrix.tobytes(order="C")).hexdigest()


def _json_float_matrix(value: object) -> np.ndarray:
    if not isinstance(value, list) or not value:
        raise ValueError("nonvisual join matrix must be a non-empty JSON array")
    for row in value:
        if not isinstance(row, list) or any(type(item) is not float for item in row):
            raise ValueError("nonvisual join matrix values must be exact JSON floats")
    try:
        return np.asarray(value, dtype=np.float64)
    except ValueError as error:
        raise ValueError("nonvisual join matrix must be rectangular") from error


def _python_float(value: float | np.floating[Any]) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("diagnostic value must be finite")
    return 0.0 if result == 0 else result


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


def _same_json_value(left: object, right: object) -> bool:
    return _canonical_json_bytes({"value": left}) == _canonical_json_bytes({"value": right})


def _validated_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be 64 lowercase hexadecimal digits")
    return value


def _require_exact_value(actual: object, expected: object, label: str) -> None:
    if type(actual) is not type(expected):
        raise ValueError(f"nonvisual join diagnostic {label} type drifted")
    if isinstance(expected, dict):
        if set(actual) != set(expected):  # type: ignore[arg-type]
            raise ValueError(f"nonvisual join diagnostic {label} schema drifted")
        for key, expected_value in expected.items():
            _require_exact_value(actual[key], expected_value, f"{label}.{key}")  # type: ignore[index]
    elif isinstance(expected, list):
        if len(actual) != len(expected):  # type: ignore[arg-type]
            raise ValueError(f"nonvisual join diagnostic {label} length drifted")
        for index, expected_value in enumerate(expected):
            _require_exact_value(actual[index], expected_value, f"{label}[{index}]")  # type: ignore[index]
    elif actual != expected:
        raise ValueError(f"nonvisual join diagnostic {label} value drifted")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _coverage_message(label: str, expected: set[str], observed: set[str]) -> str:
    missing = sorted(expected - observed, key=_source_session_sort_key)
    extra = sorted(observed - expected, key=_source_session_sort_key)
    return f"{label} source-session coverage differs: missing={missing}, extra={extra}"
