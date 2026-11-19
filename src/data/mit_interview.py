"""Strict loader and participant-grouped split contract for MIT Interview."""

from __future__ import annotations

import csv
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .mit_session_mapping import SessionMappingContext

Condition = Literal["pre", "post"]
FoldRole = Literal["train", "validation", "test"]

CORPUS_ID = "mit_interview"
TRANSCRIPT_PARSER_REVISION = "mit-candidate-transcript-parser-2"
FOLD_COUNT = 5
FOLD_SALT = "fair-avi-mit-fold-v1"
EXPECTED_PARTICIPANTS = 69
EXPECTED_SESSIONS = 138

RATING_FIELDS = (
    "Overall",
    "RecommendHiring",
    "Colleague",
    "Engaged",
    "Excited",
    "EyeContact",
    "Smiled",
    "SpeakingRate",
    "NoFillers",
    "Friendly",
    "Paused",
    "EngagingTone",
    "StructuredAnswers",
    "Calm",
    "NotStressed",
    "Focused",
    "Authentic",
    "NotAwkward",
    "Total",
)

# These aliases are the exact spellings observed in the supplied 138 transcripts. Matching is
# exact and deliberately not fuzzy: an unrecognised speaker-like marker fails inspection.
CANDIDATE_MARKER_ALIASES = frozenset(
    {"Interviewee", "INterviewee", "Intervieweee", "Interveiwee", "Inerviewee"}
)
INTERVIEWER_MARKER_ALIASES = frozenset({"Interviewer", "Intrviewer"})
# One source segment uses ``r`` instead of a speaker name. It is excluded and counted as
# unclassified rather than guessed to be candidate speech.
UNCLASSIFIED_EXCLUDED_MARKER_ALIASES = frozenset({"r"})
_SESSION_PATTERN = re.compile(r"^(PP|P)([1-9][0-9]*)$", re.ASCII)
_SPEAKER_LIKE_PATTERN = re.compile(r"^[A-Za-z]*interview[A-Za-z]*$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class CandidateTranscript:
    """Candidate-only text parsed from the supplied full-dialogue transcript."""

    reference: str
    content: str
    candidate_segment_count: int
    source_dialogue_sha256: str = ""
    excluded_unclassified_marker_count: int = 0
    excluded_colonless_fragment_count: int = 0
    retained_candidate_continuation_count: int = 0
    retained_candidate_continuation_characters: int = 0
    excluded_colonless_characters: int = 0
    excluded_interviewer_segment_count: int = 0


@dataclass(frozen=True, slots=True)
class OutcomeInventory:
    """Aggregate validation facts for all supplied rating rows, without worker identifiers."""

    columns: tuple[str, ...]
    total_rows: int
    aggregate_rows: int
    individual_rows: int
    individual_ratings_per_session: tuple[tuple[str, int], ...]
    missing_rating_cells: int
    rating_cells_outside_documented_scale: int
    rating_scale_exception_counts: tuple[tuple[str, str, int], ...] = ()


@dataclass(frozen=True, slots=True)
class InterviewOutcome:
    """EM-aggregated crowd rating, retaining raw and normalised primary values."""

    recommend_hiring_raw: float
    recommend_hiring: float
    ratings_raw: tuple[tuple[str, float], ...]

    def raw(self, field: str) -> float:
        """Return one raw EM-aggregated rating by its published column name."""
        try:
            return dict(self.ratings_raw)[field]
        except KeyError as error:
            raise KeyError(f"unknown rating field {field!r}") from error


@dataclass(frozen=True, slots=True)
class MITInterviewSession:
    """One pre- or post-coaching mock-interview session."""

    session_id: str
    source_session_id: str
    participant_id: str
    condition: Condition
    fold: int
    video_reference: str
    audio_reference: str
    transcript: CandidateTranscript
    outcome: InterviewOutcome

    @property
    def clip_id(self) -> str:
        """Stable extraction key retained by the generic feature-shard interface."""
        return self.session_id

    @property
    def split(self) -> str:
        """Return the fixed participant fold label, not a train/test role."""
        return f"fold_{self.fold}"

    @property
    def media_reference(self) -> str:
        """Return the video reference for generic visual feature code."""
        return self.video_reference


@dataclass(frozen=True, slots=True)
class MITInterviewRosterEntry:
    """Outcome-free media identity used before the Stage-A eligibility lock."""

    session_id: str
    source_session_id: str
    participant_id: str
    condition: Condition
    fold: int
    video_reference: str
    audio_reference: str


@dataclass(frozen=True, slots=True)
class MITInterviewSourceRosterEntry:
    """One immutable raw media record, without a participant-identity claim."""

    source_session_id: str
    condition: Condition
    video_reference: str
    audio_reference: str


@dataclass(frozen=True, slots=True)
class MITInterviewFeatureSession:
    """Mapped feature input containing media and candidate text but no outcome."""

    session_id: str
    source_session_id: str
    participant_id: str
    condition: Condition
    fold: int
    video_reference: str
    audio_reference: str
    transcript: CandidateTranscript

    @property
    def clip_id(self) -> str:
        """Return the canonical extraction key."""
        return self.session_id

    @property
    def split(self) -> str:
        """Return the fixed participant fold label."""
        return f"fold_{self.fold}"

    @property
    def media_reference(self) -> str:
        """Return the source-bound video reference."""
        return self.video_reference


@dataclass(frozen=True, slots=True)
class MITInterviewDataset:
    """Validated sessions ordered by numeric participant then condition."""

    records: tuple[MITInterviewSession, ...]
    outcome_inventory: OutcomeInventory

    @property
    def session_ids(self) -> tuple[str, ...]:
        """Return stable session IDs in deterministic record order."""
        return tuple(record.session_id for record in self.records)

    @property
    def source_session_ids(self) -> tuple[str, ...]:
        """Return immutable source-session IDs in canonical record order."""
        return tuple(record.source_session_id for record in self.records)

    @property
    def clip_ids(self) -> tuple[str, ...]:
        """Alias used by the generic extraction writer."""
        return self.session_ids

    @property
    def participant_ids(self) -> tuple[str, ...]:
        """Return canonical decimal participant IDs in numeric order."""
        return tuple(sorted({row.participant_id for row in self.records}, key=int))

    @property
    def fold_sizes(self) -> tuple[int, ...]:
        """Return participant counts in fixed folds 0 through 4."""
        fold_by_participant = {row.participant_id: row.fold for row in self.records}
        return tuple(
            sum(value == fold for value in fold_by_participant.values()) for fold in range(5)
        )

    def for_outer_fold(
        self, outer_test_fold: int, role: FoldRole
    ) -> tuple[MITInterviewSession, ...]:
        """Return one role from the fixed five-fold participant rotation.

        Test is ``f``, validation is ``(f + 1) mod 5``, and the other three folds train.
        """
        if isinstance(outer_test_fold, bool) or outer_test_fold not in range(FOLD_COUNT):
            raise ValueError("outer_test_fold must be an integer from 0 to 4")
        if role not in {"train", "validation", "test"}:
            raise ValueError("role must be train, validation or test")
        validation_fold = (outer_test_fold + 1) % FOLD_COUNT
        if role == "test":
            wanted = {outer_test_fold}
        elif role == "validation":
            wanted = {validation_fold}
        else:
            wanted = set(range(FOLD_COUNT)) - {outer_test_fold, validation_fold}
        return tuple(row for row in self.records if row.fold in wanted)


@dataclass(frozen=True, slots=True)
class MITInterviewFeatureDataset:
    """Validated mapped feature inputs ordered by canonical participant and phase."""

    records: tuple[MITInterviewFeatureSession, ...]

    @property
    def session_ids(self) -> tuple[str, ...]:
        return tuple(record.session_id for record in self.records)

    @property
    def source_session_ids(self) -> tuple[str, ...]:
        return tuple(record.source_session_id for record in self.records)

    @property
    def participant_ids(self) -> tuple[str, ...]:
        return tuple(sorted({row.participant_id for row in self.records}, key=int))

    @property
    def fold_sizes(self) -> tuple[int, ...]:
        fold_by_participant = {row.participant_id: row.fold for row in self.records}
        return tuple(
            sum(value == fold for value in fold_by_participant.values()) for fold in range(5)
        )


def canonical_participant_id(value: object) -> str:
    """Return an ASCII-decimal participant ID without prefix or leading zeros."""
    if not isinstance(value, str):
        raise ValueError("participant ID must be a string")
    token = value.strip().upper()
    match = _SESSION_PATTERN.fullmatch(token)
    if match is not None:
        return str(int(match.group(2)))
    if not re.fullmatch(r"[1-9][0-9]*", token, re.ASCII):
        raise ValueError(f"invalid participant ID {value!r}")
    return str(int(token))


def parse_session_id(value: object) -> tuple[str, str, Condition]:
    """Return canonical session ID, participant ID and condition."""
    if not isinstance(value, str):
        raise ValueError("session ID must be a string")
    token = value.strip().upper()
    match = _SESSION_PATTERN.fullmatch(token)
    if match is None:
        raise ValueError(f"invalid session ID {value!r}")
    prefix, number = match.groups()
    participant_id = str(int(number))
    condition: Condition = "post" if prefix == "PP" else "pre"
    return f"{prefix}{participant_id}", participant_id, condition


def assign_participant_folds(participant_ids: tuple[str, ...] | list[str]) -> dict[str, int]:
    """Assign one immutable fold by salted hash rank modulo five."""
    canonical = [canonical_participant_id(value) for value in participant_ids]
    if len(canonical) != len(set(canonical)):
        raise ValueError("participant IDs for fold assignment must be unique")
    ordered = sorted(
        canonical,
        key=lambda participant_id: (
            hashlib.sha256(
                FOLD_SALT.encode("ascii") + b"\0" + participant_id.encode("ascii")
            ).digest(),
            participant_id.encode("ascii"),
        ),
    )
    return {participant_id: rank % FOLD_COUNT for rank, participant_id in enumerate(ordered)}


def parse_candidate_transcript(full_transcript: str) -> tuple[str, int]:
    """Extract candidate turns and their unlabelled continuations from supplied dialogue."""
    if (
        not isinstance(full_transcript, str)
        or not full_transcript.strip()
        or "\0" in full_transcript
    ):
        raise ValueError("full transcript must be non-empty and NUL-free")
    content, count, *_ = _parse_candidate_transcript_with_audit(full_transcript)
    return content, count


def _parse_candidate_transcript_with_audit(
    full_transcript: str,
) -> tuple[str, int, int, int, int, int, int, int]:
    if (
        not isinstance(full_transcript, str)
        or not full_transcript.strip()
        or "\0" in full_transcript
    ):
        raise ValueError("full transcript must be non-empty and NUL-free")
    utterances: list[str] = []
    unclassified = 0
    excluded_colonless = 0
    excluded_colonless_characters = 0
    retained_continuations = 0
    retained_continuation_characters = 0
    excluded_interviewer_segments = 0
    preceding_role: str | None = None
    for segment in full_transcript.split("|"):
        segment = segment.strip()
        if not segment:
            continue
        if ":" not in segment:
            cleaned = " ".join(segment.split())
            if preceding_role == "candidate" and cleaned:
                utterances.append(cleaned)
                retained_continuations += 1
                retained_continuation_characters += len(segment)
            else:
                excluded_colonless += 1
                excluded_colonless_characters += len(segment)
            continue
        marker, content = segment.split(":", maxsplit=1)
        exact_marker = marker.strip()
        if exact_marker in CANDIDATE_MARKER_ALIASES:
            preceding_role = "candidate"
            cleaned = " ".join(content.split())
            if cleaned:
                utterances.append(cleaned)
        elif exact_marker in INTERVIEWER_MARKER_ALIASES:
            preceding_role = "interviewer"
            excluded_interviewer_segments += 1
            continue
        elif exact_marker in UNCLASSIFIED_EXCLUDED_MARKER_ALIASES:
            preceding_role = "unclassified"
            unclassified += 1
        elif _SPEAKER_LIKE_PATTERN.fullmatch(marker.strip()):
            raise ValueError(f"unknown speaker marker {marker.strip()!r}")
        else:
            raise ValueError(f"unknown colon-labelled marker {marker.strip()!r}")
    if not utterances:
        raise ValueError("transcript contains no recognised candidate utterance")
    return (
        " ".join(utterances),
        len(utterances) - retained_continuations,
        unclassified,
        excluded_colonless,
        retained_continuations,
        retained_continuation_characters,
        excluded_colonless_characters,
        excluded_interviewer_segments,
    )


def load_mit_interview_source_roster(
    dataset_root: str | Path,
) -> tuple[MITInterviewSourceRosterEntry, ...]:
    """Validate immutable source media without inferring participant pairing."""
    supplied_root = Path(dataset_root)
    if supplied_root.is_symlink() or not supplied_root.is_dir():
        raise ValueError("dataset root must be a regular non-symlink directory")
    root = supplied_root.resolve()
    videos = _session_files(root, "Videos", ".avi")
    audios = _session_files(root, "Audio", ".wav")
    if set(videos) != set(audios):
        raise ValueError(_coverage_message("audio", set(videos), set(audios)))
    phase_counts = {
        condition: sum(parse_session_id(value)[2] == condition for value in videos)
        for condition in ("pre", "post")
    }
    if len(videos) != EXPECTED_SESSIONS or phase_counts != {"pre": 69, "post": 69}:
        raise ValueError(
            "expected 138 immutable source sessions split into 69 pre and 69 post files"
        )
    records: list[MITInterviewSourceRosterEntry] = []
    for session_id in sorted(
        videos, key=lambda value: (int(parse_session_id(value)[1]), value.startswith("PP"))
    ):
        _, _, condition = parse_session_id(session_id)
        records.append(
            MITInterviewSourceRosterEntry(
                source_session_id=session_id,
                condition=condition,
                video_reference=f"Videos/{videos[session_id].name}",
                audio_reference=f"Audio/{audios[session_id].name}",
            )
        )
    return tuple(records)


def load_mit_interview_roster(
    dataset_root: str | Path,
    *,
    session_mapping: SessionMappingContext,
) -> tuple[MITInterviewRosterEntry, ...]:
    """Return the mapped outcome-free canonical media roster."""
    if not isinstance(session_mapping, SessionMappingContext):
        raise TypeError("session_mapping must be a verified SessionMappingContext")
    session_mapping.verify_current_files()
    source = load_mit_interview_source_roster(dataset_root)
    source_by_id = {row.source_session_id: row for row in source}
    lookup = session_mapping.lookup
    if set(lookup.values()) != set(source_by_id) or set(lookup) != set(source_by_id):
        raise ValueError("approved session mapping does not exactly cover the source media roster")
    participants = sorted({parse_session_id(value)[1] for value in lookup}, key=int)
    folds = assign_participant_folds(participants)
    records: list[MITInterviewRosterEntry] = []
    for session_id in sorted(
        lookup, key=lambda value: (int(parse_session_id(value)[1]), value.startswith("PP"))
    ):
        _, participant_id, condition = parse_session_id(session_id)
        source_session_id = lookup[session_id]
        source_row = source_by_id[source_session_id]
        if source_row.condition != condition:
            raise ValueError("approved session mapping crosses interview phases")
        records.append(
            MITInterviewRosterEntry(
                session_id=session_id,
                source_session_id=source_session_id,
                participant_id=participant_id,
                condition=condition,
                fold=folds[participant_id],
                video_reference=source_row.video_reference,
                audio_reference=source_row.audio_reference,
            )
        )
    return tuple(records)


def load_mit_interview_feature_dataset(
    dataset_root: str | Path,
    *,
    session_mapping: SessionMappingContext,
) -> MITInterviewFeatureDataset:
    """Join mapped media and candidate transcripts without opening the outcome table."""
    supplied_root = Path(dataset_root)
    if supplied_root.is_symlink() or not supplied_root.is_dir():
        raise ValueError("dataset root must be a regular non-symlink directory")
    root = supplied_root.resolve()
    roster = load_mit_interview_roster(root, session_mapping=session_mapping)
    source_session_ids = {row.source_session_id for row in roster}
    transcripts = _load_transcripts(root / "Labels" / "interview_transcripts_by_turkers.csv", root)
    if source_session_ids != set(transcripts):
        raise ValueError(_coverage_message("transcript", source_session_ids, set(transcripts)))
    records = [
        MITInterviewFeatureSession(
            session_id=row.session_id,
            source_session_id=row.source_session_id,
            participant_id=row.participant_id,
            condition=row.condition,
            fold=row.fold,
            video_reference=row.video_reference,
            audio_reference=row.audio_reference,
            transcript=transcripts[row.source_session_id],
        )
        for row in roster
    ]
    dataset = MITInterviewFeatureDataset(tuple(records))
    if dataset.fold_sizes != (14, 14, 14, 14, 13):
        raise ValueError(f"fixed fold contract drifted: {dataset.fold_sizes}")
    return dataset


def load_mit_interview_dataset(
    dataset_root: str | Path,
    *,
    session_mapping: SessionMappingContext,
) -> MITInterviewDataset:
    """Validate the full mapped corpus including source-bound aggregate outcomes."""
    supplied_root = Path(dataset_root)
    if supplied_root.is_symlink() or not supplied_root.is_dir():
        raise ValueError("dataset root must be a regular non-symlink directory")
    root = supplied_root.resolve()
    feature_dataset = load_mit_interview_feature_dataset(
        root, session_mapping=session_mapping
    )
    outcomes, source_inventory = _load_outcomes(
        root / "Labels" / "turker_scores_full_interview.csv", root
    )
    if set(feature_dataset.source_session_ids) != set(outcomes):
        raise ValueError(
            _coverage_message("outcome", set(feature_dataset.source_session_ids), set(outcomes))
        )
    records = tuple(
        MITInterviewSession(
            session_id=row.session_id,
            source_session_id=row.source_session_id,
            participant_id=row.participant_id,
            condition=row.condition,
            fold=row.fold,
            video_reference=row.video_reference,
            audio_reference=row.audio_reference,
            transcript=row.transcript,
            outcome=outcomes[row.source_session_id],
        )
        for row in feature_dataset.records
    )
    counts_by_source = dict(source_inventory.individual_ratings_per_session)
    canonical_inventory = OutcomeInventory(
        columns=source_inventory.columns,
        total_rows=source_inventory.total_rows,
        aggregate_rows=source_inventory.aggregate_rows,
        individual_rows=source_inventory.individual_rows,
        individual_ratings_per_session=tuple(
            (row.session_id, counts_by_source[row.source_session_id]) for row in records
        ),
        missing_rating_cells=source_inventory.missing_rating_cells,
        rating_cells_outside_documented_scale=(
            source_inventory.rating_cells_outside_documented_scale
        ),
        rating_scale_exception_counts=source_inventory.rating_scale_exception_counts,
    )
    dataset = MITInterviewDataset(records, canonical_inventory)
    if dataset.fold_sizes != (14, 14, 14, 14, 13):
        raise ValueError(f"fixed fold contract drifted: {dataset.fold_sizes}")
    return dataset


def fold_manifest_csv(dataset: MITInterviewDataset) -> str:
    """Return the canonical fixed-fold manifest as UTF-8 CSV text."""
    fold_by_participant = {row.participant_id: row.fold for row in dataset.records}
    lines = ["participant_id,fold"]
    lines.extend(
        f"{participant_id},{fold_by_participant[participant_id]}"
        for participant_id in dataset.participant_ids
    )
    return "\n".join(lines) + "\n"


def fold_manifest_sha256(dataset: MITInterviewDataset) -> str:
    """Hash the exact canonical fold manifest."""
    return hashlib.sha256(fold_manifest_csv(dataset).encode("utf-8")).hexdigest()


def _session_files(root: Path, directory_name: str, suffix: str) -> dict[str, Path]:
    directory = root / directory_name
    _require_directory(directory, root)
    indexed: dict[str, Path] = {}
    for path in directory.iterdir():
        if path.suffix.casefold() != suffix:
            continue
        _require_regular_file(path, root)
        session_id, _, _ = parse_session_id(path.stem)
        if session_id in indexed:
            raise ValueError(f"duplicate session file {session_id}")
        indexed[session_id] = path
    return indexed


def _load_transcripts(path: Path, root: Path) -> dict[str, CandidateTranscript]:
    indexed: dict[str, CandidateTranscript] = {}
    _require_regular_file(path, root)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            for row_number, row in enumerate(reader, start=1):
                if len(row) != 2:
                    raise ValueError(f"transcript row {row_number} must have exactly two fields")
                session_id, _, _ = parse_session_id(row[0])
                if session_id in indexed:
                    raise ValueError(f"duplicate transcript session {session_id}")
                (
                    content,
                    count,
                    unclassified,
                    excluded_colonless,
                    retained_continuations,
                    retained_continuation_characters,
                    excluded_colonless_characters,
                    excluded_interviewer_segments,
                ) = _parse_candidate_transcript_with_audit(row[1])
                indexed[session_id] = CandidateTranscript(
                    reference="Labels/interview_transcripts_by_turkers.csv",
                    content=content,
                    candidate_segment_count=count,
                    source_dialogue_sha256=hashlib.sha256(row[1].encode("utf-8")).hexdigest(),
                    excluded_unclassified_marker_count=unclassified,
                    excluded_colonless_fragment_count=excluded_colonless,
                    retained_candidate_continuation_count=retained_continuations,
                    retained_candidate_continuation_characters=retained_continuation_characters,
                    excluded_colonless_characters=excluded_colonless_characters,
                    excluded_interviewer_segment_count=excluded_interviewer_segments,
                )
    except OSError as error:
        raise ValueError("cannot read supplied transcript table") from error
    return indexed


def _load_outcomes(path: Path, root: Path) -> tuple[dict[str, InterviewOutcome], OutcomeInventory]:
    expected = ("Participant", "Worker", *RATING_FIELDS)
    indexed: dict[str, InterviewOutcome] = {}
    individual_counts: dict[str, int] = {}
    worker_session_keys: set[tuple[str, str]] = set()
    scale_exceptions: set[tuple[str, str, str, float]] = set()
    scale_exception_counts: dict[tuple[str, str], int] = {}
    total_rows = aggregate_rows = individual_rows = out_of_scale = 0
    _require_regular_file(path, root)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration as error:
                raise ValueError("rating table is empty") from error
            if tuple(header) != expected or len(set(header)) != len(header):
                raise ValueError("rating table schema does not match the published MIT fields")
            for row_number, values_row in enumerate(reader, start=2):
                if len(values_row) != len(expected):
                    raise ValueError(
                        f"rating row {row_number} must have exactly {len(expected)} fields"
                    )
                row = dict(zip(expected, values_row, strict=True))
                session_id, _, _ = parse_session_id(row["Participant"])
                worker = row["Worker"]
                if not worker or "\0" in worker:
                    raise ValueError(f"rating worker at row {row_number} is invalid")
                worker_key = (session_id, worker)
                if worker_key in worker_session_keys:
                    raise ValueError(f"duplicate worker rating for session {session_id}")
                worker_session_keys.add(worker_key)
                values = tuple(
                    (field, _finite_rating(row[field], field, session_id, row_number))
                    for field in RATING_FIELDS
                )
                out_of_scale += sum(
                    not 1.0 <= value <= 7.0 for field, value in values if field != "Total"
                )
                source_kind = "aggregate" if worker == "AGGR" else "individual"
                for field, value in values:
                    if field != "Total" and not 1.0 <= value <= 7.0:
                        scale_exceptions.add((source_kind, session_id, field, value))
                        key = (source_kind, field)
                        scale_exception_counts[key] = scale_exception_counts.get(key, 0) + 1
                for field, value in values:
                    if field != "Total" and not 0.0 <= value <= 7.0:
                        raise ValueError(
                            f"{field} for {session_id} at row {row_number} is outside 0--7"
                        )
                if not math.isclose(
                    dict(values)["Total"],
                    sum(dict(values)[field] for field in RATING_FIELDS if field != "Total"),
                    rel_tol=0.0,
                    abs_tol=1e-6,
                ):
                    raise ValueError(
                        f"Total for {session_id} at row {row_number} does not equal the rating sum"
                    )
                total_rows += 1
                if worker != "AGGR":
                    individual_rows += 1
                    individual_counts[session_id] = individual_counts.get(session_id, 0) + 1
                    continue
                aggregate_rows += 1
                if session_id in indexed:
                    raise ValueError(f"duplicate AGGR outcome session {session_id}")
                recommend_raw = dict(values)["RecommendHiring"]
                if not 1.0 <= recommend_raw <= 7.0:
                    raise ValueError(
                        f"aggregate RecommendHiring for {session_id} must be on the 1--7 scale"
                    )
                indexed[session_id] = InterviewOutcome(
                    recommend_hiring_raw=recommend_raw,
                    recommend_hiring=(recommend_raw - 1.0) / 6.0,
                    ratings_raw=values,
                )
    except OSError as error:
        raise ValueError("cannot read supplied rating table") from error
    if set(individual_counts) != set(indexed):
        raise ValueError("individual-rating session coverage does not match aggregate coverage")
    if (
        total_rows != 1_378
        or aggregate_rows != 138
        or individual_rows != 1_240
        or sorted(individual_counts.values()).count(8) != 2
        or sorted(individual_counts.values()).count(9) != 136
        or set(individual_counts.values()) != {8, 9}
    ):
        raise ValueError("rating table row-count and 8/9-rater coverage contract drifted")
    if (
        scale_exceptions
        != {
            ("individual", "P4", "EyeContact", 0.0),
            ("individual", "P49", "Paused", 0.0),
        }
        or out_of_scale != 2
    ):
        raise ValueError("known individual-rating scale-exception contract drifted")
    inventory = OutcomeInventory(
        columns=expected,
        total_rows=total_rows,
        aggregate_rows=aggregate_rows,
        individual_rows=individual_rows,
        individual_ratings_per_session=tuple(sorted(individual_counts.items())),
        missing_rating_cells=0,
        rating_cells_outside_documented_scale=out_of_scale,
        rating_scale_exception_counts=tuple(
            (source_kind, field, count)
            for (source_kind, field), count in sorted(scale_exception_counts.items())
        ),
    )
    return indexed, inventory


def _require_directory(path: Path, root: Path) -> None:
    if path.is_symlink() or not path.is_dir() or not path.resolve().is_relative_to(root):
        raise ValueError(f"required dataset directory is unavailable or unsafe: {path.name}")


def _require_regular_file(path: Path, root: Path) -> None:
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError(f"dataset file is unavailable or unsafe: {path.name}")


def _finite_rating(value: object, field: str, session_id: str, row_number: int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} for {session_id} at row {row_number} is not numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} for {session_id} must be finite")
    return result


def _coverage_message(label: str, expected: set[str], observed: set[str]) -> str:
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    return f"{label} session coverage mismatch: missing={missing!r}; unexpected={unexpected!r}"
