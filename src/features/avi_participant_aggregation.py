"""Outcome-blind, six-clip AVI participant aggregation.

The AVI release labels are participant-level. This module pools exactly the six
cached clip embeddings for one participant and never reads outcome or audit data.
Text remains unavailable rather than becoming a zero vector.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from src.features.avi_preprocessing import (
    AVI_AUDIO_EMBEDDING_DIM,
    AVI_VIDEO_EMBEDDING_DIM,
    pool_participant_clips,
)


PARTICIPANT_AGGREGATION_SCHEMA_VERSION = "avi-participant-video-audio-aggregation-1"
QUESTION_ORDER = ("q1", "q2", "q3", "q4", "q5", "q6")
OFFICIAL_SPLITS = {"Training": 452, "Validation": 64, "Testing": 130}
_PARTICIPANT_ID = re.compile(r"^[0-9a-f]{24}$", re.ASCII)


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_vector(path: Path, dimension: int) -> np.ndarray:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or unsafe cache vector: {path.name}")
    try:
        vector = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"unreadable cache vector: {path.name}") from error
    if vector.shape != (dimension,) or vector.dtype != np.dtype(np.float32):
        raise ValueError(f"cache vector shape/dtype drift: {path.name}")
    if not np.isfinite(vector).all():
        raise ValueError(f"non-finite cache vector: {path.name}")
    return vector


def group_manifest_clips(clips: Iterable[Any]) -> dict[str, tuple[Any, ...]]:
    """Validate official split isolation and exact six-question membership."""
    grouped: dict[str, list[Any]] = defaultdict(list)
    split_by_participant: dict[str, str] = {}
    for clip in clips:
        participant_id = clip.participant_id
        if not isinstance(participant_id, str) or _PARTICIPANT_ID.fullmatch(participant_id) is None:
            raise ValueError("AVI participant ID drifted")
        if clip.split not in OFFICIAL_SPLITS:
            raise ValueError("AVI official split drifted")
        previous = split_by_participant.setdefault(participant_id, clip.split)
        if previous != clip.split:
            raise ValueError("AVI participant occurs in multiple official splits")
        grouped[participant_id].append(clip)
    result: dict[str, tuple[Any, ...]] = {}
    for participant_id, rows in grouped.items():
        questions = tuple(row.question_id for row in rows)
        if len(rows) != 6 or set(questions) != set(QUESTION_ORDER):
            raise ValueError("AVI participant does not have exactly q1--q6")
        result[participant_id] = tuple(sorted(rows, key=lambda row: QUESTION_ORDER.index(row.question_id)))
    return result


def aggregate_participant(clips: tuple[Any, ...], *, video_root: Path, audio_root: Path) -> dict[str, np.ndarray]:
    """Pool one participant's cache vectors in the fixed question order."""
    if len(clips) != 6:
        raise ValueError("AVI aggregation needs exactly six clips")
    video = {row.question_id: _valid_vector(video_root / f"{row.clip_id}.npy", AVI_VIDEO_EMBEDDING_DIM) for row in clips}
    audio = {row.question_id: _valid_vector(audio_root / f"{row.clip_id}.npy", AVI_AUDIO_EMBEDDING_DIM) for row in clips}
    video_embedding, video_mask = pool_participant_clips(video, QUESTION_ORDER)
    audio_embedding, audio_mask = pool_participant_clips(audio, QUESTION_ORDER)
    if not np.array_equal(video_mask, np.ones(6, dtype=np.bool_)) or not np.array_equal(audio_mask, video_mask):
        raise ValueError("AVI validated cache aggregation unexpectedly has a missing clip")
    return {"video": video_embedding.astype(np.float32, copy=False), "audio": audio_embedding.astype(np.float32, copy=False), "video_mask": video_mask.astype(np.bool_, copy=False), "audio_mask": audio_mask.astype(np.bool_, copy=False), "text_available": np.asarray(False, dtype=np.bool_)}


def _save_npz_no_clobber(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            pass
        except OSError as error:
            if error.errno not in {45, 95}:
                raise
            # exFAT volumes, including the attached Seagate drive, do not
            # implement hard links. Create the destination exclusively, so
            # this fallback retains the no-overwrite publication contract.
            try:
                destination_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                pass
            else:
                try:
                    with temporary.open("rb") as source, os.fdopen(destination_fd, "wb") as destination:
                        shutil.copyfileobj(source, destination, length=1024 * 1024)
                        destination.flush()
                        os.fsync(destination.fileno())
                except BaseException:
                    path.unlink(missing_ok=True)
                    raise
    finally:
        temporary.unlink(missing_ok=True)
    validate_participant_shard(path)
    return sha256_file(path)


def validate_participant_shard(path: Path) -> None:
    """Validate the strict, target-free participant feature shard schema."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("participant shard is missing or unsafe")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"video", "audio", "video_mask", "audio_mask", "text_available"}:
                raise ValueError("participant shard schema drifted")
            video, audio = archive["video"], archive["audio"]
            video_mask, audio_mask, text_available = archive["video_mask"], archive["audio_mask"], archive["text_available"]
    except (OSError, ValueError) as error:
        raise ValueError("participant shard is unreadable") from error
    if video.shape != (AVI_VIDEO_EMBEDDING_DIM,) or video.dtype != np.dtype(np.float32):
        raise ValueError("participant video feature contract drifted")
    if audio.shape != (AVI_AUDIO_EMBEDDING_DIM,) or audio.dtype != np.dtype(np.float32):
        raise ValueError("participant audio feature contract drifted")
    if video_mask.shape != (6,) or audio_mask.shape != (6,) or video_mask.dtype != np.bool_ or audio_mask.dtype != np.bool_:
        raise ValueError("participant modality mask contract drifted")
    if not np.array_equal(video_mask, np.ones(6, dtype=np.bool_)) or not np.array_equal(audio_mask, video_mask):
        raise ValueError("participant aggregation must contain six valid video/audio clips")
    if text_available.shape != () or text_available.dtype != np.bool_ or bool(text_available):
        raise ValueError("AVI text must remain explicitly unavailable")
    if not np.isfinite(video).all() or not np.isfinite(audio).all():
        raise ValueError("participant shard has non-finite values")


def _shard_matches_arrays(path: Path, arrays: dict[str, np.ndarray]) -> bool:
    """Require an existing no-clobber shard to equal the current source result."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            return all(np.array_equal(archive[name], value) for name, value in arrays.items())
    except (OSError, ValueError):
        return False


def aggregate_release(clips: Iterable[Any], *, video_root: Path, audio_root: Path, output_root: Path, manifest_sha256: str) -> dict[str, Any]:
    """Build or verify a no-clobber aggregate release outside the repository."""
    if output_root.exists() and not output_root.is_dir():
        raise ValueError("AVI aggregation output root is not a directory")
    grouped = group_manifest_clips(clips)
    if dict(Counter(rows[0].split for rows in grouped.values())) != OFFICIAL_SPLITS or len(grouped) != sum(OFFICIAL_SPLITS.values()):
        raise ValueError("AVI participant split counts drifted")
    output_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    shards: list[dict[str, str]] = []
    for participant_id in sorted(grouped):
        rows = grouped[participant_id]
        shard_path = output_root / "shards" / f"{participant_id}.npz"
        arrays = aggregate_participant(rows, video_root=video_root, audio_root=audio_root)
        if shard_path.exists():
            validate_participant_shard(shard_path)
            if not _shard_matches_arrays(shard_path, arrays):
                raise ValueError(
                    f"T-480 refuses to replace non-identical participant shard: {shard_path}"
                )
            digest = sha256_file(shard_path)
        else:
            digest = _save_npz_no_clobber(shard_path, arrays)
        shards.append({"participant_id": participant_id, "split": rows[0].split, "sha256": digest})
    split_counts = dict(Counter(row["split"] for row in shards))
    if split_counts != OFFICIAL_SPLITS:
        raise RuntimeError("AVI release split count changed during aggregation")
    payload = {"schema_version": PARTICIPANT_AGGREGATION_SCHEMA_VERSION, "status": "complete_video_audio_only", "source_manifest_sha256": manifest_sha256, "participant_count": len(shards), "split_counts": split_counts, "clips_per_participant": 6, "video": {"dimension": AVI_VIDEO_EMBEDDING_DIM, "dtype": "float32", "pooling": "mean_over_q1_to_q6"}, "audio": {"dimension": AVI_AUDIO_EMBEDDING_DIM, "dtype": "float32", "pooling": "mean_over_q1_to_q6"}, "text": {"available": False, "reason": "no_authorised_transcript_or_ASR_feature_release"}, "participant_shard_ledger_sha256": sha256_bytes(canonical_json(shards)), "participant_shards": shards, "outcome_or_audit_fields_accessed": []}
    manifest = output_root / "participant_feature_manifest.json"
    encoded = canonical_json(payload)
    if manifest.exists():
        if manifest.read_bytes() != encoded:
            raise ValueError("refusing to replace non-identical AVI participant manifest")
    else:
        with manifest.open("xb") as stream:
            stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
    return payload
