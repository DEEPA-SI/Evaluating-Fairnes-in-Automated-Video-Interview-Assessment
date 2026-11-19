"""Sequential, resumable AVI-Personality video and audio embedding extraction.

This runner deliberately has no access to outcomes or audit attributes.  It reads the
approved clip manifest, loads each requested frozen encoder once, and writes one
validated float32 vector per clip.  Existing valid vectors are resumed; corrupt or
incomplete vectors are recomputed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.features.avi_preprocessing import (
    AVI_AUDIO_EMBEDDING_DIM,
    AVI_VIDEO_EMBEDDING_DIM,
    extract_avi_audio_embedding,
    extract_avi_video_embedding,
)
from src.features.encoders import Dinov2VideoEncoder, WavlmAudioEncoder

EXPECTED_MANIFEST_SHA256 = "dc8b9e8cbe473aa6290214b96ff8f31fce14f09a84158573f46a0e3bca2d387a"
EXPECTED_CLIP_COUNT = 3_876
DEFAULT_OUTPUT_ROOT = Path("build/avi-extraction")
MANIFEST_COLUMNS = (
    "split",
    "participant_id",
    "question_id",
    "question_category",
    "media_path",
)


@dataclass(frozen=True, slots=True)
class Clip:
    """One outcome-blind extraction unit from the immutable release manifest."""

    split: str
    participant_id: str
    question_id: str
    question_category: str
    media_path: Path

    @property
    def clip_id(self) -> str:
        return self.media_path.stem


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/default.yaml")
    parser.add_argument(
        "--manifest", type=Path, default=root / "artifacts/avi/avi_extraction_manifest.csv"
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--modality", choices=("audio", "video", "both"), default="both")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", choices=("Training", "Validation", "Testing"), default=None)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> tuple[Clip, ...]:
    """Load and validate the fixed 3,876-row AVI extraction manifest."""
    if _sha256(path) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("AVI extraction manifest SHA-256 does not match the approved release")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise ValueError("AVI extraction manifest columns do not match the approved schema")
        clips = tuple(
            Clip(
                split=row["split"],
                participant_id=row["participant_id"],
                question_id=row["question_id"],
                question_category=row["question_category"],
                media_path=Path(row["media_path"]),
            )
            for row in reader
        )
    if len(clips) != EXPECTED_CLIP_COUNT:
        raise ValueError("AVI extraction manifest clip count does not match the approved release")
    if len({clip.clip_id for clip in clips}) != len(clips):
        raise ValueError("AVI extraction manifest contains duplicate output identifiers")
    return clips


def select_clips(
    clips: Iterable[Clip], *, split: str | None, limit: int | None
) -> tuple[Clip, ...]:
    """Apply deterministic manifest-order split and limit filtering."""
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")
    selected = tuple(clip for clip in clips if split is None or clip.split == split)
    return selected if limit is None else selected[:limit]


def output_path(output_root: Path, modality: str, clip: Clip) -> Path:
    return output_root / modality / f"{clip.clip_id}.npy"


def valid_embedding(path: Path, expected_dim: int) -> bool:
    """Return whether an already-published vector is safe to resume."""
    try:
        values = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return False
    return (
        values.shape == (expected_dim,)
        and values.dtype == np.dtype(np.float32)
        and bool(np.isfinite(values).all())
    )


def publish_embedding(path: Path, values: np.ndarray, expected_dim: int) -> None:
    """Validate and atomically publish through a same-directory temporary file."""
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (expected_dim,) or not np.isfinite(array).all():
        raise ValueError(f"invalid {expected_dim}-dimensional float32 embedding")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or not isinstance(config.get("encoders"), dict):
        raise ValueError("configuration must define encoder settings")
    return config


def _event(handle: Any, **fields: object) -> None:
    record = {"timestamp_utc": datetime.now(UTC).isoformat(), **fields}
    encoded = json.dumps(record, sort_keys=True)
    print(encoded, flush=True)
    handle.write(encoded + "\n")
    handle.flush()


def run(args: argparse.Namespace) -> dict[str, int]:
    """Execute the requested sequential extraction, returning count evidence."""
    clips = select_clips(load_manifest(args.manifest), split=args.split, limit=args.limit)
    requested = ("audio", "video") if args.modality == "both" else (args.modality,)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "clips": len(clips),
                    "modalities": requested,
                    "manifest_sha256": EXPECTED_MANIFEST_SHA256,
                },
                sort_keys=True,
            )
        )
        return {"processed": 0, "resumed": 0, "failed": 0, "planned": len(clips) * len(requested)}

    config = _load_config(args.config)
    # These constructors verify pinned artifacts and are intentionally called once each.
    audio_encoder = (
        WavlmAudioEncoder(config["encoders"]["audio"], device="cpu", local_files_only=True)
        if "audio" in requested
        else None
    )
    video_encoder = (
        Dinov2VideoEncoder(config["encoders"]["video"], device="cpu", local_files_only=True)
        if "video" in requested
        else None
    )
    log_dir = args.output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"avi_embedding_{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{os.getpid()}.jsonl"
    counts = {"processed": 0, "resumed": 0, "failed": 0, "planned": len(clips) * len(requested)}
    dimensions = {"audio": AVI_AUDIO_EMBEDDING_DIM, "video": AVI_VIDEO_EMBEDDING_DIM}

    with log_path.open("x", encoding="utf-8") as log:
        _event(log, event="run_started", clips=len(clips), modalities=requested, device="cpu")
        for clip in clips:
            for modality in requested:
                destination = output_path(args.output_root, modality, clip)
                if valid_embedding(destination, dimensions[modality]):
                    counts["resumed"] += 1
                    _event(log, event="clip_resumed", clip_id=clip.clip_id, modality=modality)
                    continue
                started = time.monotonic()
                try:
                    if modality == "audio":
                        assert audio_encoder is not None
                        embedding = extract_avi_audio_embedding(clip.media_path, audio_encoder)
                    else:
                        assert video_encoder is not None
                        embedding = extract_avi_video_embedding(
                            clip.media_path, video_encoder._model, video_encoder._processor
                        )
                    publish_embedding(destination, embedding, dimensions[modality])
                    counts["processed"] += 1
                    _event(
                        log,
                        event="clip_processed",
                        clip_id=clip.clip_id,
                        modality=modality,
                        elapsed_seconds=round(time.monotonic() - started, 3),
                    )
                except Exception as error:  # keep the batch resumable after one bad clip
                    counts["failed"] += 1
                    _event(
                        log,
                        event="clip_failed",
                        clip_id=clip.clip_id,
                        modality=modality,
                        error_type=type(error).__name__,
                        error=str(error),
                    )
        _event(log, event="run_finished", **counts)
    return counts


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    counts = run(args)
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
