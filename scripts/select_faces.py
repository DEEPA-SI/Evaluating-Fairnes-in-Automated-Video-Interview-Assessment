#!/usr/bin/env python3
"""Generate deterministic candidate-face crops and a strict manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import stat
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import PIL
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_mit_interview_source_roster  # noqa: E402
from src.data.annotation_stimuli import load_face_crop_manifest  # noqa: E402
from src.features.extraction import (  # noqa: E402
    canonical_json_bytes,
    code_provenance,
    file_sha256,
)
from src.features.face_selector import (  # noqa: E402
    SELECTOR_VERSION,
    OpenCvFrameDecoder,
    ScrfdFaceDetector,
    generate_face_crop_manifest,
)

OWNED_SOURCE_PATHS = (
    "scripts/select_faces.py",
    "src/__init__.py",
    "src/data/__init__.py",
    "src/data/annotation_stimuli.py",
    "src/data/mit_interview.py",
    "src/features/__init__.py",
    "src/features/extraction.py",
    "src/features/face_selector.py",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--detector-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop-root", type=Path)
    parser.add_argument("--qc-output", type=Path)
    parser.add_argument(
        "--session",
        action="append",
        default=[],
        help="exact source session ID to process; repeat for a bounded pilot (default: all)",
    )
    parser.add_argument("--source-date-epoch", type=int)
    return parser.parse_args(argv)


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    try:
        candidate_context = config["feature_extraction"]["candidate_face_context"]
    except (KeyError, TypeError) as error:
        raise ValueError("configuration is missing candidate-face context") from error
    if not isinstance(candidate_context, dict):
        raise ValueError("candidate-face context must be a mapping")
    scoped_sha256 = hashlib.sha256(canonical_json_bytes(candidate_context)).hexdigest()
    return config, scoped_sha256


def _source_date_epoch(explicit: int | None) -> int:
    raw = explicit if explicit is not None else os.environ.get("SOURCE_DATE_EPOCH")
    try:
        result = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError("SOURCE_DATE_EPOCH or --source-date-epoch is required") from error
    if result < 0:
        raise ValueError("source date epoch must be non-negative")
    return result


def _selected_sessions(dataset: Any, requested: list[str]) -> tuple[Any, ...]:
    records = tuple(dataset.records) if hasattr(dataset, "records") else tuple(dataset)
    if not requested:
        return records
    if len(requested) != len(set(requested)):
        raise ValueError("--session values must be unique")
    by_id = {session.source_session_id: session for session in records}
    missing = sorted(set(requested) - set(by_id))
    if missing:
        raise ValueError(f"unknown session IDs: {missing!r}")
    return tuple(by_id[session_id] for session_id in requested)


def _write_validated_manifest(
    output: Path, payload: dict[str, Any], expected_session_ids: tuple[str, ...]
) -> None:
    _publish_restricted_file(
        output,
        canonical_json_bytes(payload),
        validator=lambda path: load_face_crop_manifest(
            path, expected_session_ids=expected_session_ids
        ),
    )


def _publish_restricted_file(
    output: Path,
    content: bytes,
    *,
    validator: Any | None = None,
) -> None:
    """Publish one immutable restricted file without following or clobbering paths."""
    output = Path(output)
    if output.name in {"", ".", ".."} or "\0" in output.name:
        raise ValueError("restricted output must name one file")
    parent = _non_symlink_directory(output.parent, create=True)
    destination = parent / output.name
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    published_identity: tuple[int, int] | None = None
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if validator is not None:
            validator(temporary)
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            pass
        else:
            metadata = temporary.stat(follow_symlinks=False)
            published_identity = (metadata.st_dev, metadata.st_ino)
    finally:
        temporary.unlink(missing_ok=True)
    try:
        _validate_published_file(destination, content, published_identity)
    finally:
        _fsync_directory(parent)


def _non_symlink_directory(path: Path, *, create: bool = False) -> Path:
    if ".." in path.parts:
        raise ValueError("restricted output parent cannot contain parent traversal")
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        if component in {"", "."}:
            continue
        current = current / component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if not create:
                raise
            try:
                os.mkdir(current, mode=0o700)
            except FileExistsError:
                pass
            metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("restricted output parent cannot contain symlink components")
    return current


def _validate_published_file(
    path: Path, expected: bytes, expected_identity: tuple[int, int] | None
) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise ValueError("restricted output may not be a symlink") from error
    try:
        metadata = os.fstat(descriptor)
        identity = (metadata.st_dev, metadata.st_ino)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("restricted output must be one regular unlinked file")
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            raise ValueError("restricted output must use owner-only permissions")
        if expected_identity is not None and identity != expected_identity:
            raise ValueError("restricted output changed during publication")
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        if b"".join(chunks) != expected:
            raise ValueError("restricted output already exists with different content")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    current = path.stat(follow_symlinks=False)
    if (current.st_dev, current.st_ino) != identity:
        raise ValueError("restricted output changed during publication")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv: list[str] | None = None) -> int:
    """Run deterministic face selection and write restricted outputs."""
    args = _parse_args(argv)
    epoch = _source_date_epoch(args.source_date_epoch)
    config, config_sha256 = _load_config(args.config)
    candidate_context = config["feature_extraction"]["candidate_face_context"]
    selector_config = candidate_context["selector"]
    if int(candidate_context["source_grid_hz"]) != int(selector_config["scan_fps"]):
        raise ValueError("candidate-face scan rate drifted between context and selector")
    if int(candidate_context["maximum_samples"]) != int(
        selector_config["maximum_retained_samples"]
    ):
        raise ValueError("candidate-face retained cap drifted between context and selector")
    dataset = load_mit_interview_source_roster(args.dataset_root)
    sessions = _selected_sessions(dataset, args.session)
    detector = ScrfdFaceDetector(selector_config, args.detector_model)
    decoder = OpenCvFrameDecoder()
    code = code_provenance(ROOT, OWNED_SOURCE_PATHS)
    generator = {
        "name": "fair-avi-candidate-face-generator",
        "version": SELECTOR_VERSION,
        "source_commit_sha": code.commit_sha,
        "source_tree_sha256": code.source_tree_sha256,
        "source_tree_dirty": code.dirty,
        "config_sha256": config_sha256,
        "source_date_epoch": epoch,
        "insightface_version": detector.insightface_version,
        "onnxruntime_version": detector.onnxruntime_version,
        "opencv_version": decoder.opencv_version,
        "numpy_version": str(np.__version__),
        "pillow_version": str(PIL.__version__),
        "ffprobe_version": decoder.ffprobe_version,
        "ffprobe_build_sha256": decoder.ffprobe_build_sha256,
    }
    output = Path(os.path.abspath(os.fspath(args.output)))
    crop_root = Path(
        os.path.abspath(os.fspath(args.crop_root or output.parent / "face_crops"))
    )
    started = time.perf_counter()
    manifest, qc = generate_face_crop_manifest(
        sessions,
        dataset_root=args.dataset_root,
        output_manifest=output,
        crop_root=crop_root,
        detector=detector,
        decoder=decoder,
        selector_config=selector_config,
        generator_provenance=generator,
        maximum_samples=int(candidate_context["maximum_samples"]),
        progress=lambda session_id, status: print(f"{session_id}: {status}", flush=True),
    )
    _write_validated_manifest(
        output,
        manifest,
        expected_session_ids=tuple(session.source_session_id for session in sessions),
    )
    elapsed_seconds = time.perf_counter() - started
    qc["face_crop_manifest_sha256"] = file_sha256(output)
    crop_storage_bytes = sum(
        (output.parent / row["path"]).stat().st_size
        for clip in manifest["clips"].values()
        for row in clip["frames"]
    )
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_bytes = int(peak_rss if sys.platform == "darwin" else peak_rss * 1024)
    projection_factor = len(dataset) / len(sessions)
    qc.update(
        {
            "elapsed_seconds": elapsed_seconds,
            "peak_rss_bytes": peak_rss_bytes,
            "crop_storage_bytes": crop_storage_bytes,
            "projected_full_elapsed_seconds": elapsed_seconds * projection_factor,
            "projected_full_crop_storage_bytes": round(crop_storage_bytes * projection_factor),
        }
    )
    qc_output = args.qc_output or output.with_name("face_crop_qc.json")
    _publish_restricted_file(Path(qc_output), canonical_json_bytes(qc))
    print(
        json.dumps(
            {
                "sessions": qc["session_count"],
                "succeeded": qc["succeeded"],
                "failed": qc["failed"],
                "manifest_sha256": qc["face_crop_manifest_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
