"""Read-only integrity and complete media decoding for a pinned AVI release.

The caller persists ``records`` in restricted storage. Only ``aggregate`` is suitable
for an aggregate project report. Source files are never rewritten. Hashing, probing
and decoding share one held, no-symlink file descriptor; a second streaming hash and
identity check detect a source changed while it was being examined.

Full decoding uses the following command (``FD`` is an inherited descriptor):
``ffmpeg -nostdin -hide_banner -v info -xerror -err_detect explode -threads 1
-filter_threads 1 -filter_complex_threads 1 -i /dev/fd/FD -map 0:v -map 0:a?
-af astats=metadata=0:reset=0:measure_perchannel=none:
measure_overall=Number_of_samples -c:v wrapped_avframe -c:a pcm_s16le -threads 1
-progress pipe:1 -nostats -f null -``. All video/audio streams are mapped. Completion
requires a final progress marker, positive decoded video frames and positive sample
counts for every probed audio stream, not just a readable MP4 header. One worker runs
at most one FFmpeg process, and ``workers`` bounds the concurrent file audits.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any

VALIDATION_SCHEMA_VERSION = "avi-release-validation-v1"
_HASH_CHUNK_BYTES = 1024 * 1024
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_AUDIO_SAMPLES = re.compile(r"Number of samples:\s*([0-9]+)(?:\.0+)?\s*(?:\n|$)")


def validate_release_files(
    root: str | Path,
    manifest: Mapping[str, Any],
    *,
    workers: int = 4,
    full_decode: bool = True,
    progress: Callable[[int, int, dict[str, Any]], None] | None = None,
    validated_records: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate pinned files, returning aggregate findings and restricted records.

    ``manifest`` contains nonempty ``repo_id``/``revision`` strings and a ``files``
    list. Each file has ``path``, integer ``size``, Git ``blob_id`` and nullable
    ``lfs_sha256``. An LFS object is verified against its SHA-256, never against the
    Git hash of its pointer. Ordinary files use SHA-1 of ``blob <size>\0<bytes>``.

    Invalid manifests/root paths raise ``ValueError`` before content is read.
    Missing, unsafe, corrupt or undecodable files instead produce failed records.
    ``progress(completed, total, record)`` receives an isolated record copy. Record
    order is UTF-8 path order regardless of scheduling. ``complete`` can be true
    only following full decoding of every expected MP4; probe-only runs are clearly
    distinguished. ``validated_records`` may contain trusted, restricted receipts
    from a previous invocation. Every source is still rehashed, but a same-schema
    successful full-decode receipt is reused only if its manifest identity and
    source SHA-256 match. Fresh versus reused decodes are separately counted.
    Duration warnings are diagnostics, not exclusion rules.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 32:
        raise ValueError("workers must be an integer from 1 through 32")
    if not isinstance(full_decode, bool):
        raise ValueError("full_decode must be a boolean")
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable")
    if validated_records is not None and not isinstance(validated_records, Mapping):
        raise ValueError("validated_records must be a mapping")
    receipts = validated_records or {}
    files = _validate_manifest(manifest)
    tool_identity = (
        _tool_identity() if any(entry["path"].lower().endswith(".mp4") for entry in files) else {}
    )
    root_path = Path(root).expanduser().absolute()
    if ".." in root_path.parts:
        raise ValueError("release root may not contain parent-relative components")
    try:
        root_fd = _open_root(root_path)
    except OSError as error:
        raise ValueError("release root is unavailable or has a symlink component") from error
    try:
        root_identity = _directory_identity(os.fstat(root_fd))
        records: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _validate_file,
                    root_fd,
                    entry,
                    full_decode,
                    receipts.get(entry["path"]),
                    tool_identity,
                ): entry
                for entry in files
            }
            for completed, future in enumerate(as_completed(futures), 1):
                record = future.result()
                records.append(record)
                if progress is not None:
                    progress(completed, len(files), copy.deepcopy(record))
        check_fd = _open_root(root_path)
        try:
            if _directory_identity(os.fstat(check_fd)) != root_identity:
                raise ValueError("release root changed during validation")
        finally:
            os.close(check_fd)
    finally:
        os.close(root_fd)
    if tool_identity and _tool_identity() != tool_identity:
        raise ValueError("media tools changed during validation")
    records.sort(key=lambda row: row["path"].encode("utf-8"))
    duplicate_payloads = _duplicate_payloads(records)
    aggregate = _aggregate(manifest, records, duplicate_payloads, full_decode)
    return {"aggregate": aggregate, "records": records, "duplicate_payloads": duplicate_payloads}


def _validate_manifest(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(manifest, Mapping):
        raise ValueError("source manifest must be a mapping")
    for name in ("repo_id", "revision"):
        if not isinstance(manifest.get(name), str) or not manifest[name].strip():
            raise ValueError(f"manifest {name} must be a nonempty string")
    supplied = manifest.get("files")
    if not isinstance(supplied, list) or not supplied:
        raise ValueError("manifest files must be a nonempty list")
    files: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in supplied:
        if not isinstance(entry, Mapping):
            raise ValueError("manifest file entries must be mappings")
        path = entry.get("path")
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or "\\" in path
            or any(ord(character) < 32 for character in path)
            or any(component in {"", ".", ".."} for component in path.split("/"))
        ):
            raise ValueError("manifest paths must be canonical relative POSIX paths")
        if path in seen:
            raise ValueError("manifest contains a duplicate file path")
        seen.add(path)
        size = entry.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("manifest file size must be a nonnegative integer")
        blob_id = entry.get("blob_id")
        if not isinstance(blob_id, str) or _HEX40.fullmatch(blob_id) is None:
            raise ValueError("manifest blob_id must be lowercase Git SHA-1")
        if "lfs_sha256" not in entry:
            raise ValueError("manifest file must explicitly specify nullable lfs_sha256")
        lfs_sha256 = entry["lfs_sha256"]
        if lfs_sha256 is not None and (
            not isinstance(lfs_sha256, str) or _HEX64.fullmatch(lfs_sha256) is None
        ):
            raise ValueError("manifest lfs_sha256 must be lowercase SHA-256 or null")
        files.append({"path": path, "size": size, "blob_id": blob_id, "lfs_sha256": lfs_sha256})
    return sorted(files, key=lambda row: row["path"].encode("utf-8"))


def _open_root(path: Path) -> int:
    current_fd = os.open("/", _DIRECTORY_FLAGS)
    try:
        for component in path.parts[1:]:
            child_fd = os.open(component, _DIRECTORY_FLAGS, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = child_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


@contextmanager
def _open_file(root_fd: int, path: str) -> Iterator[int]:
    parent_fd = os.dup(root_fd)
    file_fd: int | None = None
    try:
        components = path.split("/")
        for component in components[:-1]:
            child_fd = os.open(component, _DIRECTORY_FLAGS, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = child_fd
        file_fd = os.open(components[-1], _FILE_FLAGS, dir_fd=parent_fd)
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            raise ValueError("source is not a regular file")
        yield file_fd
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(parent_fd)


def _identity(observed: os.stat_result) -> tuple[int, ...]:
    return (
        observed.st_dev,
        observed.st_ino,
        observed.st_mode,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
        observed.st_nlink,
    )


def _directory_identity(observed: os.stat_result) -> tuple[int, int, int]:
    # File creation by the downloader may change directory mtime/nlink harmlessly.
    return observed.st_dev, observed.st_ino, observed.st_mode


def _tool_identity() -> dict[str, str | None]:
    identity: dict[str, str | None] = {}
    for tool in ("ffmpeg", "ffprobe"):
        executable = shutil.which(tool)
        digest = hashlib.sha256()
        try:
            if executable is None:
                raise FileNotFoundError(tool)
            with open(executable, "rb") as opened:
                before = _identity(os.fstat(opened.fileno()))
                while chunk := opened.read(_HASH_CHUNK_BYTES):
                    digest.update(chunk)
                if _identity(os.fstat(opened.fileno())) != before:
                    raise ValueError("tool binary changed during hashing")
            identity[f"{tool}_binary_sha256"] = digest.hexdigest()
        except (OSError, ValueError):
            identity[f"{tool}_binary_sha256"] = None
    return identity


def _hash_open_file(file_fd: int) -> tuple[str, str]:
    size = os.fstat(file_fd).st_size
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1(f"blob {size}\0".encode("ascii"))
    offset = 0
    while chunk := os.pread(file_fd, _HASH_CHUNK_BYTES, offset):
        offset += len(chunk)
        sha256.update(chunk)
        git_sha1.update(chunk)
    if offset != size:
        raise ValueError("source size changed during hashing")
    return sha256.hexdigest(), git_sha1.hexdigest()


def _validate_file(
    root_fd: int,
    entry: dict[str, Any],
    full_decode: bool,
    receipt: Mapping[str, Any] | None = None,
    tool_identity: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    lfs = entry["lfs_sha256"]
    record: dict[str, Any] = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "path": entry["path"],
        "expected_size": entry["size"],
        "observed_size": None,
        "expected_hash_algorithm": "sha256" if lfs else "git_blob_sha1",
        "expected_hash": lfs or entry["blob_id"],
        "sha256": None,
        "git_blob_sha1": None,
        "hash_verified": False,
        "status": "failed",
        "issues": [],
        "warnings": [],
        "media": None,
        "decode_receipt_reused": False,
    }
    try:
        with _open_file(root_fd, entry["path"]) as file_fd:
            before = _identity(os.fstat(file_fd))
            record["observed_size"] = os.fstat(file_fd).st_size
            record["sha256"], record["git_blob_sha1"] = _hash_open_file(file_fd)
            if record["observed_size"] != entry["size"]:
                record["issues"].append("size_mismatch")
            observed_hash = record["sha256"] if lfs else record["git_blob_sha1"]
            if observed_hash != record["expected_hash"]:
                record["issues"].append("hash_mismatch")
            record["hash_verified"] = not record["issues"]
            if entry["path"].lower().endswith(".mp4"):
                if full_decode and _reusable_decode_receipt(receipt, record, tool_identity or {}):
                    record["media"] = copy.deepcopy(receipt["media"])
                    record["warnings"] = copy.deepcopy(receipt["warnings"])
                    record["decode_receipt_reused"] = True
                else:
                    record["media"] = _validate_media(file_fd, full_decode)
                    record["media"]["tool_identity"] = dict(tool_identity or {})
                    record["issues"].extend(record["media"].pop("issues"))
                    record["warnings"].extend(record["media"].pop("warnings"))
            if (
                _hash_open_file(file_fd) != (record["sha256"], record["git_blob_sha1"])
                or _identity(os.fstat(file_fd)) != before
            ):
                record["issues"].append("source_changed_during_validation")
                record["hash_verified"] = False
            with _open_file(root_fd, entry["path"]) as current_fd:
                if _identity(os.fstat(current_fd)) != before:
                    record["issues"].append("source_path_changed_during_validation")
                    record["hash_verified"] = False
    except FileNotFoundError:
        record["issues"].append("missing_file")
        record["hash_verified"] = False
    except (OSError, ValueError) as error:
        record["issues"].append("unsafe_or_unreadable_file")
        record["file_error"] = str(error)[:1000]
        record["hash_verified"] = False
    if not record["issues"]:
        record["status"] = "verified"
    return record


def _reusable_decode_receipt(
    receipt: Mapping[str, Any] | None,
    current: dict[str, Any],
    tool_identity: dict[str, str | None],
) -> bool:
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema_version") != VALIDATION_SCHEMA_VERSION
        or receipt.get("status") != "verified"
        or receipt.get("hash_verified") is not True
        or receipt.get("issues") != []
        or not current["hash_verified"]
        or not isinstance(receipt.get("warnings"), list)
        or not all(isinstance(warning, str) for warning in receipt["warnings"])
    ):
        return False
    for field in (
        "path",
        "expected_size",
        "observed_size",
        "expected_hash_algorithm",
        "expected_hash",
        "sha256",
        "git_blob_sha1",
    ):
        if receipt.get(field) != current[field]:
            return False
    media = receipt.get("media")
    if (
        not isinstance(media, dict)
        or media.get("probe_ok") is not True
        or media.get("decode_requested") is not True
        or media.get("decode_ok") is not True
        or media.get("tool_identity") != tool_identity
        or not all(tool_identity.get(f"{tool}_binary_sha256") for tool in ("ffmpeg", "ffprobe"))
        or media.get("video_stream_count") != 1
        or not isinstance(media.get("audio_stream_count"), int)
        or media["audio_stream_count"] < 1
        or not isinstance(media.get("streams"), list)
        or not isinstance(media.get("decoded_video_frames"), int)
        or media["decoded_video_frames"] <= 0
        or not isinstance(media.get("decoded_audio_samples"), list)
        or len(media["decoded_audio_samples"]) != media["audio_stream_count"]
        or not all(isinstance(count, int) and count > 0 for count in media["decoded_audio_samples"])
    ):
        return False
    return True


def _run_process(
    command: list[str], file_fd: int, *, timeout: float
) -> subprocess.CompletedProcess:
    # A duplicate has its own lifetime; /dev/fd avoids reopening by source pathname.
    os.lseek(file_fd, 0, os.SEEK_SET)
    duplicate = os.dup(file_fd)
    try:
        actual = [part.replace("{FD}", str(duplicate)) for part in command]
        return subprocess.run(
            actual,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout,
            pass_fds=(duplicate,),
        )
    finally:
        os.close(duplicate)


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def _validate_media(file_fd: int, full_decode: bool) -> dict[str, Any]:
    media: dict[str, Any] = {
        "probe_ok": False,
        "streams": [],
        "video_stream_count": 0,
        "audio_stream_count": 0,
        "format_duration_seconds": None,
        "decode_requested": full_decode,
        "decode_ok": False,
        "decoded_video_frames": None,
        "decoded_audio_samples": [],
        "issues": [],
        "warnings": [],
    }
    probe_command = [
        "ffprobe",
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-of",
        "json",
        "/dev/fd/{FD}",
    ]
    try:
        probe = _run_process(probe_command, file_fd, timeout=60)
        if probe.returncode != 0:
            media["issues"].append("probe_failed")
            media["probe_error"] = probe.stderr[-4000:]
            return media
        payload = json.loads(probe.stdout)
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("streams"), list)
            or not all(isinstance(stream, dict) for stream in payload["streams"])
            or not isinstance(payload.get("format"), dict)
        ):
            raise ValueError("unexpected FFprobe JSON schema")
    except (json.JSONDecodeError, ValueError):
        media["issues"].append("invalid_probe_json")
        return media
    except subprocess.TimeoutExpired:
        media["issues"].append("probe_timeout")
        return media
    except OSError:
        media["issues"].append("probe_tool_unavailable")
        return media
    media["probe_ok"] = True
    media["format_duration_seconds"] = _number(payload["format"].get("duration"))
    fields = (
        "index",
        "codec_type",
        "codec_name",
        "width",
        "height",
        "sample_rate",
        "channels",
        "r_frame_rate",
        "avg_frame_rate",
        "time_base",
        "duration_ts",
        "nb_frames",
    )
    for stream in payload["streams"]:
        summary = {name: stream.get(name) for name in fields if name in stream}
        summary["duration_seconds"] = _number(stream.get("duration"))
        summary["start_seconds"] = _number(stream.get("start_time"))
        media["streams"].append(summary)
    videos = [stream for stream in media["streams"] if stream.get("codec_type") == "video"]
    audios = [stream for stream in media["streams"] if stream.get("codec_type") == "audio"]
    media["video_stream_count"], media["audio_stream_count"] = len(videos), len(audios)
    if len(videos) != 1:
        media["issues"].append("unexpected_video_stream_count")
    if not audios:
        media["issues"].append("missing_audio_stream")
    elif len(audios) != 1:
        media["warnings"].append("multiple_audio_streams")
    for stream in videos:
        if not all(
            isinstance(stream.get(field), int) and stream[field] > 0
            for field in ("width", "height")
        ):
            media["issues"].append("invalid_video_dimensions")
    durations = [stream["duration_seconds"] for stream in [*videos, *audios]]
    if any(duration is not None and duration <= 0 for duration in durations):
        media["warnings"].append("nonpositive_stream_duration")
    positive = [duration for duration in durations if duration is not None and duration > 0]
    if positive and max(positive) - min(positive) > max(0.25, max(positive) * 0.01):
        media["warnings"].append("audio_video_duration_mismatch")
    starts = [stream["start_seconds"] for stream in [*videos, *audios]]
    starts = [start for start in starts if start is not None]
    if starts and max(starts) - min(starts) > 0.25:
        media["warnings"].append("audio_video_start_mismatch")
    if not positive and not (
        media["format_duration_seconds"] is not None and media["format_duration_seconds"] > 0
    ):
        media["warnings"].append("missing_positive_duration")
    if full_decode:
        _decode_media(file_fd, media)
    return media


def _decode_media(file_fd: int, media: dict[str, Any]) -> None:
    decode_command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-v",
        "info",
        "-xerror",
        "-err_detect",
        "explode",
        "-threads",
        "1",
        "-filter_threads",
        "1",
        "-filter_complex_threads",
        "1",
        "-i",
        "/dev/fd/{FD}",
        "-map",
        "0:v",
        "-map",
        "0:a?",
        "-af",
        "astats=metadata=0:reset=0:measure_perchannel=none:measure_overall=Number_of_samples",
        "-c:v",
        "wrapped_avframe",
        "-c:a",
        "pcm_s16le",
        "-threads",
        "1",
        "-progress",
        "pipe:1",
        "-nostats",
        "-f",
        "null",
        "-",
    ]
    duration = media["format_duration_seconds"] or 0
    try:
        decoded = _run_process(
            decode_command, file_fd, timeout=min(3600, max(120, duration * 4 + 60))
        )
    except subprocess.TimeoutExpired:
        media["issues"].append("decode_timeout")
        return
    except OSError:
        media["issues"].append("decode_tool_unavailable")
        return
    if decoded.returncode != 0:
        media["issues"].append("decode_failed")
        media["decode_error"] = decoded.stderr[-4000:]
        return
    progress = {}
    for line in decoded.stdout.splitlines():
        if "=" in line:
            name, value = line.split("=", 1)
            progress[name] = value.strip()
    try:
        media["decoded_video_frames"] = int(progress.get("frame", "0"))
    except ValueError:
        media["decoded_video_frames"] = 0
    media["decoded_audio_samples"] = [
        int(count) for count in _AUDIO_SAMPLES.findall(decoded.stderr)
    ]
    if progress.get("progress") != "end":
        media["issues"].append("decode_completion_unproven")
    if media["decoded_video_frames"] <= 0:
        media["issues"].append("no_decoded_video_frames")
    if (
        len(media["decoded_audio_samples"]) != media["audio_stream_count"]
        or not media["decoded_audio_samples"]
        or any(count <= 0 for count in media["decoded_audio_samples"])
    ):
        media["issues"].append("audio_decode_unproven")
    media["decode_ok"] = not any(
        issue in media["issues"]
        for issue in (
            "decode_completion_unproven",
            "no_decoded_video_frames",
            "audio_decode_unproven",
        )
    )


def _duplicate_payloads(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[str]] = {}
    for record in records:
        if record["sha256"] is not None:
            groups.setdefault(record["sha256"], []).append(record["path"])
    return [
        {"sha256": digest, "paths": paths}
        for digest, paths in sorted(groups.items())
        if len(paths) > 1
    ]


def _aggregate(
    manifest: Mapping[str, Any],
    records: list[dict[str, Any]],
    duplicates: list[dict[str, Any]],
    full_decode: bool,
) -> dict[str, Any]:
    media = [record["media"] for record in records if record["media"] is not None]
    expected_videos = sum(record["path"].lower().endswith(".mp4") for record in records)
    verified = sum(record["status"] == "verified" for record in records)
    decoded = sum(item["decode_ok"] for item in media)
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "repo_id": manifest["repo_id"],
        "revision": manifest["revision"],
        "validation_mode": "full_decode" if full_decode else "hash_and_probe_only",
        "complete": full_decode and verified == len(records) and decoded == expected_videos,
        "expected_file_count": len(records),
        "present_file_count": sum(record["observed_size"] is not None for record in records),
        "hash_verified_file_count": sum(record["hash_verified"] for record in records),
        "verified_file_count": verified,
        "failed_file_count": len(records) - verified,
        "expected_bytes": sum(record["expected_size"] for record in records),
        "hash_verified_bytes": sum(
            record["observed_size"] for record in records if record["hash_verified"]
        ),
        "expected_mp4_count": expected_videos,
        "probed_mp4_count": sum(item["probe_ok"] for item in media),
        "fully_decoded_mp4_count": decoded,
        "receipt_reused_mp4_count": sum(record["decode_receipt_reused"] for record in records),
        "freshly_decoded_mp4_count": sum(
            record["media"] is not None
            and record["media"]["decode_ok"]
            and not record["decode_receipt_reused"]
            for record in records
        ),
        "decoded_video_frame_count": sum(item["decoded_video_frames"] or 0 for item in media),
        "decoded_audio_sample_count": sum(sum(item["decoded_audio_samples"]) for item in media),
        "duplicate_payload_group_count": len(duplicates),
        "duplicate_payload_extra_file_count": sum(len(group["paths"]) - 1 for group in duplicates),
        "issue_counts": dict(
            sorted(Counter(issue for record in records for issue in record["issues"]).items())
        ),
        "warning_counts": dict(
            sorted(Counter(warning for record in records for warning in record["warnings"]).items())
        ),
    }
