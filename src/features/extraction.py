"""Deterministic, resumable multimodal feature extraction.

The ordinary data loader supplies clip metadata and transcript text.  This module
adds an explicit media boundary: DINOv2 accepts only already-selected principal-
face crops, while WavLM accepts only decoded mono audio from the same validated
active-interview interval.  Face detection is deliberately not guessed here; a crop
provider must either return provenance-carrying crops or a structured missing-modality
failure.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import platform
import random
import subprocess
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.data import (
    TRANSCRIPT_PARSER_REVISION,
    MITInterviewFeatureSession,
    MITInterviewSession,
    SessionMappingContext,
)
from src.data.annotation_stimuli import (
    FACE_FRAME_FIELDS,
    FACE_MANIFEST_SCHEMA_VERSION,
    FACE_SELECTOR_FIELDS,
    load_face_crop_manifest,
    validate_face_generator,
    validate_face_selector,
)
from src.secure_io import (
    AnchoredDirectory,
    AnchoredFile,
    AnchoredOpenFile,
    AnchoredSubprocessFile,
)

EXTRACTOR_VERSION = "fair-avi-mit-extractor-4"
SCHEMA_VERSION = "fair-avi-mit-feature-shard-4"
SHARD_KEYS = (
    "video",
    "video_mask",
    "video_timestamps_seconds",
    "video_mean",
    "audio",
    "audio_mask",
    "audio_second_indices",
    "audio_mean",
    "text",
    "text_mask",
    "text_token_start",
    "text_token_count",
    "text_mean",
)
_AUDIO_TIMING_FIELDS = {
    "codec_name",
    "sample_rate",
    "start_pts",
    "duration_ts",
    "time_base_numerator",
    "time_base_denominator",
}
_AUDIO_MODALITY_FIELDS = {
    "status",
    "context",
    "boundary_source",
    "full_sample_count",
    "full_duration_seconds",
    "source_embedded_audio_timing",
    "supplied_audio_timing",
    "supplied_audio_matches_embedded_duration",
    "decoded_audio_matches_supplied_duration",
    "active_start_sample",
    "requested_active_end_sample_exclusive",
    "active_end_sample_exclusive",
    "active_sample_count",
    "active_start_seconds",
    "active_end_seconds_exclusive",
    "active_duration_seconds",
    "available_active_duration_seconds",
    "end_alignment_policy",
    "end_alignment_shortfall_samples",
    "end_alignment_shortfall_seconds",
    "end_alignment_tolerance_samples",
    "window_count_active",
    "selected_windows",
    "complete_active_interval_representation",
}
_PRODUCTION_CROP_BINDING_FIELDS = {
    "generator",
    *FACE_SELECTOR_FIELDS,
    "schema_version",
    "source_video_sha256",
    "full_duration_seconds",
    "full_one_hz_grid_count",
    "active_start_grid_index",
    "active_end_grid_index_exclusive",
    "active_close_reason",
    "frame_rate_numerator",
    "frame_rate_denominator",
    "valid_candidate_grid_indices",
    "selected_grid_indices",
    "requested_fps",
    "manifest_clip_id",
    "frames",
    "input_fingerprint",
}


class ExtractionError(RuntimeError):
    """A structured extraction error with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class _FeatureArtifactStateError(RuntimeError):
    """A non-excludable output-state violation that must abort the run."""


@dataclass(frozen=True, slots=True)
class FaceCropBatch:
    """Principal-face crops sampled at deterministic clip timestamps."""

    images: tuple[Any, ...]
    timestamps_seconds: np.ndarray
    provenance: Mapping[str, Any]


class FaceCropProvider(Protocol):
    """Boundary implemented by a separately frozen principal-face selector."""

    def __call__(self, media_path: Path | AnchoredOpenFile, *, fps: int) -> FaceCropBatch:
        """Return only selected principal-face crops, never uncropped frames."""


class VideoEncoder(Protocol):
    name: str
    revision: str
    out_dim: int

    def encode(self, images: Sequence[Any]) -> np.ndarray:
        """Return final-layer CLS vectors in input order."""


class AudioEncoder(Protocol):
    name: str
    revision: str
    out_dim: int
    sample_rate: int
    conv_stride: int
    receptive_field: int

    def encode(self, waveform: np.ndarray) -> np.ndarray:
        """Return valid final hidden frames for one unpadded mono waveform."""


class TextEncoder(Protocol):
    name: str
    revision: str
    out_dim: int
    max_sequence_tokens: int

    def tokenize(self, text: str) -> list[int]:
        """Tokenize without specials or truncation."""

    def special_token_count(self) -> int:
        """Return the number of specials added to one sequence."""

    def encode_token_chunks(self, chunks: Sequence[Sequence[int]]) -> np.ndarray:
        """Return one artifact-pooled, L2-normalised vector per chunk."""


class TranscriptProvider(Protocol):
    """Resolve a supplied transcript from inline content or its immutable reference."""

    def __call__(self, clip: MITInterviewSession | MITInterviewFeatureSession) -> str:
        """Return corpus-supplied text without transcription or cleaning."""


@dataclass(frozen=True, slots=True)
class ExtractionSettings:
    """Frozen preprocessing and output contract loaded from the governing config."""

    seed: int
    video_fps: int
    video_dim: int
    audio_sample_rate: int
    audio_dim: int
    audio_pooled_dim: int
    audio_conv_stride: int
    audio_receptive_field: int
    text_dim: int
    text_max_sequence_tokens: int
    video_max_samples: int
    audio_window_samples: int
    audio_max_windows: int
    source_date_epoch: int

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        source_date_epoch: int | None = None,
    ) -> ExtractionSettings:
        """Validate and construct settings from the feature-extraction config."""
        try:
            feature = config["feature_extraction"]
            encoders = config["encoders"]
            video = encoders["video"]
            audio = encoders["audio"]
            text = encoders["text"]
        except (KeyError, TypeError) as error:
            raise ValueError("config is missing feature extraction fields") from error
        if not all(isinstance(item, Mapping) for item in (feature, video, audio, text)):
            raise ValueError("feature extraction and encoder config entries must be mappings")
        epoch = source_date_epoch
        if epoch is None:
            raw_epoch = os.environ.get("SOURCE_DATE_EPOCH")
            if raw_epoch is None:
                raise ValueError("SOURCE_DATE_EPOCH (or an explicit source_date_epoch) is required")
            try:
                epoch = int(raw_epoch)
            except ValueError as error:
                raise ValueError("SOURCE_DATE_EPOCH must be an integer") from error
        settings = cls(
            seed=int(feature["seed"]),
            video_fps=int(video["fps"]),
            video_dim=int(video["out_dim"]),
            audio_sample_rate=int(audio["sample_rate"]),
            audio_dim=int(audio["out_dim"]),
            audio_pooled_dim=int(audio["pooled_dim"]),
            audio_conv_stride=int(audio["conv_stride_samples"]),
            audio_receptive_field=int(audio["receptive_field_samples"]),
            text_dim=int(text["out_dim"]),
            text_max_sequence_tokens=int(text["max_sequence_tokens"]),
            video_max_samples=int(feature["candidate_face_context"]["maximum_samples"]),
            audio_window_samples=(
                int(feature["active_interview_audio_context"]["nonoverlap_window_seconds"])
                * int(audio["sample_rate"])
            ),
            audio_max_windows=int(feature["active_interview_audio_context"]["maximum_windows"]),
            source_date_epoch=epoch,
        )
        if settings.seed < 0 or settings.source_date_epoch < 0:
            raise ValueError("seed and SOURCE_DATE_EPOCH must be non-negative")
        if settings.video_fps <= 0 or settings.audio_sample_rate <= 0:
            raise ValueError("sampling rates must be positive")
        if settings.video_max_samples != 128:
            raise ValueError("candidate-face context must be capped at 128 samples")
        if settings.audio_window_samples != 320_000 or settings.audio_max_windows != 16:
            raise ValueError("audio context must use at most sixteen 20-second windows")
        if settings.audio_pooled_dim != settings.audio_dim * 2:
            raise ValueError("audio pooled_dim must concatenate mean then max")
        expected_contract = {
            "feature.schema_version": (feature.get("schema_version"), SCHEMA_VERSION),
            "feature.extractor_version": (feature.get("extractor_version"), EXTRACTOR_VERSION),
            "video.name": (video.get("name"), "facebook/dinov2-large"),
            "video.architecture": (video.get("architecture"), "dinov2_vitl14"),
            "video.revision": (
                video.get("revision"),
                "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
            ),
            "video.weight_file": (video.get("weight_file"), "model.safetensors"),
            "video.weight_sha256": (
                video.get("weight_sha256"),
                "399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23",
            ),
            "video.out_dim": (settings.video_dim, 1024),
            "video.fps": (settings.video_fps, 1),
            "video.inference_batch_size": (video.get("inference_batch_size"), 4),
            "video.input": (video.get("input"), "preselected_principal_face_crop"),
            "video.processor": (
                video.get("processor"),
                "rgb_bicubic_shortest_edge_256_center_crop_224_rescale_imagenet_normalize",
            ),
            "video.output": (video.get("output"), "final_last_hidden_state_cls_index_0"),
            "audio.name": (audio.get("name"), "microsoft/wavlm-base-plus"),
            "audio.revision": (
                audio.get("revision"),
                "4c66d4806a428f2e922ccfa1a962776e232d487b",
            ),
            "audio.weight_file": (audio.get("weight_file"), "pytorch_model.bin"),
            "audio.weight_sha256": (
                audio.get("weight_sha256"),
                "3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51",
            ),
            "audio.out_dim": (settings.audio_dim, 768),
            "audio.sample_rate": (settings.audio_sample_rate, 16_000),
            "audio.waveform_normalization": (audio.get("waveform_normalization"), False),
            "audio.conv_stride_samples": (settings.audio_conv_stride, 320),
            "audio.receptive_field_samples": (settings.audio_receptive_field, 400),
            "audio.output": (audio.get("output"), "final_last_hidden_state"),
            "audio.pool": (
                audio.get("pool"),
                "per_selected_active_interval_window_center_time_half_open_one_second_mean_then_max",
            ),
            "audio.frame_center_seconds": (
                audio.get("frame_center_seconds"),
                "(j*320+199.5)/16000",
            ),
            "audio.partial_last_second": (
                audio.get("partial_last_second"),
                "retain_if_nonempty",
            ),
            "audio.end_alignment": (
                feature["active_interview_audio_context"].get("end_alignment"),
                "allow_end_of_stream_shortfall_below_one_source_audio_tick",
            ),
            "audio.embedded_audio_time_base": (
                feature["active_interview_audio_context"].get("embedded_audio_time_base"),
                "3/125",
            ),
            "audio.supplied_audio_time_base": (
                feature["active_interview_audio_context"].get("supplied_audio_time_base"),
                "1/48000",
            ),
            "text.source": (text.get("source"), "candidate_only_supplied_transcript"),
            "text.name": (text.get("name"), "sentence-transformers/all-mpnet-base-v2"),
            "text.revision": (
                text.get("revision"),
                "e8c3b32edf5434bc2275fc9bab85f82640a19130",
            ),
            "text.weight_file": (text.get("weight_file"), "model.safetensors"),
            "text.weight_sha256": (
                text.get("weight_sha256"),
                "78c0197b6159d92658e319bc1d72e4c73a9a03dd03815e70e555c5ef05615658",
            ),
            "text.out_dim": (settings.text_dim, 768),
            "text.max_sequence_tokens": (settings.text_max_sequence_tokens, 384),
            "text.content_tokens_per_chunk": (text.get("content_tokens_per_chunk"), 382),
            "text.chunks": (text.get("chunks"), "tokenizer_ids_nonoverlapping"),
            "text.pool": (
                text.get("pool"),
                "artifact_attention_mask_mean_then_l2_normalize",
            ),
        }
        drift = [
            f"{name}: expected {expected!r}, got {actual!r}"
            for name, (actual, expected) in expected_contract.items()
            if actual != expected
        ]
        if drift:
            raise ValueError("feature extraction contract drift: " + "; ".join(drift))
        for modality, encoder in (("video", video), ("audio", audio), ("text", text)):
            for field, length in (("revision", 40), ("weight_sha256", 64)):
                value = encoder.get(field)
                if (
                    not isinstance(value, str)
                    or len(value) != length
                    or any(character not in "0123456789abcdef" for character in value)
                ):
                    raise ValueError(
                        f"encoders.{modality}.{field} must be a lowercase hexadecimal pin"
                    )
            artifacts = encoder.get("artifact_files")
            if not isinstance(artifacts, Mapping) or not artifacts:
                raise ValueError(f"encoders.{modality}.artifact_files must be a non-empty mapping")
            if artifacts.get(encoder["weight_file"]) != encoder["weight_sha256"]:
                raise ValueError(
                    f"encoders.{modality}.artifact_files must contain the exact frozen weight"
                )
            for filename, digest in artifacts.items():
                if (
                    not isinstance(filename, str)
                    or not filename
                    or Path(filename).is_absolute()
                    or ".." in Path(filename).parts
                ):
                    raise ValueError(
                        f"encoders.{modality}.artifact_files names must be repository-relative"
                    )
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                ):
                    raise ValueError(
                        f"encoders.{modality}.artifact_files[{filename!r}] must be a SHA-256"
                    )
        return settings


@dataclass(frozen=True, slots=True)
class CodeProvenance:
    """Commit identity plus a checksum that identifies uncommitted source changes."""

    commit_sha: str
    source_tree_sha256: str
    dirty: bool


@dataclass(frozen=True, slots=True)
class ExtractionSummary:
    """Counts and stable report rows returned by a bounded extraction run."""

    succeeded: int
    resumed: int
    failed: int
    records: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class AudioStreamTiming:
    """Exact FFprobe timing fields for one selected audio stream."""

    codec_name: str
    sample_rate: int
    start_pts: int | None
    duration_ts: int
    time_base_numerator: int
    time_base_denominator: int


@dataclass(frozen=True, slots=True)
class ExtractionDependencies:
    """Injected decoders and encoders used by the orchestration layer."""

    face_crops: FaceCropProvider
    decode_audio: Callable[[Path | AnchoredOpenFile, int], np.ndarray]
    video_encoder: VideoEncoder
    audio_encoder: AudioEncoder
    text_encoder: TextEncoder
    transcript: TranscriptProvider
    face_crop_fingerprint: str
    face_crop_fingerprint_probe: Callable[[], str]
    probe_video_duration: Callable[[Path | AnchoredOpenFile], float] | None = None
    probe_audio_stream_timing: Callable[[Path | AnchoredOpenFile], AudioStreamTiming] | None = None
    backend: str = "huggingface"
    device: str = "cpu"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if len(self.face_crop_fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in self.face_crop_fingerprint
        ):
            raise ValueError("face_crop_fingerprint must be a lowercase SHA-256 digest")
        if not callable(self.face_crop_fingerprint_probe):
            raise ValueError("face_crop_fingerprint_probe must be callable")


def set_deterministic_runtime(seed: int) -> None:
    """Seed Python/NumPy/Torch and request deterministic Torch algorithms."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
    except ImportError:
        pass


def safe_clip_stem(clip_id: str) -> str:
    """Return a path-safe, collision-resistant shard stem for an arbitrary clip ID."""
    if not isinstance(clip_id, str) or not clip_id or "\0" in clip_id:
        raise ValueError("clip_id must be a non-empty, NUL-free string")
    digest = hashlib.sha256(clip_id.encode("utf-8")).hexdigest()
    return f"clip-{digest}"


def file_sha256(path: Path | AnchoredOpenFile) -> str:
    """Hash a file without loading it all into memory."""
    if isinstance(path, AnchoredOpenFile):
        path.verify_current_file()
        return path.sha256
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize a sidecar deterministically as UTF-8 JSON."""
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def write_canonical_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a deterministic JSON report."""
    _atomic_write(path, canonical_json_bytes(payload))


def code_provenance(repository_root: Path, owned_paths: Sequence[str]) -> CodeProvenance:
    """Identify committed and uncommitted extractor source without trusting HEAD alone."""
    commit = _git(repository_root, "rev-parse", "HEAD").strip()
    if len(commit) != 40:
        raise ExtractionError("code_provenance_unavailable", "cannot resolve a full source commit")
    digest = hashlib.sha256()
    dirty = False
    ignored_parts = {"__pycache__", ".pytest_cache", ".ruff_cache"}
    for relative in sorted(owned_paths):
        path = repository_root / relative
        digest.update(relative.encode("utf-8") + b"\0")
        if path.is_file():
            data = path.read_bytes()
            digest.update(hashlib.sha256(data).digest())
        elif path.is_dir():
            for child in sorted(
                item
                for item in path.rglob("*")
                if item.is_file()
                and not ignored_parts.intersection(item.relative_to(repository_root).parts)
                and child_suffix_allowed(item)
            ):
                child_relative = child.relative_to(repository_root).as_posix()
                digest.update(child_relative.encode("utf-8") + b"\0")
                digest.update(hashlib.sha256(child.read_bytes()).digest())
    status = _git(repository_root, "status", "--porcelain", "--", *owned_paths)
    dirty = bool(status.strip())
    return CodeProvenance(commit_sha=commit, source_tree_sha256=digest.hexdigest(), dirty=dirty)


def chunk_token_ids(
    token_ids: Sequence[int],
    *,
    max_sequence_tokens: int,
    special_token_count: int,
) -> tuple[tuple[int, ...], ...]:
    """Partition tokenizer IDs into non-overlapping chunks with room for specials."""
    capacity = max_sequence_tokens - special_token_count
    if capacity <= 0:
        raise ValueError("special tokens must leave positive content capacity")
    values = tuple(int(token_id) for token_id in token_ids)
    if not values:
        return ()
    return tuple(values[start : start + capacity] for start in range(0, len(values), capacity))


def uniformly_spaced_indices(count: int, maximum: int) -> tuple[int, ...]:
    """Select endpoint-spanning integer ranks without floating-point arithmetic."""
    if isinstance(count, bool) or isinstance(maximum, bool):
        raise TypeError("count and maximum must be integers")
    if count <= 0 or maximum <= 0:
        raise ValueError("count and maximum must be positive")
    if count <= maximum:
        return tuple(range(count))
    if maximum == 1:
        return (0,)
    return tuple((rank * (count - 1)) // (maximum - 1) for rank in range(maximum))


def bounded_audio_windows(
    sample_count: int, *, window_samples: int = 320_000, maximum_windows: int = 16
) -> tuple[tuple[int, int, int], ...]:
    """Return selected ``(window_index,start,end)`` non-overlapping audio blocks."""
    if sample_count <= 0 or window_samples <= 0 or maximum_windows <= 0:
        raise ValueError("audio window arguments must be positive")
    count = (sample_count + window_samples - 1) // window_samples
    return tuple(
        (index, index * window_samples, min((index + 1) * window_samples, sample_count))
        for index in uniformly_spaced_indices(count, maximum_windows)
    )


def wavlm_output_length(
    sample_count: int,
    kernels: Sequence[int] = (10, 3, 3, 3, 3, 2, 2),
    strides: Sequence[int] = (5, 2, 2, 2, 2, 2, 2),
) -> int:
    """Apply WavLM's pinned convolution-length formula to unpadded samples."""
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    length = sample_count
    for kernel, stride in zip(kernels, strides, strict=True):
        length = (length - kernel) // stride + 1
        if length <= 0:
            return 0
    return length


def pool_wavlm_seconds(
    hidden: np.ndarray,
    *,
    sample_count: int,
    sample_rate: int = 16_000,
    conv_stride: int = 320,
    receptive_field: int = 400,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool valid WavLM frames by half-open one-second centre-time bins.

    Frame ``j`` has centre ``(j*320 + 199.5)/16000`` seconds under the
    frozen frontend. Each selected 20-second window is encoded independently, retaining
    window-local Transformer context. Non-empty bins, including a partial final second,
    concatenate elementwise mean then maximum.
    """
    values = np.asarray(hidden, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("WavLM hidden states must be a finite two-dimensional matrix")
    valid_length = wavlm_output_length(sample_count)
    if values.shape[0] < valid_length:
        raise ValueError("WavLM output is shorter than the pinned valid length")
    if values.shape[0] > valid_length:
        raise ValueError("WavLM output is longer than the pinned unpadded valid length")
    values = values[:valid_length]
    if valid_length == 0:
        return np.empty((0, values.shape[1] * 2), dtype=np.float32), np.empty(0, dtype=np.int64)
    centres = (
        np.arange(valid_length, dtype=np.float64) * conv_stride + (receptive_field - 1) / 2
    ) / sample_rate
    second_indices = np.floor(centres).astype(np.int64)
    unique_seconds = np.unique(second_indices)
    pooled = np.stack(
        [
            np.concatenate(
                (
                    values[second_indices == second].mean(axis=0),
                    values[second_indices == second].max(axis=0),
                )
            )
            for second in unique_seconds
        ]
    ).astype(np.float32)
    return pooled, unique_seconds


def mean_valid(values: np.ndarray, mask: np.ndarray, *, out_dim: int) -> np.ndarray:
    """Return a float32 arithmetic mean over true mask rows."""
    matrix = np.asarray(values, dtype=np.float32)
    valid = np.asarray(mask, dtype=np.bool_)
    if matrix.ndim != 2 or matrix.shape[1] != out_dim or valid.shape != (matrix.shape[0],):
        raise ValueError("representation and mask shapes do not satisfy the feature schema")
    if not valid.any():
        return np.full(out_dim, np.nan, dtype=np.float32)
    return matrix[valid].mean(axis=0, dtype=np.float32)


def extract_clip_arrays(
    clip: MITInterviewSession | MITInterviewFeatureSession,
    video_path: Path | AnchoredOpenFile,
    audio_path: Path | AnchoredOpenFile,
    dependencies: ExtractionDependencies,
    settings: ExtractionSettings,
    *,
    transcript_text: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Extract all three fixed representations for one validated ordinary clip."""
    if not _media_file_available(video_path) or not _media_file_available(audio_path):
        raise ExtractionError("media_missing", "required video or audio file is unavailable")

    try:
        crops = dependencies.face_crops(video_path, fps=settings.video_fps)
    except ExtractionError:
        raise
    except Exception as error:
        raise ExtractionError("face_crop_failed", str(error)) from error
    if not crops.images:
        raise ExtractionError("face_crop_missing", "no selected principal-face crops were supplied")
    timestamps = np.asarray(crops.timestamps_seconds, dtype=np.float64)
    if timestamps.shape != (len(crops.images),) or not np.isfinite(timestamps).all():
        raise ExtractionError("face_crop_invalid", "face crop timestamps are invalid")
    crop_fingerprint = crops.provenance.get("input_fingerprint")
    if crop_fingerprint != dependencies.face_crop_fingerprint:
        raise ExtractionError(
            "face_crop_mutated_during_extraction",
            "principal-face crop inputs changed after the extraction fingerprint was computed",
        )
    if len(crops.images) > settings.video_max_samples:
        raise ExtractionError(
            "face_crop_invalid", "face-crop manifest exceeds the bounded 128-sample contract"
        )
    required_provenance = {
        "schema_version",
        "source_video_sha256",
        "full_duration_seconds",
        "full_one_hz_grid_count",
        "active_start_grid_index",
        "active_end_grid_index_exclusive",
        "active_close_reason",
        "valid_candidate_grid_indices",
        "selected_grid_indices",
        "input_fingerprint",
    }
    if not required_provenance.issubset(crops.provenance):
        raise ExtractionError("face_crop_invalid", "face-crop provenance is incomplete")
    if crops.provenance["schema_version"] != FACE_MANIFEST_SCHEMA_VERSION:
        raise ExtractionError("face_crop_invalid", "face-crop manifest schema is unsupported")
    if crops.provenance["source_video_sha256"] != file_sha256(video_path):
        raise ExtractionError("face_crop_invalid", "face crops are not bound to the source video")
    duration = _finite_number(crops.provenance["full_duration_seconds"], "video duration")
    if duration <= 0:
        raise ExtractionError("face_crop_invalid", "video duration must be positive")
    grid_count = int(crops.provenance["full_one_hz_grid_count"])
    if grid_count != max(1, math.floor(duration)):
        raise ExtractionError("face_crop_invalid", "face-crop grid does not match duration")
    duration_probe = dependencies.probe_video_duration or probe_video_duration_seconds
    measured_duration = duration_probe(video_path)
    if not math.isclose(duration, measured_duration, rel_tol=0.0, abs_tol=1e-3):
        raise ExtractionError(
            "face_crop_invalid", "face-crop duration does not match the bound source video"
        )
    active_start = crops.provenance["active_start_grid_index"]
    active_end = crops.provenance["active_end_grid_index_exclusive"]
    active_close_reason = crops.provenance["active_close_reason"]
    if (
        isinstance(active_start, bool)
        or not isinstance(active_start, int)
        or isinstance(active_end, bool)
        or not isinstance(active_end, int)
        or not 0 <= active_start < active_end <= grid_count
        or active_close_reason not in {"terminal_invalid_run", "end_of_stream"}
    ):
        raise ExtractionError("face_crop_invalid", "face-crop active interval is invalid")
    raw_valid_indices = crops.provenance["valid_candidate_grid_indices"]
    if not isinstance(raw_valid_indices, list) or any(
        isinstance(index, bool) or not isinstance(index, int) for index in raw_valid_indices
    ):
        raise ExtractionError("face_crop_invalid", "candidate-valid grid indices are invalid")
    valid_indices = tuple(raw_valid_indices)
    if (
        not valid_indices
        or valid_indices != tuple(sorted(set(valid_indices)))
        or valid_indices[0] != active_start
        or valid_indices[-1] != active_end - 1
        or any(not active_start <= index < active_end for index in valid_indices)
    ):
        raise ExtractionError(
            "face_crop_invalid", "candidate-valid indices do not bind the active interval"
        )
    selected_ranks = uniformly_spaced_indices(len(valid_indices), settings.video_max_samples)
    expected_indices = tuple(valid_indices[rank] for rank in selected_ranks)
    supplied_indices = crops.provenance["selected_grid_indices"]
    if tuple(supplied_indices) != expected_indices:
        raise ExtractionError("face_crop_invalid", "face-crop active-interval selection drifted")
    if len(crops.images) != len(expected_indices) or not np.array_equal(
        timestamps, np.asarray(expected_indices, dtype=np.float64)
    ):
        raise ExtractionError(
            "face_crop_missing", "every selected candidate-face timestamp must have one valid crop"
        )
    video = _finite_matrix(
        dependencies.video_encoder.encode(crops.images), settings.video_dim, "video"
    )
    if video.shape[0] != len(crops.images):
        raise ExtractionError(
            "video_shape_invalid", "video encoder returned one row per crop incorrectly"
        )
    video_mask = np.ones(video.shape[0], dtype=np.bool_)

    try:
        waveform = np.asarray(
            dependencies.decode_audio(audio_path, settings.audio_sample_rate), dtype=np.float32
        )
    except ExtractionError:
        raise
    except Exception as error:
        raise ExtractionError("audio_decode_failed", str(error)) from error
    if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
        raise ExtractionError(
            "audio_invalid", "decoded audio must be a non-empty finite mono waveform"
        )
    full_sample_count = int(waveform.size)
    audio_timing_probe = dependencies.probe_audio_stream_timing or probe_audio_stream_timing
    try:
        source_audio_timing = audio_timing_probe(video_path)
        supplied_audio_timing = audio_timing_probe(audio_path)
    except ExtractionError:
        raise
    except Exception as error:
        raise ExtractionError(
            "audio_source_timing_unavailable",
            "cannot verify the source video's embedded audio timing",
        ) from error
    if (
        not isinstance(source_audio_timing, AudioStreamTiming)
        or not isinstance(supplied_audio_timing, AudioStreamTiming)
        or not source_audio_timing.codec_name
        or not supplied_audio_timing.codec_name
        or source_audio_timing.sample_rate <= 0
        or supplied_audio_timing.sample_rate <= 0
        or source_audio_timing.duration_ts <= 0
        or supplied_audio_timing.duration_ts <= 0
        or source_audio_timing.start_pts != 0
        or supplied_audio_timing.start_pts not in {None, 0}
        or (
            source_audio_timing.time_base_numerator,
            source_audio_timing.time_base_denominator,
        )
        != (3, 125)
        or (
            supplied_audio_timing.time_base_numerator,
            supplied_audio_timing.time_base_denominator,
        )
        != (1, 48_000)
    ):
        raise ExtractionError(
            "audio_source_alignment_invalid",
            "supplied audio does not match the frozen embedded-audio timing contract",
        )
    source_duration_cross_product = (
        source_audio_timing.duration_ts
        * source_audio_timing.time_base_numerator
        * supplied_audio_timing.time_base_denominator
    )
    supplied_duration_cross_product = (
        supplied_audio_timing.duration_ts
        * supplied_audio_timing.time_base_numerator
        * source_audio_timing.time_base_denominator
    )
    decoded_sample_numerator = (
        supplied_audio_timing.duration_ts
        * supplied_audio_timing.time_base_numerator
        * settings.audio_sample_rate
    )
    if (
        source_duration_cross_product != supplied_duration_cross_product
        or decoded_sample_numerator % supplied_audio_timing.time_base_denominator != 0
        or decoded_sample_numerator // supplied_audio_timing.time_base_denominator
        != full_sample_count
    ):
        raise ExtractionError(
            "audio_source_alignment_invalid",
            "supplied audio duration does not exactly match the embedded audio stream",
        )
    active_start_sample = active_start * settings.audio_sample_rate
    requested_active_end_sample = active_end * settings.audio_sample_rate
    active_end_sample = min(requested_active_end_sample, full_sample_count)
    end_shortfall_samples = max(requested_active_end_sample - full_sample_count, 0)
    end_shortfall_tolerance_samples = math.ceil(
        settings.audio_sample_rate
        * source_audio_timing.time_base_numerator
        / source_audio_timing.time_base_denominator
    )
    if end_shortfall_samples and not (
        active_close_reason == "end_of_stream"
        and active_end == grid_count
        and end_shortfall_samples < end_shortfall_tolerance_samples
    ):
        raise ExtractionError(
            "audio_active_interval_unavailable",
            "decoded audio does not cover the frozen active interview interval",
        )
    active_waveform = waveform[active_start_sample:active_end_sample]
    if active_waveform.size == 0:
        raise ExtractionError(
            "audio_active_interval_unavailable", "active interview audio is empty"
        )
    selected_windows = bounded_audio_windows(
        active_waveform.size,
        window_samples=settings.audio_window_samples,
        maximum_windows=settings.audio_max_windows,
    )
    pooled_windows: list[np.ndarray] = []
    second_indices: list[int] = []
    for window_index, start, end in selected_windows:
        audio_hidden = _finite_matrix(
            dependencies.audio_encoder.encode(active_waveform[start:end]),
            settings.audio_dim,
            "audio hidden",
        )
        pooled, local_seconds = pool_wavlm_seconds(
            audio_hidden,
            sample_count=end - start,
            sample_rate=settings.audio_sample_rate,
            conv_stride=settings.audio_conv_stride,
            receptive_field=settings.audio_receptive_field,
        )
        pooled_windows.append(pooled)
        second_indices.extend(
            active_start
            + window_index * (settings.audio_window_samples // settings.audio_sample_rate)
            + int(value)
            for value in local_seconds
        )
    audio = np.concatenate(pooled_windows, axis=0)
    seconds = np.asarray(second_indices, dtype=np.int64)
    if audio.shape[0] == 0:
        raise ExtractionError("audio_too_short", "audio produced no valid WavLM frame")
    audio_mask = np.ones(audio.shape[0], dtype=np.bool_)

    transcript = transcript_text
    if transcript is None:
        try:
            transcript = dependencies.transcript(clip)
        except ExtractionError:
            raise
        except Exception as error:
            raise ExtractionError("transcript_load_failed", str(error)) from error
    if not isinstance(transcript, str) or not transcript.strip():
        raise ExtractionError("transcript_missing", "corpus-supplied transcript is missing")
    source_dialogue_sha256 = clip.transcript.source_dialogue_sha256
    if (
        not isinstance(source_dialogue_sha256, str)
        or len(source_dialogue_sha256) != 64
        or any(value not in "0123456789abcdef" for value in source_dialogue_sha256)
    ):
        raise ExtractionError(
            "transcript_provenance_invalid", "source-dialogue checksum is unavailable or invalid"
        )
    token_ids = dependencies.text_encoder.tokenize(transcript)
    special_count = dependencies.text_encoder.special_token_count()
    chunks = chunk_token_ids(
        token_ids,
        max_sequence_tokens=settings.text_max_sequence_tokens,
        special_token_count=special_count,
    )
    if not chunks:
        raise ExtractionError("transcript_empty", "supplied transcript tokenizes to no content")
    text = _finite_matrix(
        dependencies.text_encoder.encode_token_chunks(chunks), settings.text_dim, "text"
    )
    if text.shape[0] != len(chunks):
        raise ExtractionError(
            "text_shape_invalid", "text encoder returned one row per chunk incorrectly"
        )
    text_mask = np.ones(text.shape[0], dtype=np.bool_)
    starts = np.cumsum([0, *(len(chunk) for chunk in chunks[:-1])], dtype=np.int64)
    counts = np.asarray([len(chunk) for chunk in chunks], dtype=np.int64)

    arrays = {
        "video": video,
        "video_mask": video_mask,
        "video_timestamps_seconds": timestamps,
        "video_mean": mean_valid(video, video_mask, out_dim=settings.video_dim),
        "audio": audio,
        "audio_mask": audio_mask,
        "audio_second_indices": seconds,
        "audio_mean": mean_valid(audio, audio_mask, out_dim=settings.audio_pooled_dim),
        "text": text,
        "text_mask": text_mask,
        "text_token_start": starts,
        "text_token_count": counts,
        "text_mean": mean_valid(text, text_mask, out_dim=settings.text_dim),
    }
    _validate_arrays(arrays, settings)
    modality = {
        "video": {"status": "success", "crop_provider": dict(crops.provenance)},
        "audio": {
            "status": "success",
            "context": "active_interview_audio_not_candidate_voice",
            "boundary_source": "candidate_face_active_interval",
            "full_sample_count": full_sample_count,
            "full_duration_seconds": full_sample_count / settings.audio_sample_rate,
            "source_embedded_audio_timing": {
                "codec_name": source_audio_timing.codec_name,
                "sample_rate": source_audio_timing.sample_rate,
                "start_pts": source_audio_timing.start_pts,
                "duration_ts": source_audio_timing.duration_ts,
                "time_base_numerator": source_audio_timing.time_base_numerator,
                "time_base_denominator": source_audio_timing.time_base_denominator,
            },
            "supplied_audio_timing": {
                "codec_name": supplied_audio_timing.codec_name,
                "sample_rate": supplied_audio_timing.sample_rate,
                "start_pts": supplied_audio_timing.start_pts,
                "duration_ts": supplied_audio_timing.duration_ts,
                "time_base_numerator": supplied_audio_timing.time_base_numerator,
                "time_base_denominator": supplied_audio_timing.time_base_denominator,
            },
            "supplied_audio_matches_embedded_duration": True,
            "decoded_audio_matches_supplied_duration": True,
            "active_start_sample": active_start_sample,
            "requested_active_end_sample_exclusive": requested_active_end_sample,
            "active_end_sample_exclusive": active_end_sample,
            "active_sample_count": int(active_waveform.size),
            "active_start_seconds": active_start,
            "active_end_seconds_exclusive": active_end,
            "active_duration_seconds": active_end - active_start,
            "available_active_duration_seconds": (
                active_waveform.size / settings.audio_sample_rate
            ),
            "end_alignment_policy": (
                "allow_end_of_stream_shortfall_below_one_source_audio_tick"
            ),
            "end_alignment_shortfall_samples": end_shortfall_samples,
            "end_alignment_shortfall_seconds": (
                end_shortfall_samples / settings.audio_sample_rate
            ),
            "end_alignment_tolerance_samples": end_shortfall_tolerance_samples,
            "window_count_active": (active_waveform.size + settings.audio_window_samples - 1)
            // settings.audio_window_samples,
            "selected_windows": [
                {
                    "index": index,
                    "active_start_sample": start,
                    "active_end_sample_exclusive": end,
                    "source_start_sample": active_start_sample + start,
                    "source_end_sample_exclusive": active_start_sample + end,
                }
                for index, start, end in selected_windows
            ],
            "complete_active_interval_representation": len(selected_windows)
            == (active_waveform.size + settings.audio_window_samples - 1)
            // settings.audio_window_samples,
        },
        "text": {
            "status": "success",
            "source": "corpus_supplied",
            "reference": _portable_relative_path(
                clip.transcript.reference,
                code="transcript_reference_unsafe",
                label="transcript reference",
            ).as_posix(),
        },
    }
    return arrays, modality


def extract_dataset(
    clips: Sequence[MITInterviewSession | MITInterviewFeatureSession],
    *,
    media_root: Path,
    output_dir: Path,
    dependencies: ExtractionDependencies,
    settings: ExtractionSettings,
    config_sha256: str,
    code: CodeProvenance,
    script_path: str,
    overwrite: bool = False,
    session_mapping: SessionMappingContext | None = None,
) -> ExtractionSummary:
    """Extract deterministic per-clip shards, resuming only verified complete outputs."""
    if overwrite:
        raise ValueError(
            "research feature artifacts are immutable; use a new empty output directory"
        )
    clip_ids = [clip.clip_id for clip in clips]
    if len(clip_ids) != len(set(clip_ids)):
        raise ValueError("feature extraction clip IDs must be unique")
    if session_mapping is not None:
        session_mapping.verify_current_files()
        expected_sources = session_mapping.lookup
        for clip in clips:
            if expected_sources.get(clip.session_id) != clip.source_session_id:
                raise ValueError("feature input disagrees with the approved session mapping")
    set_deterministic_runtime(settings.seed)
    records: list[dict[str, Any]] = []
    succeeded = resumed = failed = 0
    with AnchoredDirectory.open(
        output_dir, create=True, directory_mode=0o700
    ) as output, output.exclusive_lock():
        for clip in clips:
            stem = safe_clip_stem(clip.clip_id)
            shard_filename = f"{stem}.npz"
            sidecar_filename = f"{stem}.json"
            failure_filename = f"{stem}.failure.json"
            present = set(output.list_names())
            has_shard = shard_filename in present
            has_sidecar = sidecar_filename in present
            has_failure = failure_filename in present
            if has_failure and (has_shard or has_sidecar):
                raise _FeatureArtifactStateError(
                    f"success and failure artifacts coexist for {clip.clip_id}"
                )
            recovering_orphan_shard = has_shard and not has_sidecar and not has_failure
            if has_shard != has_sidecar and not recovering_orphan_shard:
                raise _FeatureArtifactStateError(
                    f"incomplete immutable feature artifact for {clip.clip_id}"
                )
            video_checksum: str | None = None
            audio_checksum: str | None = None
            transcript_sha256: str | None = None
            expected: str | None = None
            try:
                with _open_anchored_media_reference(
                    media_root, clip.video_reference
                ) as video_source, _open_anchored_media_reference(
                    media_root, clip.audio_reference
                ) as audio_source:
                    video_checksum = video_source.sha256
                    audio_checksum = audio_source.sha256
                    _portable_relative_path(
                        clip.transcript.reference,
                        code="transcript_reference_unsafe",
                        label="transcript reference",
                    )
                    try:
                        transcript_text = dependencies.transcript(clip)
                    except ExtractionError:
                        raise
                    except Exception as error:
                        raise ExtractionError("transcript_load_failed", str(error)) from error
                    if not isinstance(transcript_text, str) or not transcript_text.strip():
                        raise ExtractionError(
                            "transcript_missing", "corpus-supplied transcript is missing"
                        )
                    transcript_sha256 = hashlib.sha256(
                        transcript_text.encode("utf-8")
                    ).hexdigest()
                    expected = _expected_fingerprint(
                        clip,
                        settings=settings,
                        config_sha256=config_sha256,
                        code=code,
                        video_sha256=video_checksum,
                        audio_sha256=audio_checksum,
                        transcript_sha256=transcript_sha256,
                        dependencies=dependencies,
                        session_mapping=session_mapping,
                    )
                    base_sidecar = _base_sidecar(
                        clip,
                        settings=settings,
                        config_sha256=config_sha256,
                        code=code,
                        script_path=script_path,
                        video_sha256=video_checksum,
                        audio_sha256=audio_checksum,
                        transcript_sha256=transcript_sha256,
                        dependencies=dependencies,
                        extraction_fingerprint=expected,
                        session_mapping=session_mapping,
                    )
                    if has_sidecar:
                        verified = _verified_success_artifacts(
                            output,
                            shard_filename=shard_filename,
                            sidecar_filename=sidecar_filename,
                            expected_base=base_sidecar,
                            settings=settings,
                        )
                        current_crop_fingerprint = (
                            dependencies.face_crop_fingerprint_probe()
                        )
                        current_transcript_sha256 = hashlib.sha256(
                            dependencies.transcript(clip).encode("utf-8")
                        ).hexdigest()
                        video_source.verify_current_file()
                        audio_source.verify_current_file()
                        if (
                            current_crop_fingerprint
                            != dependencies.face_crop_fingerprint
                            or current_transcript_sha256 != transcript_sha256
                        ):
                            raise _FeatureArtifactStateError(
                                f"input changed while resuming {clip.clip_id}"
                            )
                        if session_mapping is not None:
                            session_mapping.verify_current_files()
                        metadata_file, shard_file, payload = verified
                        resumed += 1
                        records.append(
                            _success_record(
                                clip,
                                metadata_file=metadata_file,
                                shard_file=shard_file,
                                payload=payload,
                                visit_result="resumed_success",
                            )
                        )
                        continue
                    arrays, modality = extract_clip_arrays(
                        clip,
                        video_source,
                        audio_source,
                        dependencies,
                        settings,
                        transcript_text=transcript_text,
                    )
                    video_source.verify_current_file()
                    audio_source.verify_current_file()
                    if session_mapping is not None:
                        session_mapping.verify_current_files()
                    shard_bytes, sidecar_bytes, sidecar = _success_artifact_bytes(
                        shard_filename,
                        arrays,
                        {**base_sidecar, "modality": modality},
                    )
                if has_failure:
                    raise _FeatureArtifactStateError(
                        f"prior failure conflicts with new success for {clip.clip_id}"
                    )
                shard_file: AnchoredFile | None = None
                try:
                    shard_file = output.publish_bytes(
                        shard_filename, shard_bytes, mode=0o600
                    )
                    metadata_file = output.publish_bytes(
                        sidecar_filename, sidecar_bytes, mode=0o600
                    )
                except (FileExistsError, OSError, RuntimeError, ValueError) as collision:
                    raise _FeatureArtifactStateError(
                        f"immutable success publication collided for {clip.clip_id}"
                    ) from collision
                succeeded += 1
                records.append(
                    _success_record(
                        clip,
                        metadata_file=metadata_file,
                        shard_file=shard_file,
                        payload=sidecar,
                        visit_result="generated_success",
                    )
                )
            except _FeatureArtifactStateError:
                raise
            except Exception as error:
                if has_sidecar:
                    raise _FeatureArtifactStateError(
                        f"cannot validate cached success for {clip.clip_id}"
                    ) from error
                if recovering_orphan_shard:
                    raise _FeatureArtifactStateError(
                        f"cannot reproduce orphan shard for {clip.clip_id}"
                    ) from error
                code_value = (
                    error.code if isinstance(error, ExtractionError) else "unexpected_error"
                )
                failure = _failure_sidecar(
                    clip,
                    error=error,
                    code_value=code_value,
                    settings=settings,
                    config_sha256=config_sha256,
                    code=code,
                    script_path=script_path,
                    dependencies=dependencies,
                    session_mapping=session_mapping,
                    video_sha256=video_checksum,
                    audio_sha256=audio_checksum,
                    transcript_sha256=transcript_sha256,
                    extraction_fingerprint=expected,
                )
                unsigned_failure = dict(failure)
                failure["failure_payload_sha256"] = hashlib.sha256(
                    canonical_json_bytes(unsigned_failure)
                ).hexdigest()
                failure_bytes = canonical_json_bytes(failure)
                try:
                    metadata_file = output.publish_bytes(
                        failure_filename, failure_bytes, mode=0o600
                    )
                except FileExistsError as collision:
                    raise _FeatureArtifactStateError(
                        f"prior failure differs for {clip.clip_id}"
                    ) from collision
                failed += 1
                records.append(
                    _failure_record(
                        clip,
                        metadata_file=metadata_file,
                        payload=failure,
                        visit_result=(
                            "reproduced_failure" if has_failure else "generated_failure"
                        ),
                    )
                )
        output.verify_path_identity()
    if session_mapping is not None:
        session_mapping.verify_current_files()
    return ExtractionSummary(
        succeeded=succeeded, resumed=resumed, failed=failed, records=tuple(records)
    )


def decode_audio_ffmpeg(
    media_path: Path | AnchoredOpenFile, sample_rate: int
) -> np.ndarray:
    """Decode a media file to deterministic mono float32 PCM with FFmpeg."""
    with _subprocess_media_reference(media_path) as media:
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-i",
            media.path,
            "-map_metadata",
            "-1",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "pipe:1",
        ]
        try:
            result = subprocess.run(
                command, check=True, capture_output=True, pass_fds=media.pass_fds
            )
        except (OSError, subprocess.CalledProcessError) as error:
            detail = (
                getattr(error, "stderr", b"")
                .decode("utf-8", errors="replace")
                .strip()
            )
            raise ExtractionError("audio_decode_failed", detail or str(error)) from error
    waveform = np.frombuffer(result.stdout, dtype="<f4").copy()
    if waveform.size == 0:
        raise ExtractionError("audio_missing", "media contains no decodable audio")
    return waveform


def probe_audio_stream_timing(
    media_path: Path | AnchoredOpenFile,
) -> AudioStreamTiming:
    """Read the first audio stream's exact timestamp fields with FFprobe."""
    _ffmpeg_runtime_metadata()
    with _subprocess_media_reference(media_path) as media:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,start_pts,duration_ts,time_base",
            "-of",
            "json",
            media.path,
        ]
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                pass_fds=media.pass_fds,
            )
            payload = json.loads(result.stdout)
            streams = payload["streams"]
            if (
                not isinstance(streams, list)
                or len(streams) != 1
                or not isinstance(streams[0], dict)
            ):
                raise ValueError("expected one selected audio stream")
            stream = streams[0]
            numerator_text, denominator_text = str(stream["time_base"]).split(
                "/", maxsplit=1
            )
            timing = AudioStreamTiming(
                codec_name=str(stream["codec_name"]),
                sample_rate=int(stream["sample_rate"]),
                start_pts=(
                    int(stream["start_pts"]) if "start_pts" in stream else None
                ),
                duration_ts=int(stream["duration_ts"]),
                time_base_numerator=int(numerator_text),
                time_base_denominator=int(denominator_text),
            )
        except (
            OSError,
            subprocess.CalledProcessError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise ExtractionError(
                "audio_source_timing_unavailable",
                "cannot verify the source video's embedded audio timing",
            ) from error
    if (
        not timing.codec_name
        or timing.sample_rate <= 0
        or timing.duration_ts <= 0
        or timing.time_base_numerator <= 0
        or timing.time_base_denominator <= 0
    ):
        raise ExtractionError(
            "audio_source_timing_unavailable",
            "source embedded-audio timing is invalid",
        )
    return timing


def probe_video_duration_seconds(media_path: Path | AnchoredOpenFile) -> float:
    """Return the source-container duration using the provenance-recorded FFprobe runtime."""
    _ffmpeg_runtime_metadata()
    with _subprocess_media_reference(media_path) as media:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            media.path,
        ]
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                pass_fds=media.pass_fds,
            )
            duration = float(result.stdout.strip())
        except (OSError, subprocess.CalledProcessError, ValueError) as error:
            raise ExtractionError(
                "video_duration_probe_failed", "cannot verify source video duration"
            ) from error
    if not math.isfinite(duration) or duration <= 0:
        raise ExtractionError("video_duration_probe_failed", "source video duration is invalid")
    return duration


class ManifestFaceCropProvider:
    """Load preselected principal-face images from an explicit JSON manifest."""

    def __init__(
        self, manifest_path: Path, *, expected_manifest_sha256: str | None = None
    ) -> None:
        supplied = manifest_path.expanduser()
        self._manifest_path = supplied if supplied.is_absolute() else Path.cwd() / supplied
        self._root = self._manifest_path.parent
        manifest_file = self._read_manifest_file(expected_manifest_sha256)
        try:
            payload = json.loads(
                manifest_file.content,
                object_pairs_hook=_unique_json_object,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON number {value}")
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError("face-crop manifest is not valid duplicate-safe JSON") from error
        if not isinstance(payload, dict):
            raise ValueError("face-crop manifest must contain one JSON object")
        if (
            expected_manifest_sha256 is not None
            and canonical_json_bytes(payload) != manifest_file.content
        ):
            raise ValueError("face-crop manifest does not use canonical JSON serialization")
        validated = load_face_crop_manifest(self._manifest_path)
        if expected_manifest_sha256 is not None and validated != payload:
            raise ValueError("face-crop manifest changed during validation")
        if expected_manifest_sha256 is None:
            payload = validated
        self._manifest_file_sha256 = manifest_file.sha256
        self._clips = payload["clips"]
        self._generator = payload["generator"]
        self._selector = payload["selector"]
        self._schema_version = payload["schema_version"]
        self._verify_manifest_file()

    def for_clip(self, clip_id: str) -> FaceCropProvider:
        """Bind this manifest to one clip for the dependency protocol."""

        def load(
            _media_path: Path | AnchoredOpenFile, *, fps: int
        ) -> FaceCropBatch:
            clip, rows = self._validated_rows(clip_id, fps=fps)
            if file_sha256(_media_path) != clip["source_video_sha256"]:
                raise ExtractionError(
                    "face_crop_invalid", "face crops are bound to a different source video"
                )
            images: list[Any] = []
            timestamps: list[float] = []
            files: list[dict[str, str]] = []
            from PIL import Image

            for row in rows:
                crop = self._read_crop(row)
                with Image.open(io.BytesIO(crop.content)) as image:
                    images.append(image.convert("RGB").copy())
                timestamps.append(float(row["timestamp_seconds"]))
                files.append({"path": str(row["path"]), "sha256": crop.sha256})
            fingerprint = self._fingerprint(clip_id, clip, rows, files, fps=fps)
            self._verify_manifest_file()
            return FaceCropBatch(
                images=tuple(images),
                timestamps_seconds=np.asarray(timestamps, dtype=np.float64),
                provenance={
                    "generator": self._generator,
                    **self._selector,
                    "schema_version": self._schema_version,
                    "source_video_sha256": clip["source_video_sha256"],
                    "full_duration_seconds": clip["duration_seconds"],
                    "full_one_hz_grid_count": clip["full_one_hz_grid_count"],
                    "active_start_grid_index": clip["active_start_grid_index"],
                    "active_end_grid_index_exclusive": clip["active_end_grid_index_exclusive"],
                    "active_close_reason": clip["active_close_reason"],
                    "frame_rate_numerator": clip["frame_rate_numerator"],
                    "frame_rate_denominator": clip["frame_rate_denominator"],
                    "valid_candidate_grid_indices": clip["valid_candidate_grid_indices"],
                    "selected_grid_indices": clip["selected_grid_indices"],
                    "requested_fps": fps,
                    "manifest_clip_id": clip_id,
                    "frames": [
                        {**row, "crop_sha256": file["sha256"]}
                        for row, file in zip(rows, files, strict=True)
                    ],
                    "input_fingerprint": fingerprint,
                },
            )

        return load

    def fingerprint_for_clip(self, clip_id: str, *, fps: int) -> str:
        """Hash selector provenance, crop metadata and every selected crop file."""
        clip, rows = self._validated_rows(clip_id, fps=fps)
        files = [
            {"path": str(row["path"]), "sha256": self._read_crop(row).sha256}
            for row in rows
        ]
        fingerprint = self._fingerprint(clip_id, clip, rows, files, fps=fps)
        self._verify_manifest_file()
        return fingerprint

    def _fingerprint(
        self,
        clip_id: str,
        clip: Mapping[str, Any],
        rows: Sequence[Mapping[str, Any]],
        files: Sequence[Mapping[str, str]],
        *,
        fps: int,
    ) -> str:
        payload = {
            "generator": self._generator,
            "selector": self._selector,
            "schema_version": self._schema_version,
            "manifest_file_sha256": self._manifest_file_sha256,
            "clip_id": clip_id,
            "clip": {key: value for key, value in clip.items() if key != "frames"},
            "rows": list(rows),
            "files": list(files),
            "requested_fps": fps,
        }
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    def _validated_rows(
        self, clip_id: str, *, fps: int
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        clip = self._clips.get(clip_id)
        if not isinstance(clip, dict):
            raise ExtractionError("face_crop_missing", "no principal-face crops in manifest")
        if clip["status"] == "failure":
            failure = clip["failure"]
            raise ExtractionError(str(failure["code"]), str(failure["message"]))
        rows = clip["frames"]
        if not rows:
            raise ExtractionError("face_crop_missing", "no principal-face crops in manifest")
        if fps <= 0:
            raise ExtractionError("face_crop_invalid", "face-crop FPS must be positive")
        if fps != 1:
            raise ExtractionError("face_crop_invalid", "MIT face-crop grid must be exactly 1 Hz")
        if [row["grid_index"] for row in rows] != clip["selected_grid_indices"]:
            raise ExtractionError(
                "face_crop_missing",
                "every retained active-interval timestamp needs a valid candidate crop",
            )
        return clip, rows

    def _read_manifest_file(self, expected_sha256: str | None) -> AnchoredFile:
        try:
            with AnchoredDirectory.open(self._root, directory_mode=None) as root:
                with root.open_file(
                    self._manifest_path.name,
                    expected_mode=None,
                    expected_sha256=expected_sha256,
                ) as source:
                    return AnchoredFile(
                        content=source.read_bytes(),
                        sha256=source.sha256,
                        size=source.size,
                        mode=source.mode,
                        device=source.device,
                        inode=source.inode,
                    )
        except (OSError, RuntimeError, ValueError) as error:
            raise ValueError("face-crop manifest path is unavailable or unsafe") from error

    def _verify_manifest_file(self) -> None:
        current = self._read_manifest_file(self._manifest_file_sha256)
        if current.sha256 != self._manifest_file_sha256:
            raise ExtractionError(
                "face_crop_manifest_mutated",
                "face-crop manifest changed after its mapping binding was verified",
            )

    def _read_crop(self, row: Mapping[str, Any]) -> AnchoredFile:
        relative = _portable_relative_path(
            str(row["path"]), code="face_crop_missing", label="face-crop path"
        )
        expected = row.get("crop_sha256")
        if not isinstance(expected, str):
            raise ExtractionError("face_crop_invalid", "face-crop checksum is unavailable")
        try:
            with AnchoredDirectory.open(
                self._root / relative.parent, directory_mode=None
            ) as parent:
                crop = parent.read_bytes(relative.name, expected_mode=None)
                if crop.sha256 != expected:
                    raise ExtractionError(
                        "face_crop_invalid", "face-crop checksum does not match the manifest"
                    )
                return crop
        except ExtractionError:
            raise
        except (OSError, RuntimeError, ValueError) as error:
            raise ExtractionError(
                "face_crop_missing", "face-crop path is unavailable or unsafe"
            ) from error
        finally:
            self._verify_manifest_file()


class SuppliedTranscriptProvider:
    """Use inline supplied text, otherwise resolve its supplied reference safely."""

    def __init__(self, transcript_root: Path, *, require_inline: bool = False) -> None:
        self._root = transcript_root.resolve()
        self._require_inline = require_inline

    def __call__(self, clip: MITInterviewSession | MITInterviewFeatureSession) -> str:
        if clip.transcript.content is not None and clip.transcript.content.strip():
            return clip.transcript.content
        if self._require_inline:
            raise ExtractionError(
                "transcript_missing",
                "research extraction requires loader-validated inline transcript bytes",
            )
        reference = _portable_relative_path(
            clip.transcript.reference,
            code="transcript_reference_unsafe",
            label="transcript reference",
        )
        path = (self._root / reference).resolve()
        if not path.is_relative_to(self._root) or not path.is_file():
            raise ExtractionError(
                "transcript_missing", "supplied transcript reference is unavailable or unsafe"
            )
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise ExtractionError("transcript_load_failed", str(error)) from error
        if not text.strip() or "\0" in text:
            raise ExtractionError(
                "transcript_missing", "supplied transcript file is empty or invalid"
            )
        return text


def _base_sidecar(
    clip: MITInterviewSession | MITInterviewFeatureSession,
    *,
    settings: ExtractionSettings,
    config_sha256: str,
    code: CodeProvenance,
    script_path: str,
    video_sha256: str,
    audio_sha256: str,
    transcript_sha256: str,
    dependencies: ExtractionDependencies,
    extraction_fingerprint: str,
    session_mapping: SessionMappingContext | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "state": "success",
        "clip_id": clip.clip_id,
        "session_id": clip.session_id,
        "source_session_id": getattr(clip, "source_session_id", clip.session_id),
        "fold": clip.fold,
        "participant_id": clip.participant_id,
        "condition": clip.condition,
        "input": {
            "video_reference": clip.video_reference,
            "video_sha256": video_sha256,
            "audio_reference": clip.audio_reference,
            "audio_sha256": audio_sha256,
            "source_dialogue_sha256": clip.transcript.source_dialogue_sha256,
            "candidate_transcript_sha256": transcript_sha256,
            "transcript_parser_revision": TRANSCRIPT_PARSER_REVISION,
            "transcript_parser_audit": {
                "explicit_candidate_segments": clip.transcript.candidate_segment_count,
                "retained_candidate_continuations": (
                    clip.transcript.retained_candidate_continuation_count
                ),
                "retained_candidate_continuation_characters": (
                    clip.transcript.retained_candidate_continuation_characters
                ),
                "excluded_colonless_fragments": clip.transcript.excluded_colonless_fragment_count,
                "excluded_colonless_characters": clip.transcript.excluded_colonless_characters,
                "excluded_unclassified_markers": clip.transcript.excluded_unclassified_marker_count,
                "excluded_interviewer_segments": clip.transcript.excluded_interviewer_segment_count,
            },
            "face_crop_fingerprint": dependencies.face_crop_fingerprint,
        },
        "script": script_path,
        "seed": settings.seed,
        "source_date_epoch": settings.source_date_epoch,
        "config_sha256": config_sha256,
        "extraction_fingerprint": extraction_fingerprint,
        "session_mapping": session_mapping.provenance if session_mapping is not None else None,
        "runtime": _runtime_metadata(dependencies),
        "code": {
            "commit_sha": code.commit_sha,
            "source_tree_sha256": code.source_tree_sha256,
            "dirty": code.dirty,
        },
        "encoders": _encoder_metadata(dependencies),
        "preprocessing": _preprocessing_metadata(settings),
    }


def current_feature_run_context(
    *,
    settings: ExtractionSettings,
    config_sha256: str,
    code: CodeProvenance,
    script_path: str,
    dependencies: ExtractionDependencies,
    session_mapping: SessionMappingContext,
) -> dict[str, Any]:
    """Recompute the common closure context at the final publication boundary."""
    session_mapping.verify_current_files()
    return {
        "feature_schema_version": SCHEMA_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "script": script_path,
        "seed": settings.seed,
        "source_date_epoch": settings.source_date_epoch,
        "config_sha256": config_sha256,
        "code": {
            "commit_sha": code.commit_sha,
            "source_tree_sha256": code.source_tree_sha256,
            "dirty": code.dirty,
        },
        "session_mapping": session_mapping.provenance,
        "runtime": _runtime_metadata(dependencies),
        "encoders": _encoder_metadata(dependencies),
        "preprocessing": _preprocessing_metadata(settings),
    }


def verify_current_feature_inputs(
    clip: MITInterviewSession | MITInterviewFeatureSession,
    metadata: Mapping[str, Any],
    *,
    media_root: Path,
    transcript: TranscriptProvider,
    face_crop_fingerprint_probe: Callable[[], str],
    session_mapping: SessionMappingContext,
) -> None:
    """Recompute one record's outcome-free input binding through stable handles."""
    session_mapping.verify_current_files()
    try:
        with _open_anchored_media_reference(
            media_root, clip.video_reference
        ) as video_source, _open_anchored_media_reference(
            media_root, clip.audio_reference
        ) as audio_source:
            transcript_text = transcript(clip)
            if not isinstance(transcript_text, str) or not transcript_text.strip():
                raise ExtractionError(
                    "transcript_missing", "corpus-supplied transcript is missing"
                )
            current = {
                "video_reference": clip.video_reference,
                "video_sha256": video_source.sha256,
                "audio_reference": clip.audio_reference,
                "audio_sha256": audio_source.sha256,
                "source_dialogue_sha256": clip.transcript.source_dialogue_sha256,
                "candidate_transcript_sha256": hashlib.sha256(
                    transcript_text.encode("utf-8")
                ).hexdigest(),
                "transcript_parser_revision": TRANSCRIPT_PARSER_REVISION,
                "transcript_parser_audit": {
                    "explicit_candidate_segments": clip.transcript.candidate_segment_count,
                    "retained_candidate_continuations": (
                        clip.transcript.retained_candidate_continuation_count
                    ),
                    "retained_candidate_continuation_characters": (
                        clip.transcript.retained_candidate_continuation_characters
                    ),
                    "excluded_colonless_fragments": (
                        clip.transcript.excluded_colonless_fragment_count
                    ),
                    "excluded_colonless_characters": (
                        clip.transcript.excluded_colonless_characters
                    ),
                    "excluded_unclassified_markers": (
                        clip.transcript.excluded_unclassified_marker_count
                    ),
                    "excluded_interviewer_segments": (
                        clip.transcript.excluded_interviewer_segment_count
                    ),
                },
                "face_crop_fingerprint": face_crop_fingerprint_probe(),
            }
            video_source.verify_current_file()
            audio_source.verify_current_file()
    except (OSError, RuntimeError, ValueError) as error:
        raise ExtractionError(
            "input_revalidation_failed",
            "cannot revalidate immutable feature inputs at closure",
        ) from error
    if metadata.get("input") != current:
        raise ExtractionError(
            "input_revalidation_failed",
            "feature input binding changed before closure",
        )
    session_mapping.verify_current_files()


def _preprocessing_metadata(settings: ExtractionSettings) -> dict[str, Any]:
    return {
        "video": {
            "input": "preselected_principal_face_crops",
            "sampling_fps": settings.video_fps,
            "maximum_samples": settings.video_max_samples,
            "source_index_set": "candidate_valid_1hz_within_active_interview_interval",
            "selection": "endpoint_rank_over_candidate_valid_indices_when_capped",
            "processor": (
                "pinned_artifact_rgb_bicubic_shortest_256_center_crop_224_imagenet_norm"
            ),
        },
        "audio": {
            "mono": True,
            "sample_rate": settings.audio_sample_rate,
            "waveform_normalization": False,
            "context": "active_interview_audio_not_candidate_voice",
            "boundary_source": "candidate_face_active_interval",
            "window_samples": settings.audio_window_samples,
            "maximum_windows": settings.audio_max_windows,
            "selection": "floor(j*(N-1)/(maximum-1))_when_capped",
            "pooling": (
                "bounded_window_final_hidden_center_time_half_open_seconds_mean_then_max"
            ),
            "frame_centre_formula": "(j*320+199.5)/16000",
            "partial_last_second": "retain_if_nonempty",
            "end_alignment": "allow_end_of_stream_shortfall_below_one_source_audio_tick",
        },
        "text": {
            "source": "corpus_supplied_only",
            "chunking": "tokenizer_ids_nonoverlapping_no_specials_then_add_specials",
            "max_sequence_tokens_including_specials": settings.text_max_sequence_tokens,
            "pooling": "attention_mask_mean_then_l2_normalize_per_chunk",
        },
    }


def _failure_sidecar(
    clip: MITInterviewSession | MITInterviewFeatureSession,
    *,
    error: Exception,
    code_value: str,
    settings: ExtractionSettings,
    config_sha256: str,
    code: CodeProvenance,
    script_path: str,
    dependencies: ExtractionDependencies,
    session_mapping: SessionMappingContext | None,
    video_sha256: str | None,
    audio_sha256: str | None,
    transcript_sha256: str | None,
    extraction_fingerprint: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "state": "failure",
        **_record_identity(clip),
        "failure": {
            "code": code_value,
            "message": _safe_failure_message(error, code_value),
            "type": type(error).__name__,
            "modality": _failure_modality(code_value),
        },
        "input": {
            "video_reference": _safe_reference_for_sidecar(clip.video_reference),
            "video_sha256": video_sha256,
            "audio_reference": _safe_reference_for_sidecar(clip.audio_reference),
            "audio_sha256": audio_sha256,
            "source_dialogue_sha256": clip.transcript.source_dialogue_sha256,
            "candidate_transcript_sha256": transcript_sha256,
            "transcript_parser_revision": TRANSCRIPT_PARSER_REVISION,
            "transcript_parser_audit": {
                "explicit_candidate_segments": clip.transcript.candidate_segment_count,
                "retained_candidate_continuations": (
                    clip.transcript.retained_candidate_continuation_count
                ),
                "retained_candidate_continuation_characters": (
                    clip.transcript.retained_candidate_continuation_characters
                ),
                "excluded_colonless_fragments": (
                    clip.transcript.excluded_colonless_fragment_count
                ),
                "excluded_colonless_characters": clip.transcript.excluded_colonless_characters,
                "excluded_unclassified_markers": (
                    clip.transcript.excluded_unclassified_marker_count
                ),
                "excluded_interviewer_segments": (
                    clip.transcript.excluded_interviewer_segment_count
                ),
            },
            "face_crop_fingerprint": dependencies.face_crop_fingerprint,
        },
        "script": script_path,
        "seed": settings.seed,
        "source_date_epoch": settings.source_date_epoch,
        "config_sha256": config_sha256,
        "extraction_fingerprint": extraction_fingerprint,
        "session_mapping": (
            session_mapping.provenance if session_mapping is not None else None
        ),
        "runtime": _runtime_metadata(dependencies),
        "code": {
            "commit_sha": code.commit_sha,
            "source_tree_sha256": code.source_tree_sha256,
            "dirty": code.dirty,
        },
        "encoders": _encoder_metadata(dependencies),
        "preprocessing": _preprocessing_metadata(settings),
    }


def _success_artifact_bytes(
    shard_filename: str,
    arrays: Mapping[str, np.ndarray],
    sidecar: Mapping[str, Any],
) -> tuple[bytes, bytes, dict[str, Any]]:
    shard_bytes = _npz_bytes(arrays)
    array_metadata = {
        key: {
            "shape": list(np.asarray(arrays[key]).shape),
            "dtype": str(np.asarray(arrays[key]).dtype),
        }
        for key in SHARD_KEYS
    }
    complete = {
        **sidecar,
        "arrays": array_metadata,
        "shard": {
            "filename": shard_filename,
            "sha256": hashlib.sha256(shard_bytes).hexdigest(),
            "keys": list(SHARD_KEYS),
        },
    }
    complete["sidecar_payload_sha256"] = hashlib.sha256(canonical_json_bytes(complete)).hexdigest()
    sidecar_bytes = canonical_json_bytes(complete)
    return shard_bytes, sidecar_bytes, complete


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Create a byte-stable uncompressed NPZ with fixed ZIP metadata."""
    import zipfile

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for key in SHARD_KEYS:
            value = io.BytesIO()
            np.lib.format.write_array(value, np.asarray(arrays[key]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, value.getvalue())
    return output.getvalue()


def _verified_success_artifacts(
    output: AnchoredDirectory,
    *,
    shard_filename: str,
    sidecar_filename: str,
    expected_base: Mapping[str, Any],
    settings: ExtractionSettings,
) -> tuple[AnchoredFile, AnchoredFile, dict[str, Any]]:
    try:
        metadata_file = output.read_bytes(sidecar_filename, expected_mode=0o600)
        shard_file = output.read_bytes(shard_filename, expected_mode=0o600)
        payload = json.loads(
            metadata_file.content,
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {value}")
            ),
        )
        if not isinstance(payload, dict) or canonical_json_bytes(payload) != metadata_file.content:
            raise ValueError("success sidecar is not canonical JSON")
        expected_sidecar_sha256 = payload.get("sidecar_payload_sha256")
        unsigned_payload = dict(payload)
        unsigned_payload.pop("sidecar_payload_sha256", None)
        if (
            not isinstance(expected_sidecar_sha256, str)
            or hashlib.sha256(canonical_json_bytes(unsigned_payload)).hexdigest()
            != expected_sidecar_sha256
        ):
            raise ValueError("success sidecar payload checksum failed")
        exact_fields = set(expected_base) | {
            "modality",
            "arrays",
            "shard",
            "sidecar_payload_sha256",
        }
        if set(payload) != exact_fields or any(
            payload.get(field) != value for field, value in expected_base.items()
        ):
            raise ValueError("cached sidecar semantic contract drifted")
        shard = payload["shard"]
        if (
            not isinstance(shard, dict)
            or set(shard) != {"filename", "sha256", "keys"}
            or shard["filename"] != shard_filename
            or shard["keys"] != list(SHARD_KEYS)
            or shard["sha256"] != shard_file.sha256
        ):
            raise ValueError("cached shard binding drifted")
        with np.load(io.BytesIO(shard_file.content), allow_pickle=False) as archive:
            if tuple(archive.files) != SHARD_KEYS:
                raise ValueError("cached shard keys drifted")
            arrays: dict[str, np.ndarray] = {}
            for key in SHARD_KEYS:
                arrays[key] = archive[key]
                metadata = payload["arrays"][key]
                if (
                    not isinstance(metadata, dict)
                    or set(metadata) != {"shape", "dtype"}
                    or
                    list(arrays[key].shape) != metadata["shape"]
                    or str(arrays[key].dtype) != metadata["dtype"]
                ):
                    raise ValueError("cached array metadata drifted")
            _validate_arrays(arrays, settings)
        if set(payload["arrays"]) != set(SHARD_KEYS):
            raise ValueError("cached array registry drifted")
        if _npz_bytes(arrays) != shard_file.content:
            raise ValueError("cached shard is not the deterministic NPZ representation")
        _validate_modality_sidecar(payload["modality"], arrays, expected_base, settings)
        output.verify_path_identity()
        return metadata_file, shard_file, payload
    except (ExtractionError, OSError, KeyError, TypeError, ValueError) as error:
        raise _FeatureArtifactStateError(
            f"cached success for {expected_base.get('clip_id')} failed full validation"
        ) from error


def _validate_modality_sidecar(
    modality: object,
    arrays: Mapping[str, np.ndarray],
    expected_base: Mapping[str, Any],
    settings: ExtractionSettings,
) -> None:
    if not isinstance(modality, dict) or set(modality) != {"video", "audio", "text"}:
        raise ValueError("modality provenance registry drifted")
    if any(
        not isinstance(modality[name], dict)
        or modality[name].get("status") != "success"
        for name in ("video", "audio", "text")
    ):
        raise ValueError("modality success state drifted")
    video = modality["video"]
    if set(video) != {"status", "crop_provider"}:
        raise ValueError("video modality provenance schema drifted")
    crop = video.get("crop_provider")
    input_payload = expected_base.get("input")
    if (
        not isinstance(crop, dict)
        or not isinstance(input_payload, dict)
        or crop.get("input_fingerprint") != input_payload.get("face_crop_fingerprint")
        or crop.get("source_video_sha256") != input_payload.get("video_sha256")
        or crop.get("selected_grid_indices")
        != np.asarray(arrays["video_timestamps_seconds"], dtype=np.int64).tolist()
    ):
        raise ValueError("video modality provenance drifted")
    runtime = expected_base.get("runtime")
    if isinstance(runtime, Mapping) and runtime.get("backend") == "huggingface":
        _validate_production_crop_provenance(crop, expected_base, arrays)
    audio = modality["audio"]
    if set(audio) != _AUDIO_MODALITY_FIELDS:
        raise ValueError("audio modality provenance schema drifted")
    seconds = np.asarray(arrays["audio_second_indices"])
    selected_windows = audio.get("selected_windows")
    window_count = audio.get("window_count_active")
    active_sample_count = audio.get("active_sample_count")
    active_start_sample = audio.get("active_start_sample")
    if (
        isinstance(active_sample_count, bool)
        or not isinstance(active_sample_count, int)
        or active_sample_count <= 0
        or isinstance(active_start_sample, bool)
        or not isinstance(active_start_sample, int)
        or active_start_sample < 0
    ):
        raise ValueError("audio active-sample provenance drifted")
    _validate_audio_timing_provenance(audio, crop, settings)
    expected_windows = [
        {
            "index": index,
            "active_start_sample": start,
            "active_end_sample_exclusive": end,
            "source_start_sample": active_start_sample + start,
            "source_end_sample_exclusive": active_start_sample + end,
        }
        for index, start, end in bounded_audio_windows(
            active_sample_count,
            window_samples=settings.audio_window_samples,
            maximum_windows=settings.audio_max_windows,
        )
    ]
    expected_window_count = (
        active_sample_count + settings.audio_window_samples - 1
    ) // settings.audio_window_samples
    if (
        audio.get("context") != "active_interview_audio_not_candidate_voice"
        or selected_windows != expected_windows
        or window_count != expected_window_count
        or audio.get("complete_active_interval_representation")
        is not (len(expected_windows) == expected_window_count)
        or not isinstance(audio.get("active_start_seconds"), int)
        or not isinstance(audio.get("active_end_seconds_exclusive"), int)
        or (seconds < audio["active_start_seconds"]).any()
        or (seconds >= audio["active_end_seconds_exclusive"]).any()
    ):
        raise ValueError("audio modality provenance drifted")
    text = modality["text"]
    reference = text.get("reference")
    if (
        set(text) != {"status", "source", "reference"}
        or text.get("source") != "corpus_supplied"
        or not isinstance(reference, str)
        or _portable_relative_path(
            reference,
            code="transcript_reference_unsafe",
            label="transcript reference",
        ).as_posix()
        != reference
    ):
        raise ValueError("text modality provenance drifted")


def _validate_production_crop_provenance(
    crop: Mapping[str, Any],
    expected_base: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> None:
    if set(crop) != _PRODUCTION_CROP_BINDING_FIELDS:
        raise ValueError("production face-crop provenance schema drifted")
    validate_face_generator(crop.get("generator"))
    validate_face_selector(
        {field: crop.get(field) for field in FACE_SELECTOR_FIELDS}
    )
    if (
        crop.get("schema_version") != FACE_MANIFEST_SCHEMA_VERSION
        or crop.get("requested_fps") != 1
        or crop.get("manifest_clip_id") != expected_base.get("source_session_id")
    ):
        raise ValueError("production face-crop manifest binding drifted")
    frames = crop.get("frames")
    timestamps = np.asarray(arrays["video_timestamps_seconds"])
    if (
        not isinstance(frames, list)
        or len(frames) != timestamps.size
        or any(
            not isinstance(frame, Mapping) or set(frame) != set(FACE_FRAME_FIELDS)
            for frame in frames
        )
        or [frame["grid_index"] for frame in frames]
        != np.asarray(timestamps, dtype=np.int64).tolist()
        or [float(frame["timestamp_seconds"]) for frame in frames]
        != timestamps.tolist()
    ):
        raise ValueError("production face-crop frame binding drifted")
    for frame in frames:
        _portable_relative_path(
            frame["path"], code="face_crop_missing", label="face-crop path"
        )
        digest = frame.get("crop_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("production face-crop checksum is invalid")


def _validate_audio_timing_provenance(
    audio: Mapping[str, Any], crop: Mapping[str, Any], settings: ExtractionSettings
) -> None:
    source = audio.get("source_embedded_audio_timing")
    supplied = audio.get("supplied_audio_timing")
    if (
        not isinstance(source, Mapping)
        or set(source) != _AUDIO_TIMING_FIELDS
        or not isinstance(supplied, Mapping)
        or set(supplied) != _AUDIO_TIMING_FIELDS
    ):
        raise ValueError("audio stream timing schema drifted")
    numeric_fields = ("sample_rate", "duration_ts", "time_base_numerator", "time_base_denominator")
    if any(
        isinstance(timing.get(field), bool)
        or not isinstance(timing.get(field), int)
        or timing[field] <= 0
        for timing in (source, supplied)
        for field in numeric_fields
    ) or any(
        not isinstance(timing.get("codec_name"), str) or not timing["codec_name"]
        for timing in (source, supplied)
    ):
        raise ValueError("audio stream timing values drifted")
    full_sample_count = audio.get("full_sample_count")
    active_end_sample = audio.get("active_end_sample_exclusive")
    requested_end_sample = audio.get("requested_active_end_sample_exclusive")
    active_start_seconds = audio.get("active_start_seconds")
    active_end_seconds = audio.get("active_end_seconds_exclusive")
    active_sample_count = audio.get("active_sample_count")
    shortfall = audio.get("end_alignment_shortfall_samples")
    integer_values = (
        full_sample_count,
        active_end_sample,
        requested_end_sample,
        active_start_seconds,
        active_end_seconds,
        active_sample_count,
        shortfall,
        audio.get("end_alignment_tolerance_samples"),
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_values):
        raise ValueError("audio interval timing values drifted")
    assert isinstance(full_sample_count, int)
    assert isinstance(active_end_sample, int)
    assert isinstance(requested_end_sample, int)
    assert isinstance(active_start_seconds, int)
    assert isinstance(active_end_seconds, int)
    assert isinstance(active_sample_count, int)
    assert isinstance(shortfall, int)
    expected_start_sample = active_start_seconds * settings.audio_sample_rate
    expected_requested_end = active_end_seconds * settings.audio_sample_rate
    expected_end_sample = min(expected_requested_end, full_sample_count)
    expected_shortfall = max(expected_requested_end - full_sample_count, 0)
    expected_tolerance = math.ceil(
        settings.audio_sample_rate
        * source["time_base_numerator"]
        / source["time_base_denominator"]
    )
    supplied_sample_numerator = (
        supplied["duration_ts"]
        * supplied["time_base_numerator"]
        * settings.audio_sample_rate
    )
    if (
        source.get("start_pts") != 0
        or supplied.get("start_pts") not in {None, 0}
        or (source["time_base_numerator"], source["time_base_denominator"])
        != (3, 125)
        or (supplied["time_base_numerator"], supplied["time_base_denominator"])
        != (1, 48_000)
        or source["duration_ts"]
        * source["time_base_numerator"]
        * supplied["time_base_denominator"]
        != supplied["duration_ts"]
        * supplied["time_base_numerator"]
        * source["time_base_denominator"]
        or supplied_sample_numerator % supplied["time_base_denominator"] != 0
        or supplied_sample_numerator // supplied["time_base_denominator"]
        != full_sample_count
        or full_sample_count <= 0
        or audio.get("full_duration_seconds")
        != full_sample_count / settings.audio_sample_rate
        or audio.get("supplied_audio_matches_embedded_duration") is not True
        or audio.get("decoded_audio_matches_supplied_duration") is not True
        or audio.get("active_start_sample") != expected_start_sample
        or requested_end_sample != expected_requested_end
        or active_end_sample != expected_end_sample
        or active_sample_count != expected_end_sample - expected_start_sample
        or audio.get("active_duration_seconds")
        != active_end_seconds - active_start_seconds
        or audio.get("available_active_duration_seconds")
        != active_sample_count / settings.audio_sample_rate
        or audio.get("end_alignment_policy")
        != "allow_end_of_stream_shortfall_below_one_source_audio_tick"
        or shortfall != expected_shortfall
        or audio.get("end_alignment_shortfall_seconds")
        != shortfall / settings.audio_sample_rate
        or audio.get("end_alignment_tolerance_samples") != expected_tolerance
        or crop.get("active_start_grid_index") != active_start_seconds
        or crop.get("active_end_grid_index_exclusive") != active_end_seconds
    ):
        raise ValueError("audio stream and active-interval binding drifted")


def _success_record(
    clip: MITInterviewSession | MITInterviewFeatureSession,
    *,
    metadata_file: AnchoredFile,
    shard_file: AnchoredFile,
    payload: Mapping[str, Any],
    visit_result: str,
) -> dict[str, Any]:
    return {
        **_record_identity(clip),
        "state": "success",
        "visit_result": visit_result,
        "metadata_filename": f"{safe_clip_stem(clip.clip_id)}.json",
        "metadata_file_sha256": metadata_file.sha256,
        "metadata_payload_sha256": payload["sidecar_payload_sha256"],
        "extraction_fingerprint": payload["extraction_fingerprint"],
        "shard_filename": f"{safe_clip_stem(clip.clip_id)}.npz",
        "shard_file_sha256": shard_file.sha256,
        "failure_code": None,
    }


def _failure_record(
    clip: MITInterviewSession | MITInterviewFeatureSession,
    *,
    metadata_file: AnchoredFile,
    payload: Mapping[str, Any],
    visit_result: str,
) -> dict[str, Any]:
    return {
        **_record_identity(clip),
        "state": "failure",
        "visit_result": visit_result,
        "metadata_filename": f"{safe_clip_stem(clip.clip_id)}.failure.json",
        "metadata_file_sha256": metadata_file.sha256,
        "metadata_payload_sha256": payload["failure_payload_sha256"],
        "extraction_fingerprint": payload["extraction_fingerprint"],
        "shard_filename": None,
        "shard_file_sha256": None,
        "failure_code": payload["failure"]["code"],
        "code": payload["failure"]["code"],
    }


def _record_identity(
    clip: MITInterviewSession | MITInterviewFeatureSession,
) -> dict[str, Any]:
    return {
        "clip_id": clip.clip_id,
        "session_id": clip.session_id,
        "source_session_id": getattr(clip, "source_session_id", clip.session_id),
        "participant_id": clip.participant_id,
        "condition": clip.condition,
        "fold": clip.fold,
    }


def _validate_arrays(arrays: Mapping[str, np.ndarray], settings: ExtractionSettings) -> None:
    if tuple(arrays) != SHARD_KEYS:
        raise ExtractionError("schema_invalid", "feature array keys do not match the frozen schema")
    expected = {
        "video": (None, settings.video_dim),
        "audio": (None, settings.audio_pooled_dim),
        "text": (None, settings.text_dim),
    }
    for key, (_, width) in expected.items():
        value = np.asarray(arrays[key])
        if (
            value.ndim != 2
            or value.shape[0] == 0
            or value.shape[1] != width
            or value.dtype != np.float32
            or not np.isfinite(value).all()
        ):
            raise ExtractionError("schema_invalid", f"{key} has an invalid shape or dtype")
    for modality in ("video", "audio", "text"):
        mask = np.asarray(arrays[f"{modality}_mask"])
        rows = np.asarray(arrays[modality]).shape[0]
        if mask.dtype != np.bool_ or mask.shape != (rows,) or not mask.all():
            raise ExtractionError("schema_invalid", f"{modality}_mask must mark every row valid")
    vector_contracts = {
        "video_timestamps_seconds": (np.float64, np.asarray(arrays["video"]).shape[0]),
        "audio_second_indices": (np.int64, np.asarray(arrays["audio"]).shape[0]),
        "text_token_start": (np.int64, np.asarray(arrays["text"]).shape[0]),
        "text_token_count": (np.int64, np.asarray(arrays["text"]).shape[0]),
    }
    for key, (dtype, rows) in vector_contracts.items():
        values = np.asarray(arrays[key])
        if values.dtype != dtype or values.shape != (rows,):
            raise ExtractionError("schema_invalid", f"{key} has an invalid shape or dtype")
    timestamps = np.asarray(arrays["video_timestamps_seconds"])
    seconds = np.asarray(arrays["audio_second_indices"])
    starts = np.asarray(arrays["text_token_start"])
    counts = np.asarray(arrays["text_token_count"])
    if (
        not np.isfinite(timestamps).all()
        or (timestamps < 0).any()
        or (timestamps.size > 1 and (np.diff(timestamps) <= 0).any())
        or not np.allclose(
            timestamps * settings.video_fps,
            np.rint(timestamps * settings.video_fps),
            atol=1e-9,
            rtol=0,
        )
        or (seconds < 0).any()
        or (seconds.size > 1 and (np.diff(seconds) <= 0).any())
        or starts[0] != 0
        or (starts.size > 1 and not np.array_equal(starts[1:], np.cumsum(counts[:-1])))
        or (counts <= 0).any()
        or (counts > settings.text_max_sequence_tokens - 2).any()
    ):
        raise ExtractionError("schema_invalid", "feature index arrays violate sequence contracts")
    for modality, width in (
        ("video", settings.video_dim),
        ("audio", settings.audio_pooled_dim),
        ("text", settings.text_dim),
    ):
        summary = np.asarray(arrays[f"{modality}_mean"])
        expected_mean = np.asarray(arrays[modality]).mean(axis=0, dtype=np.float32)
        if (
            summary.dtype != np.float32
            or summary.shape != (width,)
            or not np.isfinite(summary).all()
            or not np.array_equal(summary, expected_mean)
        ):
            raise ExtractionError("schema_invalid", f"{modality}_mean is not the exact valid mean")


def _finite_matrix(value: Any, out_dim: int, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != out_dim or matrix.shape[0] == 0:
        raise ExtractionError("encoder_shape_invalid", f"{name} must have shape [n,{out_dim}]")
    if not np.isfinite(matrix).all():
        raise ExtractionError("encoder_values_invalid", f"{name} contains non-finite values")
    return np.ascontiguousarray(matrix)


def _media_file_available(path: Path | AnchoredOpenFile) -> bool:
    if isinstance(path, AnchoredOpenFile):
        path.verify_current_file()
        return True
    return path.is_file()


@contextmanager
def _subprocess_media_reference(
    media_path: Path | AnchoredOpenFile,
) -> Iterator[AnchoredSubprocessFile]:
    if isinstance(media_path, AnchoredOpenFile):
        with media_path.duplicate_for_subprocess() as reference:
            yield reference
        return
    yield AnchoredSubprocessFile(path=str(media_path), pass_fds=())


@contextmanager
def _open_anchored_media_reference(
    media_root: Path, reference: str
) -> Iterator[AnchoredOpenFile]:
    relative = _portable_relative_path(
        reference,
        code="media_reference_unsafe",
        label="media reference",
    )
    parent = media_root / relative.parent
    anchor: AnchoredDirectory | None = None
    source: AnchoredOpenFile | None = None
    try:
        anchor = AnchoredDirectory.open(parent, directory_mode=None)
        source = anchor.open_file(relative.name, expected_mode=0o644)
    except (OSError, RuntimeError, ValueError) as error:
        if source is not None:
            source.close()
        if anchor is not None:
            anchor.close()
        raise ExtractionError(
            "media_reference_unsafe",
            "media reference is unavailable or its path identity changed",
        ) from error
    try:
        yield source
    finally:
        try:
            source.close()
        finally:
            anchor.close()


def _resolve_media_path(media_root: Path, reference: str) -> Path:
    root = media_root.resolve()
    candidate = _portable_relative_path(
        reference,
        code="media_reference_unsafe",
        label="media reference",
    )
    unresolved = root / candidate
    current = root
    for component in candidate.parts:
        current = current / component
        if current.is_symlink():
            raise ExtractionError("media_reference_unsafe", "media reference contains a symlink")
    path = unresolved.resolve()
    if not path.is_relative_to(root):
        raise ExtractionError(
            "media_reference_unsafe", "media reference escapes the configured root"
        )
    return path


def _expected_fingerprint(
    clip: MITInterviewSession | MITInterviewFeatureSession,
    *,
    settings: ExtractionSettings,
    config_sha256: str,
    code: CodeProvenance,
    video_sha256: str,
    audio_sha256: str,
    transcript_sha256: str,
    dependencies: ExtractionDependencies,
    session_mapping: SessionMappingContext | None,
) -> str:
    encoders = {}
    for modality, encoder in (
        ("video", dependencies.video_encoder),
        ("audio", dependencies.audio_encoder),
        ("text", dependencies.text_encoder),
    ):
        encoders[modality] = {
            "name": encoder.name,
            "revision": encoder.revision,
            "weight_file": getattr(encoder, "weight_file", None),
            "weight_sha256": getattr(encoder, "weight_sha256", None),
            "artifact_files": list(getattr(encoder, "artifact_files", ())),
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "clip_id": clip.clip_id,
        "session_id": clip.session_id,
        "source_session_id": getattr(clip, "source_session_id", clip.session_id),
        "fold": clip.fold,
        "participant_id": clip.participant_id,
        "condition": clip.condition,
        "video_reference": clip.video_reference,
        "audio_reference": clip.audio_reference,
        "transcript_reference": clip.transcript.reference,
        "video_sha256": video_sha256,
        "audio_sha256": audio_sha256,
        "source_dialogue_sha256": clip.transcript.source_dialogue_sha256,
        "candidate_transcript_sha256": transcript_sha256,
        "transcript_parser_revision": TRANSCRIPT_PARSER_REVISION,
        "transcript_parser_audit": {
            "explicit_candidate_segments": clip.transcript.candidate_segment_count,
            "retained_candidate_continuations": (
                clip.transcript.retained_candidate_continuation_count
            ),
            "retained_candidate_continuation_characters": (
                clip.transcript.retained_candidate_continuation_characters
            ),
            "excluded_colonless_fragments": clip.transcript.excluded_colonless_fragment_count,
            "excluded_colonless_characters": clip.transcript.excluded_colonless_characters,
            "excluded_unclassified_markers": clip.transcript.excluded_unclassified_marker_count,
            "excluded_interviewer_segments": clip.transcript.excluded_interviewer_segment_count,
        },
        "face_crop_fingerprint": dependencies.face_crop_fingerprint,
        "session_mapping": session_mapping.provenance if session_mapping is not None else None,
        "config_sha256": config_sha256,
        "code_commit_sha": code.commit_sha,
        "code_source_tree_sha256": code.source_tree_sha256,
        "settings": {name: getattr(settings, name) for name in settings.__dataclass_fields__},
        "encoders": encoders,
        "backend": dependencies.backend,
        "device": dependencies.device,
        "dtype": dependencies.dtype,
        "libraries": {
            package: _package_version(package)
            for package in (
                "numpy",
                "torch",
                "torchvision",
                "transformers",
                "sentence-transformers",
                "huggingface-hub",
                "Pillow",
            )
        },
        "external_media_runtime": _ffmpeg_runtime_metadata(),
        "runtime_contract": _runtime_metadata(dependencies),
        "preprocessing_contract": {
            "video": "selected_face_rgb_bicubic_256_center_224_imagenet_cls0",
            "audio": (
                "active_interval_one_source_audio_tick_eos_tolerance_"
                "bounded_window_center_time_half_open_seconds_mean_then_max"
            ),
            "text": "token_ids_382_content_mean_pool_l2",
        },
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _encoder_metadata(dependencies: ExtractionDependencies) -> dict[str, Any]:
    metadata = {}
    for modality, encoder in (
        ("video", dependencies.video_encoder),
        ("audio", dependencies.audio_encoder),
        ("text", dependencies.text_encoder),
    ):
        metadata[modality] = {
            "name": encoder.name,
            "revision": encoder.revision,
            "output_dim": encoder.out_dim,
            "weight_file": getattr(encoder, "weight_file", None),
            "weight_sha256": getattr(encoder, "weight_sha256", None),
            "weight_size_bytes": getattr(encoder, "weight_size_bytes", None),
            "cache_snapshot_reference": getattr(encoder, "cache_snapshot_reference", None),
            "processor_or_tokenizer_revision": encoder.revision,
            "inference_batch_size": getattr(encoder, "inference_batch_size", 1),
            "verified_at_utc": getattr(encoder, "verified_at_utc", None),
            "artifact_files": list(getattr(encoder, "artifact_files", ())),
            "license_id": getattr(encoder, "license_id", None),
            "license_urls": list(getattr(encoder, "license_urls", ())),
            "redistribution_review_required": bool(
                getattr(encoder, "redistribution_review_required", False)
            ),
        }
    return metadata


def _runtime_metadata(dependencies: ExtractionDependencies) -> dict[str, Any]:
    runtime = {
        "backend": dependencies.backend,
        "device": dependencies.device,
        "dtype": dependencies.dtype,
        "runtime_family": f"{platform.system().lower()}-{platform.machine().lower()}",
        "python_version": platform.python_version(),
        "libraries": {
            package: _package_version(package)
            for package in (
                "numpy",
                "torch",
                "torchvision",
                "transformers",
                "sentence-transformers",
                "huggingface-hub",
                "Pillow",
            )
        },
        "external_media_runtime": _ffmpeg_runtime_metadata(),
    }
    try:
        import torch

        runtime["torch_determinism"] = {
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "cuda_runtime": torch.version.cuda,
            "mps_built": bool(torch.backends.mps.is_built()),
        }
    except ImportError:
        runtime["torch_determinism"] = None
    return runtime


@lru_cache(maxsize=1)
def _ffmpeg_runtime_metadata() -> dict[str, Any]:
    """Fingerprint the exact external media tool builds used by extraction."""
    result: dict[str, Any] = {}
    for executable in ("ffmpeg", "ffprobe"):
        try:
            completed = subprocess.run(
                [executable, "-version"], check=True, capture_output=True, text=True
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise ExtractionError(
                "media_runtime_unavailable", f"required {executable} executable is unavailable"
            ) from error
        output = completed.stdout.replace("\r\n", "\n")
        first_line = output.splitlines()[0] if output.splitlines() else ""
        if not first_line.startswith(f"{executable} version "):
            raise ExtractionError("media_runtime_unavailable", f"cannot identify {executable}")
        result[executable] = {
            "version_line": first_line,
            "build_output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        }
    return result


def _failure_modality(code: str) -> str | None:
    for modality in ("video", "audio", "text", "transcript", "face", "media", "model"):
        if code.startswith(modality):
            return "text" if modality == "transcript" else modality
    return None


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON object keys instead of silently keeping the last value."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _portable_relative_path(value: str, *, code: str, label: str) -> Path:
    """Validate a canonical portable reference before resolving it within a root."""
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\0" in value
        or "\\" in value
    ):
        raise ExtractionError(code, f"{label} must be a portable relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ExtractionError(code, f"{label} must be a portable relative path")
    return path


def _safe_reference_for_sidecar(value: str) -> str | None:
    """Return a relative reference only; never serialize a host path on failure."""
    try:
        return _portable_relative_path(
            value,
            code="reference_unsafe",
            label="input reference",
        ).as_posix()
    except ExtractionError:
        return None


def _safe_failure_message(error: Exception, code: str) -> str:
    """Return a stable public message without copying arbitrary exception text or paths."""
    messages = {
        "media_reference_unsafe": "media reference is unsafe",
        "media_missing": "media input is unavailable",
        "transcript_reference_unsafe": "transcript reference is unsafe",
        "transcript_missing": "supplied transcript is unavailable",
        "face_crop_missing": "principal-face crop input is unavailable",
        "face_crop_invalid": "principal-face crop input is invalid",
        "face_crop_manifest_unavailable": "principal-face crop manifest is unavailable",
        "model_artifact_unavailable": "pinned model artifact is unavailable",
        "model_weight_hash_mismatch": "pinned model artifact checksum does not match",
        "unexpected_error": "unexpected extraction failure; inspect local run logs",
    }
    if code in messages:
        return messages[code]
    if isinstance(error, ExtractionError):
        return code.replace("_", " ")
    return messages["unexpected_error"]


def child_suffix_allowed(path: Path) -> bool:
    """Restrict provenance hashing to authored text/config source files."""
    return path.suffix in {".py", ".yaml", ".yml", ".json", ".toml"}


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ExtractionError("face_crop_invalid", f"{name} must be numeric")
    try:
        result = float(Decimal(str(value)))
    except (InvalidOperation, ValueError):
        raise ExtractionError("face_crop_invalid", f"{name} must be numeric") from None
    if not math.isfinite(result):
        raise ExtractionError("face_crop_invalid", f"{name} must be finite")
    return result


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _git(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments], cwd=root, text=True, check=True, capture_output=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ExtractionError("code_provenance_unavailable", str(error)) from error
    return result.stdout
