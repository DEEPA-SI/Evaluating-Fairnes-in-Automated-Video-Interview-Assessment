#!/usr/bin/env python3
"""Extract pinned multimodal features from the validated ordinary manifest boundary."""

from __future__ import annotations

import argparse
import gc
import hashlib
import os
import platform
import resource
import sys
import tempfile
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import (  # noqa: E402
    CandidateTranscript,
    MITInterviewFeatureSession,
    MITInterviewSession,
    SessionMappingContext,
    load_mit_interview_feature_dataset,
    load_verified_session_mapping,
)
from src.features.closure import (  # noqa: E402
    FEATURE_EXTRACTION_CLOSURE_FILENAME,
    write_feature_extraction_closure,
)
from src.features.encoders import (  # noqa: E402
    Dinov2VideoEncoder,
    MpnetTextEncoder,
    WavlmAudioEncoder,
)
from src.features.extraction import (  # noqa: E402
    AudioStreamTiming,
    CodeProvenance,
    ExtractionDependencies,
    ExtractionError,
    ExtractionSettings,
    FaceCropBatch,
    ManifestFaceCropProvider,
    SuppliedTranscriptProvider,
    canonical_json_bytes,
    chunk_token_ids,
    code_provenance,
    current_feature_run_context,
    decode_audio_ffmpeg,
    extract_dataset,
    pool_wavlm_seconds,
    set_deterministic_runtime,
    verify_current_feature_inputs,
    write_canonical_json,
)

OWNED_SOURCE_PATHS = (
    "scripts/extract_features.py",
    "src/__init__.py",
    "src/data/__init__.py",
    "src/data/mit_interview.py",
    "src/data/mit_session_mapping.py",
    "src/data/annotation_stimuli.py",
    "src/features/__init__.py",
    "src/features/extraction.py",
    "src/features/encoders.py",
    "src/features/closure.py",
    "src/frozen_json.py",
    "src/secure_io.py",
    "configs/default.yaml",
)


class _SyntheticVideoEncoder:
    name = "synthetic/dinov2-contract-double"
    revision = "0" * 40
    out_dim = 1024
    weight_file = "none"
    weight_sha256 = "0" * 64
    license_id = "test-only"
    license_urls: tuple[str, ...] = ()
    redistribution_review_required = False

    def encode(self, images: list[Any] | tuple[Any, ...]) -> np.ndarray:
        rows = np.arange(len(images) * self.out_dim, dtype=np.float32)
        return rows.reshape(len(images), self.out_dim) / self.out_dim


class _SyntheticAudioEncoder:
    name = "synthetic/wavlm-contract-double"
    revision = "0" * 40
    out_dim = 768
    sample_rate = 16_000
    conv_stride = 320
    receptive_field = 400
    weight_file = "none"
    weight_sha256 = "0" * 64
    license_id = "test-only"
    license_urls: tuple[str, ...] = ()
    redistribution_review_required = False

    def encode(self, waveform: np.ndarray) -> np.ndarray:
        from src.features.extraction import wavlm_output_length

        length = wavlm_output_length(waveform.size)
        rows = np.arange(length * self.out_dim, dtype=np.float32)
        return rows.reshape(length, self.out_dim) / self.out_dim


class _SyntheticTextEncoder:
    name = "synthetic/mpnet-contract-double"
    revision = "0" * 40
    out_dim = 768
    max_sequence_tokens = 384
    weight_file = "none"
    weight_sha256 = "0" * 64
    license_id = "test-only"
    license_urls: tuple[str, ...] = ()
    redistribution_review_required = False

    def tokenize(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def special_token_count(self) -> int:
        return 2

    def encode_token_chunks(
        self, chunks: list[list[int]] | tuple[tuple[int, ...], ...]
    ) -> np.ndarray:
        rows = np.zeros((len(chunks), self.out_dim), dtype=np.float32)
        for index, chunk in enumerate(chunks):
            rows[index, : min(len(chunk), self.out_dim)] = np.asarray(
                chunk[: self.out_dim], dtype=np.float32
            )
            norm = np.linalg.norm(rows[index])
            if norm:
                rows[index] /= norm
        return rows


class _UnavailableFaceCropManifest:
    """Turn a missing/invalid crop manifest into per-clip structured failures."""

    def __init__(self, error: Exception) -> None:
        self._message = str(error)

    def for_clip(self, _clip_id: str):
        message = self._message

        def missing(_path: Path, *, fps: int) -> FaceCropBatch:
            del fps
            raise ExtractionError("face_crop_manifest_unavailable", message)

        return missing

    def fingerprint_for_clip(self, _clip_id: str, *, fps: int) -> str:
        del fps
        raise ExtractionError("face_crop_manifest_unavailable", self._message)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["mit-interview"])
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--face-crops", type=Path)
    parser.add_argument("--session-mapping", type=Path)
    parser.add_argument("--identity-audit", type=Path)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--transcript-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--session",
        dest="session_ids",
        action="append",
        metavar="SESSION_ID",
        help="extract one exact session; repeat this option for a bounded pilot set",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--source-date-epoch", type=int)
    parser.add_argument(
        "--backend",
        choices=["real", "synthetic-smoke", "real-adapter-smoke"],
        default="real",
        help="synthetic-smoke validates orchestration only and writes no research evidence",
    )
    parser.add_argument(
        "--pilot-report",
        type=Path,
        default=Path("runs/t408_real_adapter_smoke.json"),
        help="ignored machine-readable output for real-adapter-smoke",
    )
    return parser.parse_args(argv)


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    return config, hashlib.sha256(raw).hexdigest()


def _resolved_paths(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Path]:
    data = config["data"]
    raw_dir = _lexically_absolute(args.dataset_root or Path(data["raw_dir"]))
    paths = {
        "dataset_root": raw_dir,
        "face_crops": args.face_crops
        or raw_dir / config["feature_extraction"]["face_crops_manifest"],
        "session_mapping": args.session_mapping or Path(data["session_mapping"]),
        "identity_audit": args.identity_audit or Path(data["identity_join_audit"]),
        "media_root": args.media_root or raw_dir,
        "transcript_root": args.transcript_root or raw_dir,
        "output_dir": args.output_dir or Path(data["feature_dir"]),
    }
    return {name: _lexically_absolute(path) for name, path in paths.items()}


def _lexically_absolute(path: Path) -> Path:
    """Normalise CLI dot components without following any filesystem symlink."""
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _select_clips(
    clips: tuple[MITInterviewSession | MITInterviewFeatureSession, ...],
    *,
    session_ids: list[str] | None,
    limit: int | None,
) -> tuple[MITInterviewSession | MITInterviewFeatureSession, ...]:
    """Select a validated bounded subset while preserving canonical dataset order."""
    if session_ids is not None:
        if limit is not None:
            raise ValueError("--session and --limit cannot be combined")
        if not session_ids or any(not value or value != value.strip() for value in session_ids):
            raise ValueError("--session values must be trimmed non-empty session IDs")
        if len(set(session_ids)) != len(session_ids):
            raise ValueError("--session values must be unique")
        available = {clip.clip_id for clip in clips}
        unknown = sorted(set(session_ids) - available, key=lambda value: value.encode("utf-8"))
        if unknown:
            raise ValueError(f"unknown --session value(s): {', '.join(unknown)}")
        requested = set(session_ids)
        return tuple(clip for clip in clips if clip.clip_id in requested)
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        return clips[:limit]
    return clips


def _real_dependencies(
    config: dict[str, Any],
    paths: dict[str, Path],
    clip: MITInterviewSession | MITInterviewFeatureSession,
    crop_manifest: ManifestFaceCropProvider | _UnavailableFaceCropManifest,
    *,
    device: str,
    local_files_only: bool,
) -> ExtractionDependencies:
    device = _selected_device(device)
    return ExtractionDependencies(
        face_crops=crop_manifest.for_clip(clip.source_session_id),
        decode_audio=decode_audio_ffmpeg,
        video_encoder=Dinov2VideoEncoder(
            config["encoders"]["video"], device=device, local_files_only=local_files_only
        ),
        audio_encoder=WavlmAudioEncoder(
            config["encoders"]["audio"], device=device, local_files_only=local_files_only
        ),
        text_encoder=MpnetTextEncoder(
            config["encoders"]["text"], device=device, local_files_only=local_files_only
        ),
        transcript=SuppliedTranscriptProvider(
            paths["transcript_root"], require_inline=True
        ),
        backend="huggingface",
        device=device,
        face_crop_fingerprint="0" * 64,
        face_crop_fingerprint_probe=lambda: "0" * 64,
    )


def _run_real(
    clips: tuple[MITInterviewSession | MITInterviewFeatureSession, ...],
    *,
    config: dict[str, Any],
    config_sha256: str,
    paths: dict[str, Path],
    settings: ExtractionSettings,
    code: CodeProvenance,
    args: argparse.Namespace,
    session_mapping: SessionMappingContext,
    publish_full_run_closure: bool,
) -> dict[str, Any]:
    # Models are loaded once. The face-crop callable is rebound per clip without
    # changing encoder instances.
    try:
        crop_manifest: ManifestFaceCropProvider | _UnavailableFaceCropManifest = (
            ManifestFaceCropProvider(
                paths["face_crops"],
                expected_manifest_sha256=(
                    session_mapping.face_crop_manifest_file_sha256
                ),
            )
        )
    except ValueError as error:
        crop_manifest = _UnavailableFaceCropManifest(error)
    first = clips[0]
    dependencies = _real_dependencies(
        config,
        paths,
        first,
        crop_manifest,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    records: list[dict[str, Any]] = []
    totals = {"succeeded": 0, "resumed": 0, "failed": 0}
    for clip in clips:
        face_crops, face_crop_fingerprint, face_crop_fingerprint_probe = _face_binding(
            crop_manifest, clip.source_session_id, settings.video_fps
        )
        bound = ExtractionDependencies(
            face_crops=face_crops,
            decode_audio=dependencies.decode_audio,
            video_encoder=dependencies.video_encoder,
            audio_encoder=dependencies.audio_encoder,
            text_encoder=dependencies.text_encoder,
            transcript=dependencies.transcript,
            backend=dependencies.backend,
            device=dependencies.device,
            dtype=dependencies.dtype,
            face_crop_fingerprint=face_crop_fingerprint,
            face_crop_fingerprint_probe=face_crop_fingerprint_probe,
            probe_video_duration=dependencies.probe_video_duration,
            probe_audio_stream_timing=dependencies.probe_audio_stream_timing,
        )
        summary = extract_dataset(
            [clip],
            media_root=paths["media_root"],
            output_dir=paths["output_dir"],
            dependencies=bound,
            settings=settings,
            config_sha256=config_sha256,
            code=code,
            script_path="scripts/extract_features.py",
            overwrite=args.overwrite,
            session_mapping=session_mapping,
        )
        totals["succeeded"] += summary.succeeded
        totals["resumed"] += summary.resumed
        totals["failed"] += summary.failed
        records.extend(summary.records)
    closure_summary: dict[str, str] | None = None
    if publish_full_run_closure:
        current_config, current_config_sha256 = _load_config(args.config)
        if current_config != config or current_config_sha256 != config_sha256:
            raise ValueError("feature configuration changed during extraction")
        current_code = code_provenance(ROOT, OWNED_SOURCE_PATHS)
        if current_code != code:
            raise ValueError("owned feature source changed during extraction")
        expected_run_context = current_feature_run_context(
            settings=settings,
            config_sha256=current_config_sha256,
            code=current_code,
            script_path="scripts/extract_features.py",
            dependencies=dependencies,
            session_mapping=session_mapping,
        )

        def verify_current_input(
            clip: MITInterviewSession | MITInterviewFeatureSession,
            metadata: dict[str, Any],
        ) -> None:
            _provider, _fingerprint, fingerprint_probe = _face_binding(
                crop_manifest, clip.source_session_id, settings.video_fps
            )
            verify_current_feature_inputs(
                clip,
                metadata,
                media_root=paths["media_root"],
                transcript=dependencies.transcript,
                face_crop_fingerprint_probe=fingerprint_probe,
                session_mapping=session_mapping,
            )

        closure = write_feature_extraction_closure(
            paths["output_dir"],
            records=records,
            selected_sessions=clips,
            session_mapping=session_mapping,
            expected_run_context=expected_run_context,
            verify_current_input=verify_current_input,
        )
        closure_bytes = canonical_json_bytes(closure)
        closure_summary = {
            "filename": FEATURE_EXTRACTION_CLOSURE_FILENAME,
            "file_sha256": hashlib.sha256(closure_bytes).hexdigest(),
            "payload_sha256": closure["closure_payload_sha256"],
        }
    return {
        **totals,
        "records": records,
        "feature_extraction_closure": closure_summary,
    }


def _face_binding(
    manifest: ManifestFaceCropProvider | _UnavailableFaceCropManifest,
    clip_id: str,
    fps: int,
):
    provider = manifest.for_clip(clip_id)

    def fingerprint_probe() -> str:
        return manifest.fingerprint_for_clip(clip_id, fps=fps)

    try:
        fingerprint = fingerprint_probe()
    except ExtractionError as error:
        failure = error
        fingerprint = hashlib.sha256(
            canonical_json_bytes({"clip_id": clip_id, "code": error.code, "message": str(error)})
        ).hexdigest()

        def provider(_path: Path, *, fps: int) -> FaceCropBatch:
            del fps
            raise failure

    return provider, fingerprint, fingerprint_probe


def _run_synthetic_smoke(settings: ExtractionSettings) -> dict[str, Any]:
    """Exercise the real writer/resume path with deterministic injected doubles."""
    with tempfile.TemporaryDirectory(prefix="fair-avi-smoke-") as directory:
        root = Path(directory).resolve()
        media = root / "Videos" / "P1.avi"
        audio = root / "Audio" / "P1.wav"
        media.parent.mkdir()
        audio.parent.mkdir()
        media.write_bytes(b"synthetic-video-contract-v1")
        audio.write_bytes(b"synthetic-audio-contract-v1")
        clip = MITInterviewFeatureSession(
            session_id="P1",
            source_session_id="P1",
            participant_id="1",
            condition="pre",
            fold=0,
            video_reference="Videos/P1.avi",
            audio_reference="Audio/P1.wav",
            transcript=CandidateTranscript(
                reference="Labels/interview_transcripts_by_turkers.csv",
                content="fixed supplied candidate transcript",
                candidate_segment_count=1,
                source_dialogue_sha256=hashlib.sha256(
                    b"fixed supplied candidate transcript"
                ).hexdigest(),
            ),
        )

        def crops(_path: Path, *, fps: int) -> FaceCropBatch:
            source_sha256 = hashlib.sha256(_path.read_bytes()).hexdigest()
            return FaceCropBatch(
                images=(np.zeros((224, 224, 3), dtype=np.uint8),),
                timestamps_seconds=np.array([0.0]),
                provenance={
                    "schema_version": "mit-candidate-face-crops-2",
                    "name": "synthetic-contract-double",
                    "revision": "4" * 40,
                    "principal_face_rule": "synthetic_candidate_only",
                    "confidence_threshold": 0.0,
                    "subject_role": "candidate",
                    "requested_fps": fps,
                    "source_video_sha256": source_sha256,
                    "full_duration_seconds": 1.0,
                    "full_one_hz_grid_count": 1,
                    "active_start_grid_index": 0,
                    "active_end_grid_index_exclusive": 1,
                    "active_close_reason": "end_of_stream",
                    "frame_rate_numerator": 1,
                    "frame_rate_denominator": 1,
                    "valid_candidate_grid_indices": [0],
                    "selected_grid_indices": [0],
                    "input_fingerprint": "3" * 64,
                },
            )

        dependencies = ExtractionDependencies(
            face_crops=crops,
            decode_audio=lambda _path, sample_rate: np.linspace(
                -0.5, 0.5, sample_rate + 128, dtype=np.float32
            ),
            video_encoder=_SyntheticVideoEncoder(),
            audio_encoder=_SyntheticAudioEncoder(),
            text_encoder=_SyntheticTextEncoder(),
            transcript=lambda value: value.transcript.content or "",
            backend="synthetic-smoke",
            face_crop_fingerprint="3" * 64,
            face_crop_fingerprint_probe=lambda: "3" * 64,
            probe_video_duration=lambda _path: 1.0,
            probe_audio_stream_timing=lambda path: (
                AudioStreamTiming(
                    codec_name="synthetic-embedded",
                    sample_rate=48_000,
                    start_pts=0,
                    duration_ts=42,
                    time_base_numerator=3,
                    time_base_denominator=125,
                )
                if path.suffix == ".avi"
                else AudioStreamTiming(
                    codec_name="synthetic-supplied",
                    sample_rate=48_000,
                    start_pts=None,
                    duration_ts=48_384,
                    time_base_numerator=1,
                    time_base_denominator=48_000,
                )
            ),
        )
        code = CodeProvenance("0" * 40, "1" * 64, False)
        first = extract_dataset(
            [clip],
            media_root=root,
            output_dir=root / "features",
            dependencies=dependencies,
            settings=settings,
            config_sha256="2" * 64,
            code=code,
            script_path="scripts/extract_features.py",
        )
        second = extract_dataset(
            [clip],
            media_root=root,
            output_dir=root / "features",
            dependencies=dependencies,
            settings=settings,
            config_sha256="2" * 64,
            code=code,
            script_path="scripts/extract_features.py",
        )
        if first.succeeded != 1 or second.resumed != 1:
            raise RuntimeError("synthetic extraction smoke did not write and resume exactly once")
        sidecar = next((root / "features").glob("*.json"))
        return {
            "backend": "synthetic-smoke",
            "research_evidence": False,
            "first": {"succeeded": first.succeeded, "failed": first.failed},
            "second": {"resumed": second.resumed},
            "sidecar_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        }


def _directory_bytes(path: Path) -> int:
    """Return logical bytes once per inode, including symlink targets once."""
    if not path.exists():
        return 0
    observed: set[tuple[int, int]] = set()
    total = 0
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        try:
            status = item.stat()
        except OSError:
            continue
        identity = (status.st_dev, status.st_ino)
        if identity not in observed:
            observed.add(identity)
            total += status.st_size
    return total


def _selected_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _synchronize(device: str) -> None:
    import torch

    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize(device)


def _release_accelerator(device: str) -> None:
    gc.collect()
    try:
        import torch

        if device == "mps":
            torch.mps.empty_cache()
        elif device.startswith("cuda"):
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def _smoke_tensor_record(values: np.ndarray, elapsed: float, encoder: Any) -> dict[str, Any]:
    matrix = np.asarray(values)
    return {
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "row_l2_norms": np.linalg.norm(matrix, axis=1).astype(float).tolist(),
        "finite": bool(np.isfinite(matrix).all()),
        "tensor_sha256": _array_sha256(matrix),
        "elapsed_seconds": elapsed,
        "repository": encoder.name,
        "revision": encoder.revision,
        "weight_file": encoder.weight_file,
        "weight_sha256_verified": encoder.weight_sha256,
        "weight_size_bytes": encoder.weight_size_bytes,
        "cache_snapshot_reference": encoder.cache_snapshot_reference,
    }


def _run_real_adapter_smoke(
    config: dict[str, Any],
    settings: ExtractionSettings,
    *,
    requested_device: str,
    local_files_only: bool,
) -> dict[str, Any]:
    """Run one bounded synthetic input through each pinned production adapter."""
    device = _selected_device(requested_device)
    set_deterministic_runtime(settings.seed)
    cache_root = Path(
        os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub")
    )
    cache_before = _directory_bytes(cache_root)
    total_start = time.perf_counter()
    outputs: dict[str, Any] = {}

    start = time.perf_counter()
    video_encoder = Dinov2VideoEncoder(
        config["encoders"]["video"], device=device, local_files_only=local_files_only
    )
    crop = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)
    video = video_encoder.encode([crop])
    _synchronize(device)
    outputs["video"] = _smoke_tensor_record(video, time.perf_counter() - start, video_encoder)
    del video_encoder
    _release_accelerator(device)

    start = time.perf_counter()
    audio_encoder = WavlmAudioEncoder(
        config["encoders"]["audio"], device=device, local_files_only=local_files_only
    )
    sample_count = int(1.2 * settings.audio_sample_rate)
    time_axis = np.arange(sample_count, dtype=np.float64) / settings.audio_sample_rate
    waveform = (0.1 * np.sin(2 * np.pi * 220 * time_axis)).astype(np.float32)
    hidden = audio_encoder.encode(waveform)
    audio, seconds = pool_wavlm_seconds(
        hidden,
        sample_count=sample_count,
        sample_rate=settings.audio_sample_rate,
        conv_stride=settings.audio_conv_stride,
        receptive_field=settings.audio_receptive_field,
    )
    _synchronize(device)
    outputs["audio"] = {
        **_smoke_tensor_record(audio, time.perf_counter() - start, audio_encoder),
        "final_hidden_shape": list(hidden.shape),
        "final_hidden_sha256": _array_sha256(hidden),
        "valid_second_indices": seconds.tolist(),
        "waveform_samples": sample_count,
    }
    del audio_encoder
    _release_accelerator(device)

    start = time.perf_counter()
    text_encoder = MpnetTextEncoder(
        config["encoders"]["text"], device=device, local_files_only=local_files_only
    )
    transcript = "This is a fixed corpus-supplied transcript for adapter validation."
    token_ids = text_encoder.tokenize(transcript)
    chunks = chunk_token_ids(
        token_ids,
        max_sequence_tokens=text_encoder.max_sequence_tokens,
        special_token_count=text_encoder.special_token_count(),
    )
    text = text_encoder.encode_token_chunks(chunks)
    _synchronize(device)
    outputs["text"] = {
        **_smoke_tensor_record(text, time.perf_counter() - start, text_encoder),
        "content_token_counts": [len(chunk) for chunk in chunks],
        "special_token_count": text_encoder.special_token_count(),
    }
    del text_encoder
    _release_accelerator(device)

    cache_after = _directory_bytes(cache_root)
    peak_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = int(peak_raw if sys.platform == "darwin" else peak_raw * 1024)
    return {
        "report_schema": "fair-avi-real-adapter-smoke-1",
        "research_evidence": False,
        "redistributes_model_weights": False,
        "source_date_epoch": settings.source_date_epoch,
        "device": device,
        "runtime_family": f"{platform.system().lower()}-{platform.machine().lower()}",
        "python": platform.python_version(),
        "libraries": {
            package: version(package)
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
        "elapsed_seconds": time.perf_counter() - total_start,
        "peak_rss_bytes": peak_bytes,
        "model_cache": {
            "bytes_before": cache_before,
            "bytes_after": cache_after,
            "bytes_added": max(cache_after - cache_before, 0),
        },
        "outputs": outputs,
        "status": "success",
    }


def _array_sha256(values: np.ndarray) -> str:
    """Hash tensor schema and canonical C-order bytes for smoke reproducibility."""
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(canonical_json_bytes({"shape": list(array.shape)}))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config, config_sha256 = _load_config(args.config)
        settings = ExtractionSettings.from_config(config, source_date_epoch=args.source_date_epoch)
        if args.backend == "synthetic-smoke":
            print(canonical_json_bytes(_run_synthetic_smoke(settings)).decode(), end="")
            return 0
        if args.backend == "real-adapter-smoke":
            report = _run_real_adapter_smoke(
                config,
                settings,
                requested_device=args.device,
                local_files_only=args.local_files_only,
            )
            write_canonical_json(args.pilot_report, report)
            print(canonical_json_bytes(report).decode(), end="")
            return 0
        paths = _resolved_paths(args, config)
        session_mapping = load_verified_session_mapping(
            paths["session_mapping"],
            identity_join_audit_report_path=paths["identity_audit"],
            face_crop_manifest_path=paths["face_crops"],
        )
        dataset = load_mit_interview_feature_dataset(
            paths["dataset_root"], session_mapping=session_mapping
        )
        clips = _select_clips(
            dataset.records,
            session_ids=args.session_ids,
            limit=args.limit,
        )
        code = code_provenance(ROOT, OWNED_SOURCE_PATHS)
        report = _run_real(
            clips,
            config=config,
            config_sha256=config_sha256,
            paths=paths,
            settings=settings,
            code=code,
            args=args,
            session_mapping=session_mapping,
            publish_full_run_closure=(
                len(clips) == 138
                and tuple(clip.clip_id for clip in clips)
                == tuple(clip.clip_id for clip in dataset.records)
            ),
        )
        print(canonical_json_bytes(report).decode(), end="")
        return 1 if report["failed"] else 0
    except Exception as error:
        print(f"feature extraction failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
