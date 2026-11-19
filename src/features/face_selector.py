"""Deterministic SCRFD candidate-face selection and crop generation.

The selector scans the source at one-Hz with the official InsightFace SCRFD
preprocessing and multi-stride output decoding.  It closes a deterministic
outcome-blind active-interview interval after a sustained candidate absence,
retains at most 128 candidate-valid seconds and never substitutes an uncropped
frame or another face. Detector, boundary, crop and runtime provenance are
serialized by the caller in the strict face-crop manifest.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from src.features.extraction import ExtractionError

SELECTOR_VERSION = "fair-avi-scrfd-face-selector-2"
OFFICIAL_SCRFD_PREPROCESSING = "bgr_resize_aspect_pad_top_left_blob_mean_127.5_std_128_swap_rb"
OFFICIAL_SCRFD_DECODING = "multi_stride_distance2bbox_then_nms"
CANDIDATE_RULE = "per_second_unique_most_central_with_margin"
ACTIVE_INTERVAL_RULE = "first_valid_then_first_terminal_invalid_run"
RETAINED_GRID_RULE = "valid_active_indices_endpoint_rank"
CROP_RULE = "square_longest_detector_side_scaled_translated_within_frame"


@dataclass(frozen=True, slots=True)
class Detection:
    """One finite SCRFD face detection in source-frame pixel coordinates."""

    box_xyxy: tuple[float, float, float, float]
    confidence: float

    def __post_init__(self) -> None:
        values = (*self.box_xyxy, self.confidence)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("face detection values must be finite")
        x0, y0, x1, y1 = self.box_xyxy
        if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
            raise ValueError("face detection box must have positive in-frame geometry")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("face detection confidence must be in [0,1]")

    @property
    def centroid(self) -> tuple[float, float]:
        """Return the box centre in source-frame pixels."""
        x0, y0, x1, y1 = self.box_xyxy
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    @property
    def area(self) -> float:
        """Return the continuous box area in square pixels."""
        x0, y0, x1, y1 = self.box_xyxy
        return (x1 - x0) * (y1 - y0)


@dataclass(frozen=True, slots=True)
class VideoStreamInfo:
    """Immutable source-stream facts used to map grid times to source frames."""

    duration_seconds: float
    width: int
    height: int
    frame_rate_numerator: int
    frame_rate_denominator: int
    nominal_frame_rate_numerator: int
    nominal_frame_rate_denominator: int
    time_base_numerator: int
    time_base_denominator: int
    frame_count: int
    stream_duration_seconds: float
    stream_start_time_seconds: float
    constant_frame_rate: bool

    @property
    def frame_rate(self) -> Fraction:
        """Return the exact probed average frame rate."""
        return Fraction(self.frame_rate_numerator, self.frame_rate_denominator)


@dataclass(frozen=True, slots=True)
class DecodedFrame:
    """One requested grid point and its deterministically chosen source frame."""

    grid_index: int
    timestamp_seconds: float
    source_frame_index: int
    decoded_timestamp_seconds: float
    bgr: np.ndarray


@dataclass(frozen=True, slots=True)
class _SourceFileSnapshot:
    """Stable identity and content facts for one immutable source file."""

    device: int
    inode: int
    size: int
    mtime_ns: int
    sha256: str


@dataclass(frozen=True, slots=True)
class CandidateScan:
    """Outcome-blind one-Hz candidate-visibility scan for one source session."""

    rows: tuple[dict[str, Any], ...]
    valid_detections: Mapping[int, Detection]
    active_start_grid_index: int | None
    active_end_grid_index_exclusive: int | None
    active_close_reason: str | None
    valid_candidate_grid_indices: tuple[int, ...]
    selected_grid_indices: tuple[int, ...]
    failure: ExtractionError | None


class FaceDetector(Protocol):
    """Injectable detector boundary used by production and synthetic tests."""

    name: str
    revision: str

    def detect(self, frame_bgr: np.ndarray) -> tuple[Detection, ...]:
        """Return deterministic face detections for one BGR frame."""


class FrameDecoder(Protocol):
    """Injectable video-probe and selected-frame decoder boundary."""

    def probe(self, video_path: Path) -> VideoStreamInfo:
        """Return exact source-stream facts."""

    def decode(
        self, video_path: Path, stream: VideoStreamInfo, grid_indices: Sequence[int]
    ) -> tuple[DecodedFrame, ...]:
        """Decode one source frame for every requested integer-second point."""

    def iter_decode(
        self, video_path: Path, stream: VideoStreamInfo, grid_indices: Sequence[int]
    ) -> Iterator[DecodedFrame]:
        """Yield one source frame for every requested integer-second point."""


class ScrfdFaceDetector:
    """Hash-verified official buffalo_l SCRFD-10GF ONNX detector."""

    def __init__(self, config: Mapping[str, Any], model_path: Path) -> None:
        settings = validate_selector_config(config)
        try:
            supplied = _secure_regular_source_path(Path(model_path))
            model_snapshot = _capture_source_snapshot(supplied)
        except (OSError, ValueError):
            raise ExtractionError(
                "face_detector_artifact_unavailable",
                "pinned SCRFD detector artifact is unavailable",
            ) from None
        observed_sha256 = model_snapshot.sha256
        if observed_sha256 != settings["artifact_sha256"]:
            raise ExtractionError(
                "face_detector_artifact_hash_mismatch",
                "pinned SCRFD detector artifact checksum does not match",
            )
        if model_snapshot.size != settings["artifact_size_bytes"]:
            raise ExtractionError(
                "face_detector_artifact_hash_mismatch",
                "pinned SCRFD detector artifact size does not match",
            )
        try:
            import cv2
            import insightface
            import onnxruntime
        except ImportError as error:
            raise ExtractionError(
                "face_detector_runtime_unavailable",
                "pinned insightface and onnxruntime runtimes are required for SCRFD",
            ) from error
        if "CPUExecutionProvider" not in onnxruntime.get_available_providers():
            raise ExtractionError(
                "face_detector_runtime_unavailable", "onnxruntime CPU provider is unavailable"
            )
        options = onnxruntime.SessionOptions()
        options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.use_deterministic_compute = True
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
        session = onnxruntime.InferenceSession(
            str(supplied),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.name = settings["name"]
        self.revision = settings["revision"]
        self.insightface_version = str(insightface.__version__)
        self.onnxruntime_version = str(onnxruntime.__version__)
        self._settings = settings
        from insightface.model_zoo.scrfd import SCRFD

        self._runtime = SCRFD(model_file=str(supplied), session=session)
        self._runtime.prepare(
            -1,
            input_size=tuple(settings["input_size"]),
            det_thresh=settings["confidence_threshold"],
            nms_thresh=settings["nms_threshold"],
        )
        _verify_source_snapshot(supplied, model_snapshot, verify_content=True)

    def detect(self, frame_bgr: np.ndarray) -> tuple[Detection, ...]:
        """Return official-path SCRFD detections in source-frame coordinates."""
        frame = np.asarray(frame_bgr)
        if (
            frame.ndim != 3
            or frame.shape[2] != 3
            or frame.dtype != np.uint8
            or frame.shape[0] <= 0
            or frame.shape[1] <= 0
        ):
            raise ExtractionError(
                "face_frame_invalid", "SCRFD input must be a non-empty uint8 BGR frame"
            )
        boxes, _keypoints = self._runtime.detect(
            frame, input_size=tuple(self._settings["input_size"])
        )
        values = np.asarray(boxes)
        if values.ndim != 2 or values.shape[1] != 5 or not np.isfinite(values).all():
            raise ExtractionError(
                "face_detector_output_invalid", "official SCRFD output shape drifted"
            )
        frame_height, frame_width = frame.shape[:2]
        detections: list[Detection] = []
        for x0, y0, x1, y1, confidence in values.tolist():
            clipped = (
                min(max(float(x0), 0.0), float(frame_width)),
                min(max(float(y0), 0.0), float(frame_height)),
                min(max(float(x1), 0.0), float(frame_width)),
                min(max(float(y1), 0.0), float(frame_height)),
            )
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                continue
            detections.append(Detection(clipped, float(confidence)))
        detections.sort(key=_detection_sort_key)
        return tuple(detections)

    @property
    def selector_provenance(self) -> dict[str, Any]:
        """Return JSON-safe detector/method provenance for the strict manifest."""
        return dict(self._settings)


class OpenCvFrameDecoder:
    """Decode exact probed source-frame indices with a pinned OpenCV runtime."""

    def __init__(self) -> None:
        import cv2

        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
        self.opencv_version = str(cv2.__version__)
        self.ffprobe_version, self.ffprobe_build_sha256 = _ffprobe_provenance()

    def probe(self, video_path: Path) -> VideoStreamInfo:
        """Probe one regular source video with FFprobe JSON output."""
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            (
                "stream=width,height,r_frame_rate,avg_frame_rate,time_base,start_time,"
                "duration,nb_frames:format=duration"
            ),
            "-of",
            "json",
            str(video_path),
        ]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            payload = json.loads(result.stdout)
            stream = payload["streams"][0]
            duration = float(payload["format"]["duration"])
            width = int(stream["width"])
            height = int(stream["height"])
            frame_rate = _positive_fraction(stream["avg_frame_rate"], "average frame rate")
            nominal_frame_rate = _positive_fraction(stream["r_frame_rate"], "nominal frame rate")
            time_base = _positive_fraction(stream["time_base"], "stream time base")
            stream_duration = float(stream["duration"])
            stream_start = float(stream["start_time"])
            frame_count = int(stream["nb_frames"])
        except (
            OSError,
            subprocess.CalledProcessError,
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as error:
            raise ExtractionError(
                "face_video_probe_failed", "cannot identify source video stream"
            ) from error
        if (
            not math.isfinite(duration)
            or duration <= 0
            or width <= 0
            or height <= 0
            or not math.isfinite(stream_duration)
            or stream_duration <= 0
            or not math.isfinite(stream_start)
            or stream_start != 0.0
            or frame_count <= 0
        ):
            raise ExtractionError("face_video_probe_failed", "source video stream is invalid")
        constant_frame_rate = (
            nominal_frame_rate == frame_rate
            and time_base == 1 / frame_rate
            and math.isclose(
                stream_duration * float(frame_rate),
                float(frame_count),
                rel_tol=0.0,
                abs_tol=1e-3,
            )
        )
        if not constant_frame_rate:
            raise ExtractionError(
                "face_video_not_cfr",
                "candidate-face extraction requires the frozen constant-frame-rate corpus",
            )
        return VideoStreamInfo(
            duration_seconds=duration,
            width=width,
            height=height,
            frame_rate_numerator=frame_rate.numerator,
            frame_rate_denominator=frame_rate.denominator,
            nominal_frame_rate_numerator=nominal_frame_rate.numerator,
            nominal_frame_rate_denominator=nominal_frame_rate.denominator,
            time_base_numerator=time_base.numerator,
            time_base_denominator=time_base.denominator,
            frame_count=frame_count,
            stream_duration_seconds=stream_duration,
            stream_start_time_seconds=stream_start,
            constant_frame_rate=True,
        )

    def decode(
        self, video_path: Path, stream: VideoStreamInfo, grid_indices: Sequence[int]
    ) -> tuple[DecodedFrame, ...]:
        """Seek to a deterministic nearest source-frame index at every grid time."""
        return tuple(self.iter_decode(video_path, stream, grid_indices))

    def iter_decode(
        self, video_path: Path, stream: VideoStreamInfo, grid_indices: Sequence[int]
    ) -> Iterator[DecodedFrame]:
        """Yield selected frames without retaining the full one-Hz scan in memory."""
        import cv2

        requested = tuple(int(value) for value in grid_indices)
        if not requested or requested != tuple(sorted(set(requested))):
            raise ValueError("grid indices must be a non-empty increasing unique sequence")
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ExtractionError("face_video_decode_failed", "cannot open source video")
        try:
            for grid_index in requested:
                frame_index = _nearest_source_frame_index(grid_index, stream.frame_rate)
                if frame_index < 0 or frame_index >= stream.frame_count:
                    raise ExtractionError(
                        "face_video_decode_failed", "selected source-frame index is unavailable"
                    )
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index)):
                    raise ExtractionError(
                        "face_video_decode_failed", "cannot seek selected source-frame index"
                    )
                success, frame = capture.read()
                if not success or frame is None:
                    raise ExtractionError(
                        "face_video_decode_failed", "cannot decode selected source frame"
                    )
                image = np.asarray(frame)
                if image.dtype != np.uint8 or image.shape != (stream.height, stream.width, 3):
                    raise ExtractionError(
                        "face_video_decode_failed", "decoded source-frame shape drifted"
                    )
                yield DecodedFrame(
                    grid_index=grid_index,
                    timestamp_seconds=float(grid_index),
                    source_frame_index=frame_index,
                    decoded_timestamp_seconds=float(Fraction(frame_index, 1) / stream.frame_rate),
                    bgr=image.copy(),
                )
        finally:
            capture.release()


def validate_selector_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalise the frozen candidate-face selector settings."""
    required = {
        "name",
        "revision",
        "artifact_release",
        "artifact_url",
        "archive_sha256",
        "artifact_file",
        "artifact_sha256",
        "artifact_size_bytes",
        "input_size",
        "preprocessing",
        "decoding",
        "confidence_threshold",
        "nms_threshold",
        "principal_face_rule",
        "candidate_centrality_max_ratio",
        "candidate_centrality_margin_ratio",
        "scan_fps",
        "first_valid_max_seconds",
        "minimum_active_elapsed_seconds",
        "terminal_invalid_run_seconds",
        "minimum_valid_candidate_seconds",
        "active_interval_rule",
        "retained_grid_rule",
        "maximum_retained_samples",
        "crop_rule",
        "crop_scale",
        "minimum_crop_side_pixels",
        "subject_role",
    }
    if not isinstance(config, Mapping) or set(config) != required:
        raise ValueError("candidate-face selector config has an unexpected field set")
    result = dict(config)
    for field in (
        "name",
        "artifact_release",
        "artifact_url",
        "artifact_file",
    ):
        value = result[field]
        if not isinstance(value, str) or not value or value != value.strip() or "\0" in value:
            raise ValueError(f"selector {field} must be trimmed non-empty text")
    if not str(result["artifact_url"]).startswith("https://"):
        raise ValueError("selector artifact_url must use HTTPS")
    for field in ("revision",):
        if not _lower_hex(result[field], 40):
            raise ValueError(f"selector {field} must be a lowercase SHA-1")
    for field in ("archive_sha256", "artifact_sha256"):
        if not _lower_hex(result[field], 64):
            raise ValueError(f"selector {field} must be a lowercase SHA-256")
    for field, expected in (
        ("preprocessing", OFFICIAL_SCRFD_PREPROCESSING),
        ("decoding", OFFICIAL_SCRFD_DECODING),
        ("principal_face_rule", CANDIDATE_RULE),
        ("active_interval_rule", ACTIVE_INTERVAL_RULE),
        ("retained_grid_rule", RETAINED_GRID_RULE),
        ("crop_rule", CROP_RULE),
        ("subject_role", "candidate"),
    ):
        if result[field] != expected:
            raise ValueError(f"selector {field} drifted from the frozen contract")
    size = result["input_size"]
    if (
        not isinstance(size, list)
        or len(size) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in size
        )
    ):
        raise ValueError("selector input_size must contain two positive integers")
    for field in (
        "confidence_threshold",
        "nms_threshold",
        "candidate_centrality_max_ratio",
        "candidate_centrality_margin_ratio",
    ):
        value = _finite_float(result[field], field)
        if not 0.0 < value <= 1.0:
            raise ValueError(f"selector {field} must be in (0,1]")
        result[field] = value
    crop_scale = _finite_float(result["crop_scale"], "crop_scale")
    if crop_scale < 1.0:
        raise ValueError("selector crop scale must be at least one")
    result["crop_scale"] = crop_scale
    for field in (
        "artifact_size_bytes",
        "minimum_crop_side_pixels",
        "minimum_active_elapsed_seconds",
        "terminal_invalid_run_seconds",
        "minimum_valid_candidate_seconds",
        "maximum_retained_samples",
    ):
        value = result[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"selector {field} must be a positive integer")
    first_valid = result["first_valid_max_seconds"]
    if isinstance(first_valid, bool) or not isinstance(first_valid, int) or first_valid < 0:
        raise ValueError("selector first_valid_max_seconds must be a non-negative integer")
    if result["scan_fps"] != 1:
        raise ValueError("selector scan_fps must be exactly one")
    if result["maximum_retained_samples"] != 128:
        raise ValueError("selector maximum_retained_samples must be exactly 128")
    if result["minimum_valid_candidate_seconds"] > result["maximum_retained_samples"]:
        raise ValueError("selector minimum valid count cannot exceed the retained cap")
    return result


def select_candidate_detection(
    detections: Sequence[Detection],
    *,
    frame_width: int,
    frame_height: int,
    centrality_max_ratio: float,
    centrality_margin_ratio: float,
) -> Detection:
    """Select the unique most-central face for one second or fail without substitution."""
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")
    if not 0.0 < centrality_max_ratio <= 1.0:
        raise ValueError("candidate centrality limit must be in (0,1]")
    if not 0.0 < centrality_margin_ratio <= 1.0:
        raise ValueError("candidate centrality margin must be in (0,1]")
    values = tuple(detections)
    if not values:
        raise ExtractionError("face_not_detected", "no face was detected at this grid second")
    ranked = sorted(
        (
            (
                _detection_centrality(
                    detection, frame_width=frame_width, frame_height=frame_height
                ),
                _detection_sort_key(detection),
                detection,
            )
            for detection in values
        ),
        key=lambda item: (item[0], item[1]),
    )
    best_centrality, _key, best = ranked[0]
    if best_centrality > centrality_max_ratio:
        raise ExtractionError(
            "face_candidate_off_centre",
            "the most-central face lies outside the frozen candidate region",
        )
    if len(ranked) > 1 and ranked[1][0] - best_centrality < centrality_margin_ratio:
        raise ExtractionError(
            "face_candidate_ambiguous",
            "multiple faces lack the frozen candidate centrality margin",
        )
    return best


def bounded_valid_grid_indices(
    valid_grid_indices: Sequence[int], maximum: int = 128
) -> tuple[int, ...]:
    """Select endpoint-spanning ranks from candidate-valid integer-second indices."""
    values = tuple(int(value) for value in valid_grid_indices)
    if not values or values != tuple(sorted(set(values))):
        raise ValueError("valid grid indices must be increasing, unique and non-empty")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
        raise ValueError("maximum retained samples must be a positive integer")
    if len(values) <= maximum:
        return values
    if maximum == 1:
        return (values[0],)
    return tuple(values[(rank * (len(values) - 1)) // (maximum - 1)] for rank in range(maximum))


def scan_candidate_activity(
    video_path: Path,
    stream: VideoStreamInfo,
    detector: FaceDetector,
    decoder: FrameDecoder,
    selector_config: Mapping[str, Any],
) -> CandidateScan:
    """Find the active interview from an outcome-blind one-Hz candidate-validity scan."""
    settings = validate_selector_config(selector_config)
    grid_count = max(1, math.floor(stream.duration_seconds))
    rows: list[dict[str, Any]] = []
    valid: dict[int, Detection] = {}
    first_valid: int | None = None
    last_valid: int | None = None
    consecutive_invalid = 0
    close_reason: str | None = None

    for frame in decoder.iter_decode(video_path, stream, range(grid_count)):
        detections = detector.detect(frame.bgr)
        try:
            candidate = select_candidate_detection(
                detections,
                frame_width=stream.width,
                frame_height=stream.height,
                centrality_max_ratio=settings["candidate_centrality_max_ratio"],
                centrality_margin_ratio=settings["candidate_centrality_margin_ratio"],
            )
        except ExtractionError as error:
            if error.code not in {
                "face_not_detected",
                "face_candidate_off_centre",
                "face_candidate_ambiguous",
            }:
                raise
            rows.append(
                {
                    "grid_index": frame.grid_index,
                    "timestamp_seconds": str(frame.grid_index),
                    "status": "invalid",
                    "reason": error.code,
                    "detection_count": len(detections),
                    "box_xyxy": None,
                    "confidence": None,
                    "centrality_ratio": None,
                }
            )
            if first_valid is not None:
                consecutive_invalid += 1
        else:
            centrality = _detection_centrality(
                candidate, frame_width=stream.width, frame_height=stream.height
            )
            valid[frame.grid_index] = candidate
            first_valid = frame.grid_index if first_valid is None else first_valid
            last_valid = frame.grid_index
            consecutive_invalid = 0
            rows.append(
                {
                    "grid_index": frame.grid_index,
                    "timestamp_seconds": str(frame.grid_index),
                    "status": "valid",
                    "reason": None,
                    "detection_count": len(detections),
                    "box_xyxy": [float(value) for value in candidate.box_xyxy],
                    "confidence": float(candidate.confidence),
                    "centrality_ratio": centrality,
                }
            )

        if first_valid is None and frame.grid_index >= settings["first_valid_max_seconds"]:
            return CandidateScan(
                rows=tuple(rows),
                valid_detections={},
                active_start_grid_index=None,
                active_end_grid_index_exclusive=None,
                active_close_reason=None,
                valid_candidate_grid_indices=(),
                selected_grid_indices=(),
                failure=ExtractionError(
                    "face_active_start_missing",
                    "no unambiguous candidate face was found within the frozen start boundary",
                ),
            )
        if (
            first_valid is not None
            and consecutive_invalid >= settings["terminal_invalid_run_seconds"]
            and frame.grid_index - first_valid + 1 >= settings["minimum_active_elapsed_seconds"]
        ):
            close_reason = "terminal_invalid_run"
            break
    else:
        close_reason = "end_of_stream"

    if first_valid is None or last_valid is None:
        return CandidateScan(
            rows=tuple(rows),
            valid_detections={},
            active_start_grid_index=None,
            active_end_grid_index_exclusive=None,
            active_close_reason=None,
            valid_candidate_grid_indices=(),
            selected_grid_indices=(),
            failure=ExtractionError(
                "face_active_start_missing",
                "no unambiguous candidate face was found within the frozen start boundary",
            ),
        )
    active_end = last_valid + 1
    active_valid = tuple(index for index in valid if first_valid <= index < active_end)
    if len(active_valid) < settings["minimum_valid_candidate_seconds"]:
        return CandidateScan(
            rows=tuple(rows),
            valid_detections=dict(valid),
            active_start_grid_index=first_valid,
            active_end_grid_index_exclusive=active_end,
            active_close_reason=close_reason,
            valid_candidate_grid_indices=active_valid,
            selected_grid_indices=(),
            failure=ExtractionError(
                "face_candidate_insufficient",
                "active interview has fewer than the frozen minimum candidate-valid seconds",
            ),
        )
    selected = bounded_valid_grid_indices(
        active_valid, maximum=settings["maximum_retained_samples"]
    )
    return CandidateScan(
        rows=tuple(rows),
        valid_detections={index: valid[index] for index in active_valid},
        active_start_grid_index=first_valid,
        active_end_grid_index_exclusive=active_end,
        active_close_reason=close_reason,
        valid_candidate_grid_indices=active_valid,
        selected_grid_indices=selected,
        failure=None,
    )


def crop_box_xyxy(
    detection: Detection,
    *,
    frame_width: int,
    frame_height: int,
    crop_scale: float,
    minimum_crop_side_pixels: int,
) -> tuple[int, int, int, int]:
    """Return a padded square crop translated, never reflected, into the frame."""
    if frame_width <= 0 or frame_height <= 0 or crop_scale < 1.0:
        raise ValueError("crop frame dimensions and scale are invalid")
    if minimum_crop_side_pixels <= 0:
        raise ValueError("minimum crop side must be positive")
    x0, y0, x1, y1 = detection.box_xyxy
    raw_side = max(x1 - x0, y1 - y0)
    if raw_side < minimum_crop_side_pixels:
        raise ExtractionError(
            "face_crop_too_small", "selected detector box is below the minimum crop side"
        )
    side = min(int(math.ceil(raw_side * crop_scale)), frame_width, frame_height)
    if side < minimum_crop_side_pixels:
        raise ExtractionError("face_crop_too_small", "padded crop is below the minimum side")
    centre_x, centre_y = detection.centroid
    left = int(math.floor(centre_x - side / 2.0))
    top = int(math.floor(centre_y - side / 2.0))
    left = min(max(left, 0), frame_width - side)
    top = min(max(top, 0), frame_height - side)
    return (left, top, left + side, top + side)


def build_session_crop_rows(
    decoded_frames: Sequence[DecodedFrame],
    selected_detections: Mapping[int, Detection],
    *,
    output_root: Path,
    reference_root: Path,
    session_id: str,
    crop_scale: float,
    minimum_crop_side_pixels: int,
) -> list[dict[str, Any]]:
    """Build and publish one content-addressed session crop directory atomically."""
    if not decoded_frames:
        raise ExtractionError("face_video_decode_failed", "no selected video frames decoded")
    if (
        not isinstance(session_id, str)
        or re.fullmatch(r"P{1,2}[1-9][0-9]*", session_id, re.ASCII) is None
    ):
        raise ExtractionError("face_crop_path_invalid", "session ID is not path-safe")
    if ".." in Path(output_root).parts or ".." in Path(reference_root).parts:
        raise ExtractionError(
            "face_crop_path_invalid", "crop paths cannot contain parent traversal"
        )
    try:
        restricted_root = _secure_directory_path(Path(reference_root), create=False)
        expected_output_root = restricted_root / "face_crops"
        requested_output_root = _lexical_absolute(Path(output_root))
    except (OSError, ValueError) as error:
        raise ExtractionError(
            "face_crop_path_invalid", "crop output parent is unavailable or unsafe"
        ) from error
    if requested_output_root != expected_output_root:
        raise ExtractionError(
            "face_crop_path_invalid", "crop root must be beside the restricted manifest"
        )
    first_shape = decoded_frames[0].bgr.shape
    if any(frame.bgr.shape != first_shape for frame in decoded_frames):
        raise ExtractionError("face_video_decode_failed", "source frame geometry changed")
    frame_height, frame_width = first_shape[:2]
    required = tuple(frame.grid_index for frame in decoded_frames)
    if required != tuple(sorted(set(required))) or tuple(selected_detections) != required:
        raise ExtractionError(
            "face_candidate_incomplete", "candidate detections do not cover the retained grid"
        )
    try:
        secure_output_root = _secure_directory_path(expected_output_root, create=True)
    except (OSError, ValueError) as error:
        raise ExtractionError(
            "face_crop_path_invalid", "crop output directory is unavailable or unsafe"
        ) from error
    stage = Path(tempfile.mkdtemp(prefix=f".{session_id}.", dir=secure_output_root))
    os.chmod(stage, 0o700, follow_symlinks=False)
    staged_rows: list[dict[str, Any]] = []
    session_digest = hashlib.sha256()
    try:
        for frame in decoded_frames:
            detection = selected_detections[frame.grid_index]
            crop_box = crop_box_xyxy(
                detection,
                frame_width=frame_width,
                frame_height=frame_height,
                crop_scale=crop_scale,
                minimum_crop_side_pixels=minimum_crop_side_pixels,
            )
            left, top, right, bottom = crop_box
            encoded = _encode_png_rgb(frame.bgr[top:bottom, left:right])
            filename = f"{frame.grid_index:06d}.png"
            crop_sha256 = hashlib.sha256(encoded).hexdigest()
            _atomic_write(stage / filename, encoded)
            session_digest.update(filename.encode("ascii") + b"\0")
            session_digest.update(bytes.fromhex(crop_sha256))
            staged_rows.append(
                {
                    "filename": filename,
                    "crop_sha256": crop_sha256,
                    "timestamp_seconds": str(frame.grid_index),
                    "grid_index": frame.grid_index,
                    "source_frame_index": frame.source_frame_index,
                    "decoded_timestamp_seconds": _decimal_text(frame.decoded_timestamp_seconds),
                    "box_xyxy": [float(value) for value in detection.box_xyxy],
                    "crop_box_xyxy": list(crop_box),
                    "confidence": float(detection.confidence),
                }
            )
        directory_name = f"{session_id}-{session_digest.hexdigest()[:24]}"
        published = secure_output_root / directory_name
        _fsync_directory(stage)
        _publish_crop_directory(stage, published)
        reference_directory = Path("face_crops") / directory_name
        if restricted_root / reference_directory != published:
            raise ExtractionError(
                "face_crop_path_invalid", "crop output is not beneath the manifest directory"
            )
        return [
            {
                "path": (reference_directory / str(row["filename"])).as_posix(),
                **{key: value for key, value in row.items() if key != "filename"},
            }
            for row in staged_rows
        ]
    except Exception:
        if stage.exists() and not stage.is_symlink():
            shutil.rmtree(stage)
        raise


def generate_face_crop_manifest(
    sessions: Sequence[Any],
    *,
    dataset_root: Path,
    output_manifest: Path,
    crop_root: Path,
    detector: ScrfdFaceDetector,
    decoder: OpenCvFrameDecoder,
    selector_config: Mapping[str, Any],
    generator_provenance: Mapping[str, Any],
    maximum_samples: int = 128,
    progress: Callable[[str, str], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate a strict all-session manifest plus a non-authoritative QC summary."""
    settings = validate_selector_config(selector_config)
    detector_settings = validate_selector_config(detector.selector_provenance)
    if detector_settings != settings:
        raise ValueError("detector and selector configuration provenance do not match")
    if maximum_samples != settings["maximum_retained_samples"]:
        raise ValueError("candidate-face retained cap drifted from selector provenance")
    try:
        root = _secure_directory_path(Path(dataset_root), create=False)
        manifest_root = _secure_directory_path(Path(output_manifest).parent, create=True)
    except (OSError, ValueError) as error:
        raise ValueError("source or restricted output directory is unsafe") from error
    crops = _lexical_absolute(Path(crop_root))
    if crops != manifest_root / "face_crops":
        raise ValueError("crop_root must be the face_crops directory beside the manifest")
    clips: dict[str, Any] = {}
    qc_rows: list[dict[str, Any]] = []
    for session in sessions:
        session_id = str(session.source_session_id)
        reference = Path(str(session.video_reference))
        if (
            reference.is_absolute()
            or ".." in reference.parts
            or not reference.parts
            or "\\" in str(session.video_reference)
            or "\0" in str(session.video_reference)
        ):
            raise ExtractionError("face_media_missing", "source video reference is unavailable")
        try:
            video_path = _secure_regular_source_path(root / reference, root=root)
            source_snapshot = _capture_source_snapshot(video_path)
        except (OSError, ValueError):
            raise ExtractionError(
                "face_media_missing", "source video reference is unavailable"
            ) from None
        stream = decoder.probe(video_path)
        _verify_source_snapshot(video_path, source_snapshot, verify_content=False)
        grid_count = max(1, math.floor(stream.duration_seconds))
        base = {
            "source_video_sha256": source_snapshot.sha256,
            "duration_seconds": _decimal_text(stream.duration_seconds),
            "stream_duration_seconds": _decimal_text(stream.stream_duration_seconds),
            "full_one_hz_grid_count": grid_count,
            "frame_width": stream.width,
            "frame_height": stream.height,
            "frame_rate_numerator": stream.frame_rate_numerator,
            "frame_rate_denominator": stream.frame_rate_denominator,
            "nominal_frame_rate_numerator": stream.nominal_frame_rate_numerator,
            "nominal_frame_rate_denominator": stream.nominal_frame_rate_denominator,
            "time_base_numerator": stream.time_base_numerator,
            "time_base_denominator": stream.time_base_denominator,
            "frame_count": stream.frame_count,
            "stream_start_time_seconds": _decimal_text(stream.stream_start_time_seconds),
            "constant_frame_rate": stream.constant_frame_rate,
        }
        scan: CandidateScan | None = None
        try:
            scan = scan_candidate_activity(
                video_path, stream, detector, decoder, selector_config=settings
            )
            if scan.failure is not None:
                raise scan.failure
            decoded = decoder.decode(video_path, stream, scan.selected_grid_indices)
            selected_detections = {
                index: scan.valid_detections[index] for index in scan.selected_grid_indices
            }
            _verify_source_snapshot(video_path, source_snapshot, verify_content=True)
            rows = build_session_crop_rows(
                decoded,
                selected_detections,
                output_root=crops,
                reference_root=manifest_root,
                session_id=session_id,
                crop_scale=settings["crop_scale"],
                minimum_crop_side_pixels=settings["minimum_crop_side_pixels"],
            )
            clip = {
                **base,
                **_scan_manifest_fields(scan),
                "status": "success",
                "failure": None,
                "frames": rows,
            }
            qc = {
                "session_id": session_id,
                "status": "success",
                "scanned_grid_count": len(scan.rows),
                "valid_candidate_grid_count": len(scan.valid_candidate_grid_indices),
                "selected_grid_count": len(scan.selected_grid_indices),
                "crop_count": len(rows),
                "failure_code": None,
            }
        except ExtractionError as error:
            clip = {
                **base,
                **_scan_manifest_fields(scan),
                "status": "failure",
                "failure": {"code": error.code, "message": str(error)},
                "frames": [],
            }
            qc = {
                "session_id": session_id,
                "status": "failure",
                "scanned_grid_count": len(scan.rows) if scan is not None else 0,
                "valid_candidate_grid_count": (
                    len(scan.valid_candidate_grid_indices) if scan is not None else 0
                ),
                "selected_grid_count": (len(scan.selected_grid_indices) if scan is not None else 0),
                "crop_count": 0,
                "failure_code": error.code,
            }
            _verify_source_snapshot(video_path, source_snapshot, verify_content=True)
        clips[session_id] = clip
        qc_rows.append(qc)
        if progress is not None:
            progress(session_id, str(qc["status"]))
    manifest = {
        "schema_version": "mit-candidate-face-crops-2",
        "generator": dict(generator_provenance),
        "selector": settings,
        "clips": clips,
    }
    qc_summary = {
        "schema_version": "mit-candidate-face-crop-qc-1",
        "session_count": len(qc_rows),
        "succeeded": sum(row["status"] == "success" for row in qc_rows),
        "failed": sum(row["status"] == "failure" for row in qc_rows),
        "sessions": qc_rows,
    }
    return manifest, qc_summary


def _scan_manifest_fields(scan: CandidateScan | None) -> dict[str, Any]:
    if scan is None:
        return {
            "scan": [],
            "active_start_grid_index": None,
            "active_end_grid_index_exclusive": None,
            "active_close_reason": None,
            "valid_candidate_grid_indices": [],
            "selected_grid_indices": [],
        }
    return {
        "scan": [dict(row) for row in scan.rows],
        "active_start_grid_index": scan.active_start_grid_index,
        "active_end_grid_index_exclusive": scan.active_end_grid_index_exclusive,
        "active_close_reason": scan.active_close_reason,
        "valid_candidate_grid_indices": list(scan.valid_candidate_grid_indices),
        "selected_grid_indices": list(scan.selected_grid_indices),
    }


def _lexical_absolute(path: Path) -> Path:
    """Return a normalized absolute path without resolving symlinks."""
    if "\0" in os.fspath(path):
        raise ValueError("filesystem path contains NUL")
    return Path(os.path.abspath(os.fspath(path)))


def _secure_directory_path(path: Path, *, create: bool) -> Path:
    """Walk or create a directory while rejecting every symlink component."""
    if ".." in path.parts:
        raise ValueError("directory path cannot contain parent traversal")
    absolute = _lexical_absolute(path)
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
            raise ValueError("directory path contains a symlink or non-directory component")
    return current


def _secure_regular_source_path(path: Path, *, root: Path | None = None) -> Path:
    """Return a lexical regular-file path after rejecting symlinks and escapes."""
    if ".." in path.parts:
        raise ValueError("source path cannot contain parent traversal")
    absolute = _lexical_absolute(path)
    if root is not None:
        secure_root = _lexical_absolute(root)
        if not absolute.is_relative_to(secure_root):
            raise ValueError("source path escapes its dataset root")
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current = current / component
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("source path contains a symlink component")
    if not stat.S_ISREG(current.lstat().st_mode):
        raise ValueError("source path is not a regular file")
    return current


def _capture_source_snapshot(path: Path) -> _SourceFileSnapshot:
    """Hash one source through a no-follow descriptor and prove pathname stability."""
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as error:
        raise ValueError("source file cannot be opened safely") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("source must be a singly linked regular file")
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if identity_before != identity_after:
            raise ValueError("source file changed while it was read")
    finally:
        os.close(descriptor)
    current = path.stat(follow_symlinks=False)
    if (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
    ) != identity_after:
        raise ValueError("source pathname changed while it was read")
    return _SourceFileSnapshot(*identity_after, digest.hexdigest())


def _verify_source_snapshot(
    path: Path, expected: _SourceFileSnapshot, *, verify_content: bool
) -> None:
    """Fail if a source pathname, inode, metadata or optionally bytes changed."""
    if verify_content:
        observed = _capture_source_snapshot(path)
        if observed != expected:
            raise ValueError("source file changed during candidate-face selection")
        return
    try:
        current = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError("source file changed during candidate-face selection") from error
    if stat.S_ISLNK(current.st_mode) or not stat.S_ISREG(current.st_mode):
        raise ValueError("source file changed during candidate-face selection")
    identity = (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
    if identity != (expected.device, expected.inode, expected.size, expected.mtime_ns):
        raise ValueError("source file changed during candidate-face selection")


def _crop_directory_inventory(root: Path) -> tuple[tuple[str, str], ...]:
    """Return a strict, stable inventory for one private flat crop directory."""
    metadata = root.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise ValueError("crop directory is not a private regular directory")
    rows: list[tuple[str, str]] = []
    with os.scandir(root) as entries:
        for entry in sorted(entries, key=lambda value: value.name):
            if (
                not entry.name.endswith(".png")
                or len(entry.name) != 10
                or not entry.name[:6].isdecimal()
            ):
                raise ValueError("crop directory contains an unexpected entry")
            path = root / entry.name
            file_metadata = path.lstat()
            if (
                stat.S_ISLNK(file_metadata.st_mode)
                or not stat.S_ISREG(file_metadata.st_mode)
                or file_metadata.st_nlink != 1
                or stat.S_IMODE(file_metadata.st_mode) != 0o600
            ):
                raise ValueError("crop output is not one private regular file")
            rows.append((entry.name, _capture_source_snapshot(path).sha256))
    return tuple(rows)


def _publish_crop_directory(stage: Path, destination: Path) -> None:
    """Publish staged crop files without replacing any existing path."""
    expected = _crop_directory_inventory(stage)
    created = False
    destination_identity: tuple[int, int] | None = None
    try:
        try:
            os.mkdir(destination, mode=0o700)
        except FileExistsError:
            pass
        else:
            created = True
            created_stat = destination.lstat()
            if not stat.S_ISDIR(created_stat.st_mode):
                raise ValueError("published crop path was replaced")
            destination_identity = (created_stat.st_dev, created_stat.st_ino)
            source_descriptor = os.open(
                stage, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
            )
            destination_descriptor = os.open(
                destination,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                for filename, _digest in expected:
                    os.link(
                        filename,
                        filename,
                        src_dir_fd=source_descriptor,
                        dst_dir_fd=destination_descriptor,
                        follow_symlinks=False,
                    )
                os.fsync(destination_descriptor)
            finally:
                os.close(destination_descriptor)
                os.close(source_descriptor)
            current = destination.lstat()
            if (current.st_dev, current.st_ino) != destination_identity:
                raise ValueError("published crop directory changed during publication")
            shutil.rmtree(stage)
            _fsync_directory(destination.parent)
        if _crop_directory_inventory(destination) != expected:
            raise ValueError("content-addressed crop directory content differs")
        if stage.exists():
            shutil.rmtree(stage)
            _fsync_directory(destination.parent)
    except (OSError, ValueError) as error:
        if created and destination_identity is not None:
            _remove_claimed_crop_directory(destination, destination_identity)
        raise ExtractionError(
            "face_crop_publish_collision",
            "content-addressed candidate-crop directory collision",
        ) from error


def _remove_claimed_crop_directory(path: Path, expected_identity: tuple[int, int]) -> None:
    """Remove only a failed directory claim whose inode still belongs to this call."""
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(metadata.st_mode) or (metadata.st_dev, metadata.st_ino) != expected_identity:
        return
    with os.scandir(path) as entries:
        children = list(entries)
    for entry in children:
        child = path / entry.name
        child_metadata = child.lstat()
        if stat.S_ISREG(child_metadata.st_mode) and not stat.S_ISLNK(child_metadata.st_mode):
            child.unlink()
        else:
            return
    path.rmdir()
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _detection_centrality(detection: Detection, *, frame_width: int, frame_height: int) -> float:
    centre_x, centre_y = frame_width / 2.0, frame_height / 2.0
    return math.hypot(
        detection.centroid[0] - centre_x, detection.centroid[1] - centre_y
    ) / math.hypot(frame_width, frame_height)


def _detection_sort_key(detection: Detection) -> tuple[float, float, float, float, float]:
    x0, y0, x1, y1 = detection.box_xyxy
    return (-detection.confidence, x0, y0, x1, y1)


def _nearest_source_frame_index(timestamp_seconds: int, frame_rate: Fraction) -> int:
    scaled = Fraction(timestamp_seconds, 1) * frame_rate
    # Exact round-half-up for non-negative timestamps.
    return (2 * scaled.numerator + scaled.denominator) // (2 * scaled.denominator)


def _positive_fraction(value: object, label: str) -> Fraction:
    try:
        result = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{label} is invalid") from error
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _ffprobe_provenance() -> tuple[str, str]:
    try:
        result = subprocess.run(["ffprobe", "-version"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise ExtractionError(
            "face_video_probe_failed", "required FFprobe runtime is unavailable"
        ) from error
    output = result.stdout.replace("\r\n", "\n")
    lines = output.splitlines()
    first = lines[0] if lines else ""
    if not first.startswith("ffprobe version "):
        raise ExtractionError("face_video_probe_failed", "cannot identify FFprobe runtime")
    return first, hashlib.sha256(output.encode("utf-8")).hexdigest()


def _encode_png_rgb(bgr: np.ndarray) -> bytes:
    import cv2

    rgb = cv2.cvtColor(np.asarray(bgr), cv2.COLOR_BGR2RGB)
    with io.BytesIO() as buffer:
        Image.fromarray(rgb, mode="RGB").save(
            buffer, format="PNG", optimize=False, compress_level=9
        )
        return buffer.getvalue()


def _atomic_write(path: Path, content: bytes) -> None:
    if path.name in {"", ".", ".."}:
        raise ValueError("crop output must name one file")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    published_identity: tuple[int, int] | None = None
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise ValueError("crop output path already exists") from error
        metadata = temporary.stat(follow_symlinks=False)
        published_identity = (metadata.st_dev, metadata.st_ino)
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_directory(path.parent)
    metadata = path.stat(follow_symlinks=False)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_nlink != 1
        or published_identity != (metadata.st_dev, metadata.st_ino)
        or _capture_source_snapshot(path).sha256 != hashlib.sha256(content).hexdigest()
    ):
        raise ValueError("crop output changed during publication")


def _decimal_text(value: float) -> str:
    if not math.isfinite(value) or value < 0:
        raise ValueError("decimal value must be finite and non-negative")
    text = format(value, ".12f").rstrip("0").rstrip(".")
    return text or "0"


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )
