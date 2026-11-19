"""AVI-Personality participant-level feature pooling.

This module contains outcome-blind numerical operations only.
It must not import participant outcomes or protected audit attributes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

EXPECTED_CLIPS = 6


def pool_participant_clips(
    clip_features: Mapping[str, np.ndarray],
    expected_questions: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Pool available clip vectors into one participant representation.

    Returns the mean feature vector and a six-position availability mask.
    Missing clips are excluded rather than replaced with zero vectors.
    """
    questions = tuple(expected_questions)

    if len(questions) != EXPECTED_CLIPS or len(set(questions)) != EXPECTED_CLIPS:
        raise ValueError("Exactly six unique questions are required")

    if set(clip_features) - set(questions):
        raise ValueError("Unexpected question identifiers")

    vectors = []
    mask = np.zeros(EXPECTED_CLIPS, dtype=np.bool_)
    feature_dim = None

    for index, question in enumerate(questions):
        if question not in clip_features:
            continue

        vector = np.asarray(clip_features[question], dtype=np.float32)

        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("Clip features must be nonempty one-dimensional vectors")

        if not np.isfinite(vector).all():
            raise ValueError("Clip features contain non-finite values")

        if feature_dim is None:
            feature_dim = vector.size
        elif vector.size != feature_dim:
            raise ValueError("Clip feature dimensions are inconsistent")

        vectors.append(vector)
        mask[index] = True

    if not vectors:
        raise ValueError("No valid clip features available for participant")

    pooled = np.stack(vectors).mean(axis=0, dtype=np.float32)

    return pooled, mask


AVI_AUDIO_SAMPLE_RATE = 16_000


def decode_avi_audio(media_path: str | __import__("pathlib").Path) -> np.ndarray:
    """Decode AVI clip audio as mono 16-kHz float32 PCM using FFmpeg."""
    from pathlib import Path

    from src.features.extraction import decode_audio_ffmpeg

    waveform = decode_audio_ffmpeg(
        Path(media_path),
        sample_rate=AVI_AUDIO_SAMPLE_RATE,
    )

    values = np.asarray(waveform, dtype=np.float32)

    if values.ndim != 1 or values.size == 0:
        raise ValueError("AVI audio must be a nonempty mono waveform")

    if not np.isfinite(values).all():
        raise ValueError("AVI audio contains non-finite samples")

    return values


AVI_VIDEO_SAMPLE_FPS = 1
AVI_MAX_VIDEO_FRAMES = 128


def select_avi_frame_seconds(duration_seconds: float) -> tuple[int, ...]:
    """Select deterministic integer-second frame timestamps for an AVI clip."""
    import math

    from src.features.face_selector import bounded_valid_grid_indices

    duration = float(duration_seconds)

    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("AVI video duration must be finite and positive")

    frame_count = max(1, math.floor(duration))
    grid = range(frame_count)

    return bounded_valid_grid_indices(
        grid,
        maximum=AVI_MAX_VIDEO_FRAMES,
    )


def decode_avi_frames(
    video_path: str | __import__("pathlib").Path,
    frame_seconds: Sequence[int],
    *,
    expected_fps: int = 30,
) -> tuple[np.ndarray, ...]:
    """Decode deterministic one-second video frames for a verified CFR AVI clip."""
    from pathlib import Path

    import cv2

    requested = tuple(frame_seconds)

    if not requested or any(
        isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in requested
    ):
        raise ValueError("Frame seconds must be nonempty nonnegative integers")

    if requested != tuple(sorted(set(requested))):
        raise ValueError("Frame seconds must be strictly increasing and unique")

    if expected_fps <= 0:
        raise ValueError("Expected FPS must be positive")

    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)

    capture = cv2.VideoCapture(str(Path(video_path)))

    if not capture.isOpened():
        raise ValueError("Cannot open AVI video")

    frames = []

    try:
        actual_fps = float(capture.get(cv2.CAP_PROP_FPS))

        if not np.isfinite(actual_fps) or abs(actual_fps - expected_fps) > 0.01:
            raise ValueError("AVI video frame rate differs from verified contract")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise ValueError("AVI video has no decodable frames")

        for second in requested:
            frame_index = second * expected_fps

            if frame_index >= frame_count:
                raise ValueError("Requested frame exceeds video frame count")

            if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
                raise ValueError("Cannot seek requested AVI frame")

            success, frame = capture.read()

            if not success or frame is None:
                raise ValueError("Cannot decode requested AVI frame")

            image = np.asarray(frame)

            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("Decoded AVI frame must have three colour channels")

            if image.dtype != np.uint8:
                raise ValueError("Decoded AVI frame must use uint8 pixels")

            frames.append(image.copy())

    finally:
        capture.release()

    return tuple(frames)


AVI_VIDEO_EMBEDDING_DIM = 1024


def pool_avi_video_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Mean-pool DINOv2 CLS embeddings over selected AVI video frames."""
    values = np.asarray(embeddings, dtype=np.float32)

    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != AVI_VIDEO_EMBEDDING_DIM:
        raise ValueError("Expected nonempty N x 1024 DINOv2 frame embeddings")

    if not np.isfinite(values).all():
        raise ValueError("AVI video embeddings contain non-finite values")

    return values.mean(axis=0, dtype=np.float32)


def extract_avi_video_embedding(
    video_path,
    model,
    processor,
) -> np.ndarray:
    """Extract one 1024-dimensional AVI DINOv2 embedding using batches of four."""
    import json
    import subprocess
    from pathlib import Path

    import torch
    from PIL import Image

    video_path = Path(video_path)

    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    duration = float(json.loads(result.stdout)["format"]["duration"])
    timestamps = select_avi_frame_seconds(duration)

    if not timestamps:
        raise ValueError("No AVI video frames selected")

    model.eval()
    embeddings = []
    batch_size = 4

    for start in range(0, len(timestamps), batch_size):
        batch_timestamps = timestamps[start : start + batch_size]
        images = []

        frames = decode_avi_frames(video_path, batch_timestamps)

        if len(frames) != len(batch_timestamps):
            raise ValueError("AVI frame decoding count mismatch")

        for frame in frames:
            rgb = Image.fromarray(
                frame[:, :, ::-1].copy(),
                mode="RGB",
            )
            images.append(rgb)

        inputs = processor(images=images, return_tensors="pt")

        with torch.inference_mode():
            output = model(**inputs)
            cls = output.last_hidden_state[:, 0, :]

        batch_embeddings = cls.detach().cpu().numpy().astype(np.float32)

        expected_shape = (len(images), AVI_VIDEO_EMBEDDING_DIM)

        if batch_embeddings.shape != expected_shape:
            raise ValueError("Unexpected DINOv2 embedding dimensions")

        if not np.isfinite(batch_embeddings).all():
            raise ValueError("Non-finite DINOv2 embedding")

        embeddings.append(batch_embeddings)

    return pool_avi_video_embeddings(np.concatenate(embeddings, axis=0))


AVI_AUDIO_EMBEDDING_DIM = 1536


def extract_avi_audio_embedding(
    media_path,
    encoder,
) -> np.ndarray:
    """Extract a 1536-dimensional AVI audio embedding using bounded WavLM windows."""
    from src.features.extraction import pool_wavlm_seconds

    waveform = decode_avi_audio(media_path)
    window_samples = 20 * AVI_AUDIO_SAMPLE_RATE
    pooled_parts = []

    for start in range(0, waveform.size, window_samples):
        segment = waveform[start : start + window_samples]

        # WavLM's convolutional feature extractor requires at least
        # 400 samples at 16 kHz. Ignore shorter residual fragments.
        if segment.size < 400:
            continue

        hidden = np.asarray(encoder.encode(segment), dtype=np.float32)

        if (
            hidden.ndim != 2
            or hidden.shape[0] == 0
            or hidden.shape[1] != 768
            or not np.isfinite(hidden).all()
        ):
            raise ValueError("Invalid WavLM hidden-state representation")

        pooled, _ = pool_wavlm_seconds(
            hidden,
            sample_count=int(segment.size),
            sample_rate=AVI_AUDIO_SAMPLE_RATE,
            conv_stride=320,
            receptive_field=400,
        )

        if (
            pooled.ndim != 2
            or pooled.shape[0] == 0
            or pooled.shape[1] != AVI_AUDIO_EMBEDDING_DIM
            or not np.isfinite(pooled).all()
        ):
            raise ValueError("Invalid WavLM temporal pooling output")

        pooled_parts.append(pooled)

    if not pooled_parts:
        raise ValueError("No valid audio intervals")

    embedding = np.concatenate(pooled_parts, axis=0).mean(axis=0, dtype=np.float32)

    if embedding.shape != (AVI_AUDIO_EMBEDDING_DIM,) or not np.isfinite(embedding).all():
        raise ValueError("Invalid AVI audio embedding")

    return embedding


def extract_avi_participant_multimodal(
    question_media,
    expected_questions,
    video_model,
    video_processor,
    audio_encoder,
):
    """Extract and pool outcome-blind AVI participant video/audio features."""
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    if not isinstance(question_media, Mapping):
        raise TypeError("question_media must map question IDs to media paths")

    if (
        not isinstance(expected_questions, Sequence)
        or isinstance(expected_questions, (str, bytes))
        or len(expected_questions) != 6
        or len(set(expected_questions)) != 6
    ):
        raise ValueError("Exactly six unique expected questions are required")

    unexpected = set(question_media) - set(expected_questions)
    if unexpected:
        raise ValueError("Unexpected question IDs in participant media")

    video_features = {}
    audio_features = {}

    for question in expected_questions:
        media_path = question_media.get(question)

        if media_path is None:
            continue

        media_path = Path(media_path)

        if not media_path.is_file():
            raise FileNotFoundError(f"Media unavailable for question {question}")

        video_features[question] = extract_avi_video_embedding(
            media_path,
            video_model,
            video_processor,
        )

        audio_features[question] = extract_avi_audio_embedding(
            media_path,
            audio_encoder,
        )

    video_embedding, video_mask = pool_participant_clips(
        video_features,
        expected_questions,
    )

    audio_embedding, audio_mask = pool_participant_clips(
        audio_features,
        expected_questions,
    )

    if video_embedding.shape != (1024,):
        raise ValueError("Invalid participant video embedding")

    if audio_embedding.shape != (1536,):
        raise ValueError("Invalid participant audio embedding")

    if (
        video_mask.shape != (6,)
        or audio_mask.shape != (6,)
        or not np.array_equal(video_mask, audio_mask)
        or not np.isfinite(video_embedding).all()
        or not np.isfinite(audio_embedding).all()
    ):
        raise ValueError("Invalid participant multimodal representation")

    return {
        "video": video_embedding.astype(np.float32, copy=False),
        "audio": audio_embedding.astype(np.float32, copy=False),
        "video_mask": video_mask,
        "audio_mask": audio_mask,
    }
