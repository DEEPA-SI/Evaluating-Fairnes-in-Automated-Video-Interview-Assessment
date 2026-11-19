"""Pinned Hugging Face adapters for the three frozen foundation encoders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .extraction import ExtractionError, file_sha256


def verify_cached_weight(
    repository: str,
    revision: str,
    filename: str,
    expected_sha256: str,
    *,
    local_files_only: bool = False,
) -> Path:
    """Fetch one exact artifact file and fail closed on revision/hash drift."""
    from huggingface_hub import hf_hub_download

    try:
        path = Path(
            hf_hub_download(
                repo_id=repository,
                filename=filename,
                revision=revision,
                local_files_only=local_files_only,
            )
        )
    except Exception as error:
        raise ExtractionError(
            "model_artifact_unavailable", f"cannot resolve {repository}@{revision}/{filename}"
        ) from error
    snapshot_parts = path.parts
    if "snapshots" not in snapshot_parts:
        raise ExtractionError("model_revision_drift", "artifact path has no immutable snapshot")
    snapshot_index = snapshot_parts.index("snapshots") + 1
    if snapshot_index >= len(snapshot_parts) or snapshot_parts[snapshot_index] != revision:
        raise ExtractionError("model_revision_drift", "resolved model snapshot does not match pin")
    digest = file_sha256(path)
    if digest != expected_sha256:
        raise ExtractionError(
            "model_weight_hash_mismatch",
            f"{repository}/{filename} expected {expected_sha256}, observed {digest}",
        )
    return path


class Dinov2VideoEncoder:
    """Final-layer CLS adapter for pinned DINOv2 ViT-L/14."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        device: str = "cpu",
        local_files_only: bool = False,
    ) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.name = str(config["name"])
        self.revision = str(config["revision"])
        self.out_dim = int(config["out_dim"])
        self.inference_batch_size = int(config["inference_batch_size"])
        if self.inference_batch_size != 4:
            raise ValueError("DINOv2 inference_batch_size must remain fixed at 4")
        self.weight_file = str(config["weight_file"])
        self.weight_sha256 = str(config["weight_sha256"])
        self.license_id = str(config["license_id"])
        self.license_urls = tuple(str(item) for item in config.get("license_urls", ()))
        self.redistribution_review_required = bool(
            config.get("redistribution_review_required", False)
        )
        self.artifact_files, weight_path = _verify_artifact_files(
            config,
            local_files_only=local_files_only,
        )
        self.weight_size_bytes = weight_path.stat().st_size
        self.cache_snapshot_reference = f"snapshots/{self.revision}/{self.weight_file}"
        self.verified_at_utc = str(config["verified_at_utc"])
        self._processor = AutoImageProcessor.from_pretrained(
            self.name,
            revision=self.revision,
            local_files_only=local_files_only,
        )
        self._model = AutoModel.from_pretrained(
            self.name,
            revision=self.revision,
            use_safetensors=True,
            local_files_only=local_files_only,
        ).to(device)
        self._model.float()
        self._model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)
        self._device = torch.device(device)

    def encode(self, images: Sequence[Any]) -> np.ndarray:
        """Return `last_hidden_state[:, 0, :]` for selected crops."""
        import torch

        selected = list(images)
        if not selected:
            raise ExtractionError("encoder_shape_invalid", "DINOv2 needs at least one crop")
        encoded = []
        for start in range(0, len(selected), self.inference_batch_size):
            inputs = self._processor(
                images=selected[start : start + self.inference_batch_size], return_tensors="pt"
            )
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with torch.inference_mode():
                outputs = self._model(**inputs)
                cls = outputs.last_hidden_state[:, 0, :]
            encoded.append(cls.detach().to(device="cpu", dtype=torch.float32).numpy())
        values = np.concatenate(encoded, axis=0)
        if values.shape[1:] != (self.out_dim,):
            raise ExtractionError("encoder_shape_invalid", "DINOv2 CLS dimension drifted")
        return values


class WavlmAudioEncoder:
    """Final-hidden adapter for pinned WavLM-Base+."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        device: str = "cpu",
        local_files_only: bool = False,
    ) -> None:
        import torch
        from transformers import AutoFeatureExtractor, AutoModel

        self.name = str(config["name"])
        self.revision = str(config["revision"])
        self.out_dim = int(config["out_dim"])
        self.weight_file = str(config["weight_file"])
        self.weight_sha256 = str(config["weight_sha256"])
        self.license_id = str(config["license_id"])
        self.license_urls = tuple(str(item) for item in config.get("license_urls", ()))
        self.redistribution_review_required = bool(
            config.get("redistribution_review_required", False)
        )
        self.sample_rate = int(config["sample_rate"])
        self.conv_stride = int(config["conv_stride_samples"])
        self.receptive_field = int(config["receptive_field_samples"])
        self.artifact_files, weight_path = _verify_artifact_files(
            config,
            local_files_only=local_files_only,
        )
        self.weight_size_bytes = weight_path.stat().st_size
        self.cache_snapshot_reference = f"snapshots/{self.revision}/{self.weight_file}"
        self.verified_at_utc = str(config["verified_at_utc"])
        self._processor = AutoFeatureExtractor.from_pretrained(
            self.name, revision=self.revision, local_files_only=local_files_only
        )
        self._model = AutoModel.from_pretrained(
            self.name, revision=self.revision, local_files_only=local_files_only
        ).to(device)
        self._model.float()
        self._model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)
        self._device = torch.device(device)

    def encode(self, waveform: np.ndarray) -> np.ndarray:
        """Return unpadded final hidden states for a mono 16-kHz waveform."""
        import torch

        inputs = self._processor(
            np.asarray(waveform, dtype=np.float32),
            sampling_rate=self.sample_rate,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(self._device)
        attention_mask = inputs["attention_mask"].to(self._device)
        with torch.inference_mode():
            outputs = self._model(input_values=input_values, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state[0]
        values = hidden.detach().to(device="cpu", dtype=torch.float32).numpy()
        if values.shape[1:] != (self.out_dim,):
            raise ExtractionError("encoder_shape_invalid", "WavLM hidden dimension drifted")
        return values


class MpnetTextEncoder:
    """Tokenizer-aware chunk adapter for pinned all-mpnet-base-v2."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        device: str = "cpu",
        local_files_only: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.name = str(config["name"])
        self.revision = str(config["revision"])
        self.out_dim = int(config["out_dim"])
        self.weight_file = str(config["weight_file"])
        self.weight_sha256 = str(config["weight_sha256"])
        self.license_id = str(config["license_id"])
        self.license_urls = tuple(str(item) for item in config.get("license_urls", ()))
        self.redistribution_review_required = bool(
            config.get("redistribution_review_required", False)
        )
        self.max_sequence_tokens = int(config["max_sequence_tokens"])
        self.artifact_files, weight_path = _verify_artifact_files(
            config,
            local_files_only=local_files_only,
        )
        self.weight_size_bytes = weight_path.stat().st_size
        self.cache_snapshot_reference = f"snapshots/{self.revision}/{self.weight_file}"
        self.verified_at_utc = str(config["verified_at_utc"])
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.name, revision=self.revision, local_files_only=local_files_only
        )
        self._model = AutoModel.from_pretrained(
            self.name,
            revision=self.revision,
            use_safetensors=True,
            local_files_only=local_files_only,
        ).to(device)
        self._model.float()
        self._model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)
        self._device = torch.device(device)

    def tokenize(self, text: str) -> list[int]:
        """Return all tokenizer IDs without specials and without truncation."""
        values = self._tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )["input_ids"]
        return [int(value) for value in values]

    def special_token_count(self) -> int:
        """Return the tokenizer-defined single-sequence special-token count."""
        return int(self._tokenizer.num_special_tokens_to_add(pair=False))

    def encode_token_chunks(self, chunks: Sequence[Sequence[int]]) -> np.ndarray:
        """Apply artifact-equivalent mean pooling and L2 normalisation per chunk."""
        import torch
        import torch.nn.functional as functional

        bos_token_id = self._tokenizer.bos_token_id
        eos_token_id = self._tokenizer.eos_token_id
        if bos_token_id is None or eos_token_id is None:
            raise ExtractionError("tokenizer_contract_invalid", "MPNet BOS/EOS IDs are required")
        rows = []
        for chunk in chunks:
            input_ids = [int(bos_token_id), *(int(value) for value in chunk), int(eos_token_id)]
            if len(input_ids) > self.max_sequence_tokens:
                raise ExtractionError(
                    "tokenizer_contract_invalid", "MPNet chunk exceeds the 384-token contract"
                )
            rows.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
        batch = self._tokenizer.pad(rows, padding=True, return_tensors="pt")
        batch = {key: value.to(self._device) for key, value in batch.items()}
        with torch.inference_mode():
            token_embeddings = self._model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
            pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = functional.normalize(pooled, p=2, dim=1)
        values = pooled.detach().to(device="cpu", dtype=torch.float32).numpy()
        if values.shape[1:] != (self.out_dim,):
            raise ExtractionError("encoder_shape_invalid", "MPNet embedding dimension drifted")
        return values


def _verify_artifact_files(
    config: Mapping[str, Any],
    *,
    local_files_only: bool,
) -> tuple[tuple[dict[str, str | int], ...], Path]:
    """Verify the explicit, cache-independent artifact manifest for one encoder."""
    repository = str(config["name"])
    revision = str(config["revision"])
    weight_file = str(config["weight_file"])
    weight_sha256 = str(config["weight_sha256"])
    configured = config.get("artifact_files")
    if not isinstance(configured, Mapping) or not configured:
        raise ExtractionError(
            "model_artifact_manifest_invalid", "encoder artifact_files must be a non-empty mapping"
        )
    if configured.get(weight_file) != weight_sha256:
        raise ExtractionError(
            "model_artifact_manifest_invalid",
            "artifact_files must contain the frozen weight file and matching SHA-256",
        )

    records: list[dict[str, str | int]] = []
    weight_path: Path | None = None
    for raw_filename in sorted(configured):
        raw_sha256 = configured[raw_filename]
        if not isinstance(raw_filename, str) or not raw_filename or "\0" in raw_filename:
            raise ExtractionError(
                "model_artifact_manifest_invalid", "artifact filename must be a non-empty string"
            )
        filename_path = Path(raw_filename)
        if filename_path.is_absolute() or ".." in filename_path.parts:
            raise ExtractionError(
                "model_artifact_manifest_invalid", "artifact filename must be repository-relative"
            )
        if (
            not isinstance(raw_sha256, str)
            or len(raw_sha256) != 64
            or any(character not in "0123456789abcdef" for character in raw_sha256)
        ):
            raise ExtractionError(
                "model_artifact_manifest_invalid",
                f"artifact SHA-256 is invalid for {raw_filename}",
            )
        path = verify_cached_weight(
            repository,
            revision,
            raw_filename,
            raw_sha256,
            local_files_only=local_files_only,
        )
        records.append(
            {
                "filename": raw_filename,
                "sha256": raw_sha256,
                "size_bytes": path.stat().st_size,
            }
        )
        if raw_filename == weight_file:
            weight_path = path
    if weight_path is None:
        raise ExtractionError(
            "model_artifact_manifest_invalid", "frozen weight file is absent from artifact_files"
        )
    return tuple(records), weight_path
