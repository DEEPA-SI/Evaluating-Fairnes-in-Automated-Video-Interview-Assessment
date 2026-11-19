"""Low-capacity common scorer for 69 independent MIT participants."""

from __future__ import annotations

import torch
import torch.nn as nn


class _ModalityBlock(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 8)
        self.norm = nn.LayerNorm(8)
        self.activation = nn.GELU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.linear(value)))


class LowCapacityMultimodalScorer(nn.Module):
    """Project three train-standardised summaries to 8D and predict one score.

    Train-only per-modality standardisation occurs in the data pipeline and is persisted in the
    run manifest.  This module has exactly 26,721 trainable parameters.
    """

    EXPECTED_TRAINABLE_PARAMETERS = 26_721

    def __init__(self) -> None:
        super().__init__()
        self.video = _ModalityBlock(1024)
        self.audio = _ModalityBlock(1536)
        self.text = _ModalityBlock(768)
        self.head = nn.Linear(24, 1)
        self.reset_parameters()
        if self.trainable_parameter_count != self.EXPECTED_TRAINABLE_PARAMETERS:
            raise RuntimeError("common scorer parameter contract drifted")

    def reset_parameters(self) -> None:
        """Apply the frozen Xavier/zero and LayerNorm one/zero initialisation."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @property
    def trainable_parameter_count(self) -> int:
        """Return the exact number of trainable scalar parameters."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(
        self,
        video_mean: torch.Tensor,
        audio_mean: torch.Tensor,
        text_mean: torch.Tensor,
    ) -> torch.Tensor:
        """Return one normalised crowd-perceived-hireability score per row."""
        fused = self.fused_representation(video_mean, audio_mean, text_mean)
        return torch.sigmoid(self.head(fused))

    def fused_representation(
        self,
        video_mean: torch.Tensor,
        audio_mean: torch.Tensor,
        text_mean: torch.Tensor,
    ) -> torch.Tensor:
        """Return the exact 24-dimensional pre-head representation.

        ARL may consume a detached copy of this value during training.  Exposing it here leaves
        the scorer architecture, parameter names and prediction calculation unchanged.
        """
        if video_mean.ndim != 2 or video_mean.shape[1] != 1024:
            raise ValueError("video_mean must have shape (batch,1024)")
        if audio_mean.ndim != 2 or audio_mean.shape[1] != 1536:
            raise ValueError("audio_mean must have shape (batch,1536)")
        if text_mean.ndim != 2 or text_mean.shape[1] != 768:
            raise ValueError("text_mean must have shape (batch,768)")
        if len({video_mean.shape[0], audio_mean.shape[0], text_mean.shape[0]}) != 1:
            raise ValueError("all modality summaries must have the same batch size")
        return torch.cat(
            (self.video(video_mean), self.audio(audio_mean), self.text(text_mean)), dim=1
        )
