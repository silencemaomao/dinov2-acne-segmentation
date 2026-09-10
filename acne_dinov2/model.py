from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class DINOv2Segmenter(nn.Module):
    def __init__(
        self,
        pretrained_name: str = "facebook/dinov2-small",
        hidden_indices: list[int] | None = None,
        decoder_channels: int = 256,
        num_classes: int = 1,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        gradient_checkpointing: bool = False,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(
            pretrained_name, local_files_only=local_files_only
        )
        self.hidden_indices = hidden_indices or [-1, -3, -5, -7]
        hidden_size = int(self.backbone.config.hidden_size)
        self.patch_size = self.backbone.config.patch_size
        self.projections = nn.ModuleList(
            [nn.Linear(hidden_size, decoder_channels) for _ in self.hidden_indices]
        )
        fused_channels = decoder_channels * len(self.hidden_indices)
        self.decoder = nn.Sequential(
            nn.Conv2d(fused_channels, decoder_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(decoder_channels, decoder_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels // 2),
            nn.GELU(),
            nn.Conv2d(decoder_channels // 2, num_classes, 1),
        )
        if freeze_backbone:
            self.backbone.requires_grad_(False)
        elif gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

    def _patch_hw(self, images: torch.Tensor) -> tuple[int, int]:
        patch = self.patch_size
        if isinstance(patch, (tuple, list)):
            patch_h, patch_w = int(patch[0]), int(patch[1])
        else:
            patch_h = patch_w = int(patch)
        height, width = images.shape[-2:]
        if height % patch_h or width % patch_w:
            raise ValueError(
                f"Input {(height, width)} must be divisible by DINOv2 patch size {(patch_h, patch_w)}."
            )
        return height // patch_h, width // patch_w

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.backbone(pixel_values=images, output_hidden_states=True, return_dict=True)
        if output.hidden_states is None:
            raise RuntimeError("DINOv2 did not return hidden states.")
        grid_h, grid_w = self._patch_hw(images)
        patch_count = grid_h * grid_w
        maps = []
        for hidden_index, projection in zip(self.hidden_indices, self.projections, strict=True):
            tokens = output.hidden_states[hidden_index]
            # Taking the final patch_count tokens also supports backbones with register tokens.
            patch_tokens = tokens[:, -patch_count:, :]
            projected = projection(patch_tokens)
            feature_map = projected.transpose(1, 2).reshape(
                images.shape[0], -1, grid_h, grid_w
            )
            maps.append(feature_map)
        logits = self.decoder(torch.cat(maps, dim=1))
        return F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)


class TinySegmenter(nn.Module):
    """Small offline model used only to smoke-test the complete pipeline."""

    def __init__(self, channels: int = 16, num_classes: int = 1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, num_classes, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


def build_model(config: dict[str, Any]) -> nn.Module:
    model_config = dict(config["model"])
    model_type = model_config.pop("type")
    class_count = len(config["data"]["classes"])
    configured_count = model_config.pop("num_classes", None)
    if configured_count is not None and int(configured_count) != class_count:
        raise ValueError(
            f"model.num_classes={configured_count} does not match "
            f"len(data.classes)={class_count}. Use null for automatic inference."
        )
    model_config["num_classes"] = class_count
    if model_type == "dinov2_segmenter":
        return DINOv2Segmenter(**model_config)
    if model_type == "tiny_segmenter":
        return TinySegmenter(**model_config)
    raise ValueError(f"Unknown model.type: {model_type}")
