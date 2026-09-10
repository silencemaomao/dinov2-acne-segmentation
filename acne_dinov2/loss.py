from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class MaskedBCEDiceLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        positive_weight: float | list[float] = 1.0,
        smooth: float = 1.0,
        **_: Any,
    ) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.positive_weight = positive_weight
        self.smooth = float(smooth)

    def pixel_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pos_weight = torch.as_tensor(self.positive_weight, device=logits.device, dtype=logits.dtype)
        if pos_weight.ndim == 1:
            pos_weight = pos_weight.view(1, -1, 1, 1)
        return F.binary_cross_entropy_with_logits(
            logits, target, reduction="none", pos_weight=pos_weight
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        if logits.shape != target.shape:
            raise ValueError(f"logits shape {tuple(logits.shape)} != target shape {tuple(target.shape)}")
        valid = target >= 0
        safe_target = target.clamp(0, 1)
        if not torch.any(valid):
            zero = logits.sum() * 0.0
            return {"total": zero, "pixel": zero.detach(), "dice": zero.detach()}

        per_pixel = self.pixel_loss(logits, safe_target)
        pixel = per_pixel[valid].mean()
        probabilities = torch.sigmoid(logits)
        valid_float = valid.to(logits.dtype)
        reduce_dims = (0, 2, 3)
        intersection = (probabilities * safe_target * valid_float).sum(dim=reduce_dims)
        denominator = ((probabilities + safe_target) * valid_float).sum(dim=reduce_dims)
        dice_per_class = 1.0 - (2.0 * intersection + self.smooth) / (
            denominator + self.smooth
        )
        supervised_classes = valid.sum(dim=reduce_dims) > 0
        dice = dice_per_class[supervised_classes].mean()
        total = self.bce_weight * pixel + self.dice_weight * dice
        return {"total": total, "pixel": pixel.detach(), "dice": dice.detach()}


class MaskedFocalDiceLoss(MaskedBCEDiceLoss):
    def __init__(self, focal_alpha: float = 0.75, focal_gamma: float = 2.0, **kwargs: Any):
        super().__init__(**kwargs)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)

    def pixel_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        probability = torch.sigmoid(logits)
        p_t = probability * target + (1.0 - probability) * (1.0 - target)
        alpha_t = self.focal_alpha * target + (1.0 - self.focal_alpha) * (1.0 - target)
        return alpha_t * (1.0 - p_t).pow(self.focal_gamma) * bce


def build_loss(config: dict[str, Any]) -> nn.Module:
    loss_config = dict(config["loss"])
    loss_type = loss_config.pop("type")
    positive_weight = loss_config.get("positive_weight", 1.0)
    class_names = list(config["data"]["classes"])
    if isinstance(positive_weight, dict):
        missing = [name for name in class_names if name not in positive_weight]
        if missing:
            raise ValueError(f"loss.positive_weight is missing classes: {missing}")
        loss_config["positive_weight"] = [
            float(positive_weight[name]) for name in class_names
        ]
    elif isinstance(positive_weight, list) and len(positive_weight) != len(class_names):
        raise ValueError(
            "loss.positive_weight list length must equal len(data.classes), "
            f"got {len(positive_weight)} and {len(class_names)}."
        )
    if loss_type == "masked_bce_dice":
        return MaskedBCEDiceLoss(**loss_config)
    if loss_type == "masked_focal_dice":
        return MaskedFocalDiceLoss(**loss_config)
    raise ValueError(f"Unknown loss.type: {loss_type}")
