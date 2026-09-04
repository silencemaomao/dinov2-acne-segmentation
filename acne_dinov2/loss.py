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
        positive_weight: float = 1.0,
        smooth: float = 1.0,
        **_: Any,
    ) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.positive_weight = float(positive_weight)
        self.smooth = float(smooth)

    def pixel_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pos_weight = torch.as_tensor(self.positive_weight, device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(
            logits, target, reduction="none", pos_weight=pos_weight
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = logits.squeeze(1)
        valid = target >= 0
        safe_target = target.clamp(0, 1)
        if not torch.any(valid):
            zero = logits.sum() * 0.0
            return {"total": zero, "pixel": zero.detach(), "dice": zero.detach()}

        per_pixel = self.pixel_loss(logits, safe_target)
        pixel = per_pixel[valid].mean()
        probabilities = torch.sigmoid(logits)
        valid_float = valid.to(logits.dtype)
        intersection = (probabilities * safe_target * valid_float).sum()
        denominator = ((probabilities + safe_target) * valid_float).sum()
        dice = 1.0 - (2.0 * intersection + self.smooth) / (denominator + self.smooth)
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
    if loss_type == "masked_bce_dice":
        return MaskedBCEDiceLoss(**loss_config)
    if loss_type == "masked_focal_dice":
        return MaskedFocalDiceLoss(**loss_config)
    raise ValueError(f"Unknown loss.type: {loss_type}")
