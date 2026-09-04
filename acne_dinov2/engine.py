from __future__ import annotations

from typing import Any

import torch

from .metrics import confusion_counts, metrics_from_counts


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    dataloader: Any,
    criterion: torch.nn.Module,
    accelerator: Any,
    threshold: float,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    counts = torch.zeros(4, dtype=torch.float64, device=accelerator.device)
    loss_sum = torch.zeros(1, dtype=torch.float64, device=accelerator.device)
    sample_count = torch.zeros(1, dtype=torch.float64, device=accelerator.device)

    for batch in dataloader:
        logits = model(batch["image"])
        losses = criterion(logits, batch["target"])
        batch_size = batch["image"].shape[0]
        counts += confusion_counts(logits, batch["target"], threshold)
        loss_sum += losses["total"].detach().double() * batch_size
        sample_count += batch_size

    counts = accelerator.reduce(counts, reduction="sum")
    loss_sum = accelerator.reduce(loss_sum, reduction="sum")
    sample_count = accelerator.reduce(sample_count, reduction="sum")
    metrics = metrics_from_counts(counts.cpu())
    metrics["loss"] = float((loss_sum / sample_count.clamp_min(1)).item())
    if was_training:
        model.train()
    return metrics
