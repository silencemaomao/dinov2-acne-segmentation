from __future__ import annotations

import torch


def confusion_counts(
    logits: torch.Tensor, target: torch.Tensor, threshold: float
) -> torch.Tensor:
    predictions = torch.sigmoid(logits.squeeze(1)) >= threshold
    valid = target >= 0
    positive = target == 1
    negative = target == 0
    true_positive = (predictions & positive & valid).sum()
    false_positive = (predictions & negative & valid).sum()
    false_negative = ((~predictions) & positive & valid).sum()
    true_negative = ((~predictions) & negative & valid).sum()
    return torch.stack([true_positive, false_positive, false_negative, true_negative]).double()


def metrics_from_counts(counts: torch.Tensor) -> dict[str, float]:
    tp, fp, fn, tn = [float(value) for value in counts.tolist()]
    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = 2.0 * tp / (2.0 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)
    return {
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "f1": dice,
        "iou": iou,
        "accuracy": accuracy,
    }
