from __future__ import annotations

import torch


def confusion_counts(
    logits: torch.Tensor, target: torch.Tensor, threshold: float
) -> torch.Tensor:
    if logits.shape != target.shape:
        raise ValueError(f"logits shape {tuple(logits.shape)} != target shape {tuple(target.shape)}")
    predictions = torch.sigmoid(logits) >= threshold
    valid = target >= 0
    positive = target == 1
    negative = target == 0
    reduce_dims = (0, 2, 3)
    true_positive = (predictions & positive & valid).sum(dim=reduce_dims)
    false_positive = (predictions & negative & valid).sum(dim=reduce_dims)
    false_negative = ((~predictions) & positive & valid).sum(dim=reduce_dims)
    true_negative = ((~predictions) & negative & valid).sum(dim=reduce_dims)
    return torch.stack(
        [true_positive, false_positive, false_negative, true_negative], dim=1
    ).double()


def _binary_metrics(counts: torch.Tensor) -> dict[str, float]:
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


def metrics_from_counts(counts: torch.Tensor, class_names: list[str]) -> dict[str, float]:
    if counts.shape != (len(class_names), 4):
        raise ValueError(
            f"Expected counts shape {(len(class_names), 4)}, received {tuple(counts.shape)}"
        )
    per_class = [_binary_metrics(counts[index]) for index in range(len(class_names))]
    result: dict[str, float] = {}
    for class_name, values in zip(class_names, per_class, strict=True):
        for metric_name, value in values.items():
            result[f"class/{class_name}/{metric_name}"] = value
    for metric_name in ("precision", "recall", "dice", "f1", "iou", "accuracy"):
        result[f"macro_{metric_name}"] = sum(
            values[metric_name] for values in per_class
        ) / len(per_class)
    micro = _binary_metrics(counts.sum(dim=0))
    for metric_name, value in micro.items():
        result[f"micro_{metric_name}"] = value
    return result
