from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from .config import resolve_path


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_config = config["optimizer"]
    if optimizer_config.get("type", "adamw") != "adamw":
        raise ValueError("Only optimizer.type=adamw is currently supported.")
    learning_rate = float(optimizer_config["learning_rate"])
    backbone_multiplier = float(optimizer_config.get("backbone_lr_multiplier", 1.0))
    backbone_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (backbone_parameters if name.startswith("backbone.") else head_parameters).append(parameter)
    groups = []
    if backbone_parameters:
        groups.append({"params": backbone_parameters, "lr": learning_rate * backbone_multiplier})
    if head_parameters:
        groups.append({"params": head_parameters, "lr": learning_rate})
    return torch.optim.AdamW(
        groups,
        lr=learning_rate,
        weight_decay=float(optimizer_config.get("weight_decay", 0.01)),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: dict[str, Any]
) -> torch.optim.lr_scheduler.LambdaLR:
    scheduler_config = config.get("scheduler", {"type": "constant"})
    scheduler_type = scheduler_config.get("type", "constant")
    warmup_steps = int(scheduler_config.get("warmup_steps", 0))
    max_steps = int(config["train"]["max_steps"])

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(step, 1) / warmup_steps
        if scheduler_type == "constant":
            return 1.0
        if scheduler_type != "cosine":
            raise ValueError(f"Unknown scheduler.type: {scheduler_type}")
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def output_path(config: dict[str, Any]) -> Path:
    return resolve_path(config, config["train"]["output_dir"])


def save_exported_model(accelerator: Any, model: torch.nn.Module, folder: Path) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        folder.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        accelerator.save(unwrapped.state_dict(), folder / "model.pt")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
