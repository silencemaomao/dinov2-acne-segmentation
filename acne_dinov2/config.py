from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"YAML root must be a mapping: {config_path}")
    _validate_config(config)
    config["_config_path"] = str(config_path)
    return config


def _validate_config(config: dict[str, Any]) -> None:
    required = ["data", "dataloader", "model", "loss", "optimizer", "accelerator"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing required YAML sections: {', '.join(missing)}")

    image_size = config["data"].get("image_size")
    if not isinstance(image_size, list) or len(image_size) != 2:
        raise ValueError("data.image_size must be [height, width].")
    if any(int(value) <= 0 for value in image_size):
        raise ValueError("data.image_size values must be positive.")

    for split in ("train", "val", "test"):
        if split in config["data"]:
            for key in ("input_dir", "label_dir"):
                if not config["data"][split].get(key):
                    raise ValueError(f"data.{split}.{key} is required.")

    loader_type = config["dataloader"].get("type")
    if loader_type not in {"standard", "weighted"}:
        raise ValueError("dataloader.type must be 'standard' or 'weighted'.")

    precision = str(config["accelerator"].get("mixed_precision", "no"))
    if precision not in {"no", "fp16", "bf16"}:
        raise ValueError("accelerator.mixed_precision must be no, fp16, or bf16.")


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    project_root = Path(config["_config_path"]).parent.parent
    return (project_root / path).resolve()
