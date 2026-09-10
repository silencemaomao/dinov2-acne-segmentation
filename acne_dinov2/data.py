from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .config import resolve_path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LABEL_VALUES = {0, 128, 255}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def _relative_key(path: Path, root: Path) -> str:
    return path.relative_to(root).with_suffix("").as_posix().lower()


def _index_files(root: Path, kind: str) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"{kind} directory does not exist: {root}")
    index: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            key = _relative_key(path, root)
            if key in index:
                raise ValueError(f"Duplicate {kind} key {key!r}: {index[key]} and {path}")
            index[key] = path
    if not index:
        raise FileNotFoundError(f"No supported {kind} files found under {root}")
    return index


class AcneHeatmapDataset(Dataset[dict[str, Any]]):
    """Paired RGB images and 0/128/255 heatmaps."""

    def __init__(
        self,
        input_dir: str | Path,
        label_dir: str | Path,
        classes: list[str],
        image_size: tuple[int, int],
        horizontal_flip: float = 0.0,
        strict_labels: bool = True,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.label_dir = Path(label_dir)
        self.classes = [str(name) for name in classes]
        if not self.classes or len(set(self.classes)) != len(self.classes):
            raise ValueError("data.classes must contain unique, non-empty class names.")
        self.image_size = tuple(int(value) for value in image_size)
        self.horizontal_flip = float(horizontal_flip)
        self.strict_labels = bool(strict_labels)

        images = _index_files(self.input_dir, "input image")
        labels_by_class: dict[str, dict[str, Path]] = {}
        for class_name in self.classes:
            class_labels = _index_files(
                self.label_dir / class_name, f"label for class {class_name}"
            )
            missing = sorted(set(images) - set(class_labels))
            extra = sorted(set(class_labels) - set(images))
            if missing:
                raise FileNotFoundError(
                    f"Class {class_name!r} is missing masks for {len(missing)} images, e.g. {missing[:3]}"
                )
            if extra:
                raise ValueError(f"Class {class_name!r} has masks without images: {extra[:3]}")
            labels_by_class[class_name] = class_labels
        self.samples = [
            (
                key,
                images[key],
                {class_name: labels_by_class[class_name][key] for class_name in self.classes},
            )
            for key in sorted(images)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def _read_label(self, path: Path) -> tuple[Image.Image, np.ndarray]:
        label_image = Image.open(path).convert("L")
        raw = np.asarray(label_image, dtype=np.uint8)
        unique = set(np.unique(raw).tolist())
        if self.strict_labels and not unique.issubset(LABEL_VALUES):
            invalid = sorted(unique - LABEL_VALUES)
            raise ValueError(
                f"Label {path} contains invalid values {invalid[:20]}; expected only 0, 128, 255. "
                "Save labels as lossless PNG and resize with nearest interpolation."
            )
        return label_image, raw

    def has_positive(self, index: int) -> bool:
        _, _, label_paths = self.samples[index]
        return any(
            np.any(self._read_label(label_paths[class_name])[1] == 255)
            for class_name in self.classes
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        key, image_path, label_paths = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size
        label_images = [
            self._read_label(label_paths[class_name])[0] for class_name in self.classes
        ]

        height, width = self.image_size
        image = image.resize((width, height), Image.Resampling.BICUBIC)
        label_images = [
            label.resize((width, height), Image.Resampling.NEAREST) for label in label_images
        ]
        should_flip = self.horizontal_flip > 0 and random.random() < self.horizontal_flip
        if should_flip:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            label_images = [
                label.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for label in label_images
            ]

        image_array = np.asarray(image, dtype=np.float32).copy() / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        image_tensor = (image_tensor - IMAGENET_MEAN) / IMAGENET_STD

        targets = []
        for label_image in label_images:
            label_array = np.asarray(label_image, dtype=np.uint8).copy()
            target = torch.full((height, width), -1.0, dtype=torch.float32)
            target[torch.from_numpy(label_array == 0)] = 0.0
            target[torch.from_numpy(label_array == 255)] = 1.0
            targets.append(target)
        return {
            "image": image_tensor,
            "target": torch.stack(targets),
            "name": key,
            "original_size": torch.tensor([original_height, original_width], dtype=torch.int64),
        }


def build_dataset(config: dict[str, Any], split: str) -> AcneHeatmapDataset:
    data_config = config["data"]
    if split not in data_config:
        raise ValueError(f"YAML has no data.{split} section.")
    split_config = data_config[split]
    return AcneHeatmapDataset(
        input_dir=resolve_path(config, split_config["input_dir"]),
        label_dir=resolve_path(config, split_config["label_dir"]),
        classes=list(data_config["classes"]),
        image_size=tuple(data_config["image_size"]),
        horizontal_flip=split_config.get("horizontal_flip", 0.0) if split == "train" else 0.0,
        strict_labels=data_config.get("strict_labels", True),
    )


def build_dataloader(config: dict[str, Any], split: str) -> DataLoader:
    dataset = build_dataset(config, split)
    loader_config = config["dataloader"]
    is_train = split == "train"
    loader_type = loader_config.get("type", "standard")
    sampler = None
    shuffle = is_train

    if is_train and loader_type == "weighted":
        positive_weight = float(loader_config.get("positive_sample_weight", 2.0))
        weights = [positive_weight if dataset.has_positive(i) else 1.0 for i in range(len(dataset))]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False

    workers = int(loader_config.get("num_workers", 4))
    return DataLoader(
        dataset,
        batch_size=int(
            loader_config.get("train_batch_size", 4)
            if is_train
            else loader_config.get("eval_batch_size", 4)
        ),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=workers,
        pin_memory=bool(loader_config.get("pin_memory", True)),
        persistent_workers=bool(loader_config.get("persistent_workers", True)) and workers > 0,
        drop_last=is_train and len(dataset) > int(loader_config.get("train_batch_size", 4)),
    )
