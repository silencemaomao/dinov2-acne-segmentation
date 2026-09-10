from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


CLASSES = ("acne", "pigmentation", "scar")


def create_sample(image_path: Path, label_root: Path, seed: int, index: int) -> None:
    rng = np.random.default_rng(seed)
    image_array = np.full((64, 64, 3), [188, 142, 122], dtype=np.int16)
    image_array += rng.normal(0, 6, image_array.shape).astype(np.int16)
    image = Image.fromarray(np.uint8(np.clip(image_array, 0, 255)))
    image_draw = ImageDraw.Draw(image)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    for class_index, class_name in enumerate(CLASSES):
        label = Image.new("L", (64, 64), 0)
        label_draw = ImageDraw.Draw(label)
        label_draw.rectangle((class_index * 3, 0, class_index * 3 + 2, 63), fill=128)
        positive = index % (class_index + 2) == 0
        if positive:
            x, y = int(rng.integers(14, 50)), int(rng.integers(14, 50))
            radius = int(rng.integers(3, 7))
            color = [(205, 65, 65), (135, 95, 70), (190, 125, 115)][class_index]
            image_draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius), fill=color
            )
            label_draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
        label_path = label_root / class_name / f"sample_{index:03d}.png"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label.save(label_path)
    image.save(image_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("smoke_data"))
    args = parser.parse_args()
    counts = {"train": 8, "val": 4, "test": 4}
    for split, count in counts.items():
        for index in range(count):
            create_sample(
                args.output_dir / split / "images" / f"sample_{index:03d}.jpg",
                args.output_dir / split / "labels",
                seed=index + {"train": 0, "val": 100, "test": 200}[split],
                index=index,
            )
    print(f"Created smoke-test dataset under {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
