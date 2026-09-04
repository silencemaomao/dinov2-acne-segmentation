from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def create_sample(image_path: Path, label_path: Path, seed: int, positive: bool) -> None:
    rng = np.random.default_rng(seed)
    image_array = np.full((64, 64, 3), [188, 142, 122], dtype=np.int16)
    image_array += rng.normal(0, 6, image_array.shape).astype(np.int16)
    image = Image.fromarray(np.uint8(np.clip(image_array, 0, 255)))
    label = Image.new("L", (64, 64), 0)
    image_draw = ImageDraw.Draw(image)
    label_draw = ImageDraw.Draw(label)
    if positive:
        x, y = int(rng.integers(14, 50)), int(rng.integers(14, 50))
        radius = int(rng.integers(3, 7))
        image_draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(205, 65, 65))
        label_draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
        label_draw.rectangle((0, 0, 5, 63), fill=128)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)
    label.save(label_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("smoke_data"))
    args = parser.parse_args()
    counts = {"train": 8, "val": 4, "test": 4}
    for split, count in counts.items():
        for index in range(count):
            create_sample(
                args.output_dir / split / "images" / f"sample_{index:03d}.jpg",
                args.output_dir / split / "labels" / f"sample_{index:03d}.png",
                seed=index + {"train": 0, "val": 100, "test": 200}[split],
                positive=index % 2 == 0,
            )
    print(f"Created smoke-test dataset under {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
