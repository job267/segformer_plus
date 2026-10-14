from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.config import load_config


@pytest.fixture
def tiny_camvid(tmp_path):
    config = load_config()
    config["dataset"]["root"] = str(tmp_path)
    config["dataset"]["expected_counts"] = {"train": 2, "val": 1, "test": 1}
    config["preprocessing"]["size"] = [32, 32]
    for split, count in config["dataset"]["expected_counts"].items():
        entries = []
        for index in range(count):
            stem = f"{split}_{index}"
            image_relative = f"images/{stem}.png"
            mask_relative = f"masks/{stem}_L.png"
            image_path, mask_path = tmp_path / image_relative, tmp_path / mask_relative
            image_path.parent.mkdir(exist_ok=True)
            mask_path.parent.mkdir(exist_ok=True)
            seed = sum(map(ord, stem))
            pixels = np.random.default_rng(seed).integers(0, 256, (32, 48, 3), dtype=np.uint8)
            labels = np.zeros((32, 48), dtype=np.uint8)
            labels[:, 16:32], labels[:, 32:] = 3, 8
            labels[0, 0] = 255
            Image.fromarray(pixels).save(image_path)
            Image.fromarray(labels).save(mask_path)
            entries.append(f"{image_relative} {mask_relative}")
        (tmp_path / config["dataset"]["splits"][split]).write_text(
            "\n".join(entries) + "\n", encoding="utf-8"
        )
    return config
