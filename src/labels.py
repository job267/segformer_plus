"""CamVid 11-class IDs, verified against the pinned lih627/CamVid conversion."""

import numpy as np
from PIL import Image

CLASS_NAMES = (
    "Sky", "Building", "Pole", "Road", "Sidewalk", "Tree",
    "SignSymbol", "Fence", "Car", "Pedestrian", "Bicyclist",
)
PALETTE = np.array(
    [
        (128, 128, 128), (128, 0, 0), (192, 192, 128), (128, 64, 128),
        (0, 0, 192), (128, 128, 0), (192, 128, 128), (64, 64, 128),
        (64, 0, 128), (64, 64, 0), (0, 128, 192),
    ],
    dtype=np.uint8,
)


def validate_labels(mask: np.ndarray, num_classes: int = 11, ignore_index: int = 255) -> None:
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("An index mask must be a two-dimensional integer array")
    legal = ((mask >= 0) & (mask < num_classes)) | (mask == ignore_index)
    if not legal.all():
        raise ValueError(f"Illegal label IDs: {np.unique(mask[~legal]).tolist()}")


def decode_mask(
    image: Image.Image, encoding: str = "index", num_classes: int = 11, ignore_index: int = 255
) -> np.ndarray:
    if encoding == "index":
        # Palette-mode PNGs retain stored indices. Never convert them to grayscale.
        mask = np.array(image)
    elif encoding == "rgb":
        rgb = np.array(image.convert("RGB"))
        mask = np.full(rgb.shape[:2], -1, dtype=np.int64)
        for class_id, color in enumerate(PALETTE):
            mask[np.all(rgb == color, axis=-1)] = class_id
        mask[np.all(rgb == (0, 0, 0), axis=-1)] = ignore_index
        if (mask < 0).any():
            unknown = np.unique(rgb[mask < 0], axis=0)[:10].tolist()
            raise ValueError(f"Unknown RGB labels (first 10): {unknown}")
    else:
        raise ValueError(f"Unsupported mask encoding: {encoding}")
    validate_labels(mask, num_classes, ignore_index)
    return mask.astype(np.int64)


def colorize_mask(mask: np.ndarray, ignore_index: int = 255) -> Image.Image:
    validate_labels(mask, len(CLASS_NAMES), ignore_index)
    colors = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = mask != ignore_index
    colors[valid] = PALETTE[mask[valid]]
    return Image.fromarray(colors)


def save_preview(image: Image.Image, mask: np.ndarray, directory, name: str) -> None:
    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    colored = colorize_mask(mask)
    image.convert("RGB").save(directory / f"{name}_image.png")
    Image.fromarray(mask.astype(np.uint8)).save(directory / f"{name}_index.png")
    colored.save(directory / f"{name}_mask.png")
    Image.blend(image.convert("RGB"), colored, alpha=0.5).save(directory / f"{name}_overlay.png")
