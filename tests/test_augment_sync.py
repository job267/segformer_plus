import numpy as np
import torch
from PIL import Image

from src.transforms import PairedTransform


def test_forced_flip_moves_both_image_and_mask():
    mask = np.array([[0, 0, 1, 1], [0, 0, 1, 255]], dtype=np.int64)
    image = Image.fromarray(np.repeat((mask == 1)[..., None], 3, axis=2).astype(np.uint8) * 255)
    transform = PairedTransform(
        size=(2, 4), mean=(0, 0, 0), std=(1, 1, 1), training=True,
        augmentation={
            "enabled": True, "horizontal_flip_probability": 1.0,
            "brightness": 0, "contrast": 0, "saturation": 0,
        },
    )
    pixels, labels = transform(image, mask)
    assert np.array_equal(labels.numpy(), np.fliplr(mask))
    assert torch.equal(pixels[0] == 1, labels == 1)


def test_validation_ignores_random_augmentation_and_resize_preserves_ids():
    image = Image.new("RGB", (3, 2), (120, 50, 200))
    mask = np.array([[0, 3, 255], [3, 10, 0]], dtype=np.int64)
    transform = PairedTransform(
        size=(17, 19), training=False,
        augmentation={"enabled": True, "horizontal_flip_probability": 1.0},
    )
    first, second = transform(image, mask), transform(image, mask)
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
    assert set(first[1].unique().tolist()) <= {0, 3, 10, 255}
