"""Shared geometry for image/mask pairs; only train receives random augmentation."""

import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps


class PairedTransform:
    def __init__(
        self, size=(384, 384), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
        training=False, augmentation=None,
    ):
        self.size = tuple(size)
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        self.training = training
        self.augmentation = augmentation or {"enabled": False}

    def __call__(self, image: Image.Image, mask: np.ndarray):
        image = image.convert("RGB")
        label = Image.fromarray(mask.astype(np.uint8))
        if self.training and self.augmentation["enabled"]:
            if random.random() < self.augmentation["horizontal_flip_probability"]:
                image, label = ImageOps.mirror(image), ImageOps.mirror(label)
            operations = [
                (ImageEnhance.Brightness, self.augmentation["brightness"]),
                (ImageEnhance.Contrast, self.augmentation["contrast"]),
                (ImageEnhance.Color, self.augmentation["saturation"]),
            ]
            random.shuffle(operations)
            for operation, amplitude in operations:
                if amplitude:
                    image = operation(image).enhance(random.uniform(1 - amplitude, 1 + amplitude))
        height, width = self.size
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        label = label.resize((width, height), Image.Resampling.NEAREST)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        pixels = (pixels - self.mean) / self.std
        return (
            torch.from_numpy(pixels.transpose(2, 0, 1).copy()),
            torch.from_numpy(np.array(label, dtype=np.int64)),
        )


def transform_from_config(config: dict, split: str, *, disable_augmentation=False):
    preprocessing = config["preprocessing"]
    return PairedTransform(
        size=preprocessing["size"], mean=preprocessing["mean"], std=preprocessing["std"],
        training=split == "train" and not disable_augmentation,
        augmentation=config["augmentation"],
    )
