"""Manifest-driven CamVid loading; no guessed pairing or automatic label remapping."""

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from src.config import project_path
from src.labels import decode_mask
from src.transforms import transform_from_config


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image_path: Path
    mask_path: Path


def contained_path(root: Path, value: str) -> Path:
    relative = Path(value.replace("\\", "/"))
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(
            f"Manifest paths must stay inside the dataset root: {value} "
            f"(resolved={path}, root={root.resolve()})"
        )
    return path


def read_manifest(root: Path, manifest: str, *, require_files=True) -> list[Sample]:
    root = Path(root).resolve()
    manifest_path = contained_path(root, manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing split manifest: {manifest_path}. Run scripts/download_camvid.py "
            "or place the pinned dataset according to data/README.md."
        )
    samples = []
    images_seen, masks_seen, ids_seen = set(), set(), set()
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"{manifest}:{line_number}: expected image_path mask_path")
        image_path, mask_path = [contained_path(root, field) for field in fields]
        sample_id = image_path.stem
        if mask_path.stem.removesuffix("_L") != sample_id:
            raise ValueError(f"{manifest}:{line_number}: mismatched image/mask filenames")
        if image_path in images_seen or mask_path in masks_seen or sample_id in ids_seen:
            raise ValueError(f"{manifest}:{line_number}: duplicate sample")
        images_seen.add(image_path)
        masks_seen.add(mask_path)
        ids_seen.add(sample_id)
        if require_files:
            for path in (image_path, mask_path):
                if not path.is_file():
                    raise FileNotFoundError(f"Missing dataset file: {path}")
        samples.append(Sample(sample_id, image_path, mask_path))
    if not samples:
        raise ValueError(f"Empty split: {manifest}")
    return samples


class CamVidDataset(Dataset):
    def __init__(self, config: dict, split="train", *, disable_augmentation=False):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split}")
        self.config, self.split = config, split
        self.samples = read_manifest(
            project_path(config["dataset"]["root"]), config["dataset"]["splits"][split]
        )
        self.transform = transform_from_config(
            config, split, disable_augmentation=disable_augmentation
        )

    def __len__(self):
        return len(self.samples)

    def raw_sample(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.image_path) as opened:
            image = opened.convert("RGB")
        with Image.open(sample.mask_path) as opened:
            mask = decode_mask(
                opened, self.config["dataset"]["label_encoding"],
                self.config["model"]["num_labels"], self.config["dataset"]["ignore_index"],
            )
        if image.size != (mask.shape[1], mask.shape[0]):
            raise ValueError(f"Image/mask dimensions differ: {sample.sample_id}")
        return image, mask

    def __getitem__(self, index: int):
        image, mask = self.raw_sample(index)
        pixels, labels = self.transform(image, mask)
        return {"pixel_values": pixels, "labels": labels, "sample_id": self.samples[index].sample_id}
