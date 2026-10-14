"""Inspect every listed sample and detect split leakage without training."""

from collections import Counter

import numpy as np
from PIL import Image

from src.config import project_path
from src.dataset import read_manifest, contained_path
from src.labels import CLASS_NAMES, decode_mask, save_preview
from src.utils import file_sha256, utc_now


def audit_dataset(config: dict, *, preview_dir=None, preview_count=3) -> dict:
    dataset = config["dataset"]
    root = project_path(dataset["root"])
    report = {
        "status": "checking", "timestamp_utc": utc_now(), "root": str(root),
        "source": dataset["source"], "source_revision": dataset["source_revision"],
        "label_encoding": dataset["label_encoding"], "ignore_index": dataset["ignore_index"],
        "class_names": list(CLASS_NAMES), "splits": {}, "errors": [], "error_count": 0,
        "files": [], "note": "Manifest counts do not prove that image/mask files are present.",
    }
    seen_ids, seen_paths, seen_hashes = {}, {}, {}

    def add_error(message):
        report["error_count"] += 1
        if len(report["errors"]) < 50:
            report["errors"].append(message)

    for split, manifest in dataset["splits"].items():
        counts = np.zeros(11, dtype=np.int64)
        stats = {
            "manifest": manifest, "listed_samples": 0, "audited_samples": 0,
            "missing_files": 0, "total_pixels": 0, "ignored_pixels": 0,
            "image_modes": {}, "mask_modes": {}, "resolutions": {},
        }
        report["splits"][split] = stats
        image_modes, mask_modes, resolutions = Counter(), Counter(), Counter()
        try:
            samples = read_manifest(root, manifest, require_files=False)
            stats["manifest_sha256"] = file_sha256(contained_path(root, manifest))
        except (OSError, ValueError) as error:
            add_error(f"{split}: {type(error).__name__}: {error}")
            samples = []
        stats["listed_samples"] = len(samples)
        expected = dataset["expected_counts"][split]
        stats["expected_samples"] = expected
        if samples and len(samples) != expected:
            add_error(f"{split}: manifest count {len(samples)} differs from expected {expected}")
        for sample in samples:
            for key, tracker in (
                (sample.sample_id, seen_ids),
                (str(sample.image_path), seen_paths),
                (str(sample.mask_path), seen_paths),
            ):
                if key in tracker:
                    add_error(f"Split overlap: {sample.sample_id} in {tracker[key]} and {split}")
                tracker[key] = split
            missing = [path for path in (sample.image_path, sample.mask_path) if not path.is_file()]
            if missing:
                stats["missing_files"] += len(missing)
                add_error(f"{split}/{sample.sample_id}: missing {', '.join(str(x) for x in missing)}")
                continue
            try:
                with Image.open(sample.image_path) as opened:
                    image_modes[opened.mode] += 1
                    image = opened.convert("RGB")
                with Image.open(sample.mask_path) as opened:
                    mask_modes[opened.mode] += 1
                    mask = decode_mask(opened, dataset["label_encoding"], 11, dataset["ignore_index"])
                if image.size != (mask.shape[1], mask.shape[0]):
                    raise ValueError(f"Image size {image.size} differs from mask {mask.shape}")
                image_hash = file_sha256(sample.image_path)
                mask_hash = file_sha256(sample.mask_path)
                if image_hash in seen_hashes and seen_hashes[image_hash][0] != split:
                    add_error(f"Identical image content across splits: {sample.sample_id} and {seen_hashes[image_hash]}")
                seen_hashes[image_hash] = (split, sample.sample_id)
                valid = mask != dataset["ignore_index"]
                counts += np.bincount(mask[valid], minlength=11)
                stats["total_pixels"] += mask.size
                stats["ignored_pixels"] += int((~valid).sum())
                resolutions[f"{image.width}x{image.height}"] += 1
                stats["audited_samples"] += 1
                report["files"].append({
                    "split": split, "sample_id": sample.sample_id,
                    "image": str(sample.image_path.relative_to(root)),
                    "mask": str(sample.mask_path.relative_to(root)),
                    "image_sha256": image_hash, "mask_sha256": mask_hash,
                })
                if preview_dir and split == "train" and stats["audited_samples"] <= preview_count:
                    save_preview(image, mask, preview_dir, sample.sample_id)
            except (OSError, ValueError) as error:
                add_error(f"{split}/{sample.sample_id}: {type(error).__name__}: {error}")
        stats.update(
            class_pixels=dict(zip(CLASS_NAMES, map(int, counts))),
            image_modes=dict(image_modes), mask_modes=dict(mask_modes),
            resolutions=dict(resolutions),
            ignored_ratio=(
                stats["ignored_pixels"] / stats["total_pixels"] if stats["total_pixels"] else None
            ),
        )
    report["status"] = "passed" if report["error_count"] == 0 else "failed"
    return report
