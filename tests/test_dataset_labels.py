import numpy as np
import pytest
from PIL import Image

from src.audit import audit_dataset
from src.dataset import CamVidDataset, read_manifest
from src.labels import PALETTE, decode_mask


def test_index_masks_keep_class_zero_and_palette_indices():
    values = np.array([[0, 3, 10, 255]], dtype=np.uint8)
    indexed = Image.fromarray(values).convert("P")
    assert np.array_equal(decode_mask(indexed), values)


def test_rgb_palette_mapping_and_unknown_colors():
    colors = np.vstack([PALETTE, [0, 0, 0]]).astype(np.uint8).reshape(1, 12, 3)
    assert decode_mask(Image.fromarray(colors), "rgb").tolist() == [list(range(11)) + [255]]
    with pytest.raises(ValueError, match="Unknown RGB"):
        decode_mask(Image.fromarray(np.array([[[1, 2, 3]]], dtype=np.uint8)), "rgb")


def test_illegal_index_and_rgb_in_index_mode():
    with pytest.raises(ValueError, match="Illegal label"):
        decode_mask(Image.fromarray(np.array([[11]], dtype=np.uint8)))
    with pytest.raises(ValueError, match="two-dimensional"):
        decode_mask(Image.new("RGB", (2, 2)))


def test_dataset_and_full_audit(tiny_camvid):
    dataset = CamVidDataset(tiny_camvid)
    assert len(dataset) == 2
    assert dataset[0]["pixel_values"].shape == (3, 32, 32)
    assert set(dataset[0]["labels"].unique().tolist()) <= {0, 3, 8, 255}
    audit = audit_dataset(tiny_camvid)
    assert audit["status"] == "passed"
    assert audit["splits"]["train"]["audited_samples"] == 2
    assert audit["splits"]["train"]["class_pixels"]["Sky"] > 0


def test_split_overlap_is_rejected(tiny_camvid):
    from pathlib import Path
    root = Path(tiny_camvid["dataset"]["root"])
    train_line = (root / "camvid_train.txt").read_text(encoding="utf-8").splitlines()[0]
    (root / "camvid_val.txt").write_text(train_line + "\n", encoding="utf-8")
    audit = audit_dataset(tiny_camvid)
    assert audit["status"] == "failed"
    assert any("Split overlap" in error for error in audit["errors"])


def test_manifest_path_traversal_is_rejected(tmp_path):
    (tmp_path / "train.txt").write_text("../outside.png masks/outside_L.png\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inside"):
        read_manifest(tmp_path, "train.txt", require_files=False)


def test_missing_pairs_and_dimension_mismatch(tiny_camvid):
    from pathlib import Path
    root = Path(tiny_camvid["dataset"]["root"])
    Image.new("L", (1, 1)).save(root / "masks/train_0_L.png")
    assert audit_dataset(tiny_camvid)["status"] == "failed"
    (root / "masks/train_0_L.png").unlink()
    with pytest.raises(FileNotFoundError, match="Missing dataset"):
        CamVidDataset(tiny_camvid)
