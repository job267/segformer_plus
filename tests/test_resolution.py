from copy import deepcopy

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from src.config import load_config
from src.dataset import CamVidDataset
from src.experiment import fit
from src.resolution import (
    CONTROL_CONFIG, RESOLUTION_CONFIG, EvaluationGridTransform, ResolutionDataset,
    validate_evaluation_grids, validate_resolution,
)
from src.transforms import PairedTransform
from src.utils import seed_everything


@pytest.mark.parametrize("change", ["batch", "seed", "evaluation"])
def test_resolution_rejects_unplanned_changes(change):
    control = load_config(CONTROL_CONFIG)
    config = load_config(RESOLUTION_CONFIG)
    validate_resolution(config, control)
    if change == "batch":
        config["training"]["batch_size"] = 2
    elif change == "seed":
        config["seed"] = 43
    else:
        config["preprocessing"]["evaluation_size"] = [512, 512]
    with pytest.raises(ValueError):
        validate_resolution(config, control)


@pytest.mark.parametrize("split", ["val", "test"])
def test_evaluation_labels_match_control_exactly(tiny_camvid, split):
    candidate = deepcopy(tiny_camvid)
    candidate["preprocessing"].update(size=[64, 64], evaluation_size=[32, 32])
    candidate["augmentation"]["enabled"] = True
    data = ResolutionDataset(candidate, split)
    expected = CamVidDataset(tiny_camvid, split)[0]
    first, second = data[0], data[0]
    assert first["pixel_values"].shape == (3, 64, 64)
    assert first["labels"].shape == (32, 32)
    assert torch.equal(first["labels"], expected["labels"])
    assert torch.equal(first["pixel_values"], second["pixel_values"])
    assert torch.equal(first["labels"], second["labels"])
    assert first["labels"][0, 0] == 255
    assert ResolutionDataset(candidate, "train")[0]["labels"].shape == (64, 64)


def test_labels_are_resized_directly_from_original_mask():
    mask = (np.indices((33, 47)).sum(axis=0) % 11).astype(np.uint8)
    mask[0, 0] = 255
    image = Image.new("RGB", (47, 33))
    transform = EvaluationGridTransform(PairedTransform(size=[64, 64]), [32, 32])
    _, actual = transform(image, mask)
    original = Image.fromarray(mask)
    expected = np.array(original.resize((32, 32), Image.Resampling.NEAREST))
    wrong = np.array(original.resize((64, 64), Image.Resampling.NEAREST).resize((32, 32), Image.Resampling.NEAREST))
    assert not np.array_equal(expected, wrong), "Fixture must detect accidental double resize"
    assert np.array_equal(actual.numpy(), expected)


def test_native_fit_supports_different_train_and_eval_grids(tiny_camvid, tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        seed_everything(42)
        config = deepcopy(tiny_camvid)
        config["preprocessing"].update(size=[64, 64], evaluation_size=[32, 32])
        config["training"].update(epochs=1, batch_size=2, amp=False, loss="ce_dice")
        model = SegformerForSemanticSegmentation(SegformerConfig(
            num_labels=11, hidden_sizes=[4, 4, 8, 8], depths=[1, 1, 1, 1],
            num_attention_heads=[1, 1, 1, 1], sr_ratios=[1, 1, 1, 1], decoder_hidden_size=8,
        ))
        train = DataLoader(ResolutionDataset(config, "train"), batch_size=2,
                           generator=torch.Generator().manual_seed(42), shuffle=True)
        val = DataLoader(ResolutionDataset(config, "val"), batch_size=1)
        result = fit(model, train, val, config, torch.device("cpu"), tmp_path)
        state = torch.load(tmp_path / "last_checkpoint/state.pt", weights_only=False)
        valid = int((val.dataset[0]["labels"] != 255).sum())
        assert state["best_validation"]["valid_pixels"] == valid
        assert sum(state["best_validation"]["class_pixels"]) == valid
        assert np.isfinite(result["history"][0]["train_loss"])
        assert (tmp_path / "best_model/model.safetensors").is_file()
    finally:
        torch.set_num_threads(previous)


def test_comparison_rejects_different_label_grid():
    control = {split: {"class_pixels": [1] * 11} for split in ("val", "test")}
    actual = deepcopy(control)
    validate_evaluation_grids(actual, control)
    actual["val"]["class_pixels"][2] += 1
    with pytest.raises(ValueError, match="val labels differ"):
        validate_evaluation_grids(actual, control)


def test_final_report_selects_validation_even_if_test_is_lower(tmp_path, monkeypatch):
    import csv
    import json
    import src.resolution as resolution
    from src.utils import write_json

    control_config = load_config(CONTROL_CONFIG)
    config = load_config(RESOLUTION_CONFIG)
    control = {
        "best_val_epoch": 1, "best_val_miou": 0.7, "test_miou": 0.6,
        "pixel_accuracy": 0.9, "checkpoint_sha256": "control-hash",
        "class_pixels": [1] * 11, "per_class_iou": [0.6] * 11,
    }
    candidate = {
        **control, "best_val_miou": 0.71, "test_miou": 0.59,
        "per_class_iou": [0.59] * 11, "evaluation_size": [384, 384],
        "input_size": [512, 512], "control_checkpoint_sha256": "control-hash",
    }
    history = [{"epoch_time_sec": 60, "peak_vram_mb": 100}]
    info = {"data_audit_sha256": "same-data", "source_sha256": {"src/model.py": "same-code"}}
    monkeypatch.setattr(resolution, "project_path", lambda value: tmp_path / value)
    monkeypatch.setattr(resolution, "load_config", lambda path: control_config)
    monkeypatch.setattr(resolution, "load_completed", lambda item: (
        candidate if item["experiment_id"] == "E4_resolution" else control, history, info
    ))
    (tmp_path / "reports").mkdir()
    for item in (config, control_config):
        directory = tmp_path / item["output_dir"]
        write_json(directory / "status.json", {"status": "completed"})
        write_json(directory / "best_validation_metrics.json", {"class_pixels": [1] * 11})
    rows = resolution.write_resolution_comparison(config)
    selected = json.loads((tmp_path / "results/resolution_validation_selected_experiment.json").read_text())
    assert selected["experiment_id"] == "E4_resolution"
    assert rows[1]["delta_test_pp"] == pytest.approx(-1)
    with (tmp_path / "results/resolution_per_class.csv").open() as handle:
        classes = list(csv.DictReader(handle))
    assert len(classes) == 11
    assert float(classes[0]["E4_iou"]) == 0.59
    assert float(classes[0]["delta_pp"]) == pytest.approx(-1)
