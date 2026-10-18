from copy import deepcopy

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from src.config import load_config
from src.experiment import fit
from src.losses import SegmentationLoss
from src.utils import seed_everything
from src.weighted import (
    CONTROL_CONFIG, WEIGHTED_CONFIG, WEIGHT_RULE, WeightedSegmentationLoss,
    compute_weight_record, frequency_weights, validate_weighted,
)
from src.weighted_training import fit_weighted


def test_frequency_rule_is_bounded_and_uses_relative_frequency():
    counts = [10**12] + list(range(1, 11))
    weights = frequency_weights(counts)
    assert min(weights) >= 0.5 and max(weights) <= 3
    assert weights[0] < weights[1]
    assert frequency_weights(np.asarray(counts) * 7) == pytest.approx(weights)
    assert frequency_weights([1] * 11) == pytest.approx([1] * 11)
    with pytest.raises(ValueError, match="positive finite"):
        frequency_weights([0] + [1] * 10)


def test_weighted_ce_matches_pytorch_and_ignores_255():
    generator = torch.Generator().manual_seed(8)
    logits = torch.randn(2, 11, 4, 4, generator=generator, requires_grad=True)
    target = torch.randint(0, 11, (2, 4, 4), generator=generator)
    target[0, 0, 0] = 255
    weights = torch.linspace(0.5, 3, 11)
    criterion = WeightedSegmentationLoss(weights, dice_weight=0)
    expected = torch.nn.functional.cross_entropy(logits, target, weight=weights, ignore_index=255)
    actual = criterion(logits, target)
    assert torch.allclose(actual, expected)
    assert torch.allclose(actual, WeightedSegmentationLoss(weights * 7, dice_weight=0)(logits, target))
    actual.backward()
    assert torch.equal(logits.grad[0, :, 0, 0], torch.zeros(11))
    assert torch.isfinite(logits.grad).all()


def test_ones_weights_reproduce_original_ce_dice_and_all_ignore():
    logits = torch.randn(2, 11, 4, 4, requires_grad=True)
    labels = torch.randint(0, 11, (2, 8, 8))
    labels[:, 0] = 255
    criterion = WeightedSegmentationLoss([1] * 11)
    assert torch.equal(criterion(logits, labels), SegmentationLoss("ce_dice")(logits, labels))
    ignored = torch.full_like(labels, 255)
    loss = criterion(logits, ignored)
    assert loss.item() == 0
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


@pytest.mark.parametrize("change", ["seed", "size", "rule"])
def test_e5_rejects_extra_factors(change):
    config, control = load_config(WEIGHTED_CONFIG), load_config(CONTROL_CONFIG)
    validate_weighted(config, control)
    if change == "seed":
        config["seed"] = 1
    elif change == "size":
        config["preprocessing"]["size"] = [384, 384]
    else:
        config["training"]["class_weighting"]["maximum"] = 10
    with pytest.raises(ValueError):
        validate_weighted(config, control)


def test_weight_record_reads_train_masks_only(tiny_camvid, monkeypatch):
    import src.weighted as weighted
    control = deepcopy(tiny_camvid)
    control["experiment_id"] = "E4_resolution"
    control["output_dir"] = "results/test_control"
    control["dataset"]["splits"].update(val="MUST_NOT_READ_VAL", test="MUST_NOT_READ_TEST")
    root = weighted.project_path(control["dataset"]["root"])
    for path in (root / "masks").glob("train_*.png"):
        mask = (np.indices((32, 48)).sum(axis=0) % 11).astype(np.uint8)
        mask[0, 0] = 255
        Image.fromarray(mask).save(path)
    config = deepcopy(control)
    config.update(experiment_id="E5_weighted", output_dir="results/test_weighted")
    config["training"]["class_weighting"] = deepcopy(WEIGHT_RULE)
    monkeypatch.setattr(weighted, "load_config", lambda path: control)
    monkeypatch.setattr(weighted, "file_sha256", lambda path: "fixture-sha")
    record = compute_weight_record(config)
    assert record["source_split"] == "train" and record["samples"] == 2
    assert all(name.startswith("train_") for name in record["sample_ids"])
    assert all(count > 0 for count in record["class_pixels"])


def setup_run():
    seed_everything(42)
    model = SegformerForSemanticSegmentation(SegformerConfig(
        num_labels=11, hidden_sizes=[4, 4, 8, 8], depths=[1, 1, 1, 1],
        num_attention_heads=[1, 1, 1, 1], sr_ratios=[1, 1, 1, 1], decoder_hidden_size=8,
    ))
    generator = torch.Generator().manual_seed(7)
    data = [{"pixel_values": torch.randn(3, 32, 32, generator=generator),
             "labels": torch.randint(0, 11, (32, 32), generator=generator)} for _ in range(4)]
    return model, DataLoader(data, batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(42)), DataLoader(data, batch_size=2)


@pytest.fixture
def cpu_config():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    config = load_config()
    config["training"].update(epochs=2, amp=False, loss="ce_dice")
    yield config
    torch.set_num_threads(previous)


def test_training_adapter_with_ones_matches_frozen_engine(tmp_path, cpu_config):
    model, train, val = setup_run()
    expected = fit(model, train, val, cpu_config, torch.device("cpu"), tmp_path / "original")
    state = {key: value.clone() for key, value in model.state_dict().items()}
    model, train, val = setup_run()
    actual = fit_weighted(model, train, val, cpu_config, torch.device("cpu"), tmp_path / "adapter", WeightedSegmentationLoss([1] * 11))
    assert all(torch.equal(value, state[key]) for key, value in model.state_dict().items())
    for actual_row, expected_row in zip(actual["history"], expected["history"]):
        for field in ("train_loss", "val_loss", "val_miou"):
            assert actual_row[field] == expected_row[field]


def test_weighted_resume_matches_continuous_and_rejects_changed_weights(tmp_path, cpu_config):
    weights = np.linspace(0.5, 3, 11)
    model, train, val = setup_run()
    fit_weighted(model, train, val, cpu_config, torch.device("cpu"), tmp_path / "full", WeightedSegmentationLoss(weights))
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    model, train, val = setup_run()
    def stop(row):
        raise InterruptedError("after committed epoch")
    with pytest.raises(InterruptedError):
        fit_weighted(model, train, val, cpu_config, torch.device("cpu"), tmp_path / "resume", WeightedSegmentationLoss(weights), on_epoch=stop)
    model, train, val = setup_run()
    fit_weighted(model, train, val, cpu_config, torch.device("cpu"), tmp_path / "resume", WeightedSegmentationLoss(weights), resume=True)
    assert all(torch.equal(value, expected[key]) for key, value in model.state_dict().items())
    with pytest.raises(ValueError, match="weights changed"):
        fit_weighted(model, train, val, cpu_config, torch.device("cpu"), tmp_path / "resume", WeightedSegmentationLoss([1] * 11), resume=True)


def test_final_comparison_uses_validation_even_if_test_falls(tmp_path, monkeypatch):
    import json
    import src.weighted as weighted
    from src.utils import write_json

    control_config, config = load_config(CONTROL_CONFIG), load_config(WEIGHTED_CONFIG)
    control = {"best_val_epoch": 1, "best_val_miou": 0.7, "test_miou": 0.6,
               "pixel_accuracy": 0.9, "checkpoint_sha256": "control-sha",
               "class_pixels": [1] * 11, "per_class_iou": [0.6] * 11}
    candidate = {**control, "best_val_miou": 0.71, "test_miou": 0.59,
                 "class_weights_sha256": "weight-sha", "input_size": [512, 512],
                 "evaluation_size": [384, 384], "control_checkpoint_sha256": "control-sha",
                 "per_class_iou": [0.59] * 11}
    history = [{"epoch_time_sec": 60, "peak_vram_mb": 100}]
    info = {"data_audit_sha256": "same-data", "source_sha256": {"src/train.py": "same-code"}}
    monkeypatch.setattr(weighted, "project_path", lambda value: tmp_path / value)
    monkeypatch.setattr(weighted, "load_config", lambda path: control_config)
    monkeypatch.setattr(weighted, "file_sha256", lambda path: "weight-sha")
    monkeypatch.setattr(weighted, "load_completed", lambda item: (
        candidate if item["experiment_id"] == "E5_weighted" else control, history, info
    ))
    (tmp_path / "reports").mkdir()
    for item in (config, control_config):
        path = tmp_path / item["output_dir"]
        write_json(path / "status.json", {"status": "completed"})
        write_json(path / "best_validation_metrics.json", {"class_pixels": [1] * 11, "per_class_iou": [0.6] * 11})
    rows = weighted.write_weighted_comparison(config)
    selected = json.loads((tmp_path / "results/weighted_validation_selected_experiment.json").read_text())
    assert selected["experiment_id"] == "E5_weighted"
    assert rows[1]["delta_test_pp"] == pytest.approx(-1)
