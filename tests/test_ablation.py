from copy import deepcopy

import pytest
import torch

from src.comparison import CONFIG_PATHS, validate_matrix, verify_metric_record
from src.config import load_config
from src.labels import CLASS_NAMES
from src.suite import run_sequence


def test_unplanned_seed_change_is_rejected():
    configurations = [load_config(path) for path in CONFIG_PATHS]
    validate_matrix(configurations)
    configurations[2]["seed"] = 123
    with pytest.raises(ValueError, match="unplanned factor"):
        validate_matrix(configurations)


def test_queue_is_sequential_and_skips_completed():
    finished = {"E1"}
    calls = []
    phases = []

    def launch(item):
        calls.append(item)
        finished.add(item)
        return 0

    result = run_sequence(
        ["E1", "E2", "E3"], launch, lambda item: item in finished,
        lambda phase, item, completed: phases.append((phase, item)),
    )
    assert calls == ["E2", "E3"]
    assert result == ["E1", "E2", "E3"]
    assert phases == [
        ("skipped_completed", "E1"), ("running", "E2"), ("completed_experiment", "E2"),
        ("running", "E3"), ("completed_experiment", "E3"),
    ]


def test_queue_failure_does_not_start_later_jobs():
    calls = []

    def launch(item):
        calls.append(item)
        return 7

    with pytest.raises(RuntimeError, match="code 7"):
        run_sequence(["E1", "E2", "E3"], launch, lambda item: False, lambda *args: None)
    assert calls == ["E1"]


def test_zero_exit_still_requires_completed_archive():
    with pytest.raises(RuntimeError, match="verified completed archive"):
        run_sequence(["E1", "E2"], lambda item: 0, lambda item: False, lambda *args: None)


def test_comparison_checks_measured_metrics_and_validation_selection():
    config = load_config()
    config["training"]["epochs"] = 2
    matrix = [[10 if row == column else 0 for column in range(11)] for row in range(11)]
    matrix[2][1] = 4
    metrics = {
        "confusion_matrix": matrix, "test_miou": (9 + 20 / 14) / 11,
        "pixel_accuracy": 110 / 114, "class_pixels": [10, 10, 14] + [10] * 8,
        "per_class_iou": [1, 10 / 14, 10 / 14] + [1] * 8,
        "best_val_epoch": 1, "best_val_miou": 0.3, "seed": 42, "ignore_index": 255,
        "best_checkpoint_selected_by": "val_miou", "split": "test", "class_names": list(CLASS_NAMES),
    }
    history = [{"epoch": 1, "val_miou": 0.3}, {"epoch": 2, "val_miou": 0.2}]
    verify_metric_record(metrics, history, config)
    tampered = deepcopy(metrics)
    tampered["test_miou"] += 0.01
    with pytest.raises(ValueError, match="mIoU does not match"):
        verify_metric_record(tampered, history, config)
    tampered = deepcopy(metrics)
    tampered["best_val_epoch"] = 2
    with pytest.raises(ValueError, match="highest validation"):
        verify_metric_record(tampered, history, config)


def test_augmented_configs_keep_validation_deterministic(tiny_camvid):
    from src.dataset import CamVidDataset
    for loss in ("ce", "ce_dice"):
        config = deepcopy(tiny_camvid)
        config["augmentation"]["enabled"] = True
        config["training"]["loss"] = loss
        data = CamVidDataset(config, "val")
        first, second = data[0], data[0]
        assert torch.equal(first["pixel_values"], second["pixel_values"])
        assert torch.equal(first["labels"], second["labels"])
