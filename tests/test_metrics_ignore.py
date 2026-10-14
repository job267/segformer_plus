import pytest
import torch

from src.metrics import SegmentationMetrics, resize_logits


def test_confusion_and_ignore_match_hand_calculation():
    metrics = SegmentationMetrics(3)
    metrics.update(torch.tensor([0, 0, 2, 99]), torch.tensor([0, 1, 2, 255]))
    result = metrics.compute()
    assert result["confusion_matrix"] == [[1, 0, 0], [1, 0, 0], [0, 0, 1]]
    assert result["per_class_iou"] == [0.5, 0.0, 1.0]
    assert result["miou"] == pytest.approx(0.5)
    assert result["pixel_accuracy"] == pytest.approx(2 / 3)


def test_absent_classes_are_null_but_false_positives_are_included():
    metrics = SegmentationMetrics(3)
    metrics.update(torch.tensor([0, 1]), torch.tensor([0, 0]))
    result = metrics.compute()
    assert result["per_class_iou"] == [0.5, 0.0, None]
    assert result["miou"] == pytest.approx(0.25)
    assert result["classes_in_miou"] == 2


def test_all_ignored_and_invalid_predictions():
    metrics = SegmentationMetrics(2)
    metrics.update(torch.tensor([999]), torch.tensor([255]))
    assert metrics.compute()["miou"] is None
    with pytest.raises(ValueError, match="invalid"):
        metrics.update(torch.tensor([2]), torch.tensor([0]))


def test_logits_resize_before_argmax():
    logits = torch.randn(2, 11, 8, 8)
    labels = torch.zeros(2, 32, 40, dtype=torch.long)
    assert resize_logits(logits, labels).shape == (2, 11, 32, 40)
