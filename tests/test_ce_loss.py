import pytest
import torch

from src.losses import SegmentationLoss


def test_ce_matches_manual_valid_pixel_log_probability_mean():
    logits = torch.tensor(
        [[[[2.0, -1.0], [0.5, 100.0]], [[0.0, 3.0], [-0.5, -100.0]], [[-2.0, 0.0], [4.0, 0.0]]]],
        requires_grad=True,
    )
    labels = torch.tensor([[[0, 1], [2, 255]]])
    log_probabilities = logits.log_softmax(dim=1)
    expected = -(
        log_probabilities[0, 0, 0, 0]
        + log_probabilities[0, 1, 0, 1]
        + log_probabilities[0, 2, 1, 0]
    ) / 3
    actual = SegmentationLoss("ce")(logits, labels)
    assert actual.item() == pytest.approx(expected.item())
    actual.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[:, :, 1, 1]) == 0
