import pytest
import torch

from src.losses import SegmentationLoss, soft_dice_loss


@pytest.mark.parametrize("kind", ["ce", "ce_dice"])
def test_ignored_logits_do_not_affect_loss_or_gradient(kind):
    logits = torch.randn(1, 3, 2, 2, requires_grad=True)
    labels = torch.tensor([[[0, 1], [2, 255]]])
    criterion = SegmentationLoss(kind)
    loss = criterion(logits, labels)
    modified = logits.detach().clone()
    modified[:, :, 1, 1] = torch.tensor([1000.0, -1000.0, 500.0])
    assert float(loss) == pytest.approx(float(criterion(modified, labels)))
    loss.backward()
    assert torch.equal(logits.grad[:, :, 1, 1], torch.zeros(1, 3))
    assert torch.isfinite(logits.grad).all()


def test_perfect_prediction_dice_and_class_absence():
    labels = torch.zeros((1, 2, 2), dtype=torch.long)
    logits = torch.full((1, 3, 2, 2), -30.0)
    logits[:, 0] = 30.0
    assert float(soft_dice_loss(logits, labels)) == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize("kind", ["ce", "ce_dice"])
def test_all_ignored_is_zero_and_has_zero_gradient(kind):
    logits = torch.randn(1, 3, 2, 2, requires_grad=True)
    labels = torch.full((1, 2, 2), 255, dtype=torch.long)
    loss = SegmentationLoss(kind)(logits, labels)
    assert float(loss) == 0
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))
