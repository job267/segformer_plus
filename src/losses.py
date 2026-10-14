"""Masked CE and multiclass soft Dice, computed in FP32 even under autocast."""

import torch
from torch import nn
from torch.nn import functional as F

from src.metrics import resize_logits


def soft_dice_loss(logits, labels, ignore_index=255, epsilon=1e-6):
    valid = labels != ignore_index
    if not valid.any():
        return logits.sum() * 0.0
    safe_labels = labels.masked_fill(~valid, 0)
    target = F.one_hot(safe_labels, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
    mask = valid.unsqueeze(1)
    probabilities = logits.float().softmax(dim=1) * mask
    target = target * mask
    intersection = (probabilities * target).sum(dim=(0, 2, 3))
    denominator = (probabilities + target).sum(dim=(0, 2, 3))
    present = target.sum(dim=(0, 2, 3)) > 0
    dice = (2 * intersection + epsilon) / (denominator + epsilon)
    # Classes absent from this batch's GT are excluded from Dice; CE still supervises them.
    return 1 - dice[present].mean()


class SegmentationLoss(nn.Module):
    def __init__(self, kind="ce", dice_weight=0.5, ignore_index=255):
        super().__init__()
        if kind not in ("ce", "ce_dice"):
            raise ValueError(f"Unsupported loss: {kind}")
        self.kind, self.dice_weight, self.ignore_index = kind, dice_weight, ignore_index

    def forward(self, logits, labels):
        logits = resize_logits(logits, labels).float()
        valid = labels != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0
        # torch 2.3's fused CUDA NLL spatial sum emits a nondeterminism warning.
        # Compute per-pixel CE first, then use an explicit reduction over valid pixels.
        per_pixel = F.cross_entropy(
            logits, labels, ignore_index=self.ignore_index, reduction="none"
        )
        ce = per_pixel.sum() / valid.sum()
        if self.kind == "ce_dice":
            ce = ce + self.dice_weight * soft_dice_loss(logits, labels, self.ignore_index)
        return ce


def loss_from_config(config):
    return SegmentationLoss(
        config["training"]["loss"], config["training"]["dice_weight"],
        config["dataset"]["ignore_index"],
    )
