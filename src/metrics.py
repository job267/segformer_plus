"""Dataset-level confusion matrix: rows=ground truth, columns=prediction."""

import numpy as np
import torch
from torch.nn import functional as F


def resize_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.shape[-2:] == labels.shape[-2:]:
        return logits
    return F.interpolate(logits, labels.shape[-2:], mode="bilinear", align_corners=False)


class SegmentationMetrics:
    def __init__(self, num_classes=11, ignore_index=255):
        self.num_classes, self.ignore_index = num_classes, ignore_index
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, prediction, target) -> None:
        prediction = torch.as_tensor(prediction).detach().cpu().numpy()
        target = torch.as_tensor(target).detach().cpu().numpy()
        if prediction.shape != target.shape:
            raise ValueError("Prediction and target shapes must match")
        valid = target != self.ignore_index
        truth, predicted = target[valid], prediction[valid]
        for name, values in (("target", truth), ("prediction", predicted)):
            if not np.issubdtype(values.dtype, np.integer):
                raise ValueError(f"{name} must contain integer class IDs")
            if ((values < 0) | (values >= self.num_classes)).any():
                raise ValueError(f"{name} contains invalid class IDs")
        bins = np.bincount(
            self.num_classes * truth.astype(np.int64) + predicted.astype(np.int64),
            minlength=self.num_classes**2,
        )
        self.confusion += bins.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict:
        true_positive = np.diag(self.confusion)
        truth_pixels = self.confusion.sum(axis=1)
        union = truth_pixels + self.confusion.sum(axis=0) - true_positive
        present = union > 0
        iou = np.divide(true_positive, union, out=np.zeros_like(union, dtype=float), where=present)
        total = int(truth_pixels.sum())
        return {
            "miou": float(iou[present].mean()) if present.any() else None,
            "pixel_accuracy": float(true_positive.sum() / total) if total else None,
            "per_class_iou": [float(x) if valid else None for x, valid in zip(iou, present)],
            "class_pixels": truth_pixels.tolist(),
            "classes_in_miou": int(present.sum()),
            "valid_pixels": total,
            "absent_class_policy": "Exclude classes whose TP+FP+FN=0; encode their IoU as null.",
            "confusion_matrix": self.confusion.tolist(),
        }
