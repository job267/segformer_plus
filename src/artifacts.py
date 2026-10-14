"""Export measured metrics, fixed prediction examples, and baseline plots."""

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from src.labels import CLASS_NAMES, colorize_mask


def export_metric_tables(metrics, directory):
    directory = Path(directory)
    with (directory / "per_class_iou.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class_name", "iou", "gt_pixels"])
        writer.writerows(
            (index, name, metrics["per_class_iou"][index], metrics["class_pixels"][index])
            for index, name in enumerate(CLASS_NAMES)
        )
    with (directory / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ground_truth/prediction", *CLASS_NAMES])
        writer.writerows((name, *row) for name, row in zip(CLASS_NAMES, metrics["confusion_matrix"]))


def make_plots(history, metrics, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for field, label in [("train_loss", "Train"), ("val_loss", "Validation")]:
        axes[0].plot(epochs, [row[field] for row in history], label=label)
    axes[0].set(xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_miou"] for row in history])
    axes[1].set(xlabel="Epoch", ylabel="Validation mIoU", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.savefig(directory / "training_curves.png", dpi=180)
    plt.close(figure)
    matrix = np.array(metrics["confusion_matrix"], dtype=float)
    totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)
    figure, axis = plt.subplots(figsize=(10, 8), layout="constrained")
    plot = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    axis.set(
        xticks=range(11), yticks=range(11), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        xlabel="Prediction", ylabel="Ground truth", title="Test confusion matrix (row normalized)",
    )
    plt.setp(axis.get_xticklabels(), rotation=55, ha="right")
    for row in range(11):
        for column in range(11):
            axis.text(column, row, f"{normalized[row, column]:.2f}", ha="center", va="center",
                      fontsize=8, color="white" if normalized[row, column] > 0.5 else "black")
    figure.colorbar(plot, ax=axis, label="Fraction within each ground-truth class")
    figure.savefig(directory / "confusion_matrix.png", dpi=180)
    plt.close(figure)


@torch.no_grad()
def export_predictions(model, dataset, device, directory, count=6):
    directory = Path(directory) / "predictions"
    directory.mkdir(parents=True, exist_ok=True)
    model.eval()
    sample_ids = []
    for index in np.linspace(0, len(dataset) - 1, min(count, len(dataset)), dtype=int):
        image, target = dataset.raw_sample(int(index))
        pixels, _ = dataset.transform(image, target)
        logits = model(pixel_values=pixels.unsqueeze(0).to(device)).logits.float()
        prediction = F.interpolate(
            logits, size=(image.height, image.width), mode="bilinear", align_corners=False
        ).argmax(1)[0].cpu().numpy()
        name = dataset.samples[index].sample_id
        sample_ids.append(name)
        image.save(directory / f"{name}_input.png")
        colorize_mask(target).save(directory / f"{name}_gt.png")
        Image.fromarray(prediction.astype(np.uint8)).save(directory / f"{name}_index.png")
        colored = colorize_mask(prediction)
        colored.save(directory / f"{name}_pred.png")
        Image.blend(image, colored, alpha=0.5).save(directory / f"{name}_overlay.png")
    return sample_ids
