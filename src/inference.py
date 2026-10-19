"""Verified local checkpoint loading, native-size predictions and precision evaluation."""

from contextlib import contextmanager
from html import escape
from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F
from transformers import SegformerForSemanticSegmentation

from src.comparison import load_completed
from src.config import load_config, project_path
from src.labels import CLASS_NAMES, PALETTE, colorize_mask
from src.metrics import SegmentationMetrics
from src.transforms import PairedTransform
from src.utils import file_sha256, write_json


@contextmanager
def precision_context(device, precision):
    if precision not in ("fp32", "fp16"):
        raise ValueError("precision must be fp32 or fp16")
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("FP16 autocast requires CUDA in this project")
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=precision == "fp16"):
        yield


def load_run_model(run_directory, device):
    directory = project_path(run_directory)
    config = load_config(directory / "config.yaml")
    if project_path(config["output_dir"]) != directory:
        raise ValueError("Run directory differs from its archived configuration")
    metrics, _, _ = load_completed(config)
    model, info = SegformerForSemanticSegmentation.from_pretrained(
        directory / "best_model", local_files_only=True, output_loading_info=True
    )
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Local checkpoint did not load strictly: {info}")
    if model.config.num_labels != 11 or [model.config.id2label[i] for i in range(11)] != list(CLASS_NAMES):
        raise ValueError("Local checkpoint class order differs")
    return model.to(device).eval(), config, metrics


def open_rgb(path):
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def image_tensor(image, config):
    settings = config["preprocessing"]
    transform = PairedTransform(size=settings["size"], mean=settings["mean"], std=settings["std"])
    pixels, _ = transform(image, np.zeros((image.height, image.width), dtype=np.uint8))
    return pixels.unsqueeze(0)


@torch.inference_mode()
def predict_tensor(model, pixels, output_size, device, precision, *, check_finite=True):
    with precision_context(device, precision):
        logits = model(pixel_values=pixels).logits
    if check_finite and not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite prediction logits")
    logits = F.interpolate(logits.float(), output_size, mode="bilinear", align_corners=False)
    return logits.argmax(1)


@torch.inference_mode()
def predict_image(model, image, config, device, precision="fp32"):
    pixels = image_tensor(image, config).to(device)
    return predict_tensor(model, pixels, (image.height, image.width), device, precision)[0].cpu().numpy().astype(np.uint8)


@torch.inference_mode()
def evaluate_precision(model, loader, device, precision):
    metrics = SegmentationMetrics(11, 255)
    for batch in loader:
        labels = batch["labels"]
        prediction = predict_tensor(model, batch["pixel_values"].to(device), labels.shape[-2:], device, precision)
        metrics.update(prediction, labels)
    return metrics.compute()


def write_prediction(model, image_path, config, device, precision, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError(f"Prediction output must be empty: {directory}")
    image = open_rgb(image_path)
    labels = predict_image(model, image, config, device, precision)
    colored = colorize_mask(labels)
    image.save(directory / "input.png")
    Image.fromarray(labels).save(directory / "index.png")
    colored.save(directory / "segmentation.png")
    Image.blend(image, colored, alpha=0.5).save(directory / "overlay.png")
    return {
        "input": Path(image_path).name, "input_sha256": file_sha256(image_path),
        "original_size": [image.height, image.width], "directory": directory.name,
        "present_classes": [CLASS_NAMES[index] for index in np.unique(labels)],
    }


def save_demo_index(directory, items, metadata):
    directory = Path(directory)
    rows = []
    for item in items:
        name = quote(item["directory"], safe="")
        rows.append(f'<section><h2>{escape(Path(item["input"]).name)}</h2><div class="grid">' + "".join(
            f'<figure><img src="predictions/{name}/{file}.png"><figcaption>{caption}</figcaption></figure>'
            for file, caption in (("input", "原图"), ("segmentation", "分割结果"), ("overlay", "叠加结果"))
        ) + "</div></section>")
    legend = "".join(
        f'<span><i style="background:rgb({int(color[0])},{int(color[1])},{int(color[2])})"></i>{escape(name)}</span>'
        for name, color in zip(CLASS_NAMES, PALETTE)
    )
    content = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>SegFormer 图片分割</title><style>' + (
        'body{font:16px system-ui;margin:32px;background:#f4f6f8;color:#182638}h1{font-size:28px}'
        '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}'
        'figure{margin:0}img{width:100%;border-radius:8px}section{background:white;padding:20px;margin:24px 0;border-radius:12px}'
        'figcaption{padding:8px 0}aside{display:flex;gap:16px;flex-wrap:wrap}i{display:inline-block;width:14px;height:14px;margin-right:6px}'
    ) + '</style><h1>SegFormer 街景语义分割</h1><p>' + escape(
        f'{metadata["experiment_id"]} · {metadata["precision"].upper()} · 输入 {metadata["input_size"][0]}×{metadata["input_size"][1]} · 11 类'
    ) + '</p><aside>' + legend + '</aside>' + "".join(rows) + '</html>'
    (directory / "index.html").write_text(content, encoding="utf-8")
    write_json(directory / "prediction_summary.json", {**metadata, "images": items})
