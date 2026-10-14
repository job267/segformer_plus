"""Shared training and FP32 evaluation epoch functions."""

import math

import torch

from src.metrics import SegmentationMetrics, resize_logits


def make_scaler(device, amp, *, initial_scale=65536.0):
    return torch.amp.GradScaler(
        "cuda", enabled=amp and device.type == "cuda", init_scale=initial_scale
    )


def train_one_epoch(model, loader, optimizer, criterion, device, *, amp=False, scaler=None):
    model.train()
    scaler = scaler or make_scaler(device, amp)
    weighted_loss, valid_pixels = 0.0, 0
    for batch in loader:
        pixels, labels = batch["pixel_values"].to(device), batch["labels"].to(device)
        count = int((labels != criterion.ignore_index).sum().item())
        if count == 0:
            continue
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            loss = criterion(model(pixel_values=pixels).logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        weighted_loss += float(loss.detach()) * count
        valid_pixels += count
    if not valid_pixels:
        raise ValueError("Training split contains no valid pixels")
    return weighted_loss / valid_pixels


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    metrics = SegmentationMetrics(model.config.num_labels, criterion.ignore_index)
    weighted_loss, valid_pixels = 0.0, 0
    for batch in loader:
        pixels, labels = batch["pixel_values"].to(device), batch["labels"].to(device)
        logits = resize_logits(model(pixel_values=pixels).logits, labels)
        loss = criterion(logits, labels)
        if not math.isfinite(float(loss)):
            raise FloatingPointError("Non-finite evaluation loss")
        count = int((labels != criterion.ignore_index).sum().item())
        weighted_loss += float(loss) * count
        valid_pixels += count
        metrics.update(logits.argmax(1), labels)
    if not valid_pixels:
        raise ValueError("Evaluation split contains no valid pixels")
    return {"loss": weighted_loss / valid_pixels, **metrics.compute()}
