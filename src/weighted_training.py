"""E5 training adapter; frozen epoch/checkpoint helpers and explicit weighted criterion."""

import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from src.experiment import (
    atomic_torch_save, capture_rng, config_digest, export_best, restore_rng, write_history,
)
from src.train import evaluate, make_scaler, train_one_epoch
from src.utils import utc_now, write_json


def fit_weighted(model, train_loader, val_loader, config, device, directory, criterion,
                 *, resume=False, on_epoch=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "last_checkpoint/state.pt"
    criterion = criterion.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                                  weight_decay=config["training"]["weight_decay"])
    scaler = make_scaler(device, config["training"]["amp"])
    history, best_miou, best_epoch, best_model, best_validation = [], -1.0, None, None, None
    if resume:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["config_digest"] != config_digest(config):
            raise ValueError("Resume configuration differs from the frozen E5 configuration")
        if not torch.equal(state["class_weights"], criterion.class_weights.detach().cpu()):
            raise ValueError("Resume class weights changed")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        history, best_miou, best_epoch, best_model = state["history"], state["best_miou"], state["best_epoch"], state["best_model"]
        best_validation = state["best_validation"]
        restore_rng(state["rng"], train_loader.generator)
        write_history(history, directory)
        export_best(model, state, directory)
    elif checkpoint.exists():
        raise FileExistsError("Existing E5 checkpoint; use --resume")
    for epoch in range(len(history) + 1, config["training"]["epochs"] + 1):
        started = perf_counter()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device,
                                     amp=config["training"]["amp"], scaler=scaler)
        validation = evaluate(model, val_loader, criterion, device)
        if validation["miou"] is None or not np.isfinite(validation["miou"]):
            raise ValueError("Validation mIoU is undefined/non-finite")
        improved = validation["miou"] > best_miou
        if improved:
            best_miou, best_epoch = validation["miou"], epoch
            best_model = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_validation = validation
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        row = {
            "epoch": epoch, "train_loss": train_loss, "val_loss": validation["loss"],
            "val_miou": validation["miou"], "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": perf_counter() - started,
            "peak_vram_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None,
        }
        history.append(row)
        state = {
            "epoch": epoch, "config_digest": config_digest(config),
            "class_weights": criterion.class_weights.detach().cpu().clone(),
            "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "history": history,
            "best_epoch": best_epoch, "best_miou": best_miou, "best_model": best_model,
            "rng": capture_rng(train_loader.generator), "best_validation": best_validation,
        }
        atomic_torch_save(state, checkpoint)
        write_history(history, directory)
        if improved:
            export_best(model, state, directory)
        write_json(directory / "status.json", {
            "status": "training", "updated_utc": utc_now(), "epoch": epoch,
            "pid": os.getpid(), "epochs": config["training"]["epochs"],
            "best_val_epoch": best_epoch, "best_val_miou": best_miou, "latest": row,
        })
        print(json.dumps(row, allow_nan=False), flush=True)
        if on_epoch:
            on_epoch(row)
    return {"history": history, "best_epoch": best_epoch, "best_miou": best_miou}
