"""Validation-selected training, transactional epoch archives, and exact epoch resume."""

import csv
import hashlib
import json
import os
import random
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from src.losses import loss_from_config
from src.train import evaluate, make_scaler, train_one_epoch
from src.utils import utc_now, write_json

LOG_FIELDS = [
    "epoch", "train_loss", "val_loss", "val_miou",
    "learning_rate", "epoch_time_sec", "peak_vram_mb",
]


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


@contextmanager
def experiment_lock(directory):
    """OS-released file lock also works after an unexpected process exit."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "training.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def capture_rng(generator):
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(), "generator": generator.get_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    generator.set_state(state["generator"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def write_history(history, directory):
    path = Path(directory) / "train_log.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(path)


def export_best(model, state, directory):
    # Restoring the best head/BatchNorm buffers must not alter the ongoing training model.
    best = deepcopy(model).cpu()
    best.load_state_dict(state["best_model"], strict=True)
    best.save_pretrained(Path(directory) / "best_model", safe_serialization=True)
    write_json(Path(directory) / "best_checkpoint.json", {
        "selected_by": "val_miou", "best_val_epoch": state["best_epoch"],
        "best_val_miou": state["best_miou"],
    })
    write_json(Path(directory) / "best_validation_metrics.json", state["best_validation"])


def fit(model, train_loader, val_loader, config, device, directory, *, resume=False, on_epoch=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "last_checkpoint" / "state.pt"
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    criterion = loss_from_config(config)
    scaler = make_scaler(device, config["training"]["amp"])
    history, best_miou, best_epoch, best_model, best_validation = [], -1.0, None, None, None
    if resume:
        # Only load the trusted checkpoint written by this project.
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["config_digest"] != config_digest(config):
            raise ValueError("Resume configuration differs from the frozen run configuration")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        history, best_miou, best_epoch, best_model = (
            state["history"], state["best_miou"], state["best_epoch"], state["best_model"]
        )
        best_validation = state["best_validation"]
        restore_rng(state["rng"], train_loader.generator)
        write_history(history, directory)
        export_best(model, state, directory)
    elif checkpoint.exists():
        raise FileExistsError("Existing checkpoint: use --resume, never silently overwrite")
    for epoch in range(len(history) + 1, config["training"]["epochs"] + 1):
        started = perf_counter()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            amp=config["training"]["amp"], scaler=scaler,
        )
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
            "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "history": history, "best_epoch": best_epoch, "best_miou": best_miou,
            "best_model": best_model, "rng": capture_rng(train_loader.generator),
            "best_validation": best_validation,
        }
        # This checkpoint is the authoritative commit of the completed epoch.
        atomic_torch_save(state, checkpoint)
        write_history(history, directory)
        if improved:
            export_best(model, state, directory)
        write_json(directory / "status.json", {
            "status": "training", "updated_utc": utc_now(), "epoch": epoch,
            "pid": os.getpid(),
            "epochs": config["training"]["epochs"], "best_val_epoch": best_epoch,
            "best_val_miou": best_miou, "latest": row,
        })
        print(json.dumps(row, allow_nan=False), flush=True)
        if on_epoch:
            on_epoch(row)
    return {"history": history, "best_epoch": best_epoch, "best_miou": best_miou}
