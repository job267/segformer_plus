"""Prepare train-only E5 weights and check a real full-size optimizer step."""

import argparse
import json
import traceback

from _bootstrap import ROOT
from train import verify_prerequisites

import torch
from torch.utils.data import DataLoader, Subset

from src.config import load_config, project_path, save_config
from src.model import build_model
from src.resolution import ResolutionDataset
from src.train import evaluate, make_scaler
from src.utils import choose_device, file_sha256, seed_everything, utc_now, write_json
from src.weighted import WEIGHTED_CONFIG, WEIGHTS_PATH, WeightedSegmentationLoss, prepare_weights, verify_e4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    directory = project_path("results/stage6/E5_smoke")
    directory.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "timestamp_utc": utc_now(), "scope": "one_train_step_and_four_val_samples", "test_inference": False}
    try:
        config = load_config(WEIGHTED_CONFIG)
        _, _, info = verify_e4(config)
        if verify_prerequisites(config) != info["data_audit_sha256"]:
            raise ValueError("E5 data audit differs from E4")
        record = prepare_weights(config)
        save_config(config, directory / "config.yaml")
        seed_everything(config["seed"])
        device = choose_device(config["device"])
        model, loading = build_model(config, local_files_only=True if args.offline else None,
                                     report_path=directory / "model_loading.json")
        model = model.to(device).train()
        criterion = WeightedSegmentationLoss(record["weights"], config["training"]["dice_weight"]).to(device)
        batch = next(iter(DataLoader(ResolutionDataset(config, "train"), batch_size=4)))
        pixels, labels = batch["pixel_values"].to(device), batch["labels"].to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                                      weight_decay=config["training"]["weight_decay"])
        scaler = make_scaler(device, config["training"]["amp"], initial_scale=1024.0)
        before = model.decode_head.classifier.weight.detach().clone()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda" and config["training"]["amp"]):
            loss = criterion(model(pixel_values=pixels).logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite E5 loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise FloatingPointError("Missing or non-finite E5 gradients")
        norm = float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.float()) for g in gradients])))
        scaler.step(optimizer)
        scaler.update()
        if norm <= 0 or torch.equal(before, model.decode_head.classifier.weight.detach()):
            raise ValueError("Weighted loss did not update the model")
        val = ResolutionDataset(config, "val")
        metrics = evaluate(model, DataLoader(Subset(val, range(4)), batch_size=4), criterion, device)
        report.update(
            status="passed", config_sha256=file_sha256(project_path(WEIGHTED_CONFIG)),
            class_weights_sha256=file_sha256(project_path(WEIGHTS_PATH)),
            initialization=loading["initialization"], device=str(device), optimizer_updated=True,
            train_loss=float(loss.detach()), gradient_norm=norm, labels_shape=list(labels.shape),
            validation_size=[384, 384], validation_miou=metrics["miou"],
            peak_vram_mb=torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None,
        )
    except Exception as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
    write_json(directory / "weighted_check.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
