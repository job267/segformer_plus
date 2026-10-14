"""One forward/backward/optimizer step; synthetic checks are explicitly distinguished."""

import argparse
import json
import traceback
from time import perf_counter

import torch
from torch.utils.data import DataLoader

from _bootstrap import ROOT
from src.config import load_config, project_path, save_config
from src.dataset import CamVidDataset
from src.losses import loss_from_config
from src.metrics import resize_logits
from src.model import build_model
from src.train import make_scaler
from src.utils import choose_device, seed_everything, utc_now, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--synthetic", action="store_true", help="Logic check, not CamVid acceptance")
    parser.add_argument("--random-init", action="store_true", help="Allowed only with --synthetic")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--size", type=int, help="Synthetic-only square input size")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    if args.random_init and not args.synthetic:
        parser.error("--random-init requires --synthetic; real data checks must use pretrained weights")
    if args.size and (not args.synthetic or args.size < 32):
        parser.error("--size is synthetic-only and must be >=32")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    directory = project_path(args.output_dir or (
        "results/stage1/synthetic_smoke" if args.synthetic else "results/stage1/real_smoke"
    ))
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running", "timestamp_utc": utc_now(), "command_scope": "one_optimizer_step",
        "validation_scope": "synthetic_logic_only" if args.synthetic else "CamVid_train_batch",
        "random_init": args.random_init,
    }
    started = perf_counter()
    try:
        config = load_config(args.config)
        if args.size:
            config["preprocessing"]["size"] = [args.size, args.size]
        save_config(config, directory / "config.yaml")
        seed_everything(config["seed"])
        device = choose_device(args.device or config["device"])
        if device.type == "cpu":
            torch.set_num_threads(min(4, torch.get_num_threads()))
        if args.synthetic:
            height, width = config["preprocessing"]["size"]
            pixels = torch.randn(args.batch_size, 3, height, width)
            labels = torch.randint(0, 11, (args.batch_size, height, width))
            labels[:, :2, :2] = config["dataset"]["ignore_index"]
            sample_ids = [f"synthetic_{i}" for i in range(args.batch_size)]
        else:
            dataset = CamVidDataset(config, "train")
            batch = next(iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=False)))
            pixels, labels, sample_ids = batch["pixel_values"], batch["labels"], batch["sample_id"]
        model, loading = build_model(
            config, random_init=args.random_init,
            local_files_only=True if args.offline else None,
            report_path=directory / "model_loading.json",
        )
        model = model.to(device).train()
        pixels, labels = pixels.to(device), labels.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )
        criterion = loss_from_config(config)
        amp = config["training"]["amp"] and device.type == "cuda"
        scaler = make_scaler(device, amp, initial_scale=1024.0)
        before = model.decode_head.classifier.weight.detach().clone()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(pixel_values=pixels).logits
            loss = criterion(logits, labels)
        if logits.shape[1] != 11 or not torch.isfinite(loss) or not torch.isfinite(logits).all():
            raise FloatingPointError("Invalid logits or non-finite loss")
        if not (labels != criterion.ignore_index).any():
            raise ValueError("The selected batch contains only ignored pixels")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise FloatingPointError("Missing/non-finite gradients")
        gradient_norm = float(torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(gradient.float()) for gradient in gradients])
        ))
        if gradient_norm <= 0:
            raise RuntimeError("All gradients are zero")
        scaler.step(optimizer)
        scaler.update()
        changed = not torch.equal(before, model.decode_head.classifier.weight.detach())
        if not changed:
            raise RuntimeError("Optimizer did not update the segmentation classifier")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        report.update(
            status="passed", device=str(device), amp=amp, seed=config["seed"],
            loss=float(loss.detach()), gradient_norm=gradient_norm, optimizer_updated=changed,
            logits_shape=list(logits.shape),
            resized_logits_shape=list(resize_logits(logits.detach(), labels).shape),
            labels_shape=list(labels.shape), sample_ids=sample_ids,
            initialization=loading["initialization"],
            peak_vram_mb=(
                torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
            ),
        )
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        report["traceback"] = traceback.format_exc()
    report["elapsed_seconds"] = perf_counter() - started
    write_json(directory / "smoke_test.json", report)
    log = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    (directory / "smoke_test.log").write_text(log + "\n", encoding="utf-8")
    print(log)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
