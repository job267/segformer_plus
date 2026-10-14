"""Diagnostic training on 20-50 train samples; never reads validation/test labels."""

import argparse
import csv
import json
import traceback
from copy import deepcopy
from time import perf_counter

import torch
from torch.utils.data import DataLoader, Subset

from _bootstrap import ROOT
from src.config import load_config, project_path, save_config
from src.dataset import CamVidDataset
from src.labels import save_preview
from src.losses import loss_from_config
from src.model import build_model
from src.train import evaluate, make_scaler, train_one_epoch
from src.utils import choose_device, seed_everything, seed_worker, utc_now, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Diagnostic override")
    parser.add_argument("--minimum-loss-decrease", type=float, default=0.2)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output-dir", default="results/stage1/overfit")
    args = parser.parse_args()
    if not 20 <= args.samples <= 50 or args.epochs < 1 or args.learning_rate <= 0:
        parser.error("Use 20-50 samples, positive epochs and a positive learning rate")
    if not 0 < args.minimum_loss_decrease < 1:
        parser.error("--minimum-loss-decrease must be in (0, 1)")
    directory = project_path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running", "timestamp_utc": utc_now(), "validation_scope": "train_subset_only",
        "note": "Learning diagnostic; these values are not validation/test experiment results.",
        "arguments": vars(args),
    }
    started = perf_counter()
    try:
        config = load_config(args.config)
        seed_everything(config["seed"])
        device = choose_device(config["device"])
        dataset = CamVidDataset(config, "train", disable_augmentation=True)
        if len(dataset) < args.samples:
            raise ValueError("Not enough training samples for the requested diagnostic")
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(config["seed"]))[:args.samples].tolist()
        subset = Subset(dataset, indices)
        report["sample_ids"] = [dataset.samples[index].sample_id for index in indices]
        report["preview_sample_ids"] = report["sample_ids"][:3]
        for index in indices[:3]:
            image, mask = dataset.raw_sample(index)
            save_preview(image, mask, directory / "previews", dataset.samples[index].sample_id)
        generator = torch.Generator().manual_seed(config["seed"])
        loader_kwargs = {
            "batch_size": config["training"]["batch_size"],
            "num_workers": config["training"]["num_workers"], "worker_init_fn": seed_worker,
        }
        train_loader = DataLoader(subset, shuffle=True, generator=generator, **loader_kwargs)
        diagnostic_loader = DataLoader(subset, shuffle=False, **loader_kwargs)
        model, _ = build_model(
            config, local_files_only=True if args.offline else None,
            report_path=directory / "model_loading.json",
        )
        model = model.to(device)
        criterion = loss_from_config(config)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=config["training"]["weight_decay"]
        )
        scaler = make_scaler(device, config["training"]["amp"])
        report["initial"] = evaluate(model, diagnostic_loader, criterion, device)
        diagnostic_config = deepcopy(config)
        diagnostic_config.update(experiment_id="diagnostic_overfit", output_dir=args.output_dir)
        diagnostic_config["training"].update(
            epochs=args.epochs, learning_rate=args.learning_rate
        )
        diagnostic_config["augmentation"]["enabled"] = False
        if args.offline:
            diagnostic_config["model"]["local_files_only"] = True
        save_config(config, directory / "source_config.yaml")
        save_config(diagnostic_config, directory / "config.yaml")
        with (directory / "overfit_log.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["epoch", "train_loss", "subset_eval_loss", "subset_miou", "epoch_seconds"]
            )
            writer.writeheader()
            for epoch in range(1, args.epochs + 1):
                epoch_started = perf_counter()
                train_loss = train_one_epoch(
                    model, train_loader, optimizer, criterion, device,
                    amp=config["training"]["amp"], scaler=scaler,
                )
                final = evaluate(model, diagnostic_loader, criterion, device)
                row = {
                    "epoch": epoch, "train_loss": train_loss,
                    "subset_eval_loss": final["loss"], "subset_miou": final["miou"],
                    "epoch_seconds": perf_counter() - epoch_started,
                }
                writer.writerow(row)
                handle.flush()
                print(json.dumps(row), flush=True)
        decrease = 1 - final["loss"] / report["initial"]["loss"]
        report.update(final=final, relative_loss_decrease=decrease, device=str(device))
        report["status"] = (
            "passed" if decrease >= args.minimum_loss_decrease
            and final["miou"] > report["initial"]["miou"] else "failed"
        )
        if report["status"] == "failed":
            report["error"] = "Learning criterion not met; inspect labels/loss/transforms before E0."
    except Exception as error:
        report.update(
            status="failed", error_type=type(error).__name__, error=str(error),
            traceback=traceback.format_exc(),
        )
    report["elapsed_seconds"] = perf_counter() - started
    write_json(directory / "overfit_check.json", report)
    log = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    (directory / "overfit_check.log").write_text(log + "\n", encoding="utf-8")
    print(log)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
