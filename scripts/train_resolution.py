"""Run E4 independently, selecting and evaluating on the frozen 384x384 label grid."""

import argparse
import json
import os
import subprocess
import sys
import traceback
from time import perf_counter

from _bootstrap import ROOT
from train import source_manifest, verify_prerequisites

import torch
from torch.utils.data import DataLoader

from src.artifacts import export_metric_tables, export_predictions, make_plots
from src.comparison import read_json
from src.config import load_config, project_path, save_config
from src.experiment import experiment_lock, fit
from src.labels import CLASS_NAMES
from src.losses import loss_from_config
from src.model import build_model
from src.resolution import (
    CONTROL_CONFIG, RESOLUTION_CONFIG, ResolutionDataset, validate_evaluation_grids,
    verify_control, write_resolution_comparison,
)
from src.train import evaluate
from src.utils import choose_device, file_sha256, seed_everything, seed_worker, utc_now, write_json


def run(config, args, directory):
    frozen = directory / "config.yaml"
    if args.resume:
        if not frozen.is_file() or load_config(frozen) != config:
            raise ValueError("Resume requires the exact saved E4 configuration")
    elif any((directory / name).exists() for name in ("config.yaml", "run_info.json", "train_log.csv")):
        raise FileExistsError("E4 already exists; use --resume")
    control, _, control_info = verify_control(config)
    audit_hash = verify_prerequisites(config)
    if audit_hash != control_info["data_audit_sha256"]:
        raise ValueError("E4 must use the E3 dataset audit")
    checks = project_path("results/stage5/E4_smoke")
    smoke, grid_check = read_json(checks / "smoke_test.json"), read_json(checks / "resolution_check.json")
    if smoke["status"] != "passed" or grid_check["status"] != "passed":
        raise ValueError("E4 real training and validation checks must pass first")
    if load_config(checks / "config.yaml") != config or grid_check["config_sha256"] != file_sha256(project_path(RESOLUTION_CONFIG)):
        raise ValueError("E4 checks refer to a different configuration")
    if smoke["random_init"] or not smoke["optimizer_updated"] or smoke["labels_shape"] != [4, 512, 512]:
        raise ValueError("E4 check must use real pretrained training at the full batch size")
    sources = source_manifest()
    info_path = directory / "run_info.json"
    if args.resume:
        info = read_json(info_path)
        if info["source_sha256"] != sources or info["data_audit_sha256"] != audit_hash:
            raise ValueError("Source or data changed; exact E4 resume rejected")
        if info["control_checkpoint_sha256"] != control["checkpoint_sha256"]:
            raise ValueError("E3 control checkpoint changed")
        info.setdefault("resumes", []).append({"timestamp_utc": utc_now(), "pid": os.getpid()})
        info["pid"] = os.getpid()
    else:
        save_config(config, frozen)
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        info = {
            "started_utc": utc_now(), "argv": [sys.executable, *sys.argv], "pid": os.getpid(),
            "git_commit": git.stdout.strip() or None, "source_sha256": sources,
            "data_audit_sha256": audit_hash, "seed": config["seed"],
            "checkpoint": config["model"]["checkpoint"], "revision": config["model"]["revision"],
            "initialization": "Independent pinned ImageNet MiT-B0; fresh 11-class head",
            "input_size": [512, 512], "train_label_size": [512, 512], "evaluation_size": [384, 384],
            "control_experiment": "E3_combined", "control_checkpoint_sha256": control["checkpoint_sha256"],
            "test_policy": "One completed final evaluation after validation selection; no test tuning.",
        }
    args.accepted = True
    write_json(info_path, info)
    subprocess.run([
        sys.executable, str(ROOT / "scripts/check_environment.py"), "--output",
        str(directory / ("environment_resume.json" if args.resume else "environment.json")),
    ], cwd=ROOT, check=True)
    seed_everything(config["seed"])
    device = choose_device(config["device"])
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    generator = torch.Generator().manual_seed(config["seed"])
    kwargs = dict(batch_size=config["training"]["batch_size"],
                  num_workers=config["training"]["num_workers"], worker_init_fn=seed_worker)
    train_data, val_data = ResolutionDataset(config, "train"), ResolutionDataset(config, "val")
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **kwargs)
    model, _ = build_model(config, local_files_only=True if args.offline else None,
                           report_path=directory / ("model_loading_resume.json" if args.resume else "model_loading.json"))
    model = model.to(device)
    write_json(directory / "status.json", {
        "status": "training", "updated_utc": utc_now(), "pid": os.getpid(),
        "epochs": config["training"]["epochs"], "resuming": args.resume,
    })
    write_resolution_comparison(config, allow_incomplete=True)
    summary = fit(model, train_loader, val_loader, config, device, directory,
                  resume=args.resume and (directory / "last_checkpoint/state.pt").is_file())
    state = torch.load(directory / "last_checkpoint/state.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["best_model"], strict=True)
    weights_hash = file_sha256(directory / "best_model/model.safetensors")
    metrics_path = directory / "metrics.json"
    if metrics_path.exists():
        metrics = read_json(metrics_path)
        if metrics["checkpoint_sha256"] != weights_hash or metrics.get("evaluation_size") != [384, 384]:
            raise ValueError("Saved E4 test metrics refer to different weights or label grid")
    else:
        write_json(directory / "status.json", {
            "status": "evaluating_test", "updated_utc": utc_now(), "epoch": len(summary["history"]),
            "best_val_epoch": summary["best_epoch"],
        })
        journal_path = directory / "test_evaluation.json"
        journal = read_json(journal_path) if journal_path.exists() else {"attempts": []}
        journal["attempts"].append({"started_utc": utc_now(), "checkpoint_sha256": weights_hash})
        write_json(journal_path, journal)
        test_data = ResolutionDataset(config, "test")
        metrics = evaluate(model, DataLoader(test_data, shuffle=False, **kwargs), loss_from_config(config), device)
        metrics.update(
            experiment_id=config["experiment_id"], dataset="CamVid_11", split="test", seed=config["seed"],
            best_checkpoint_selected_by="val_miou", best_val_epoch=summary["best_epoch"],
            best_val_miou=summary["best_miou"], test_miou=metrics["miou"], checkpoint_sha256=weights_hash,
            ignore_index=255, class_names=list(CLASS_NAMES), input_size=[512, 512], evaluation_size=[384, 384],
            control_checkpoint_sha256=control["checkpoint_sha256"],
            metric_definition="FP32; original masks resized directly to 384x384; logits resized before argmax; dataset-level confusion.",
        )
        write_json(metrics_path, metrics)
        journal["attempts"][-1]["completed_utc"] = utc_now()
        write_json(journal_path, journal)
    validate_evaluation_grids(
        {"val": state["best_validation"], "test": metrics},
        {"val": read_json(project_path(load_config(CONTROL_CONFIG)["output_dir"]) / "best_validation_metrics.json"),
         "test": control},
    )
    export_metric_tables(metrics, directory)
    make_plots(summary["history"], metrics, directory)
    samples = export_predictions(model, ResolutionDataset(config, "test"), device, directory)
    write_json(directory / "prediction_samples.json", {"selection": "six evenly spaced test-list positions", "sample_ids": samples})
    write_json(directory / "status.json", {
        "status": "completed", "updated_utc": utc_now(), "epoch": len(summary["history"]),
        "epochs": config["training"]["epochs"], "best_val_epoch": summary["best_epoch"],
        "best_val_miou": summary["best_miou"], "test_miou": metrics["test_miou"],
    })
    rows = write_resolution_comparison(config)
    print(json.dumps({"status": "completed", "comparison": rows}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.accepted = False
    config = load_config(RESOLUTION_CONFIG)
    directory = project_path(config["output_dir"])
    started = perf_counter()
    try:
        with experiment_lock(directory):
            try:
                run(config, args, directory)
            except BaseException as error:
                path = directory / ("status.json" if args.accepted else "launch_error.json")
                status = read_json(path) if path.exists() else {}
                status.update(
                    status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    updated_utc=utc_now(), error_type=type(error).__name__, error=str(error),
                    traceback=traceback.format_exc(), elapsed_seconds=perf_counter() - started,
                )
                write_json(path, status)
                raise
    except BaseException:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
