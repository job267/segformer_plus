"""Run one frozen experiment; training/selection use train/val, test is final-only."""

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from time import perf_counter

from _bootstrap import ROOT

import torch
from torch.utils.data import DataLoader

from src.artifacts import export_metric_tables, export_predictions, make_plots
from src.config import load_config, project_path, save_config
from src.dataset import CamVidDataset, contained_path
from src.experiment import experiment_lock, fit
from src.losses import loss_from_config
from src.labels import CLASS_NAMES
from src.model import build_model
from src.train import evaluate
from src.utils import choose_device, file_sha256, seed_everything, seed_worker, utc_now, write_json


def verify_prerequisites(config):
    audit_path = ROOT / "results/stage1/data_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["status"] != "passed" or audit["source_revision"] != config["dataset"]["source_revision"]:
        raise ValueError("A passed audit for this exact dataset revision is required")
    root = project_path(config["dataset"]["root"])
    if Path(audit["root"]).resolve() != root or audit["label_encoding"] != config["dataset"]["label_encoding"]:
        raise ValueError("Dataset configuration differs from the passed audit")
    for split, stats in audit["splits"].items():
        if stats["audited_samples"] != config["dataset"]["expected_counts"][split]:
            raise ValueError(f"Audit sample count differs: {split}")
        manifest = contained_path(root, config["dataset"]["splits"][split])
        if file_sha256(manifest) != stats["manifest_sha256"]:
            raise ValueError(f"Split manifest changed after audit: {split}")
    for item in audit["files"]:
        for key in ("image", "mask"):
            if file_sha256(contained_path(root, item[key])) != item[f"{key}_sha256"]:
                raise ValueError(f"Dataset file changed after audit: {item[key]}")
    for relative in [
        "results/stage1/model_loading.json", "results/stage1/real_smoke/smoke_test.json",
        "results/stage1/overfit/overfit_check.json",
    ]:
        if json.loads((ROOT / relative).read_text(encoding="utf-8"))["status"] != "passed":
            raise ValueError(f"Stage 1 gate not passed: {relative}")
    return file_sha256(audit_path)


def source_manifest():
    paths = sorted(
        [*ROOT.joinpath("src").glob("*.py"), *ROOT.joinpath("scripts").glob("*.py"),
         *ROOT.joinpath("configs").glob("*.yaml"), ROOT / "requirements.txt"]
    )
    return {path.relative_to(ROOT).as_posix(): file_sha256(path) for path in paths}


def run(config, args, directory):
    frozen_path = directory / "config.yaml"
    if args.resume:
        if not frozen_path.is_file() or load_config(frozen_path) != config:
            raise ValueError("Resume requires the exact saved experiment configuration")
    elif any((directory / name).exists() for name in ["config.yaml", "run_info.json", "train_log.csv"]):
        raise FileExistsError("Run directory already contains an experiment; use --resume")
    audit_hash = verify_prerequisites(config)
    sources = source_manifest()
    info_path = directory / "run_info.json"
    if args.resume:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if info["source_sha256"] != sources or info["data_audit_sha256"] != audit_hash:
            raise ValueError("Source or audited data changed; exact resume rejected")
        info.setdefault("resumes", []).append({
            "timestamp_utc": utc_now(), "argv": sys.argv, "pid": os.getpid(),
        })
        info["pid"] = os.getpid()
    else:
        save_config(config, frozen_path)
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        info = {
            "started_utc": utc_now(), "argv": [sys.executable, *sys.argv],
            "seed": config["seed"], "pid": os.getpid(), "git_commit": git.stdout.strip() or None,
            "checkpoint": config["model"]["checkpoint"], "revision": config["model"]["revision"],
            "source_sha256": sources, "data_audit_sha256": audit_hash,
            "test_policy": "One completed final evaluation of the validation-selected checkpoint.",
        }
    args.accepted = True
    write_json(info_path, info)
    environment_path = directory / ("environment_resume.json" if args.resume else "environment.json")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_environment.py"), "--output", str(environment_path)],
        cwd=ROOT, check=True,
    )
    seed_everything(config["seed"])
    device = choose_device(config["device"])
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    generator = torch.Generator().manual_seed(config["seed"])
    kwargs = dict(
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"], worker_init_fn=seed_worker,
    )
    train_data, val_data = CamVidDataset(config, "train"), CamVidDataset(config, "val")
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **kwargs)
    model, _ = build_model(
        config, local_files_only=True if args.offline else None,
        report_path=directory / ("model_loading_resume.json" if args.resume else "model_loading.json"),
    )
    model = model.to(device)
    write_json(directory / "status.json", {
        "status": "training", "updated_utc": utc_now(), "pid": os.getpid(),
        "epochs": config["training"]["epochs"], "resuming": args.resume,
    })
    summary = fit(
        model, train_loader, val_loader, config, device, directory,
        resume=args.resume and (directory / "last_checkpoint/state.pt").is_file(),
    )
    # Model selection is finished before any test label enters model evaluation.
    state = torch.load(directory / "last_checkpoint/state.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["best_model"], strict=True)
    weights_hash = file_sha256(directory / "best_model/model.safetensors")
    metrics_path = directory / "metrics.json"
    test_data = CamVidDataset(config, "test")
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics["checkpoint_sha256"] != weights_hash:
            raise ValueError("Existing test results refer to a different checkpoint")
    else:
        write_json(directory / "status.json", {
            "status": "evaluating_test", "updated_utc": utc_now(),
            "epoch": len(summary["history"]), "best_val_epoch": summary["best_epoch"],
        })
        journal_path = directory / "test_evaluation.json"
        journal = json.loads(journal_path.read_text(encoding="utf-8")) if journal_path.exists() else {"attempts": []}
        journal["attempts"].append({"started_utc": utc_now(), "checkpoint_sha256": weights_hash})
        write_json(journal_path, journal)
        metrics = evaluate(
            model, DataLoader(test_data, shuffle=False, **kwargs), loss_from_config(config), device
        )
        metrics.update(
            experiment_id=config["experiment_id"], dataset="CamVid_11", split="test",
            seed=config["seed"], best_checkpoint_selected_by="val_miou",
            best_val_epoch=summary["best_epoch"], best_val_miou=summary["best_miou"],
            test_miou=metrics["miou"], checkpoint_sha256=weights_hash,
            ignore_index=config["dataset"]["ignore_index"],
            metric_definition=(
                f"FP32; resized {config['preprocessing']['size']} labels; "
                "dataset-level confusion; union-zero classes excluded."
            ),
            class_names=list(CLASS_NAMES),
        )
        write_json(metrics_path, metrics)
        journal["attempts"][-1]["completed_utc"] = utc_now()
        write_json(journal_path, journal)
    export_metric_tables(metrics, directory)
    make_plots(summary["history"], metrics, directory)
    sample_ids = export_predictions(model, test_data, device, directory)
    write_json(directory / "prediction_samples.json", {"selection": "six evenly spaced test-list positions", "sample_ids": sample_ids})
    write_json(directory / "status.json", {
        "status": "completed", "updated_utc": utc_now(), "epoch": len(summary["history"]),
        "epochs": config["training"]["epochs"], "best_val_epoch": summary["best_epoch"],
        "best_val_miou": summary["best_miou"], "test_miou": metrics["test_miou"],
    })
    print(json.dumps({"status": "completed", "best_val_miou": summary["best_miou"], "test_miou": metrics["test_miou"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.accepted = False
    config = load_config(args.config)
    directory = project_path(config["output_dir"])
    started = perf_counter()
    try:
        with experiment_lock(directory):
            try:
                run(config, args, directory)
            except BaseException as error:
                status_path = directory / ("status.json" if args.accepted else "launch_error.json")
                failure = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
                failure.update({
                    "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    "updated_utc": utc_now(), "error_type": type(error).__name__,
                    "error": str(error), "traceback": traceback.format_exc(),
                    "elapsed_seconds": perf_counter() - started,
                })
                write_json(status_path, failure)
                raise
    except BaseException:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
