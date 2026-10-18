"""One frozen E4/E5 replicate; original engines and validation-only selection."""

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
from src.multiseed import (
    FAMILIES, PREFLIGHT_PATH, SEEDS, model_state_sha256, replicate_config,
    validate_replicate, verify_references, write_multiseed_comparison,
)
from src.resolution import ResolutionDataset, validate_evaluation_grids
from src.train import evaluate
from src.utils import choose_device, file_sha256, seed_everything, seed_worker, utc_now, write_json
from src.weighted import WEIGHTED_CONFIG, WEIGHTS_PATH, WeightedSegmentationLoss, prepare_weights
from src.weighted_training import fit_weighted


def run(config, args, directory):
    validate_replicate(config, args.family, args.seed)
    frozen = directory / "config.yaml"
    if args.resume:
        if not frozen.is_file() or load_config(frozen) != config:
            raise ValueError("Resume requires the exact saved replicate configuration")
    elif any((directory / name).exists() for name in ("config.yaml", "run_info.json", "train_log.csv")):
        raise FileExistsError("Replicate already exists; use --resume")
    references = verify_references()
    reference = references[args.family]
    audit_hash = verify_prerequisites(config)
    weights = prepare_weights(load_config(WEIGHTED_CONFIG))
    preflight_path = project_path(PREFLIGHT_PATH)
    preflight = read_json(preflight_path)
    sources = source_manifest()
    weights_hash = file_sha256(project_path(WEIGHTS_PATH)) if args.family == "E5" else None
    if preflight["status"] != "passed" or preflight["source_sha256"] != sources:
        raise ValueError("A passed preflight for the exact current implementation is required")
    if audit_hash != preflight["data_audit_sha256"] or audit_hash != reference["info"]["data_audit_sha256"]:
        raise ValueError("Replicate data audit differs from the frozen preflight")
    if preflight["class_weights_sha256"] != file_sha256(project_path(WEIGHTS_PATH)):
        raise ValueError("Frozen class weights changed")
    preflight_hash = file_sha256(preflight_path)
    reference_hash = reference["metrics"]["checkpoint_sha256"]
    info_path = directory / "run_info.json"
    if args.resume:
        info = read_json(info_path)
        if info["source_sha256"] != sources or info["data_audit_sha256"] != audit_hash:
            raise ValueError("Source or data changed; exact replicate resume rejected")
        if info["preflight_sha256"] != preflight_hash or info["class_weights_sha256"] != weights_hash or info["reference_checkpoint_sha256"] != reference_hash:
            raise ValueError("Replicate preflight, class weights or reference changed")
        info.setdefault("resumes", []).append({"timestamp_utc": utc_now(), "pid": os.getpid()})
        info["pid"] = os.getpid()
    else:
        save_config(config, frozen)
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        info = {
            "started_utc": utc_now(), "argv": [sys.executable, *sys.argv], "pid": os.getpid(),
            "git_commit": git.stdout.strip() or None, "source_sha256": sources,
            "data_audit_sha256": audit_hash, "seed": args.seed, "family": args.family,
            "checkpoint": config["model"]["checkpoint"], "revision": config["model"]["revision"],
            "initialization": "Independent pinned ImageNet MiT-B0; fresh 11-class head",
            "input_size": [512, 512], "train_label_size": [512, 512], "evaluation_size": [384, 384],
            "reference_checkpoint_sha256": reference_hash, "class_weights_sha256": weights_hash,
            "preflight_sha256": preflight_hash,
            "test_policy": "One completed final evaluation after validation selection; no best-seed selection.",
        }
    args.accepted = True
    write_json(info_path, info)
    subprocess.run([sys.executable, str(ROOT / "scripts/check_environment.py"), "--output",
                    str(directory / ("environment_resume.json" if args.resume else "environment.json"))], cwd=ROOT, check=True)
    seed_everything(args.seed)
    device = choose_device(config["device"])
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    generator = torch.Generator().manual_seed(args.seed)
    kwargs = dict(batch_size=config["training"]["batch_size"],
                  num_workers=config["training"]["num_workers"], worker_init_fn=seed_worker)
    train_data, val_data = ResolutionDataset(config, "train"), ResolutionDataset(config, "val")
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **kwargs)
    model, _ = build_model(config, local_files_only=True if args.offline else None,
                           report_path=directory / ("model_loading_resume.json" if args.resume else "model_loading.json"))
    initial_hash = model_state_sha256(model)
    if initial_hash != preflight["initial_hashes"][str(args.seed)][args.family]:
        raise ValueError("Replicate initial model differs from paired preflight")
    if args.resume and info.get("initial_model_sha256") != initial_hash:
        raise ValueError("Saved initial model identity changed")
    info["initial_model_sha256"] = initial_hash
    write_json(info_path, info)
    model = model.to(device)
    criterion = (WeightedSegmentationLoss(weights["weights"], config["training"]["dice_weight"])
                 if args.family == "E5" else loss_from_config(config)).to(device)
    write_json(directory / "status.json", {
        "status": "training", "updated_utc": utc_now(), "pid": os.getpid(),
        "epochs": config["training"]["epochs"], "resuming": args.resume,
    })
    options = {"resume": args.resume and (directory / "last_checkpoint/state.pt").is_file()}
    if args.family == "E5":
        summary = fit_weighted(model, train_loader, val_loader, config, device, directory, criterion, **options)
    else:
        summary = fit(model, train_loader, val_loader, config, device, directory, **options)
    state = torch.load(directory / "last_checkpoint/state.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["best_model"], strict=True)
    selected_hash = file_sha256(directory / "best_model/model.safetensors")
    path = directory / "metrics.json"
    if path.exists():
        metrics = read_json(path)
        if metrics["checkpoint_sha256"] != selected_hash or metrics["evaluation_size"] != [384, 384] or metrics.get("class_weights_sha256") != weights_hash:
            raise ValueError("Saved test metrics refer to different weights or policy")
    else:
        write_json(directory / "status.json", {
            "status": "evaluating_test", "updated_utc": utc_now(), "epoch": len(summary["history"]),
        })
        journal_path = directory / "test_evaluation.json"
        journal = read_json(journal_path) if journal_path.exists() else {"attempts": []}
        journal["attempts"].append({"started_utc": utc_now(), "checkpoint_sha256": selected_hash})
        write_json(journal_path, journal)
        metrics = evaluate(model, DataLoader(ResolutionDataset(config, "test"), shuffle=False, **kwargs), criterion, device)
        metrics.update(
            experiment_id=config["experiment_id"], family=args.family, dataset="CamVid_11", split="test", seed=args.seed,
            best_checkpoint_selected_by="val_miou", best_val_epoch=summary["best_epoch"],
            best_val_miou=summary["best_miou"], test_miou=metrics["miou"], checkpoint_sha256=selected_hash,
            ignore_index=255, class_names=list(CLASS_NAMES), input_size=[512, 512], evaluation_size=[384, 384],
            reference_checkpoint_sha256=reference_hash, class_weights_sha256=weights_hash,
            metric_definition="FP32; original masks directly resized to 384x384; logits resized before argmax; dataset-level confusion.",
        )
        write_json(path, metrics)
        journal["attempts"][-1]["completed_utc"] = utc_now()
        write_json(journal_path, journal)
    validate_evaluation_grids({"val": state["best_validation"], "test": metrics},
                              {"val": reference["validation"], "test": reference["metrics"]})
    export_metric_tables(metrics, directory)
    make_plots(summary["history"], metrics, directory)
    samples = export_predictions(model, ResolutionDataset(config, "test"), device, directory)
    write_json(directory / "prediction_samples.json", {"selection": "six evenly spaced test-list positions", "sample_ids": samples})
    write_json(directory / "status.json", {
        "status": "completed", "updated_utc": utc_now(), "epoch": len(summary["history"]),
        "epochs": config["training"]["epochs"], "best_val_epoch": summary["best_epoch"],
        "best_val_miou": summary["best_miou"], "test_miou": metrics["test_miou"],
    })
    comparison = write_multiseed_comparison(allow_incomplete=True)
    print(json.dumps({"status": "completed", "experiment_id": config["experiment_id"],
                      "completed_runs": comparison["completed_runs"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=FAMILIES)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS[1:])
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.accepted = False
    config = replicate_config(args.family, args.seed)
    directory = project_path(config["output_dir"])
    started = perf_counter()
    try:
        with experiment_lock(directory):
            try:
                run(config, args, directory)
            except BaseException as error:
                path = directory / ("status.json" if args.accepted else "launch_error.json")
                status = read_json(path) if path.exists() else {}
                status.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                              updated_utc=utc_now(), error_type=type(error).__name__, error=str(error),
                              traceback=traceback.format_exc(), elapsed_seconds=perf_counter() - started)
                write_json(path, status)
                raise
    except BaseException:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
