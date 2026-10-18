"""Check real E4 FP32 validation and exact E3 label-grid parity, without test inference."""

import argparse
import json
import traceback

from _bootstrap import ROOT

import torch
from torch.utils.data import DataLoader, Subset

from src.comparison import read_json
from src.config import load_config, project_path
from src.dataset import CamVidDataset
from src.losses import loss_from_config
from src.model import build_model
from src.resolution import CONTROL_CONFIG, RESOLUTION_CONFIG, ResolutionDataset, verify_control
from src.train import evaluate
from src.utils import choose_device, file_sha256, seed_everything, utc_now, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    directory = project_path("results/stage5/E4_smoke")
    directory.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "timestamp_utc": utc_now(), "test_inference": False}
    try:
        config = load_config(RESOLUTION_CONFIG)
        control_metrics, _, _ = verify_control(config)
        control_config = load_config(CONTROL_CONFIG)
        # Label checks inspect annotations only; they never evaluate the test model.
        parity = {}
        torch.set_num_threads(min(4, torch.get_num_threads()))
        for split in ("val", "test"):
            candidate = ResolutionDataset(config, split)
            control = CamVidDataset(control_config, split)
            if candidate.samples != control.samples:
                raise ValueError("E3/E4 manifests differ")
            counts = torch.zeros(11, dtype=torch.int64)
            for index in range(len(candidate)):
                image, mask = candidate.raw_sample(index)
                _, actual = candidate.transform(image, mask)
                _, expected = control.transform(image, mask)
                if not torch.equal(actual, expected):
                    raise ValueError(f"E3/E4 label grid differs at {split}/{index}")
                counts += torch.bincount(actual[actual != 255], minlength=11)
            expected_metrics = control_metrics if split == "test" else read_json(
                project_path(control_config["output_dir"]) / "best_validation_metrics.json"
            )
            if counts.tolist() != expected_metrics["class_pixels"]:
                raise ValueError(f"{split} class pixels differ from the archived E3 result")
            parity[split] = {"samples": len(candidate), "all_labels_equal": True,
                             "class_pixels": counts.tolist()}
        seed_everything(config["seed"])
        device = choose_device(config["device"])
        model, _ = build_model(config, local_files_only=True if args.offline else None)
        model = model.to(device)
        data = ResolutionDataset(config, "val")
        batch = next(iter(DataLoader(data, batch_size=4)))
        metrics = evaluate(model, DataLoader(Subset(data, range(4)), batch_size=4),
                           loss_from_config(config), device)
        report.update(
            status="passed", config_sha256=file_sha256(project_path(RESOLUTION_CONFIG)),
            input_size=config["preprocessing"]["size"], evaluation_size=config["preprocessing"]["evaluation_size"],
            device=str(device), pixels_shape=list(batch["pixel_values"].shape),
            labels_shape=list(batch["labels"].shape), label_parity=parity,
            validation_scope="first_four_validation_samples_only", validation_metrics=metrics,
        )
    except Exception as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
    write_json(directory / "resolution_check.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
