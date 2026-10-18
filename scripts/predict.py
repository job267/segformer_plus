"""Predict a local image or directory, exporting native-size class IDs, colors and overlays."""

import argparse
import json

from _bootstrap import ROOT
from src.config import project_path
from src.inference import load_run_model, save_demo_index, write_prediction
from src.utils import choose_device, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Image file or directory")
    parser.add_argument("--run-dir", default="results/E4_resolution")
    parser.add_argument("--output-dir", default="results/inference_demo")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    source, directory = project_path(args.input), project_path(args.output_dir)
    if not source.exists():
        parser.error(f"Input does not exist: {source}")
    if directory.exists() and any(directory.iterdir()):
        parser.error("Choose an empty output directory to preserve previous predictions")
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    paths = sorted(path for path in source.iterdir() if path.is_file() and path.suffix.lower() in extensions) if source.is_dir() else [source]
    if not paths:
        parser.error("No supported images found")
    device = choose_device(args.device)
    if args.precision == "fp16" and device.type != "cuda":
        parser.error("FP16 requires CUDA")
    seed_everything(42)
    model, config, metrics = load_run_model(args.run_dir, device)
    items = [write_prediction(model, path, config, device, args.precision,
                              directory / "predictions" / f"{index:03d}_{path.stem}")
             for index, path in enumerate(paths)]
    metadata = {
        "experiment_id": config["experiment_id"], "checkpoint_sha256": metrics["checkpoint_sha256"],
        "precision": args.precision, "device": str(device), "input_size": config["preprocessing"]["size"],
        "output_grid": "original image size; logits resized before argmax; class IDs 0-10",
    }
    save_demo_index(directory, items, metadata)
    print(json.dumps({"images": len(items), "preview": str(directory / "index.html")}, indent=2))


if __name__ == "__main__":
    main()
