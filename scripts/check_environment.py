"""Read-only environment probes and Stage 0 configuration validation."""

import argparse
import importlib
import importlib.metadata
import inspect
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="environment.json")
    args = parser.parse_args()
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "python_executable": Path(sys.executable).name,
        "platform": platform.platform(), "packages": {}, "errors": [],
    }
    for distribution, module in (
        ("torch", "torch"), ("transformers", "transformers"), ("numpy", "numpy"),
        ("Pillow", "PIL"), ("PyYAML", "yaml"), ("huggingface-hub", "huggingface_hub"),
        ("safetensors", "safetensors"), ("requests", "requests"),
    ):
        try:
            report["packages"][distribution] = importlib.metadata.version(distribution)
            importlib.import_module(module)
        except Exception as error:
            report["errors"].append(f"{distribution}: {type(error).__name__}")
    try:
        import torch
        from transformers import SegformerForSemanticSegmentation

        report["cuda"] = {
            "torch_cuda_build": torch.version.cuda,
            "available": torch.cuda.is_available(), "devices": [],
        }
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            report["cuda"]["devices"].append({
                "index": index, "name": properties.name,
                "memory_bytes": properties.total_memory,
                "compute_capability": [properties.major, properties.minor],
            })
        if torch.cuda.is_available():
            tensor = torch.ones((16, 16), device="cuda")
            report["cuda"]["compute_probe_sum"] = float((tensor @ tensor).sum().item())
        report["segformer_forward_signature"] = str(inspect.signature(SegformerForSemanticSegmentation.forward))
        report["segformer_implementation"] = SegformerForSemanticSegmentation.__module__
    except Exception as error:
        report["errors"].append(f"Model/CUDA probe: {type(error).__name__}")
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True,
        )
        report["nvidia_smi"] = completed.stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        report["nvidia_smi"] = {"unavailable": type(error).__name__}
    try:
        from src.config import load_config
        report["configurations"] = {
            name: load_config(ROOT / "configs" / name)["experiment_id"]
            for name in ("baseline.yaml", "augmentation.yaml", "dice_loss.yaml", "combined.yaml")
        }
    except Exception as error:
        report["errors"].append(f"Configuration: {type(error).__name__}")
    report["status"] = "passed" if not report["errors"] else "failed"
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
