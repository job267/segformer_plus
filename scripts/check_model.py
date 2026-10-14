"""Download/load only the pinned MiT-B0 encoder and inspect loading keys."""

import argparse
import json

from _bootstrap import ROOT
from src.config import load_config, project_path
from src.model import build_model
from src.utils import seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", default="results/stage1/model_loading.json")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config["seed"])
    try:
        _, report = build_model(
            config, local_files_only=True if args.offline else None,
            report_path=project_path(args.output),
        )
    except Exception as error:
        print(f"Model loading failed: {type(error).__name__}: {error}")
        print(f"Diagnostic saved: {project_path(args.output)}")
        return 1
    print(json.dumps({k: v for k, v in report.items() if k != "fresh_head_parameter_names"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
