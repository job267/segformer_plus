"""Audit split manifests, pairing, label values, dimensions, and leakage."""

import argparse
import json

from _bootstrap import ROOT
from src.audit import audit_dataset
from src.config import load_config, project_path
from src.utils import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--output", default="results/stage1/data_audit.json")
    parser.add_argument("--previews", default="results/stage1/previews")
    args = parser.parse_args()
    report = audit_dataset(load_config(args.config), preview_dir=project_path(args.previews))
    write_json(project_path(args.output), report)
    summary = {key: value for key, value in report.items() if key != "files"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Audit saved: {project_path(args.output)}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
