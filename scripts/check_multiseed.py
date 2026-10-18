"""Verify references, data and paired pretrained initializations before replication."""

import argparse
import json
import traceback

from _bootstrap import ROOT
from train import source_manifest, verify_prerequisites

from src.comparison import read_json
from src.config import load_config, project_path
from src.model import build_model
from src.multiseed import (
    FAMILIES, PREFLIGHT_PATH, QUEUE_ORDER, SEEDS, model_state_sha256,
    replicate_config, verify_references,
)
from src.utils import file_sha256, seed_everything, utc_now, write_json
from src.weighted import WEIGHTED_CONFIG, WEIGHTS_PATH, prepare_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    path = project_path(PREFLIGHT_PATH)
    report = {"status": "running", "timestamp_utc": utc_now()}
    try:
        references = verify_references()
        config = load_config(WEIGHTED_CONFIG)
        audit_hash = verify_prerequisites(config)
        if audit_hash != references["E5"]["info"]["data_audit_sha256"]:
            raise ValueError("Current data differs from E4/E5 reference audits")
        prepare_weights(config)
        hashes = {}
        for seed in SEEDS[1:]:
            hashes[str(seed)] = {}
            for family in FAMILIES:
                seed_everything(seed)
                model, _ = build_model(replicate_config(family, seed),
                                       local_files_only=True if args.offline else None)
                hashes[str(seed)][family] = model_state_sha256(model)
                del model
            if len(set(hashes[str(seed)].values())) != 1:
                raise ValueError("E4/E5 paired initial model weights differ")
        if len({value["E4"] for value in hashes.values()}) != len(SEEDS) - 1:
            raise ValueError("Different new seeds unexpectedly produced identical complete initial weights")
        report.update(
            status="passed", seeds=list(SEEDS), order=[replicate_config(*item)["experiment_id"] for item in QUEUE_ORDER],
            initial_hashes=hashes, source_sha256=source_manifest(), data_audit_sha256=audit_hash,
            class_weights_sha256=file_sha256(project_path(WEIGHTS_PATH)),
            reference_checkpoints={family: references[family]["metrics"]["checkpoint_sha256"] for family in FAMILIES},
            test_inference=False, scope="reference archive verification and CPU pretrained initialization only",
        )
        if path.exists() and read_json(path).get("status") == "passed":
            previous = read_json(path)
            if {k: v for k, v in previous.items() if k != "timestamp_utc"} != {k: v for k, v in report.items() if k != "timestamp_utc"}:
                raise ValueError("Existing passed preflight differs; refusing to overwrite frozen replication evidence")
            report = previous
        else:
            write_json(path, report)
    except Exception as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
        write_json(path.with_name("preflight_error.json"), report)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
