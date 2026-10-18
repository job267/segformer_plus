"""Sequential paired replication queue, stopping on failure and freezing all sources."""

import argparse
import json
import os
import subprocess
import sys
import traceback

from _bootstrap import ROOT
from train import source_manifest
from src.comparison import read_json
from src.config import project_path
from src.experiment import experiment_lock
from src.multiseed import (
    PREFLIGHT_PATH, QUEUE_ORDER, load_replicate, replicate_config,
    verify_references, write_multiseed_comparison,
)
from src.suite import run_sequence
from src.utils import file_sha256, utc_now, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    queue = project_path("results/multiseed/queue")
    queue.mkdir(parents=True, exist_ok=True)
    with experiment_lock(queue):
        snapshot = {"pid": os.getpid(), "started_utc": utc_now(),
                    "order": [replicate_config(*pair)["experiment_id"] for pair in QUEUE_ORDER]}
        try:
            references = verify_references()
            sources = source_manifest()
            preflight = read_json(project_path(PREFLIGHT_PATH))
            if preflight["status"] != "passed" or preflight["source_sha256"] != sources:
                raise ValueError("Passed current-source preflight required before queue launch")
            plan = {"runs": [replicate_config(*pair) for pair in QUEUE_ORDER], "source_sha256": sources,
                    "preflight_sha256": file_sha256(project_path(PREFLIGHT_PATH))}
            plan_path = queue / "plan.json"
            if plan_path.exists() and read_json(plan_path) != plan:
                raise ValueError("Existing queue plan differs; refusing to change the fixed experiment")
            write_json(plan_path, plan)
            write_multiseed_comparison(allow_incomplete=True)
            by_id = {replicate_config(*pair)["experiment_id"]: pair for pair in QUEUE_ORDER}

            def completed(experiment):
                family, seed = by_id[experiment]
                status = project_path(replicate_config(family, seed)["output_dir"]) / "status.json"
                if not status.exists() or read_json(status)["status"] != "completed":
                    return False
                load_replicate(family, seed, references)
                return True

            def launch(experiment):
                if source_manifest() != sources or file_sha256(project_path(PREFLIGHT_PATH)) != plan["preflight_sha256"]:
                    raise ValueError("Code/config/preflight changed during the replication queue")
                family, seed = by_id[experiment]
                directory = project_path(replicate_config(family, seed)["output_dir"])
                directory.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, "-u", str(ROOT / "scripts/train_replicate.py"),
                           "--family", family, "--seed", str(seed)]
                if args.offline:
                    command.append("--offline")
                if (directory / "config.yaml").exists():
                    if not args.resume:
                        raise FileExistsError(f"{experiment} exists; use --resume")
                    command.append("--resume")
                print(json.dumps({"starting": experiment, "argv": command}), flush=True)
                with (directory / "stdout.log").open("ab") as stdout, (directory / "stderr.log").open("ab") as stderr:
                    return subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr).returncode

            def state(phase, current, finished):
                write_json(queue / "status.json", {
                    **snapshot, "status": "running", "updated_utc": utc_now(),
                    "phase": phase, "current_experiment": current, "completed": finished,
                })
                if phase != "running":
                    write_multiseed_comparison(allow_incomplete=True)

            finished = run_sequence(snapshot["order"], launch, completed, state)
            comparison = write_multiseed_comparison()
            write_json(queue / "status.json", {
                **snapshot, "status": "completed", "updated_utc": utc_now(),
                "current_experiment": None, "completed": finished, "comparison": comparison,
            })
        except BaseException as error:
            path = queue / "status.json"
            status = read_json(path) if path.exists() else snapshot
            status.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                          updated_utc=utc_now(), error=str(error), traceback=traceback.format_exc())
            write_json(path, status)
            traceback.print_exc()
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
