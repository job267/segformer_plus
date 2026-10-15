"""Sequential E1/E2/E3 queue, independent initialization, stop on failure."""

import argparse
import json
import os
import subprocess
import sys
import traceback

from _bootstrap import ROOT
from src.comparison import CONFIG_PATHS, load_completed, verify_baseline, write_comparison
from src.config import load_config, project_path
from src.experiment import experiment_lock
from src.suite import run_sequence
from src.utils import file_sha256, utc_now, write_json


def source_hashes():
    sources = {
        str(path.relative_to(ROOT)): file_sha256(path)
        for folder, pattern in [("src", "*.py"), ("scripts", "*.py"), ("configs", "*.yaml")]
        for path in (ROOT / folder).glob(pattern)
    }
    sources["requirements.txt"] = file_sha256(ROOT / "requirements.txt")
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="Resume unfinished runs; never changes configuration")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    queue = project_path("results/ablation_queue")
    queue.mkdir(parents=True, exist_ok=True)
    with experiment_lock(queue):
        started = utc_now()
        try:
            configs = [load_config(path) for path in CONFIG_PATHS]
            verify_baseline(configs)
            if not project_path("reports/baseline_analysis.md").is_file():
                raise ValueError("Baseline analysis must be completed before E1-E3")
            by_id = {config["experiment_id"]: (path, config) for path, config in zip(CONFIG_PATHS[1:], configs[1:])}
            plan = list(by_id)
            snapshot = {
                "pid": os.getpid(), "started_utc": started, "order": plan,
                "initialization": "Each run starts independently from pinned MiT-B0, seed=42",
                "baseline": configs[0]["experiment_id"],
            }
            write_json(queue / "plan.json", snapshot)
            write_comparison(configs, allow_incomplete=True)
            # Freeze the entire implementation before the first child process.
            sources = source_hashes()

            def completed(experiment):
                _, config = by_id[experiment]
                status = project_path(config["output_dir"]) / "status.json"
                if not status.exists():
                    return False
                if json.loads(status.read_text(encoding="utf-8"))["status"] != "completed":
                    return False
                load_completed(config)
                return True

            def launch(experiment):
                if source_hashes() != sources:
                    raise ValueError("Source/config/requirements changed while queue was running")
                path, config = by_id[experiment]
                directory = project_path(config["output_dir"])
                directory.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, "-u", str(ROOT / "scripts/train.py"), "--config", path]
                if args.offline:
                    command.append("--offline")
                if (directory / "config.yaml").exists():
                    if not args.resume:
                        raise FileExistsError(f"{experiment} already has a run; use --resume")
                    command.append("--resume")
                print(json.dumps({"starting": experiment, "argv": command}), flush=True)
                with (directory / "stdout.log").open("ab") as stdout, (directory / "stderr.log").open("ab") as stderr:
                    return subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr).returncode

            def state(phase, current, completed_ids):
                write_json(queue / "status.json", {
                    **snapshot, "status": "running", "phase": phase, "updated_utc": utc_now(),
                    "current_experiment": current, "completed": completed_ids,
                })
                if phase != "running":
                    write_comparison(configs, allow_incomplete=True)

            finished = run_sequence(plan, launch, completed, state)
            rows = write_comparison(configs)
            write_json(queue / "status.json", {
                **snapshot, "status": "completed", "updated_utc": utc_now(),
                "current_experiment": None, "completed": finished, "comparison": rows,
            })
        except BaseException as error:
            path = queue / "status.json"
            status = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            status.update(
                status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                updated_utc=utc_now(), error=str(error), traceback=traceback.format_exc(),
            )
            write_json(path, status)
            traceback.print_exc()
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
