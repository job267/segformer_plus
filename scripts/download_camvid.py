"""Fetch the pinned source's manifests and 11-class data, without mixing dataset versions."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import local
from time import sleep

import requests
from PIL import Image

from _bootstrap import ROOT
from src.config import load_config, project_path
from src.dataset import contained_path, read_manifest
from src.utils import file_sha256, utc_now, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--metadata-only", action="store_true", help="Lists/licenses only; not data acceptance")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be in [1, 8]")
    config = load_config(args.config)
    dataset = config["dataset"]
    root = project_path(dataset["root"])
    root.mkdir(parents=True, exist_ok=True)
    revision = dataset["source_revision"]
    base = f"https://raw.githubusercontent.com/lih627/CamVid/{revision}"
    metadata_path = root / "source_metadata.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("source_revision") != revision:
            raise ValueError("Existing dataset belongs to a different revision; use a separate directory")
    report = {
        "status": "downloading", "source": dataset["source"], "source_revision": revision,
        "timestamp_utc": utc_now(), "metadata_only": args.metadata_only,
        "downloaded_files": 0, "reused_files": 0, "failed_files": [],
        "license_note": "Source repository LICENSE is recorded; original CamVid terms also apply.",
    }
    connections = local()

    def fetch(relative: str, *, verify_image=False, destination=None):
        destination = destination or contained_path(root, relative)
        if destination.exists():
            if verify_image:
                with Image.open(destination) as image:
                    image.verify()
            return "reused"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        last_error = None
        if not hasattr(connections, "session"):
            connections.session = requests.Session()
            connections.session.headers["User-Agent"] = "segformer-course-project"
        for attempt in range(3):
            try:
                with connections.session.get(
                    f"{base}/{relative}", stream=True, timeout=(10, 20)
                ) as response, temporary.open("wb") as handle:
                    response.raise_for_status()
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                if verify_image:
                    with Image.open(temporary) as image:
                        image.verify()
                temporary.replace(destination)
                return "downloaded"
            except Exception as error:
                last_error = error
                temporary.unlink(missing_ok=True)
                connections.session.close()
                connections.session = requests.Session()
                connections.session.headers["User-Agent"] = "segformer-course-project"
                if attempt < 2:
                    sleep(1 + attempt)
        raise RuntimeError(f"{relative}: {type(last_error).__name__}: {last_error}")

    try:
        metadata_files = list(dataset["splits"].values()) + ["LICENSE", "README.md", "camvid_data.py"]
        for relative in metadata_files:
            outcome = fetch(relative)
            report[f"{outcome}_files"] = report.get(f"{outcome}_files", 0) + 1
        report["metadata_sha256"] = {
            relative: file_sha256(contained_path(root, relative)) for relative in metadata_files
        }
        file_paths = []
        report["manifest_counts"] = {}
        for split, manifest in dataset["splits"].items():
            samples = read_manifest(root, manifest, require_files=False)
            report["manifest_counts"][split] = len(samples)
            for sample in samples:
                for path in (sample.image_path, sample.mask_path):
                    file_paths.append(path.relative_to(root).as_posix())
        if not args.metadata_only:
            file_paths = sorted(set(file_paths))
            # Resolve and check all target paths before concurrent directory/file creation.
            destinations = {relative: contained_path(root, relative) for relative in file_paths}
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                pending = {
                    pool.submit(fetch, relative, verify_image=True, destination=destinations[relative]): relative
                    for relative in file_paths
                }
                for count, future in enumerate(as_completed(pending), 1):
                    try:
                        outcome = future.result()
                        report[f"{outcome}_files"] = report.get(f"{outcome}_files", 0) + 1
                    except Exception as error:
                        report["failed_files"].append(str(error))
                        if len(report["failed_files"]) >= 8:
                            for task in pending:
                                task.cancel()
                            print("Stopped after repeated network/file errors; rerun to resume.", flush=True)
                            break
                    if count % 50 == 0 or count == len(file_paths):
                        print(f"Files checked/downloaded: {count}/{len(file_paths)}", flush=True)
        report["status"] = "failed" if report["failed_files"] else (
            "metadata_only" if args.metadata_only else "downloaded_pending_audit"
        )
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
    write_json(metadata_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("Run audit_dataset.py to verify files and labels before training.")
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
