"""Create a local, hash-manifested course bundle or an empty-results training workspace."""

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
RUNS = ["E0_baseline", "E1_augmentation", "E2_dice", "E3_combined", "E4_resolution", "E5_weighted",
        "multiseed/E4_seed2024", "multiseed/E5_seed2024", "multiseed/E4_seed3407", "multiseed/E5_seed3407"]
CASES = ("0001TP_008550", "Seq05VD_f02310")
TEXT_SUFFIXES = {".py", ".yaml", ".md", ".txt", ".json", ".toml", ".html"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(root, relative):
    """Reject platform-independent traversal before resolving within the bundle."""
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError(f"Invalid manifest path: {relative!r}")
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in ("", ".", "..") for part in relative.split("/")):
        raise ValueError(f"Invalid manifest path: {relative!r}")
    resolved = (Path(root) / Path(*path.parts)).resolve()
    if not resolved.is_relative_to(Path(root).resolve()):
        raise ValueError("Manifest path escapes bundle")
    return resolved


def source_files(root):
    root = Path(root)
    files = set()
    for path in root.iterdir():
        if path.is_file() and (path.suffix in TEXT_SUFFIXES or path.name in {".gitignore", ".gitattributes"}):
            files.add(path)
    for directory in ("src", "scripts", "configs", "tests", "docs", "demo", "delivery"):
        for path in (root / directory).rglob("*"):
            relative = path.relative_to(root)
            if not path.is_file() or any(part in {"__pycache__", "artifacts"} for part in relative.parts):
                continue
            if relative.as_posix() in {"delivery/release.json", "delivery/verification.json"}:
                continue
            if path.suffix in TEXT_SUFFIXES:
                files.add(path)
    if (root / "data/README.md").is_file():
        files.add(root / "data/README.md")
    return sorted(files)


def delivery_files(root):
    root = Path(root)
    files = set(source_files(root))
    for path in (root / "reports").rglob("*"):
        if path.is_file() and path.suffix in {".md", ".html", ".json", ".png"}:
            files.add(path)
    for path in (root / "results").rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if any(part in {"last_checkpoint", "checkpoints", "previews", "best_model"} for part in path.relative_to(root).parts):
            continue
        if "/predictions/" in relative:
            selected = any(relative == f"results/{run}/predictions/{case}_{suffix}.png"
                           for case in CASES for run, suffix in (("E0_baseline", "input"), ("E0_baseline", "gt"), ("E0_baseline", "pred"), ("E5_weighted", "pred")))
            selected |= relative == "results/inference_demo/predictions/000_0016E5_07959/input.png"
            if selected:
                files.add(path)
            continue
        if relative.startswith("results/inference_demo/"):
            continue
        if path.suffix in {".json", ".yaml", ".csv", ".png"}:
            files.add(path)
    for run in RUNS:
        for filename in ("config.json", "model.safetensors"):
            path = root / "results" / run / "best_model" / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing completed best model: {path}")
            files.add(path)
    for name in ("demo/example.png", "reports/final_report.md", "reports/final_report.html", "docs/reproduction.md", "THIRD_PARTY_NOTICES.md"):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing delivery requirement: {path}")
        files.add(path)
    return sorted(files)


def training_workspace(root, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Training workspace must not already exist; choose a new directory")
    paths = source_files(root)
    output.mkdir(parents=True)
    for path in paths:
        target = output / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    return {"workspace": str(output), "source_files": len(paths), "copied_experiment_results": False}


def build_archive(root, output, *, allow_dirty=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Archive exists; choose a new filename")
    git_status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, capture_output=True, text=True, check=True).stdout
    if git_status.strip() and not allow_dirty:
        raise ValueError("Commit delivery source first, or use --allow-dirty for a clearly marked validation candidate")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    files = delivery_files(root)
    entries = [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files]
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_git_commit": revision, "source_dirty": bool(git_status.strip()),
                "purpose": "Course research/evaluation; see THIRD_PARTY_NOTICES.md",
                "best_model_runs": RUNS, "full_dataset_included": False,
                "last_training_states_included": False, "files": entries}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
        for path, entry in zip(files, entries):
            archive.write(path, entry["path"])
        archive.writestr("MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return {"archive": str(output), "archive_bytes": output.stat().st_size,
            "archive_sha256": sha256(output), "file_count": len(entries),
            "source_git_commit": revision, "source_dirty": manifest["source_dirty"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New ZIP filename")
    parser.add_argument("--training-workspace", type=Path, help="New source-only directory, without any experiment archives")
    parser.add_argument("--allow-dirty", action="store_true", help="Validation candidates only; dirty state is recorded")
    args = parser.parse_args()
    if bool(args.output) == bool(args.training_workspace):
        parser.error("Choose exactly one of --output / --training-workspace")
    result = training_workspace(ROOT, args.training_workspace) if args.training_workspace else build_archive(ROOT, args.output, allow_dirty=args.allow_dirty)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
