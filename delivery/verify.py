"""Verify a relocated bundle and one CPU image prediction, with all HF networking disabled."""

import argparse
import base64
import io
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from delivery.package import safe_path, sha256


def verify_manifest(root):
    root = Path(root)
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1 or not manifest["files"]:
        raise ValueError("Unsupported or empty manifest")
    seen = set()
    for entry in manifest["files"]:
        path = safe_path(root, entry["path"])
        if entry["path"] in seen or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ValueError("Duplicate path or invalid SHA256 in manifest")
        seen.add(entry["path"])
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Manifest integrity failed: {entry['path']}")
    return {"files_verified": len(seen), "source_git_commit": manifest["source_git_commit"],
            "source_dirty": manifest["source_dirty"]}


def verify_bundle(*, manifest=True, prediction=True):
    result = verify_manifest(ROOT) if manifest else {}
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(ROOT / ".cache/offline_verification")
    os.environ["HF_HUB_CACHE"] = str(ROOT / ".cache/offline_verification/hub")
    cache_files = list((ROOT / ".cache/offline_verification").rglob("*.safetensors")) + list((ROOT / ".cache/offline_verification").rglob("*.bin"))
    # Imports happen only after the manifest check and explicit offline configuration.
    import numpy as np
    import torch
    from PIL import Image
    from delivery.build_report import verified_results
    from demo.app import Predictor
    from src.utils import seed_everything

    singles, records, families, pairs, delta, selection = verified_results()
    result.update(status="passed", verified_runs=len(singles)+4,
                  selected_family=selection["family"], representative_seed=selection["representative_seed"],
                  family_statistics=families, paired_statistics_pp=delta,
                  full_dataset_present=(ROOT / "data/CamVid").exists(),
                  hf_cached_weight_files_before_prediction=len(cache_files),
                  hf_hub_offline=True, transformers_offline=True,
                  test_inference_executed=False)
    if prediction:
        seed_everything(42)
        torch.set_num_threads(min(4, torch.get_num_threads()))
        example = ROOT / "demo/example.png"
        predictor = Predictor(torch.device("cpu"))
        payload = {"model": selection["family"], "precision": "fp32", "image": base64.b64encode(example.read_bytes()).decode("ascii")}
        response = predictor.predict(payload)
        raw = base64.b64decode(response["index_mask"].split(",", 1)[1])
        with Image.open(io.BytesIO(raw)) as mask, Image.open(example) as original:
            array = np.array(mask)
            if mask.size != original.size or array.min() < 0 or array.max() > 10:
                raise ValueError("Offline prediction dimensions or label range differ")
        result["offline_prediction"] = {"device": "cpu", "precision": "fp32", "model": response["model"],
                                        "width": response["width"], "height": response["height"],
                                        "label_ids": np.unique(array).tolist(),
                                        "index_png_sha256": __import__("hashlib").sha256(raw).hexdigest(),
                                        "pipeline_ms": response["pipeline_ms"]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--working-tree", action="store_true", help="Verify before packaging, without a MANIFEST")
    parser.add_argument("--skip-prediction", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional JSON record; use a path outside the extracted bundle")
    args = parser.parse_args()
    result = verify_bundle(manifest=not args.working_tree, prediction=not args.skip_prediction)
    result["verified_root"] = str(ROOT)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
