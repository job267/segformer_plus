"""Delivery checks focus on path containment and detecting damaged bundled files."""

import json

import pytest
import torch

from delivery.package import RUNS, delivery_files, safe_path, sha256, training_workspace
from delivery.verify import verify_manifest
from demo.app import Predictor


@pytest.mark.parametrize("relative", ["../x", "/absolute", "C:/escape", "a\\b", "a/../../escape", "a//b", "./file"])
def test_manifest_rejects_escaping_or_ambiguous_paths(tmp_path, relative):
    with pytest.raises(ValueError):
        safe_path(tmp_path, relative)


def test_manifest_detects_same_size_tampering(tmp_path):
    path = tmp_path / "model.txt"
    path.write_bytes(b"original")
    manifest = {"schema_version": 1, "source_git_commit": "abc", "source_dirty": False,
                "files": [{"path": "model.txt", "bytes": 8, "sha256": sha256(path)}]}
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_manifest(tmp_path)["files_verified"] == 1
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="integrity failed"):
        verify_manifest(tmp_path)


def test_new_training_workspace_has_no_run_history_or_weights(tmp_path):
    root, target = tmp_path / "source", tmp_path / "fresh"
    (root / "src").mkdir(parents=True)
    (root / "src/model.py").write_bytes(b"# preserve bytes\r\n")
    (root / "results/run/best_model").mkdir(parents=True)
    (root / "results/run/best_model/model.safetensors").write_bytes(b"weights")
    (root / "demo").mkdir()
    (root / "demo/example.png").write_bytes(b"data")
    training_workspace(root, target)
    assert (target / "src/model.py").read_bytes() == b"# preserve bytes\r\n"
    assert not (target / "results").exists()
    assert not (target / "demo/example.png").exists()
    with pytest.raises(FileExistsError):
        training_workspace(root, target)


def test_bundle_keeps_archived_configs_but_excludes_training_states(tmp_path):
    for run in RUNS:
        for suffix in ("config.yaml", "best_model/config.json", "best_model/model.safetensors", "last_checkpoint/state.pt"):
            path = tmp_path / "results" / run / suffix
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
    for relative in ("demo/example.png", "reports/final_report.md", "reports/final_report.html", "docs/reproduction.md", "THIRD_PARTY_NOTICES.md", "data/CamVid/private_image.png"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    paths = {path.relative_to(tmp_path).as_posix() for path in delivery_files(tmp_path)}
    assert "results/E0_baseline/config.yaml" in paths
    assert "results/multiseed/E5_seed3407/best_model/model.safetensors" in paths
    assert not any("last_checkpoint" in path or path.startswith("data/CamVid") for path in paths)


@pytest.mark.parametrize("payload,message", [({"model": "unknown"}, "模型"),
    ({"precision": "fp16"}, "CUDA"), ({"image": "not-base64"}, "无效")])
def test_demo_rejects_invalid_request_without_loading_models(payload, message):
    predictor = Predictor(torch.device("cpu"))
    with pytest.raises(ValueError, match=message):
        predictor.predict(payload)
    assert not predictor.cache
