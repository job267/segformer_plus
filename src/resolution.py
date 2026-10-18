"""Resolution ablation with the frozen E3 evaluation label grid."""

import csv
from copy import deepcopy

import numpy as np
import torch
from PIL import Image

from src.comparison import load_completed, read_json
from src.config import load_config, project_path
from src.dataset import CamVidDataset
from src.labels import CLASS_NAMES
from src.utils import file_sha256, write_json

RESOLUTION_CONFIG = "configs/resolution.yaml"
CONTROL_CONFIG = "configs/combined.yaml"


def validate_resolution(config, control):
    if config["experiment_id"] != "E4_resolution":
        raise ValueError("This entry point requires E4_resolution")
    if project_path(config["output_dir"]) == project_path(control["output_dir"]):
        raise ValueError("E4 must use a separate output directory")
    normalized = deepcopy(config)
    if normalized["preprocessing"]["size"] != [512, 512]:
        raise ValueError("E4 input size must be 512x512")
    if normalized["preprocessing"].pop("evaluation_size", None) != control["preprocessing"]["size"]:
        raise ValueError("E4 evaluation labels must use the E3 label grid")
    normalized["preprocessing"]["size"] = control["preprocessing"]["size"]
    normalized["experiment_id"] = control["experiment_id"]
    normalized["output_dir"] = control["output_dir"]
    if normalized != control:
        raise ValueError("An unplanned factor differs from E3")


class EvaluationGridTransform:
    def __init__(self, image_transform, evaluation_size):
        self.image_transform = image_transform
        self.evaluation_size = tuple(evaluation_size)

    def __call__(self, image, mask):
        pixels, _ = self.image_transform(image, mask)
        height, width = self.evaluation_size
        # Resize directly from the ORIGINAL mask, never via the 512x512 train grid.
        label = Image.fromarray(mask.astype(np.uint8)).resize(
            (width, height), Image.Resampling.NEAREST
        )
        return pixels, torch.from_numpy(np.array(label, dtype=np.int64))


class ResolutionDataset(CamVidDataset):
    def __init__(self, config, split="train", *, disable_augmentation=False):
        super().__init__(config, split, disable_augmentation=disable_augmentation)
        if split != "train":
            self.transform = EvaluationGridTransform(
                self.transform, config["preprocessing"]["evaluation_size"]
            )


def verify_control(config):
    control = load_config(CONTROL_CONFIG)
    validate_resolution(config, control)
    metrics, history, info = load_completed(control)
    for relative, digest in info["source_sha256"].items():
        if file_sha256(project_path(relative)) != digest:
            raise ValueError(f"Frozen E3 source changed: {relative}")
    return metrics, history, info


def validate_evaluation_grids(candidate, control):
    for split in ("val", "test"):
        if candidate[split]["class_pixels"] != control[split]["class_pixels"]:
            raise ValueError(f"{split} labels differ from the E3 evaluation grid")


def write_resolution_comparison(config, *, allow_incomplete=False):
    control_config = load_config(CONTROL_CONFIG)
    validate_resolution(config, control_config)
    control, control_history, control_info = load_completed(control_config)
    directory = project_path(config["output_dir"])
    status_path = directory / "status.json"
    status = read_json(status_path)["status"] if status_path.exists() else "pending"
    entries = [(control_config, control, control_history, "completed")]
    if status == "completed":
        metrics, history, info = load_completed(config)
        if info["data_audit_sha256"] != control_info["data_audit_sha256"]:
            raise ValueError("E4 data audit differs from E3")
        if any(info["source_sha256"].get(path) != digest for path, digest in control_info["source_sha256"].items()):
            raise ValueError("E4 did not use the frozen E3 implementation")
        grid = config["preprocessing"]["evaluation_size"]
        if metrics.get("evaluation_size") != grid or metrics.get("input_size") != [512, 512]:
            raise ValueError("E4 metric resolution metadata differs")
        if metrics.get("control_checkpoint_sha256") != control["checkpoint_sha256"]:
            raise ValueError("E4 comparison refers to a different E3 checkpoint")
        validate_evaluation_grids(
            {"val": read_json(directory / "best_validation_metrics.json"), "test": metrics},
            {"val": read_json(project_path(control_config["output_dir"]) / "best_validation_metrics.json"),
             "test": control},
        )
        entries.append((config, metrics, history, status))
    elif allow_incomplete:
        entries.append((config, None, None, status))
    else:
        raise ValueError(f"E4 has not completed: {status}")
    rows = []
    for item, metrics, history, item_status in entries:
        row = {
            "experiment_id": item["experiment_id"], "status": item_status,
            "input_size": "x".join(map(str, item["preprocessing"]["size"])),
            "evaluation_size": "384x384",
        }
        if metrics is not None:
            row.update(
                best_val_epoch=metrics["best_val_epoch"], best_val_miou=metrics["best_val_miou"],
                test_miou=metrics["test_miou"], pixel_accuracy=metrics["pixel_accuracy"],
                delta_val_pp=100 * (metrics["best_val_miou"] - control["best_val_miou"]),
                delta_test_pp=100 * (metrics["test_miou"] - control["test_miou"]),
                train_val_minutes=sum(float(record["epoch_time_sec"]) for record in history) / 60,
                peak_vram_mb=max((float(record["peak_vram_mb"]) for record in history if record["peak_vram_mb"]), default=None),
            )
        rows.append(row)
    path = project_path("results/resolution_comparison.csv")
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        fields = [*rows[0]]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    report = [
        "# E3–E4 分辨率消融", "",
        "E3 输入/训练标签为 384×384；E4 输入/训练标签为 512×512。",
        "两组验证和测试的原始标签均直接缩放到 384×384，预测 logits 在 argmax 前插值到该网格。",
        "相同 seed=42、预训练初始化来源、增强、损失、batch=4、50 轮及优化器。",
        "E4 独立从固定预训练编码器开始，不接续 E3 权重。", "",
        "| 实验 | 状态 | 最佳 Val mIoU | Test mIoU | Val 变化 | Test 变化 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = (
            f"{row['best_val_miou']:.2%}", f"{row['test_miou']:.2%}",
            f"{row['delta_val_pp']:+.2f} pp", f"{row['delta_test_pp']:+.2f} pp",
        ) if row["status"] == "completed" else ("", "", "", "")
        report.append(f"| {row['experiment_id']} | {row['status']} | {' | '.join(values)} |")
    if status == "completed":
        selected = max(rows, key=lambda row: row["best_val_miou"])
        report += [
            "", f"两组中按验证集选择 **{selected['experiment_id']}**。",
            "单 seed 结果不证明稳定性；训练分辨率改变后训练损失不可直接跨组比较。",
            "此表评价网格与原 E0–E3 一致，但当前测试集已查看过，不能作为新增未见盲测。",
        ]
        write_json(project_path("results/resolution_validation_selected_experiment.json"), {
            "selected_by": "best_val_miou", "experiment_id": selected["experiment_id"],
            "best_val_miou": selected["best_val_miou"], "evaluation_size": [384, 384],
            "checkpoint": str(project_path(next(item["output_dir"] for item, *_ in entries
                                                   if item["experiment_id"] == selected["experiment_id"])) / "best_model"),
        })
        with project_path("results/resolution_per_class.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["class_id", "class_name", "E3_iou", "E4_iou", "delta_pp"])
            for index, name in enumerate(CLASS_NAMES):
                before, after = control["per_class_iou"][index], metrics["per_class_iou"][index]
                writer.writerow([index, name, before, after,
                                 None if before is None or after is None else 100 * (after - before)])
    else:
        report += ["", "E4 尚未完成，空白不表示零分；完成后自动核验并汇总，不预设分辨率提高必然有收益。"]
    project_path("reports/resolution_ablation.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return rows
