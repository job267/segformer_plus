"""Verify the frozen factorial experiment and summarize only measured results."""

import csv
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from src.config import load_config, project_path
from src.labels import CLASS_NAMES
from src.utils import file_sha256, write_json

CONFIG_PATHS = [
    "configs/baseline.yaml", "configs/augmentation.yaml",
    "configs/dice_loss.yaml", "configs/combined.yaml",
]
FACTORS = [(False, "ce"), (True, "ce"), (False, "ce_dice"), (True, "ce_dice")]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_matrix(configurations):
    if len(configurations) != 4:
        raise ValueError("Exactly E0/E1/E2/E3 are required")
    canonical = []
    directories = set()
    for config, expected in zip(configurations, FACTORS):
        if (config["augmentation"]["enabled"], config["training"]["loss"]) != expected:
            raise ValueError("Experiment factors differ from the frozen E0-E3 matrix")
        directory = project_path(config["output_dir"])
        if directory in directories:
            raise ValueError("Experiments must have separate output directories")
        directories.add(directory)
        normalized = deepcopy(config)
        normalized.pop("experiment_id")
        normalized.pop("output_dir")
        normalized["augmentation"]["enabled"] = False
        normalized["training"]["loss"] = "ce"
        canonical.append(normalized)
    if any(config != canonical[0] for config in canonical[1:]):
        raise ValueError("An unplanned factor differs: seed/data/model/budget/optimizer/preprocessing")


def verify_metric_record(metrics, history, config):
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    if matrix.shape != (11, 11) or (matrix < 0).any() or matrix.sum() == 0:
        raise ValueError("Invalid confusion matrix")
    diagonal = np.diag(matrix)
    support = matrix.sum(axis=1)
    union = support + matrix.sum(axis=0) - diagonal
    valid = union > 0
    iou = np.divide(diagonal, union, out=np.zeros(11, dtype=float), where=valid)
    if abs(float(iou[valid].mean()) - metrics["test_miou"]) > 1e-12:
        raise ValueError("Reported test mIoU does not match the confusion matrix")
    if abs(float(diagonal.sum() / matrix.sum()) - metrics["pixel_accuracy"]) > 1e-12:
        raise ValueError("Reported pixel accuracy does not match the confusion matrix")
    if list(support) != metrics["class_pixels"]:
        raise ValueError("Class pixel counts do not match the confusion matrix")
    for index, expected in enumerate(metrics["per_class_iou"]):
        if (not valid[index] and expected is not None) or (
            valid[index] and (expected is None or abs(expected - iou[index]) > 1e-12)
        ):
            raise ValueError("Per-class IoU does not match the confusion matrix")
    expected_epochs = config["training"]["epochs"]
    if [int(row["epoch"]) for row in history] != list(range(1, expected_epochs + 1)):
        raise ValueError("Training history is incomplete")
    best = max(history, key=lambda row: float(row["val_miou"]))
    if int(best["epoch"]) != metrics["best_val_epoch"] or abs(
        float(best["val_miou"]) - metrics["best_val_miou"]
    ) > 1e-12:
        raise ValueError("The checkpoint was not selected by highest validation mIoU")
    if metrics["seed"] != config["seed"] or metrics["ignore_index"] != 255:
        raise ValueError("Metric protocol differs from the configuration")
    if metrics["best_checkpoint_selected_by"] != "val_miou" or metrics["split"] != "test":
        raise ValueError("Final test/validation selection policy differs")
    if metrics["class_names"] != list(CLASS_NAMES):
        raise ValueError("Class order differs")


def load_completed(config):
    directory = project_path(config["output_dir"])
    status = read_json(directory / "status.json")
    if status["status"] != "completed":
        raise ValueError(f"{config['experiment_id']} has not completed")
    if load_config(directory / "config.yaml") != config:
        raise ValueError(f"{config['experiment_id']} configuration changed")
    metrics = read_json(directory / "metrics.json")
    with (directory / "train_log.csv").open(encoding="utf-8", newline="") as handle:
        history = list(csv.DictReader(handle))
    verify_metric_record(metrics, history, config)
    if file_sha256(directory / "best_model/model.safetensors") != metrics["checkpoint_sha256"]:
        raise ValueError("The tested checkpoint file changed")
    return metrics, history, read_json(directory / "run_info.json")


def verify_baseline(configurations):
    validate_matrix(configurations)
    baseline = configurations[0]
    metrics, history, info = load_completed(baseline)
    # Every pre-existing source file is held fixed; newly added queue/report code is allowed.
    for relative, expected in info["source_sha256"].items():
        if file_sha256(project_path(relative)) != expected:
            raise ValueError(f"Baseline implementation changed: {relative}")
    return metrics, history, info


def summarize(configurations, *, allow_incomplete=False):
    validate_matrix(configurations)
    baseline_metrics, _, baseline_info = load_completed(configurations[0])
    rows, records = [], {}
    for config in configurations:
        directory = project_path(config["output_dir"])
        status_path = directory / "status.json"
        status = read_json(status_path)["status"] if status_path.exists() else "pending"
        row = {"experiment_id": config["experiment_id"], "status": status}
        if status != "completed":
            if not allow_incomplete:
                raise ValueError(f"Cannot make a final comparison: {config['experiment_id']} is {status}")
        else:
            metrics, history, info = load_completed(config)
            if info["data_audit_sha256"] != baseline_info["data_audit_sha256"]:
                raise ValueError("Experiments used different dataset audits")
            if any(info["source_sha256"].get(path) != digest for path, digest in baseline_info["source_sha256"].items()):
                raise ValueError("Experiments did not use the same baseline implementation")
            if metrics["class_pixels"] != baseline_metrics["class_pixels"]:
                raise ValueError("Experiments evaluated different test pixels")
            row.update(
                best_val_epoch=metrics["best_val_epoch"],
                best_val_miou=metrics["best_val_miou"], test_miou=metrics["test_miou"],
                pixel_accuracy=metrics["pixel_accuracy"],
                delta_test_miou_pp=100 * (metrics["test_miou"] - baseline_metrics["test_miou"]),
                train_val_minutes=sum(float(item["epoch_time_sec"]) for item in history) / 60,
                peak_vram_mb=max(
                    (float(item["peak_vram_mb"]) for item in history if item["peak_vram_mb"]),
                    default=None,
                ),
            )
            records[config["experiment_id"]] = metrics
        rows.append(row)
    return rows, records


def write_comparison(configurations, *, allow_incomplete=False):
    rows, records = summarize(configurations, allow_incomplete=allow_incomplete)
    results = project_path("results")
    fields = [
        "experiment_id", "status", "best_val_epoch", "best_val_miou", "test_miou",
        "pixel_accuracy", "delta_test_miou_pp", "train_val_minutes", "peak_vram_mb",
    ]
    path = results / "comparison.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    with (results / "comparison_per_class.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        ids = [config["experiment_id"] for config in configurations]
        writer.writerow(["class_id", "class_name", *ids])
        for index, name in enumerate(CLASS_NAMES):
            writer.writerow([index, name, *[
                records[experiment]["per_class_iou"][index] if experiment in records else None
                for experiment in ids
            ]])
    complete = len(records) == 4
    report = [
        "# E0–E3 消融实验", "",
        f"已完成 {len(records)}/4 组。所有数字来自各组原始 metrics.json 和逐轮日志。",
        "空白表示尚未完成，不是零分；测试集不参与选模。", "",
        "| 实验 | 状态 | 最佳 Val mIoU | Test mIoU | 相对 E0 的 Test 变化 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        values = (
            f"{row['best_val_miou']:.2%}", f"{row['test_miou']:.2%}",
            f"{row['delta_test_miou_pp']:+.2f} 个百分点",
        ) if row["status"] == "completed" else ("", "", "")
        report.append(f"| {row['experiment_id']} | {row['status']} | {' | '.join(values)} |")
    if complete:
        selected = max(rows, key=lambda row: row["best_val_miou"])
        baseline = rows[0]
        report += [
            "", f"按验证集选择的方案为 **{selected['experiment_id']}**，"
            f"最佳 Val mIoU={selected['best_val_miou']:.2%}；该方案的最终 Test mIoU="
            f"{selected['test_miou']:.2%}。方案选择依据为验证集，不按 Test 排名改选。", "",
        ]
        for row in rows[1:]:
            direction = "高于" if row["delta_test_miou_pp"] > 0 else (
                "低于" if row["delta_test_miou_pp"] < 0 else "等于"
            )
            report.append(
                f"- {row['experiment_id']}：Test mIoU {direction} E0 "
                f"{abs(row['delta_test_miou_pp']):.2f} 个百分点；"
                f"最佳 Val 变化 {100*(row['best_val_miou']-baseline['best_val_miou']):+.2f} 个百分点。"
            )
        report += [
            "", "单 seed 对照不证明统计显著性或因果机制。训练损失的 CE 与 CE+Dice "
            "数值不能直接作跨组优劣比较。输入、预算、优化器、种子、数据及原有训练代码一致，"
            "唯一计划差异是增强开关与 Dice 项。每类指标见 results/comparison_per_class.csv。",
        ]
        write_json(results / "validation_selected_experiment.json", {
            "selected_by": "best_val_miou", "experiment_id": selected["experiment_id"],
            "best_val_miou": selected["best_val_miou"],
            "checkpoint": str(project_path(next(
                config["output_dir"] for config in configurations
                if config["experiment_id"] == selected["experiment_id"]
            )) / "best_model"),
        })
    else:
        report += ["", "其余实验完成前不判断增强、Dice 或组合是否有效。"]
    project_path("reports/ablation.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return rows
