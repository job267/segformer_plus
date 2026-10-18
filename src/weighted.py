"""Train-only frequency weights and a separate E4/E5 comparison."""

import csv
from copy import deepcopy

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from src.comparison import load_completed, read_json
from src.config import load_config, project_path
from src.dataset import read_manifest
from src.labels import CLASS_NAMES, decode_mask
from src.losses import soft_dice_loss
from src.metrics import resize_logits
from src.resolution import validate_evaluation_grids
from src.utils import file_sha256, write_json

WEIGHTED_CONFIG = "configs/weighted.yaml"
CONTROL_CONFIG = "configs/resolution.yaml"
WEIGHTS_PATH = "results/stage6/class_weights.json"
WEIGHT_RULE = {"method": "inverse_sqrt_mean_clipped", "minimum": 0.5, "maximum": 3.0}


def validate_weighted(config, control):
    candidate = deepcopy(config)
    if candidate["experiment_id"] != "E5_weighted":
        raise ValueError("Expected E5_weighted")
    if project_path(candidate["output_dir"]) == project_path(control["output_dir"]):
        raise ValueError("E5 must have its own output directory")
    if candidate["training"].pop("class_weighting", None) != WEIGHT_RULE:
        raise ValueError("E5 class-weight rule differs from the frozen protocol")
    candidate["experiment_id"] = control["experiment_id"]
    candidate["output_dir"] = control["output_dir"]
    if candidate != control:
        raise ValueError("An unplanned factor differs from E4")


def verify_e4(config):
    control = load_config(CONTROL_CONFIG)
    validate_weighted(config, control)
    metrics, history, info = load_completed(control)
    for relative, digest in info["source_sha256"].items():
        if file_sha256(project_path(relative)) != digest:
            raise ValueError(f"Frozen E4 source changed: {relative}")
    return metrics, history, info


def frequency_weights(counts, minimum=0.5, maximum=3.0):
    counts = np.asarray(counts, dtype=np.float64)
    if counts.shape != (11,) or not np.isfinite(counts).all() or (counts <= 0).any():
        raise ValueError("Every training class must have positive finite pixel counts")
    raw = (counts / counts.sum()) ** -0.5
    return np.clip(raw / raw.mean(), minimum, maximum).tolist()


def compute_weight_record(config):
    validate_weighted(config, load_config(CONTROL_CONFIG))
    root = project_path(config["dataset"]["root"])
    manifest = config["dataset"]["splits"]["train"]
    samples = read_manifest(root, manifest)
    counts = np.zeros(11, dtype=np.int64)
    height, width = config["preprocessing"]["size"]
    for sample in samples:
        with Image.open(sample.mask_path) as opened:
            mask = decode_mask(opened, config["dataset"]["label_encoding"])
        labels = np.array(Image.fromarray(mask.astype(np.uint8)).resize(
            (width, height), Image.Resampling.NEAREST
        ))
        counts += np.bincount(labels[labels != 255], minlength=11)
    if len(samples) != config["dataset"]["expected_counts"]["train"]:
        raise ValueError("Training sample count differs from the frozen protocol")
    return {
        "source_split": "train", "sample_ids": [sample.sample_id for sample in samples],
        "samples": len(samples), "label_size": [height, width], "ignore_index": 255,
        "class_names": list(CLASS_NAMES), "class_pixels": counts.tolist(),
        "weights": frequency_weights(counts), "rule": WEIGHT_RULE,
        "data_audit_sha256": file_sha256(project_path("results/stage1/data_audit.json")),
        "train_manifest_sha256": file_sha256(root / manifest),
    }


def prepare_weights(config):
    record = compute_weight_record(config)
    path = project_path(WEIGHTS_PATH)
    if path.exists():
        if read_json(path) != record:
            raise ValueError("Existing class-weight archive differs; refusing to overwrite")
    else:
        write_json(path, record)
    return record


class WeightedSegmentationLoss(nn.Module):
    def __init__(self, weights, dice_weight=0.5, ignore_index=255):
        super().__init__()
        weights = torch.as_tensor(weights, dtype=torch.float32)
        if weights.shape != (11,) or not torch.isfinite(weights).all() or not (weights > 0).all():
            raise ValueError("Expected 11 finite positive class weights")
        self.register_buffer("class_weights", weights.clone())
        self.dice_weight, self.ignore_index = dice_weight, ignore_index

    def forward(self, logits, labels):
        logits = resize_logits(logits, labels).float()
        valid = labels != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0
        weights = self.class_weights.to(logits.device)
        per_pixel = F.cross_entropy(logits, labels, weight=weights,
                                    ignore_index=self.ignore_index, reduction="none")
        denominator = weights[labels[valid]].sum()
        ce = per_pixel.sum() / denominator
        return ce + self.dice_weight * soft_dice_loss(logits, labels, self.ignore_index)


def write_weighted_comparison(config, *, allow_incomplete=False):
    control_config = load_config(CONTROL_CONFIG)
    validate_weighted(config, control_config)
    control, history, control_info = load_completed(control_config)
    directory = project_path(config["output_dir"])
    status = read_json(directory / "status.json")["status"] if (directory / "status.json").exists() else "pending"
    entries = [(control_config, control, history, "completed")]
    candidate = None
    if status == "completed":
        candidate, candidate_history, info = load_completed(config)
        if info["data_audit_sha256"] != control_info["data_audit_sha256"] or any(
            info["source_sha256"].get(path) != digest for path, digest in control_info["source_sha256"].items()
        ):
            raise ValueError("E5 differs from the E4 data or frozen implementation")
        if candidate.get("control_checkpoint_sha256") != control["checkpoint_sha256"]:
            raise ValueError("E5 control checkpoint changed")
        if candidate.get("class_weights_sha256") != file_sha256(project_path(WEIGHTS_PATH)):
            raise ValueError("E5 class weights changed")
        if candidate.get("input_size") != [512, 512] or candidate.get("evaluation_size") != [384, 384]:
            raise ValueError("E5 evaluation grid changed")
        candidate_val = read_json(directory / "best_validation_metrics.json")
        control_val = read_json(project_path(control_config["output_dir"]) / "best_validation_metrics.json")
        validate_evaluation_grids({"val": candidate_val, "test": candidate},
                                  {"val": control_val, "test": control})
        entries.append((config, candidate, candidate_history, status))
    elif allow_incomplete:
        entries.append((config, None, None, status))
    else:
        raise ValueError(f"E5 has not completed: {status}")
    rows = []
    for item, metrics, records, item_status in entries:
        row = {"experiment_id": item["experiment_id"], "status": item_status}
        if metrics is not None:
            row.update(
                best_val_epoch=metrics["best_val_epoch"], best_val_miou=metrics["best_val_miou"],
                test_miou=metrics["test_miou"], pixel_accuracy=metrics["pixel_accuracy"],
                delta_val_pp=100 * (metrics["best_val_miou"] - control["best_val_miou"]),
                delta_test_pp=100 * (metrics["test_miou"] - control["test_miou"]),
                train_val_minutes=sum(float(record["epoch_time_sec"]) for record in records) / 60,
                peak_vram_mb=max((float(record["peak_vram_mb"]) for record in records if record["peak_vram_mb"]), default=None),
            )
        rows.append(row)
    path = project_path("results/weighted_comparison.csv")
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    report = [
        "# E4–E5 类别加权实验", "",
        "输入/训练标签 512×512，验证/测试标签 384×384，batch=4、50 轮、seed=42。",
        "E5 仅将 CE 改为训练频率生成的类别加权 CE，Dice 权重仍为 0.5。",
        "频率来自未增强的 367 张训练 mask；逆平方根权重按均值归一化后截断到 [0.5,3.0]。",
        "CE 以有效像素权重之和为分母；不使用验证或测试频率生成权重。", "",
        "| 实验 | 状态 | 最佳 Val mIoU | Test mIoU | Val 变化 | Test 变化 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = (f"{row['best_val_miou']:.2%}", f"{row['test_miou']:.2%}",
                  f"{row['delta_val_pp']:+.2f} pp", f"{row['delta_test_pp']:+.2f} pp") if row["status"] == "completed" else ("", "", "", "")
        report.append(f"| {row['experiment_id']} | {row['status']} | {' | '.join(values)} |")
    if candidate is not None:
        selected = max(rows, key=lambda row: row["best_val_miou"])
        report += ["", f"按验证集选择 **{selected['experiment_id']}**，同分保留 E4。", "",
                   "预先关注的验证集类别：", "",
                   "| 类别 | E4 Val IoU | E5 Val IoU | 变化 |", "|---|---:|---:|---:|"]
        for index in (2, 6, 9):
            before, after = control_val["per_class_iou"][index], candidate_val["per_class_iou"][index]
            report.append(f"| {CLASS_NAMES[index]} | {before:.2%} | {after:.2%} | {100*(after-before):+.2f} pp |")
        report += ["", "单 seed，不证明稳定性；测试集已查看，不作为未见盲测；两组 loss 不直接比较。"]
        write_json(project_path("results/weighted_validation_selected_experiment.json"), {
            "selected_by": "best_val_miou", "experiment_id": selected["experiment_id"],
            "best_val_miou": selected["best_val_miou"], "evaluation_size": [384, 384],
            "checkpoint": str(project_path(next(item["output_dir"] for item, *_ in entries
                                                  if item["experiment_id"] == selected["experiment_id"])) / "best_model"),
        })
        for filename, baseline_metrics, candidate_metrics in (
            ("weighted_per_class.csv", control, candidate),
            ("weighted_validation_per_class.csv", control_val, candidate_val),
        ):
            with project_path(f"results/{filename}").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["class_id", "class_name", "E4_iou", "E5_iou", "delta_pp"])
                for index, name in enumerate(CLASS_NAMES):
                    before, after = baseline_metrics["per_class_iou"][index], candidate_metrics["per_class_iou"][index]
                    writer.writerow([index, name, before, after,
                                     None if before is None or after is None else 100 * (after - before)])
    else:
        report += ["", "E5 尚未完成，暂不判断加权是否有效，空白不代表零分。"]
    project_path("reports/weighted_ablation.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return rows
