"""Frozen paired E4/E5 seed replication, verified archives and descriptive statistics."""

import csv
import hashlib
import json

import numpy as np

from src.comparison import load_completed, read_json
from src.config import load_config, project_path, validate_config
from src.labels import CLASS_NAMES
from src.metrics import SegmentationMetrics
from src.utils import file_sha256, write_json
from src.weighted import WEIGHTS_PATH, validate_weighted

SEEDS = (42, 2024, 3407)
FAMILIES = ("E4", "E5")
REFERENCE_CONFIGS = {"E4": "configs/resolution.yaml", "E5": "configs/weighted.yaml"}
QUEUE_ORDER = tuple((family, seed) for seed in SEEDS[1:] for family in FAMILIES)
PREFLIGHT_PATH = "results/multiseed/preflight.json"


def replicate_config(family, seed):
    if family not in FAMILIES or seed not in SEEDS:
        raise ValueError("Only the frozen E4/E5 seed matrix is allowed")
    config = load_config(REFERENCE_CONFIGS[family])
    if seed != 42:
        config["seed"] = seed
        config["experiment_id"] = f"{family}_seed{seed}"
        config["output_dir"] = f"results/multiseed/{config['experiment_id']}"
    validate_config(config)
    return config


def validate_replicate(config, family, seed):
    if config != replicate_config(family, seed):
        raise ValueError("Replicate configuration differs from the frozen seed matrix")


def model_state_sha256(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(json.dumps([name, str(array.dtype), list(array.shape)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def verify_confusion_record(metrics):
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    if matrix.shape != (11, 11) or (matrix < 0).any() or matrix.sum() <= 0:
        raise ValueError("Invalid archived confusion matrix")
    calculator = SegmentationMetrics()
    calculator.confusion = matrix
    expected = calculator.compute()
    for key in ("class_pixels", "valid_pixels", "classes_in_miou"):
        if metrics[key] != expected[key]:
            raise ValueError(f"Archived {key} does not match confusion matrix")
    for key in ("miou", "pixel_accuracy"):
        if not np.isfinite(metrics[key]) or abs(metrics[key] - expected[key]) > 1e-12:
            raise ValueError(f"Archived {key} does not match confusion matrix")
    if len(metrics["per_class_iou"]) != 11:
        raise ValueError("Archived class IoU count differs")
    for actual, wanted in zip(metrics["per_class_iou"], expected["per_class_iou"]):
        if (actual is None) != (wanted is None) or (wanted is not None and (not np.isfinite(actual) or abs(actual - wanted) > 1e-12)):
            raise ValueError("Archived class IoU does not match confusion matrix")


def verify_references():
    configs = {family: replicate_config(family, 42) for family in FAMILIES}
    validate_weighted(configs["E5"], configs["E4"])
    references = {}
    for family in FAMILIES:
        config = configs[family]
        metrics, history, info = load_completed(config)
        validation = read_json(project_path(config["output_dir"]) / "best_validation_metrics.json")
        verify_confusion_record(validation)
        if abs(validation["miou"] - metrics["best_val_miou"]) > 1e-12:
            raise ValueError("Reference validation result differs from selected checkpoint")
        for relative, digest in info["source_sha256"].items():
            if file_sha256(project_path(relative)) != digest:
                raise ValueError(f"Frozen {family} source changed: {relative}")
        journal = read_json(project_path(config["output_dir"]) / "test_evaluation.json")
        completed = [attempt for attempt in journal["attempts"] if attempt.get("completed_utc")]
        if len(completed) != 1 or completed[0]["checkpoint_sha256"] != metrics["checkpoint_sha256"]:
            raise ValueError("Reference must retain one completed test of its selected checkpoint")
        references[family] = {"config": config, "metrics": metrics, "validation": validation,
                              "history": history, "info": info}
    e4, e5 = references["E4"], references["E5"]
    if e4["info"]["data_audit_sha256"] != e5["info"]["data_audit_sha256"]:
        raise ValueError("Reference data audits differ")
    for source in ("metrics", "validation"):
        if e4[source]["class_pixels"] != e5[source]["class_pixels"]:
            raise ValueError("Reference evaluation pixels differ")
    if e5["metrics"]["class_weights_sha256"] != file_sha256(project_path(WEIGHTS_PATH)):
        raise ValueError("Frozen train-only class weights changed")
    if e5["metrics"]["control_checkpoint_sha256"] != e4["metrics"]["checkpoint_sha256"]:
        raise ValueError("E5 refers to a different E4 control")
    return references


def load_replicate(family, seed, references):
    if seed == 42:
        return references[family]
    config = replicate_config(family, seed)
    directory = project_path(config["output_dir"])
    metrics, history, info = load_completed(config)
    reference = references[family]
    if info["family"] != family or info["seed"] != seed:
        raise ValueError("Replicate family/seed identity differs")
    if info["data_audit_sha256"] != reference["info"]["data_audit_sha256"]:
        raise ValueError("Replicate data differs from reference")
    if info["reference_checkpoint_sha256"] != reference["metrics"]["checkpoint_sha256"]:
        raise ValueError("Replicate reference checkpoint changed")
    if any(info["source_sha256"].get(path) != digest for path, digest in reference["info"]["source_sha256"].items()):
        raise ValueError("Replicate changed the reference implementation")
    if metrics["input_size"] != [512, 512] or metrics["evaluation_size"] != [384, 384]:
        raise ValueError("Replicate evaluation grid differs")
    if metrics["class_pixels"] != reference["metrics"]["class_pixels"]:
        raise ValueError("Replicate test pixels differ")
    validation = read_json(directory / "best_validation_metrics.json")
    verify_confusion_record(validation)
    if validation["class_pixels"] != reference["validation"]["class_pixels"] or abs(validation["miou"] - metrics["best_val_miou"]) > 1e-12:
        raise ValueError("Replicate validation pixels or selected score differ")
    expected_weights = file_sha256(project_path(WEIGHTS_PATH)) if family == "E5" else None
    if info.get("class_weights_sha256") != expected_weights or metrics.get("class_weights_sha256") != expected_weights:
        raise ValueError("Replicate weighting policy differs")
    journal = read_json(directory / "test_evaluation.json")
    completed = [attempt for attempt in journal["attempts"] if attempt.get("completed_utc")]
    if len(completed) != 1 or completed[0]["checkpoint_sha256"] != metrics["checkpoint_sha256"]:
        raise ValueError("Replicate must have one completed final test for its selected checkpoint")
    preflight = read_json(project_path(PREFLIGHT_PATH))
    if info["source_sha256"] != preflight["source_sha256"] or info["preflight_sha256"] != file_sha256(project_path(PREFLIGHT_PATH)):
        raise ValueError("Replicate workflow differs from the frozen preflight")
    if info["initial_model_sha256"] != preflight["initial_hashes"][str(seed)][family]:
        raise ValueError("Replicate initialization differs from the paired preflight")
    return {"config": config, "metrics": metrics, "validation": validation, "history": history, "info": info}


def sample_statistics(values):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("Statistics require finite scalar observations")
    return {"n": len(array), "mean": float(array.mean()) if len(array) else None,
            "std": float(array.std(ddof=1)) if len(array) > 1 else None}


def aggregate_records(records):
    paired = [seed for seed in SEEDS if all((family, seed) in records for family in FAMILIES)]
    families = []
    for family in FAMILIES:
        row = {"family": family, "paired_n": len(paired), "planned_n": len(SEEDS)}
        for source, field, prefix in (("metrics", "best_val_miou", "val_miou"),
                                      ("metrics", "test_miou", "test_miou"),
                                      ("metrics", "pixel_accuracy", "test_pixel_accuracy")):
            summary = sample_statistics([records[family, seed][source][field] for seed in paired])
            row[f"{prefix}_mean"], row[f"{prefix}_std"] = summary["mean"], summary["std"]
        families.append(row)
    differences = []
    for seed in SEEDS:
        row = {"seed": seed, "paired_completed": seed in paired}
        if seed in paired:
            e4, e5 = records["E4", seed]["metrics"], records["E5", seed]["metrics"]
            row.update(val_delta_pp=100*(e5["best_val_miou"]-e4["best_val_miou"]),
                       test_delta_pp=100*(e5["test_miou"]-e4["test_miou"]))
        differences.append(row)
    delta = {key: sample_statistics([row[key] for row in differences if row["paired_completed"]])
             for key in ("val_delta_pp", "test_delta_pp")}
    selected = max(families, key=lambda row: row["val_miou_mean"])["family"] if len(paired) == len(SEEDS) else None
    return families, differences, delta, selected


def write_csv(path, fields, rows):
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_multiseed_comparison(*, allow_incomplete=False):
    references = verify_references()
    records, runs = {}, []
    for seed in SEEDS:
        for family in FAMILIES:
            config = replicate_config(family, seed)
            status_path = project_path(config["output_dir"]) / "status.json"
            status = read_json(status_path)["status"] if status_path.exists() else "pending"
            row = {"family": family, "seed": seed, "experiment_id": config["experiment_id"], "status": status}
            if status == "completed":
                record = load_replicate(family, seed, references)
                records[family, seed] = record
                metrics = record["metrics"]
                row.update(best_val_epoch=metrics["best_val_epoch"], best_val_miou=metrics["best_val_miou"],
                           test_miou=metrics["test_miou"], pixel_accuracy=metrics["pixel_accuracy"])
            elif not allow_incomplete:
                raise ValueError(f"Replicate is not complete: {config['experiment_id']} {status}")
            runs.append(row)
    for seed in SEEDS[1:]:
        if all((family, seed) in records for family in FAMILIES):
            if records["E4", seed]["info"]["initial_model_sha256"] != records["E5", seed]["info"]["initial_model_sha256"]:
                raise ValueError("Paired initial model weights differ")
    families, differences, delta, selected = aggregate_records(records)
    write_csv("results/multiseed/runs.csv", ["family", "seed", "experiment_id", "status", "best_val_epoch", "best_val_miou", "test_miou", "pixel_accuracy"], runs)
    write_csv("results/multiseed/summary.csv", list(families[0]), families)
    write_csv("results/multiseed/paired_differences.csv", ["seed", "paired_completed", "val_delta_pp", "test_delta_pp"], differences)
    report = [
        "# E4/E5 多随机种子复核", "",
        f"固定种子 {list(SEEDS)}；已完成 {len(records)}/6 次训练归档。seed=42 复用原有 E4/E5，不重复训练或测试。",
        "下表仅统计两组都已完成的相同种子，避免中途样本数不一致；std 为样本标准差（ddof=1），不足两对时留空。",
        "每次均按最高 Val mIoU 选 checkpoint，Test 不参与模型或种子选择。", "",
        "| 方案 | 已配对种子数 | Val mIoU 均值±std | Test mIoU 均值±std |",
        "|---|---:|---:|---:|",
    ]
    def formatted(row, prefix):
        if row[f"{prefix}_mean"] is None:
            return ""
        mean, std = row[f"{prefix}_mean"], row[f"{prefix}_std"]
        return f"{mean:.2%}" + (f" ± {100*std:.2f} pp" if std is not None else "（std 待配对）")
    for row in families:
        report.append(f"| {row['family']} | {row['paired_n']}/3 | {formatted(row, 'val_miou')} | {formatted(row, 'test_miou')} |")
    report += ["", "| 种子 | E5−E4 Val 变化 | E5−E4 Test 变化 |", "|---|---:|---:|"]
    for row in differences:
        values = (f"{row['val_delta_pp']:+.3f} pp", f"{row['test_delta_pp']:+.3f} pp") if row["paired_completed"] else ("", "")
        report.append(f"| {row['seed']} | {' | '.join(values)} |")
    if selected is not None:
        report += ["", f"按三个种子的平均最佳 Val mIoU 选择 **{selected}**；同分保留 E4。",
                   "部署代表固定使用胜出方案原有 seed=42 的最佳模型，不挑选测试最高的种子，不集成。",
                   f"配对 Val 差值均值 {delta['val_delta_pp']['mean']:+.3f} pp，样本 std={delta['val_delta_pp']['std']:.3f} pp；"
                   f"配对 Test 差值均值 {delta['test_delta_pp']['mean']:+.3f} pp，样本 std={delta['test_delta_pp']['std']:.3f} pp。"]
        write_json(project_path("results/multiseed/selection.json"), {
            "selected_by": "mean_best_val_miou_over_fixed_seeds", "family": selected,
            "seeds": list(SEEDS), "representative_seed": 42,
            "checkpoint": str(project_path(references[selected]["config"]["output_dir"]) / "best_model"),
            "family_summary": families, "paired_delta_statistics_pp": delta,
        })
    else:
        report += ["", "全部三对完成前不判定哪组更稳定或更好，不生成最终方案选择。"]
    report += ["", "三种子仅为描述性复核，不证明统计显著性，也不反映场景抽样的不确定性；当前测试集已查看，不能称为新增盲测。",
               "Val/Test 每类均值与样本标准差见 results/multiseed/per_class_summary.csv，仍只使用完整配对的种子。"]
    classes = []
    paired = [seed for seed in SEEDS if all((family, seed) in records for family in FAMILIES)]
    for family in FAMILIES:
        for split, source in (("val", "validation"), ("test", "metrics")):
            for index, name in enumerate(CLASS_NAMES):
                observations = [records[family, seed][source]["per_class_iou"][index] for seed in paired]
                if any(value is None for value in observations):
                    raise ValueError("Missing class IoU would change the fixed comparison")
                classes.append({"family": family, "split": split, "class_id": index,
                                "class_name": name, **sample_statistics(observations)})
    write_csv("results/multiseed/per_class_summary.csv", ["family", "split", "class_id", "class_name", "n", "mean", "std"], classes)
    project_path("reports/multiseed.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"completed_runs": len(records), "planned_runs": 6, "families": families,
            "paired_differences": differences, "selected_family": selected}
