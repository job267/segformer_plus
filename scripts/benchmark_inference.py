"""Measure E4 FP32 versus CUDA FP16 autocast on validation only, before training."""

import argparse
import csv
import json
from time import perf_counter

from _bootstrap import ROOT

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.comparison import read_json
from src.config import project_path
from src.inference import evaluate_precision, load_run_model, open_rgb, predict_image, predict_tensor
from src.labels import colorize_mask
from src.resolution import ResolutionDataset
from src.utils import choose_device, seed_everything, utc_now, write_json


def timing_summary(seconds):
    milliseconds = np.asarray(seconds) * 1000
    return {
        "samples": len(seconds), "mean_ms": float(milliseconds.mean()),
        "p50_ms": float(np.percentile(milliseconds, 50)),
        "p95_ms": float(np.percentile(milliseconds, 95)),
        "images_per_second": float(1000 / milliseconds.mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="results/E4_resolution")
    parser.add_argument("--output-dir", default="results/benchmark_E4")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 20:
        parser.error("warmup >=1 and iterations >=20 are required")
    device = choose_device("cuda")
    directory = project_path(args.output_dir)
    if directory.exists() and any(directory.iterdir()):
        parser.error("Choose an empty output directory to preserve measured benchmarks")
    directory.mkdir(parents=True, exist_ok=True)
    seed_everything(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model, config, archived = load_run_model(args.run_dir, device)
    val = ResolutionDataset(config, "val")
    resident = next(iter(DataLoader(val, batch_size=1)))["pixel_values"].to(device)
    paths = [sample.image_path for sample in val.samples[:10]]
    measurements = {}
    for precision in ("fp32", "fp16"):
        metrics = evaluate_precision(model, DataLoader(val, batch_size=config["training"]["batch_size"]), device, precision)
        if precision == "fp32":
            expected = read_json(project_path(args.run_dir) / "best_validation_metrics.json")
            if metrics["confusion_matrix"] != expected["confusion_matrix"]:
                raise ValueError("FP32 validation did not reproduce the archived E4 confusion matrix")
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated(device) / 1024**2
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(args.warmup):
            predict_tensor(model, resident, config["preprocessing"]["evaluation_size"], device, precision,
                           check_finite=False)
        torch.cuda.synchronize(device)
        times = []
        for _ in range(args.iterations):
            torch.cuda.synchronize(device)
            started = perf_counter()
            predict_tensor(model, resident, config["preprocessing"]["evaluation_size"], device, precision,
                           check_finite=False)
            torch.cuda.synchronize(device)
            times.append(perf_counter() - started)
        peak = torch.cuda.max_memory_allocated(device) / 1024**2
        pipeline_times = []
        for _ in range(3):
            for path in paths:
                torch.cuda.synchronize(device)
                started = perf_counter()
                image = open_rgb(path)
                labels = predict_image(model, image, config, device, precision)
                colored = colorize_mask(labels)
                Image.blend(image, colored, alpha=0.5)
                torch.cuda.synchronize(device)
                pipeline_times.append(perf_counter() - started)
        measurements[precision] = {
            "validation": metrics, "model_latency": timing_summary(times),
            "image_pipeline_latency": timing_summary(pipeline_times),
            "baseline_allocated_mb": baseline, "peak_allocated_mb": peak,
            "incremental_peak_mb": peak - baseline,
        }
    fp32, fp16 = measurements["fp32"], measurements["fp16"]
    report = {
        "status": "completed", "timestamp_utc": utc_now(), "experiment_id": config["experiment_id"],
        "checkpoint_sha256": archived["checkpoint_sha256"], "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device), "device": str(device),
        "model_weight_dtype": "float32", "fp16_mode": "CUDA float16 autocast; logits postprocessing float32",
        "input_size": config["preprocessing"]["size"], "evaluation_size": config["preprocessing"]["evaluation_size"],
        "accuracy_scope": "all 101 validation images; batch=4; no test inference",
        "model_latency_scope": "batch=1; GPU-resident input; forward + FP32 resize to 384x384 + argmax; synchronized wall time; excludes I/O and finite checks",
        "pipeline_scope": "first 10 val images repeated 3 times; open/decode/resize/normalize/H2D/forward/native-size prediction/D2H/colorize/overlay; excludes model load and saving files",
        "memory_scope": "torch allocated bytes for model + resident batch=1 input + timed inference; not total GPU usage",
        "warmup": args.warmup, "iterations": args.iterations, "measurements": measurements,
        "delta_val_miou_pp": 100 * (fp16["validation"]["miou"] - fp32["validation"]["miou"]),
        "model_speedup": fp32["model_latency"]["mean_ms"] / fp16["model_latency"]["mean_ms"],
    }
    write_json(directory / "benchmark.json", report)
    with (directory / "benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["precision", "val_miou", "pixel_accuracy", "model_mean_ms", "model_p95_ms", "model_fps", "pipeline_mean_ms", "peak_allocated_mb"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for precision, item in measurements.items():
            writer.writerow({
                "precision": precision, "val_miou": item["validation"]["miou"],
                "pixel_accuracy": item["validation"]["pixel_accuracy"],
                "model_mean_ms": item["model_latency"]["mean_ms"], "model_p95_ms": item["model_latency"]["p95_ms"],
                "model_fps": item["model_latency"]["images_per_second"],
                "pipeline_mean_ms": item["image_pipeline_latency"]["mean_ms"], "peak_allocated_mb": item["peak_allocated_mb"],
            })
    lines = [
        "# E4 FP32 / FP16 推理评测", "",
        f"GPU：{report['gpu']}；输入 512×512，模型测速 batch=1，验证评价 batch=4、标签 384×384。",
        "FP16 指 CUDA autocast，模型权重保留 FP32，输出插值/argmax 前转 FP32。",
        "完整 101 张验证图像；没有重新评价测试集。FP32 混淆矩阵与 E4 归档完全一致。", "",
        "| 精度 | Val mIoU | 模型均时/ms | P95/ms | 模型 FPS | 图片流程均时/ms | allocated 峰值/MiB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for precision, item in measurements.items():
        lines.append(f"| {precision} | {item['validation']['miou']:.2%} | {item['model_latency']['mean_ms']:.2f} | {item['model_latency']['p95_ms']:.2f} | {item['model_latency']['images_per_second']:.2f} | {item['image_pipeline_latency']['mean_ms']:.2f} | {item['peak_allocated_mb']:.2f} |")
    lines += [
        "", f"FP16 相对 FP32 的 Val mIoU 变化 {report['delta_val_miou_pp']:+.4f} 个百分点；模型均时加速比 {report['model_speedup']:.2f}×。",
        "", "模型计时包含 forward、384×384 logits 插值及 argmax，不含磁盘/预处理/传输；图片流程另外包括读图、预处理、GPU/CPU 传输、原图尺寸预测和着色叠加，不含保存文件。",
        f"模型先预热 {args.warmup} 次，再同步测量 {args.iterations} 次；图片流程为固定前 10 张验证图各 3 次。",
        "显存数字为 PyTorch allocated 峰值，含权重和常驻输入，不等于 GPU 总占用。桌面硬件运行的一次测量，不证明跨设备速度。E5 在本次测量之后启动。",
    ]
    project_path("reports/inference_benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"delta_val_miou_pp": report["delta_val_miou_pp"], "model_speedup": report["model_speedup"],
                      "results": str(directory / "benchmark.csv")}, indent=2))


if __name__ == "__main__":
    main()
