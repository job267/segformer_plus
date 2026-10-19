"""Offline local image-upload demo; reuses verified archived models without new dependencies."""

import argparse
import base64
import binascii
import io
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src  # sets local caches before transformers
import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from src.inference import load_run_model, predict_image
from src.labels import CLASS_NAMES, PALETTE, colorize_mask
from src.utils import choose_device, seed_everything

RUNS = {"E5": ("类别加权 · 最终方案", "results/E5_weighted"),
        "E4": ("高分辨率对照", "results/E4_resolution"),
        "E0": ("初始基线", "results/E0_baseline")}
MAX_BODY = 24 * 1024**2


def png_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class Predictor:
    def __init__(self, device):
        self.device = device
        self.cache = {}
        self.lock = threading.Lock()

    def models(self):
        return [{"id": name, "label": label, "available": (ROOT / directory / "best_model/model.safetensors").is_file()}
                for name, (label, directory) in RUNS.items()]

    def predict(self, payload):
        name, precision = payload.get("model", "E5"), payload.get("precision", "fp32")
        if name not in RUNS or precision not in ("fp32", "fp16"):
            raise ValueError("请选择页面中提供的模型和精度")
        if precision == "fp16" and self.device.type != "cuda":
            raise ValueError("FP16 需要 CUDA，请选择 FP32")
        try:
            raw = base64.b64decode(payload["image"], validate=True)
        except (KeyError, ValueError, binascii.Error):
            raise ValueError("图片上传内容无效") from None
        if not raw or len(raw) > 16 * 1024**2:
            raise ValueError("请选择小于 16 MiB 的图片")
        try:
            with Image.open(io.BytesIO(raw)) as opened:
                if opened.width * opened.height > 24_000_000:
                    raise ValueError("图片尺寸过大，请先缩小到 2400 万像素以内")
                image = ImageOps.exif_transpose(opened).convert("RGB")
        except (UnidentifiedImageError, OSError):
            raise ValueError("无法解码图片，请使用 PNG、JPEG 等常见格式") from None
        with self.lock:
            started = perf_counter()
            if name not in self.cache:
                self.cache[name] = load_run_model(RUNS[name][1], self.device)
            model, config, _ = self.cache[name]
            load_ms = 1000 * (perf_counter() - started)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            started = perf_counter()
            labels = predict_image(model, image, config, self.device, precision)
            colored = colorize_mask(labels)
            overlay = Image.blend(image, colored, alpha=0.5)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            pipeline_ms = 1000 * (perf_counter() - started)
        counts = np.bincount(labels.reshape(-1), minlength=11)
        return {
            "model": name, "precision": precision, "width": image.width, "height": image.height,
            "pipeline_ms": pipeline_ms, "model_load_ms": load_ms,
            "timing_scope": "图片预处理、推理、原尺寸输出和叠加；不含模型加载、PNG 编码和传输",
            "segmentation": png_url(colored), "overlay": png_url(overlay),
            "index_mask": png_url(Image.fromarray(labels)),
            "classes": [{"name": label, "pixels": int(counts[index]), "color": color.tolist()}
                        for index, (label, color) in enumerate(zip(CLASS_NAMES, PALETTE)) if counts[index]],
        }


def handler_for(predictor):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/report/"):
                relative = unquote(self.path.removeprefix("/report/"))
                path = (ROOT / "reports" / relative).resolve()
                allowed = relative == "final_report.html" or (relative.startswith("figures/") and path.suffix == ".png")
                if not allowed or not path.is_relative_to((ROOT / "reports").resolve()) or not path.is_file():
                    self.send_json(404, {"error": "报告文件不存在"})
                    return
                body = path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8" if path.suffix == ".html" else "image/png")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/":
                body = (ROOT / "demo/index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/status":
                self.send_json(200, {"ready": True, "device": str(predictor.device),
                                     "fp16_available": predictor.device.type == "cuda", "models": predictor.models()})
            elif self.path == "/api/example":
                path = ROOT / "demo/example.png"
                if path.exists():
                    with Image.open(path) as example:
                        self.send_json(200, {"name": "CamVid 验证集示例", "image": png_url(example)})
                else:
                    self.send_json(404, {"error": "当前目录未附带示例，请上传本地图片"})
            else:
                self.send_json(404, {"error": "页面不存在"})

        def do_POST(self):
            if self.path != "/api/predict":
                self.send_json(404, {"error": "接口不存在"})
                return
            host = self.headers.get("Host", "")
            origin = self.headers.get("Origin")
            if host not in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"} or (origin and origin not in {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}):
                self.send_json(403, {"error": "请从本地演示页面发起预测"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_BODY or self.headers.get_content_type() != "application/json":
                    raise ValueError("上传内容需为 JSON，且小于 24 MiB")
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError("上传内容格式不正确")
                self.send_json(200, predictor.predict(payload))
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                self.send_json(400, {"error": str(error)})
            except Exception as error:
                self.send_json(500, {"error": f"预测失败：{type(error).__name__}：{error}"})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    seed_everything(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(Predictor(choose_device(args.device))))
    print(json.dumps({"url": f"http://127.0.0.1:{server.server_port}", "offline": True}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
