from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from src.config import load_config
from src.inference import evaluate_precision, precision_context, predict_image, predict_tensor, write_prediction


class RoadModel(torch.nn.Module):
    def forward(self, pixel_values):
        logits = torch.zeros(len(pixel_values), 11, 4, 4)
        logits[:, 3] = 1
        return SimpleNamespace(logits=logits)


@pytest.mark.parametrize("precision", ["fp16", "invalid"])
def test_unsupported_cpu_precision_is_explicit(precision):
    with pytest.raises(ValueError):
        with precision_context(torch.device("cpu"), precision):
            pass


def test_native_size_export_and_ignore_metrics(tmp_path):
    config = load_config()
    config["preprocessing"]["size"] = [32, 32]
    image = Image.new("RGB", (53, 37), (10, 20, 30))
    prediction = predict_image(RoadModel(), image, config, torch.device("cpu"))
    assert prediction.shape == (37, 53) and np.all(prediction == 3)
    path = tmp_path / "input.png"
    image.save(path)
    output = tmp_path / "predictions"
    write_prediction(RoadModel(), path, config, torch.device("cpu"), "fp32", output)
    with Image.open(output / "index.png") as saved:
        assert saved.mode == "L" and saved.size == (53, 37)
        assert np.all(np.asarray(saved) == 3)
    with pytest.raises(FileExistsError):
        write_prediction(RoadModel(), path, config, torch.device("cpu"), "fp32", output)
    labels = torch.full((1, 8, 8), 3)
    labels[:, 0] = 255
    metrics = evaluate_precision(RoadModel(), [{"pixel_values": torch.zeros(1, 3, 32, 32), "labels": labels}], torch.device("cpu"), "fp32")
    assert metrics["miou"] == 1 and metrics["valid_pixels"] == 56


def test_nonfinite_logits_are_rejected():
    class BadModel(torch.nn.Module):
        def forward(self, pixel_values):
            return SimpleNamespace(logits=torch.full((1, 11, 4, 4), float("nan")))
    with pytest.raises(FloatingPointError):
        predict_tensor(BadModel(), torch.zeros(1, 3, 32, 32), [32, 32], torch.device("cpu"), "fp32")
