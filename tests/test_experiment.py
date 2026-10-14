import csv
from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from src.artifacts import export_metric_tables
from src.config import load_config
from src.experiment import fit
from src.utils import seed_everything


@pytest.fixture
def tiny_run():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    config = load_config()
    config["training"].update(epochs=2, batch_size=2, amp=False, learning_rate=0.001)
    config["device"] = "cpu"
    yield config
    torch.set_num_threads(previous)


def setup_run():
    seed_everything(42)
    model = SegformerForSemanticSegmentation(SegformerConfig(
        num_labels=11, hidden_sizes=[4, 4, 8, 8], depths=[1, 1, 1, 1],
        num_attention_heads=[1, 1, 1, 1], sr_ratios=[1, 1, 1, 1],
        decoder_hidden_size=8, classifier_dropout_prob=0.1,
    ))
    # Separate fixture generator prevents accidental consumption of the model RNG.
    data_generator = torch.Generator().manual_seed(7)
    data = [
        {"pixel_values": torch.randn(3, 32, 32, generator=data_generator),
         "labels": torch.randint(0, 11, (32, 32), generator=data_generator)}
        for _ in range(4)
    ]
    train = DataLoader(data, batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(42))
    val = DataLoader(data, batch_size=2)
    return model, train, val


def test_resume_matches_uninterrupted_weights_and_logs(tmp_path, tiny_run):
    model, train, val = setup_run()
    complete = fit(model, train, val, tiny_run, torch.device("cpu"), tmp_path / "complete")
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model, train, val = setup_run()

    def interrupt_after_checkpoint(row):
        raise InterruptedError("simulated interruption after a committed epoch")

    with pytest.raises(InterruptedError):
        fit(model, train, val, tiny_run, torch.device("cpu"), tmp_path / "resume",
            on_epoch=interrupt_after_checkpoint)
    # Reconstructing a model/loader consumes RNG, which must be restored by resume.
    model, train, val = setup_run()
    resumed = fit(model, train, val, tiny_run, torch.device("cpu"), tmp_path / "resume", resume=True)
    assert resumed["best_epoch"] == complete["best_epoch"]
    assert all(torch.equal(value, expected[key]) for key, value in model.state_dict().items())
    for actual, wanted in zip(resumed["history"], complete["history"]):
        for key in ("epoch", "train_loss", "val_loss", "val_miou", "learning_rate"):
            assert actual[key] == wanted[key]
    with (tmp_path / "resume/train_log.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 2
    assert (tmp_path / "resume/best_model/model.safetensors").is_file()


def test_best_uses_validation_and_changed_resume_is_rejected(tmp_path, tiny_run, monkeypatch):
    model, train, val = setup_run()
    scores = iter([0.8, 0.2])
    monkeypatch.setattr("src.experiment.evaluate", lambda *args: {"miou": next(scores), "loss": 1.0})
    result = fit(model, train, val, tiny_run, torch.device("cpu"), tmp_path)
    assert result["best_epoch"] == 1
    assert result["best_miou"] == 0.8
    changed = deepcopy(tiny_run)
    changed["training"]["learning_rate"] = 0.1
    with pytest.raises(ValueError, match="configuration differs"):
        fit(model, train, val, changed, torch.device("cpu"), tmp_path, resume=True)


def test_confusion_csv_contains_numeric_columns(tmp_path):
    export_metric_tables({
        "per_class_iou": [0.5] * 11, "class_pixels": [10] * 11,
        "confusion_matrix": torch.eye(11, dtype=torch.int64).tolist(),
    }, tmp_path)
    with (tmp_path / "confusion_matrix.csv").open() as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 12
    assert all(len(row) == 12 for row in rows)
    assert [int(value) for value in rows[1][1:]] == [1] + [0] * 10
