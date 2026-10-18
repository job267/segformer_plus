from copy import deepcopy

import numpy as np
import pytest
import torch

from src.config import load_config
from src.metrics import SegmentationMetrics
from src.multiseed import (
    FAMILIES, QUEUE_ORDER, REFERENCE_CONFIGS, SEEDS, aggregate_records,
    model_state_sha256, replicate_config, sample_statistics, validate_replicate,
    verify_confusion_record,
)


def record(val, test, seed=42, family="E4"):
    return {
        "config": replicate_config(family, seed),
        "metrics": {"best_val_miou": val, "test_miou": test, "pixel_accuracy": 0.9,
                    "best_val_epoch": 50, "per_class_iou": [test] * 11},
        "validation": {"per_class_iou": [val] * 11},
        "info": {"initial_model_sha256": f"seed{seed}"},
    }


def test_frozen_matrix_changes_only_seed_and_output_identity():
    assert QUEUE_ORDER == (("E4", 2024), ("E5", 2024), ("E4", 3407), ("E5", 3407))
    directories = set()
    for family in FAMILIES:
        reference = load_config(REFERENCE_CONFIGS[family])
        assert replicate_config(family, 42) == reference
        for seed in SEEDS[1:]:
            candidate = replicate_config(family, seed)
            directories.add(candidate["output_dir"])
            normalized = deepcopy(candidate)
            normalized.update(seed=42, experiment_id=reference["experiment_id"], output_dir=reference["output_dir"])
            assert normalized == reference
    assert len(directories) == 4
    with pytest.raises(ValueError):
        replicate_config("E5", 123)


def test_unplanned_optimizer_change_is_rejected():
    config = replicate_config("E5", 2024)
    config["training"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="configuration differs"):
        validate_replicate(config, "E5", 2024)


def test_statistics_use_sample_std_and_do_not_invent_one_seed_uncertainty():
    assert sample_statistics([]) == {"n": 0, "mean": None, "std": None}
    assert sample_statistics([0.7]) == {"n": 1, "mean": 0.7, "std": None}
    result = sample_statistics([0.6, 0.7, 0.8])
    assert result["mean"] == pytest.approx(0.7)
    assert result["std"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="finite"):
        sample_statistics([float("nan")])


def test_unpaired_run_is_excluded_and_no_early_selection():
    records = {("E4", 42): record(0.7, 0.6), ("E5", 42): record(0.71, 0.61),
               ("E4", 2024): record(0.99, 0.99, 2024)}
    families, pairs, delta, selected = aggregate_records(records)
    assert families[0]["paired_n"] == families[1]["paired_n"] == 1
    assert families[0]["val_miou_mean"] == 0.7
    assert families[0]["val_miou_std"] is None
    assert "val_delta_pp" not in pairs[1]
    assert delta["val_delta_pp"]["n"] == 1 and selected is None


def test_complete_pairs_select_by_mean_validation_and_report_paired_variation():
    records = {}
    for index, seed in enumerate(SEEDS):
        records["E4", seed] = record(0.7, 0.9, seed, "E4")
        records["E5", seed] = record(0.71 + 0.01*index, 0.6, seed, "E5")
    families, pairs, delta, selected = aggregate_records(records)
    assert selected == "E5"  # Its test score is deliberately lower.
    assert families[1]["val_miou_mean"] == pytest.approx(0.72)
    assert families[1]["val_miou_std"] == pytest.approx(0.01)
    assert delta["val_delta_pp"]["mean"] == pytest.approx(2)
    assert delta["val_delta_pp"]["std"] == pytest.approx(1)
    assert delta["test_delta_pp"]["mean"] == pytest.approx(-30)


def test_validation_tie_retains_e4():
    records = {(family, seed): record(0.7, 0.9 if family == "E5" else 0.6, seed, family)
               for seed in SEEDS for family in FAMILIES}
    assert aggregate_records(records)[-1] == "E4"


def test_validation_confusion_is_recomputed_and_corruption_rejected():
    calculator = SegmentationMetrics()
    calculator.confusion = np.eye(11, dtype=np.int64) * 10
    metrics = calculator.compute()
    verify_confusion_record(metrics)
    broken = deepcopy(metrics)
    broken["per_class_iou"][2] = float("nan")
    with pytest.raises(ValueError, match="class IoU"):
        verify_confusion_record(broken)
    broken = deepcopy(metrics)
    broken["miou"] = 0.9
    with pytest.raises(ValueError, match="miou"):
        verify_confusion_record(broken)


def test_initial_weight_hash_is_deterministic_and_does_not_consume_rng():
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4))
    state = torch.get_rng_state().clone()
    expected = model_state_sha256(model)
    assert model_state_sha256(deepcopy(model)) == expected
    assert torch.equal(state, torch.get_rng_state())
    with torch.no_grad():
        model[0].weight[0, 0] += 1
    assert model_state_sha256(model) != expected


@pytest.mark.parametrize("complete", [False, True])
def test_report_requires_all_pairs_and_keeps_seed42_representative(tmp_path, monkeypatch, complete):
    import json
    import src.multiseed as multiseed
    from src.utils import write_json

    records = {(family, seed): record(0.7 if family == "E4" else 0.72,
                                     0.9 if family == "E4" else 0.6, seed, family)
               for family in FAMILIES for seed in SEEDS}
    references = {family: records[family, 42] for family in FAMILIES}
    monkeypatch.setattr(multiseed, "project_path", lambda value: tmp_path / value)
    monkeypatch.setattr(multiseed, "verify_references", lambda: references)
    monkeypatch.setattr(multiseed, "load_replicate", lambda family, seed, refs: records[family, seed])
    (tmp_path / "reports").mkdir()
    for (family, seed), item in records.items():
        if complete or seed == 42:
            write_json(tmp_path / item["config"]["output_dir"] / "status.json", {"status": "completed"})
    result = multiseed.write_multiseed_comparison(allow_incomplete=True)
    selection = tmp_path / "results/multiseed/selection.json"
    if complete:
        selected = json.loads(selection.read_text())
        assert selected["family"] == "E5" and selected["representative_seed"] == 42
        assert selected["checkpoint"].endswith("E5_weighted\\best_model") or selected["checkpoint"].endswith("E5_weighted/best_model")
        assert result["completed_runs"] == 6
    else:
        assert not selection.exists() and result["selected_family"] is None
        assert result["completed_runs"] == 2
