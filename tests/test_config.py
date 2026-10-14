from copy import deepcopy

import pytest

from src.config import load_config, validate_config


def test_experiments_change_only_the_planned_factors():
    names = ["baseline", "augmentation", "dice_loss", "combined"]
    configurations = [load_config(f"configs/{name}.yaml") for name in names]
    assert [(c["augmentation"]["enabled"], c["training"]["loss"]) for c in configurations] == [
        (False, "ce"), (True, "ce"), (False, "ce_dice"), (True, "ce_dice"),
    ]
    comparable = []
    for config in configurations:
        config = deepcopy(config)
        config.pop("experiment_id")
        config.pop("output_dir")
        config["augmentation"]["enabled"] = False
        config["training"]["loss"] = "ce"
        comparable.append(config)
    assert all(config == comparable[0] for config in comparable)


def test_reducing_class_zero_is_rejected():
    config = load_config()
    config["model"]["do_reduce_labels"] = True
    with pytest.raises(ValueError, match="class 0"):
        validate_config(config)


def test_inheritance_cycle(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cycle"):
        load_config(tmp_path / "a.yaml")
