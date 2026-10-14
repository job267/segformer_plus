"""Small YAML configuration loader with explicit inheritance and validation."""

from copy import deepcopy
from pathlib import Path

import yaml

from src import PROJECT_ROOT


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def deep_merge(base: dict, overrides: dict) -> dict:
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path = "configs/baseline.yaml", _seen=None) -> dict:
    path = project_path(path)
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError(f"Configuration inheritance cycle: {path}")
    seen.add(path)
    with path.open(encoding="utf-8-sig") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    parent = config.pop("extends", None)
    if parent:
        config = deep_merge(load_config(path.parent / parent, seen), config)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    required = {
        "experiment_id", "seed", "device", "output_dir", "model", "dataset",
        "preprocessing", "augmentation", "training",
    }
    if set(config) != required:
        raise ValueError(f"Configuration keys differ from schema: {set(config) ^ required}")
    if type(config["seed"]) is not int:
        raise ValueError("seed must be an integer")
    if config["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if config["model"]["num_labels"] != 11:
        raise ValueError("This frozen project uses exactly 11 classes")
    if config["model"]["do_reduce_labels"] is not False:
        raise ValueError("CamVid class 0 is valid; do_reduce_labels must be false")
    if config["dataset"]["ignore_index"] != 255:
        raise ValueError("This CamVid profile uses ignore_index=255")
    if config["dataset"]["label_encoding"] not in ("index", "rgb"):
        raise ValueError("label_encoding must be explicitly index or rgb")
    if set(config["dataset"]["splits"]) != {"train", "val", "test"}:
        raise ValueError("Exactly train/val/test manifests are required")
    size = config["preprocessing"]["size"]
    if len(size) != 2 or any(type(x) is not int or x < 32 for x in size):
        raise ValueError("size must be [height, width], both >= 32")
    for key in ("mean", "std"):
        values = config["preprocessing"][key]
        if len(values) != 3 or any(not isinstance(x, (int, float)) for x in values):
            raise ValueError(f"{key} requires three numeric values")
    if any(x <= 0 for x in config["preprocessing"]["std"]):
        raise ValueError("std must be positive")
    augmentation = config["augmentation"]
    if type(augmentation["enabled"]) is not bool:
        raise ValueError("augmentation.enabled must be boolean")
    if not 0 <= augmentation["horizontal_flip_probability"] <= 1:
        raise ValueError("Flip probability must be in [0, 1]")
    if any(not 0 <= augmentation[key] <= 1 for key in ("brightness", "contrast", "saturation")):
        raise ValueError("Color jitter amplitudes must be in [0, 1]")
    training = config["training"]
    for key in ("batch_size", "epochs"):
        if type(training[key]) is not int or training[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    if type(training["num_workers"]) is not int or training["num_workers"] < 0:
        raise ValueError("num_workers must be a non-negative integer")
    if training["learning_rate"] <= 0 or training["weight_decay"] < 0:
        raise ValueError("Invalid optimizer parameters")
    if training["loss"] not in ("ce", "ce_dice") or training["dice_weight"] < 0:
        raise ValueError("Invalid loss configuration")
    if type(training["amp"]) is not bool:
        raise ValueError("training.amp must be boolean")


def save_config(config: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
