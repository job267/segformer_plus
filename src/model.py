"""Load a verified MiT-B0 encoder and create a fresh 11-class segmentation head."""

from copy import deepcopy
from pathlib import Path

from transformers import SegformerConfig, SegformerForSemanticSegmentation, SegformerModel

from src import PROJECT_ROOT
from src.labels import CLASS_NAMES
from src.utils import utc_now, write_json

B0_STRUCTURE = {
    "hidden_sizes": [32, 64, 160, 256], "depths": [2, 2, 2, 2],
    "num_attention_heads": [1, 2, 5, 8], "sr_ratios": [8, 4, 2, 1],
    "decoder_hidden_size": 256,
}


def build_model(config: dict, *, random_init=False, local_files_only=None, report_path=None):
    settings = config["model"]
    report = {
        "status": "loading", "timestamp_utc": utc_now(),
        "checkpoint": settings["checkpoint"], "requested_revision": settings["revision"],
        "initialization": "random_logic_test_only" if random_init else "ImageNet_MiT_B0",
        "segmentation_head": "randomly initialized; CamVid training required",
        "num_labels": settings["num_labels"], "do_reduce_labels": False,
    }
    try:
        if random_init:
            encoder_config = SegformerConfig(**B0_STRUCTURE)
            encoder = None
            loading_info = {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []}
        else:
            encoder, loading_info = SegformerModel.from_pretrained(
                settings["checkpoint"], revision=settings["revision"],
                cache_dir=str(PROJECT_ROOT / ".cache" / "huggingface" / "hub"),
                local_files_only=(
                    settings["local_files_only"] if local_files_only is None else local_files_only
                ),
                output_loading_info=True,
            )
            encoder_config = encoder.config
            report["resolved_revision"] = getattr(encoder_config, "_commit_hash", None)
        report["encoder_loading_info"] = loading_info
        for key, expected in B0_STRUCTURE.items():
            if getattr(encoder_config, key) != expected:
                raise RuntimeError(f"Checkpoint is not MiT-B0: {key} != {expected}")
        unexpected = set(loading_info.get("unexpected_keys", []))
        if unexpected - {"classifier.weight", "classifier.bias"}:
            raise RuntimeError(f"Unexplained unexpected encoder keys: {sorted(unexpected)}")
        if any(loading_info.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError(f"Encoder checkpoint did not load completely: {loading_info}")
        parameters = deepcopy(encoder_config.to_dict())
        parameters.update(
            num_labels=11, id2label=dict(enumerate(CLASS_NAMES)),
            label2id={label: i for i, label in enumerate(CLASS_NAMES)},
            semantic_loss_ignore_index=config["dataset"]["ignore_index"],
            architectures=["SegformerForSemanticSegmentation"],
        )
        model = SegformerForSemanticSegmentation(SegformerConfig(**parameters))
        if encoder is not None:
            copied = model.segformer.load_state_dict(encoder.state_dict(), strict=True)
            report["encoder_copy"] = {
                "missing_keys": copied.missing_keys, "unexpected_keys": copied.unexpected_keys,
            }
        report["fresh_head_parameter_names"] = [
            name for name, _ in model.named_parameters() if name.startswith("decode_head.")
        ]
        report["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
        report["status"] = "passed"
        if report_path:
            write_json(report_path, report)
        return model, report
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        if report_path:
            write_json(report_path, report)
        raise
