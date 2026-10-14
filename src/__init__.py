"""SegFormer-B0 / CamVid course project, Stage 0-1 foundation."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Set before importing transformers; caches belong to this workspace.
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(PROJECT_ROOT / ".cache" / "huggingface" / "hub"))
