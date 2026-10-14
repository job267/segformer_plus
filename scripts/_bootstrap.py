"""Allow documented commands to work without installing a generic 'src' package."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src  # noqa: E402,F401 -- sets writable model caches before transformers import
