"""Verify E3/E4 archives and regenerate the separate resolution comparison."""

import argparse
import json

from _bootstrap import ROOT
from src.config import load_config
from src.resolution import RESOLUTION_CONFIG, write_resolution_comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    print(json.dumps(write_resolution_comparison(load_config(RESOLUTION_CONFIG),
                                                allow_incomplete=args.allow_incomplete), indent=2))
