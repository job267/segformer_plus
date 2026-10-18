"""Regenerate E4/E5 comparison from independently verified archives."""

import argparse
import json

from _bootstrap import ROOT
from src.config import load_config
from src.weighted import WEIGHTED_CONFIG, write_weighted_comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    print(json.dumps(write_weighted_comparison(load_config(WEIGHTED_CONFIG),
                                              allow_incomplete=args.allow_incomplete), indent=2))
