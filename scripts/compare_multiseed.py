"""Recompute paired E4/E5 seed statistics without retraining or new test inference."""

import argparse
import json

from _bootstrap import ROOT
from src.multiseed import write_multiseed_comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    print(json.dumps(write_multiseed_comparison(allow_incomplete=args.allow_incomplete), indent=2))
