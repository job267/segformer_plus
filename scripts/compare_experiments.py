"""Recompute E0-E3 comparisons from verified raw experiment archives."""

import argparse
import json

from _bootstrap import ROOT
from src.comparison import CONFIG_PATHS, write_comparison
from src.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    rows = write_comparison(
        [load_config(path) for path in CONFIG_PATHS], allow_incomplete=args.allow_incomplete
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
