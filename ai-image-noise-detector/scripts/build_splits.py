"""Build a metadata + split CSV from a raw-data directory.

Examples:
    # random 70/15/15 split, stratified per generator
    python scripts/build_splits.py \
        --data-root data/raw/dummy --dataset-name Dummy \
        --output data/splits/dummy_random.csv

    # cross-generator split: listed generators go entirely to test
    python scripts/build_splits.py \
        --data-root data/raw/dummy --dataset-name Dummy \
        --split-type cross_generator --held-out dummy_gen_c \
        --output data/splits/dummy_crossgen.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.split_generators import (
    assign_cross_generator_split,
    assign_random_split,
    build_metadata,
    save_split_csv,
    summarize_split,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="directory containing real/ and fake/<generator>/")
    parser.add_argument("--dataset-name", required=True, help="e.g. Synthbuster")
    parser.add_argument("--output", type=Path, required=True,
                        help="output CSV path, e.g. data/splits/synthbuster_random.csv")
    parser.add_argument("--split-type", choices=("random", "cross_generator"),
                        default="random")
    parser.add_argument("--held-out", nargs="+", default=[],
                        help="generators held out for test (cross_generator only)")
    parser.add_argument("--fracs", nargs=3, type=float, default=(0.70, 0.15, 0.15),
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-by-source", action="store_true",
                        help="partition by source_id so a real photo and all fakes "
                             "derived from it stay in the same split (use for "
                             "paired datasets like Synthbuster)")
    args = parser.parse_args()

    df = build_metadata(args.data_root, args.dataset_name)

    if args.split_type == "random":
        df = assign_random_split(df, tuple(args.fracs), args.seed, args.group_by_source)
    else:
        if not args.held_out:
            parser.error("--held-out is required for --split-type cross_generator")
        df = assign_cross_generator_split(
            df, args.held_out, tuple(args.fracs), args.seed, args.group_by_source
        )

    path = save_split_csv(df, args.output)
    print(f"Wrote {len(df)} rows to {path}\n")
    print(summarize_split(df).to_string(index=False))


if __name__ == "__main__":
    main()
