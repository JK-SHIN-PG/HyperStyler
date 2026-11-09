#!/usr/bin/env python3
"""Rename the News author column for HyperStyler training."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    from datasets import DatasetDict, load_from_disk

    if args.output_path.exists():
        raise FileExistsError(f"Output path already exists: {args.output_path}")
    dataset = load_from_disk(str(args.input_path))
    for split in ("train", "val"):
        subset = dataset[split]
        if "author" not in subset.column_names:
            raise KeyError(f"{split!r} has no 'author' column")
    DatasetDict({split: dataset[split].rename_column("author", "author_id") for split in ("train", "val")}).save_to_disk(str(args.output_path))


if __name__ == "__main__":
    main()
