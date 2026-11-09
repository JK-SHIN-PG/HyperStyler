#!/usr/bin/env python3
"""Download the original All-the-News source into the release data layout."""

import argparse
from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "rjac/all-the-news-2-1-Component-one"
DEFAULT_OUTPUT = Path("data/raw/news/all_the_news_component_one")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {args.output_dir}")
    dataset = load_dataset(DATASET_NAME)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(args.output_dir))


if __name__ == "__main__":
    main()
