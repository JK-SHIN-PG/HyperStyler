#!/usr/bin/env python3
"""Write the filtered All-the-News DatasetDict as TinyStyler JSONL shards.

Keeps the train/valid author assignment of the input; does not resplit or
subsample texts.
"""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-splits", nargs=2, default=("train", "valid"))
    parser.add_argument("--author-column", default="author")
    parser.add_argument("--text-column", default="text")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only this stage's train/val JSONL and statistics file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import load_from_disk

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory already contains files: {args.output_dir}")
        allowed = {"train.jsonl", "val.jsonl", "statistics.json"}
        unexpected = [path for path in args.output_dir.iterdir() if path.name not in allowed]
        if unexpected:
            raise FileExistsError(
                "Refusing to overwrite a prepared directory containing other outputs: "
                + ", ".join(str(path) for path in unexpected)
            )

    dataset = load_from_disk(str(args.input_path))
    split_names = ("train", "val")
    source_to_output = dict(zip(args.source_splits, split_names))
    authors_by_split: dict[str, set[str]] = {}
    text_counts: dict[str, int] = {}

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for source_split, output_split in source_to_output.items():
        if source_split not in dataset:
            raise KeyError(f"Missing split {source_split!r}; available: {list(dataset)}")
        source = dataset[source_split]
        required = {args.author_column, args.text_column}
        missing = required - set(source.column_names)
        if missing:
            raise KeyError(f"{source_split!r} is missing columns: {sorted(missing)}")

        authors: set[str] = set()
        output_path = args.output_dir / f"{output_split}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for row in source:
                author = row[args.author_column]
                text = row[args.text_column]
                if not isinstance(author, str) or not author.strip():
                    raise ValueError(f"{source_split!r} contains an empty author")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"{source_split!r} contains empty text for {author!r}")
                authors.add(author)
                handle.write(json.dumps({"author": author, "text": text}, ensure_ascii=False) + "\n")
        authors_by_split[output_split] = authors
        text_counts[output_split] = len(source)

    author_sets = authors_by_split
    overlap = author_sets["train"] & author_sets["val"]
    if overlap:
        raise ValueError(f"Author leakage in source DatasetDict: train/val={len(overlap)}")

    statistics = {
        "source_splits": source_to_output,
        "author_counts": {split: len(authors) for split, authors in authors_by_split.items()},
        "text_counts": text_counts,
    }
    (args.output_dir / "statistics.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
