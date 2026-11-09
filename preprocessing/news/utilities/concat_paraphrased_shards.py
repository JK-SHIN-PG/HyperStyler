#!/usr/bin/env python3
"""Concatenate News paraphrase JSONL shards in an explicit caller-supplied order."""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--skip-invalid-json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in args.input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with args.output_path.open("w", encoding="utf-8") as output:
        for path in args.input_paths:
            with path.open(encoding="utf-8") as input_file:
                for line_number, line in enumerate(input_file, start=1):
                    value = line.strip()
                    if not value:
                        continue
                    try:
                        json.loads(value)
                    except json.JSONDecodeError as error:
                        if args.skip_invalid_json:
                            skipped += 1
                            continue
                        raise ValueError(f"Invalid JSON in {path}:{line_number}") from error
                    output.write(value + "\n")
                    written += 1
    print(f"files: {len(args.input_paths)}")
    print(f"written: {written}")
    print(f"skipped invalid rows: {skipped}")


if __name__ == "__main__":
    main()
