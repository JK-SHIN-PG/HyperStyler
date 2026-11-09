#!/usr/bin/env python3
"""Remove records whose generated paraphrase is empty."""

import argparse
import json
from pathlib import Path


def is_empty(paraphrase):
    if isinstance(paraphrase, list):
        return not paraphrase or not isinstance(paraphrase[0], str) or not paraphrase[0].strip()
    return not isinstance(paraphrase, str) or not paraphrase.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(args.in_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL files found in {args.in_dir}")
    if args.out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)

    total = 0
    removed = 0
    for input_path in files:
        output_path = args.out_dir / input_path.name
        file_total = 0
        file_removed = 0
        with input_path.open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as output:
            for line_number, line in enumerate(source, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON: {input_path}:{line_number}") from error
                file_total += 1
                if is_empty(row.get("paraphrase")):
                    file_removed += 1
                    continue
                output.write(line if line.endswith("\n") else line + "\n")

        total += file_total
        removed += file_removed
        print(f"{input_path.name}: kept {file_total - file_removed}, removed {file_removed}")

    print(f"total: kept {total - removed}, removed {removed}")


if __name__ == "__main__":
    main()
