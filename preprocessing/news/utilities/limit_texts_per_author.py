#!/usr/bin/env python3
"""Keep the first N valid paraphrased records for each News author."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--per-author", type=int, default=10)
    parser.add_argument("--author-key", default="author")
    parser.add_argument("--bad-lines-path", type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.per_author < 1:
        raise ValueError("per-author must be positive")
    valid_rows = []
    counts = Counter()
    bad_rows = []
    with args.input_path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = line.strip()
            if not value:
                continue
            try:
                row = json.loads(value)
                if not isinstance(row, dict) or args.author_key not in row:
                    raise ValueError(f"missing '{args.author_key}'")
            except (json.JSONDecodeError, ValueError) as error:
                if args.strict:
                    raise ValueError(f"Invalid row {line_number}: {error}") from error
                bad_rows.append(value)
                continue
            counts[row[args.author_key]] += 1
            valid_rows.append(row)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    retained = defaultdict(int)
    written = 0
    with args.output_path.open("w", encoding="utf-8") as handle:
        for row in valid_rows:
            author = row[args.author_key]
            if retained[author] < args.per_author:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                retained[author] += 1
                written += 1
    if args.bad_lines_path:
        args.bad_lines_path.parent.mkdir(parents=True, exist_ok=True)
        with args.bad_lines_path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(bad_rows))
            if bad_rows:
                handle.write("\n")
    print(f"authors: {len(counts)}")
    print(f"written: {written}")
    print(f"invalid rows: {len(bad_rows)}")


if __name__ == "__main__":
    main()
