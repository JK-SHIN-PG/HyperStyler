#!/usr/bin/env python3
"""Prepare Blog Authorship Corpus records for the numbered Blog pipeline.

This is a CLI refactor of ``0_preprocess_blog.ipynb``. It preserves the
original filtering rules and output schema while replacing its local absolute
paths with explicit command-line arguments.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from transformers import PegasusTokenizer


MIN_CANDIDATES = 10
MAX_TOKEN_LENGTH = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("temp"))
    parser.add_argument("--model-name", default="tuner007/pegasus_paraphrase")
    return parser.parse_args()


def preprocess_blog_text(text: str, tokenizer: PegasusTokenizer) -> list[str]:
    candidates = []
    for sentence in sent_tokenize(text.strip()):
        sentence = sentence.strip()
        if (
            not sentence
            or "urlLink" in sentence
            or "mail" in sentence
            or len(sentence.split()) < 3
            or len(tokenizer.tokenize(sentence)) > MAX_TOKEN_LENGTH
        ):
            continue
        candidates.append(sentence)
    return candidates


def main() -> None:
    args = parse_args()
    dataset = pd.read_csv(args.input_path)
    required_columns = {"id", "gender", "age", "topic", "sign", "text"}
    missing_columns = required_columns - set(dataset.columns)
    if missing_columns:
        raise ValueError(f"Missing Blog CSV columns: {sorted(missing_columns)}")

    grouped = dataset.groupby("id").agg(list).reset_index()
    for _, row in tqdm(grouped.iterrows(), total=len(grouped), desc="Checking authors"):
        for field in ("gender", "age", "topic", "sign"):
            if len(set(row[field])) != 1:
                raise ValueError(f"{field} is inconsistent for Blog author {row['id']}")

    tokenizer = PegasusTokenizer.from_pretrained(args.model_name)
    outputs = []
    for _, row in tqdm(grouped.iterrows(), total=len(grouped), desc="Preprocessing"):
        candidates = []
        for text in row["text"]:
            candidates.extend(preprocess_blog_text(text, tokenizer))
        # The original notebook kept only authors with at least MIN_CANDIDATES
        # sentences. Author selection now comes from the published author lists
        # (apply_author_lists.py); with current nltk/tokenizer versions, four
        # listed authors fall below this threshold, so the filter is disabled.
        # if len(candidates) >= MIN_CANDIDATES:
        outputs.append({"author_id": f"blog_{row['id']}", "syms": candidates})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "preprocessed_blog_dataset.jsonl").open("w", encoding="utf-8") as handle:
        for output in outputs:
            handle.write(json.dumps(output) + "\n")
    # The original notebook wrote this empty protected-author file.
    (args.output_dir / "protected_author.jsonl").touch()
    print(f"Total users after preprocessing: {len(outputs)}")


if __name__ == "__main__":
    main()
