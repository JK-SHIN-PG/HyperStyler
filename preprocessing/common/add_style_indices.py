#!/usr/bin/env python3
"""Add the StyleNavigator index required by HyperStyler reconstruction training.

For each example, the index identifies the most cosine-similar vector in the
same author's split-specific LMDB entry.  The LMDB must be generated from the
same split and embedding model before running this script.
"""

import argparse
import pickle
from pathlib import Path

import lmdb
import torch
import torch.nn.functional as functional
from datasets import load_from_disk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--train-lmdb", type=Path, required=True)
    parser.add_argument("--val-lmdb", type=Path, required=True)
    parser.add_argument("--author-column", default="author_id")
    parser.add_argument("--embedding-column", default="style_embedding")
    return parser.parse_args()


class AuthorEmbeddings:
    def __init__(self, path: Path):
        self.env = lmdb.open(str(path), readonly=True, lock=False, readahead=True)

    def __getitem__(self, author_id: str) -> torch.Tensor:
        with self.env.begin() as txn:
            value = txn.get(author_id.encode("utf-8"))
        if value is None:
            raise KeyError(f"Author {author_id!r} is absent from the LMDB")
        return torch.as_tensor(pickle.loads(value), dtype=torch.float32)

    def close(self) -> None:
        self.env.close()


def main() -> None:
    args = parse_args()
    if args.output_path.exists():
        raise FileExistsError(f"Output path already exists: {args.output_path}")
    dataset = load_from_disk(str(args.dataset_path))
    lmdb_paths = {"train": args.train_lmdb, "val": args.val_lmdb}
    outputs = {}
    for split in ('train', 'val'):
        subset = dataset[split]
        embeddings = AuthorEmbeddings(lmdb_paths[split])

        def assign_index(example: dict) -> dict:
            query = functional.normalize(
                torch.as_tensor(example[args.embedding_column], dtype=torch.float32), dim=-1
            )
            candidates = functional.normalize(embeddings[example[args.author_column]], dim=-1)
            example["idx"] = int(torch.argmax(candidates @ query).item())
            return example

        try:
            outputs[split] = subset.map(assign_index)
        finally:
            embeddings.close()
    from datasets import DatasetDict

    DatasetDict(outputs).save_to_disk(str(args.output_path))


if __name__ == "__main__":
    main()
