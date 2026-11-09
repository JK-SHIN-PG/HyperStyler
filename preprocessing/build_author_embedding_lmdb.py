import argparse
import json
import os
import pickle
import shutil
from collections import defaultdict

import lmdb
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from datasets import load_from_disk


DEFAULT_SPLIT_NAME_MAP = {
    "train": "train",
    "val": "val",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Group per-example style embeddings by author_id and write them to "
            "LMDB files compatible with AuthorEmbLMDB."
        )
    )
    parser.add_argument(
        "--dataset-path",
        default="data/processed/blog/dataset_with_idx",
        help="Path to a Hugging Face DatasetDict saved with load_from_disk.",
    )
    parser.add_argument(
        "--output-root",
        default="data/processed/blog",
        help="Directory where author_embs_<split>.lmdb folders will be created.",
    )
    parser.add_argument(
        "--author-column",
        default="author_id",
        help="Column name containing author identifiers.",
    )
    parser.add_argument(
        "--embedding-column",
        default="style_embedding",
        help="Column name containing per-example embedding vectors.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Dataset splits to export.",
    )
    parser.add_argument(
        "--map-size-bytes",
        type=int,
        default=2_147_483_648,
        help="LMDB map size in bytes. Increase this if your dataset is larger.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output LMDB directory before rebuilding it.",
    )
    return parser.parse_args()


def ensure_output_dir(path: str, overwrite: bool):
    if os.path.exists(path):
        if not overwrite:
            raise FileExistsError(
                f"Output path already exists: {path}. Use --overwrite to rebuild it."
            )
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def iter_rows_from_arrow_split(split_path: str):
    state_path = os.path.join(split_path, "state.json")
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    for data_file in state["_data_files"]:
        arrow_path = os.path.join(split_path, data_file["filename"])
        with pa.memory_map(arrow_path, "r") as source:
            reader = ipc.open_stream(source)
            for batch in reader:
                yield from batch.to_pylist()


def load_dataset_split(dataset_path: str, split: str):
    split_path = os.path.join(dataset_path, split)
    return iter_rows_from_arrow_split(split_path)


def collect_author_embeddings(rows, author_column: str, embedding_column: str):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[author_column]].append(
            np.asarray(row[embedding_column], dtype=np.float32)
        )

    stacked = {}
    for author_id, embeddings in grouped.items():
        stacked[author_id] = np.stack(embeddings, axis=0)
    return stacked


def write_lmdb(author_embeddings: dict, output_path: str, map_size_bytes: int):
    env = lmdb.open(output_path, map_size=map_size_bytes)
    try:
        with env.begin(write=True) as txn:
            for author_id, embeddings in author_embeddings.items():
                txn.put(author_id.encode("utf-8"), pickle.dumps(embeddings, protocol=4))
    finally:
        env.sync()
        env.close()


def main():
    args = parse_args()
    dataset = None
    available_splits = None
    try:
        dataset = load_from_disk(args.dataset_path)
        available_splits = list(dataset.keys())
    except Exception as exc:
        print(
            "load_from_disk failed, falling back to direct Arrow reading: "
            f"{type(exc).__name__}: {exc}"
        )
        available_splits = [
            name
            for name in os.listdir(args.dataset_path)
            if os.path.isdir(os.path.join(args.dataset_path, name))
            and os.path.exists(os.path.join(args.dataset_path, name, "state.json"))
        ]

    os.makedirs(args.output_root, exist_ok=True)

    for split in args.splits:
        if split not in available_splits:
            raise KeyError(
                f"Split '{split}' not found in dataset. Available splits: {available_splits}"
            )

        output_split_name = DEFAULT_SPLIT_NAME_MAP.get(split, split)
        output_path = os.path.join(
            args.output_root, f"author_embs_{output_split_name}.lmdb"
        )
        ensure_output_dir(output_path, args.overwrite)
        print(f"[{split}] collecting author embeddings...", flush=True)

        author_embeddings = collect_author_embeddings(
            dataset[split] if dataset is not None else load_dataset_split(args.dataset_path, split),
            author_column=args.author_column,
            embedding_column=args.embedding_column,
        )
        write_lmdb(author_embeddings, output_path, args.map_size_bytes)

        num_authors = len(author_embeddings)
        num_rows = sum(embeddings.shape[0] for embeddings in author_embeddings.values())
        example_shape = next(iter(author_embeddings.values())).shape
        print(
            f"[{split}] wrote {num_authors} authors / {num_rows} rows to {output_path} "
            f"(example author tensor shape: {example_shape})"
        , flush=True)


if __name__ == "__main__":
    main()
