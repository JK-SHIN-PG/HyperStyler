"""Convert the train/val split into a Hugging Face DatasetDict."""

import os

from datasets import disable_caching, load_dataset


def convert_to_dataset(in_dir, out_dir):
    """Save ``in_dir/{train,val}.jsonl`` as a DatasetDict (val is reused as test)."""
    disable_caching()
    train_path = os.path.join(in_dir, 'train.jsonl')
    val_path = os.path.join(in_dir, 'val.jsonl')
    dataset = load_dataset(
        "json", data_files={"train": [train_path], "val": [val_path], "test": [val_path]}
    )
    dataset.save_to_disk(str(out_dir))
    print(f"Saved DatasetDict to {out_dir}: " + ", ".join(f"{k} {len(v)}" for k, v in dataset.items()))
