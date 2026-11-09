"""Split filtered pairs into training and validation sets."""

import glob
import math
import os
import random


def split_train_val(in_dir, out_dir, percent_val=0.05, seed=42):
    """Shuffle all lines of ``in_dir/*.jsonl`` and write train.jsonl / val.jsonl."""
    random.seed(seed)

    all_file_paths = sorted(glob.glob(os.path.join(in_dir, "*.jsonl")))
    print(f"Found {len(all_file_paths)} files in {in_dir}")

    all_data = []
    for file_path in all_file_paths:
        with open(file_path, 'r') as f:
            all_data.extend(f)

    if not all_data:
        raise SystemExit(f"No generations passed filtering in {in_dir}; nothing to split.")

    random.shuffle(all_data)
    num_val = math.ceil(len(all_data) * percent_val)
    val_data = all_data[:num_val]
    train_data = all_data[num_val:]

    os.makedirs(out_dir, exist_ok=True)
    print(f"Writing {len(train_data)} training and {len(val_data)} validation examples to {out_dir}")
    with open(os.path.join(out_dir, 'train.jsonl'), 'w') as f:
        f.writelines(train_data)
    with open(os.path.join(out_dir, 'val.jsonl'), 'w') as f:
        f.writelines(val_data)
