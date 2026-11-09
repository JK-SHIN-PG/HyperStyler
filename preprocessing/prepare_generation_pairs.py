"""Build source-text inputs for HyperStyler synthetic-pair generation.

The input pair file contains one JSON object per line with ``author_id`` and
``target_author_id``.  For every pair, this script deterministically selects a
source text from a prepared DatasetDict and writes JSONL shards that can be
passed to ``common/generate_paraphrases.py``.  The paraphraser preserves both
author IDs and adds the required five ``paraphrase`` candidates.
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import click
from datasets import load_from_disk


def seeded_index(seed: int, author_id: str, size: int) -> int:
    """Choose one text per source author reproducibly across Python versions."""
    digest = hashlib.sha256(f"{seed}:{author_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % size


def load_pairs(path: Path) -> list[dict]:
    pairs = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            pair = json.loads(line)
            missing = {"author_id", "target_author_id"} - pair.keys()
            if missing:
                raise ValueError(
                    f"{path}:{line_number} is missing required field(s): "
                    f"{', '.join(sorted(missing))}"
                )
            pairs.append(pair)
    return pairs


@click.command()
@click.option(
    "--pairs-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="JSONL containing author_id and target_author_id pairs.",
)
@click.option(
    "--dataset-path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Prepared Hugging Face DatasetDict containing author_id and text.",
)
@click.option("--output-dir", type=click.Path(path_type=Path), required=True)
@click.option("--split", default="train", show_default=True)
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--max-per-file", default=5000, show_default=True, type=int)
@click.option("--overwrite", is_flag=True, help="Replace existing pairs-*.jsonl shards.")
def main(pairs_path, dataset_path, output_dir, split, seed, max_per_file, overwrite):
    """Select one source text per source author and write generation shards."""
    if max_per_file <= 0:
        raise click.BadParameter("must be positive", param_hint="--max-per-file")

    pairs = load_pairs(pairs_path)
    dataset_dict = load_from_disk(str(dataset_path))
    if split not in dataset_dict:
        raise click.ClickException(
            f"Split '{split}' not found; available: {', '.join(dataset_dict.keys())}"
        )

    source_authors = {pair["author_id"] for pair in pairs}
    texts_by_author = defaultdict(list)
    for row in dataset_dict[split]:
        author_id = row["author_id"]
        text = row["text"]
        if author_id in source_authors and isinstance(text, str) and text.strip():
            texts_by_author[author_id].append(text)

    missing_authors = sorted(source_authors - texts_by_author.keys())
    if missing_authors:
        preview = ", ".join(missing_authors[:10])
        suffix = " ..." if len(missing_authors) > 10 else ""
        raise click.ClickException(
            f"{len(missing_authors)} source author(s) have no usable text in "
            f"split '{split}': {preview}{suffix}"
        )

    selected_texts = {
        author_id: texts[seeded_index(seed, author_id, len(texts))]
        for author_id, texts in texts_by_author.items()
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    old_files = list(output_dir.glob("pairs-*.jsonl"))
    if old_files and not overwrite:
        raise click.ClickException(
            f"Output already contains {len(old_files)} pairs shard(s). "
            "Choose a new directory or pass --overwrite."
        )
    if overwrite:
        for old_file in old_files:
            old_file.unlink()

    for shard_index, start in enumerate(range(0, len(pairs), max_per_file)):
        output_path = output_dir / f"pairs-{shard_index:05d}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for pair in pairs[start : start + max_per_file]:
                handle.write(
                    json.dumps(
                        {
                            "author_id": pair["author_id"],
                            "target_author_id": pair["target_author_id"],
                            "text": selected_texts[pair["author_id"]],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    click.echo(
        f"Wrote {len(pairs)} pairs in {(len(pairs) + max_per_file - 1) // max_per_file} "
        f"shard(s); selected one text for each of {len(selected_texts)} source authors."
    )


if __name__ == "__main__":
    main()
