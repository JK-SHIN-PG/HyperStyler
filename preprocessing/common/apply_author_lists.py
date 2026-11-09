'''Build author_splits.json from the published author lists.

Replaces the random split in split_authors.py so that preprocessing uses the
exact train/validation/held-out authors of the paper experiments. The output
has the same format ({"train", "val", "test"}) and is read by subsample_data.py.
'''

import json
import os

import click


SPLIT_FILES = {
    'train': 'train_authors.json',
    'val': 'valid_authors.json',
    'test': 'held_out_authors.json',
}


@click.command()
@click.option('--path', type=str, required=True, help='preprocessed JSONL with an author_id field')
@click.option('--author_list_dir', type=str, required=True, help='directory with the published *_authors.json files')
@click.option('--out_dir', type=str, required=True, help='directory to write author_splits.json')
def main(path, author_list_dir, out_dir):
    author_lists = {}
    for split, file_name in SPLIT_FILES.items():
        with open(os.path.join(author_list_dir, file_name), 'r') as f:
            author_lists[split] = json.load(f)

    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        assert not set(author_lists[a]) & set(author_lists[b]), f'{a}/{b} author lists overlap'

    present = set()
    with open(path, 'r') as f:
        for line in f:
            present.add(json.loads(line)['author_id'])

    listed = set().union(*author_lists.values())
    author_splits = {}
    for split, authors in author_lists.items():
        author_splits[split] = [a for a in authors if a in present]
        missing = len(authors) - len(author_splits[split])
        print(f'{split}: {len(author_splits[split])} authors' + (f' ({missing} listed authors not found in {path})' if missing else ''))

    unlisted = len(present - listed)
    if unlisted:
        print(f'{unlisted} authors in {path} are not in the author lists and are excluded')

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'author_splits.json'), 'w') as f:
        json.dump(author_splits, f)


if __name__ == '__main__':
    main()
