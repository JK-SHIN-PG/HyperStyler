# MUD preprocessing

```bash
bash preprocessing/mud/run_preprocessing.sh \
  data/raw/mud/data.jsonl data/evaluation/reddit/protected_authors.txt
```

`data.jsonl` has one record per author with `author_id` and `syms` (a list of texts). Authors are split by the lists in `data/splits/mud/`; the evaluation authors in `protected_authors.txt` are held out. Outputs: `data/processed/mud/dataset_with_idx`, `author_embs_train.lmdb`, `author_embs_val.lmdb`.
