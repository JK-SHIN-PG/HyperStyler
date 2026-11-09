# Preprocessing

| Dataset | Command |
| --- | --- |
| Blog | `bash preprocessing/blog/run_preprocessing.sh data/raw/blog/blogtext.csv` |
| News | `python preprocessing/news/download_raw_news.py && bash preprocessing/news/run_preprocessing.sh` |
| MUD | `bash preprocessing/mud/run_preprocessing.sh data/raw/mud/data.jsonl data/evaluation/reddit/protected_authors.txt` |

Each runner writes intermediate files to `data/intermediate/<dataset>/` and the training DatasetDict and author LMDBs to `data/processed/<dataset>/`. Set `GPU_ID` to choose the GPU (default 0).

- `blog/`, `news/`, `mud/`: dataset-specific runners and stages.
- `common/`: shared stages (author split, paraphrasing, style embeddings, `idx` assignment).
- `build_author_embedding_lmdb.py`: author LMDBs from a DatasetDict with style embeddings.
- `prepare_generation_input.sh <dataset>`: stage-2 input (author pairs from `data/splits/<dataset>/generation_author_pairs.jsonl`, then paraphrases).
- `prepare_generation_pairs.py`: source and target texts for each author pair (called by `prepare_generation_input.sh`).
