# News preprocessing

```bash
python preprocessing/news/download_raw_news.py   # rjac/all-the-news-2-1-Component-one
bash preprocessing/news/run_preprocessing.sh
```

`0_preprocess_news.py` keeps articles with a single individual author and selects 16 sentences per author. Authors are split 90/5/5 with a fixed seed; the split used in the paper is in `data/splits/news/`. If an output from an earlier run exists, the runner asks whether to rebuild it. Outputs: `data/processed/news/dataset_with_idx`, `author_embs_train.lmdb`, `author_embs_val.lmdb`.
