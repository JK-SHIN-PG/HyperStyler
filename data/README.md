# Data

```
data/
  raw/<dataset>/                 source corpora (not included)
  intermediate/<dataset>/        preprocessing outputs (not included)
  processed/<dataset>/           dataset_with_idx/, author_embs_{train,val}.lmdb (not included)
  splits/
    blog/, news/, mud/           train_authors.json, valid_authors.json, held_out_authors.json,
                                 generation_author_pairs.jsonl
  evaluation/
    blog/evaluation_dataset.json
    news/evaluation_dataset.json
    reddit/                      random.json, single.json, diverse.json, protected_authors.txt
```

The Blog and News evaluation authors (15 source, 15 target, 16 texts each) are held-out authors.

## Sources

- Blog: Blog Authorship Corpus (Schler et al., 2006) [link](https://www.kaggle.com/datasets/rtatman/blog-authorship-corpus/data).
- News: All the News 2.1, [`rjac/all-the-news-2-1-Component-one`](https://huggingface.co/datasets/rjac/all-the-news-2-1-Component-one).
- Reddit: Million Reddit User Dataset (MUD; Khan et al., 2021, [arXiv:2105.07263](https://arxiv.org/abs/2105.07263)).
- Reddit evaluation sets: STYLL (Patel et al., 2022, *Low-Resource Authorship Style Transfer: Can Non-Famous Authors Be Imitated?*).
