# HyperStyler

Official code for **HyperStyler: Low-resource Authorship Style Transfer via Context-aware Style Navigation and Hypernetworks** (EMNLP 2026) [paper](https://aclanthology.org/2026.emnlp-main.1045/).

## Repository structure

```
configs/                 one config per dataset: blog.yaml, news.yaml, mud.yaml
preprocessing/           data preparation for each dataset
data/
  splits/                author splits and generation author pairs
  evaluation/            evaluation sets
src/                     model (HyperStyler, modified T5) and data collators
utility/                 metrics, style embeddings, experiment helpers
0_base_model_trainer.py  stage 0: base reconstruction model
1_hyperstyler_trainer.py stage 1: HyperStyler reconstruction
2_generate_main.py       stage 2: synthetic pair generation
3_data_filtering.py      stage 3: filtering
4_self_distillation.py   stage 4: self-distillation
5_evaluate.py            stage 5: evaluation
```

## Setup

Python 3.9, PyTorch 2.5.1, Transformers 4.50.0.

```bash
pip install -r requirements.txt
bash utility/download_luar.sh   # LUAR checkpoint for evaluation
```

## Data

| Dataset | Source | Preparation |
| --- | --- | --- |
| Blog | Blog Authorship Corpus (`blogtext.csv`) | `bash preprocessing/blog/run_preprocessing.sh data/raw/blog/blogtext.csv` |
| News | All the News | `python preprocessing/news/download_raw_news.py && bash preprocessing/news/run_preprocessing.sh` |
| Reddit | Million Reddit User Dataset (MUD) | `bash preprocessing/mud/run_preprocessing.sh data/raw/mud/data.jsonl` |

Download links and references for the raw data are in [data/README.md](data/README.md). For reproducibility, the authors in each split are stored in `data/splits/<dataset>/`, and the preprocessing code builds the datasets from these author lists.

## Training

Replace `blog` with `news` or `mud` for the other datasets.

```bash
export CUDA_VISIBLE_DEVICES=0,1   # --num_processes must match the number of GPUs

# stage 0, once per dataset -> outputs/blog/0_base/
accelerate launch --num_processes 2 --main_process_port 29500 \
  0_base_model_trainer.py --config configs/blog.yaml

EXP=outputs/blog/hyperstyler
accelerate launch --num_processes 2 --main_process_port 29500 \
  1_hyperstyler_trainer.py --exp_dir $EXP --config configs/blog.yaml

# stage 2 input, once per dataset: paraphrased source texts for the
# author pairs in data/splits/blog/generation_author_pairs.jsonl
bash preprocessing/prepare_generation_input.sh blog
python 2_generate_main.py --exp_dir $EXP     # one worker per visible GPU
python 3_data_filtering.py --exp_dir $EXP
accelerate launch --num_processes 2 --main_process_port 29500 \
  4_self_distillation.py --exp_dir $EXP
```

## Evaluation

```bash
python 5_evaluate.py --exp_dir $EXP
```

This evaluates `$EXP/4_self_distillation/best_model.pt`. Use `--checkpoint` to evaluate another checkpoint file or folder. For example, the released Reddit model ([JKSHINPG/HyperStyler](https://huggingface.co/JKSHINPG/HyperStyler)):

```bash
hf download JKSHINPG/HyperStyler --local-dir checkpoints/reddit
python 5_evaluate.py --exp_dir outputs/mud/eval --config configs/mud.yaml --checkpoint checkpoints/reddit
```

Results are written to `$EXP/5_evaluation/<checkpoint folder>/num{N}_temp{T}_topp{P}/`. `scores_avg.json` holds the joint, away, towards and sim scores averaged over author pairs. Away and towards use LUAR embeddings, and sim uses Mutual Implication Score.

## Acknowledgements

The synthetic-data pipeline (generation, filtering, self-distillation) and the evaluation follow [TinyStyler](https://github.com/zacharyhorvitz/TinyStyler) (MIT), and parts of the code are adapted from it. The T5 model code is modified from [Hugging Face Transformers](https://github.com/huggingface/transformers) (Apache-2.0). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

MIT. See [LICENSE](LICENSE).

## Citation

```bibtex
@inproceedings{shin-etal-2026-hyperstyler,
    title = "{H}yper{S}tyler: Low-resource Authorship Style Transfer via Context-aware Style Navigation and Hypernetworks",
    author = "Shin, Jongkyung  and
      Jeon, Minguk  and
      Park, Chan Woo  and
      Lim, Chiehyeon",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Junyi Jessy  and
      Elliott, Desmond",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.emnlp-main.1045/",
    doi = "10.18653/2026.emnlp-main.1045",
    pages = "24006--24028",
    ISBN = "979-8-89176-459-0",
    abstract = "Low-resource authorship style transfer (LAST) aims to rewrite text into the style of an arbitrary target author using only a few reference examples while preserving the original meaning. Existing methods often struggle to achieve both high style fidelity and semantic preservation because they compress diverse references into a single static author embedding, which averages out context-dependent stylistic variation, and rely on hidden representations for style control, which entangle style with content. We propose HyperStyler, a novel architecture that decouples LAST into style selection and style realization. Stylo-navigator predicts style coordinates by jointly modeling the source context and target-author references, and Stylo-hypernet realizes them via dynamic parameter modulation instead of hidden-state injection. Our experiments on Reddit, Blog, and News datasets demonstrate that HyperStyler consistently outperforms prior methods including LLM-based approaches and generalizes robustly across domains. Notably, HyperStyler achieves superior performance with as few as 2.4{\%} additional parameters over T5-large, while being over 1.8{\texttimes} faster than LLMs at inference."
}
```
