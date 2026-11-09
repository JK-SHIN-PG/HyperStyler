# Third-party notices

## TinyStyler

https://github.com/zacharyhorvitz/TinyStyler (MIT License, as stated in its README)

Files adapted from TinyStyler:

- `utility/filtering_metrics.py` (`data/data_filtering_metrics.py`)
- `utility/style_utils.py` (`tinystyler/style_utils.py`)
- `utility/uar.py`, `utility/evaluation_metric.py` (`authorship_evaluation/styll_eval_metrics/`)
- `utility/authorship_eval.py` (`authorship_evaluation/run_authorship_eval.py`)
- the training, filtering, and evaluation entry points (`0_base_model_trainer.py`, `1_hyperstyler_trainer.py`, `3_data_filtering.py`, `4_self_distillation.py`, `5_evaluate.py`)
- files in `preprocessing/common/`

```
MIT License

Copyright (c) the TinyStyler authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Hugging Face Transformers

https://github.com/huggingface/transformers (Apache License 2.0)

`src/modeling_t5_for_hyperstyler_rerankcache.py` and `src/config_t5.py` are modified from the Transformers T5 implementation. The original copyright and license notice is kept at the top of each file.
