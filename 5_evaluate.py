"""Stage 5: generate on the evaluation set and score the generations.

Outputs go to ``<exp_dir>/5_evaluation/<checkpoint folder>/num{N}_temp{T}_topp{P}/``:
``generations.jsonl``, ``scores.json`` (per author pair), ``scores_avg.json``
and ``timing.json``. Reddit data types prefix each file name (``random_...``).
"""

import json
import os
import random
from pathlib import Path
from time import perf_counter

import click
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src.hyperstyler import HyperStyler
from utility.checkpoint import load_state_dict
from utility.authorship_eval import score_generations
from utility.evaluation_metric import away_score_batch, sim_score_batch, towards_score_batch
from utility.experiment import confirm_overwrite, init_experiment, resolve_checkpoint, run_log
from utility.style_utils import get_text_style_embedding


def get_authorshipstyler_model(model_path, device, base_config, styler_config):
    tokenizer = AutoTokenizer.from_pretrained(base_config['model_name'], legacy=True)
    model = HyperStyler(base_config['model_name'], styler_config, use_style=True)
    model.load_state_dict(load_state_dict(model_path))
    model.to(device)
    return tokenizer, model


def create_dataset(data, tgt_author_to_txtembs, src_author_to_txtembs=None):

    pair_dataset = []
    use_source_embs = src_author_to_txtembs is not None
    for source_author, source_comments in data['source_authors'].items():
        for target_author, target_comments in data['target_authors'].items():
            for s_idx, source_comment in enumerate(source_comments):
                sample = {
                    'source_author': source_author,
                    'target_author': target_author,
                    'source_text': source_comment,
                    'source_text_idx' : s_idx,
                    'source_author_texts': source_comments,
                    'target_author_texts': target_comments,
                }
                if use_source_embs:
                    sample['source_embedding'] = src_author_to_txtembs[source_author][s_idx]
                # tgt_embs : [num_samples, 768]
                sample['target_embedding'] = torch.stack(tgt_author_to_txtembs[target_author], dim=0)
                sample['target_embedding_mask'] = torch.ones((sample['target_embedding'].shape[0]))
                pair_dataset.append(sample)

    return pair_dataset

def generate_batch(model_path, device, source_data, source_embs_data, target_embs_data, target_embs_mask,
                   config,
                   batch_size, num_sequences, temperature, top_p, do_sample):
    base_config = config['base_config']
    styler_config = config['styler_config']
    model_load_start = perf_counter()
    tokenizer, model = get_authorshipstyler_model(model_path, device, base_config, styler_config)
    model_load_time = perf_counter() - model_load_start

    output_texts = []
    total_generation_time = 0.0

    for i in tqdm(range(0, len(source_data), batch_size)):
        results, generation_time = run_styler_batch(
            model = model,
            tokenizer = tokenizer,
            source_texts=source_data[i:i + batch_size], 
            source_embs_batch=None if source_embs_data is None else source_embs_data[i:i + batch_size],
            target_embs_batch=target_embs_data[i:i + batch_size],
            target_embs_mask = target_embs_mask[i:i+batch_size],
            temperature=temperature,
            top_p=top_p,
            max_input_token= base_config['max_encoder_len'],
            max_new_tokens=base_config['max_decoder_len'],
            device=device,
            num_sequences=num_sequences,
            do_sample=do_sample,
        )
        total_generation_time += generation_time
        output_texts += results

    timing_stats = {
        "model_load_time_sec": model_load_time,
        "generation_total_time_sec": total_generation_time,
        "generation_avg_time_per_batch_sec": total_generation_time / max(1, (len(source_data) + batch_size - 1) // batch_size),
        "generation_avg_time_per_sample_sec": total_generation_time / max(1, len(source_data)),
    }

    return output_texts, timing_stats

@torch.no_grad()
def run_styler_batch(
    model,
    tokenizer,
    source_texts,
    source_embs_batch,
    target_embs_batch,
    target_embs_mask,
    temperature,
    top_p,
    max_input_token,
    max_new_tokens,
    device,
    num_sequences,
    do_sample,
):

    bz = len(source_texts)

    inputs = tokenizer(
        source_texts,
        return_tensors='pt',
        padding='max_length',
        max_length=max_input_token,
        truncation=True,
    ).to(device)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    generation_start = perf_counter()
    output = model.generate(
    input_ids = inputs['input_ids'],
    attention_mask = inputs['attention_mask'],
    style_embs=target_embs_batch.to(device),
    style_masks=target_embs_mask.to(device),
    do_sample=do_sample,
    temperature=temperature,
    top_p=top_p,
    max_new_tokens=max_new_tokens,
    num_return_sequences=num_sequences,
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    generation_time = perf_counter() - generation_start
    
    generated_texts = tokenizer.batch_decode(output, skip_special_tokens=True)

    if num_sequences > 1:
        selected_embs = getattr(model.model, "_last_selected_embs", None)
        if selected_embs is None:
            with torch.no_grad():
                forward_output = model(
                input_ids = inputs['input_ids'],
                labels= inputs['input_ids'],  # placeholder; only selected_embs is used
                attention_mask = inputs['attention_mask'],
                style_embs=target_embs_batch.to(device),
                style_masks=target_embs_mask.to(device))
                selected_embs = forward_output.selected_embs
        batch_text_embs = []
        batch_texts = []
        for i in range(0, len(generated_texts), num_sequences):
            texts = generated_texts[i : i + num_sequences]
            batch_texts.append(texts)
            batch_text_embs.append(get_text_style_embedding(texts))

        batch_text_embs = torch.stack(batch_text_embs) # [batch_size, num_sequences, embedding_dim]
        batch_text_embs = batch_text_embs.permute(1,0,2)
        joint_scores = []
        for i in range(num_sequences):
            style_transferred_texts = [text[i] for text in batch_texts]
            source_asim = away_score_batch(batch_text_embs[i], source_embs_batch)

            towards_asim = towards_score_batch(batch_text_embs[i], selected_embs.detach().cpu().numpy())
            mis_score = sim_score_batch(style_transferred_texts, source_texts)
            joint_score = ((source_asim*towards_asim)**(1/2)*mis_score)**(1/2)

            joint_scores.append(joint_score)

        joint_scores = np.stack(joint_scores)
        max_idx = np.argmax(joint_scores, axis=0)
        best_texts = [batch_texts[b][max_idx[b]] for b in range(bz)]
    else:
        best_texts = generated_texts

    return best_texts, generation_time

def run_inference(*, config, model_path, data_path, out_path, timing_path):
    ev = config['evaluation']
    num_samples, temperature, top_p = ev['num_samples'], ev['temperature'], ev['top_p']
    batch_size = ev['batch_size']

    torch.cuda.reset_peak_memory_stats()
    random.seed(42)
    torch.manual_seed(42)
    overall_start = perf_counter()

    with open(data_path, 'r') as f:
        data = json.load(f)

    tgt_author_to_txtembs = {}
    target_embed_start = perf_counter()
    for target_author, target_comments in data['target_authors'].items():
        target_text_embs = get_text_style_embedding(target_comments)
        tgt_author_to_txtembs[target_author] = [x.detach().cpu() for x in target_text_embs]
    target_embed_time = perf_counter() - target_embed_start

    src_author_to_txtembs = None
    source_embed_time = 0.0
    if num_samples > 1:
        src_author_to_txtembs = {}
        source_embed_start = perf_counter()
        for source_author, source_comments in data['source_authors'].items():
            source_text_embs = get_text_style_embedding(source_comments)
            src_author_to_txtembs[source_author] = [x.detach().cpu() for x in source_text_embs]
        source_embed_time = perf_counter() - source_embed_start

    pair_dataset = create_dataset(data, tgt_author_to_txtembs, src_author_to_txtembs)

    source_data = [d["source_text"] for d in pair_dataset]
    target_embs_data = [d["target_embedding"] for d in pair_dataset]
    target_embs_mask = [d["target_embedding_mask"] for d in pair_dataset]
    source_embs_data = None
    if num_samples > 1:
        source_embs_data = torch.stack([d["source_embedding"] for d in pair_dataset])

    output, timing_stats = generate_batch(
        model_path,
        "cuda",
        source_data,
        source_embs_data,
        torch.stack(target_embs_data),
        torch.stack(target_embs_mask),
        config=config,
        batch_size=batch_size,
        num_sequences=num_samples,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
    )
    overall_time = perf_counter() - overall_start

    timing_summary = {
        "num_examples": len(pair_dataset),
        "batch_size": batch_size,
        "num_samples": num_samples,
        "target_embedding_time_sec": target_embed_time,
        "source_embedding_time_sec": source_embed_time,
        **timing_stats,
        "end_to_end_time_sec": overall_time,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }
    print(json.dumps(timing_summary, indent=2))
    with open(timing_path, "w") as f:
        json.dump(timing_summary, f, indent=2)

    with open(out_path, 'w') as f:
        for sample, transferred_text in zip(pair_dataset, output):
            sample['output'] = transferred_text
            sample.pop('source_embedding', None)
            del sample['target_embedding']
            del sample['target_embedding_mask']
            f.write(json.dumps(sample) + "\n")


@click.command()
@click.option('--exp_dir', type=str, required=True, help='Experiment directory, e.g. outputs/blog/<experiment>.')
@click.option('--config', 'config_path', type=str, default=None, help='Configuration to copy on first use.')
@click.option('--checkpoint', type=str, default=None,
              help='Checkpoint file (.pt or .safetensors) or a folder containing best_model.pt or '
                   'model.safetensors. Default: <exp_dir>/4_self_distillation/best_model.pt.')
def main(exp_dir, config_path, checkpoint):
    exp_dir = Path(exp_dir)
    config = init_experiment(exp_dir, config_path)
    ev = config['evaluation']
    model_path = Path(checkpoint) if checkpoint else resolve_checkpoint(config, exp_dir, 'self_distillation')
    if model_path.is_dir():
        found = [model_path / n for n in ('best_model.pt', 'model.safetensors') if (model_path / n).exists()]
        if not found:
            raise SystemExit(f'No best_model.pt or model.safetensors in {model_path}.')
        model_path = found[0]
    if not model_path.exists():
        raise SystemExit(f'{model_path} not found.')
    model = model_path.absolute().parent.name  # results folder: name of the checkpoint's folder

    setting = f"num{ev['num_samples']}_temp{ev['temperature']}_topp{ev['top_p']}"
    save_dir = exp_dir / '5_evaluation' / model / setting
    runs = []
    for data_type in ev['data_types']:
        prefix = f'{data_type}_' if data_type else ''
        data_path = config['paths']['evaluation_data']
        if data_type:
            data_path = os.path.join(data_path, f'{data_type}.json')
        files = {name: save_dir / f'{prefix}{name}'
                 for name in ('generations.jsonl', 'scores.json', 'scores_avg.json', 'timing.json')}
        runs.append((data_path, files))

    if not confirm_overwrite([p for _, files in runs for p in files.values()]):
        return
    save_dir.mkdir(parents=True, exist_ok=True)

    with run_log(exp_dir, f'5_evaluation/{model}/{setting}'):
        for data_path, files in runs:
            run_inference(
                config=config, model_path=model_path, data_path=data_path,
                out_path=files['generations.jsonl'], timing_path=files['timing.json'],
            )
            torch.cuda.empty_cache()
            score_generations(files['generations.jsonl'], files['scores.json'], files['scores_avg.json'])


if __name__ == '__main__':
    main()
