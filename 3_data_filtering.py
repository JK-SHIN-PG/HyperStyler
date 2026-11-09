"""Stage 3: filter generated pairs, split train/val and build the DatasetDict."""

import json
import os
import re
from pathlib import Path

import click
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from utility.filtering_metrics import away_and_towards_score, mis_alternative_compute, mis_compute
from src.convert_to_dataset_format import convert_to_dataset
from src.hyperstyler import HyperStyler
from utility.checkpoint import load_state_dict
from src.split_finetune_train_val import split_train_val
from utility.experiment import (
    confirm_overwrite, init_experiment, load_experiment_config, resolve_checkpoint, run_log, run_workers, worker_share,
)
from utility.utils import AuthorEmbLMDB

STYLE_MODEL = "AnnaWegmann/Style-Embedding"
LINK_REGEX = r'(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])'


def get_valid_indices(outputs):
    return [i for i, text in enumerate(outputs) if text is not None]


def count_valid_tasks(tasks):
    count = 0
    for task in tasks:
        if any(task['output']):
            count += 1
    return count


def filter_based_on_identity(tasks):
    for task in tqdm(tasks):
        source_text = task['text'] # source_text
        output = task['output']

        for i in range(len(output)):
            if output[i] is None:
                continue
            if output[i].strip() == source_text.strip():
                output[i] = None

    tasks_remaining = count_valid_tasks(tasks)

    return tasks_remaining


def filter_to_max_results(tasks, max_per):
    for task in tqdm(tasks):
        output = task['output']

        for i in range(len(output)):
            if i >= max_per:
                output[i] = None


def flatten_refs_cands(tasks):
    all_indices = []
    all_refs = []
    all_cands = []

    for i, task in enumerate(tasks):
        source_text = task['text'] # source_text
        output = task['output']
        valid_indices = get_valid_indices(output)

        references = [source_text] * len(valid_indices)
        candidates = [output[i] for i in valid_indices]
        indices = [[i, val_idx] for val_idx in valid_indices]

        all_indices.extend(indices)
        all_refs.extend(references)
        all_cands.extend(candidates)

    return all_indices, all_refs, all_cands


def filter_based_on_model_score(tasks, threshold, model_fn=lambda refs, cands: 0):
    tasks_remaining = 0

    i = 1
    while f'sim_{i}' in tasks[0]:
        i += 1

    key = f'sim_{i}'

    for task in tasks:
        task[key] = [None for _ in range(len(tasks[0]['output']))]

    indices, references, candidates = flatten_refs_cands(tasks)
    flatten_scores = model_fn(references, candidates)

    for (task_idx, inference_idx), score in zip(indices, flatten_scores):
        tasks[task_idx][key][inference_idx] = score
        if score < threshold:
            tasks[task_idx]['output'][inference_idx] = None

    tasks_remaining = count_valid_tasks(tasks)
    return tasks_remaining



def add_style_features(tasks, author_embs_dict, model, tokenizer):
    for task in tqdm(tasks):
        source_author = task['author_id']
        source_emb = author_embs_dict[source_author]
        # Target: style embeddings predicted by the model during generation.
        target_emb = task['selected_embs']
        inputs = tokenizer(
            task['paraphrase'],
            return_tensors='pt',
            padding='max_length',
            max_length=80,
            truncation=True,
        ).to('cuda')
        # Source: style embeddings predicted by the model from the source author's
        # embeddings, one per paraphrase.
        num_txts = len(task['paraphrase'])
        source_emb = source_emb.unsqueeze(dim=0)
        source_emb = source_emb.repeat_interleave(num_txts, dim=0)  # [num_txts, num_author_texts, 768]
        source_embedding_mask = torch.ones((source_emb.shape[0], source_emb.shape[1]))
        with torch.no_grad():
            forward_output = model(
                input_ids=inputs['input_ids'],
                labels=inputs['input_ids'],  # placeholder; only selected_embs is used
                attention_mask=inputs['attention_mask'],
                style_embs=source_emb.to('cuda'),
                style_masks=source_embedding_mask.to('cuda'),
            )
            source_selected_embs = forward_output.selected_embs

        outputs = task['output']
        task['away_scores'] = [None for _ in range(len(outputs))]
        task['towards_scores'] = [None for _ in range(len(outputs))]
        for i in range(len(outputs)):
            if outputs[i] is not None:
                away, towards = away_and_towards_score(
                    style_transferred_texts=[outputs[i]],
                    embedding_type=f'style_{STYLE_MODEL}',
                    source_emb=source_selected_embs[i].unsqueeze(0).cpu().numpy(),
                    target_emb=np.array(target_emb[i]).reshape(1, -1),
                )
                task['away_scores'][i] = away
                task['towards_scores'][i] = towards


def get_style_transfer_stats(tasks):
    away_scores = []
    towards_scores = []

    for task in tasks:
        if any(task['away_scores']):
            away_scores.append(max([s for s in task['away_scores'] if s is not None]))
            towards_scores.append(
                max([s for s in task['towards_scores'] if s is not None])
            )

    away_scores = np.array(away_scores)
    towards_scores = np.array(towards_scores)

    away_mean = away_scores.mean()
    away_median = np.median(away_scores)
    away_std = away_scores.std()

    towards_mean = towards_scores.mean()
    towards_median = np.median(towards_scores)
    towards_std = towards_scores.std()

    print('Style transfer stats:')
    print(f'\tAway: (mean: {away_mean}, median: {away_median}, std: {away_std})')
    print(
        f'\tTowards: (mean: {towards_mean}, median: {towards_median}, std: {towards_std})'
    )

    print('Possible away threshold:', away_mean - away_std)
    print('Possible towards threshold:', towards_mean - towards_std)


def filter_based_on_style(*, tasks, away_threshold, towards_threshold):
    for task in tasks:
        outputs = task['output']
        for i in range(len(outputs)):
            if (
                task['away_scores'][i] is not None
                and task['away_scores'][i] < away_threshold
            ):
                outputs[i] = None
            if (
                task['towards_scores'][i] is not None
                and task['towards_scores'][i] < towards_threshold
            ):
                outputs[i] = None

    tasks_remaining = count_valid_tasks(tasks)

    return tasks_remaining


def filter_hallucinated_links(tasks, link_regex):
    # only keep links that are not hallucinated
    for task in tasks:
        outputs = task['output']
        source = task['text'] # source_text
        for i in range(len(outputs)):
            if outputs[i] is not None:
                links = re.findall(link_regex, outputs[i], re.IGNORECASE)
                if links:
                    source_links = re.findall(link_regex, source, re.IGNORECASE)
                    if not any([link in source_links for link in links]):
                        outputs[i] = None

    tasks_remaining = count_valid_tasks(tasks)

    return tasks_remaining


def filter_to_best_result(task):
    outputs = task['output']
    mis_scores = task['sim_1']
    away_scores = task['away_scores']
    towards_scores = task['towards_scores']

    best_idx = None
    best_score = -1

    for i in range(len(outputs)):
        if outputs[i] is not None:
            # score = mis_scores[i]
            score = (
                ((towards_scores[i] * away_scores[i]) ** (1 / 2)) * mis_scores[i]
            ) ** (
                1 / 2
            )  # NOTE: This is unnormalized, assumes mis is close to zero as is.
            if score > best_score:
                best_score = score
                best_idx = i

    for key in ['output', 'away_scores', 'towards_scores', 'sim_1', 'sim_2', 'paraphrase', 'selected_embs']:
        task[key] = task[key][best_idx]


def process_file(*, path, out_path, filter_config, author_embs_dict, tokenizer, model):
    with open(path, 'r') as f:
        tasks = [json.loads(line) for line in f]
    initial_samples = len(tasks)
    print('Initial samples:', initial_samples)

    for task in tasks:
        task['output'] = [out[0] for out in task['output']]

    print('Filtering based on x == x and max per')
    filter_to_max_results(tasks, filter_config['max_per_row'])
    num_remaining = filter_based_on_identity(tasks)
    print(f'\tRemaining samples: {num_remaining}/{initial_samples}\n')

    print('Filtering based on MIS score')
    num_remaining = filter_based_on_model_score(
        tasks,
        threshold=filter_config['mis_threshold'],
        model_fn=lambda refs, cands: mis_compute(refs, cands),
    )
    print(f'\tRemaining samples: {num_remaining}/{initial_samples}\n')

    print('Filtering based on Alternative MIS score')
    num_remaining = filter_based_on_model_score(
        tasks,
        threshold=filter_config['simcse_threshold'],
        model_fn=lambda refs, cands: mis_alternative_compute(refs, cands),
    )
    print(f'\tRemaining samples: {num_remaining}/{initial_samples}\n')

    print('Adding style features')
    add_style_features(tasks, author_embs_dict, model, tokenizer)
    get_style_transfer_stats(tasks)

    print('Filtering based on style transfer')
    num_remaining = filter_based_on_style(
        tasks=tasks,
        away_threshold=filter_config['away_threshold'],
        towards_threshold=filter_config['towards_threshold'],
    )
    print(f'\tRemaining samples: {num_remaining}/{initial_samples}')

    print('Filtering based on hallucinated links')
    num_remaining = filter_hallucinated_links(tasks, link_regex=LINK_REGEX)
    print(f'\tRemaining samples: {num_remaining}/{initial_samples}')

    with open(out_path, 'w') as f:
        for task in tasks:
            if any(task['output']):
                filter_to_best_result(task)
                f.write(json.dumps(task) + '\n')


def get_authorshipstyler_model(model_path, device, base_config, styler_config):
    tokenizer = AutoTokenizer.from_pretrained(base_config['model_name'], legacy=True)
    model = HyperStyler(base_config['model_name'], styler_config, use_style=True)
    model.load_state_dict(load_state_dict(model_path))
    model.to(device)
    return tokenizer, model


@click.command()
@click.option('--exp_dir', type=str, required=True, help='Experiment directory, e.g. outputs/blog/<experiment>.')
@click.option('--config', 'config_path', type=str, default=None, help='Configuration to copy on first use.')
@click.option('--worker_idx', type=int, default=None, hidden=True)
@click.option('--num_workers', type=int, default=None, hidden=True)
def main(exp_dir, config_path, worker_idx, num_workers):
    """Filter stage-2 generations on every GPU in CUDA_VISIBLE_DEVICES, then split and convert."""
    exp_dir = Path(exp_dir)
    in_dir = exp_dir / '2_generation'
    stage_dir = exp_dir / '3_filtering'
    filtered_dir = stage_dir / 'filtered'

    if worker_idx is None:
        config = init_experiment(exp_dir, config_path)
        if not in_dir.exists():
            raise SystemExit(f'{in_dir} not found. Run stage 2 first.')
        if not confirm_overwrite(stage_dir):
            return
        filtered_dir.mkdir(parents=True)
        with run_log(exp_dir, '3_filtering'):
            run_workers(__file__, exp_dir)
            filter_config = config['filtering']
            split_train_val(filtered_dir, stage_dir / 'split',
                            percent_val=filter_config['val_fraction'], seed=filter_config['split_seed'])
            convert_to_dataset(stage_dir / 'split', stage_dir / 'dataset')
        return

    config = load_experiment_config(exp_dir)
    model_path = resolve_checkpoint(config, exp_dir, 'hyperstyler')
    tokenizer, model = get_authorshipstyler_model(model_path, 'cuda', config['base_config'], config['styler_config'])
    author_embs_dict = AuthorEmbLMDB(config['paths']['author_embs_train'])

    transfer_results = sorted(str(p) for p in in_dir.glob('*.jsonl'))
    for path in tqdm(worker_share(transfer_results, worker_idx, num_workers)):
        process_file(
            path=path,
            out_path=filtered_dir / os.path.basename(path),
            filter_config=config['filtering'],
            author_embs_dict=author_embs_dict,
            tokenizer=tokenizer,
            model=model,
        )


if __name__ == '__main__':
    main()
