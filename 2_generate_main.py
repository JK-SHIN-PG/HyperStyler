import os
import torch
import json
from tqdm.auto import tqdm
import random
from transformers import AutoTokenizer
import click
from pathlib import Path
from src.hyperstyler import HyperStyler
from utility.checkpoint import load_state_dict
from utility.utils import AuthorEmbLMDB
from utility.experiment import (
    confirm_overwrite, init_experiment, load_experiment_config, resolve_checkpoint, run_log, run_workers, worker_share,
)

def get_authorshipstyler_model(model_path, device, base_config, styler_config):

    tokenizer = AutoTokenizer.from_pretrained(base_config['model_name'] , legacy=True)
    model = HyperStyler(base_config['model_name'] , styler_config, use_style= True)

    model.load_state_dict(load_state_dict(model_path))
    model.to(device)

    return tokenizer, model

@torch.no_grad()
def run_styler_batch(
    model,
    tokenizer,
    source_texts,
    style_embs,
    style_masks,
    temperature,
    top_p,
    max_new_tokens=100,
    device="cpu",
    n_samples=1,
):

    inputs = tokenizer(
            source_texts, padding="longest", truncation=True, return_tensors="pt"
        ).to(device)
    

    generated_outputs = model.generate(
        input_ids = inputs['input_ids'],
        attention_mask = inputs['attention_mask'],
        style_embs=style_embs.to(device), # [B, max_num, 768]
        style_masks=style_masks.to(device), # [B, max_num]
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        num_return_sequences=n_samples,
    )
    batch_size = inputs['input_ids'].shape[0]

    if isinstance(generated_outputs, list):
        n_samples = len(generated_outputs)

        samples_by_batch = [
            tokenizer.batch_decode(generated_outputs[i], skip_special_tokens=True)
            for i in range(n_samples)
        ]
        
        results = [
            [samples_by_batch[s][b] for s in range(n_samples)]
            for b in range(batch_size)
        ]

    else:
        generated_texts = tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
        results = [
            generated_texts[i : i + n_samples]
            for i in range(0, len(generated_texts), n_samples)
        ]

    selected_embs = getattr(model.model, "_last_selected_embs", None)

    return results, selected_embs.detach().cpu().numpy()


def get_style_mask(target_embs_batch):
    
    padded_embs = []
    masks = []
    max_num = 0
    for embs in target_embs_batch:
        max_num = max(max_num, embs.shape[0])

    for emb in target_embs_batch:
        num_pads = max_num - emb.shape[0]
        if num_pads > 0:
            padding = torch.zeros(num_pads, emb.shape[1], dtype=torch.float)
            emb_padded = torch.cat([emb, padding], dim=0)
        else:
            emb_padded = emb

        mask = torch.cat(
            [torch.ones(emb.shape[0], dtype=torch.float),
             torch.zeros(num_pads, dtype=torch.float)]
        )

        padded_embs.append(emb_padded)
        masks.append(mask)

    return torch.stack(padded_embs), torch.stack(masks)

def generate_batch(model, tokenizer, author_embs_dict, device, dataset, out_path, batch_size, num_samples, temperature, top_p):

    with open(out_path, 'w') as file:
        for i in tqdm(range(0, len(dataset), batch_size)):
            data_list = dataset[i:i + batch_size]

            input_data = []
            target_embs_batch = []
            num_txts = len(data_list[0]['paraphrase'])
            for data in data_list:
                input_data += data['paraphrase']
                target_embs_batch.append(author_embs_dict[data['target_author_id']]) #

            style_embs, style_masks = get_style_mask(target_embs_batch)
            style_embs = style_embs.repeat_interleave(num_txts, dim=0)
            style_masks = style_masks.repeat_interleave(num_txts, dim=0)
 
            results, selected_embs = run_styler_batch(
                model = model,
                tokenizer = tokenizer,
                source_texts= input_data,
                style_embs=style_embs,
                style_masks=style_masks,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=80,
                device=device,
                n_samples=num_samples,
            )

            grouped_results = [
                results[i:i + num_txts]
                for i in range(0, len(results), num_txts)
            ]

            grouped_selected_embs = [
                selected_embs[i:i + num_txts]
                for i in range(0, len(selected_embs), num_txts)
            ]

            for i in range(len(data_list)):
                temp = data_list[i]
                temp['output'] = grouped_results[i]
                temp["selected_embs"] = grouped_selected_embs[i].tolist()
                file.write(json.dumps(temp) + "\n")

@click.command()
@click.option('--exp_dir', type=str, required=True, help='Experiment directory, e.g. outputs/blog/<experiment>.')
@click.option('--config', 'config_path', type=str, default=None, help='Configuration to copy on first use.')
@click.option('--worker_idx', type=int, default=None, hidden=True)
@click.option('--num_workers', type=int, default=None, hidden=True)
def main(exp_dir, config_path, worker_idx, num_workers):
    """Generate synthetic pairs on every GPU in CUDA_VISIBLE_DEVICES."""
    exp_dir = Path(exp_dir)
    out_dir = exp_dir / '2_generation'

    if worker_idx is None:
        init_experiment(exp_dir, config_path)
        if not confirm_overwrite(out_dir):
            return
        out_dir.mkdir(parents=True)
        with run_log(exp_dir, '2_generation'):
            run_workers(__file__, exp_dir)
        return

    config = load_experiment_config(exp_dir)
    gen_config = config['generation']
    model_path = resolve_checkpoint(config, exp_dir, 'hyperstyler')

    random.seed(42)
    torch.manual_seed(42)

    author_embs_dict = AuthorEmbLMDB(config['paths']['author_embs_train'])

    data_path = config['paths']['generation_input']
    paraphrased_files = sorted(os.path.join(data_path, x) for x in os.listdir(data_path))
    selected_files = worker_share(paraphrased_files, worker_idx, num_workers)
    print('Num files:', len(paraphrased_files))
    print('For worker:', worker_idx, 'Processing:', len(selected_files))

    device = 'cuda'
    tokenizer, model = get_authorshipstyler_model(model_path, device, config['base_config'], config['styler_config'])

    for paraphrase_file in tqdm(selected_files):
        print('Processing:', paraphrase_file)
        data = []
        with open(paraphrase_file, 'r') as f:
            for line in f:
                data.append(json.loads(line))

        generate_batch(
            model, tokenizer, author_embs_dict, device, data,
            out_dir / os.path.basename(paraphrase_file),
            batch_size=gen_config['batch_size'],
            num_samples=gen_config['num_samples'],
            temperature=gen_config['temperature'],
            top_p=gen_config['top_p'],
        )


if __name__ == '__main__':
    main()
