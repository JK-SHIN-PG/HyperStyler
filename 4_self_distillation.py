"""Stage 4: self-distillation on the filtered synthetic pairs."""

import json
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import click
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_scheduler

from src.dataloader import collator_for_self_distillation
from src.hyperstyler import HyperStyler
from utility.checkpoint import load_state_dict
from utility.experiment import confirm_overwrite, init_experiment, load_experiment_config, resolve_checkpoint, run_log
from utility.utils import AuthorEmbLMDB, count_parameters, log_and_print, validate_config


def run_model(*, batch, model, device):

    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    style_embs = batch['style_embs'].to(device)
    style_masks = batch['style_masks'].to(device)
    
    idx_labels = None
    target_style_embs = None
    if 'idx_labels' in batch:
        idx_labels = batch['idx_labels'].to(device)

    if 'target_style_embs' in batch:
        target_style_embs = batch['target_style_embs'].to(device)

    result = model(
        input_ids=input_ids, attention_mask=attention_mask, 
        labels=labels, style_embs=style_embs,  style_masks=style_masks,
        idx_labels = idx_labels, target_style_embs= target_style_embs,
    )
    loss = result.loss
    sft_loss = result.sft_loss
    select_loss = result.select_loss

    return loss, sft_loss, select_loss


@click.command()
@click.option('--exp_dir', type=str, required=True, help='Experiment directory, e.g. outputs/blog/<experiment>.')
@click.option('--config', 'config_path', type=str, default=None, help='Configuration to copy on first use.')
def main(exp_dir, config_path):
    """Self-distill the stage-1 HyperStyler on the stage-3 dataset. Launch with accelerate."""
    exp_dir = Path(exp_dir)
    out_dir = exp_dir / '4_self_distillation'
    data_file_path = exp_dir / '3_filtering' / 'dataset'

    accelerator = Accelerator()
    device = accelerator.device

    # The main process prepares the experiment and asks before overwriting.
    proceed = [None]
    if accelerator.is_main_process:
        init_experiment(exp_dir, config_path)
        proceed[0] = data_file_path.exists() and confirm_overwrite(out_dir)
        if not data_file_path.exists():
            print(f'{data_file_path} not found. Run stage 3 first.')
        if proceed[0]:
            out_dir.mkdir(parents=True)
    broadcast_object_list(proceed)
    if not proceed[0]:
        return
    accelerator.wait_for_everyone()

    config = load_experiment_config(exp_dir)
    validate_config(config)
    sd_config = config['self_distillation']
    styler_config = config['styler_config']
    model_path = resolve_checkpoint(config, exp_dir, 'hyperstyler')

    seed = sd_config['seed']
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    with run_log(exp_dir, '4_self_distillation') if accelerator.is_main_process else nullcontext():
        train(config, sd_config, styler_config, model_path, data_file_path, out_dir, accelerator, device)


def train(config, sd_config, styler_config, model_path, data_file_path, out_dir, accelerator, device):
    lr = float(sd_config['lr'])
    batch_size = sd_config['batch_size']
    accumulation_steps = sd_config['accumulation_steps']
    max_steps = sd_config['max_steps']

    model_name = config['base_config']['model_name']
    # prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="right",
        use_fast=True,  # Fast tokenizer giving issues.
        trust_remote_code=True,
    )

    # prepare model
    model = HyperStyler(base_model=model_name, use_style=True, config=styler_config)
    
    saved_state_dict = load_state_dict(model_path)
    model.load_state_dict(saved_state_dict)

    if sd_config['freeze_base_model']:
        model.base_model_freeze()
    if sd_config['freeze_stylonavigator']:
        model.freeze_stylonavigator()

    model.to(device)

    # prepare dataset

    dataset = load_from_disk(str(data_file_path))
    # The synthetic validation split targets training authors, so both splits
    # use the training LMDB.
    author_embs_dict = AuthorEmbLMDB(config['paths']['author_embs_train'])

    train_dataset = dataset['train']
    valid_dataset = dataset['val']

    collator_fn = collator_for_self_distillation
    collator_args = {
        'tokenizer': tokenizer,
        'max_length_src': config['base_config']['max_encoder_len'],
        'max_length_tgt': config['base_config']['max_decoder_len'],
        'ignore_idx': -100,
        'author_embs_dict' : author_embs_dict,
        'use_stylonavigator' : config['styler_config']['use_stylonavigator'],
    }

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=lambda x: collator_fn(batch=x, **collator_args),
        num_workers=4,
        pin_memory=True,
    )

    eval_dataloader = DataLoader(
        valid_dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=lambda x: collator_fn(batch=x, **collator_args),
        num_workers=4,
        pin_memory=True,
    )
    # Warmup length assumes two processes, as in the original runs.
    num_training_steps = (sd_config['max_num_epochs'] * len(train_dataloader)) // 2


    model_optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=0.01)
    model_scheduler = get_scheduler(
        name = 'constant',
        optimizer=model_optimizer,
        num_warmup_steps=int(0.05 * num_training_steps),
        num_training_steps=max_steps,
    )
    (
        model,
        model_optimizer,
        train_dataloader,
        model_scheduler,
    ) = accelerator.prepare(
        model, model_optimizer, train_dataloader, model_scheduler
    )
    model_optimizer.zero_grad()
    wandb.init(project='HyperStyler_self_distillation', config=config)

    total_params, trainable_params = count_parameters(model)

    REPORT_FILE_PATH = os.path.join(out_dir, "report.txt")
    
    if accelerator.is_main_process:
        start_time = time.time()
        with open(REPORT_FILE_PATH, "w") as report:
            log_and_print("total_params : {}".format(total_params), report)
            log_and_print("trainable_params : {}".format(trainable_params), report)
            log_and_print("num_training_steps : {}".format(max_steps), report)

    best_val_loss = None

    steps = 0
    counter = 0
    for epoch in range(sd_config['max_num_epochs']):
        model.train()
        wandb.log({"epoch": epoch})

        with tqdm(total=len(train_dataloader)) as pbar:
            for i, data in enumerate(train_dataloader):

                loss, sft_loss, select_loss = run_model(batch=data, model=model, device=device)
                train_logs = {"train_loss": loss.item()}
                train_logs['sft_loss'] = sft_loss.item()
                train_logs["train_select"] = select_loss

                wandb.log(train_logs)
                loss = loss / accumulation_steps
                accelerator.backward(loss)
                pbar.update(1)
                if (counter + 1) % accumulation_steps == 0:
                    model_optimizer.step()
                    model_scheduler.step()
                    model_optimizer.zero_grad()
                    steps += 1

                if (counter + 1) % (sd_config['eval_freq'] * accumulation_steps) == 0:
                    model.eval()
                    losses = []
                    sft_losses = []
                    with torch.no_grad():
                        for j, val_data in enumerate(eval_dataloader):
                            if j > sd_config['max_val_batch']:
                                break
                            loss, sft_loss, select_loss = run_model(batch=val_data, model=model, device=device)

                            losses.append(loss.item())
                            sft_losses.append(sft_loss.item())

                    val_loss = sum(losses) / len(losses)
                    val_logs = {"val_loss": val_loss}
                    val_sft = sum(sft_losses) / len(sft_losses)
                    val_logs['val_sft'] = val_sft
                    
                    wandb.log(val_logs)

                    if accelerator.is_main_process:
                        if best_val_loss is None or val_loss < best_val_loss:
                            best_val_loss = val_loss
                            print(epoch, i, 'New best val loss: ', best_val_loss)
                            with open(
                                os.path.join(out_dir, 'checkpoint_info.json'), 'w+'
                            ) as out_:
                                json.dump(
                                    {
                                        'epoch': epoch,
                                        'i': i,
                                        'counter': counter,
                                        'steps': steps,
                                        'loss': best_val_loss,
                                    },
                                    out_,
                                )

                            torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(out_dir, 'best_model.pt'))

                    model.train()

                counter += 1

                if steps >= max_steps:
                    break

    wandb.finish()
        
    if accelerator.is_main_process:
        torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(out_dir, 'last_model.pt'))
        end_time = time.time()
        with open(REPORT_FILE_PATH, "a") as report:
            log_and_print(f"Training time: {end_time-start_time:.4f}", report)


if __name__ == '__main__':
    main()
