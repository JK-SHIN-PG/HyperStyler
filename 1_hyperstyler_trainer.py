from transformers import AutoTokenizer, get_scheduler
import os
import random
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import json
import time
from datasets import load_from_disk

from contextlib import nullcontext
from pathlib import Path

import wandb
import click
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list

from src.hyperstyler import HyperStyler
from utility.checkpoint import load_state_dict
from utility.experiment import confirm_overwrite, init_experiment, load_experiment_config, run_log
from utility.utils import count_parameters, log_and_print, validate_config, AuthorEmbLMDB
from src.dataloader import collator_for_training

def run_model(*, batch, model, device):

    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    style_embs = batch['style_embs'].to(device)
    style_masks = batch['style_masks'].to(device)

    idx_labels = None
    if 'idx_labels' in batch:
        idx_labels = batch['idx_labels'].to(device)

    result = model(
        input_ids=input_ids, attention_mask=attention_mask, 
        labels=labels, style_embs=style_embs, style_masks=style_masks,
        idx_labels = idx_labels,
    )
    loss = result.loss
    select_loss = result.select_loss
    select_acc = result.select_acc

    return loss, select_loss, select_acc


@click.command()
@click.option('--exp_dir', type=str, required=True, help='Experiment directory, e.g. outputs/blog/<experiment>.')
@click.option('--config', 'config_path', type=str, default=None, help='Configuration to copy on first use.')
@click.option('--wandb_project', type=str, default='HyperStyler', show_default=True)
def main(exp_dir, config_path, wandb_project):
    """Train HyperStyler reconstruction from the dataset's base model. Launch with accelerate."""
    exp_dir = Path(exp_dir)
    out_dir = exp_dir / '1_hyperstyler'

    accelerator = Accelerator()
    device = accelerator.device

    # The main process prepares the experiment and asks before overwriting.
    proceed = [None]
    if accelerator.is_main_process:
        init_experiment(exp_dir, config_path)
        proceed[0] = confirm_overwrite(out_dir)
        if proceed[0]:
            out_dir.mkdir(parents=True)
    broadcast_object_list(proceed)
    if not proceed[0]:
        return
    accelerator.wait_for_everyone()

    config = load_experiment_config(exp_dir)
    validate_config(config)

    seed = config['train_config']['seed']
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    with run_log(exp_dir, '1_hyperstyler') if accelerator.is_main_process else nullcontext():
        train(config, out_dir, accelerator, device, wandb_project)


def train(config, out_dir, accelerator, device, wandb_project):
    train_config = config['train_config']
    styler_config = config['styler_config']
    paths = config['paths']

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

    # Start from the dataset's base reconstruction model (stage 0).
    base_checkpoint = paths['base_checkpoint']
    if base_checkpoint is not None:
        saved_state_dict = load_state_dict(base_checkpoint)
        model.load_state_dict(saved_state_dict, strict=False)

    freeze_config = train_config['freeze']
    if 'all' in freeze_config:
        model.base_model_freeze()
    else:
        if 'encoder' in freeze_config:
            model.freeze_encoder_only()

    model.to(device)

    # prepare dataset

    dataset = load_from_disk(paths['dataset'])
    author_embs_dict = AuthorEmbLMDB(paths['author_embs_train'])
    author_embs_dict_val = AuthorEmbLMDB(paths['author_embs_val'])

    train_dataset = dataset['train']
    valid_dataset = dataset['val']

    collator_fn = collator_for_training


    collator_args = {
        'tokenizer': tokenizer,
        'max_length_src': config['base_config']['max_encoder_len'],
        'max_length_tgt': config['base_config']['max_decoder_len'],
        'ignore_idx': -100,
        'author_embs_dict' : author_embs_dict,
        'use_stylonavigator' : config['styler_config']['use_stylonavigator'],
    }
    collator_args_val = collator_args.copy()
    collator_args_val['author_embs_dict'] = author_embs_dict_val

    batch_size = train_config['batch_size']
    # DataLoaders creation:
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
        collate_fn=lambda x: collator_fn(batch=x, **collator_args_val),
        num_workers=4,
        pin_memory=True,
    )
    lr = train_config['learning_rate']
    lr = float(lr)

    if styler_config['use_hypernet']:
        hyper_optimizer = torch.optim.AdamW(params=model.hypernet.parameters(), lr=lr, weight_decay=0.01)
        hyper_scheduler = get_scheduler(
            name='cosine',
            optimizer=hyper_optimizer,
            num_warmup_steps = train_config['warmup_steps'],
            num_training_steps=train_config['max_steps'],
        )
        (
            hyper_scheduler,
            hyper_optimizer
        ) = accelerator.prepare(hyper_scheduler , hyper_optimizer)
        hyper_optimizer.zero_grad()
    else:
        proj_optimizer = torch.optim.AdamW(params=model.internal_controller.parameters(), lr=lr, weight_decay=0.01)
        proj_scheduler = get_scheduler(
            name = 'cosine',
            optimizer=proj_optimizer,
            num_warmup_steps = train_config['warmup_steps'],
            num_training_steps=train_config['max_steps'],
        )
        (
            proj_scheduler,
            proj_optimizer,
        ) = accelerator.prepare(proj_scheduler , proj_optimizer)
        proj_optimizer.zero_grad()

    if styler_config['use_stylonavigator']:
        navigator_optimizer = torch.optim.AdamW(params=model.style_navigator.parameters(), lr=lr, weight_decay=0.01)
        navigator_scheduler = get_scheduler(
            name = 'cosine',
            optimizer=navigator_optimizer,
            num_warmup_steps = train_config['warmup_steps'],
            num_training_steps=train_config['max_steps'],
        )
        (
            navigator_scheduler,
            navigator_optimizer,
        ) = accelerator.prepare(navigator_scheduler , navigator_optimizer)
        navigator_optimizer.zero_grad()

    model_optimizer = torch.optim.AdamW(params=model.model.parameters(), lr=lr, weight_decay=0.01)
    model_scheduler = get_scheduler(
        name = 'cosine',
        optimizer=model_optimizer,
        num_warmup_steps = train_config['warmup_steps'],
        num_training_steps=train_config['max_steps'],
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

    if accelerator.is_main_process:
        wandb.init(project=wandb_project, config=config)

    total_params, trainable_params = count_parameters(model)

    REPORT_FILE_PATH = os.path.join(out_dir, "report.txt")
    
    if accelerator.is_main_process:
        start_time = time.time()
        with open(REPORT_FILE_PATH, "w") as report:
            log_and_print("total_params : {}".format(total_params), report)
            log_and_print("trainable_params : {}".format(trainable_params), report)
            log_and_print("num_training_steps : {}".format(train_config['max_steps']), report)

    best_val_loss = None
    accumulation_steps = train_config['accumulation_steps']

    steps = 0
    counter = 0
    for epoch in range(train_config['max_num_epochs']):
        model.train()
        if accelerator.is_main_process:
            wandb.log({"epoch": epoch})

        with tqdm(total=len(train_dataloader)) as pbar:
            for i, data in enumerate(train_dataloader):

                loss, select_loss, select_acc = run_model(batch=data, model=model, device=device)
                train_logs = {"train_loss": loss.item()}
                
                if styler_config['use_stylonavigator']:
                    train_logs["train_select"] = select_loss.item()
                    train_logs["train_acc"] = select_acc.item()

                if accelerator.is_main_process:
                    wandb.log(train_logs)
                loss = loss / accumulation_steps
                accelerator.backward(loss)
                if styler_config['use_stylonavigator']:
                    select_loss = select_loss / accumulation_steps
                    accelerator.backward(select_loss)
                pbar.update(1)

                if (counter + 1) % accumulation_steps == 0:
                    if styler_config['use_hypernet']:
                        hyper_optimizer.step()
                        hyper_scheduler.step()
                        hyper_optimizer.zero_grad()
                    else:
                        proj_optimizer.step()
                        proj_scheduler.step()
                        proj_optimizer.zero_grad()
                    if styler_config['use_stylonavigator']:
                        navigator_optimizer.step()
                        navigator_scheduler.step()
                        navigator_optimizer.zero_grad()
                    model_optimizer.step()
                    model_scheduler.step()
                    model_optimizer.zero_grad()
                    
                    steps += 1

                if (counter + 1) % (train_config['eval_freq'] * accumulation_steps) == 0:
                    model.eval()
                    losses = []
                    select_losses = []
                    select_acces = []
                    with torch.no_grad():
                        for j, val_data in enumerate(eval_dataloader):
                            if j > train_config['max_val_batch']:
                                break
                            loss, select_loss, select_acc = run_model(batch=val_data, model=model, device=device)

                            losses.append(loss.item())
                            if styler_config['use_stylonavigator']:
                                select_losses.append(select_loss.item())
                                select_acces.append(select_acc.item())

                    val_loss = sum(losses) / len(losses)
                    val_logs = {"val_loss": val_loss}
                    if styler_config['use_stylonavigator']:
                        val_select = sum(select_losses) / len(select_losses)
                        val_acc = sum(select_acces) / len(select_acces)
                        val_logs['val_select'] = val_select
                        val_logs['val_select_acc'] = val_acc
                    
                    if accelerator.is_main_process:
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

                if steps >= train_config['max_steps']:
                    break

        if steps >= train_config['max_steps']:
            break

    if accelerator.is_main_process:
        wandb.finish()
        
    if accelerator.is_main_process:
        torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(out_dir, 'last_model.pt'))
        end_time = time.time()
        with open(REPORT_FILE_PATH, "a") as report:
            log_and_print(f"Training time: {end_time-start_time:.4f}", report)


if __name__ == '__main__':
    main()
