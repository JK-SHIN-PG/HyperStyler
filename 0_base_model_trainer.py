"""Stage 0: train the dataset's base reconstruction model (shared by experiments).

Writes to the directory of ``paths.base_checkpoint`` (e.g. outputs/blog/0_base/),
together with a copy of the configuration and run_log.txt.
"""

import json
import os
import random
import shutil
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

from src.dataloader import collator_cycle_reconstruction
from src.hyperstyler import HyperStyler
from utility.experiment import CONFIG_NAME, confirm_overwrite, load_yaml, run_log


def run_model(*, batch, model, device):

    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    result = model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels
    )
    loss = result.loss

    return loss

@click.command()
@click.option('--config', 'config_path', type=str, required=True, help='Dataset configuration, e.g. configs/blog.yaml.')
def main(config_path):
    config = load_yaml(config_path)
    out_dir = Path(config['paths']['base_checkpoint']).parent

    accelerator = Accelerator()
    device = accelerator.device

    # The main process asks before overwriting and copies the configuration.
    proceed = [None]
    if accelerator.is_main_process:
        proceed[0] = confirm_overwrite(out_dir)
        if proceed[0]:
            out_dir.mkdir(parents=True)
            shutil.copyfile(config_path, out_dir / CONFIG_NAME)
    broadcast_object_list(proceed)
    if not proceed[0]:
        return
    accelerator.wait_for_everyone()

    base_config = config['base_config']
    bt = config['base_training']
    seed = bt['seed']
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    with run_log(out_dir, '0_base') if accelerator.is_main_process else nullcontext():
        train(
            accelerator=accelerator, device=device, out_dir=out_dir, config=config,
            model_name=base_config['model_name'],
            max_encoder_len=base_config['max_encoder_len'],
            max_decoder_len=base_config['max_decoder_len'],
            data_file_path=config['paths']['dataset'],
            learning_rate=float(bt['learning_rate']),
            batch_size=bt['batch_size'],
            accumulation_steps=bt['accumulation_steps'],
            warmup_steps=bt['warmup_steps'],
            max_steps=bt['max_steps'],
            max_num_epochs=bt['max_num_epochs'],
            eval_freq=bt['eval_freq'],
            max_val_batch=bt['max_val_batch'],
        )


def train(*, accelerator, device, out_dir, config, model_name, max_encoder_len, max_decoder_len, data_file_path,
          learning_rate, batch_size, accumulation_steps, warmup_steps, max_steps, max_num_epochs, eval_freq,
          max_val_batch):
    # prepare tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="right",
        use_fast=True,  # Fast tokenizer giving issues.
        trust_remote_code=True,
    )

    # prepare model
    model = HyperStyler(base_model=model_name)
    model.to(device)

    # prepare dataset
    dataset = load_from_disk(data_file_path)
    train_dataset = dataset['train']
    valid_dataset = dataset['val']

    collator_fn = collator_cycle_reconstruction

    collator_args = {
        'tokenizer': tokenizer,
        'max_length_src': max_encoder_len,
        'max_length_tgt': max_decoder_len,
        'ignore_idx': -100,
    }

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
        collate_fn=lambda x: collator_fn(batch=x, **collator_args),
        num_workers=4,
        pin_memory=True,
    )

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    scheduler = get_scheduler(name='constant', optimizer=optimizer, num_warmup_steps=warmup_steps)

    (
        model,
        optimizer,
        train_dataloader,
        scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader,  scheduler
    )

    if accelerator.is_main_process:
        wandb.init(project='base_model_training', config=config)

    best_val_loss = None
    optimizer.zero_grad()
    steps = 0
    counter = 0
    for epoch in range(max_num_epochs):
        model.train()
        if accelerator.is_main_process:
            wandb.log({"epoch": epoch})

        with tqdm(total=len(train_dataloader)) as pbar:
            for i, data in enumerate(train_dataloader):

                loss = run_model(batch=data, model=model, device=device)

                if accelerator.is_main_process:
                    wandb.log({"train_loss": loss.item()})
                loss = loss / accumulation_steps

                accelerator.backward(loss)
                pbar.update(1)

                if (counter + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    steps += 1

                if (counter + 1) % (eval_freq * accumulation_steps) == 0:
                    model.eval()
                    losses = []
                    with torch.no_grad():
                        for j, val_data in enumerate(eval_dataloader):
                            if j > max_val_batch:
                                break
                            loss = run_model(batch=val_data, model=model, device=device)

                            losses.append(loss.item())
                    val_loss = sum(losses) / len(losses)
                    if accelerator.is_main_process:
                        wandb.log({"val_loss": val_loss})

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
                            # save optimizer state, save scheduler state
                            torch.save(
                                optimizer.state_dict(),
                                os.path.join(out_dir, 'optimizer.pt'),
                            )
                            torch.save(
                                scheduler.state_dict(),
                                os.path.join(out_dir, 'scheduler.pt'),
                            )
                    model.train()

                counter += 1

                if steps >= max_steps:
                    break

        if steps >= max_steps:
            break

    if accelerator.is_main_process:
        torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(out_dir, 'last_model.pt'))
        wandb.finish()


if __name__ == '__main__':
    main()
