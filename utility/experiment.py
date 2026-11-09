"""Experiment directory helpers shared by the stage scripts.

An experiment directory (``outputs/<dataset>/<experiment>``) holds:

- ``config.yaml``: a copy of the configuration made on first use. Every later
  stage reads this copy, so all stages of an experiment share one setting.
- ``run_log.txt``: one line per stage run (start, end, stage, status, command).
- one fixed output directory per stage (``2_generation``, ``3_filtering``, ...).
"""

import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import yaml

CONFIG_NAME = 'config.yaml'
RUN_LOG_NAME = 'run_log.txt'


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def _flatten(d, prefix=''):
    items = {}
    for key, value in d.items():
        name = f'{prefix}{key}'
        if isinstance(value, dict):
            items.update(_flatten(value, f'{name}.'))
        else:
            items[name] = value
    return items


def _warn_config_diff(snapshot, source, snapshot_path, source_path):
    a, b = _flatten(snapshot), _flatten(source)
    changed = sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))
    if not changed:
        return
    print(f'WARNING: {source_path} differs from {snapshot_path}; using {snapshot_path}.')
    for key in changed:
        print(f'  {key}: {a.get(key)!r} (used) vs {b.get(key)!r} ({source_path})')
    print('  Start a new experiment directory to use the changed configuration.')


def init_experiment(exp_dir, config_path=None):
    """Return the experiment configuration, creating the copy on first use."""
    exp_dir = Path(exp_dir)
    snapshot_path = exp_dir / CONFIG_NAME
    if snapshot_path.exists():
        config = load_yaml(snapshot_path)
        if config_path is not None:
            _warn_config_diff(config, load_yaml(config_path), snapshot_path, config_path)
        return config
    if config_path is None:
        raise SystemExit(f'{snapshot_path} not found. Pass --config to start this experiment.')
    exp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, snapshot_path)
    print(f'Copied {config_path} to {snapshot_path}')
    return load_yaml(snapshot_path)


def load_experiment_config(exp_dir):
    """Load the copied configuration (for worker processes)."""
    return load_yaml(Path(exp_dir) / CONFIG_NAME)


def confirm_overwrite(paths):
    """Ask before replacing existing outputs. Returns True if it is safe to write.

    Existing paths are removed when the user answers 'y'. Without an
    interactive terminal, nothing is removed and False is returned.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    existing = [Path(p) for p in paths if Path(p).exists()]
    if not existing:
        return True
    for path in existing:
        print(f'File exists: {path}')
    if not sys.stdin.isatty():
        print('No interactive terminal; not overwriting. Remove the outputs above or run interactively.')
        return False
    answer = input('Overwrite? [y/N]: ').strip().lower()
    if answer != 'y':
        print('Not overwriting.')
        return False
    for path in existing:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    return True


def resolve_checkpoint(config, exp_dir, model):
    """Checkpoint of 'hyperstyler' (stage 1) or 'self_distillation' (stage 4)."""
    exp_dir = Path(exp_dir)
    if model == 'hyperstyler':
        path = config.get('paths', {}).get('hyperstyler_checkpoint')
        return Path(path) if path else exp_dir / '1_hyperstyler' / 'best_model.pt'
    if model == 'self_distillation':
        return exp_dir / '4_self_distillation' / 'best_model.pt'
    raise ValueError(f'Unknown model: {model}')


@contextmanager
def run_log(exp_dir, stage):
    """Append the outcome of a stage run to run_log.txt."""
    start = datetime.now()
    status = 'failed'
    try:
        yield
        status = 'done'
    finally:
        end = datetime.now()
        command = ' '.join([os.path.basename(sys.executable)] + sys.argv)
        gpus = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        line = (
            f'{start:%Y-%m-%d %H:%M:%S}\t{end:%Y-%m-%d %H:%M:%S}\t{stage}\t{status}\t'
            f'CUDA_VISIBLE_DEVICES={gpus}\t{command}\n'
        )
        with open(Path(exp_dir) / RUN_LOG_NAME, 'a') as f:
            f.write(line)


def run_workers(script, exp_dir):
    """Run ``script`` once per visible GPU, each on a disjoint share of the inputs."""
    gpus = [g for g in os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',') if g.strip()]
    procs = []
    for idx, gpu in enumerate(gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu.strip())
        cmd = [sys.executable, script, '--exp_dir', str(exp_dir),
               '--worker_idx', str(idx), '--num_workers', str(len(gpus))]
        procs.append(subprocess.Popen(cmd, env=env))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise RuntimeError(f'Worker exit codes: {codes}')


def worker_share(items, worker_idx, num_workers):
    """Contiguous share of ``items`` for one worker (ceil split)."""
    per_worker = -(-len(items) // num_workers)
    return items[worker_idx * per_worker:(worker_idx + 1) * per_worker]
