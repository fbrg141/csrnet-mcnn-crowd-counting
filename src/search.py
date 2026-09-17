"""Validation-only grid search; explicit confirmation and held-out test stages.

Examples: python -m src.search --config configs/mcnn_search.json --stage dry-run
Use one process per output directory. Checkpoints must be trusted local artifacts.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from src.config import SHANGHAITECH_DIR, DENSITY_CACHE_DIR
from src.runs import atomic_json, completed, digest, file_hash, source_hash, validate_hparams


def expand_grid(config):
    allowed = {'model', 'part', 'seed', 'epochs', 'batch_size', 'num_workers', 'grid',
               'baseline', 'confirmation_seeds', 'smoke'}
    if set(config) - allowed:
        raise ValueError(f'unknown config keys: {set(config) - allowed}')
    if config['model'] != 'mcnn' or config['part'] not in ('A', 'B'):
        raise ValueError('search supports MCNN, part A or B')
    for name in ('seed', 'epochs', 'batch_size', 'num_workers'):
        value = config[name]
        if type(value) is not int or value < (1 if name in ('epochs', 'batch_size') else 0):
            raise ValueError(f'invalid {name}')
    if config['seed'] >= 2**32:
        raise ValueError('seed must fit uint32')
    seeds = config.get('confirmation_seeds', [123, 2026])
    if len(set(seeds)) != len(seeds) or config['seed'] in seeds or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError('confirmation seeds must be unique uint32 and exclude search seed')
    grid = config['grid']
    keys = ('lr', 'momentum', 'weight_decay')
    if set(grid) != set(keys) or any(not isinstance(grid[k], list) or not grid[k] for k in keys):
        raise ValueError('grid needs nonempty lr, momentum, weight_decay lists')
    validate_hparams(**config['baseline'])
    trials = []
    for values in itertools.product(*(grid[k] for k in keys)):
        hp = dict(zip(keys, values))
        validate_hparams(**hp)
        trial = {k: config[k] for k in ('model', 'part', 'seed', 'epochs', 'batch_size', 'num_workers')}
        trial.update(hp, smoke=bool(config.get('smoke', False)))
        if trial['smoke']:
            trial['epochs'] = 1
        trial['trial_id'] = 'trial-' + digest(trial)[:16]
        trials.append(trial)
    if len({t['trial_id'] for t in trials}) != len(trials):
        raise ValueError('duplicate grid combinations')
    if not any(all(t[k] == config['baseline'][k] for k in keys) for t in trials):
        raise ValueError('baseline must be included in grid')
    return trials


def rank_trials(rows):
    if not rows or any(not math.isfinite(r['val_mae']) for r in rows):
        raise ValueError('ranking requires finite validation MAE for every trial')
    return sorted(rows, key=lambda r: (r['val_mae'], r['trial_id']))


def run_command(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as f:
        f.write('\nCOMMAND ' + json.dumps(command) + '\n')
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as p:
            for line in p.stdout:
                print(line, end='', flush=True)
                f.write(line)
                f.flush()
            code = p.wait()
    if code:
        raise RuntimeError(f'command failed ({code}); see {log}')


def train_trial(trial, directory, args):
    command = [sys.executable, '-u', '-m', 'src.train', '--resume', '--root', str(args.root),
               '--out-dir', str(directory), '--cache-dir', str(args.cache_dir), '--device', args.device]
    for key in ('model', 'part', 'seed', 'epochs', 'batch_size', 'num_workers', 'lr', 'momentum', 'weight_decay'):
        command += ['--' + key.replace('_', '-'), str(trial[key])]
    if trial.get('smoke'):
        command.append('--smoke')
    run_command(command, directory / 'train.log')
    summary = completed(directory)
    if summary is None:
        raise ValueError(f'incomplete trial {directory}')
    return summary


def lock_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'{path.name} changed; use a new output directory')
    else:
        atomic_json(path, value)


def selection_rows(trials, out):
    rows = []
    fixed = None
    for trial in trials:
        directory = out / trial['trial_id']
        summary = completed(directory)
        if summary is None:
            raise ValueError(f'all trials must complete before selection: {directory}')
        if any(summary['config'][k] != v for k, v in trial.items() if k != 'trial_id'):
            raise ValueError('trial config does not match completed run')
        current = {k: v for k, v in summary['config'].items()
                   if k not in ('lr', 'momentum', 'weight_decay')}
        if fixed is not None and current != fixed:
            raise ValueError('fixed training configuration differs across trials (including dataset/source)')
        fixed = current
        rows.append(dict(trial, val_mae=summary['val_mae'], val_rmse=summary['val_rmse'],
                         best_epoch=summary['best_epoch'], checkpoint=summary['checkpoint']))
    return rank_trials(rows)


def validate_final_run(summary, trial, reference):
    expected = dict(reference)
    for key in ('seed', 'lr', 'momentum', 'weight_decay'):
        if key in trial:
            expected[key] = trial[key]
    if summary['config'] != expected:
        raise ValueError('final run configuration does not match frozen selection and dataset')


def final_runs(selection, config, out, with_confirmation):
    runs = []
    for label in ('baseline', 'winner'):
        trial = selection[label]
        runs.append((label, trial, out / trial['trial_id']))
        if with_confirmation:
            for seed in config.get('confirmation_seeds', [123, 2026]):
                other = dict(trial, seed=seed)
                # Baseline==winner intentionally shares artifacts, never double-trains.
                directory = out / 'confirmation' / f"{trial['trial_id']}-seed{seed}"
                runs.append((label, other, directory))
    return runs


def write_reports(out):
    selection = json.loads((out / 'selection.json').read_text())
    rows = selection['ranking']
    with (out / 'search_results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    for row in rows:
        history = json.loads((out / row['trial_id'] / 'history.json').read_text())
        ax.plot([h['epoch'] for h in history], [h['val_mae'] for h in history],
                label=f"{row['trial_id']} lr={row['lr']:g} m={row['momentum']} wd={row['weight_decay']:g}")
    ax.set(xlabel='Epoch (fixed budget)', ylabel='Validation MAE', title='MCNN search (validation only)')
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / 'validation_curves.png', dpi=150)
    plt.close(fig)
    path = out / 'final_metrics.json'
    if path.exists():
        final = json.loads(path.read_text())
        with (out / 'test_results.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(final['runs'][0]))
            writer.writeheader()
            writer.writerows(final['runs'])
        atomic_json(out / 'aggregate.json', final['aggregate'])
    print(f'[report] {out}; test files only exist after explicit evaluation')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/mcnn_search.json')
    p.add_argument('--out-dir', default='reports/search/mcnn-A')
    p.add_argument('--root', default=str(SHANGHAITECH_DIR))
    p.add_argument('--cache-dir', default=str(DENSITY_CACHE_DIR))
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--stage', choices=['dry-run', 'baseline', 'search', 'confirm', 'evaluate', 'report'], default='dry-run')
    p.add_argument('--with-confirmation', action='store_true', help='evaluate search seed plus all configured confirmation seeds')
    p.add_argument('--smoke', action='store_true', help='synthetic one-epoch 64x64 trials, never real results')
    args = p.parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    if args.smoke:
        config['smoke'] = True
    trials = expand_grid(config)
    out = Path(args.out_dir)
    if args.stage == 'dry-run':
        print(json.dumps({'trials': trials, 'n_trials': len(trials),
                          'epoch_budget': sum(t['epochs'] for t in trials),
                          'confirmation_max_runs': 2 * len(config.get('confirmation_seeds', [123, 2026])),
                          'selection': 'best validation MAE only; tie: trial_id'}, indent=2))
        return
    out.mkdir(parents=True, exist_ok=True)
    protocol = dict(config=config, source_hash=source_hash(), root=str(Path(args.root).resolve()),
                    device=args.device)
    lock_json(out / 'protocol.json', protocol)
    if args.stage == 'baseline':
        trial = next(t for t in trials if all(t[k] == v for k, v in config['baseline'].items()))
        train_trial(trial, out / trial['trial_id'], args)
        return
    if args.stage == 'search':
        for trial in trials:
            train_trial(trial, out / trial['trial_id'], args)
        ranking = selection_rows(trials, out)
        baseline = next(t for t in ranking if all(t[k] == v for k, v in config['baseline'].items()))
        selection = dict(protocol_hash=digest(protocol), metric='best validation MAE',
                         tie_break='trial_id ascending', winner=ranking[0], baseline=baseline, ranking=ranking)
        lock_json(out / 'selection.json', selection)
        write_reports(out)
        return
    selection = json.loads((out / 'selection.json').read_text())
    if selection['protocol_hash'] != digest(protocol) or selection['ranking'] != selection_rows(trials, out):
        raise ValueError('frozen selection no longer matches validated trials')
    if args.stage == 'confirm':
        seen = set()
        for _, trial, directory in final_runs(selection, config, out, True):
            if directory not in seen:
                train_trial(trial, directory, args)
                seen.add(directory)
    elif args.stage == 'evaluate':
        runs = final_runs(selection, config, out, args.with_confirmation)
        reference = completed(out / selection['baseline']['trial_id'])['config']
        for _, trial, directory in runs:
            summary = completed(directory)
            if summary is None:
                raise ValueError('finish confirmation training before test evaluation')
            validate_final_run(summary, trial, reference)
        # Freeze seeds/labels/checkpoint hashes before the first test access.
        plan = {'selection_hash': file_hash(out / 'selection.json'),
                'runs': [{'label': label, 'seed': t['seed'], 'directory': str(d.relative_to(out)),
                          'checkpoint_hash': file_hash(d / completed(d)['checkpoint'])} for label, t, d in runs]}
        lock_json(out / 'evaluation_plan.json', plan)
        metrics = []
        evaluated = {}
        with tempfile.TemporaryDirectory() as temporary:
            root = args.root
            if config.get('smoke'):
                from src.evaluate import _make_fake_dataset
                root = _make_fake_dataset(temporary, config['part'], n=2)
            for label, trial, directory in runs:
                if directory not in evaluated:
                    summary = completed(directory)
                    dest = directory / 'test_metrics.json'
                    command = [sys.executable, '-u', '-m', 'src.evaluate', '--model', 'mcnn',
                               '--part', config['part'], '--ckpt', str(directory / summary['checkpoint']),
                               '--root', str(root), '--out', str(dest), '--device', args.device,
                               '--cache-dir', args.cache_dir]
                    if config.get('smoke'):
                        command.append('--no-cache')
                    run_command(command, directory / 'evaluate.log')
                    evaluated[directory] = json.loads(dest.read_text())
                result = evaluated[directory]
                metrics.append(dict(label=label, seed=trial['seed'], trial_id=trial['trial_id'],
                                    mae=result['mae'], rmse=result['rmse'], n_test=result['n_test'],
                                    checkpoint_hash=result['checkpoint_hash'], synthetic=bool(config.get('smoke'))))
        aggregate = {}
        for label in ('baseline', 'winner'):
            subset = [r for r in metrics if r['label'] == label]
            aggregate[label] = {'n': len(subset), 'seeds': [r['seed'] for r in subset]}
            for metric in ('mae', 'rmse'):
                values = [r[metric] for r in subset]
                aggregate[label][metric + '_mean'] = statistics.mean(values)
                aggregate[label][metric + '_sample_std'] = statistics.stdev(values) if len(values) > 1 else None
        atomic_json(out / 'final_metrics.json', dict(evaluation_plan=plan, runs=metrics, aggregate=aggregate,
                                                    synthetic=bool(config.get('smoke'))))
        write_reports(out)
    else:
        write_reports(out)


if __name__ == '__main__':
    main()
