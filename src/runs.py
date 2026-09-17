"""Small, local-only experiment persistence helpers. Load only trusted checkpoints."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import torch


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def source_hash() -> str:
    root = Path(__file__).resolve().parent.parent
    paths = sorted((root / 'src').rglob('*.py')) + [root / 'uv.lock']
    return digest({str(p.relative_to(root)): file_hash(p) for p in paths})


def _atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_json(path, value):
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    _atomic(path, lambda f: f.write(data))


def atomic_checkpoint(path, value):
    _atomic(path, lambda f: torch.save(value, f))


def validate_hparams(lr, momentum, weight_decay):
    if not all(math.isfinite(x) for x in (lr, momentum, weight_decay)):
        raise ValueError('hyperparameters must be finite')
    if lr <= 0 or not 0 <= momentum < 1 or weight_decay < 0:
        raise ValueError('require lr > 0, 0 <= momentum < 1, weight_decay >= 0')


def completed(directory, config=None):
    """Fail closed: completion is valid only with matching artifact hashes/config."""
    directory = Path(directory)
    marker = directory / 'complete.json'
    if not marker.exists():
        return None
    try:
        manifest = json.loads(marker.read_text())
        summary = json.loads((directory / 'summary.json').read_text())
        stored = json.loads((directory / 'config.json').read_text())
        if config is not None and stored != config:
            raise ValueError('completed run configuration changed; use a new output directory')
        if manifest['config_hash'] != digest(stored) or summary['config'] != stored:
            raise ValueError('completion config mismatch')
        for name in ('config.json', 'history.json', 'summary.json', summary['checkpoint'], 'last.pth'):
            if Path(name).name != name or file_hash(directory / name) != manifest['files'][name]:
                raise ValueError(f'completion artifact mismatch: {name}')
        if not math.isfinite(summary['val_mae']) or summary['epochs_completed'] != stored['epochs']:
            raise ValueError('invalid completed metrics/budget')
        history = json.loads((directory / 'history.json').read_text())
        if len(history) != stored['epochs']:
            raise ValueError('incomplete history')
        return summary
    except (KeyError, OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'invalid completion manifest in {directory}') from exc


def mark_complete(directory, summary):
    directory = Path(directory)
    names = ['config.json', 'history.json', 'summary.json', summary['checkpoint'], 'last.pth']
    atomic_json(directory / 'complete.json', {
        'config_hash': digest(summary['config']),
        'files': {name: file_hash(directory / name) for name in names},
    })
