"""Real small CPU integration checks: no mocked optimizer/model/metrics."""
import json

import pytest
import torch

from src import train
from src.runs import completed


def test_epoch_resume_matches_uninterrupted_and_rejects_changes(tmp_path, monkeypatch):
    monkeypatch.setattr('src.datasets.dataset.DEFAULT_IMAGE_SIZE', (32, 32))
    torch.set_num_threads(2)
    root = train._make_fake_dataset(tmp_path / 'data', n=5)
    args = ['--root', root, '--epochs', '2', '--batch-size', '2', '--device', 'cpu', '--no-cache']
    uninterrupted, resumed = tmp_path / 'whole', tmp_path / 'resumed'
    train.main(args + ['--out-dir', str(uninterrupted)])
    original = train.train_one_epoch
    calls = 0

    def interrupt(*a, **kw):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('simulated disconnect')
        return original(*a, **kw)

    monkeypatch.setattr(train, 'train_one_epoch', interrupt)
    with pytest.raises(RuntimeError, match='disconnect'):
        train.main(args + ['--out-dir', str(resumed)])
    assert not (resumed / 'complete.json').exists()
    monkeypatch.setattr(train, 'train_one_epoch', original)
    train.main(args + ['--out-dir', str(resumed), '--resume'])
    assert json.loads((resumed / 'history.json').read_text()) == json.loads((uninterrupted / 'history.json').read_text())
    left = torch.load(resumed / 'last.pth', weights_only=False)
    right = torch.load(uninterrupted / 'last.pth', weights_only=False)
    assert all(torch.equal(left['state_dict'][k], v) for k, v in right['state_dict'].items())
    assert left['optimizer']['param_groups'] == right['optimizer']['param_groups']
    before = (resumed / 'last.pth').stat().st_mtime_ns
    train.main(args + ['--out-dir', str(resumed), '--resume'])
    assert (resumed / 'last.pth').stat().st_mtime_ns == before
    with pytest.raises(ValueError, match='configuration changed'):
        train.main(args + ['--out-dir', str(resumed), '--resume', '--momentum', '.9'])
    (resumed / 'history.json').write_text('[]')
    with pytest.raises(ValueError, match='artifact mismatch'):
        completed(resumed)


def test_invalid_empty_and_nonfinite_evaluation():
    with pytest.raises(ValueError, match='empty'):
        train.evaluate(torch.nn.Identity(), [], torch.device('cpu'))
    batch = [(torch.full((1, 1, 2, 2), float('nan')), torch.zeros(1, 1, 2, 2))]
    with pytest.raises(ValueError, match='non-finite'):
        train.evaluate(torch.nn.Identity(), batch, torch.device('cpu'))


def test_csrnet_offline_factory():
    from src.models import build_model
    model = build_model('csrnet', pretrained=False)
    assert model.output_stride == 8
