import json
import math

import pytest

from src.search import expand_grid, rank_trials
from src.runs import atomic_json, validate_hparams


def test_default_grid_baseline_and_unique_ids():
    config = json.load(open('configs/mcnn_search.json'))
    trials = expand_grid(config)
    assert len(trials) == len({t['trial_id'] for t in trials}) == 12
    assert sum(t['lr'] == 1e-6 and t['momentum'] == .95 and t['weight_decay'] == 0 for t in trials) == 1
    assert sorted({t['lr'] for t in trials}) == [1e-6, 3e-6, 1e-5]
    changed = dict(config, epochs=51)
    assert expand_grid(changed)[0]['trial_id'] != trials[0]['trial_id']


def test_selection_rejects_mixed_datasets(monkeypatch, tmp_path):
    from src import search
    trials = expand_grid(json.load(open('configs/mcnn_search.json')))[:2]
    summaries = {}
    for i, trial in enumerate(trials):
        config = {k: v for k, v in trial.items() if k != 'trial_id'}
        config['dataset_hash'] = str(i)
        summaries[trial['trial_id']] = dict(config=config, val_mae=1., val_rmse=2.,
                                          best_epoch=1, checkpoint='best.pth')
    monkeypatch.setattr(search, 'completed', lambda d: summaries[d.name])
    with pytest.raises(ValueError, match='fixed training configuration'):
        search.selection_rows(trials, tmp_path)


def test_confirmation_rejects_wrong_seed_or_dataset():
    from src.search import validate_final_run
    reference = {'seed': 42, 'dataset_hash': 'a', 'lr': 1e-6}
    trial = {'seed': 123, 'lr': 1e-6}
    for wrong in ({'seed': 42, 'dataset_hash': 'a', 'lr': 1e-6},
                  {'seed': 123, 'dataset_hash': 'b', 'lr': 1e-6}):
        with pytest.raises(ValueError, match='final run configuration'):
            validate_final_run({'config': wrong}, trial, reference)
    validate_final_run({'config': dict(reference, seed=123)}, trial, reference)


def test_ranking_validation_only():
    rows = [{'trial_id': 'b', 'val_mae': 2., 'test_mae': 0.},
            {'trial_id': 'a', 'val_mae': 1., 'test_mae': 1000.}]
    assert rank_trials(rows)[0]['trial_id'] == 'a'
    with pytest.raises(ValueError):
        rank_trials([{'trial_id': 'x', 'val_mae': math.nan}])


@pytest.mark.parametrize('kwargs', [{'lr': 0}, {'lr': float('nan')}, {'momentum': 1.1}, {'weight_decay': -1}])
def test_invalid_hparams(kwargs):
    with pytest.raises(ValueError):
        validate_hparams(**dict({'lr': 1e-6, 'momentum': .95, 'weight_decay': 0}, **kwargs))


def test_atomic_json_refuses_nonfinite(tmp_path):
    p = tmp_path / 'result.json'
    atomic_json(p, {'ok': 1})
    with pytest.raises(ValueError):
        atomic_json(p, {'bad': float('nan')})
    assert json.loads(p.read_text()) == {'ok': 1}
