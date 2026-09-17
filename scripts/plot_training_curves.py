#!/usr/bin/env python3
"""Verify original lightweight evidence and regenerate all final figures/tables.
Run from any directory; no dataset, checkpoint or GPU required.
"""
from pathlib import Path
import hashlib
import json
import statistics
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'reports/evidence'

def load(path):
    return json.loads(path.read_text())

def derive():
    manifest = load(EVIDENCE / 'manifest.json')
    for name, digest in manifest['files'].items():
        assert hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest() == digest, name
    search = EVIDENCE / 'mcnn/search'
    selection = load(search / 'selection.json')
    trials = sorted(search.glob('trial-*/summary.json'))
    confirmations = sorted((search / 'confirmation').glob('*/summary.json'))
    assert len(trials) == 12 and len(confirmations) == 4
    for path in trials + confirmations + [EVIDENCE / 'csrnet/summary.json']:
        s = load(path)
        h = load(path.with_name('history.json'))
        c = load(path.with_name('config.json'))
        assert s['config'] == c
        assert len(h) == s['epochs_completed'] == c['epochs'] == 50
        assert [row['epoch'] for row in h] == list(range(1, 51))
        best = min(h, key=lambda row: row['val_mae'])
        assert best['epoch'] == s['best_epoch'] and best['val_mae'] == s['val_mae']
        assert c['part'] == 'A' and c['cache_version'] == 2 and not c['smoke']
        assert (c['n_train'], c['n_val'], c['density_mode']) == (270, 30, 'fixed')
        for name, digest in load(path.with_name('complete.json'))['files'].items():
            if name.endswith('.json'):
                assert hashlib.sha256(path.with_name(name).read_bytes()).hexdigest() == digest
    ranking = sorted(selection['ranking'], key=lambda x: (x['val_mae'], x['trial_id']))
    assert selection['winner'] == ranking[0]
    assert all(row['best_epoch'] == 50 for row in ranking)
    for row in ranking:
        s = load(search / row['trial_id'] / 'summary.json')
        assert row['val_mae'] == s['val_mae']
    final = load(search / 'final_metrics.json')
    assert len(final['runs']) == 6
    for row, plan in zip(final['runs'], final['evaluation_plan']['runs']):
        original = load(search / plan['directory'] / 'test_metrics.json')
        for key in ('mae', 'rmse', 'seed', 'n_test', 'checkpoint_hash'):
            assert original[key] == row[key]
        assert row['n_test'] == 182 and not row['synthetic']
    aggregate = {}
    for label in ('baseline', 'winner'):
        rows = [row for row in final['runs'] if row['label'] == label]
        assert sorted(row['seed'] for row in rows) == [42, 123, 2026]
        aggregate[label] = {f'{metric}_{stat}': fn([row[metric] for row in rows])
                            for metric in ('mae', 'rmse')
                            for stat, fn in [('mean', statistics.mean), ('sample_std', statistics.stdev)]}
        for key, value in aggregate[label].items():
            assert math.isclose(value, final['aggregate'][label][key], rel_tol=1e-12)
    seed42 = [dict(row) for row in final['runs'] if row['seed'] == 42]
    csr = load(EVIDENCE / 'csrnet/test_metrics.json')
    assert csr['n_test'] == 182 and csr['seed'] == 42 and not csr['synthetic']
    assert csr['checkpoint_hash'] == load(EVIDENCE / 'csrnet/complete.json')['files']['csrnet_partA_seed42_best.pth']
    seed42.append(dict(csr, label='CSRNet'))
    return {'seed42': seed42, 'aggregate': aggregate, 'ranking': ranking,
            'weight_decay_delta': ranking[1]['val_mae'] - ranking[0]['val_mae'],
            'csrnet_reduction_percent': {m: 100 * (1 - csr[m] / seed42[1][m]) for m in ('mae','rmse')}}

def main():
    result = derive()
    out = ROOT / 'reports/figures'
    out.mkdir(exist_ok=True)
    (ROOT / 'reports/results_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    plt.rcParams.update({'font.size': 11})
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / (name + '.png'), dpi=180, bbox_inches='tight')
        plt.close(fig)
    labels = ['MCNN početni', 'MCNN podešeni', 'CSRNet']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ('mae', 'rmse')):
        values = [row[metric] for row in result['seed42']]
        bars = ax.bar(labels, values, color=['#8b9bb4','#3274a1','#d78034'])
        ax.bar_label(bars, fmt='%.2f', padding=3)
        ax.set(title=f'Test {metric.upper()} — seed 42', ylabel='Broj osoba')
        ax.set_ylim(0, max(values)*1.18)
    save(fig, 'test_comparison')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    search = EVIDENCE / 'mcnn/search'
    paths = [search / result['seed42'][i]['trial_id'] / 'history.json' for i in (0, 1)]
    paths.append(EVIDENCE / 'csrnet/history.json')
    for label, path in zip(labels, paths):
        history = load(path)
        axes[0].plot([r['epoch'] for r in history], [r['val_mae'] for r in history], label=label)
    axes[0].set(yscale='log', title='Validacioni MAE (log skala)', xlabel='Epoha', ylabel='MAE')
    h = load(paths[-1]); best = min(h, key=lambda r:r['val_mae'])
    for key, label in [('train_mae','Trening'),('val_mae','Validacija')]:
        axes[1].plot([r['epoch'] for r in h],[r[key] for r in h],label=label)
    axes[1].axvline(best['epoch'],ls='--',color='gray',label=f"Najbolja epoha: {best['epoch']}")
    axes[1].set(title='CSRNet — originalna istorija',xlabel='Epoha',ylabel='MAE')
    for ax in axes: ax.legend(); ax.grid(alpha=.25)
    save(fig, 'training_curves')
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ranking = result['ranking']
    bars=ax.bar(range(12),[r['val_mae'] for r in ranking],color='#3274a1')
    ax.bar_label(bars,fmt='%.2f',fontsize=8,padding=2)
    ax.set_xticks(range(12),[f"{r['lr']:g}\n{r['momentum']:g}\n{r['weight_decay']:g}" for r in ranking],fontsize=9)
    ax.set(xlabel='Stopa učenja / momentum / weight decay',ylabel='Najbolji validacioni MAE',title='MCNN: svih 12 konfiguracija, seed 42, 50 epoha',ylim=(0,4000))
    save(fig,'search_results')
    fig, axes=plt.subplots(1,2,figsize=(12,4))
    final=load(search/'final_metrics.json')['runs']
    for ax,metric in zip(axes,('mae','rmse')):
        for x,label in enumerate(('baseline','winner')):
            a=result['aggregate'][label]
            ax.errorbar(x,a[metric+'_mean'],yerr=a[metric+'_sample_std'],fmt='o',capsize=8,color='#3274a1')
            rows=[r for r in final if r['label']==label]
            for offset,row in zip((-.08,0,.08),rows):
                ax.scatter(x+offset,row[metric],s=28,color='#d78034')
                ax.annotate(str(row['seed']),(x+offset,row[metric]),xytext=(5,3),textcoords='offset points',fontsize=8)
        ax.set(xticks=[0,1],xticklabels=labels[:2],xlim=(-.4,1.5),ylabel=metric.upper(),title='MCNN: srednja vrednost ± uzoračka SD (n=3)')
        ax.grid(axis='y',alpha=.25)
    save(fig,'seed_variability')
    print(json.dumps({'verified_mcnn_trials':12,'verified_confirmations':4,'test_records':7,'aggregate':result['aggregate'],'reduction':result['csrnet_reduction_percent'],'weight_decay_delta':result['weight_decay_delta']},indent=2))

if __name__ == '__main__':
    main()
