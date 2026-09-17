"""Offline checks for the final report's original evidence and derived numbers."""
import re

from scripts.plot_training_curves import derive, ROOT


def test_presentation_uses_existing_shared_figures():
    source = ROOT / 'presentation/presentation.tex'
    figures = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', source.read_text())
    assert len(figures) == 5
    for name in figures:
        path = (source.parent / name).resolve()
        assert path.parent == ROOT / 'reports/figures'
        assert path.is_file()
    assert not (ROOT / 'presentation/figures').exists()


def test_final_evidence_and_scope():
    result = derive()  # checks every source hash, epoch history and original test metric
    assert len(result['ranking']) == 12
    assert {row['seed'] for row in result['seed42']} == {42}
    assert {row['n_test'] for row in result['seed42']} == {182}
    assert 0 < result['weight_decay_delta'] < 0.02
    assert {(r['lr'], r['momentum'], r['weight_decay']) for r in result['ranking']} == {
        (lr, momentum, wd) for lr in (1e-6, 3e-6, 1e-5)
        for momentum in (.9, .95) for wd in (0, 1e-4)
    }


def test_report_and_slides_match_derived_values():
    result = derive()
    for name in ('README.md', 'reports/report.md', 'presentation/presentation.tex'):
        content = (ROOT / name).read_text()
        for row in result['seed42']:
            for metric in ('mae', 'rmse'):
                assert f"{row[metric]:.4f}" in content, (name, row['label'], metric)
        for aggregate in result['aggregate'].values():
            for value in aggregate.values():
                assert f"{value:.4f}" in content, (name, value)
