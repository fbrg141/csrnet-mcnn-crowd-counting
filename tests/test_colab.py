import ast
import json
from pathlib import Path


def test_colab_notebook_is_safe_and_compiles():
    path = Path('notebooks/03_hyperparameters.ipynb')
    nb = json.loads(path.read_text())
    assert nb['nbformat'] == 4
    assert nb['metadata']['colab']['name'] == path.name
    sources = []
    for cell in nb['cells']:
        text = ''.join(cell['source'])
        sources.append(text)
        if cell['cell_type'] == 'code':
            ast.parse(text)
            assert cell['outputs'] == []
    text = '\n'.join(sources)
    assert 'rm -rf' not in text
    assert 'RUN_SEARCH = False' in text
    assert 'RUN_FINAL_EVALUATION = False' in text
    assert '--locked' in text
    assert 'files.upload()' in text
    assert 'CACHE_VERSION == 2' in text


def test_colab_baseline_plan_is_frozen_before_test_access(tmp_path):
    import pytest
    nb = json.loads(Path('notebooks/03_hyperparameters.ipynb').read_text())
    cell = next(''.join(c['source']) for c in nb['cells']
                if ''.join(c['source']).startswith('if RUN_FINAL_EVALUATION:'))
    calls = []
    scope = dict(RUN_FINAL_EVALUATION=True, EVALUATE_WITH_CONFIRMATION=False,
                 EVALUATE_CSRNET=True, CSRNET_SEEDS=[42, 123], OUTPUT=tmp_path,
                 SEARCH_ARGS=['-m', 'src.search'], SEARCH_OUT=tmp_path / 'search',
                 PART='A', EPOCHS=50, BATCH_SIZE=4, json=json,
                 run=lambda *a: calls.append(a))
    with pytest.raises(RuntimeError, match='Finish all requested CSRNet'):
        exec(cell, scope)
    assert not any('--stage' in call[0] for call in calls)
    # An already frozen plan cannot be silently changed to omit CSRNet.
    (tmp_path / 'other_evaluation_plan.json').write_text('{"runs": [{"seed": 42}]}')
    scope['EVALUATE_CSRNET'] = False
    with pytest.raises(RuntimeError, match='baseline evaluation plan changed'):
        exec(cell, scope)
    assert not any('--stage' in call[0] for call in calls)


def test_bundle_refuses_output_inside_source_tree(tmp_path, monkeypatch):
    import pytest
    from scripts import build_colab_bundle
    root = tmp_path / 'source'
    root.mkdir()
    monkeypatch.setattr(build_colab_bundle, 'ROOT', root)
    with pytest.raises(ValueError, match='outside the repository'):
        build_colab_bundle.build(root / 'notebooks')
    assert not (root / 'notebooks').exists()


def test_bundle_is_allowlisted_and_notebook_pins_snapshot(tmp_path):
    import hashlib
    import zipfile
    from scripts.build_colab_bundle import build
    build(tmp_path)
    archive = tmp_path / 'crowd-hyperparameters-source.zip'
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    notebook = json.loads((tmp_path / '03_hyperparameters.ipynb').read_text())
    assert notebook['metadata']['source_bundle_sha256'] == checksum
    assert any(f'EXPECTED_BUNDLE_SHA256 = "{checksum}"' in ''.join(c['source']) for c in notebook['cells'])
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        assert all(not set(Path(n).parts) & {'.git', '.venv', 'data', 'reports', '__pycache__'} for n in names)
        assert 'PLAN.md' not in names
        assert 'notebooks/03_hyperparameters.ipynb' in names
        assert 'scripts/plot_training_curves.py' not in names
        assert 'tests/test_results_evidence.py' not in names
        assert 'EXPECTED_BUNDLE_SHA256 = None' in z.read('notebooks/03_hyperparameters.ipynb').decode()
        manifest = json.loads(z.read('BUNDLE_MANIFEST.json'))
        for name, expected in manifest['files'].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == expected
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
