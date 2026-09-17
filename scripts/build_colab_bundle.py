"""Export an allowlisted source snapshot and runnable Colab notebook (no git writes)."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = '03_hyperparameters.ipynb'


def build(output):
    output = Path(output).resolve()
    if output.is_relative_to(ROOT.resolve()):
        raise ValueError("bundle output must be outside the repository")
    output.mkdir(parents=True, exist_ok=True)
    # Explicit allowlist: never include datasets, outputs, credentials, .git or .venv.
    paths = [ROOT / p for p in ('pyproject.toml', 'uv.lock', '.python-version',
                               'README.md', 'docs/hyperparameter-search.md',
                               'notebooks/' + NOTEBOOK)]
    for folder, pattern in [('src', '*.py'), ('scripts', '*.py'), ('tests', '*.py'), ('configs', '*.json')]:
        paths.extend(sorted((ROOT / folder).rglob(pattern)))
    # Final-report reproduction belongs to the full repo, not a training-only zip.
    excluded = {'scripts/plot_training_curves.py', 'tests/test_results_evidence.py'}
    data = {str(p.relative_to(ROOT)): p.read_bytes() for p in paths
            if str(p.relative_to(ROOT)) not in excluded}
    if (ROOT / '.git').exists():
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
    else:
        # Colab uploads intentionally have no .git; preserve snapshot provenance.
        inherited_path = ROOT / 'BUNDLE_MANIFEST.json'
        inherited = json.loads(inherited_path.read_text()) if inherited_path.exists() else {}
        revision = inherited.get('base_revision')
        dirty = inherited.get('working_tree_dirty', True) or any(
            inherited.get('files', {}).get(name) != hashlib.sha256(value).hexdigest()
            for name, value in data.items())
    manifest = {'base_revision': revision, 'working_tree_dirty': dirty,
                'description': 'Exact local source snapshot; not a claim of a pushed revision',
                'files': {name: hashlib.sha256(value).hexdigest() for name, value in data.items()}}
    data['BUNDLE_MANIFEST.json'] = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
    archive = output / 'crowd-hyperparameters-source.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, value in sorted(data.items()):
            # Deterministic member metadata; identical sources produce the same zip.
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, value)
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    notebook = json.loads((ROOT / 'notebooks' / NOTEBOOK).read_text())
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        source = source.replace('EXPECTED_BUNDLE_SHA256 = None',
                                f'EXPECTED_BUNDLE_SHA256 = "{checksum}"')
        cell['source'] = source.splitlines(True)
    notebook['metadata']['source_bundle_sha256'] = checksum
    (output / NOTEBOOK).write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + '\n')
    (output / 'crowd-hyperparameters-source.sha256').write_text(f'{checksum}  {archive.name}\n')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert set(z.namelist()) == set(data)
        for name, expected in manifest['files'].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == expected
    print(json.dumps({'bundle': str(archive), 'notebook': str(output / NOTEBOOK),
                      'source_files': len(manifest['files']), 'sha256': checksum}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='artifact directory outside the repository')
    build(parser.parse_args().output)
