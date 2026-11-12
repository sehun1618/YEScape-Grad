"""The standalone distribution must not depend on the development checkout."""

import ast
from pathlib import Path

import yescape_grad


def test_molecular_package_boundary():
    root = Path(yescape_grad.__file__).parent
    assert not list(root.rglob('pbc'))
    assert not (root / 'tools' / 'pbc_hf_state.py').exists()
    for path in root.rglob('*.py'):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [entry.name for entry in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or '']
            else:
                continue
            assert not any(name.startswith('experiments') or
                           (name.startswith('yescape_grad.') and
                            'pbc' in name.split('.')) for name in modules), path
