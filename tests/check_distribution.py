'''Inspect a built wheel and sdist: python tests/check_distribution.py ARCHIVE...'''

from pathlib import Path, PurePosixPath
import sys
import tarfile
import zipfile


def check_archive(path):
    if path.suffix == '.whl':
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    else:
        with tarfile.open(path) as archive:
            names = [str(PurePosixPath(n).relative_to(PurePosixPath(n).parts[0]))
                     for n in archive.getnames()]
    forbidden = {'dev', 'results', 'result', '__pycache__', '.pytest_cache', '.git'}
    for name in names:
        parts = PurePosixPath(name)
        assert not forbidden.intersection(parts.parts), name
        assert parts.name != 'THEORY.md', name
        assert parts.suffix not in {'.pyc', '.pyo', '.chk', '.h5', '.log'}, name
    assert 'dcforge/cpu/energy/rnsc.py' in names
    assert 'dcforge/gpu/energy/rnsc.py' in names
    if path.suffix != '.whl':
        assert 'dcforge/cpu/grad/c_ext/grid_response_becke.c' in names
    print(f'PASS {path.name}: {len(names)} archive entries')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for argument in sys.argv[1:]:
        check_archive(Path(argument))
