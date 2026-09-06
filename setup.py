'''Build the one PySCF-compatible CPU PBC gradient callback.'''

from pathlib import Path

import pyscf
from setuptools import Extension, setup


PYSCF_LIB = Path(pyscf.__file__).resolve().parent / 'lib'


setup(
    ext_modules=[
        Extension(
            'dcforge.cpu.pbc.df.gaussian._ft_ip_ovlp',
            sources=[
                'dcforge/cpu/pbc/df/gaussian/c_ext/ft_ip_ovlp.c',
            ],
            include_dirs=[
                str(PYSCF_LIB),
                str(PYSCF_LIB / 'deps' / 'include'),
            ],
            # ft.py loads libcgto/libpbc with RTLD_GLOBAL before importing
            # this callback, avoiding an environment-specific PySCF rpath.
            extra_link_args=['-Wl,--allow-shlib-undefined'],
        ),
    ],
)
