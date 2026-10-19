'''Build CPU gradient extensions (the molecular OpenMP backend is optional).'''

from pathlib import Path

import pyscf
from setuptools import Extension, setup


PYSCF_LIB = Path(pyscf.__file__).resolve().parent / 'lib'


setup(
    ext_modules=[
        Extension(
            'dcforge.cpu.grad._grid_response_becke',
            sources=['dcforge/cpu/grad/c_ext/grid_response_becke.c'],
            extra_compile_args=['-O3', '-std=c99', '-fopenmp', '-ffp-contract=off'],
            extra_link_args=['-fopenmp'],
            libraries=['m'],
            optional=True,
        ),
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
