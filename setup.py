'''Build CPU gradient extensions (the molecular OpenMP backend is optional).'''

from setuptools import Extension, setup


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
    ],
)
