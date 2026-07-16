"""Build the small PySCF-compatible Fourier AO derivative extension."""

from pathlib import Path

import pyscf
from setuptools import Extension, setup


PYSCF_LIB = Path(pyscf.__file__).resolve().parent / "lib"


setup(
    ext_modules=[
        Extension(
            "pyscf_pbc_dcdft.df.gdf._ft_ip_ovlp",
            sources=[
                "src/pyscf_pbc_dcdft/df/gdf/c_ext/ft_ip_ovlp.c",
            ],
            include_dirs=[
                str(PYSCF_LIB),
                str(PYSCF_LIB / "deps" / "include"),
            ],
            # PySCF shared libraries are loaded RTLD_GLOBAL by the Python
            # loader before this module is imported.  Leaving these symbols
            # unresolved avoids embedding one environment-specific rpath in
            # the wheel.
            extra_link_args=["-Wl,--allow-shlib-undefined"],
        )
    ]
)
