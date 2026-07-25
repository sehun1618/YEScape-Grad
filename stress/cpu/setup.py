"""Build the isolated PySCF-compatible GDF stress extensions."""

from pathlib import Path

import pyscf
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


PYSCF_LIB = Path(pyscf.__file__).resolve().parent / "lib"


class LinkedBuildExt(build_ext):
    """Relink when setup-level PySCF library ownership changes."""

    def finalize_options(self):
        super().finalize_options()
        self.force = True


setup(
    cmdclass={"build_ext": LinkedBuildExt},
    ext_modules=[
        Extension(
            "pyscf_pbc_dcdft_stress._sr3c_strain",
            sources=[
                "src/pyscf_pbc_dcdft_stress/c_ext/sr3c_strain.c",
                "src/pyscf_pbc_dcdft_stress/c_ext/sr2c_strain.c",
                "src/pyscf_pbc_dcdft_stress/c_ext/ft_ip_ovlp_fused.c",
            ],
            include_dirs=[
                str(PYSCF_LIB),
                str(PYSCF_LIB / "deps" / "include"),
            ],
            library_dirs=[
                str(PYSCF_LIB),
                str(PYSCF_LIB / "deps" / "lib"),
            ],
            libraries=["cgto", "cint"],
            runtime_library_dirs=[
                "$ORIGIN/../pyscf/lib",
                "$ORIGIN/../pyscf/lib/deps/lib",
            ],
            extra_compile_args=["-fopenmp"],
            extra_link_args=["-fopenmp", "-Wl,--allow-shlib-undefined"],
        )
    ]
)
