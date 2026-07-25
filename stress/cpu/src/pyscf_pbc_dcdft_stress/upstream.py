"""Version-guarded access to supported PySCF periodic stress primitives.

PySCF currently exposes these functions from ``pyscf.pbc.grad.rks_stress``.
Keeping every dependency in this adapter prevents private implementation
details from leaking into the rest of the stress package.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect

import numpy as np


SUPPORTED_PYSCF_PREFIXES = ("2.13.", "2.14.")


def supports_pyscf_version(version) -> bool:
    """Return whether the private stress API was validated for ``version``."""

    return str(version).startswith(SUPPORTED_PYSCF_PREFIXES)


@dataclass(frozen=True)
class PySCFStressPrimitives:
    version: str
    module: object

    @classmethod
    def load(cls):
        import pyscf
        from pyscf.pbc.grad import rks_stress

        version = str(pyscf.__version__)
        if not supports_pyscf_version(version):
            raise RuntimeError(
                "periodic stress adapters are validated for PySCF 2.13.x "
                "and 2.14.x; "
                f"found {version}"
            )
        expected = {
            "get_vxc": ("ks_grad", "cell", "dm", "with_j", "with_nuc"),
            "_eval_ao_strain_derivatives": (
                "cell", "coords", "kpts", "deriv", "out"
            ),
        }
        for name, parameters in expected.items():
            function = getattr(rks_stress, name, None)
            if function is None:
                raise RuntimeError(f"PySCF rks_stress is missing {name}()")
            observed = tuple(inspect.signature(function).parameters)
            if observed != parameters:
                raise RuntimeError(
                    f"PySCF rks_stress.{name} signature changed: {observed}"
                )
        return cls(version=version, module=rks_stress)

    def xc_derivative(self, target_gradient, cell, density):
        return np.asarray(
            self.module.get_vxc(
                target_gradient, cell, density, with_j=False, with_nuc=False
            ),
            dtype=float,
        )

    def xc_and_pseudopotential_derivative(self, target_gradient, cell, density):
        return np.asarray(
            self.module.get_vxc(
                target_gradient, cell, density, with_j=False, with_nuc=True
            ),
            dtype=float,
        )

    def ao_strain_derivatives(self, cell, coordinates, *, deriv=0):
        return self.module._eval_ao_strain_derivatives(
            cell, coordinates, deriv=int(deriv)
        )


@dataclass(frozen=True)
class PySCFKPointStressPrimitives:
    """Version-guarded sampled-k uniform-grid XC strain primitive."""

    version: str
    module: object

    @classmethod
    def load(cls):
        import pyscf
        from pyscf.pbc.grad import krks_stress

        version = str(pyscf.__version__)
        if not supports_pyscf_version(version):
            raise RuntimeError(
                "sampled-k stress adapters are validated for PySCF 2.13.x "
                "and 2.14.x; "
                f"found {version}"
            )
        expected = (
            "ks_grad", "cell", "dm_kpts", "kpts", "with_j", "with_nuc"
        )
        observed = tuple(inspect.signature(krks_stress.get_vxc).parameters)
        if observed != expected:
            raise RuntimeError(
                f"PySCF krks_stress.get_vxc signature changed: {observed}"
            )
        return cls(version=version, module=krks_stress)

    def xc_derivative(self, target_gradient, cell, density, kpts):
        return np.asarray(
            self.module.get_vxc(
                target_gradient, cell, density, np.asarray(kpts),
                with_j=False, with_nuc=False,
            ),
            dtype=float,
        )
