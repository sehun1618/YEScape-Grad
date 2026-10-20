# DC-Forge
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

"""Density-fitted unrestricted molecular GPU HF-DFT gradients."""

from gpu4pyscf.df.grad import uks as df_uks_grad
from gpu4pyscf.lib import logger

from dcforge.gpu.df.grad.memory import (
    DEFAULT_FOCK_DERIV_PLANNER_LIMIT_BYTES,
)
from dcforge.gpu.df.grad.uhf_nv import grad_fock_direct
from dcforge.gpu.df.grad.uhf_zvec import (
    gen_vind as gen_vind_mo,
)
from dcforge.gpu.df.grad.uhf_zvec import (
    get_z_response_oo as get_z_response_oo_mo,
)
from dcforge.gpu.grad import uhf_dft


class Gradients(uhf_dft.Gradients, df_uks_grad.Gradients):
    """Unrestricted density-fitted target-DFT-on-UHF GPU gradient driver."""

    _keys = (
        uhf_dft.Gradients._keys
        | df_uks_grad.Gradients._keys
        | {
            "fock_deriv_planner_limit_bytes",
            "fock_deriv_planner",
        }
    )

    def __init__(self, mf):
        if not hasattr(mf, "with_df") or not hasattr(mf.mf_hf, "with_df"):
            raise TypeError(
                "GPU DF-UHF Gradients requires density-fitted source and target"
            )
        if mf.with_df is not mf.mf_hf.with_df:
            raise ValueError(
                "GPU DF-UHF gradient requires one shared source/target with_df"
            )
        if mf.only_dfj or mf.mf_hf.only_dfj:
            raise NotImplementedError("GPU DF-UHF gradient requires only_dfj=False")
        uhf_dft.Gradients.__init__(self, mf)
        self.auxbasis_response = True
        self.fock_deriv_method = "direct"
        self.response_backend = "mo_direct"
        self.fock_deriv_planner_limit_bytes = DEFAULT_FOCK_DERIV_PLANNER_LIMIT_BYTES
        self.fock_deriv_planner = None

    def grad_fock(self, Co, Cv, zvec, dm0, mo_coeff, mo_occ, atmlst):
        """Return the source-Fock term through the selected DF-UHF route."""
        method = self.fock_deriv_method
        self.fock_deriv_method_used = method
        self.fock_deriv_planner = None
        if method == "direct":
            return grad_fock_direct(self, Co, Cv, zvec, dm0, atmlst)
        return uhf_dft.Gradients.grad_fock(
            self, Co, Cv, zvec, dm0, mo_coeff, mo_occ, atmlst
        )

    def gen_vind(self, Co, Cv, fvind):
        """Return the DF MO-direct spin-coupled response action."""
        if self.response_backend == "ao":
            return uhf_dft.Gradients.gen_vind(self, Co, Cv, fvind)
        if self.response_backend == "mo_direct":
            self.response_backend_used = "mo_direct"
            return gen_vind_mo(self.base.mf_hf, Co, Cv)
        raise ValueError(
            "DF GPU response_backend must be 'ao' or 'mo_direct'; "
            f"got {self.response_backend!r}"
        )

    def get_z_response_oo(self, Co, Cv, zvec, fvind):
        """Return DF MO-direct spin adjoint response OO blocks."""
        if self.response_backend == "ao":
            return uhf_dft.Gradients.get_z_response_oo(self, Co, Cv, zvec, fvind)
        return get_z_response_oo_mo(self.base.mf_hf, Co, Cv, zvec)

    def dump_flags(self, verbose=None):
        """Log common GPU UHF and DF auxiliary-response controls."""
        uhf_dft.Gradients.dump_flags(self, verbose)
        logger.info(self, "auxbasis_response = %s", self.auxbasis_response)
        logger.info(
            self,
            "fock_deriv_planner_limit_bytes = %s",
            self.fock_deriv_planner_limit_bytes,
        )
        return self

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        """Evaluate the complete DF-UHF gradient with representation response."""
        if not self.auxbasis_response:
            raise NotImplementedError(
                "GPU UHF gradient requires auxbasis_response=True"
            )
        return uhf_dft.Gradients.kernel(self, mo_energy, mo_coeff, mo_occ, atmlst)


Grad = Gradients
