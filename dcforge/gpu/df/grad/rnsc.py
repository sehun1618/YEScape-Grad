# DC-Forge
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

"""Density-fitted restricted molecular GPU non-SCF gradients."""

from gpu4pyscf.df.grad import rks as df_rks_grad
from gpu4pyscf.lib import logger

from dcforge.gpu.df.grad.memory import (
    DEFAULT_FOCK_DERIV_PLANNER_LIMIT_BYTES,
)
from dcforge.gpu.df.grad.rhf_nv import grad_fock_direct
from dcforge.gpu.df.grad.rhf_zvec import (
    gen_vind as gen_vind_mo,
)
from dcforge.gpu.df.grad.rhf_zvec import (
    get_z_response_oo as get_z_response_oo_mo,
)
from dcforge.gpu.grad import rnsc


class Gradients(rnsc.Gradients, df_rks_grad.Gradients):
    """Restricted density-fitted target-DFT-on-source GPU gradient driver.

    The common DC-DFT assembly is inherited from
    :class:`dcforge.gpu.grad.rnsc.Gradients`.  GPU4PySCF DF-RKS supplies
    provider-specific J/K and auxiliary-basis derivatives through
    ``energy_ee`` and ``jk_energy_per_atom``.
    """

    _keys = (
        rnsc.Gradients._keys
        | df_rks_grad.Gradients._keys
        | {
            "fock_deriv_planner_limit_bytes",
            "fock_deriv_planner",
        }
    )

    def __init__(self, mf):
        if not hasattr(mf, "with_df") or not hasattr(mf.mf_source, "with_df"):
            raise TypeError(
                "GPU DF Gradients requires density-fitted source and target"
            )
        if mf.with_df is not mf.mf_source.with_df:
            raise ValueError(
                "GPU DF gradient requires one shared source/target with_df"
            )
        if mf.only_dfj or mf.mf_source.only_dfj:
            raise NotImplementedError("GPU DF gradient requires only_dfj=False")
        rnsc.Gradients.__init__(self, mf)
        self.auxbasis_response = True
        self.fock_deriv_method = "direct"
        # The existing MO-direct action is HF-only: it omits the source fxc.
        self.response_backend = "ao" if hasattr(mf.mf_source, "xc") else "mo_direct"
        self.fock_deriv_planner_limit_bytes = DEFAULT_FOCK_DERIV_PLANNER_LIMIT_BYTES
        self.fock_deriv_planner = None

    def grad_fock(self, Co, Cv, zvec, dm0, mo_coeff, mo_occ, atmlst):
        """Return the source-Fock term through the selected DF route."""
        if hasattr(self.base.mf_source, "xc"):
            return rnsc.Gradients.grad_fock(
                self, Co, Cv, zvec, dm0, mo_coeff, mo_occ, atmlst)
        method = self.fock_deriv_method
        self.fock_deriv_method_used = method
        self.fock_deriv_planner = None
        if method == "direct":
            return grad_fock_direct(self, Co, Cv, zvec, dm0, atmlst)
        return rnsc.Gradients.grad_fock(
            self, Co, Cv, zvec, dm0, mo_coeff, mo_occ, atmlst
        )

    def gen_vind(self, Co, Cv, fvind):
        """Return the DF MO-direct source-RHF VO response action."""
        if self.response_backend == "ao":
            return rnsc.Gradients.gen_vind(self, Co, Cv, fvind)
        if self.response_backend == "mo_direct":
            if hasattr(self.base.mf_source, "xc"):
                raise ValueError("DFA source requires response_backend='ao' to include fxc")
            self.response_backend_used = "mo_direct"
            return gen_vind_mo(self.base.mf_source, Co, Cv)
        raise ValueError(
            "DF GPU response_backend must be 'ao' or 'mo_direct'; "
            f"got {self.response_backend!r}"
        )

    def get_z_response_oo(self, Co, Cv, zvec, fvind):
        """Return the DF MO-direct adjoint response OO block."""
        if self.response_backend == "ao":
            return rnsc.Gradients.get_z_response_oo(self, Co, Cv, zvec, fvind)
        return get_z_response_oo_mo(self.base.mf_source, Co, Cv, zvec)

    def dump_flags(self, verbose=None):
        """Log common GPU and DF auxiliary-response controls."""
        rnsc.Gradients.dump_flags(self, verbose)
        logger.info(self, "auxbasis_response = %s", self.auxbasis_response)
        logger.info(
            self,
            "fock_deriv_planner_limit_bytes = %s",
            self.fock_deriv_planner_limit_bytes,
        )
        return self

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        """Evaluate the complete DF gradient with all representation response."""
        if not self.auxbasis_response:
            raise NotImplementedError("GPU gradient requires auxbasis_response=True")
        return rnsc.Gradients.kernel(self, mo_energy, mo_coeff, mo_occ, atmlst)


Grad = Gradients
