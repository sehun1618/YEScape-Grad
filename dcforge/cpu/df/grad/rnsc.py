# DC-Forge
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

'''Density-fitted restricted molecular CPU non-SCF gradients.'''

from pyscf.df.grad import rks as df_rks_grad
from pyscf.lib import logger

from dcforge.cpu.df.grad.response import make_response
from dcforge.cpu.grad import grid_response
from dcforge.cpu.grad import rnsc


class Gradients(rnsc.Gradients, df_rks_grad.Gradients):
    '''Restricted density-fitted target-DFT-on-source gradient driver.

    This class keeps the DC-DFT Z-vector assembly from
    :class:`dcforge.cpu.grad.rnsc.Gradients` and obtains all target J/K
    derivatives from :class:`pyscf.df.grad.rks.Gradients`.  PySCF therefore
    supplies the three-center integral and auxiliary-metric derivatives; only
    the nonvariational non-SCF assembly remains in DC-Forge.

    Attributes:
        grid_response : bool
            Include target numerical-grid motion.  This must remain ``True``
            for the supported analytic DF gradient.
        auxbasis_response : bool
            Include target and source auxiliary-basis response.  This must
            remain ``True`` for the supported analytic DF gradient.
        cphf_conv_tol : float
            Tolerance passed to the source DF-RHF CPHF solve.
        cphf_max_cycle : int
            Maximum number of source DF-RHF CPHF cycles.
    '''

    _keys = rnsc.Gradients._keys | df_rks_grad.Gradients._keys

    def __init__(self, mf):
        if not hasattr(mf, 'with_df') or not hasattr(mf.mf_source, 'with_df'):
            raise TypeError('DF Gradients requires density-fitted source and target')
        if mf.only_dfj or mf.mf_source.only_dfj:
            raise NotImplementedError(
                'analytic non-SCF gradients with only_dfj=True are not supported'
            )
        rnsc.Gradients.__init__(self, mf)
        self.auxbasis_response = True
        self.fock_deriv_method = 'direct'
        self.response_backend = 'ao' if hasattr(mf.mf_source, 'xc') else 'mo_direct'

    def make_response(self, Co, Cv):
        return make_response(self, Co, Cv, unrestricted=False)

    def get_veff(self, mol=None, dm=None):
        return grid_response.get_veff(self, df_rks_grad.get_veff, mol, dm)

    def dump_flags(self, verbose=None):
        '''Log the DC-DFT, grid-response and auxiliary-response controls.'''
        rnsc.Gradients.dump_flags(self, verbose)
        logger.info(self, 'auxbasis_response = %s', self.auxbasis_response)
        return self

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        '''Evaluate the complete DF gradient with all representation response.'''
        if not self.grid_response:
            raise NotImplementedError(
                'DF non-SCF gradients require grid_response=True'
            )
        if not self.auxbasis_response:
            raise NotImplementedError(
                'DF non-SCF gradients require auxbasis_response=True'
            )
        return rnsc.Gradients.kernel(
            self, mo_energy, mo_coeff, mo_occ, atmlst
        )


Grad = Gradients
