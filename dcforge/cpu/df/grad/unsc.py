# DC-Forge
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

'''Density-fitted unrestricted molecular CPU non-SCF gradients.'''

from pyscf.df.grad import uks as df_uks_grad
from pyscf.lib import logger

from dcforge.cpu.grad import grid_response
from dcforge.cpu.grad import unsc


class Gradients(unsc.Gradients, df_uks_grad.Gradients):
    '''Unrestricted density-fitted target-DFT-on-source gradient driver.'''

    _keys = unsc.Gradients._keys | df_uks_grad.Gradients._keys

    def __init__(self, mf):
        if not hasattr(mf, 'with_df') or not hasattr(mf.mf_source, 'with_df'):
            raise TypeError('DF Gradients requires density-fitted source and target')
        if mf.only_dfj or mf.mf_source.only_dfj:
            raise NotImplementedError(
                'analytic unrestricted non-SCF gradients with only_dfj=True are not supported'
            )
        unsc.Gradients.__init__(self, mf)
        self.auxbasis_response = True
        self.fock_deriv_method = 'direct'

    def get_veff(self, mol=None, dm=None):
        return grid_response.get_veff(self, df_uks_grad.get_veff, mol, dm)

    def dump_flags(self, verbose=None):
        '''Log unrestricted non-SCF and complete representation-response controls.'''
        unsc.Gradients.dump_flags(self, verbose)
        logger.info(self, 'auxbasis_response = %s', self.auxbasis_response)
        return self

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        '''Evaluate the DF unrestricted gradient with grid and auxiliary response.'''
        if not self.grid_response:
            raise NotImplementedError(
                'DF unrestricted non-SCF gradients require grid_response=True'
            )
        if not self.auxbasis_response:
            raise NotImplementedError(
                'DF unrestricted non-SCF gradients require auxbasis_response=True'
            )
        return unsc.Gradients.kernel(
            self, mo_energy, mo_coeff, mo_occ, atmlst)


Grad = Gradients
