'''Density-fitted unrestricted molecular CPU HF-DFT gradients.'''

from pyscf.df.grad import uks as df_uks_grad
from pyscf.lib import logger

from dcforge.cpu.grad import uhf_dft


class Gradients(uhf_dft.Gradients, df_uks_grad.Gradients):
    '''Unrestricted density-fitted target-DFT-on-UHF gradient driver.'''

    _keys = uhf_dft.Gradients._keys | df_uks_grad.Gradients._keys

    def __init__(self, mf):
        if not hasattr(mf, 'with_df') or not hasattr(mf.mf_hf, 'with_df'):
            raise TypeError('DF Gradients requires density-fitted source and target')
        if mf.only_dfj or mf.mf_hf.only_dfj:
            raise NotImplementedError(
                'analytic UHF-DFT gradients with only_dfj=True are not supported'
            )
        uhf_dft.Gradients.__init__(self, mf)
        self.auxbasis_response = True
        self.fock_deriv_method = 'direct'

    def dump_flags(self, verbose=None):
        '''Log UHF-DFT and complete representation-response controls.'''
        uhf_dft.Gradients.dump_flags(self, verbose)
        logger.info(self, 'auxbasis_response = %s', self.auxbasis_response)
        return self

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        '''Evaluate the DF-UHF gradient with grid and auxiliary response.'''
        if not self.grid_response:
            raise NotImplementedError(
                'DF UHF-DFT gradients require grid_response=True'
            )
        if not self.auxbasis_response:
            raise NotImplementedError(
                'DF UHF-DFT gradients require auxbasis_response=True'
            )
        return uhf_dft.Gradients.kernel(
            self, mo_energy, mo_coeff, mo_occ, atmlst)


Grad = Gradients
