'''PySCF GDF provider with an explicit DC-Forge contract.'''

import numpy as np
from pyscf.pbc import df


class GDF(df.GDF):
    '''Gamma-point periodic Gaussian density-fitting provider.'''

    backend_name = 'gdf'
    device = 'cpu'
    _keys = df.GDF._keys | {
        'fourier_response_aux_storage',
        'fourier_response_ao_pair_bytes',
        'cderi_response_storage', 'cderi_response_blksize',
        'cderi_response_batch_bytes', 'cderi_value_parity_max_abs',
        'cderi_central_validated',
    }

    def __init__(self, cell, auxbasis=None):
        df.GDF.__init__(self, cell, kpts=np.zeros((1, 3)))
        self.fourier_response_aux_storage = None
        self.fourier_response_ao_pair_bytes = None
        self.cderi_response_storage = None
        self.cderi_response_blksize = None
        self.cderi_response_batch_bytes = None
        self.cderi_value_parity_max_abs = None
        self.cderi_central_validated = False
        if auxbasis is not None:
            self.auxbasis = auxbasis

    def capability(self):
        return {
            'backend': self.backend_name,
            'device': self.device,
            'sampling': 'gamma',
            'reference': 'RHF',
            'target': 'PBE',
            'energy': True,
            'analytic_gradient': True,
            'stress': False,
            'auxbasis': self.auxbasis,
        }

    def reset(self, cell=None):
        '''Reset provider data and invalidate all derivative diagnostics.'''
        df.GDF.reset(self, cell)
        self.fourier_response_aux_storage = None
        self.fourier_response_ao_pair_bytes = None
        self.cderi_response_storage = None
        self.cderi_response_blksize = None
        self.cderi_response_batch_bytes = None
        self.cderi_value_parity_max_abs = None
        self.cderi_central_validated = False
        return self
