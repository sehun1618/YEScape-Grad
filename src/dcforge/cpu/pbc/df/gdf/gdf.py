'''PySCF GDF provider with an explicit DC-Forge contract.'''

import numpy as np
from pyscf.pbc import df

from dcforge.cpu.pbc.kpts import normalize_kpts


class GDF(df.GDF):
    '''Periodic Gaussian density-fitting provider.'''

    backend_name = 'gdf'
    device = 'cpu'
    _keys = df.GDF._keys | {
        'fourier_response_aux_storage',
        'fourier_response_ao_pair_bytes',
        'cderi_response_storage', 'cderi_response_blksize',
        'cderi_response_batch_bytes', 'cderi_value_parity_max_abs',
        'cderi_central_validated', 'cderi_pair_records',
        'cderi_pair_response_records',
    }

    def __init__(self, cell, kpts=None, *, auxbasis=None):
        kpts = normalize_kpts(kpts, required=False)
        df.GDF.__init__(self, cell, kpts=kpts)
        self.fourier_response_aux_storage = None
        self.fourier_response_ao_pair_bytes = None
        self.cderi_response_storage = None
        self.cderi_response_blksize = None
        self.cderi_response_batch_bytes = None
        self.cderi_value_parity_max_abs = None
        self.cderi_central_validated = False
        self.cderi_pair_records = None
        self.cderi_pair_response_records = None
        if auxbasis is not None:
            self.auxbasis = auxbasis

    def capability(self):
        sampled_k = len(self.kpts) > 1 or np.any(np.asarray(self.kpts) != 0)
        return {
            'backend': self.backend_name,
            'device': self.device,
            'sampling': 'full_bz' if sampled_k else 'gamma',
            'reference': 'KRHF' if sampled_k else 'RHF',
            'target': ['LDA', 'GGA', 'R2SCAN'] if sampled_k else [
                'PBE', 'R2SCAN'],
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
        self.cderi_pair_records = None
        self.cderi_pair_response_records = None
        return self
