'''PySCF RSDF provider with an explicit DC-Forge contract.'''

import numpy as np
from pyscf.pbc.df import rsdf


class RSDF(rsdf.RSGDF):
    '''Gamma-point range-separated Gaussian density-fitting provider.'''

    backend_name = 'rsdf'
    device = 'cpu'
    _keys = rsdf.RSGDF._keys | {
        'fourier_response_aux_storage',
        'fourier_response_ao_pair_bytes',
        'cderi_response_storage', 'cderi_response_blksize',
        'cderi_response_batch_bytes', 'cderi_value_parity_max_abs',
        'cderi_central_validated',
    }

    def __init__(self, cell, auxbasis=None):
        rsdf.RSGDF.__init__(self, cell, kpts=np.zeros((1, 3)))
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
            'target': ['PBE', 'R2SCAN'],
            'energy': True,
            'analytic_gradient': True,
            'stress': False,
            'auxbasis': self.auxbasis,
        }

    def response_settings(self):
        '''Return resolved source/metric range-separation settings.'''
        required = ('omega', 'mesh_compact', 'omega_j2c', 'mesh_j2c')
        missing = [key for key in required if getattr(self, key, None) is None]
        if missing:
            raise RuntimeError(
                'RSDF settings are unresolved; run build/kernel first: '
                + ', '.join(missing)
            )
        return {
            'source_omega': float(self.omega),
            'source_mesh': np.asarray(self.mesh_compact, dtype=int).tolist(),
            'metric_omega': float(self.omega_j2c),
            'metric_mesh': np.asarray(self.mesh_j2c, dtype=int).tolist(),
        }

    def reset(self, cell=None):
        '''Reset provider data and invalidate all derivative diagnostics.'''
        rsdf.RSGDF.reset(self, cell)
        self.fourier_response_aux_storage = None
        self.fourier_response_ao_pair_bytes = None
        self.cderi_response_storage = None
        self.cderi_response_blksize = None
        self.cderi_response_batch_bytes = None
        self.cderi_value_parity_max_abs = None
        self.cderi_central_validated = False
        return self
