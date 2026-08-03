'''PySCF FFTDF provider with an explicit DC-Forge contract.'''

import numpy as np
from pyscf.pbc import df


class FFTDF(df.FFTDF):
    '''FFTDF provider shared by source RHF and target RKS.

    Args:
        cell : pyscf.pbc.gto.Cell
            Built three-dimensional periodic cell.

    Kwargs:
        mesh : sequence of three int or None
            FFT mesh.  When omitted, PySCF's built ``cell.mesh`` is used.

    Notes:
        The first analytic-gradient slice requires ``with_df.mesh`` and
        ``cell.mesh`` to be identical.  Energy and every derivative kernel
        must differentiate the same finite reciprocal representation.
    '''

    backend_name = 'fftdf'
    device = 'cpu'

    def __init__(self, cell, mesh=None):
        df.FFTDF.__init__(self, cell, kpts=np.zeros((1, 3)))
        if mesh is not None:
            mesh = np.asarray(mesh, dtype=int).reshape(3)
            if np.any(mesh <= 0):
                raise ValueError('FFT mesh dimensions must be positive')
            self.mesh = mesh

    def capability(self):
        '''Return the currently validated provider boundary.'''
        return {
            'backend': self.backend_name,
            'device': self.device,
            'sampling': 'gamma',
            'reference': 'RHF',
            'target': ['PBE', 'R2SCAN'],
            'energy': True,
            'analytic_gradient': True,
            'stress': False,
            'mesh': [int(x) for x in self.mesh],
        }
