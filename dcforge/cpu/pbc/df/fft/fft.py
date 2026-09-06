'''PySCF FFTDF provider with an explicit DC-Forge contract.'''

import numpy as np
from pyscf.pbc import df

from dcforge.cpu.pbc.kpts import normalize_kpts


class FFTDF(df.FFTDF):
    '''FFTDF provider shared by source RHF and target RKS.

    Args:
        cell : pyscf.pbc.gto.Cell
            Built three-dimensional periodic cell.

    Kwargs:
        mesh : sequence of three int or None
            FFT mesh.  When omitted, PySCF's built ``cell.mesh`` is used.
        kpts : array_like or None
            Explicit full-BZ k-points with shape ``(nkpts, 3)``.  ``None``
            retains the existing one-Gamma provider behavior.

    Notes:
        The first analytic-gradient slice requires ``with_df.mesh`` and
        ``cell.mesh`` to be identical.  Energy and every derivative kernel
        must differentiate the same finite reciprocal representation.
    '''

    backend_name = 'fftdf'
    device = 'cpu'

    def __init__(self, cell, mesh=None, kpts=None):
        kpts = normalize_kpts(kpts, required=False)
        df.FFTDF.__init__(self, cell, kpts=kpts)
        if mesh is not None:
            mesh = np.asarray(mesh, dtype=int).reshape(3)
            if np.any(mesh <= 0):
                raise ValueError('FFT mesh dimensions must be positive')
            self.mesh = mesh

    def capability(self):
        '''Return the currently validated provider boundary.'''
        sampled_k = len(self.kpts) > 1 or np.any(np.asarray(self.kpts) != 0)
        return {
            'backend': self.backend_name,
            'device': self.device,
            'sampling': 'full_bz' if sampled_k else 'gamma',
            'reference': 'KRHF' if sampled_k else 'RHF',
            'target': ['LDA', 'GGA', 'MGGA'] if sampled_k else [
                'PBE', 'R2SCAN'],
            'energy': True,
            'analytic_gradient': True,
            'stress': False,
            'mesh': [int(x) for x in self.mesh],
        }
