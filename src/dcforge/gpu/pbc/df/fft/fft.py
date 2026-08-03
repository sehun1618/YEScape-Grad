'''GPU4PySCF FFTDF provider with the DC-Forge PBC contract.'''

import numpy as np
from gpu4pyscf.pbc import df


class FFTDF(df.FFTDF):
    '''Gamma-point GPU FFTDF provider shared by source and target.

    Args:
        cell : pyscf.pbc.gto.Cell
            Built three-dimensional periodic cell.

    Kwargs:
        mesh : sequence of three int or None
            FFT mesh.  When omitted, GPU4PySCF uses the built ``cell.mesh``.

    Notes:
        The fixed-density energy route requires this provider to be shared by
        source RHF and target RKS.  The analytic-gradient route uses the same
        finite ``mesh`` through ``df.fft.response``; it is therefore part of
        the provider contract rather than a gradient-only representation.
    '''

    backend_name = 'fftdf'
    device = 'gpu'

    def __init__(self, cell, mesh=None):
        df.FFTDF.__init__(self, cell, kpts=np.zeros((1, 3)))
        if mesh is not None:
            mesh = np.asarray(mesh, dtype=int).reshape(3)
            if np.any(mesh <= 0):
                raise ValueError('FFT mesh dimensions must be positive')
            self.mesh = mesh

    def capability(self):
        '''Return the validated GPU FFTDF capability boundary.'''
        return {
            'backend': self.backend_name,
            'device': self.device,
            'sampling': 'gamma',
            'reference': 'RHF',
            'target': ['PBE', 'R2SCAN'],
            'energy': True,
            'analytic_gradient': True,
            'stress': False,
            'mesh': [int(value) for value in self.mesh],
        }
