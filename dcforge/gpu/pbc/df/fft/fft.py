'''GPU4PySCF FFTDF provider with the DC-Forge PBC contract.'''

import numpy as np
from gpu4pyscf.pbc import df

from dcforge.cpu.pbc.kpts import normalize_kpts


class FFTDF(df.FFTDF):
    '''GPU FFTDF provider shared by source and target.

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
        Analytic gradients are currently validated only at real Gamma.
    '''

    backend_name = 'fftdf'
    device = 'gpu'
    _keys = df.FFTDF._keys | {
        'kpts_reference', 'kpts_roundtrip_max_abs',
    }

    def __init__(self, cell, mesh=None, kpts=None):
        kpts = normalize_kpts(kpts, required=False)
        df.FFTDF.__init__(self, cell, kpts=kpts)
        # GPU4PySCF stores scaled coordinates and exposes Cartesian k-points
        # through a round trip.  Retain the exact user array for state
        # identity and report the harmless numerical round-trip residual.
        self.kpts_reference = kpts.copy()
        self.kpts_roundtrip_max_abs = float(np.max(np.abs(
            np.asarray(self.kpts) - self.kpts_reference)))
        if mesh is not None:
            mesh = np.asarray(mesh, dtype=int).reshape(3)
            if np.any(mesh <= 0):
                raise ValueError('FFT mesh dimensions must be positive')
            self.mesh = mesh

    def capability(self):
        '''Return the validated GPU FFTDF capability boundary.'''
        sampled_k = len(self.kpts) > 1 or np.any(np.asarray(self.kpts) != 0)
        return {
            'backend': self.backend_name,
            'device': self.device,
            'sampling': 'full_bz' if sampled_k else 'gamma',
            'reference': 'KRHF' if sampled_k else 'RHF',
            'target': ['PBE', 'R2SCAN'],
            'energy': True,
            'analytic_gradient': True,
            'stress': False,
            'mesh': [int(value) for value in self.mesh],
            'kpts_roundtrip_max_abs': self.kpts_roundtrip_max_abs,
        }
