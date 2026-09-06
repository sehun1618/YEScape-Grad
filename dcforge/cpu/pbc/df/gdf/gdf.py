'''PySCF GDF provider with an explicit DC-Forge contract.'''

import numpy as np
from pyscf.pbc import df
from pyscf.pbc.df.rsdf_builder import _RSGDFBuilder
from pyscf.pbc.lib.kpts_helper import unique

from dcforge.cpu.pbc.df.gdf.metric import auxiliary_metric
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
        'omega', 'metric_mesh',
        'resolved_omega', 'resolved_source_mesh', 'resolved_metric_mesh',
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
        # Optional numerical GDF splitting controls.  ``omega`` is the
        # algorithmic real/reciprocal partition used by PySCF's private
        # _RSGDFBuilder; it is not a physical range-separated functional.
        # Defaults preserve upstream auto-resolution exactly.
        self.omega = None
        self.metric_mesh = None
        self.resolved_omega = None
        self.resolved_source_mesh = None
        self.resolved_metric_mesh = None
        if auxbasis is not None:
            self.auxbasis = auxbasis

    def _make_j3c(self, cell=None, auxcell=None, kptij_lst=None,
                  cderi_file=None):
        r'''Build CDERI with optional fixed GDF splitting parameters.

        PySCF normally resolves the RSGDF ``omega`` and the two reciprocal
        meshes independently for every Cell.  Explicit values make numerical
        representation studies reproducible across otherwise equivalent
        provider constructions.

        Setting ``omega``, ``mesh`` and ``metric_mesh`` opts into that fixed
        representation.  With all three left as ``None`` this method delegates
        unchanged to :class:`pyscf.pbc.df.GDF`.
        '''
        fixed = (
            self.omega is not None or self.mesh is not None
            or self.metric_mesh is not None)
        if not fixed:
            self.resolved_omega = None
            self.resolved_source_mesh = None
            self.resolved_metric_mesh = None
            return df.GDF._make_j3c(
                self, cell=cell, auxcell=auxcell,
                kptij_lst=kptij_lst, cderi_file=cderi_file)

        if cell is None:
            cell = self.cell
        if auxcell is None:
            auxcell = self.auxcell
        if cderi_file is None:
            cderi_file = self._cderi_to_save
        if self._prefer_ccdf or float(cell.omega) > 0:
            raise NotImplementedError(
                'fixed GDF splitting currently requires the RSGDF builder')
        if self.metric_mesh is None:
            raise ValueError(
                'fixed GDF splitting requires an explicit metric_mesh')

        if self.kpts_band is None:
            kpts_union = self.kpts
        else:
            kpts_union = unique(np.vstack((self.kpts, self.kpts_band)))[0]
        builder = _FixedMetricRSGDFBuilder(
            cell, auxcell, kpts_union, metric_mesh=self.metric_mesh)
        if self.mesh is not None:
            builder.mesh = np.asarray(self.mesh, dtype=int)
        builder.linear_dep_threshold = float(self.linear_dep_threshold)
        builder.build(omega=self.omega)
        j_only = self._j_only or len(kpts_union) == 1
        builder.make_j3c(
            cderi_file, j_only=j_only, dataname=self._dataname,
            kptij_lst=kptij_lst)
        self.resolved_omega = float(builder.omega)
        self.resolved_source_mesh = np.asarray(
            builder.mesh, dtype=int).copy()
        self.resolved_metric_mesh = np.asarray(
            builder.metric_mesh, dtype=int).copy()

    def capability(self):
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
        self.resolved_omega = None
        self.resolved_source_mesh = None
        self.resolved_metric_mesh = None
        return self


class _FixedMetricRSGDFBuilder(_RSGDFBuilder):
    '''PySCF RSGDF builder with an explicit two-center metric mesh.'''

    def __init__(self, cell, auxcell, kpts, metric_mesh):
        _RSGDFBuilder.__init__(self, cell, auxcell, kpts)
        metric_mesh = np.asarray(metric_mesh, dtype=int)
        if metric_mesh.shape != (3,) or np.any(metric_mesh <= 0):
            raise ValueError(
                'fixed GDF metric_mesh must contain three positive integers')
        self.metric_mesh = metric_mesh.copy()

    def get_2c2e(self, uniq_kpts):
        r'''Return ``M_SR+M_LR+M_G0`` on the fixed central metric mesh.'''
        return [
            auxiliary_metric(
                self.cell, self.auxcell, self.omega, self.metric_mesh,
                q=kpt)
            for kpt in np.asarray(uniq_kpts, dtype=float).reshape(-1, 3)
        ]
