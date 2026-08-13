'''Gamma-point fixed-density GPU RHF-DFT stress skeleton terms.'''

import cupy as cp
import numpy as np

from dcforge.cpu.pbc.stress.ewald import (
    ewald_strain,
)
from dcforge.gpu.pbc.df.gaussian.strain import (
    contract_kinetic_strain,
    contract_overlap_strain,
)
from dcforge.gpu.pbc.stress.multigrid import (
    get_vxc_strain_analytic,
)
from dcforge.gpu.pbc.stress.profiler import (
    stage_start,
    stage_stop,
)


def work_hcore(cell, dm0, pseudopotential):
    r'''Return ``Tr[P(T^(xy)+V_PP^(xy))]`` in Hartree.

    Kinetic strain is contracted by GPU4PySCF's native device primitive.
    The provider-matched GTH response is already density-contracted: local
    SR/LR/G0 and exact nonlocal projector terms never enter this function as
    AO derivative matrices.
    '''
    dm0 = cp.asarray(dm0)
    kinetic = np.asarray(contract_kinetic_strain(cell, dm0))
    pp_batch = cp.asarray(pseudopotential['raw_contraction'])
    if pp_batch.ndim != 3 or pp_batch.shape[1:] != (3, 3):
        raise ValueError(
            'pseudopotential strain must contain contracted density channels')
    pp = cp.asnumpy(pp_batch[0])
    return kinetic + pp


def work_j(gdf_response):
    r'''Return target ``1/2 Tr[P J^(xy)[P]]`` from the first GDF channel.'''
    names = tuple(gdf_response.get('observable_names', ()))
    if names != ('target_j', 'source_fock'):
        raise ValueError('unexpected GDF strain observable ordering')
    raw = np.asarray(gdf_response['raw_contraction'])
    if raw.shape != (2, 3, 3):
        raise ValueError('GDF strain response must have shape (2,3,3)')
    return raw[0]


def work_xc(mf_grad, dm0, image_block_size=None):
    r'''Return the analytic matched-MG XC-only strain and metadata.

    The active multigrid-v2 screened Gaussian-pair tasks and pair meshes are
    differentiated directly.  PBE includes the density, Cartesian-gradient
    and integration-volume terms; r2SCAN additionally includes the analytic
    kinetic-energy-density ``tau`` response.  No displaced Cell is used.
    '''
    return get_vxc_strain_analytic(
        mf_grad, dm=dm0, with_j=False, with_nuc=False,
        image_block_size=image_block_size,
        return_metadata=True)


def work_ovlp(cell, Co, foo):
    r'''Return the target moving-metric term ``-Tr[D_E S^(xy)]``.

    ``D_E=2 C_o F^target_oo C_o^dagger`` is constructed on the device and
    passed directly to GPU4PySCF's complete overlap-strain contraction.
    '''
    Co = cp.asarray(Co)
    foo = cp.asarray(foo)
    if foo.shape != (Co.shape[1], Co.shape[1]):
        raise ValueError('target occupied Fock block has incompatible shape')
    energy_density = 2 * Co @ foo @ Co.conj().T
    return -np.asarray(contract_overlap_strain(cell, energy_density))


def skeleton_work(
        mf_grad, state, gdf_response, hcore_components,
        xc_image_block_size=None, profile=None):
    r'''Assemble the fixed-density target GPU stress skeleton.

    The raw energy-like tensor is

    .. math::

        W_{skel}^{(xy)}=W_{hcore}^{(xy)}+W_J^{(xy)}
        +W_{xc}^{(xy)}+W_{ovlp}^{(xy)}+W_{Ewald}^{(xy)}.

    Target Hartree remains owned by GDF.  Multigrid is called with
    ``with_j=False`` and ``with_nuc=False`` and owns target XC only.
    '''
    mf = mf_grad.base
    dm0 = cp.asarray(state['dm0'])
    pseudopotential = hcore_components['pseudopotential']
    _progress(mf_grad, 'skeleton_hcore_j')
    hcore_stage = stage_start(profile)
    work_hcore1 = work_hcore(mf.cell, dm0, pseudopotential)
    work_j1 = work_j(gdf_response)
    stage_stop(profile, 'skeleton_hcore_j', hcore_stage)

    _progress(mf_grad, 'multigrid_xc')
    xc_stage = stage_start(profile)
    work_xc1, xc_metadata = work_xc(
        mf_grad, dm0, image_block_size=xc_image_block_size)
    stage_stop(profile, 'multigrid_xc', xc_stage)

    _progress(mf_grad, 'skeleton_overlap_ewald')
    overlap_stage = stage_start(profile)
    work_ovlp1 = work_ovlp(mf.cell, state['Co'], state['foo'])
    ewald = ewald_strain(mf.cell)
    work_ewald1 = np.asarray(ewald['raw_derivative'])
    stage_stop(profile, 'skeleton_overlap_ewald', overlap_stage)
    work_skeleton = (
        work_hcore1 + work_j1 + work_xc1
        + work_ovlp1 + work_ewald1)

    mf_grad.work_hcore = work_hcore1
    mf_grad.work_j = work_j1
    mf_grad.work_xc = work_xc1
    mf_grad.work_ovlp = work_ovlp1
    mf_grad.work_ewald = work_ewald1
    mf_grad.work_skeleton = work_skeleton
    mf_grad.target_xc_strain_metadata = xc_metadata
    mf_grad.ewald_strain = ewald
    mf_grad.stress_native_gpu_contracted_primitives = (
        'overlap', 'kinetic', 'multigrid_xc',
        *(f'GDF_PP_{name}' for name in
          pseudopotential['device_contracted_primitives']))
    return work_skeleton


def _progress(mf_grad, label):
    '''Emit one stress substage marker for the external queue monitor.'''
    callback = getattr(mf_grad, 'progress_callback', None)
    if callback is not None:
        callback(f'stress/{label}')


__all__ = [
    'skeleton_work',
    'work_hcore',
    'work_j',
    'work_ovlp',
    'work_xc',
]
