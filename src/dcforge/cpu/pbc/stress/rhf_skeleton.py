'''Gamma-point fixed-density RHF-DFT stress skeleton terms.'''

import numpy as np
from pyscf import lib

from dcforge.cpu.pbc.df.gaussian.strain import (
    kinetic_strain,
    overlap_strain,
)
from dcforge.cpu.pbc.df.gdf.pp_strain import (
    gdf_pseudopotential_strain,
)
from dcforge.cpu.pbc.stress.ewald import (
    ewald_strain,
)
from dcforge.cpu.pbc.stress.multigrid import (
    get_vxc_strain_finite_difference,
)


def work_hcore(dm0, kinetic, pseudopotential):
    r'''Return ``Tr[P_HF(T^(xy)+V_PP^(xy))]`` in Hartree.

    Args:
        dm0 : ndarray of shape (nao,nao)
            Spin-summed source-RHF density.
        kinetic, pseudopotential : dict
            Analytic strain ledgers returned by ``kinetic_strain`` and
            ``gdf_pseudopotential_strain``.
    '''
    dm0 = np.asarray(dm0)
    hcore1 = (
        np.asarray(kinetic['raw_derivative'])
        + np.asarray(pseudopotential['raw_derivative']))
    if hcore1.shape != (3, 3) + dm0.shape:
        raise ValueError('Hcore strain and source density shapes differ')
    return lib.einsum('xyij,ji->xy', hcore1, dm0).real


def work_j(gdf_response):
    r'''Return target ``1/2 Tr[P_HF J^(xy)[P_HF]]`` in Hartree.

    The first reverse-CDERI observable owns target Hartree.  It contains both
    three-center-source and auxiliary-metric strain and must not be combined
    with a multigrid Hartree term.
    '''
    names = tuple(gdf_response.get('observable_names', ()))
    if names != ('target_j', 'source_fock'):
        raise ValueError('unexpected GDF strain observable ordering')
    raw = np.asarray(gdf_response['raw_contraction'])
    if raw.shape != (2, 3, 3):
        raise ValueError('GDF strain response must have shape (2,3,3)')
    return raw[0]


def work_xc(mf_grad, dm0, step=1.e-4):
    r'''Return the matched pair-MG fixed-density XC strain reference.

    This is deliberately semi-analytic: the current CPU PySCF pair-MG
    integrator has no public PBE stress primitive, so the Cell dependence of
    the XC energy is differentiated by central finite difference.  The
    metadata remains attached to prevent this term from being advertised as
    production analytic.

    Returns:
        tuple
            Energy-like raw ``(3,3)`` XC derivative and owner metadata.
    '''
    return get_vxc_strain_finite_difference(
        mf_grad, dm=np.asarray(dm0), step=step,
        return_metadata=True)


def work_ovlp(Co, foo, s1):
    r'''Return the target moving-metric term ``-Tr[D_E S^(xy)]``.

    The energy-weighted density is

    .. math::

        D_E=2C_oF^{target}_{oo}C_o^\dagger.

    ``s1`` is already the complete strain overlap derivative.  The AO-row
    factor of two used by the nuclear-gradient implementation is therefore
    absent.
    '''
    Co = np.asarray(Co)
    foo = np.asarray(foo)
    s1 = np.asarray(s1)
    if foo.shape != (Co.shape[1], Co.shape[1]):
        raise ValueError('target occupied Fock block has incompatible shape')
    if s1.shape != (3, 3, Co.shape[0], Co.shape[0]):
        raise ValueError('overlap strain and occupied coefficients differ')
    energy_density = 2 * Co @ foo @ Co.conj().T
    return -lib.einsum(
        'xyij,ji->xy', s1, energy_density).real


def skeleton_work(
        mf_grad, state, gdf_response, hcore_components=None,
        xc_step=1.e-4, pp_reciprocal_mesh=None, pp_block_size=None,
        source_block_size=None):
    r'''Assemble the fixed-density target PBE stress skeleton.

    The returned raw energy-like derivative is

    .. math::

        W_{skel}^{(xy)}
        =W_{hcore}^{(xy)}+W_J^{(xy)}+W_{xc}^{(xy)}
         +W_{ovlp}^{(xy)}+W_{Ewald}^{(xy)}.

    Nuclear Ewald is included here because stress does not inherit PySCF's
    nuclear-gradient driver.  The target Hartree owner is GDF and the target
    XC owner is pair multigrid with ``with_j=False`` and ``with_nuc=False``.

    Args:
        mf_grad : dcforge.cpu.pbc.grad.rhf_dft.Gradients
            Active owner of the central fixed-density target state.
        state : dict
            Perturbation-independent state from
            :func:`dcforge.cpu.pbc.stress.rhf_nv.make_z_state`.
        gdf_response : dict
            Shared reverse-CDERI result used by target J and source Fock.

    Kwargs:
        hcore_components : dict or None
            Shared ``kinetic`` and ``pseudopotential`` ledgers.  They are
            evaluated only when omitted.
        xc_step : float
            Pair-MG XC Cell finite-difference displacement.
        pp_reciprocal_mesh, pp_block_size, source_block_size
            Numerical controls forwarded only when Hcore components are not
            already available.

    Returns:
        ndarray of shape (3,3)
            Energy-like target skeleton derivative in Hartree.
    '''
    mf = mf_grad.base
    dm0 = np.asarray(state['dm0'])
    if hcore_components is None:
        hcore_components = {
            'kinetic': kinetic_strain(mf.cell),
            'pseudopotential': gdf_pseudopotential_strain(
                mf.cell, reciprocal_mesh=pp_reciprocal_mesh,
                block_size=pp_block_size,
                source_block_size=source_block_size),
        }
    kinetic = hcore_components['kinetic']
    pseudopotential = hcore_components['pseudopotential']
    overlap = overlap_strain(mf.cell)
    ewald = ewald_strain(mf.cell)

    work_hcore1 = work_hcore(dm0, kinetic, pseudopotential)
    work_j1 = work_j(gdf_response)
    work_xc1, xc_metadata = work_xc(
        mf_grad, dm0, step=xc_step)
    work_ovlp1 = work_ovlp(
        state['Co'], state['foo'], overlap['raw_derivative'])
    work_ewald1 = np.asarray(ewald['raw_derivative'])
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
    return work_skeleton


__all__ = [
    'skeleton_work',
    'work_hcore',
    'work_j',
    'work_ovlp',
    'work_xc',
]
