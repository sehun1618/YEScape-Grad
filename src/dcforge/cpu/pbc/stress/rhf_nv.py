'''Gamma-point RHF nonvariational DC-DFT strain response.'''

import numpy as np
from pyscf import lib
from pyscf.lib import logger

from dcforge.cpu.pbc.df.gaussian.strain import (
    kinetic_strain,
    overlap_strain,
)
from dcforge.cpu.pbc.df.gdf.pp_strain import (
    gdf_pseudopotential_strain,
)
from dcforge.cpu.pbc.df.gdf.stress import (
    make_gdf_strain_response,
)
from dcforge.cpu.pbc.energy.rhf_dft import (
    _target_fock_signature,
)
from dcforge.cpu.pbc.grad.rhf_dft import (
    _gen_hf_response,
)
from dcforge.cpu.pbc.grad.rhf_zvec import (
    solve_z_vector,
)
from dcforge.cpu.pbc.stress.ewald import (
    gamma_ewald_exchange_strain,
)


def make_s1_oo_vo(Co, Cv, s1):
    r'''Project the complete strain overlap derivative into OO and VO blocks.

    The raw strain primitive is already the complete bra-plus-ket derivative
    :math:`S^{(xy)}`.  This differs from PySCF's atom-owned AO-row nuclear
    derivative: no extra factor of two or explicit Hermitian completion is
    needed here.

    Args:
        Co : ndarray of shape (nao,nocc)
            Occupied source-RHF coefficients.
        Cv : ndarray of shape (nao,nvir)
            Virtual source-RHF coefficients.
        s1 : ndarray of shape (3,3,nao,nao)
            Full AO overlap derivative with respect to raw strain
            :math:`\varepsilon_{xy}`.

    Returns:
        tuple
            ``s1oo`` with shape ``(3,3,nocc,nocc)`` and ``s1vo`` with shape
            ``(3,3,nvir,nocc)``.
    '''
    Co = np.asarray(Co)
    Cv = np.asarray(Cv)
    s1 = np.asarray(s1)
    nao = Co.shape[0]
    if Cv.shape[0] != nao or s1.shape != (3, 3, nao, nao):
        raise ValueError('strain overlap and MO coefficient shapes differ')
    s1oo = lib.einsum(
        'pi,xypq,qj->xyij', Co.conj(), s1, Co)
    s1vo = lib.einsum(
        'pa,xypq,qi->xyai', Cv.conj(), s1, Co)
    return s1oo, s1vo


def make_z_density(Co, Cv, zvec):
    r'''Return :math:`D_Z=C_v Z C_o^\dagger+\mathrm{h.c.}`.

    ``D_Z`` contains no restricted occupation factor.  With this convention
    the source-Fock contribution is

    .. math::

        -4\operatorname{Re}\langle Z,F^{(xy)}_{vo}\rangle
        =-2\operatorname{Tr}[D_ZF^{(xy)}].
    '''
    Co = np.asarray(Co)
    Cv = np.asarray(Cv)
    zvec = np.asarray(zvec)
    if zvec.shape != (Cv.shape[1], Co.shape[1]):
        raise ValueError('Z-vector has incompatible virtual/occupied shape')
    dz = Cv @ zvec @ Co.conj().T
    return dz + dz.conj().T


def work_fock_hcore(dz, hcore1):
    r'''Return the one-electron part of ``-4 Re <Z,F_HF^(xy)_vo>``.

    ``hcore1`` is the complete kinetic plus GTH pseudopotential matrix
    derivative.  The contraction returns an energy-like raw ``(3,3)``
    strain derivative in Hartree.
    '''
    dz = np.asarray(dz)
    hcore1 = np.asarray(hcore1)
    if hcore1.shape != (3, 3) + dz.shape:
        raise ValueError('Hcore strain and Z-density shapes differ')
    return -2 * lib.einsum('xyij,ji->xy', hcore1, dz).real


def work_fock_gdf(gdf_response):
    r'''Return the GDF J/K part of the source-Fock Z contraction.

    :func:`make_gdf_strain_response` orders its observables as
    ``('target_j', 'source_fock')``.  The second observable already equals

    .. math::

        -2\operatorname{Tr}
        [D_Z(J^{(xy)}[P]-\tfrac12K^{(xy)}[P])].
    '''
    names = tuple(gdf_response.get('observable_names', ()))
    if names != ('target_j', 'source_fock'):
        raise ValueError('unexpected GDF strain observable ordering')
    raw = np.asarray(gdf_response['raw_contraction'])
    if raw.shape != (2, 3, 3):
        raise ValueError('GDF strain response must have shape (2,3,3)')
    return raw[1]


def work_fock_ewald(dz, exchange1):
    r'''Return the Gamma ``exxdiv='ewald'`` source-Fock contribution.

    PySCF adds :math:`K_{G=0}=mSPS` to the exchange matrix, whereas the RHF
    Fock contains ``-K/2``.  Combining this with the global
    ``-2 Tr[D_Z F^(xy)]`` factor leaves

    .. math::

        +\operatorname{Tr}[D_ZK_{G=0}^{(xy)}].
    '''
    dz = np.asarray(dz)
    exchange1 = np.asarray(exchange1)
    if exchange1.shape != (3, 3) + dz.shape:
        raise ValueError('Ewald exchange strain and Z-density shapes differ')
    return lib.einsum('xyij,ji->xy', exchange1, dz).real


def work_ovlp_energy(s1vo, e_occ, zvec):
    r'''Return ``+4 Re <Z,S^(xy)_vo epsilon_o>`` in Hartree.'''
    s1vo = np.asarray(s1vo)
    e_occ = np.asarray(e_occ)
    zvec = np.asarray(zvec)
    if s1vo.shape[:2] != (3, 3) or s1vo.shape[2:] != zvec.shape:
        raise ValueError('VO overlap strain and Z-vector shapes differ')
    if e_occ.shape != (zvec.shape[1],):
        raise ValueError('occupied energies have incompatible shape')
    se_vo = s1vo * e_occ[None, None, None, :]
    return 4 * lib.einsum(
        'ai,xyai->xy', zvec.conj(), se_vo).real


def work_ovlp_response(Co, dz, s1oo, fvind):
    r'''Return the OO moving-metric source-response contribution.

    The physical occupied-space density source is

    .. math::

        P_{oo}^{(xy)}=-2C_oS_{oo}^{(xy)}C_o^\dagger.

    Directly forming all nine ``P_oo`` response matrices is unnecessary.
    Self-adjointness of the RHF J/K kernel moves it onto the one Z density:

    .. math::

        -4\operatorname{Re}
        \langle Z,G[P_{oo}^{(xy)}]_{vo}\rangle
        =4\operatorname{Tr}
        [S_{oo}^{(xy)}C_o^\dagger G[D_Z]C_o].

    Args:
        Co : ndarray of shape (nao,nocc)
            Occupied source-RHF coefficients.
        dz : ndarray of shape (nao,nao)
            Hermitian Z transition density.
        s1oo : ndarray of shape (3,3,nocc,nocc)
            Full strain overlap derivative in the occupied block.
        fvind : callable
            Source-RHF self-consistent J/K response action.
    '''
    Co = np.asarray(Co)
    dz = np.asarray(dz)
    s1oo = np.asarray(s1oo)
    if dz.shape != (Co.shape[0], Co.shape[0]):
        raise ValueError('Z-density and occupied coefficients differ')
    if s1oo.shape != (3, 3, Co.shape[1], Co.shape[1]):
        raise ValueError('OO overlap strain has incompatible shape')
    vz = np.asarray(fvind(dz))
    vzoo = Co.conj().T @ vz @ Co
    return 4 * lib.einsum('xyij,ji->xy', s1oo, vzoo).real


def make_strain_source(Co, Cv, e_occ, s1oo, s1vo, fock1, fvind):
    r'''Build the explicit positive-RHS strain CPHF source.

    This function is a validation oracle for

    .. math::

        V^{(xy)}
        =S_{vo}^{(xy)}\epsilon_o-F_{\mathrm{HF},vo}^{(xy)}
         -G[P_{oo}^{(xy)}]_{vo}.

    Production uses the three adjoint contractions and does not retain nine
    complete Fock derivatives.  Tests may supply such a forward ``fock1`` to
    audit every sign and restricted-spin factor.

    Returns:
        ndarray of shape (3,3,nvir,nocc)
            Positive-RHS source in the same convention as
            :func:`dcforge.cpu.pbc.grad.rhf_zvec.solve_z_vector`.
    '''
    Co = np.asarray(Co)
    Cv = np.asarray(Cv)
    e_occ = np.asarray(e_occ)
    s1oo = np.asarray(s1oo)
    s1vo = np.asarray(s1vo)
    fock1 = np.asarray(fock1)
    nao = Co.shape[0]
    if Cv.shape[0] != nao:
        raise ValueError('occupied and virtual AO dimensions differ')
    if fock1.shape != (3, 3, nao, nao):
        raise ValueError('source Fock strain must have shape (3,3,nao,nao)')

    source = s1vo * e_occ[None, None, None, :]
    source -= lib.einsum(
        'pa,xypq,qi->xyai', Cv.conj(), fock1, Co)
    for x in range(3):
        for y in range(3):
            dm_oo = -2 * Co @ s1oo[x, y] @ Co.conj().T
            v_oo = np.asarray(fvind(dm_oo))
            source[x, y] -= Cv.conj().T @ v_oo @ Co
    return source


def contract_strain_source(zvec, source):
    r'''Return ``+4 Re <Z,V^(xy)>`` for an explicit compact source.'''
    zvec = np.asarray(zvec)
    source = np.asarray(source)
    if source.shape[:2] != (3, 3) or source.shape[2:] != zvec.shape:
        raise ValueError('strain source and Z-vector shapes differ')
    return 4 * lib.einsum(
        'ai,xyai->xy', zvec.conj(), source).real


def make_z_state(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None):
    r'''Build one central target Fock and solve one source-HF Z-vector.

    The returned state is perturbation independent and can therefore be
    shared by all six stress components and by a nuclear gradient evaluated
    at the same immutable central state.  This routine never starts a target
    DFA SCF.

    Args:
        mf_grad : dcforge.cpu.pbc.grad.rhf_dft.Gradients
            Gradient/stress owner whose base is a converged ``RHFDFT``.

    Kwargs:
        mo_energy, mo_coeff, mo_occ : ndarray or None
            Central source-RHF orbitals.  Omitted values are taken from the
            active ``RHFDFT`` object.

    Returns:
        dict
            Central MO blocks, density, target Fock, response action,
            positive-RHS Z-vector and solver diagnostics.
    '''
    mf = mf_grad.base
    mf_hf = mf.mf_hf
    if mf.e_dft is None:
        raise RuntimeError('run the periodic RHFDFT energy kernel first')
    if mo_energy is None:
        mo_energy = mf.mo_energy
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ
    mo_energy = np.asarray(mo_energy)
    mo_coeff = np.asarray(mo_coeff)
    mo_occ = np.asarray(mo_occ)
    occupied = mo_occ > 0
    virtual = mo_occ == 0
    if not np.all((mo_occ == 2) | virtual):
        raise ValueError('periodic RHF stress requires exact 2/0 occupations')
    Co = mo_coeff[:, occupied]
    Cv = mo_coeff[:, virtual]
    dm0 = np.asarray(mf_hf.make_rdm1(mo_coeff, mo_occ))

    signature = _target_fock_signature(mf, dm0)
    if (mf.target_fock is not None and mf.dm_hf is not None
            and mf.target_fock_signature == signature
            and np.array_equal(np.asarray(mf.dm_hf), dm0)):
        target_fock = np.asarray(mf.target_fock)
        target_fock_reused = True
    else:
        target_fock = np.asarray(mf.get_fock(dm=dm0))
        target_fock_reused = False
    fock_mo = mo_coeff.conj().T @ target_fock @ mo_coeff
    fvo = fock_mo[virtual][:, occupied]
    foo = fock_mo[occupied][:, occupied]

    fvind = _gen_hf_response(mf_hf, mo_coeff, mo_occ)
    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)
    zvec, z_info = solve_z_vector(
        mo_energy, mo_occ, Co, Cv, fvo, fvind,
        conv_tol=mf_grad.cphf_conv_tol,
        residual_tol=mf_grad.cphf_residual_tol,
        max_cycle=mf_grad.cphf_max_cycle,
        log=log,
    )
    mf_grad.target_fock_reused = target_fock_reused
    mf_grad.zvec = zvec
    mf_grad.z_rhs_max_abs = z_info['rhs_max_abs']
    mf_grad.z_residual_max_abs = z_info['residual_max_abs']
    mf_grad.z_residual_relative = z_info['residual_relative']
    mf_grad.z_solver = z_info['solver']
    mf_grad.z_dimension = z_info['dimension']
    return {
        'mo_energy': mo_energy,
        'mo_coeff': mo_coeff,
        'mo_occ': mo_occ,
        'Co': Co,
        'Cv': Cv,
        'e_occ': mo_energy[occupied],
        'dm0': dm0,
        'target_fock': target_fock,
        'fvo': fvo,
        'foo': foo,
        'fvind': fvind,
        'zvec': zvec,
        'z_info': z_info,
        'target_fock_reused': target_fock_reused,
    }


def nonvariational_correction(
        mf_grad, state=None, reciprocal_block_size=None,
        pp_reciprocal_mesh=None, pp_block_size=None,
        validate_central=False, hcore_value_tol=1.e-7):
    r'''Assemble the three Gamma RHF nonvariational strain terms.

    The term ledger is

    .. math::

        W_{\mathrm{nv}}^{(xy)}
        =\underbrace{-4\operatorname{Re}
          \langle Z,F_{\mathrm{HF},vo}^{(xy)}\rangle}_{W_{\mathrm{fock}}}
         +\underbrace{4\operatorname{Re}
          \langle Z,S_{vo}^{(xy)}\epsilon_o\rangle}
          _{W_{\mathrm{ovlp-energy}}}
         +\underbrace{-4\operatorname{Re}
          \langle Z,G[P_{oo}^{(xy)}]_{vo}\rangle}
          _{W_{\mathrm{ovlp-response}}}.

    Kinetic, GTH pseudopotential, GDF source/metric J/K and Gamma Madelung
    exchange responses remain separately visible.  No target SCF and no
    perturbation-specific CPHF solve occurs.

    Kwargs:
        state : dict or None
            Result of :func:`make_z_state`.  ``None`` builds it once.
        reciprocal_block_size : int or None
            Bounded GDF source/metric strain block.
        pp_reciprocal_mesh : array_like or None
            Fixed reciprocal mesh for local-part2/nonlocal GTH PP strain.
        pp_block_size : int or None
            Bounded reciprocal GTH PP block.
        validate_central : bool
            Enable central CDERI and Hcore representation parity checks.
        hcore_value_tol : float
            Maximum central analytic/provider Hcore difference.

    Returns:
        ndarray of shape (3,3)
            Energy-like nonvariational derivative in Hartree.
    '''
    if state is None:
        state = make_z_state(mf_grad)
    mf = mf_grad.base
    cell = mf.cell
    Co = np.asarray(state['Co'])
    Cv = np.asarray(state['Cv'])
    dm0 = np.asarray(state['dm0'])
    zvec = np.asarray(state['zvec'])
    dz = make_z_density(Co, Cv, zvec)

    overlap = overlap_strain(cell)
    kinetic = kinetic_strain(cell)
    pseudopotential = gdf_pseudopotential_strain(
        cell, reciprocal_mesh=pp_reciprocal_mesh,
        block_size=pp_block_size,
        source_block_size=reciprocal_block_size)
    hcore_value = (
        np.asarray(kinetic['value'])
        + np.asarray(pseudopotential['value']))
    hcore_value_parity = float(np.max(np.abs(
        hcore_value - np.asarray(mf.mf_hf.get_hcore()))))
    if validate_central and hcore_value_parity > float(hcore_value_tol):
        raise RuntimeError(
            'analytic strain Hcore does not reproduce the source provider: '
            f'{hcore_value_parity:.3e} > {float(hcore_value_tol):.3e}')
    hcore1 = (
        np.asarray(kinetic['raw_derivative'])
        + np.asarray(pseudopotential['raw_derivative']))

    gdf_response = make_gdf_strain_response(
        mf.with_df, dm0, dz,
        reciprocal_block_size=reciprocal_block_size,
        validate_central=validate_central)
    work_hcore = work_fock_hcore(dz, hcore1)
    work_gdf = work_fock_gdf(gdf_response)

    if mf.mf_hf.exxdiv == 'ewald':
        ewald_exchange = gamma_ewald_exchange_strain(cell, dm0)
        work_ewald = work_fock_ewald(
            dz, ewald_exchange['raw_derivative'])
    elif mf.mf_hf.exxdiv is None:
        ewald_exchange = None
        work_ewald = np.zeros((3, 3))
    else:
        raise NotImplementedError(
            "GDF stress supports source exxdiv=None or 'ewald'")
    work_fock = work_hcore + work_gdf + work_ewald

    s1oo, s1vo = make_s1_oo_vo(
        Co, Cv, overlap['raw_derivative'])
    work_energy = work_ovlp_energy(
        s1vo, state['e_occ'], zvec)
    work_response = work_ovlp_response(
        Co, dz, s1oo, state['fvind'])
    work_nvar = work_fock + work_energy + work_response

    mf_grad.s1oo_strain = s1oo
    mf_grad.s1vo_strain = s1vo
    mf_grad.z_density = dz
    mf_grad.work_fock_hcore = work_hcore
    mf_grad.work_fock_gdf = work_gdf
    mf_grad.work_fock_ewald = work_ewald
    mf_grad.work_fock = work_fock
    mf_grad.work_ovlp_energy = work_energy
    mf_grad.work_ovlp_response = work_response
    mf_grad.work_nvar = work_nvar
    mf_grad.gdf_strain_response = gdf_response
    mf_grad.hcore_strain_value_parity_max_abs = hcore_value_parity
    mf_grad.hcore_strain_components = {
        'kinetic': kinetic,
        'pseudopotential': pseudopotential,
    }
    mf_grad.ewald_exchange_strain = ewald_exchange
    return work_nvar


__all__ = [
    'contract_strain_source',
    'make_s1_oo_vo',
    'make_strain_source',
    'make_z_density',
    'make_z_state',
    'nonvariational_correction',
    'work_fock_ewald',
    'work_fock_gdf',
    'work_fock_hcore',
    'work_ovlp_energy',
    'work_ovlp_response',
]
