'''Gamma-point FFTDF/GDF analytic DFA@RHF nuclear gradients.'''

import time

import numpy as np
from pyscf.lib import logger
from pyscf.pbc.dft import multigrid
from pyscf.pbc.grad import rks

from dcforge.cpu.pbc.df.fft import FFTDF
from dcforge.cpu.pbc.df.gaussian.hcore import (
    contract_hcore_deriv as contract_gaussian_hcore_deriv,
)
from dcforge.cpu.pbc.df.gaussian.hcore import (
    get_hcore_deriv as get_gaussian_hcore_deriv,
)
from dcforge.cpu.pbc.df.gaussian.jk import (
    grad_ewald_exchange,
    grad_jk,
)
from dcforge.cpu.pbc.df.gdf import GDF
from dcforge.cpu.pbc.df.gdf import response as gdf_response
from dcforge.cpu.pbc.energy.rhf_dft import (
    RHFDFT,
    _check_input,
    _target_fock_is_current,
)
from dcforge.cpu.pbc.grad.rhf_nv import (
    grad_fock_direct,
    nonvariational_correction,
)
from dcforge.cpu.pbc.grad.rhf_zvec import solve_z_vector
from dcforge.cpu.pbc.grad.skeleton import (
    get_hcore_deriv,
    get_ovlp_deriv,
    skeleton_gradient,
)
from dcforge.tools.pbc_hf_state import assert_pbc_source_state


def grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None,
              atmlst=None):
    r'''Evaluate the electronic Gamma-point DFA@RHF gradient.

    The term ledger follows the molecular implementation,

    ``de = de_hcore + de_veff + de_ovlp``
    ``     + de_fock + de_ovlp_energy + de_ovlp_response``.

    The inherited PySCF kernel adds the Ewald nuclear derivative afterward.
    Every term is an energy derivative ``dE/dR`` in Ha/Bohr.

    Args:
        mf_grad : dcforge.cpu.pbc.grad.rhf_dft.Gradients
            Active Gamma-point periodic gradient object.

    Kwargs:
        mo_energy, mo_coeff, mo_occ : ndarray or None
            Canonical source-RHF quantities.  The converged values are used
            when omitted.
        atmlst : sequence of int or None
            Nuclear centers included in the returned gradient.

    Returns:
        ndarray of shape (len(atmlst), 3)
            Electronic target-DFA@RHF gradient in Ha/Bohr.
    '''
    profile = mf_grad.timing_profile
    if profile:
        electronic_start = time.perf_counter()
        timings = {}
    mf = mf_grad.base
    mf_hf = mf.mf_hf
    cell = mf_grad.cell
    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)
    _check_input(mf)
    assert_pbc_source_state(mf)
    if mf.e_dft is None:
        raise RuntimeError('run the periodic RHFDFT energy kernel first')
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    if mo_occ is None:    mo_occ = mf.mo_occ
    if atmlst is None:    atmlst = range(cell.natm)
    atmlst = list(atmlst)

    mo_energy = np.asarray(mo_energy)
    mo_coeff = np.asarray(mo_coeff)
    mo_occ = np.asarray(mo_occ)
    so = mo_occ > 0
    sv = mo_occ == 0
    if not np.all((mo_occ == 2) | sv):
        raise ValueError('periodic RHF-DFT requires exact 2/0 occupations')
    Co = mo_coeff[:, so]
    Cv = mo_coeff[:, sv]
    dm0 = np.asarray(mf_hf.make_rdm1(mo_coeff, mo_occ))

    if profile:
        start = time.perf_counter()
    if _target_fock_is_current(mf, dm0):
        fock = np.asarray(mf.target_fock)
        mf_grad.target_fock_reused = True
    else:
        fock = np.asarray(mf.get_fock(dm=dm0))
        mf_grad.target_fock_reused = False
    fock_mo = mo_coeff.conj().T @ fock @ mo_coeff
    fvo = fock_mo[sv][:, so]
    foo = fock_mo[so][:, so]
    if profile:
        timings['target_fock_and_mo_s'] = time.perf_counter() - start

    if profile:
        start = time.perf_counter()
    fvind = _gen_hf_response(mf_hf, mo_coeff, mo_occ)
    zvec, z_info = solve_z_vector(
        mo_energy, mo_occ, Co, Cv, fvo, fvind,
        conv_tol=mf_grad.cphf_conv_tol,
        residual_tol=mf_grad.cphf_residual_tol,
        max_cycle=mf_grad.cphf_max_cycle,
        log=log,
    )
    mf_grad.zvec = zvec
    mf_grad.z_rhs_max_abs = z_info['rhs_max_abs']
    mf_grad.z_residual_max_abs = z_info['residual_max_abs']
    mf_grad.z_residual_relative = z_info['residual_relative']
    mf_grad.z_solver = z_info['solver']
    mf_grad.z_dimension = z_info['dimension']
    dz = Cv @ zvec @ Co.conj().T
    dz = dz + dz.conj().T
    if profile:
        timings['z_vector_s'] = time.perf_counter() - start

    kpts = np.zeros((1, 3))
    if profile:
        start = time.perf_counter()
    if isinstance(mf.with_df, GDF):
        if mf_grad.hcore_deriv_method == 'contracted':
            hcore_contractions, hcore_components = (
                contract_gaussian_hcore_deriv(
                    cell, kpts, np.asarray((dm0, dz)), atmlst))
            de_hcore = hcore_contractions[0]
            de_fock_hcore = -2 * hcore_contractions[1]
            mf_grad.hcore_deriv_components = hcore_components
            h1ao = None
        else:
            h1ao = get_gaussian_hcore_deriv(cell, kpts)
            de_hcore = de_fock_hcore = None
            mf_grad.hcore_deriv_components = None
    else:
        h1ao = get_hcore_deriv(cell, kpts)
        de_hcore = de_fock_hcore = None
        mf_grad.hcore_deriv_components = None
    s1 = get_ovlp_deriv(cell, kpts)
    if profile:
        timings['hcore_and_overlap_s'] = time.perf_counter() - start

    if isinstance(mf.with_df, GDF):
        if profile:
            start = time.perf_counter()
        cderi, jk1, jk1_source, jk1_metric = _gaussian_jk_response(
            mf_grad, dm0, dz, atmlst)
        if jk1 is None:
            vj1 = vk1 = None
            vj1_source = vk1_source = None
            vj1_metric = vk1_metric = None
        else:
            vj1, vk1 = jk1
            vj1_source, vk1_source = jk1_source
            vj1_metric, vk1_metric = jk1_metric
        s1_full = _full_overlap_derivative(cell, s1[0])
        shape = (cell.natm, 3, cell.nao_nr(), cell.nao_nr())
        vk1_ewald = np.zeros(shape, dtype=dm0.dtype)
        if mf_hf.exxdiv == 'ewald':
            vk1_ewald = grad_ewald_exchange(cell, dm0, s1_full)
            if vk1 is not None:
                vk1 += vk1_ewald
        elif mf_hf.exxdiv is not None:
            raise NotImplementedError(
                'GDF gradient supports exxdiv=None or ewald')
        mf_grad.cderi = cderi
        mf_grad.vj1 = vj1
        mf_grad.vk1 = vk1
        mf_grad.vj1_source = vj1_source
        mf_grad.vk1_source = vk1_source
        mf_grad.vj1_metric = vj1_metric
        mf_grad.vk1_metric = vk1_metric
        mf_grad.vk1_ewald = vk1_ewald
        if profile:
            timings['gaussian_jk_response_s'] = time.perf_counter() - start

    if profile:
        start = time.perf_counter()
    de_fock = grad_fock_direct(
        mf_grad, Co, Cv, zvec, dm0, h1ao, atmlst,
        de_hcore=de_fock_hcore)
    if profile:
        timings['fock_contraction_s'] = time.perf_counter() - start

    if profile:
        start = time.perf_counter()
    de_skel = skeleton_gradient(
        mf_grad, dm0, foo, Co, h1ao, s1, atmlst,
        de_hcore=de_hcore)
    if profile:
        timings['skeleton_contraction_s'] = time.perf_counter() - start

    if profile:
        start = time.perf_counter()
    de_nvar = nonvariational_correction(
        mf_grad, Co, Cv, mo_energy, zvec, s1[0],
        de_fock, fvind, atmlst)
    if profile:
        timings['nonvariational_contraction_s'] = time.perf_counter() - start
        timings['electronic_pipeline_s'] = (
            time.perf_counter() - electronic_start)
        mf_grad.electronic_timings_s = timings
    return de_skel + de_nvar


def _gen_hf_response(mf_hf, mo_coeff, mo_occ):
    '''Return the source-RHF response action used by the Z equation.

    PySCF GDF normally factorizes a Hermitian density before its exchange
    contraction.  A CPHF trial density is Hermitian but indefinite, and the
    PySCF 2.14.0 Gamma path can produce an empty imaginary MO factor and call
    ``max`` on it.  ``force_dm_kbuild`` selects PySCF's mathematically
    equivalent density-matrix contraction for these response calls only.
    The provider setting is restored immediately, so central SCF J/K builds
    retain the user's original ordering.
    '''
    response = mf_hf.gen_response(
        mo_coeff=mo_coeff, mo_occ=mo_occ, hermi=1)
    with_df = mf_hf.with_df
    if not isinstance(with_df, GDF):
        return response

    def response_dm(dm1):
        force_dm_kbuild = with_df.force_dm_kbuild
        try:
            with_df.force_dm_kbuild = True
            return response(dm1)
        finally:
            with_df.force_dm_kbuild = force_dm_kbuild

    return response_dm


class Gradients(rks.Gradients):
    '''Analytic nuclear gradient of Gamma-point ``E_DFA[P_RHF]``.'''

    _keys = rks.Gradients._keys | {
        'cell',
        'cphf_conv_tol', 'cphf_residual_tol', 'cphf_max_cycle',
        'cderi_response_storage', 'cderi_response_blksize',
        'cderi_response_blksize_used',
        'cderi_value_validation',
        'fock_deriv_method', 'grid_response', 'target_fock_reused',
        'hcore_deriv_method', 'xc_deriv_backend',
        'de_j_xc_multigrid',
        'multigrid_xc_deriv_method',
        'multigrid_xc_deriv_method_used',
        'timing_profile', 'electronic_timings_s', 'gradient_timings_s',
    }

    def __init__(self, mf):
        if not isinstance(mf, RHFDFT):
            raise TypeError('Gradients requires dcforge CPU PBC RHFDFT')
        rks.Gradients.__init__(self, mf)
        self.cphf_conv_tol = 1e-9
        self.cphf_residual_tol = 1e-10
        self.cphf_max_cycle = 200
        self.cderi_response_storage = 'contracted'
        self.cderi_response_blksize = None
        self.cderi_response_blksize_used = None
        self.cderi_value_validation = False
        self.fock_deriv_method = 'direct'
        self.hcore_deriv_method = 'contracted'
        self.grid_response = True
        self.target_fock_reused = False
        self.xc_deriv_backend = None
        self.de_j_xc_multigrid = None
        self.multigrid_xc_deriv_method = 'single_pass'
        self.multigrid_xc_deriv_method_used = None
        # This opt-in diagnostic records pipeline wall times without changing
        # the mathematical term ledger.  It is for validation and profiling,
        # so default production gradients avoid even timer-call overhead.
        self.timing_profile = False
        self.electronic_timings_s = None
        self.gradient_timings_s = None

    def dump_flags(self, verbose=None):
        rks.Gradients.dump_flags(self, verbose)
        logger.info(self, 'cphf_conv_tol = %g', self.cphf_conv_tol)
        logger.info(self, 'cphf_residual_tol = %g', self.cphf_residual_tol)
        logger.info(self, 'cphf_max_cycle = %d', self.cphf_max_cycle)
        logger.info(
            self, 'cderi_response_storage = %s',
            self.cderi_response_storage)
        logger.info(
            self, 'cderi_response_blksize = %s',
            ('auto' if self.cderi_response_blksize is None
             else self.cderi_response_blksize))
        logger.info(self, 'fock_deriv_method = %s', self.fock_deriv_method)
        logger.info(self, 'hcore_deriv_method = %s', self.hcore_deriv_method)
        logger.info(self, 'timing_profile = %s', self.timing_profile)
        logger.info(
            self, 'multigrid_xc_deriv_method = %s',
            self.multigrid_xc_deriv_method)
        ni = self.base._numint
        if isinstance(ni, multigrid.MultiGridNumInt2):
            logger.info(self, 'multigrid_ntasks = %d', ni.ntasks)
            logger.info(self, 'multigrid_ke_ratio = %g', ni.ke_ratio)
            logger.info(self, 'multigrid_rel_cutoff = %g', ni.rel_cutoff)
        logger.info(
            self, 'target XC numerical integrator = %s',
            type(self.base._numint).__name__)
        logger.info(
            self, 'target Hartree provider = %s',
            ('multigrid' if self.base.multigrid_with_j
             else type(self.base.with_df).__name__))
        return self

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None,
               atmlst=None):
        if not isinstance(self.base.with_df, (FFTDF, GDF)):
            raise NotImplementedError('unsupported PBC gradient provider')
        if isinstance(self.base.with_df, GDF):
            storage = str(self.cderi_response_storage).lower()
            if storage not in ('contracted', 'atom_stream', 'full'):
                raise ValueError(
                    "cderi_response_storage must be 'contracted', "
                    "'atom_stream' or 'full'")
            if (self.cderi_response_blksize is not None
                    and int(self.cderi_response_blksize) < 1):
                raise ValueError('cderi_response_blksize must be positive')
        if self.fock_deriv_method != 'direct':
            raise NotImplementedError(
                "periodic gradients currently support fock_deriv_method='direct'"
            )
        if self.hcore_deriv_method not in ('contracted', 'full'):
            raise ValueError(
                "hcore_deriv_method must be 'contracted' or 'full'")
        if self.timing_profile:
            total_start = rks_start = time.perf_counter()
            self.electronic_timings_s = None
        de = rks.Gradients.kernel(
            self, mo_energy, mo_coeff, mo_occ, atmlst)
        if self.timing_profile:
            rks_seconds = time.perf_counter() - rks_start
            nuclear_start = time.perf_counter()
        self.de_nuc = self.grad_nuc(atmlst=self.atmlst)
        if self.timing_profile:
            nuclear_seconds = time.perf_counter() - nuclear_start
        self.de_skel = self.de_skel_elec + self.de_nuc
        self.de_terms = (
            self.de_hcore + self.de_veff + self.de_extra + self.de_ovlp
            + self.de_nuc + self.de_fock + self.de_ovlp_energy
            + self.de_ovlp_response
        )
        if self.timing_profile:
            timings = dict(self.electronic_timings_s or {})
            timings.update({
                'rks_kernel_s': rks_seconds,
                'nuclear_gradient_s': nuclear_seconds,
                'gradient_total_s': time.perf_counter() - total_start,
            })
            self.gradient_timings_s = timings
        return de

    grad_elec = grad_elec

    def get_stress(self):
        '''Fail closed: periodic stress is not supported.'''
        raise NotImplementedError(
            'periodic stress is outside the FFTDF/GDF migration scope')

    def as_scanner(self):
        raise NotImplementedError('periodic HF-DFT scanner is not implemented')


Grad = Gradients


def _gaussian_jk_response(mf_grad, dm0, dz, atmlst):
    r'''Return contracted production or matrix-reference GDF response.

    ``full`` materializes :math:`C^\lambda` and is retained as the numerical
    reference.  ``atom_stream`` stores only J/K matrix derivatives.  Production
    ``contracted`` returns resolved source/metric gradient scalars without
    retaining :math:`C^\lambda`, :math:`J^\lambda`, or :math:`K^\lambda`.
    '''
    with_df = mf_grad.base.with_df
    storage = str(mf_grad.cderi_response_storage).lower()
    if storage not in ('contracted', 'atom_stream', 'full'):
        raise ValueError(
            "cderi_response_storage must be 'contracted', 'atom_stream' "
            "or 'full'")

    full_response = gdf_response.make_cderi_response
    stream_response = gdf_response.make_jk_response
    contracted_response = gdf_response.make_gradient_response

    if storage == 'contracted':
        terms = contracted_response(
            with_df, dm0, dz, atmlst=atmlst,
            blksize=mf_grad.cderi_response_blksize,
            validate_central=mf_grad.cderi_value_validation)
        mf_grad._de_j_source_contracted = terms['de_j_source']
        mf_grad._de_j_metric_contracted = terms['de_j_metric']
        mf_grad._de_fock_source_contracted = terms['de_fock_source']
        mf_grad._de_fock_metric_contracted = terms['de_fock_metric']
        mf_grad.cderi1 = None
        mf_grad.cderi1_source = None
        mf_grad.cderi1_metric = None
        cderi = jk1 = jk1_source = jk1_metric = None
    elif storage == 'atom_stream':
        result = stream_response(
            with_df, dm0, atmlst=atmlst,
            blksize=mf_grad.cderi_response_blksize)
        cderi, jk1, jk1_source, jk1_metric = result
        mf_grad.cderi1 = None
        mf_grad.cderi1_source = None
        mf_grad.cderi1_metric = None
    else:
        cderi, cderi1, cderi1_source, cderi1_metric = full_response(
            with_df)
        jk1_source = grad_jk(cderi, cderi1_source, dm0)
        jk1_metric = grad_jk(cderi, cderi1_metric, dm0)
        jk1 = (
            jk1_source[0] + jk1_metric[0],
            jk1_source[1] + jk1_metric[1],
        )
        mf_grad.cderi1 = cderi1
        mf_grad.cderi1_source = cderi1_source
        mf_grad.cderi1_metric = cderi1_metric

    mf_grad.cderi_response_storage_used = storage
    mf_grad.cderi_response_blksize_used = int(
        with_df.cderi_response_blksize)
    mf_grad.cderi_response_batch_bytes = int(
        with_df.cderi_response_batch_bytes)
    return cderi, jk1, jk1_source, jk1_metric


def _full_overlap_derivative(cell, s1):
    '''Expand PySCF AO-row overlap derivatives for every atom.'''
    nao = cell.nao_nr()
    s1_full = np.zeros((cell.natm, 3, nao, nao), dtype=s1.dtype)
    aoslices = cell.aoslice_by_atom()
    for ia in range(cell.natm):
        p0, p1 = aoslices[ia, 2:]
        s1_full[ia, :, p0:p1] += s1[:, p0:p1]
        s1_full[ia, :, :, p0:p1] += s1[:, p0:p1].transpose(
            0, 2, 1).conj()
    return s1_full
