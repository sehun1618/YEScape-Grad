'''Sampled-k FFTDF/GDF/RSDF semilocal-DFA@KRHF nuclear gradients.'''

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.grad import krks

from dcforge.cpu.pbc.df.fft import FFTDF
from dcforge.cpu.pbc.df.gaussian.hcore import (
    get_hcore_deriv as get_gaussian_hcore_deriv,
)
from dcforge.cpu.pbc.df.gaussian.jk import (
    grad_ewald_exchange_kpts,
    grad_jk_kpts,
)
from dcforge.cpu.pbc.df.gdf import (
    GDF,
)
from dcforge.cpu.pbc.df.gdf import (
    make_cderi_response_kpts as make_gdf_cderi_response_kpts,
)
from dcforge.cpu.pbc.df.rsdf import (
    RSDF,
)
from dcforge.cpu.pbc.df.rsdf import (
    make_cderi_response_kpts as make_rsdf_cderi_response_kpts,
)
from dcforge.cpu.pbc.energy.krhf_dft import KRHFDFT, _check_input
from dcforge.cpu.pbc.grad.krhf_nv import (
    grad_fock_direct,
    make_s1_ao,
    nonvariational_correction,
)
from dcforge.cpu.pbc.grad.krhf_skeleton import skeleton_gradient
from dcforge.cpu.pbc.grad.krhf_zvec import (
    gen_hf_response,
    make_fvo,
    make_mo_partitions,
    make_z_dm,
    solve_z_vector,
    solve_z_vector_matrix_free,
)
from dcforge.cpu.pbc.grad.skeleton import (
    get_hcore_deriv,
    get_ovlp_deriv,
)
from dcforge.cpu.pbc.kpts import IMAGINARY_RESIDUAL_TOL


def grad_elec(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None,
              atmlst=None):
    r'''Evaluate the electronic full-BZ semilocal-DFA@KRHF gradient.

    The term ledger mirrors the current Gamma implementation,

    ``de_elec = de_hcore + de_veff + de_ovlp``
    ``          + de_fock + de_ovlp_energy + de_ovlp_response``.

    Every complete k-resolved contraction carries the equal full-BZ factor
    ``1/nkpts``.  Returned values are analytic energy derivatives ``dE/dR``
    in Ha/Bohr; force is their negative.

    Args:
        mf_grad : dcforge.cpu.pbc.grad.krhf_dft.Gradients
            Active sampled-k FFTDF gradient object.

    Kwargs:
        mo_energy, mo_coeff, mo_occ : ndarray or None
            Canonical source-KRHF quantities.  Converged values are used when
            omitted.
        atmlst : sequence of int or None
            Nuclear centers included in the returned gradient.

    Returns:
        ndarray of shape (len(atmlst),3)
            Electronic sampled-k DFA@KRHF gradient in Ha/Bohr.
    '''
    mf = mf_grad.base
    mf_hf = mf.mf_hf
    cell = mf_grad.cell
    _check_input(mf)
    if mf.e_dft is None:
        raise RuntimeError('run the sampled-k KRHFDFT energy kernel first')
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    if mo_occ is None:    mo_occ = mf.mo_occ
    if atmlst is None:    atmlst = range(cell.natm)
    atmlst = list(atmlst)

    mo_energy = np.asarray(mo_energy)
    mo_coeff = np.asarray(mo_coeff)
    mo_occ = np.asarray(mo_occ)
    Co, Cv, _ = make_mo_partitions(mo_energy, mo_coeff, mo_occ)
    dm0 = np.asarray(mf_hf.make_rdm1(mo_coeff, mo_occ))
    fvo = make_fvo(mf, Co, Cv)
    foo = []
    for Co_k, fock_k in zip(Co, mf.target_fock, strict=True):
        cof = lib.dot(Co_k.conj().T, fock_k)
        foo.append(lib.dot(cof, Co_k))
    foo = tuple(foo)

    fvind = gen_hf_response(mf_hf)
    dimension = 2 * sum(np.asarray(block).size for block in fvo)
    zvec_method = mf_grad.zvec_method
    if zvec_method == 'auto':
        zvec_method = (
            'dense' if dimension <= mf_grad.dense_oracle_size
            else 'matrix_free')
    if zvec_method == 'dense':
        zvec, z_info = solve_z_vector(
            mo_energy, mo_occ, Co, Cv, fvo, fvind,
            residual_tol=mf_grad.cphf_residual_tol,
            dense_oracle_size=mf_grad.dense_oracle_size,
            equation='adjoint', kpts_weights=mf.kpts_weights)
    else:
        zvec, z_info = solve_z_vector_matrix_free(
            mo_energy, mo_occ, Co, Cv, fvo, fvind,
            conv_tol=mf_grad.cphf_conv_tol,
            residual_tol=mf_grad.cphf_residual_tol,
            max_cycle=mf_grad.cphf_max_cycle,
            kpts_weights=mf.kpts_weights,
            self_adjoint_tol=mf_grad.self_adjoint_tol,
            restart=mf_grad.gmres_restart)
    mf_grad.zvec_method_used = zvec_method
    mf_grad.zvec = zvec
    mf_grad.z_rhs_max_abs = z_info['rhs_max_abs']
    mf_grad.z_residual_max_abs = z_info['residual_max_abs']
    mf_grad.z_residual_relative = z_info['residual_relative']
    mf_grad.z_solver = z_info['solver']
    mf_grad.z_dimension = z_info['dimension']
    mf_grad.z_condition_number = z_info['condition_number']
    mf_grad.z_matrix_asymmetry_max_abs = (
        z_info['matrix_asymmetry_max_abs'])
    mf_grad.z_direct_adjoint_max_abs = z_info['direct_adjoint_max_abs']
    mf_grad.z_self_adjoint_probe_error = z_info.get(
        'self_adjoint_probe_error')
    mf_grad.z_iterations = z_info.get('iterations')
    mf_grad.z_operator_calls = z_info.get('operator_calls')
    mf_grad.z_preconditioner_calls = z_info.get('preconditioner_calls')
    mf_grad.z_dense_matrix_bytes = z_info.get('dense_matrix_bytes')
    mf_grad.z_workspace_estimate_bytes = z_info.get(
        'workspace_estimate_bytes')
    mf_grad.z_wall_seconds = z_info.get('wall_seconds')
    mf_grad.dz = make_z_dm(Co, Cv, zvec)

    if isinstance(mf.with_df, (GDF, RSDF)):
        h1ao = get_gaussian_hcore_deriv(cell, mf.kpts)
    else:
        h1ao = get_hcore_deriv(cell, mf.kpts)
    s1 = get_ovlp_deriv(cell, mf.kpts)
    if isinstance(mf.with_df, (GDF, RSDF)):
        _make_gaussian_jk_response(mf_grad, dm0, s1)
    de_fock = grad_fock_direct(
        mf_grad, Co, Cv, zvec, dm0, h1ao, atmlst)
    de_skel = skeleton_gradient(
        mf_grad, dm0, foo, Co, h1ao, s1, atmlst)
    de_nvar = nonvariational_correction(
        mf_grad, Co, Cv, mo_energy, mo_occ, zvec, s1,
        de_fock, fvind, atmlst)
    return de_skel + de_nvar


def _make_gaussian_jk_response(mf_grad, dm0, s1):
    r'''Build resolved sampled-k GDF/RSDF derivatives for the term ledger.

    The canonical pair map represents

    ``C_ij = L(q)^-1 B_ij``, ``q=k_j-k_i``.

    Source and auxiliary-metric derivatives remain separate through the J/K
    contraction.  Their sum is the complete fixed-density Gaussian two-
    electron derivative.  PySCF's ``exxdiv='ewald'`` correction is an overlap
    response rather than a CDERI response and is therefore recorded as a
    third, separate exchange component.

    Args:
        mf_grad : dcforge.cpu.pbc.grad.krhf_dft.Gradients
            Active GDF gradient object.
        dm0 : ndarray of shape (nkpts,nao,nao)
            Converged closed-shell KRHF density.
        s1 : ndarray of shape (nkpts,3,nao,nao)
            PySCF AO-row overlap derivative.
    '''
    mf = mf_grad.base
    if isinstance(mf.with_df, RSDF):
        make_response = make_rsdf_cderi_response_kpts
    else:
        make_response = make_gdf_cderi_response_kpts
    cderi, cderi1, cderi1_source, cderi1_metric = make_response(
        mf.with_df)
    vj1_source, vk1_source = grad_jk_kpts(
        cderi, cderi1_source, dm0)
    vj1_metric, vk1_metric = grad_jk_kpts(
        cderi, cderi1_metric, dm0)
    vj1 = vj1_source + vj1_metric
    vk1 = vk1_source + vk1_metric

    vk1_ewald = np.zeros_like(vk1)
    if mf.mf_hf.exxdiv == 'ewald':
        vk1_ewald = grad_ewald_exchange_kpts(
            mf.cell, mf.kpts, dm0, make_s1_ao(mf.cell, s1))
        vk1 = np.asarray(vk1, dtype=np.result_type(vk1, vk1_ewald))
        vk1 += vk1_ewald
    elif mf.mf_hf.exxdiv is not None:
        raise NotImplementedError(
            'sampled-k Gaussian gradient supports exxdiv=None or ewald')

    mf_grad.cderi = cderi
    mf_grad.cderi1 = cderi1
    mf_grad.cderi1_source = cderi1_source
    mf_grad.cderi1_metric = cderi1_metric
    mf_grad.vj1 = vj1
    mf_grad.vk1 = vk1
    mf_grad.vj1_source = vj1_source
    mf_grad.vk1_source = vk1_source
    mf_grad.vj1_metric = vj1_metric
    mf_grad.vk1_metric = vk1_metric
    mf_grad.vk1_ewald = vk1_ewald


class Gradients(krks.Gradients):
    '''Analytic fixed-cell gradient of sampled-k ``E_DFA[P_KRHF]``.'''

    _keys = krks.Gradients._keys | {
        'cphf_conv_tol', 'cphf_residual_tol', 'cphf_max_cycle',
        'dense_oracle_size', 'zvec_method', 'zvec_method_used',
        'self_adjoint_tol', 'gmres_restart', 'fock_deriv_method',
        'grid_response', 'target_fock_reused', 'xc_deriv_backend',
        'zvec', 'z_rhs_max_abs', 'z_residual_max_abs',
        'z_residual_relative', 'z_solver', 'z_dimension',
        'z_condition_number', 'z_matrix_asymmetry_max_abs',
        'z_direct_adjoint_max_abs', 'imaginary_residual_tol',
        'z_self_adjoint_probe_error', 'z_iterations',
        'z_operator_calls', 'z_preconditioner_calls',
        'z_dense_matrix_bytes', 'z_workspace_estimate_bytes',
        'z_wall_seconds',
        'cderi', 'cderi1', 'cderi1_source', 'cderi1_metric',
        'vj1', 'vk1', 'vj1_source', 'vk1_source',
        'vj1_metric', 'vk1_metric', 'vk1_ewald',
        'de_j_source', 'de_j_metric', 'de_fock_source',
        'de_fock_metric', 'de_fock_ewald',
        'term_imaginary_residuals',
        'term_imaginary_residual_max_abs',
        'term_raw_imaginary_components',
        'term_raw_imaginary_component_max_abs', 'translation_residual',
        'translation_residual_max_abs', 'term_recomposition_max_abs',
    }

    def __init__(self, mf):
        if not isinstance(mf, KRHFDFT):
            raise TypeError('Gradients requires sampled-k CPU PBC KRHFDFT')
        krks.Gradients.__init__(self, mf)
        self.cphf_conv_tol = 1e-9
        self.cphf_residual_tol = 1e-8
        self.cphf_max_cycle = 200
        self.dense_oracle_size = 512
        self.zvec_method = 'auto'
        self.zvec_method_used = None
        self.self_adjoint_tol = 1e-10
        self.gmres_restart = 80
        self.fock_deriv_method = 'direct'
        # The uniform periodic quadrature is fixed in the Cell and has no
        # atom-centered grid-motion term.  Keep the familiar option visible
        # to state that the complete target-XC derivative is nevertheless
        # required and evaluated.
        self.grid_response = True
        self.target_fock_reused = False
        self.xc_deriv_backend = None
        self.zvec = None
        self.z_rhs_max_abs = None
        self.z_residual_max_abs = None
        self.z_residual_relative = None
        self.z_solver = None
        self.z_dimension = None
        self.z_condition_number = None
        self.z_matrix_asymmetry_max_abs = None
        self.z_direct_adjoint_max_abs = None
        self.z_self_adjoint_probe_error = None
        self.z_iterations = None
        self.z_operator_calls = None
        self.z_preconditioner_calls = None
        self.z_dense_matrix_bytes = None
        self.z_workspace_estimate_bytes = None
        self.z_wall_seconds = None
        self.cderi = None
        self.cderi1 = None
        self.cderi1_source = None
        self.cderi1_metric = None
        self.vj1 = None
        self.vk1 = None
        self.vj1_source = None
        self.vk1_source = None
        self.vj1_metric = None
        self.vk1_metric = None
        self.vk1_ewald = None
        self.imaginary_residual_tol = IMAGINARY_RESIDUAL_TOL
        self.term_imaginary_residuals = {}
        self.term_imaginary_residual_max_abs = 0.0
        self.term_raw_imaginary_components = {}
        self.term_raw_imaginary_component_max_abs = 0.0
        self.translation_residual = None
        self.translation_residual_max_abs = None
        self.term_recomposition_max_abs = None

    def dump_flags(self, verbose=None):
        krks.Gradients.dump_flags(self, verbose)
        logger.info(self, 'cphf_conv_tol = %g', self.cphf_conv_tol)
        logger.info(self, 'cphf_residual_tol = %g', self.cphf_residual_tol)
        logger.info(self, 'cphf_max_cycle = %d', self.cphf_max_cycle)
        logger.info(self, 'dense_oracle_size = %d', self.dense_oracle_size)
        logger.info(self, 'zvec_method = %s', self.zvec_method)
        logger.info(self, 'self_adjoint_tol = %g', self.self_adjoint_tol)
        logger.info(self, 'gmres_restart = %d', self.gmres_restart)
        logger.info(self, 'fock_deriv_method = %s', self.fock_deriv_method)
        logger.info(self, 'imaginary_residual_tol = %g',
                    self.imaginary_residual_tol)
        logger.info(self, 'target XC numerical integrator = %s',
                    type(self.base._numint).__name__)
        logger.info(self, 'Hartree/response provider = %s',
                    type(self.base.with_df).__name__)
        return self

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None,
               atmlst=None):
        '''Evaluate sampled-k analytic ``dE/dR`` in Ha/Bohr.'''
        if not isinstance(self.base.with_df, (FFTDF, GDF, RSDF)):
            raise NotImplementedError(
                'sampled-k analytic gradient supports FFTDF, GDF and RSDF '
                'only')
        if not self.grid_response:
            raise NotImplementedError(
                'sampled-k target derivative requires grid_response=True')
        if self.fock_deriv_method != 'direct':
            raise NotImplementedError(
                "sampled-k gradients require fock_deriv_method='direct'")
        if self.dense_oracle_size < 1:
            raise ValueError('dense_oracle_size must be positive')
        if self.zvec_method not in ('auto', 'dense', 'matrix_free'):
            raise ValueError(
                "zvec_method must be 'auto', 'dense' or 'matrix_free'")
        if self.cphf_conv_tol <= 0:
            raise ValueError('cphf_conv_tol must be positive')
        if self.cphf_residual_tol <= 0:
            raise ValueError('cphf_residual_tol must be positive')
        if self.cphf_max_cycle < 1:
            raise ValueError('cphf_max_cycle must be positive')
        if self.self_adjoint_tol <= 0:
            raise ValueError('self_adjoint_tol must be positive')
        if self.gmres_restart < 1:
            raise ValueError('gmres_restart must be positive')
        if self.imaginary_residual_tol <= 0:
            raise ValueError('imaginary_residual_tol must be positive')

        self.term_imaginary_residuals = {}
        self.term_imaginary_residual_max_abs = 0.0
        self.term_raw_imaginary_components = {}
        self.term_raw_imaginary_component_max_abs = 0.0

        de = krks.Gradients.kernel(
            self, mo_energy, mo_coeff, mo_occ, atmlst)
        self.de_nuc = self.grad_nuc(atmlst=self.atmlst)
        self.de_skel = self.de_skel_elec + self.de_nuc
        self.de_terms = (
            self.de_hcore + self.de_veff + self.de_extra + self.de_ovlp
            + self.de_nuc + self.de_fock + self.de_ovlp_energy
            + self.de_ovlp_response)
        self.de = np.asarray(de)
        self.term_recomposition_max_abs = float(
            np.max(np.abs(self.de - self.de_terms)))
        self.translation_residual = np.sum(self.de, axis=0)
        self.translation_residual_max_abs = float(
            np.max(np.abs(self.translation_residual)))
        self.target_fock_reused = True
        return self.de

    grad_elec = grad_elec

    def as_scanner(self):
        raise NotImplementedError(
            'sampled-k gradient scanner is outside the K3 release')

    def get_stress(self):
        raise NotImplementedError(
            'sampled-k stress and cell derivatives are a separate goal')


Grad = Gradients
