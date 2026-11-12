# YEScape-Grad
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

'''Restricted non-SCF energy workflow built on PySCF RKS.'''

from pyscf import scf
from pyscf.lib import logger
from pyscf.df import df_jk
from pyscf.dft import rks, libxc

from yescape_grad.tools.logger import log_kernel_time
from yescape_grad.tools.source import SourceAPI, make_source
from yescape_grad.cpu.disp.d2c import evaluate_d2c, normalize_dispersion


def _check_df_owner(mf, with_df=None):
    '''Require every molecular DF provider to own the active ``Mole``.

    A DF object is not a geometry-independent option container.  Its auxiliary
    basis and three-center integrals belong to the ``Mole`` used to construct
    it.  Equal AO dimensions are therefore insufficient: attaching a provider
    from another geometry can produce shape-compatible but physically mixed
    one- and two-electron Hamiltonian terms.

    ``with_df`` validates a provider supplied to :meth:`density_fit` before
    PySCF copies it into the source and target objects.  The attached providers
    are checked as well so direct attribute mutation fails at the next energy
    or gradient entry point.
    '''
    providers = [('with_df', with_df)]
    target_df = getattr(mf, 'with_df', None)
    source = getattr(mf, 'mf_source', None)
    source_df = getattr(source, 'with_df', None)
    providers.extend((('target with_df', target_df),
                      ('source with_df', source_df)))

    if source is not None and source.mol is not mf.mol:
        raise ValueError(
            'source SCF and the non-SCF workflow must reference the same Mole object')

    for name, provider in providers:
        if provider is not None and getattr(provider, 'mol', None) is not mf.mol:
            raise ValueError(
                f'{name} and the non-SCF workflow must reference the same Mole object')

    if source is not None:
        if (target_df is None) != (source_df is None):
            raise ValueError(
                'source SCF and target DFT must use the same DF representation')
        if target_df is not None and target_df is not source_df:
            raise ValueError(
                'source SCF and target DFT must share one molecular with_df')


def _make_source_signature(mf):
    '''Describe the molecular Hamiltonian represented by the source SCF state.

    The signature intentionally excludes execution-only controls such as
    ``max_memory``, ``verbose`` and SCF tolerances.  It records only molecule
    and DF choices that can change the stationary source-SCF orbitals used by
    the Z-vector gradient.
    '''
    _check_df_owner(mf)
    mol = mf.mol
    mf_source = mf.mf_source
    with_df = getattr(mf_source, 'with_df', None)
    if with_df is None:
        df_signature = ('conventional',)
    else:
        df_signature = (
            'density_fit',
            type(with_df).__module__,
            type(with_df).__qualname__,
            repr(with_df.auxbasis),
            bool(getattr(mf_source, 'only_dfj', False)),
        )

    return (
        tuple(mol.atom_charges().tolist()),
        mol.atom_coords().tobytes(),
        repr(mol._basis),
        repr(mol._ecp),
        repr(mol._pseudo),
        repr(mol.nucmod),
        int(mol.charge),
        int(mol.spin),
        bool(mol.cart),
        df_signature,
        _source_dfa_signature(mf_source),
    )


def _source_dfa_signature(source):
    '''Record source XC and grid settings, without hashing quadrature arrays.'''
    if not hasattr(source, 'xc'):
        return ('HF',)
    grids = source.grids
    return (source.xc, source.omega, source.small_rho_cutoff,
            grids.level, grids.cutoff, repr(grids.atom_grid), repr(grids.prune),
            repr(grids.radi_method), repr(grids.becke_scheme),
            repr(grids.radii_adjust), repr(grids.atomic_radii))


def _record_source_signature(mf):
    '''Record source settings for provenance, not as a reuse permission check.'''
    mf._source_signature = _make_source_signature(mf)


def _check_source_state(mf):
    '''Check usable orbital dimensions, without requiring identical old arrays.

    Explicit source reuse is the caller's choice. Neither this check nor a
    stored converged flag proves stationarity in the current Hamiltonian.
    Nonstationary orbitals need not give an analytic gradient consistent with
    a reoptimized-source finite difference.
    '''
    _check_df_owner(mf)
    source = mf.mf_source
    energy, coeff, occ = (getattr(source, name, None)
                          for name in ('mo_energy', 'mo_coeff', 'mo_occ'))
    if any(value is None for value in (energy, coeff, occ)):
        raise RuntimeError('source SCF calculation has no canonical orbital state')
    if getattr(coeff, 'ndim', None) == 2:
        blocks = [(energy, coeff, occ)]
    elif isinstance(coeff, (tuple, list)) or getattr(coeff, 'ndim', None) == 3:
        if len(coeff) != 2 or len(energy) != 2 or len(occ) != 2:
            raise ValueError('source orbitals require one restricted or two spin blocks')
        blocks = zip(energy, coeff, occ)
    else:
        raise ValueError('source orbital shapes require a matrix or two spin matrices')
    for e, c, o in blocks:
        if (getattr(c, 'ndim', None) != 2 or c.shape[0] != mf.mol.nao_nr()
                or getattr(e, 'shape', None) != (c.shape[1],)
                or getattr(o, 'shape', None) != (c.shape[1],)):
            raise ValueError('source orbital shapes do not match the active AO space')
    if isinstance(coeff, (tuple, list)) or coeff.ndim == 3:
        # mol.spin is Nalpha-Nbeta = 2S, not the target <S^2> itself.
        spin = abs(mf.mol.spin) / 2
        target_s2 = spin * (spin + 1)
        s2 = float(source.spin_square()[0])
        # Compare endpoints to include e.g. 0.85 versus 0.75 despite subtraction rounding.
        if s2 >= target_s2 + 0.1 or s2 <= target_s2 - 0.1:
            source_name = 'UKS' if hasattr(source, 'xc') else 'UHF'
            method_name = 'NSC-DFT' if hasattr(source, 'xc') else 'HF-DFT'
            logger.warn(mf, '%s source is spin contaminated: <S^2>=%.6f, '
                        'target S(S+1)=%.6f, absolute difference=%.6f >= 0.1. '
                        'Be very cautious when using the %s energy and gradient '
                        'and consider a different source.',
                        source_name, s2, target_s2, abs(s2 - target_s2), method_name)
    if not source.converged:
        logger.warn(mf, 'Reusing source orbitals marked unconverged; source '
                    'stationarity and analytic-gradient/FD agreement are not guaranteed.')
    if hasattr(source, 'grids') and source.grids.coords is None:
        # Native checkpoints contain orbitals, not quadrature arrays.
        source.grids.build()


def _clear_target_state(mf):
    '''Discard old results before source SCF or a representation change.

    Keep source orbitals available for an explicit guess/reuse, but never
    advertise the previous target energy as a successful new calculation.
    '''
    mf._last_energy_timing = None
    mf.e_source = None
    mf.e_dft = mf.e_d2c = mf.dm_source = None
    mf.d2c_result = None
    mf.e_tot = None
    mf.converged = False


def _check_xc(mf):
    '''Validate molecular representation ownership and target-XC terms.'''
    _check_df_owner(mf)
    source = getattr(mf, 'mf_source', None)
    if source is not None and hasattr(source, 'xc'):
        # Source fxc belongs in the orbital Hessian, independently of target XC.
        _check_xc(source)
        if libxc.needs_laplacian(source.xc):
            raise NotImplementedError('source MGGA must depend on tau, not Laplacian')
        libxc.test_deriv_order(source.xc, 2, raise_error=True)
        if source.grids.mol is not mf.mol:
            raise ValueError('source grids must own the active Mole')
        if source.small_rho_cutoff != 0:
            raise ValueError('DFA source requires small_rho_cutoff=0 before SCF')
        if getattr(mf, 'dispersion', None):
            raise NotImplementedError('D2C parameters are fitted for HF sources, not DFA sources')
    if not hasattr(mf, '_numint'):
        return

    xc, _, disp = rks.parse_dft(mf.xc)
    has_nlc = mf._numint.libxc.is_nlc(xc)
    if has_nlc or mf.nlc not in ('', 0, False):
        raise NotImplementedError('nonlocal correlation such as VV10 is not supported')
    if disp is not None or mf.do_disp():
        raise NotImplementedError(
            "PySCF mf.disp corrections are not supported; "
            "use set_dispersion('d2c') for the published HF-DFT parameters"
        )


class RNSC(SourceAPI, rks.RKS):
    '''Molecular non-self-consistent target energy on a stationary source.

    Args:
        mol : pyscf.gto.Mole
            Shared molecule for source SCF and target evaluation.
        xc : str
            Target functional, including semilocal, hybrid or RSH terms.
        mf_source : native HF/KS object or None
            Optional conventional source with matching spin/backend. Its grids
            and XC are retained. Do not also pass source_xc.

    Kwargs:
        source_xc : str or None
            Source functional. None or 'HF' constructs HF; otherwise KS.
        mf_hf : native HF/KS object or None
            Compatibility spelling of mf_source; do not supply both.

    Notes:
        Public source-neutral name: RNSC. The historical class names are
        retained as compatibility aliases. Target SCF is never performed. Configure
        source grids on mf_source.grids, independently of target self.grids.
    '''

    _keys = rks.RKS._keys | {
        'mf_source', 'e_source', 'e_dft', 'e_d2c', 'dm_source',
        'dispersion', 'd2c_result', '_source_signature',
    }
    _source_options = (
        'max_memory', 'verbose', 'stdout', 'chkfile',
        'conv_tol', 'conv_tol_grad', 'conv_tol_cpscf', 'max_cycle',
        'init_guess', 'DIIS', 'diis', 'diis_space', 'diis_start_cycle',
        'diis_file', 'diis_space_rollback', 'diis_damp', 'damp',
        'level_shift', 'direct_scf', 'direct_scf_tol', 'conv_check',
        'callback',
    )

    def __init__(self, mol, xc='LDA,VWN', mf_source=None, *,
                 source_xc=None, mf_hf=None):
        rks.RKS.__init__(self, mol, xc=xc)
        mf_source, supplied_source = make_source(
            mol, scf.hf.RHF, rks.RKS, mf_hf, mf_source, source_xc)

        self.mf_source = mf_source
        if supplied_source:
            self._copy_source_options(mf_source, self)
        self.e_source = None
        self.e_dft = None
        self.e_d2c = None
        self.dm_source = None
        self.dispersion = None
        self.d2c_result = None
        self._source_signature = _make_source_signature(self)

    def set_dispersion(self, correction=None):
        '''Select an additive geometry-only dispersion correction.

        Args:
            correction : str, bool, or None
                ``'d2c'`` selects the published D2C dispersion parameters matching
                the current target ``xc``.  ``None`` disables dispersion.

        Returns:
            RNSC
                This workflow object, following PySCF modifier style.

        Notes:
            D2C is deliberately separate from PySCF's standard ``mf.disp``
            option.  This prevents accidental use of ordinary D4 parameters
            or double counting of two dispersion corrections.
        '''
        self.dispersion = normalize_dispersion(correction)
        self.e_d2c = self.d2c_result = None
        self.e_tot = self.e_dft if self.dispersion is None else None
        return self

    def kernel_source(self, dm0=None, **kwargs):
        '''Run the restricted source SCF kernel.'''
        _check_xc(self)
        _clear_target_state(self)
        self.mf_source.converged = False
        self._copy_source_options(self, self.mf_source)
        self._prepare_source_grid()
        e_source = self.mf_source.kernel(dm0=dm0, **kwargs)
        if not self.mf_source.converged:
            raise RuntimeError('restricted source SCF calculation did not converge')
        self._sync_source_results()
        _record_source_signature(self)
        return e_source

    def run_source(self, dm0=None, **kwargs):
        self.kernel_source(dm0=dm0, **kwargs)
        return self

    def make_source_rdm1(self):
        _check_source_state(self)
        return self.mf_source.make_rdm1()

    @log_kernel_time('Energy workflow')
    def kernel(self, dm0=None, run_source=True, *, run_hf=None, **kwargs):
        '''Run source SCF, then evaluate the target functional at its density.'''
        if run_hf is not None:
            if run_source is not True and run_source != run_hf:
                raise ValueError("Conflicting run_source and run_hf arguments")
            run_source = run_hf
        _check_xc(self)
        if run_source:
            self.kernel_source(dm0=dm0, **kwargs)
        else:
            if dm0 is not None or kwargs:
                raise ValueError('dm0/kwargs require run_source=True')
            _check_source_state(self)
            self._sync_source_results()

        self.dm_source = self.make_source_rdm1()
        self.e_tot = float(self.energy_tot(dm=self.dm_source))
        from yescape_grad.tools.logger import energy_summary
        energy_summary(self)
        return self.e_tot

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        '''Evaluate target non-SCF energy and optional D2C dispersion correction.'''
        _check_xc(self)
        if dm is None:
            dm = self.make_source_rdm1()
            self._sync_source_results()
        self.e_dft = float(rks.RKS.energy_tot(self, dm, h1e, vhf))
        self.dispersion = normalize_dispersion(self.dispersion)
        self.e_d2c = 0.0
        self.d2c_result = None
        self.scf_summary.pop('dispersion', None)
        if self.dispersion == 'd2c':
            self.d2c_result = evaluate_d2c(self.mol, self.xc)
            self.e_d2c = self.d2c_result.energy
            self.scf_summary['dispersion'] = self.e_d2c
        self.e_tot = self.e_dft + self.e_d2c
        return self.e_tot

    def reset(self, mol=None):
        rks.RKS.reset(self, mol)
        self.mf_source.reset(mol)
        _clear_target_state(self)
        return self

    def nuc_grad_method(self):
        from yescape_grad.cpu.grad.rnsc import Gradients
        return Gradients(self)

    Gradients = nuc_grad_method
    scf = kernel

    def Hessian(self):
        raise NotImplementedError('Non-SCF Hessians are not implemented')

    def as_scanner(self):
        from yescape_grad.tools.scanner import energy_scanner
        return energy_scanner(self)

    def density_fit(self, auxbasis=None, with_df=None, only_dfj=False):
        '''Apply the same PySCF density-fitting provider to source and target.

        Args:
            auxbasis : str, basis dict, or None
                Auxiliary basis passed to :func:`pyscf.df.df_jk.density_fit`.
                When omitted, PySCF selects the matching HF/JK-fit basis and
                uses its normal even-tempered fallback if no match exists.

        Kwargs:
            with_df : pyscf.df.df.DF or None
                Existing molecular DF provider.  When supplied, the source
                Source SCF and target RKS energy evaluator share this provider.
            only_dfj : bool
                Use density fitting for Coulomb only.  PySCF supports this for
                energies; YEScape-Grad analytic DF gradients currently require
                ``only_dfj=False``.

        Returns:
            RNSC
                A copied workflow whose restricted source SCF and target DFT evaluator
                both use PySCF's molecular density-fitting implementation.

        Notes:
            The source is transformed first using an exchange-capable basis,
            also suitable when a semilocal source feeds a hybrid target.
            The resulting ``with_df`` is
            then attached to the target evaluator, keeping source and target
            on one explicit three-center-integral representation.
        '''
        _check_df_owner(self, with_df)
        source_was_converged = bool(self.mf_source.converged)
        if auxbasis is None and with_df is None:
            # A semilocal source can still feed a hybrid target. Resolve one
            # exchange-capable provider, as in the original HF-source path.
            from pyscf.df.addons import make_auxbasis
            auxbasis = make_auxbasis(self.mol, xc='HF')
        mf_source = df_jk.density_fit(
            self.mf_source,
            auxbasis=auxbasis,
            with_df=with_df,
            only_dfj=only_dfj,
        )
        mf = _DFRNSC(self, mf_source.with_df, only_dfj)
        mf.mf_source = mf_source
        _clear_target_state(mf)
        if not source_was_converged:
            _record_source_signature(mf)
        return mf

    def _sync_source_results(self):
        mf_source = self.mf_source
        self.e_source = float(mf_source.e_tot)
        self.e_dft = self.e_d2c = self.dm_source = None
        self.d2c_result = None
        self.e_tot = self.e_source
        self.mo_energy = mf_source.mo_energy
        self.mo_coeff = mf_source.mo_coeff
        self.mo_occ = mf_source.mo_occ
        self.converged = mf_source.converged
        self.cycles = mf_source.cycles

    def _copy_source_options(self, source, target):
        for key in self._source_options:
            if hasattr(source, key) and hasattr(target, key):
                setattr(target, key, getattr(source, key))


class _DFRNSC(df_jk._DFHF, RNSC):
    '''PySCF density-fitting mixin specialized for the non-SCF workflow.'''

    # The native DF mixin otherwise exposes a target-SCF Hessian.
    Hessian = RNSC.Hessian

    def undo_df(self):
        '''Remove density fitting from both source and target workflow objects.'''
        source_was_converged = bool(self.mf_source.converged)
        mf = df_jk._DFHF.undo_df(self)
        mf.mf_source = self.mf_source.undo_df()
        _clear_target_state(mf)
        if not source_was_converged:
            _record_source_signature(mf)
        return mf

    def nuc_grad_method(self):
        from yescape_grad.cpu.df.grad.rnsc import Gradients
        return Gradients(self)

    Gradients = nuc_grad_method


def energy_tot(mf_source, mf_dft):
    '''Evaluate a target DFT energy with a converged HF or KS density.

    Args:
        mf_source : pyscf.scf.hf.SCF
            Converged restricted or unrestricted HF/KS source object.
        mf_dft : pyscf.dft.rks.RKS or pyscf.dft.uks.UKS
            Target supplied-density evaluator on the same molecule.

    Returns:
        float
            Target total energy in Hartree.  This function calls
            ``mf_dft.energy_tot(dm=mf_source.make_rdm1())`` and never target SCF.
    '''
    if not mf_source.converged:
        logger.warn(mf_dft, 'Evaluating a supplied density from an unconverged source.')
    if mf_source.mol is not mf_dft.mol:
        raise ValueError('mf_source and mf_dft must reference the same Mole object')

    _check_xc(mf_dft)
    dm_source = mf_source.make_rdm1()
    return float(mf_dft.energy_tot(dm=dm_source))


# Compatibility name; internal code uses the source-neutral class.
RHFDFT = RNSC
