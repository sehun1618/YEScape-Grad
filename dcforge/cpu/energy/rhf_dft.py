# DC-Forge
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

'''Restricted HF-DFT energy workflow built on PySCF RKS.'''

from pyscf import scf
from pyscf.lib import logger
from pyscf.df import df_jk
from pyscf.dft import rks

from dcforge.tools.logger import log_kernel_time
from dcforge.cpu.disp.d2c import evaluate_d2c, normalize_dispersion


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
    source = getattr(mf, 'mf_hf', None)
    source_df = getattr(source, 'with_df', None)
    providers.extend((('target with_df', target_df),
                      ('source with_df', source_df)))

    if source is not None and source.mol is not mf.mol:
        raise ValueError(
            'source HF and the HF-DFT workflow must reference the same Mole object')

    for name, provider in providers:
        if provider is not None and getattr(provider, 'mol', None) is not mf.mol:
            raise ValueError(
                f'{name} and the HF-DFT workflow must reference the same Mole object')

    if source is not None:
        if (target_df is None) != (source_df is None):
            raise ValueError(
                'source HF and target DFT must use the same DF representation')
        if target_df is not None and target_df is not source_df:
            raise ValueError(
                'source HF and target DFT must share one molecular with_df')


def _make_source_signature(mf):
    '''Describe the molecular Hamiltonian represented by the source HF state.

    The signature intentionally excludes execution-only controls such as
    ``max_memory``, ``verbose`` and SCF tolerances.  It records only molecule
    and DF choices that can change the stationary source-HF orbitals used by
    the Z-vector gradient.
    '''
    _check_df_owner(mf)
    mol = mf.mol
    mf_hf = mf.mf_hf
    with_df = getattr(mf_hf, 'with_df', None)
    if with_df is None:
        df_signature = ('conventional',)
    else:
        df_signature = (
            'density_fit',
            type(with_df).__module__,
            type(with_df).__qualname__,
            repr(with_df.auxbasis),
            bool(getattr(mf_hf, 'only_dfj', False)),
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
    )


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
    source = mf.mf_hf
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
    if not source.converged:
        logger.warn(mf, 'Reusing source orbitals marked unconverged; source '
                    'stationarity and analytic-gradient/FD agreement are not guaranteed.')


def _clear_target_state(mf):
    '''Discard copied target results after a source representation change.'''
    for name in ('e_hf', 'e_uhf'):
        if hasattr(mf, name):
            setattr(mf, name, None)
    mf.e_dft = mf.e_d2c = mf.dm_hf = None
    mf.d2c_result = None
    mf.e_tot = None
    mf.converged = False


def _check_xc(mf):
    '''Validate molecular representation ownership and target-XC terms.'''
    _check_df_owner(mf)
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


class RHFDFT(rks.RKS):
    '''Evaluate a target RKS energy on converged RHF orbitals.'''

    _keys = rks.RKS._keys | {
        'mf_hf', 'e_hf', 'e_dft', 'e_d2c', 'dm_hf',
        'dispersion', 'd2c_result', '_source_signature',
    }
    _hf_options = (
        'max_memory', 'verbose', 'stdout', 'chkfile',
        'conv_tol', 'conv_tol_grad', 'conv_tol_cpscf', 'max_cycle',
        'init_guess', 'DIIS', 'diis', 'diis_space', 'diis_start_cycle',
        'diis_file', 'diis_space_rollback', 'diis_damp', 'damp',
        'level_shift', 'direct_scf', 'direct_scf_tol', 'conv_check',
        'callback',
    )

    def __init__(self, mol, xc='LDA,VWN', mf_hf=None):
        rks.RKS.__init__(self, mol, xc=xc)
        supplied_mf_hf = mf_hf is not None
        if mf_hf is None:
            mf_hf = scf.RHF(mol)
        if type(mf_hf) is not scf.hf.RHF:
            raise TypeError('mf_hf must be a conventional pyscf.scf.hf.RHF object')
        if mf_hf.mol is not mol:
            raise ValueError('mf_hf and RHFDFT must reference the same Mole object')

        self.mf_hf = mf_hf
        if supplied_mf_hf:
            self._copy_hf_options(mf_hf, self)
        self.e_hf = None
        self.e_dft = None
        self.e_d2c = None
        self.dm_hf = None
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
            RHFDFT
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

    def kernel_hf(self, dm0=None, **kwargs):
        '''Run the source RHF kernel.'''
        self._copy_hf_options(self, self.mf_hf)
        e_hf = self.mf_hf.kernel(dm0=dm0, **kwargs)
        if not self.mf_hf.converged:
            raise RuntimeError('source RHF calculation did not converge')
        self._sync_hf_results()
        _record_source_signature(self)
        return e_hf

    def run_hf(self, dm0=None, **kwargs):
        self.kernel_hf(dm0=dm0, **kwargs)
        return self

    def make_hf_rdm1(self):
        _check_source_state(self)
        return self.mf_hf.make_rdm1()

    @log_kernel_time('Energy workflow')
    def kernel(self, dm0=None, run_hf=True, **kwargs):
        '''Run RHF, then evaluate the target functional at the RHF density.'''
        _check_xc(self)
        if run_hf:
            self.kernel_hf(dm0=dm0, **kwargs)
        else:
            if dm0 is not None or kwargs:
                raise ValueError('dm0/kwargs require run_hf=True')
            _check_source_state(self)
            self._sync_hf_results()

        self.dm_hf = self.make_hf_rdm1()
        self.e_tot = float(self.energy_tot(dm=self.dm_hf))
        from dcforge.tools.logger import energy_summary
        energy_summary(self)
        return self.e_tot

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        '''Evaluate target HF-DFT energy and optional D2C dispersion correction.'''
        _check_xc(self)
        if dm is None:
            dm = self.make_hf_rdm1()
            self._sync_hf_results()
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
        self.mf_hf.reset(mol)
        self.e_hf = self.e_dft = self.e_d2c = self.dm_hf = None
        self.d2c_result = None
        return self

    def nuc_grad_method(self):
        from dcforge.cpu.grad.rhf_dft import Gradients
        return Gradients(self)

    def as_scanner(self):
        from dcforge.tools.scanner import energy_scanner
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
                RHF and target RKS energy evaluator share this provider.
            only_dfj : bool
                Use density fitting for Coulomb only.  PySCF supports this for
                energies; DC-Forge analytic DF gradients currently require
                ``only_dfj=False``.

        Returns:
            RHFDFT
                A copied workflow whose source RHF and target DFT evaluator
                both use PySCF's molecular density-fitting implementation.

        Notes:
            The source RHF is transformed first because it always requires an
            exchange-capable auxiliary basis.  The resulting ``with_df`` is
            then attached to the target evaluator, keeping source and target
            on one explicit three-center-integral representation.
        '''
        _check_df_owner(self, with_df)
        source_was_converged = bool(self.mf_hf.converged)
        mf_hf = df_jk.density_fit(
            self.mf_hf,
            auxbasis=auxbasis,
            with_df=with_df,
            only_dfj=only_dfj,
        )
        mf = _DFRHFDFT(self, mf_hf.with_df, only_dfj)
        mf.mf_hf = mf_hf
        _clear_target_state(mf)
        if not source_was_converged:
            _record_source_signature(mf)
        return mf

    def _sync_hf_results(self):
        mf_hf = self.mf_hf
        self.e_hf = float(mf_hf.e_tot)
        self.e_dft = self.e_d2c = self.dm_hf = None
        self.d2c_result = None
        self.e_tot = self.e_hf
        self.mo_energy = mf_hf.mo_energy
        self.mo_coeff = mf_hf.mo_coeff
        self.mo_occ = mf_hf.mo_occ
        self.converged = mf_hf.converged
        self.cycles = mf_hf.cycles

    def _copy_hf_options(self, source, target):
        for key in self._hf_options:
            if hasattr(source, key) and hasattr(target, key):
                setattr(target, key, getattr(source, key))


class _DFRHFDFT(df_jk._DFHF, RHFDFT):
    '''PySCF density-fitting mixin specialized for the HF-DFT workflow.'''

    def undo_df(self):
        '''Remove density fitting from both source and target workflow objects.'''
        source_was_converged = bool(self.mf_hf.converged)
        mf = df_jk._DFHF.undo_df(self)
        mf.mf_hf = self.mf_hf.undo_df()
        _clear_target_state(mf)
        if not source_was_converged:
            _record_source_signature(mf)
        return mf

    def nuc_grad_method(self):
        from dcforge.cpu.df.grad.rhf_dft import Gradients
        return Gradients(self)

    Gradients = nuc_grad_method


def energy_tot(mf_hf, mf_dft):
    '''Evaluate a target DFT energy with a converged RHF or UHF density.

    Args:
        mf_hf : pyscf.scf.hf.SCF
            Converged source RHF or UHF object.
        mf_dft : pyscf.dft.rks.RKS or pyscf.dft.uks.UKS
            Target supplied-density evaluator on the same molecule.

    Returns:
        float
            Target total energy in Hartree.  This function calls
            ``mf_dft.energy_tot(dm=mf_hf.make_rdm1())`` and never target SCF.
    '''
    if not mf_hf.converged:
        logger.warn(mf_dft, 'Evaluating a supplied density from an unconverged source.')
    if mf_hf.mol is not mf_dft.mol:
        raise ValueError('mf_hf and mf_dft must reference the same Mole object')

    _check_xc(mf_dft)
    dm_hf = mf_hf.make_rdm1()
    return float(mf_dft.energy_tot(dm=dm_hf))
