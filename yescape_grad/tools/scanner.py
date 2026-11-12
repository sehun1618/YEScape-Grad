# YEScape-Grad
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

'''Molecular non-SCF scanners using the PySCF scanner lifecycle.'''

from pyscf import gto, lib
from pyscf.scf.hf import SCF_Scanner as PySCFScanner
from pyscf.grad.rhf import SCF_GradScanner as PySCFGradScanner


def _guess_signature(mol):
    '''Identify compatible AO density matrices without comparing coordinates.'''
    return (tuple(mol.atom_symbol(i) for i in range(mol.natm)),
            mol.charge, mol.spin, mol.cart, repr(mol._basis), repr(mol._ecp))


class SCFScanner(PySCFScanner):
    '''Evaluate non-SCF energies along a molecular geometry sequence.

    Previous source-SCF density is the default guess when atom order, basis,
    charge and spin match. This is AO-density reuse, not orbital projection.
    Pass ``dm0=None`` for a fresh guess or ``dm0=array`` for an explicit guess.
    Every call reconverges source SCF and evaluates the target functional;
    it never runs a target-DFT SCF. Geometry-dependent DF/grid intermediates
    are reset by the underlying workflow. Failed calls cannot report stale
    convergence or provide an automatic guess for the next call.

    Like PySCF scanners, this is a shallow view sharing source/DF objects with
    the original workflow. Do not alternate calls to the original and scanner.
    CPU NumPy and GPU CuPy density arrays remain on their original backend.
    '''

    def __init__(self, mf):
        super().__init__(mf)
        self._last_guess_signature = _guess_signature(mf.mol)

    def __call__(self, mol_or_geom, **kwargs):
        mol = (mol_or_geom if isinstance(mol_or_geom, gto.MoleBase)
               else self.mol.set_geom_(mol_or_geom, inplace=False))
        if (kwargs.get('run_hf', True) is not True
                or kwargs.get('run_source', True) is not True):
            raise ValueError('A scanner must reconverge source SCF at each geometry')
        if _guess_signature(mol) != self._last_guess_signature:
            # PySCF checks AO layout only; equal-size bases/charge changes also
            # need a fresh default guess. An explicit dm0 is still respected.
            kwargs.setdefault('dm0', None)
        self.converged = False
        self.e_tot = None
        # Native scanners clear their own mo_coeff before kernel(). Our SCF
        # lives in mf_source, so clear that copy too: otherwise PySCF can silently
        # recreate an old density even when dm0=None requests a fresh guess.
        # The workflow copy remains available to the native scanner's default
        # make_rdm1() until it has selected the initial density.
        self.mf_source.mo_coeff = None
        try:
            energy = super().__call__(mol, **kwargs)
        except BaseException:
            # Invalidate state on Ctrl-C too; always re-raise the original error.
            self.converged = self.mf_source.converged = False
            self.mo_coeff = None
            self.e_tot = None
            raise
        self._last_guess_signature = _guess_signature(mol)
        return energy


class GradientScanner(PySCFGradScanner):
    '''Return ``(target-on-source energy, dE/dR)`` in Ha and Ha/Bohr.

    The energy scanner owns default source-density reuse. ``dm0`` is forwarded
    to it (including explicit None); other keywords are gradient-kernel
    arguments. D2C, when selected, is evaluated at the new geometry in both
    energy and gradient. Z residual warnings retain their normal meaning and
    are not converted into a new convergence criterion by this scanner.
    '''

    def __call__(self, mol_or_geom, **kwargs):
        mol = (mol_or_geom if isinstance(mol_or_geom, gto.MoleBase)
               else self.mol.set_geom_(mol_or_geom, inplace=False))
        source_kwargs = {}
        if 'dm0' in kwargs:
            source_kwargs['dm0'] = kwargs.pop('dm0')
        self.de = None
        self.base.converged = False
        try:
            self.reset(mol)
            energy = self.base(mol, **source_kwargs)
            gradient = self.kernel(**kwargs)
        except BaseException:
            self.de = None
            self.base.converged = False
            raise
        return energy, gradient


def energy_scanner(mf):
    '''Return an idempotent scanner view, preserving the workflow's class.'''
    if isinstance(mf, lib.SinglePointScanner):
        return mf
    return lib.set_class(SCFScanner(mf), (SCFScanner, mf.__class__))


def gradient_scanner(gradient):
    '''Return a PySCF-compatible CPU/GPU non-SCF gradient scanner view.'''
    if isinstance(gradient, lib.GradScanner):
        return gradient
    return lib.set_class(GradientScanner(gradient),
                         (GradientScanner, gradient.__class__))
