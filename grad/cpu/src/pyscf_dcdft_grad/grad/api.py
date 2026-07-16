"""
DC-DFT Analytic Gradients - Unified Interface
==============================================

This module provides a unified interface for DC-DFT gradients,
automatically dispatching to restricted or unrestricted implementations.

Usage:
    from pyscf_dcdft_grad import dc_grad

    mf = scf.RHF(mol).run()  # or UHF
    grad_calc = dc_grad(mf, xc='B3LYP')
    gradient = grad_calc.kernel()
"""

import numpy as np
from typing import Tuple, Optional
from pyscf.lib import logger


# ============================================================================
# Utility Functions
# ============================================================================

def maxabs(arr: np.ndarray) -> float:
    """Maximum absolute value, safe for empty arrays."""
    arr = np.asarray(arr)
    return 0.0 if arr.size == 0 else float(np.max(np.abs(arr)))


def is_unrestricted(mo_coeff, mo_occ) -> bool:
    """
    Detect unrestricted (spin-polarized) data.

    Unrestricted data is stored as:
        - Tuple/list of (alpha, beta)
        - 3D array with shape (2, ...)
    """
    if isinstance(mo_coeff, (tuple, list)) or isinstance(mo_occ, (tuple, list)):
        return True
    if isinstance(mo_coeff, np.ndarray) and mo_coeff.ndim == 3 and mo_coeff.shape[0] == 2:
        return True
    if isinstance(mo_occ, np.ndarray) and mo_occ.ndim == 2 and mo_occ.shape[0] == 2:
        return True
    return False


def as_spin_tuple(obj) -> Tuple[np.ndarray, np.ndarray]:
    """Convert spin-polarized data to (alpha, beta) tuple."""
    if isinstance(obj, (tuple, list)):
        return (np.asarray(obj[0]), np.asarray(obj[1]))
    arr = np.asarray(obj)
    if arr.ndim >= 1 and arr.shape[0] == 2:
        return (arr[0], arr[1])
    raise ValueError(f'Expected spin tuple or 2-stacked array, got shape {arr.shape}')


def split_occ_vir(mo_coeff: np.ndarray, mo_occ: np.ndarray) -> Tuple[np.ndarray, np.ndarray, slice, slice]:
    """
    Split MO coefficients into occupied and virtual spaces.

    Returns:
        Co: Occupied orbital coefficients (nao, nocc)
        Cv: Virtual orbital coefficients (nao, nvir)
        so: Slice for occupied orbitals
        sv: Slice for virtual orbitals
    """
    nocc = int(np.count_nonzero(mo_occ > 0))
    so = slice(0, nocc)
    sv = slice(nocc, mo_coeff.shape[1])
    return mo_coeff[:, so], mo_coeff[:, sv], so, sv


def has_df(mf) -> bool:
    """Check if mean-field has density fitting."""
    return getattr(mf, 'with_df', None) is not None


def get_auxbasis(mf) -> Optional[str]:
    """Get auxiliary basis from mean-field object."""
    if has_df(mf):
        return getattr(mf.with_df, 'auxbasis', None)
    return None


def normalize_verbose(verbose, default):
    """Normalize verbose parameter to integer level."""
    if verbose is None:
        return default
    if isinstance(verbose, bool):
        return logger.NOTE if verbose else logger.QUIET
    return verbose


def print_gradient(mol, g: np.ndarray, title: str = "Gradient"):
    """Pretty-print gradient in PySCF style."""
    print(f"\n{title}:")
    for ia in range(mol.natm):
        symbol = mol.atom_symbol(ia)
        print(f'    {symbol:>2s}  {g[ia,0]:16.10f}  {g[ia,1]:16.10f}  {g[ia,2]:16.10f}')


# ============================================================================
# Main Interface
# ============================================================================

# Import after utility functions to avoid circular import
from . import rhf
from . import uhf


def dc_grad(mf_hf, xc: str, **kwargs):
    """
    Create DC-DFT gradient calculator (auto-dispatch).

    Args:
        mf_hf: Converged HF/UHF mean-field object
        xc: DFT functional
        **kwargs: Additional options (grids, cphf_tol, etc.)

    Returns:
        RDC_Gradients or UDC_Gradients instance

    Example:
        >>> from pyscf import gto, scf
        >>> from pyscf_dcdft_grad import dc_grad
        >>>
        >>> mol = gto.M(atom='H 0 0 0; F 0 0 1.1', basis='6-31g')
        >>> mf = scf.RHF(mol).run()
        >>> grad_calc = dc_grad(mf, xc='B3LYP')
        >>> gradient = grad_calc.kernel()
    """
    if is_unrestricted(mf_hf.mo_coeff, mf_hf.mo_occ):
        return uhf.Gradients(mf_hf, xc, **kwargs)
    else:
        return rhf.Gradients(mf_hf, xc, **kwargs)


# Convenience functions
def energy(mol, xc, mo_coeff, mo_occ, **kwargs):
    """
    Compute DC-DFT energy (auto-dispatch).

    Returns:
        DC-DFT total energy
    """
    if is_unrestricted(mo_coeff, mo_occ):
        return uhf.energy(mol, xc, mo_coeff, mo_occ, **kwargs)
    else:
        return rhf.energy(mol, xc, mo_coeff, mo_occ, **kwargs)


def kernel(mf_hf, xc: str, mo_coeff=None, mo_occ=None,
           grids=None, cphf_tol=1e-10, grid_response=True,
           auxbasis_response=True, verbose=None, output=None):
    """
    One-shot DC-DFT gradient computation with output control.

    Args:
        mf_hf: HF/UHF mean-field object
        xc: DFT functional
        mo_coeff: MO coefficients (default: mf_hf.mo_coeff)
        mo_occ: MO occupations (default: mf_hf.mo_occ)
        grids: Custom grids
        cphf_tol: CPHF tolerance
        grid_response: Include grid response
        auxbasis_response: Include auxbasis response (for DF)
        verbose: Verbosity level (int or bool)
        output: Output file path (str) or file object

    Returns:
        de: Nuclear gradients (natm, 3)

    Example with output file:
        >>> grad = kernel(mf, 'B3LYP', verbose=True, output='gradient.log')
    """
    # Handle output parameter
    original_stdout = None
    close_on_exit = False

    if output is not None:
        original_stdout = mf_hf.stdout

        if isinstance(output, str):
            # String: open file
            mf_hf.stdout = open(output, 'w')
            close_on_exit = True
        else:
            # File object
            mf_hf.stdout = output

    try:
        grad_calc = dc_grad(
            mf_hf, xc,
            grids=grids,
            cphf_tol=cphf_tol,
            grid_response=grid_response,
            auxbasis_response=auxbasis_response,
        )

        de = grad_calc.kernel(mo_coeff=mo_coeff, mo_occ=mo_occ, verbose=verbose)

        return de

    finally:
        # Restore stdout
        if original_stdout is not None:
            if close_on_exit:
                mf_hf.stdout.flush()  # Ensure all output is written
                mf_hf.stdout.close()
            mf_hf.stdout = original_stdout


# Aliases for backward compatibility
Gradients = dc_grad
DC_Gradients = dc_grad
dc_dft_gradient = kernel


__all__ = [
    "DC_Gradients",
    "Gradients",
    "dc_dft_gradient",
    "dc_grad",
    "energy",
    "kernel",
]
