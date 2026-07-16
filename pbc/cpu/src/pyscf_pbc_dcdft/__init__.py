"""Restricted periodic density-corrected DFT.

The package mirrors the public shape of :mod:`pyscf.pbc` while keeping the
source HF calculation, target DFA evaluation, and nonvariational gradient
response explicit.  Experimental modules from the earlier research workspace
are deliberately not imported here.
"""

from . import df, dft, grad, lib, pseudo, scf
from ._version import __version__
from .dft import DCKRKS, DCRKS

__all__ = [
    "DCKRKS",
    "DCRKS",
    "__version__",
    "df",
    "dft",
    "grad",
    "lib",
    "pseudo",
    "scf",
]
