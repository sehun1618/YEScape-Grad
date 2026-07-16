"""Restricted GPU density-corrected DFT methods."""

from .krks import DCKRKS
from .rks import DCRKS

__all__ = ["DCRKS", "DCKRKS"]
