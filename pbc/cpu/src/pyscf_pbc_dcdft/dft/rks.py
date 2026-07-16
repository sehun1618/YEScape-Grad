"""Gamma-point restricted DC-DFT driver."""

from .base import DCRestrictedBase


class DCRKS(DCRestrictedBase):
    """Compute ``E_DFA[n_RHF]`` for a restricted Gamma-point cell."""

    kpoint_mode = False
