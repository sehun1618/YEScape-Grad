"""Gamma-point restricted GPU DC-DFT driver."""

from .base import DCRestrictedBase


class DCRKS(DCRestrictedBase):
    """Evaluate restricted Gamma-point ``E_DFA[n_RHF]`` on the GPU."""

    kpoint_mode = False
