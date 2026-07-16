"""Sampled-k-point restricted GPU DC-DFT driver."""

from .base import DCRestrictedBase


class DCKRKS(DCRestrictedBase):
    """Evaluate restricted sampled-k-point ``E_DFA[n_KRHF]`` on the GPU."""

    kpoint_mode = True
