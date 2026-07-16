"""Sampled-k-point restricted DC-DFT driver."""

from .base import DCRestrictedBase


class DCKRKS(DCRestrictedBase):
    """Compute ``E_DFA[n_KRHF]`` using equal-weight sampled k-points."""

    kpoint_mode = True
