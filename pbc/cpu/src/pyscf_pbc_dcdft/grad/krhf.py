"""Sampled-k-point DC-DFT gradient public boundary."""

from .rhf import DCGradients


class DCKGradients(DCGradients):
    """Nuclear gradient of ``E_DFA[n_KRHF]`` with explicit k-point weights."""
