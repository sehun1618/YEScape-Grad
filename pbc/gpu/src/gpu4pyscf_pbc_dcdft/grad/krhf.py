"""Sampled-k-point GPU DC-DFT gradient public API."""

from .rhf import DCGradients


class DCKGradients(DCGradients):
    """Analytic nuclear gradient of ``E_DFA[n_KRHF]`` on the GPU."""
