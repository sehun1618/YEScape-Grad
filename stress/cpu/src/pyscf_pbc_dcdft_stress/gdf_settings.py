"""Immutable GDF settings copied from one converged central provider."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GDFProviderSettings:
    auxbasis: object | None
    mesh: tuple[int, int, int] | None
    eta: float | None
    exp_to_discard: float | None
    linear_dep_threshold: float
    max_memory: float
    prefer_ccdf: bool
    verbose: int

    @classmethod
    def from_provider(cls, provider):
        mesh = getattr(provider, "mesh", None)
        return cls(
            auxbasis=getattr(provider, "auxbasis", None),
            mesh=None if mesh is None else tuple(int(value) for value in mesh),
            eta=_optional_float(getattr(provider, "eta", None)),
            exp_to_discard=_optional_float(
                getattr(provider, "exp_to_discard", None)
            ),
            linear_dep_threshold=float(provider.linear_dep_threshold),
            max_memory=float(provider.max_memory),
            prefer_ccdf=bool(getattr(provider, "_prefer_ccdf", False)),
            verbose=int(provider.verbose),
        )

    def build(self, cell):
        from pyscf_pbc_dcdft import df

        provider = df.GDF(cell, auxbasis=self.auxbasis)
        provider.mesh = None if self.mesh is None else np.asarray(self.mesh, dtype=int)
        provider.eta = self.eta
        provider.exp_to_discard = self.exp_to_discard
        provider.linear_dep_threshold = self.linear_dep_threshold
        provider.max_memory = self.max_memory
        if hasattr(provider, "_prefer_ccdf") or self.prefer_ccdf:
            provider._prefer_ccdf = self.prefer_ccdf
        provider.verbose = self.verbose
        return provider

    def as_dict(self):
        return {
            "auxbasis": self.auxbasis,
            "mesh": None if self.mesh is None else list(self.mesh),
            "eta": self.eta,
            "exp_to_discard": self.exp_to_discard,
            "linear_dep_threshold": self.linear_dep_threshold,
            "max_memory": self.max_memory,
            "prefer_ccdf": self.prefer_ccdf,
            "verbose": self.verbose,
        }


def _optional_float(value):
    return None if value is None else float(value)
