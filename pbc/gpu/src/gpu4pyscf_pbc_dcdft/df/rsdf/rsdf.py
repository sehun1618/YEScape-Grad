"""Project-local GPU RSDF with explicit central-factor ownership."""

from __future__ import annotations

import numpy as np
from pyscf_pbc_dcdft import df as cpu_df
from pyscf_pbc_dcdft.df.rsdf.parameters import parameters_from_provider

from ..gdf.build_signature import make_gdf_build_signature
from ..gdf.gdf import GDF
from .central_exact import build_exact_cderi
from .pp_value import build_pp_value


_RSDF_SETTINGS = (
    "use_bvk",
    "precision_R",
    "precision_G",
    "npw_max",
    "_omega_min",
    "omega",
    "ke_cutoff",
    "mesh_compact",
    "omega_j2c",
    "mesh_j2c",
    "precision_j2c",
    "j2c_eig_always",
)
_RESOLVED_SETTINGS = (
    "omega",
    "ke_cutoff",
    "mesh_compact",
    "omega_j2c",
    "mesh_j2c",
)


class RSDF(GDF):
    """Range-separated GDF with selectable exact central setup.

    ``central_setup='reference'`` retains the original PySCF central CDERI.
    ``central_setup='exact_gpu_source'`` resolves PySCF's independent source
    and metric parameters without building host CDERI, generates the large
    three-center source through GPU4PySCF, and keeps only the small screened
    two-center metric primitive on the host.  Both modes expose identical
    canonical CuPy factors to SCF, target DFA, AX/Z-vector, and gradients.
    """

    backend_name = "rsdf"
    _keys = GDF._keys | {"central_setup"}

    def __init__(
        self,
        cell,
        kpts=None,
        *,
        auxbasis=None,
        central_setup="reference",
    ):
        # RSDF owns an independent CPU-identical source/metric contract.  Do
        # not inherit GDF's native central-builder default implicitly.
        super().__init__(
            cell,
            kpts=kpts,
            auxbasis=auxbasis,
            setup_mode="reference",
        )
        self.central_setup = _validate_central_setup(central_setup)
        self._dc_central_build_metadata = None
        self._dc_cderi_eviction_metadata = None
        template = cpu_df.RSDF(
            cell, kpts=np.asarray(self.kpts), auxbasis=auxbasis
        )
        _copy_settings(template, self)

    def reset(self, cell=None):
        """Discard geometry-dependent factors and central-build metadata."""

        super().reset(cell)
        self._dc_central_build_metadata = None
        self._dc_cderi_eviction_metadata = None
        return self

    def build(self, j_only=None, kpts_band=None):
        """Build full exchange-capable factors through the selected route."""

        self._dc_cderi_eviction_metadata = None

        if self.central_setup == "reference":
            result = super().build(j_only=j_only, kpts_band=kpts_band)
            self._dc_central_build_metadata = {
                "builder": "PySCF RSDF host central reference",
                "source_owner": "PySCF host",
                "short_range_metric_owner": "PySCF host",
            }
            return result
        if kpts_band is not None:
            raise NotImplementedError("RSDF kpts_band is not supported")

        current_signature = make_gdf_build_signature(self)
        if (
            self._dc_build_signature is not None
            and self._dc_build_signature != current_signature
        ):
            self._clear_pp_cache()
        points = np.asarray(self.kpts, dtype=float).reshape(-1, 3)
        witness = self._build_parameter_witness(points)
        if bool(witness.j2c_eig_always):
            raise NotImplementedError(
                "exact GPU-source central setup requires Cholesky metric "
                "factorization; j2c_eig_always=True is unsupported"
            )
        parameters = parameters_from_provider(witness)
        central = build_exact_cderi(
            self.cell,
            witness.auxcell,
            points,
            parameters,
        )
        self._cpu_witness = witness
        self._dc_cderi_blocks = central.blocks
        self._cderi = self._dc_cderi_blocks
        self.auxcell = witness.auxcell
        self.nao = int(self.cell.nao_nr())
        self.kmesh = np.asarray(parameters.bvk_kmesh, dtype=int)
        self._j_only = False
        self._dc_central_build_metadata = central.metadata
        self._dc_build_signature = make_gdf_build_signature(self)
        return self

    def release_cderi_cache(self, *, reason):
        """Release rebuildable central factors after their final gradient use."""

        blocks = self._dc_cderi_blocks
        released_bytes = 0
        if blocks is not None:
            released_bytes = sum(
                int(value.nbytes) for value in blocks.values()
            )
        self._dc_cderi_blocks = None
        self._cderi = None
        self._dc_cderi_eviction_metadata = {
            "released": True,
            "reason": str(reason),
            "released_cderi_bytes": int(released_bytes),
            "rebuild_on_next_jk_request": True,
        }
        return dict(self._dc_cderi_eviction_metadata)

    def _build_cpu_witness(self, points):
        witness = self._configured_cpu_witness(points)
        witness.build(j_only=False)
        self._adopt_resolved_settings(witness)
        return witness

    def _build_parameter_witness(self, points):
        """Resolve PySCF RSDF settings without constructing host CDERI."""

        witness = self._configured_cpu_witness(points)
        witness._rs_build()
        self._adopt_resolved_settings(witness)
        return witness

    def _configured_cpu_witness(self, points):
        witness = cpu_df.RSDF(
            self.cell,
            kpts=points,
            auxbasis=self.auxbasis,
        )
        _copy_settings(self, witness)
        witness.linear_dep_threshold = float(self.linear_dep_threshold)
        return witness

    def _adopt_resolved_settings(self, witness):
        for name in _RESOLVED_SETTINGS:
            value = getattr(witness, name)
            if isinstance(value, np.ndarray):
                value = value.copy()
            setattr(self, name, value)

    def get_pp(self, kpts=None):
        """Return the GPU4PySCF RSDF PP value with geometry-safe caching."""

        self._ensure_built()

        def builder():
            result = build_pp_value(self.cell, kpts)
            self._dc_pp_build_metadata = result.metadata
            return result.value

        return self._cached_pp_value(
            kpts,
            builder,
        )

    def capability(self) -> dict[str, object]:
        """Return the current GPU RSDF ownership contract."""

        return {
            "backend": self.backend_name,
            "device": self.device_name,
            "auxbasis": self.auxbasis,
            "energy": True,
            "analytic_gradient": True,
            "gradient_status": "fused_adjoint_release_gate_passed",
            "target_xc": ["PBE", "SCAN", "R2SCAN"],
            "central_setup": self.central_setup,
            "central_cderi": (
                "GPU three-center source + exact screened host metric"
                if self.central_setup == "exact_gpu_source"
                else "PySCF RSDF host setup, canonical CuPy storage"
            ),
            "central_build_metadata": self._dc_central_build_metadata,
            "cderi_eviction_metadata": self._dc_cderi_eviction_metadata,
            "pseudopotential_value_cache": {
                "builder": "project-local bounded GPU4PySCF PP",
                "scope": "same geometry and k-point argument",
                "entries": len(self._dc_pp_cache),
                "hits": self._dc_pp_cache_hits,
                "misses": self._dc_pp_cache_misses,
                "build_metadata": self._dc_pp_build_metadata,
            },
            "gradient_response": (
                "GPU-native source derivative, reciprocal metric derivative, "
                "and fused target-J/Z-bilinear adjoint contraction"
            ),
            "response_mode": "native_source_native_metric_fused_adjoint",
            "jk_contraction_metadata": self._dc_jk_contraction_metadata,
            "silent_cpu_fallback": False,
            "host_boundaries": [
                self._central_host_boundary(),
                "RSDF PP local-part-2 value primitive",
                "exact screened short-range metric ip1/ip2",
                "GTH local-part-2 H1 primitive generation",
                "PySCF Cell/AuxCell metadata",
            ],
        }

    def to_cpu(self):
        """Create a matching project-local CPU RSDF provider."""

        provider = cpu_df.RSDF(
            self.cell,
            kpts=np.asarray(self.kpts),
            auxbasis=self.auxbasis,
        )
        _copy_settings(self, provider)
        return provider

    def _discard_cached_build(self):
        super()._discard_cached_build()
        self._dc_central_build_metadata = None

    def _central_host_boundary(self):
        if self.central_setup == "exact_gpu_source":
            return (
                "RSDF parameter resolution and screened two-center metric "
                "value primitive"
            )
        return "RSDF central 2-center/3-center integral and CDERI setup"


def _copy_settings(source, target):
    for name in _RSDF_SETTINGS:
        value = getattr(source, name)
        if isinstance(value, np.ndarray):
            value = value.copy()
        setattr(target, name, value)


def _validate_central_setup(value):
    mode = str(value).strip().lower()
    if mode not in {"reference", "exact_gpu_source"}:
        raise ValueError(
            "RSDF central_setup must be 'reference' or "
            f"'exact_gpu_source', got {value!r}"
        )
    return mode
