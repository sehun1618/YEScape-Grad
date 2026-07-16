"""GTH local/nonlocal derivative support metadata."""


def capability() -> dict[str, object]:
    """Return backend-resolved restricted GTH derivative support."""

    return {
        "family": "GTH",
        "energy": True,
        "local_gradient": True,
        "nonlocal_gradient": True,
        "fftdf_representation": "finite_reciprocal_mesh",
        "gdf_representation": "analytic_projector_integrals",
        "status": "fftdf_and_gdf_cpu_fd_validated",
    }
