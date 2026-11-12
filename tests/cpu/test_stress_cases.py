"""Selected finite differences for deliberately difficult structures."""

import numpy as np
import pytest
from pyscf import lib

from yescape_grad.cpu.energy import RNSC, UNSC
from tests.common import finite_difference, make_mol, run_workflow


CASES = [
    ("no_cation.xyz", RNSC, 1, 2, 1e-4, 3),
    ("h2_stretched.xyz", RNSC, 1, 2, 1e-4, 3),
    ("water1_asymmetric_stretch.xyz", RNSC, 1, 2, 1e-4, 3),
    # OH is deliberately retained as the known difficult CPHF/grid case. Its
    # 1e-4-Bohr energy difference is dominated by quadrature/SCF noise, so the
    # diagnostic uses a converged 2e-3-Bohr step and a denser grid.
    ("oh.xyz", UNSC, 1, 2, 2e-3, 4),
    ("ch2_triplet_distorted.xyz", UNSC, 1, 2, 1e-4, 3),
    ("ch3_doublet_distorted.xyz", UNSC, 1, 0, 1e-4, 3),
]


@pytest.mark.cpu
@pytest.mark.fd
@pytest.mark.slow
@pytest.mark.parametrize(
    "name,workflow,atom,axis,step,grid_level",
    CASES,
    ids=[case[0].removesuffix(".xyz") for case in CASES],
)
def test_stress_structure_selected_gradient_component(
        name, workflow, atom, axis, step, grid_level):
    # These are numerical acceptance tests, not thread-scaling benchmarks.
    # Fixing the reduction order keeps the deliberately ill-conditioned OH
    # tail reproducible across machines with different OpenMP defaults.
    with lib.with_omp_threads(1):
        mol = make_mol(name, basis="sto-3g")
        mf, _ = run_workflow(workflow, mol, grid_level=grid_level)
        mf_grad = mf.nuc_grad_method()
        analytic = mf_grad.kernel()[atom, axis]
        fd = finite_difference(
            workflow, mol, atom, axis, step=step, grid_level=grid_level)

    # Some open-shell tails expose a loose Krylov stopping norm even though
    # the independently recomputed physical residual and force remain useful
    # diagnostics. Do not hide that behavior by weakening the regular tests.
    assert np.isfinite(mf_grad.z_residual_max_abs)
    assert mf_grad.z_residual_warning == (
        mf_grad.z_residual_max_abs > mf_grad.cphf_residual_warn_tol)
    assert abs(analytic - fd) < 3e-6
