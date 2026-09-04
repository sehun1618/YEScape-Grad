"""Selected finite differences for deliberately difficult structures."""

import numpy as np
import pytest

from dcforge.cpu.energy import RHFDFT, UHFDFT
from tests.common import finite_difference, make_mol, run_workflow


CASES = [
    ("no_cation.xyz", RHFDFT, 1, 2, 1e-4, 3),
    ("h2_stretched.xyz", RHFDFT, 1, 2, 1e-4, 3),
    ("water1_asymmetric_stretch.xyz", RHFDFT, 1, 2, 1e-4, 3),
    # OH is deliberately retained as the known difficult CPHF/grid case. Its
    # 1e-4-Bohr energy difference is dominated by quadrature/SCF noise, so the
    # diagnostic uses a converged 2e-3-Bohr step and a denser grid.
    ("oh.xyz", UHFDFT, 1, 2, 2e-3, 4),
    ("ch2_triplet_distorted.xyz", UHFDFT, 1, 2, 1e-4, 3),
    ("ch3_doublet_distorted.xyz", UHFDFT, 1, 0, 1e-4, 3),
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
    assert abs(analytic - fd) < 3e-6
