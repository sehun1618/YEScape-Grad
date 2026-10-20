"""GPU fixed-density energies and CPU/GPU parity."""

import numpy as np
import pytest

from dcforge.cpu.energy import RHFDFT as CPU_RHFDFT
from dcforge.cpu.energy import UHFDFT as CPU_UHFDFT
from dcforge.gpu.energy import RHFDFT as GPU_RHFDFT
from dcforge.gpu.energy import UHFDFT as GPU_UHFDFT
from tests.common import make_mol, run_workflow


pytestmark = pytest.mark.gpu


@pytest.mark.parametrize(
    "cpu_workflow,gpu_workflow,name",
    [
        (CPU_RHFDFT, GPU_RHFDFT, "water1.xyz"),
        (CPU_UHFDFT, GPU_UHFDFT, "o2_triplet.xyz"),
    ],
)
@pytest.mark.parametrize("density_fit", [False, True], ids=["conventional", "df"])
def test_gpu_energy_matches_cpu(cpu_workflow, gpu_workflow, name, density_fit):
    mol = make_mol(name, basis="def2-svp")
    auxbasis = "def2-svp-jkfit" if density_fit else None
    _, energy_cpu = run_workflow(
        cpu_workflow, mol, density_fit=density_fit, auxbasis=auxbasis)
    mf_gpu, energy_gpu = run_workflow(
        gpu_workflow, mol, density_fit=density_fit, auxbasis=auxbasis)
    assert np.isfinite(energy_gpu)
    assert abs(energy_gpu - energy_cpu) < 2e-8
    assert mf_gpu.device == "gpu"
    if density_fit:
        assert mf_gpu.with_df is mf_gpu.mf_hf.with_df


def test_gpu_d2c_energy_is_added_by_the_shared_molecular_provider():
    mol = make_mol("water1.xyz", basis="sto-3g")
    mf, energy = run_workflow(
        GPU_RHFDFT, mol, xc="r2SCAN", dispersion="d2c")
    assert mf.d2c_result is not None
    assert abs(energy - (mf.e_dft + mf.e_d2c)) < 1e-12
