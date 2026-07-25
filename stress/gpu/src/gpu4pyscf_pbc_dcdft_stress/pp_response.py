"""Density-contracted GPU4PySCF pseudopotential strain response."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from types import MappingProxyType
from typing import Mapping

import cupy as cp
import numpy as np
from gpu4pyscf.pbc.gto.cell import get_Gv_weights

from .profiling import CUDAEventProfiler


@dataclass(frozen=True)
class ContractedPseudopotentialStrain:
    """PP and XC derivatives already contracted with the required densities."""

    density_derivative: cp.ndarray
    adjoint_derivative: cp.ndarray
    target_xc_derivative: cp.ndarray
    timings_s: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "density_derivative", cp.asarray(self.density_derivative))
        object.__setattr__(self, "adjoint_derivative", cp.asarray(self.adjoint_derivative))
        object.__setattr__(
            self, "target_xc_derivative", cp.asarray(self.target_xc_derivative)
        )
        object.__setattr__(self, "timings_s", MappingProxyType(dict(self.timings_s)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def contracted_gpu4pyscf_pp_strain(
    context, adjoint_state, *, grid_block_size=None, detailed_profile=False,
    nonlocal_response_backend="centered_cell_displacement",
    nonlocal_host_block_size=8192,
):
    """Contract periodic PP strain with source and adjoint densities on GPU.

    The conventional stress route first builds six PP derivative AO matrices.
    DC-DFT needs these matrices only through two traces: one with the source
    density and one with the adjoint transition density.  GPU4PySCF evaluates
    those traces directly, avoiding the host AO-matrix construction.

    The local-potential response is analytic.  The nonlocal-projector backend
    is explicit: it may use centered cell displacements, the exact host
    derivative matrix, or a device-contracted analytic Fourier-moment route.
    SCF, target density, Z-vector and AO-density response are never recomputed.
    """

    sampled_k = context.nkpts != 1 or bool(
        getattr(context.dc_mf, "kpoint_mode", False)
    )
    if sampled_k:
        from gpu4pyscf.pbc.grad import krks as gpu_krks_grad
        from gpu4pyscf.pbc.grad import krks_stress

        gradient = gpu_krks_grad.Gradients(context.target_ks)
        kpts = np.asarray(context.kpts, dtype=float).reshape(-1, 3)

        def evaluate_xc(density):
            return krks_stress.get_vxc(
                gradient, context.cell, cp.asarray(density), kpts,
                with_j=False, with_nuc=False,
            )

        source_density = cp.asarray(context.density)
        transition_density = cp.stack(adjoint_state.transition_density)
        driver = "gpu4pyscf.pbc.grad.krks_stress.get_vxc"
    else:
        from gpu4pyscf.pbc.grad import rks as gpu_rks_grad
        from gpu4pyscf.pbc.grad import rks_stress

        gradient = gpu_rks_grad.Gradients(context.target_ks)

        def evaluate_xc(density):
            return rks_stress.get_vxc(
                gradient, context.cell, cp.asarray(density),
                with_j=False, with_nuc=False,
            )

        source_density = cp.asarray(context.density[0])
        transition_density = cp.asarray(adjoint_state.transition_density[0])
        driver = "gpu4pyscf.pbc.grad.rks_stress.get_vxc"

    timings = {}
    device_profiler = CUDAEventProfiler(detailed_profile)
    started = perf_counter()
    source_xc = cp.asarray(device_profiler.call(
        "source_xc", lambda: evaluate_xc(source_density)
    ))
    cp.cuda.Stream.null.synchronize()
    timings["source_xc"] = perf_counter()-started

    started = perf_counter()
    pp_tensors, pp_phase_timings, pp_work_profile = _fused_pp_strain(
        context, source_density, transition_density, sampled_k=sampled_k,
        grid_block_size=grid_block_size, device_profiler=device_profiler,
        nonlocal_response_backend=nonlocal_response_backend,
        nonlocal_host_block_size=nonlocal_host_block_size,
    )
    cp.cuda.Stream.null.synchronize()
    timings["fused_source_adjoint_pp"] = perf_counter()-started
    timings.update({
        f"pp_{name}": value for name, value in pp_phase_timings.items()
    })
    timings["total"] = (
        timings["source_xc"]+timings["fused_source_adjoint_pp"]
    )
    pp_work_profile["cuda_event_profile"] = device_profiler.result()

    return ContractedPseudopotentialStrain(
        density_derivative=_tensor_to_voigt(pp_tensors[0]),
        adjoint_derivative=2.0*_tensor_to_voigt(pp_tensors[1]),
        target_xc_derivative=_tensor_to_voigt(source_xc),
        timings_s=timings,
        metadata={
            "owner": (
                "gpu4pyscf_native_energy_contracted_analytic_nonlocal_device"
                if nonlocal_response_backend == "gpu_analytic"
                else "gpu4pyscf_native_energy_contracted_analytic_nonlocal"
                if nonlocal_response_backend == "host_analytic_matrix"
                else "gpu4pyscf_native_energy_contracted"
            ),
            "driver": driver,
            "density_count": 2,
            "density_execution": "fused_source_and_adjoint",
            "pp_density_variables": ["rho"],
            "grid_block_size": (
                None if grid_block_size is None else int(grid_block_size)
            ),
            "ao_derivative": "gpu_analytic",
            "pp_factor_derivative": {
                "local_3d_gth": "analytic_reciprocal_chain_rule",
                "nonlocal": (
                    "gpu_analytic_augmented_shell_fourier_moment"
                    if nonlocal_response_backend == "gpu_analytic"
                    else
                    "host_analytic_fourier_projector_strain_device_contracted"
                    if nonlocal_response_backend == "host_analytic_matrix"
                    else "symmetric_voigt_centered_cell_displacement"
                ),
            },
            "nonlocal_derivative_numerical": (
                nonlocal_response_backend
                not in {"host_analytic_matrix", "gpu_analytic"}
            ),
            "full_scf_finite_difference": False,
            "sampled_k": sampled_k,
            "detailed_profile": bool(detailed_profile),
            "work_profile": pp_work_profile,
        },
    )


def _fused_pp_strain(
    context, source_density, transition_density, *, sampled_k,
    grid_block_size=None, device_profiler=None,
    nonlocal_response_backend="centered_cell_displacement",
    nonlocal_host_block_size=8192,
):
    """Evaluate only PP strain for two densities while sharing AO/grid work."""

    profiler = device_profiler or CUDAEventProfiler(False)
    if sampled_k:
        from gpu4pyscf.pbc.grad import krks_stress as module
        from gpu4pyscf.pbc.grad import rks_stress as primitive_module

        density_sets = cp.stack((
            cp.asarray(source_density), cp.asarray(transition_density)
        ))
        kpts = np.asarray(context.kpts, dtype=float).reshape(-1, 3)
    else:
        from gpu4pyscf.pbc.grad import rks_stress as module

        primitive_module = module
        density_sets = cp.stack((
            cp.asarray(source_density), cp.asarray(transition_density)
        ))
        kpts = None

    cell = context.cell
    if not cell.cart:
        c2s = module.asarray(cell.cart2sph_coeff())
        density_sets = cp.stack(tuple(
            module.sandwich_dot(density, c2s.T) for density in density_sets
        ))
        cell = cell.copy()
        cell.cart = True

    grids = context.target_ks.grids
    grids_idx = grids.argsort(tile=8)
    grids_coords = grids.coords[grids_idx]
    ngrids = len(grids_coords)
    mesh = grids.mesh
    ndensity = int(density_sets.shape[0])
    nao = int(density_sets.shape[-1])
    rho0 = cp.zeros((ndensity, ngrids))
    phase_timings = {}
    started_phase = perf_counter()
    vpploc_g, vpploc_g1, vpploc_analytic = _vpploc_g_strain(
        primitive_module, cell, mesh
    )
    vpploc_r = profiler.call(
        "local_pp_ifft",
        lambda: primitive_module.pbctools.ifft(vpploc_g, mesh).real,
    )
    vpploc_r_sorted = vpploc_r[grids_idx]
    local_density_response = cp.zeros((ndensity, 3, 3), dtype=cp.float64)
    cp.cuda.Stream.null.synchronize()
    phase_timings["local_potential_setup"] = perf_counter()-started_phase

    if sampled_k:
        nkpts = len(kpts)
        evaluator = module._GTOvalOpt(cell, kpts, deriv=1)
        denominator = max(1, nkpts*ndensity*10*nao)
    else:
        nkpts = 1
        evaluator = module._GTOvalOpt(cell, deriv=1)
        denominator = max(1, ndensity*10*nao)
    strain_weight = 2.0/nkpts
    block_size = int(
        (4e9/16/denominator)/module.ALIGNED
    )*module.ALIGNED
    block_size = max(module.ALIGNED, block_size)
    if grid_block_size is not None:
        requested = int(grid_block_size)
        if requested <= 0:
            raise ValueError("PP grid block size must be positive")
        aligned = max(
            module.ALIGNED,
            requested//module.ALIGNED*module.ALIGNED,
        )
        block_size = min(block_size, aligned)

    started_phase = perf_counter()
    grid_block_count = 0
    for p0, p1 in module.lib.prange(0, ngrids, block_size):
        grid_block_count += 1
        coords = cp.asarray(grids_coords[p0:p1].T, order="C").T
        rho1_block = cp.zeros(
            (ndensity, 3, 3, p1-p0), dtype=cp.float64
        )
        if sampled_k:
            ao_kpts = profiler.call(
                "local_ao_values",
                lambda: module.eval_ao_kpts(
                    cell, coords, kpts, deriv=1, opt=evaluator
                ),
            )
            ao_strain_kpts = profiler.call(
                "local_ao_strain",
                lambda: module._eval_ao_strain_derivatives(
                    cell, coords, kpts, deriv=0, opt=evaluator
                ),
            )
            for kpoint, (ao_raw, strain_raw) in enumerate(
                zip(ao_kpts, ao_strain_kpts)
            ):
                rho1_block += profiler.call(
                    "local_density_response",
                    lambda ao_raw=ao_raw, strain_raw=strain_raw,
                    kpoint=kpoint: _accumulate_density_strain(
                        module, coords, ao_raw, strain_raw,
                        density_sets[:, kpoint], rho0[:, p0:p1],
                    ),
                )
        else:
            ao_raw = profiler.call(
                "local_ao_values",
                lambda: module.eval_ao_kpts(
                    cell, coords, deriv=1, opt=evaluator
                )[0],
            )
            strain_raw = profiler.call(
                "local_ao_strain",
                lambda: module._eval_ao_strain_derivatives(
                    cell, coords, deriv=0, opt=evaluator
                )[0],
            )
            rho1_block += profiler.call(
                "local_density_response",
                lambda: _accumulate_density_strain(
                    module, coords, ao_raw, strain_raw, density_sets,
                    rho0[:, p0:p1],
                ),
            )
        local_density_response += profiler.call(
            "local_density_contraction",
            lambda: strain_weight*module.contract(
                "dxyg,g->dxy", rho1_block, vpploc_r_sorted[p0:p1]
            ).real,
        )
    cp.cuda.Stream.null.synchronize()
    phase_timings["ao_density_strain"] = perf_counter()-started_phase

    started_phase = perf_counter()
    if sampled_k:
        rho0 *= 1.0/nkpts
    rho0_fft = cp.empty_like(rho0)
    rho0_fft[:, grids_idx] = rho0

    output = local_density_response.copy()
    for index in range(ndensity):
        rho_g = profiler.call(
            "local_density_fft",
            lambda index=index: primitive_module.pbctools.fft(
                rho0_fft[index], mesh
            ),
        )
        if vpploc_analytic is None:
            output[index] += profiler.call(
                "local_reciprocal_contraction",
                lambda: module.contract(
                    "g,xyg->xy", rho_g.conj(), vpploc_g1
                ).real*(1.0/ngrids),
            )
        else:
            gvectors, weighted_slope = vpploc_analytic
            for row in range(3):
                for column in range(3):
                    output[index, row, column] += profiler.call(
                        "local_reciprocal_contraction",
                        lambda row=row, column=column: 2.0*cp.einsum(
                            "g,g,g,g->", rho_g.conj(), gvectors[:, row],
                            gvectors[:, column], weighted_slope,
                            optimize=True,
                        ).real*(1.0/ngrids),
                    )
    cp.cuda.Stream.null.synchronize()
    phase_timings["local_reciprocal"] = perf_counter()-started_phase

    started_phase = perf_counter()
    nonlocal_value, nonlocal_profile = _fused_pp_nonlocal_strain(
        primitive_module, cell, mesh, density_sets, kpts=kpts,
        return_profile=True, device_profiler=profiler,
        response_backend=nonlocal_response_backend,
        host_block_size=nonlocal_host_block_size,
        gpu_block_size=block_size,
    )
    output += nonlocal_value
    cp.cuda.Stream.null.synchronize()
    phase_timings["nonlocal_projector"] = perf_counter()-started_phase
    phase_timings["total"] = sum(phase_timings.values())
    complex_bytes = np.dtype(np.complex128).itemsize
    real_bytes = np.dtype(np.float64).itemsize
    peak_block_points = min(block_size, ngrids)
    estimated_grid_workspace = (
        nkpts*(4+9)*nao*peak_block_points*complex_bytes
        + ndensity*3*3*peak_block_points*real_bytes
    )
    work_profile = {
        "density_count": ndensity,
        "kpoint_count": nkpts,
        "grid_point_count": int(ngrids),
        "grid_block_size": int(block_size),
        "grid_block_count": int(grid_block_count),
        "ao_value_batch_call_count": int(grid_block_count),
        "ao_strain_batch_call_count": int(grid_block_count),
        "ao_kpoint_evaluation_count": int(grid_block_count*nkpts),
        "density_contraction_count": int(grid_block_count*nkpts*ndensity),
        "local_fft_count": int(ndensity),
        "local_ifft_count": 1,
        "estimated_explicit_grid_workspace_bytes": int(
            estimated_grid_workspace
        ),
        "retained_grid_density_bytes": int(rho0.nbytes),
        "nonlocal": nonlocal_profile,
    }
    return output, phase_timings, work_profile


def _accumulate_density_strain(
    module, coords, ao_raw, strain_raw, density_sets, rho0,
):
    """Accumulate scalar density and its strain response for one k-point."""

    ao = ao_raw.transpose(0, 2, 1)
    ao_strain = strain_raw[:, :, 0]
    ao_strain += module.contract(
        "xig,yg->xyig", ao[1:4], coords.T
    )
    response = cp.empty(
        (len(density_sets), 3, 3, len(coords)), dtype=cp.float64
    )
    for index, density in enumerate(density_sets):
        occupied_projection = density.T.dot(ao[0])
        rho0[index] += _partial_dot(ao[0], occupied_projection)
        response[index] = module.contract(
            "xyig,ig->xyg", ao_strain, occupied_projection.conj()
        ).real
    return response


def _partial_dot(bra, ket):
    result = cp.einsum("ig,ig->g", bra.real, ket.real)
    result += cp.einsum("ig,ig->g", bra.imag, ket.imag)
    return result




def _vpploc_g_strain(module, cell, mesh):
    """Return 3D GTH local potential and analytic reciprocal strain."""

    if int(cell.dimension) != 3:
        value, derivative = module._get_vpplocG_strain_derivatives(cell, mesh)
        return value, derivative, None
    gvectors, _gvbase, _weights = get_Gv_weights(cell, mesh)
    g2 = cp.einsum("gi,gi->g", gvectors, gvectors)
    nonzero = g2 > 0.0
    safe_g2 = cp.where(nonzero, g2, 1.0)
    weighted_value = cp.zeros(len(g2), dtype=cp.complex128)
    weighted_slope = cp.zeros(len(g2), dtype=cp.complex128)
    four_pi = 4.0*np.pi
    correction_prefactor = (2.0*np.pi)**1.5
    atom_coordinates = cp.asarray(np.asarray(cell.atom_coords(), dtype=float))

    atoms_by_symbol = {}
    for atom in range(cell.natm):
        atoms_by_symbol.setdefault(cell.atom_symbol(atom), []).append(atom)
    for symbol, atoms in atoms_by_symbol.items():
        charge = float(cell.atom_charge(atoms[0]))
        value = four_pi*charge/safe_g2
        slope = -four_pi*charge/safe_g2**2
        if symbol in cell._pseudo:
            pseudo = cell._pseudo[symbol]
            radius, count, coefficients = pseudo[1:4]
            radius2 = radius**2
            reduced = radius2*g2
            damping = cp.exp(-0.5*reduced)
            value *= damping
            slope = value*(-0.5*radius2-1.0/safe_g2)
            value = cp.where(
                nonzero, value, -2.0*np.pi*charge*radius2
            )

            polynomial, derivative = _gth_local_polynomial(
                reduced, count, coefficients
            )
            correction = (
                -correction_prefactor*radius**3*damping*polynomial
            )
            correction_slope = (
                -correction_prefactor*radius**5*damping
                *(derivative-0.5*polynomial)
            )
            value += correction
            slope += correction_slope
            slope = cp.where(nonzero, slope, 0.0)
        else:
            value = cp.where(nonzero, value, 0.0)
            slope = cp.where(nonzero, slope, 0.0)
        symbol_structure_factor = cp.zeros(len(g2), dtype=cp.complex128)
        for atom in atoms:
            symbol_structure_factor += cp.exp(
                -1j*gvectors.dot(atom_coordinates[atom])
            )
        weighted_value += symbol_structure_factor*value
        weighted_slope += symbol_structure_factor*slope

    value = -weighted_value
    analytic_factors = (gvectors, weighted_slope)
    return value, None, analytic_factors


def _gth_local_polynomial(reduced_g2, count, coefficients):
    """Evaluate the GTH local polynomial and derivative in reduced G2."""

    array_module = cp.get_array_module(reduced_g2)
    polynomial = array_module.zeros_like(reduced_g2)
    derivative = array_module.zeros_like(reduced_g2)
    if count >= 1:
        polynomial += coefficients[0]
    if count >= 2:
        polynomial += coefficients[1]*(3.0-reduced_g2)
        derivative -= coefficients[1]
    if count >= 3:
        polynomial += coefficients[2]*(
            15.0-10.0*reduced_g2+reduced_g2**2
        )
        derivative += coefficients[2]*(-10.0+2.0*reduced_g2)
    if count >= 4:
        polynomial += coefficients[3]*(
            105.0-105.0*reduced_g2+21.0*reduced_g2**2-reduced_g2**3
        )
        derivative += coefficients[3]*(
            -105.0+42.0*reduced_g2-3.0*reduced_g2**2
        )
    return polynomial, derivative


def _fused_pp_nonlocal_strain(
    module, cell, mesh, density_sets, *, kpts=None, return_profile=False,
    device_profiler=None, response_backend="centered_cell_displacement",
    host_block_size=8192, gpu_block_size=None,
):
    """Reuse every strained-cell nonlocal projector for all densities."""

    profiler = device_profiler or CUDAEventProfiler(False)
    if kpts is None:
        if density_sets.ndim != 3:
            raise ValueError("Gamma PP densities must have shape (ndm,nao,nao)")
        densities = density_sets[:, None]
        points = np.zeros((1, 3))
    else:
        if density_sets.ndim != 4:
            raise ValueError(
                "sampled-k PP densities must have shape (ndm,nkpts,nao,nao)"
            )
        densities = density_sets
        points = np.asarray(kpts, dtype=float).reshape(-1, 3)
    if densities.shape[1] != len(points):
        raise ValueError("PP density and k-point counts differ")
    if response_backend not in {
        "centered_cell_displacement", "host_analytic_matrix", "gpu_analytic",
    }:
        raise ValueError(
            "nonlocal response backend must be 'centered_cell_displacement' "
            "'host_analytic_matrix', or 'gpu_analytic'"
        )
    if response_backend == "host_analytic_matrix":
        result, profile = _host_analytic_nonlocal_strain(
            cell, mesh, densities, points, block_size=host_block_size,
            device_profiler=profiler,
        )
        if return_profile:
            return result, profile
        return result
    if response_backend == "gpu_analytic":
        analytic_cell = cell
        analytic_densities = densities
        if not cell.cart:
            c2s = module.asarray(cell.cart2sph_coeff())
            analytic_densities = cp.stack(tuple(
                cp.stack(tuple(
                    module.sandwich_dot(density, c2s.T)
                    for density in density_batch
                ))
                for density_batch in densities
            ))
            analytic_cell = cell.copy()
            analytic_cell.cart = True
        result, profile = _gpu_analytic_nonlocal_strain(
            module, analytic_cell, mesh, analytic_densities, points,
            block_size=gpu_block_size, device_profiler=profiler,
        )
        if return_profile:
            return result, profile
        return result

    fakemol = module.gto.Mole()
    fakemol._atm = np.zeros(
        (1, module.gto.ATM_SLOTS), dtype=np.int32
    )
    fakemol._bas = np.zeros(
        (1, module.gto.BAS_SLOTS), dtype=np.int32
    )
    ptr = module.gto.PTR_ENV_START
    fakemol._env = np.zeros(ptr+10)
    fakemol._bas[0, module.gto.NPRIM_OF] = 1
    fakemol._bas[0, module.gto.NCTR_OF] = 1
    fakemol._bas[0, module.gto.PTR_EXP] = ptr+3
    fakemol._bas[0, module.gto.PTR_COEFF] = ptr+4

    ngrids = int(np.prod(mesh))
    scaled_kpts = points.dot(cell.lattice_vectors().T)
    nkpts = len(points)
    ndensity = int(densities.shape[0])
    projector_atoms = [
        atom for atom in range(cell.natm)
        if _has_nonlocal_projector(cell, atom)
    ]
    projector_elements = {
        cell.atom_symbol(atom) for atom in projector_atoms
    }
    if not projector_atoms:
        work = {
            "strain_component_count": 6,
            "displaced_cell_evaluation_count": 0,
            "ft_ao_call_count": 0,
            "projector_template_build_count": 0,
            "projector_atom_contraction_count": 0,
            "explicit_host_to_device_bytes": 0,
            "explicit_device_to_host_bytes": 0,
            "estimated_explicit_live_workspace_bytes": 0,
            "structure_factor_device_cache_bytes": 0,
            "projector_atom_count": 0,
            "projector_element_count": 0,
            "projector_channel_group_count": 0,
            "projector_atom_batch_max": 0,
            "kpoint_count": int(nkpts),
            "density_count": int(ndensity),
            "cpu_projector_template_wall_s": 0.0,
            "device_to_host_scalar_wall_s": 0.0,
            "empty_projector_fast_path": True,
        }
        result = cp.zeros((ndensity, 3, 3), dtype=cp.float64)
        if return_profile:
            return result, work
        return result
    # Fractional atom coordinates and reciprocal mesh indices are fixed under
    # homogeneous strain, so SI is identical for all +/- cells.  Atoms with no
    # nonlocal channel do not need an SI row.
    structure_factor = cell.get_SI(
        mesh=mesh, atmlst=projector_atoms
    )
    structure_factor_device = profiler.call(
        "nonlocal_structure_factor_upload",
        lambda: module.asarray(structure_factor),
    )
    work = {
        "strain_component_count": 6,
        "displaced_cell_evaluation_count": 0,
        "ft_ao_call_count": 0,
        "projector_template_build_count": 0,
        "projector_atom_contraction_count": 0,
        "explicit_host_to_device_bytes": 0,
        "explicit_device_to_host_bytes": 0,
        "estimated_explicit_live_workspace_bytes": 0,
        "structure_factor_device_cache_bytes": int(
            structure_factor_device.nbytes
        ),
        "projector_atom_count": int(len(projector_atoms)),
        "projector_element_count": int(len(projector_elements)),
        "kpoint_count": int(nkpts),
        "density_count": int(ndensity),
        "empty_projector_fast_path": False,
    }
    work["explicit_host_to_device_bytes"] += int(structure_factor.nbytes)
    cpu_template_wall_s = 0.0
    device_to_host_wall_s = 0.0

    def evaluate(strained_cell):
        nonlocal cpu_template_wall_s, device_to_host_wall_s
        work["displaced_cell_evaluation_count"] += 1
        volume = strained_cell.vol
        reciprocal = strained_cell.reciprocal_vectors(norm_to=1)
        gvectors = strained_cell.get_Gv(mesh)
        values = np.zeros(ndensity)
        for kpoint, density_batch in enumerate(
            densities.transpose(1, 0, 2, 3)
        ):
            cartesian_kpoint = scaled_kpts[kpoint].dot(reciprocal)
            gplusk = gvectors+cartesian_kpoint
            radial_g = module.lib.norm(gplusk, axis=1)
            ao_fourier = profiler.call(
                "nonlocal_ft_ao",
                lambda: module.ft_ao.ft_ao(
                    strained_cell, gvectors, kpt=cartesian_kpoint
                )*(1/volume)**0.5,
            )
            work["ft_ao_call_count"] += 1
            templates = {}
            for structure_index, atom in enumerate(projector_atoms):
                symbol = strained_cell.atom_symbol(atom)
                pseudo = strained_cell._pseudo[symbol]
                if symbol not in templates:
                    template_started = perf_counter()
                    template, layout = _nonlocal_projector_template(
                        module, fakemol, ptr, gplusk, radial_g, pseudo
                    )
                    cpu_template_wall_s += perf_counter()-template_started
                    work["projector_template_build_count"] += 1
                    work["explicit_host_to_device_bytes"] += int(
                        template.nbytes
                    )
                    templates[symbol] = (
                        profiler.call(
                            "nonlocal_template_upload",
                            lambda template=template: module.asarray(template),
                        ),
                        layout,
                    )
                template_device, layout = templates[symbol]
                if not layout:
                    continue
                projector_g = profiler.call(
                    "nonlocal_phase_apply",
                    lambda: (
                        template_device
                        * structure_factor_device[structure_index].conj()
                    ),
                )
                work["projector_atom_contraction_count"] += 1
                projector_ao = profiler.call(
                    "nonlocal_projector_ao",
                    lambda: projector_g.dot(ao_fourier),
                )
                projected_device = profiler.call(
                    "nonlocal_density_contraction",
                    lambda: cp.einsum(
                        "pi,dij,qj->dpq", projector_ao, density_batch,
                        projector_ao.conj(), optimize=True,
                    ).real,
                )
                work["explicit_device_to_host_bytes"] += int(
                    projected_device.nbytes
                )
                retained_template_bytes = sum(
                    int(item[0].nbytes)
                    for item in templates.values()
                )
                live_workspace = (
                    int(structure_factor_device.nbytes)
                    + retained_template_bytes+int(ao_fourier.nbytes)
                    + int(projector_g.nbytes)
                    + int(projector_ao.nbytes)+int(projected_device.nbytes)
                )
                work["estimated_explicit_live_workspace_bytes"] = max(
                    work["estimated_explicit_live_workspace_bytes"],
                    live_workspace,
                )
                transfer_started = perf_counter()
                projected_density = projected_device.get()
                device_to_host_wall_s += perf_counter()-transfer_started
                for start, end, count, multiplicity, coupling in layout:
                    subblock = projected_density[:, start:end, start:end]
                    subblock = subblock.reshape(
                        ndensity, count, multiplicity, count, multiplicity
                    )
                    values += np.einsum(
                        "ij,djmim->d", np.asarray(coupling), subblock
                    )
                del projected_device, projector_ao, projector_g
        return values/(nkpts*volume)

    displacement = max(1e-5, (cell.precision*0.1)**0.5)
    output = np.empty((ndensity, 3, 3))
    components = (
        ((0, 0, 1.0),),
        ((1, 1, 1.0),),
        ((2, 2, 1.0),),
        ((1, 2, 0.5), (2, 1, 0.5)),
        ((0, 2, 0.5), (2, 0, 0.5)),
        ((0, 1, 0.5), (1, 0, 0.5)),
    )
    for terms in components:
        plus, minus = _finite_diff_symmetric_cells(
            module, cell, terms, displacement
        )
        derivative = (
            evaluate(plus)-evaluate(minus)
        )/(2*displacement)
        row, column, _ = terms[0]
        output[:, row, column] = derivative
        output[:, column, row] = derivative
    result = cp.asarray(output)
    work["cpu_projector_template_wall_s"] = float(cpu_template_wall_s)
    work["device_to_host_scalar_wall_s"] = float(device_to_host_wall_s)
    if return_profile:
        return result, work
    return result


def _host_analytic_nonlocal_strain(
    cell, mesh, densities, points, *, block_size, device_profiler,
):
    """Build the exact host nonlocal PP derivative and contract it on GPU."""

    from pyscf_pbc_dcdft_stress.gdf_pp_strain import (
        kpoint_gdf_nonlocal_strain,
    )

    size = int(block_size)
    if size <= 0:
        raise ValueError("nonlocal host block size must be positive")
    started = perf_counter()
    primitive = kpoint_gdf_nonlocal_strain(
        cell, points, mesh=mesh, block_size=size,
    )
    host_wall = perf_counter()-started
    derivative_host = np.asarray(primitive.raw_derivative)
    derivative_device = device_profiler.call(
        "nonlocal_analytic_derivative_upload",
        lambda: cp.asarray(derivative_host),
    )
    contracted = device_profiler.call(
        "nonlocal_analytic_density_contraction",
        lambda: cp.einsum(
            "xykij,dkji->dxy", derivative_device, densities,
            optimize=True,
        ).real/len(points),
    )
    contracted = 0.5*(contracted+contracted.transpose(0, 2, 1))
    grid_points = int(np.prod(mesh))
    profile = {
        "owner": "host_analytic_fourier_projector_strain_device_contracted",
        "strain_component_count": 6,
        "displaced_cell_evaluation_count": 0,
        "ft_ao_call_count": 0,
        "host_ft_ao_block_count": int(
            (grid_points+size-1)//size*len(points)
        ),
        "projector_template_build_count": 0,
        "projector_atom_contraction_count": 0,
        "explicit_host_to_device_bytes": int(derivative_host.nbytes),
        "explicit_device_to_host_bytes": 0,
        "estimated_explicit_live_workspace_bytes": int(
            derivative_device.nbytes+contracted.nbytes
        ),
        "structure_factor_device_cache_bytes": 0,
        "projector_atom_count": int(sum(
            _has_nonlocal_projector(cell, atom) for atom in range(cell.natm)
        )),
        "projector_element_count": int(len({
            cell.atom_symbol(atom) for atom in range(cell.natm)
            if _has_nonlocal_projector(cell, atom)
        })),
        "kpoint_count": int(len(points)),
        "density_count": int(densities.shape[0]),
        "cpu_projector_template_wall_s": float(host_wall),
        "host_analytic_primitive_wall_s": float(host_wall),
        "device_to_host_scalar_wall_s": 0.0,
        "host_block_size": size,
        "empty_projector_fast_path": not bool(primitive.grid_points),
    }
    return contracted, profile


def _gpu_analytic_nonlocal_strain(
    module, cell, mesh, densities, points, *, block_size, device_profiler,
):
    """Contract the analytic GTH nonlocal strain without AO derivative matrices.

    GPU4PySCF does not currently expose ``ft_ao_ip1`` for periodic AOs.  For a
    Cartesian Gaussian, multiplication by ``(r_y-R_y)`` raises one Cartesian
    exponent.  Fourier transforming a copy of the cell whose shell angular
    momenta are raised by one therefore supplies the exact first moment.  The
    six symmetric strain components are accumulated directly into projector
    overlaps and finally contracted with all requested densities on device.

    Small radial/angular projector templates are still generated on the host.
    They are independent of atom count and are transferred once per reciprocal
    block and k point; the large AO Fourier transforms and density contractions
    remain on the GPU.
    """

    from pyscf_pbc_dcdft_stress.gdf_pp_strain import (
        _projector_angular_molecule,
        _projector_value_gradient,
    )

    grid_vectors = np.asarray(cell.get_Gv(mesh), dtype=float)
    ngrids = len(grid_vectors)
    size = ngrids if block_size is None else int(block_size)
    if size <= 0:
        raise ValueError("nonlocal GPU block size must be positive")
    size = min(size, max(1, ngrids))
    nkpts = len(points)
    ndensity = int(densities.shape[0])
    nao = int(densities.shape[-1])
    projector_atoms = [
        atom for atom in range(cell.natm)
        if _has_nonlocal_projector(cell, atom)
    ]
    projector_elements = {
        cell.atom_symbol(atom) for atom in projector_atoms
    }
    if not projector_atoms:
        return cp.zeros((ndensity, 3, 3), dtype=cp.float64), {
            "owner": "gpu_analytic_augmented_shell_fourier_moment_contracted",
            "strain_component_count": 6,
            "displaced_cell_evaluation_count": 0,
            "ft_ao_call_count": 0,
            "moment_ft_ao_call_count": 0,
            "projector_template_build_count": 0,
            "projector_atom_contraction_count": 0,
            "explicit_host_to_device_bytes": 0,
            "explicit_device_to_host_bytes": 0,
            "estimated_explicit_live_workspace_bytes": 0,
            "structure_factor_device_cache_bytes": 0,
            "projector_atom_count": 0,
            "projector_element_count": 0,
            "kpoint_count": int(nkpts),
            "density_count": int(ndensity),
            "gpu_block_size": int(size),
            "gpu_block_count": 0,
            "cpu_projector_template_wall_s": 0.0,
            "device_to_host_scalar_wall_s": 0.0,
            "empty_projector_fast_path": True,
            "host_boundaries": [],
        }

    moment_cell, moment_indices, moment_scales = (
        _augmented_cartesian_moment_cell(module, cell)
    )
    moment_indices_device = cp.asarray(moment_indices)
    moment_scales_device = cp.asarray(moment_scales)
    atom_coordinates = cp.asarray(
        np.asarray(cell.atom_coords(), dtype=float)[projector_atoms]
    )
    inverse_sqrt_volume = float(cell.vol)**-0.5
    fake = _projector_angular_molecule()
    channel_groups = {}
    for structure_index, atom in enumerate(projector_atoms):
        symbol = cell.atom_symbol(atom)
        for angular, projector_data in enumerate(cell._pseudo[symbol][5:]):
            radius, count, coupling = projector_data
            if int(count) <= 0:
                continue
            key = (symbol, int(angular), int(count), float(radius))
            if key not in channel_groups:
                channel_groups[key] = {
                    "key": key,
                    "structure_indices": [],
                    "symbol": symbol,
                    "angular": int(angular),
                    "radius": float(radius),
                    "count": int(count),
                    "coupling": cp.asarray(np.asarray(coupling, dtype=float)),
                }
            channel_groups[key]["structure_indices"].append(structure_index)
    channels = []
    for channel in channel_groups.values():
        atom_count = len(channel["structure_indices"])
        count = channel["count"]
        multiplicity = 2*channel["angular"]+1
        channel["structure_indices"] = cp.asarray(
            channel["structure_indices"], dtype=cp.int64
        )
        channel["overlap"] = cp.zeros(
            (nkpts, atom_count, count, multiplicity, nao),
            dtype=cp.complex128,
        )
        channel["response"] = cp.zeros(
            (6, nkpts, atom_count, count, multiplicity, nao),
            dtype=cp.complex128,
        )
        channels.append(channel)

    components = (
        ((0, 0, 1.0),),
        ((1, 1, 1.0),),
        ((2, 2, 1.0),),
        ((1, 2, 0.5), (2, 1, 0.5)),
        ((0, 2, 0.5), (2, 0, 0.5)),
        ((0, 1, 0.5), (1, 0, 0.5)),
    )
    work = {
        "owner": "gpu_analytic_augmented_shell_fourier_moment_contracted",
        "strain_component_count": 6,
        "displaced_cell_evaluation_count": 0,
        "ft_ao_call_count": 0,
        "moment_ft_ao_call_count": 0,
        "projector_template_build_count": 0,
        "projector_atom_contraction_count": 0,
        "explicit_host_to_device_bytes": int(
            moment_indices.nbytes+moment_scales.nbytes
        ),
        "explicit_device_to_host_bytes": 0,
        "estimated_explicit_live_workspace_bytes": 0,
        "structure_factor_device_cache_bytes": 0,
        "projector_atom_count": int(len(projector_atoms)),
        "projector_element_count": int(len(projector_elements)),
        "projector_channel_group_count": int(len(channels)),
        "projector_atom_batch_max": int(max(
            len(channel["structure_indices"]) for channel in channels
        )),
        "kpoint_count": int(nkpts),
        "density_count": int(ndensity),
        "gpu_block_size": int(size),
        "gpu_block_count": 0,
        "cpu_projector_template_wall_s": 0.0,
        "device_to_host_scalar_wall_s": 0.0,
        "empty_projector_fast_path": False,
        "host_boundaries": ["small_projector_value_gradient_template"],
    }
    retained_channel_bytes = sum(
        int(channel["structure_indices"].nbytes+channel["coupling"].nbytes
            + channel["overlap"].nbytes+channel["response"].nbytes)
        for channel in channels
    )

    for p0 in range(0, ngrids, size):
        p1 = min(ngrids, p0+size)
        work["gpu_block_count"] += 1
        base_host = grid_vectors[p0:p1]
        base_device = cp.asarray(base_host)
        structure_factors = device_profiler.call(
            "nonlocal_analytic_structure_factor",
            lambda: cp.exp(-1j*atom_coordinates.dot(base_device.T)),
        )
        work["structure_factor_device_cache_bytes"] = max(
            work["structure_factor_device_cache_bytes"],
            int(structure_factors.nbytes),
        )
        for kpoint_id, kpt in enumerate(points):
            momentum_host = base_host+np.asarray(kpt, dtype=float)
            momentum_device = base_device+cp.asarray(kpt, dtype=float)
            ao_value = device_profiler.call(
                "nonlocal_analytic_ft_ao",
                lambda: module.ft_ao.ft_ao(
                    cell, base_host, kpt=np.asarray(kpt, dtype=float)
                )*inverse_sqrt_volume,
            )
            work["ft_ao_call_count"] += 1
            moment_all = device_profiler.call(
                "nonlocal_analytic_moment_ft_ao",
                lambda: module.ft_ao.ft_ao(
                    moment_cell, base_host, kpt=np.asarray(kpt, dtype=float)
                ),
            )
            work["moment_ft_ao_call_count"] += 1
            moments = device_profiler.call(
                "nonlocal_analytic_moment_gather",
                lambda: tuple(
                    cp.take(moment_all, moment_indices_device[axis], axis=1)
                    * moment_scales_device[None, :]*inverse_sqrt_volume
                    for axis in range(3)
                ),
            )

            template_started = perf_counter()
            templates = {}
            for channel in channels:
                projector, gradient = _projector_value_gradient(
                    fake, momentum_host, channel["angular"],
                    channel["count"], channel["radius"],
                )
                templates[channel["key"]] = device_profiler.call(
                    "nonlocal_analytic_template_upload",
                    lambda projector=projector, gradient=gradient: (
                        cp.asarray(projector), cp.asarray(gradient)
                    ),
                )
                work["projector_template_build_count"] += 1
                work["explicit_host_to_device_bytes"] += int(
                    projector.nbytes+gradient.nbytes
                )
            work["cpu_projector_template_wall_s"] += (
                perf_counter()-template_started
            )

            for channel in channels:
                projector, _ = templates[channel["key"]]
                phase = structure_factors[
                    channel["structure_indices"]
                ].conj()
                phased_projector = device_profiler.call(
                    "nonlocal_analytic_phase_apply",
                    lambda: projector[None]*phase[:, None, None, :],
                )
                channel["overlap"][kpoint_id] += device_profiler.call(
                    "nonlocal_analytic_projector_overlap",
                    lambda: cp.einsum(
                        "znmg,gp->znmp", phased_projector, ao_value,
                        optimize=True,
                    ),
                )
                work["projector_atom_contraction_count"] += int(
                    len(channel["structure_indices"])
                )

            for component_id, terms in enumerate(components):
                def build_ao_response():
                    response = cp.zeros_like(ao_value)
                    for x, y, weight in terms:
                        response += (
                            1j*weight*momentum_device[:, x, None]*moments[y]
                        )
                    if component_id < 3:
                        response -= 0.5*ao_value
                    return response

                ao_response = device_profiler.call(
                    "nonlocal_analytic_ao_response_build", build_ao_response
                )
                for channel in channels:
                    projector, gradient = templates[channel["key"]]
                    phase = structure_factors[
                        channel["structure_indices"]
                    ].conj()
                    phased_projector = projector[None]*phase[:, None, None, :]

                    def build_projector_response():
                        response = cp.zeros_like(gradient[0])
                        for x, y, weight in terms:
                            response -= (
                                weight*momentum_device[:, x][None, None, :]
                                * gradient[y]
                            )
                        return response

                    projector_response = device_profiler.call(
                        "nonlocal_analytic_projector_response_build",
                        build_projector_response,
                    )
                    phased_response = device_profiler.call(
                        "nonlocal_analytic_response_phase_apply",
                        lambda: (
                            projector_response[None]
                            * phase[:, None, None, :]
                        ),
                    )
                    channel["response"][component_id, kpoint_id] += (
                        device_profiler.call(
                            "nonlocal_analytic_projector_response_overlap",
                            lambda: cp.einsum(
                                "znmg,gp->znmp", phased_response, ao_value,
                                optimize=True,
                            ),
                        )
                    )
                    channel["response"][component_id, kpoint_id] += (
                        device_profiler.call(
                            "nonlocal_analytic_ao_response_overlap",
                            lambda: cp.einsum(
                                "znmg,gp->znmp", phased_projector, ao_response,
                                optimize=True,
                            ),
                        )
                    )

            template_bytes = sum(
                int(projector.nbytes+gradient.nbytes)
                for projector, gradient in templates.values()
            )
            live_bytes = (
                retained_channel_bytes+int(base_device.nbytes)
                + int(structure_factors.nbytes)+int(ao_value.nbytes)
                + int(moment_all.nbytes)
                + sum(int(moment.nbytes) for moment in moments)
                + template_bytes*(1+work["projector_atom_batch_max"])
            )
            work["estimated_explicit_live_workspace_bytes"] = max(
                work["estimated_explicit_live_workspace_bytes"], live_bytes
            )
            del templates, moments, moment_all, ao_value
        del structure_factors, base_device

    values = cp.zeros(ndensity, dtype=cp.complex128)
    derivatives = cp.zeros((ndensity, 6), dtype=cp.complex128)
    for channel in channels:
        overlap = channel["overlap"]
        response = channel["response"]
        coupling = channel["coupling"]
        values += device_profiler.call(
            "nonlocal_analytic_density_energy_contraction",
            lambda: cp.einsum(
                "kzamp,ab,kzbmq,dkqp->d", overlap.conj(), coupling,
                overlap, densities, optimize=True,
            ),
        )
        derivatives += device_profiler.call(
            "nonlocal_analytic_density_response_left",
            lambda: cp.einsum(
                "ckzamp,ab,kzbmq,dkqp->dc", response.conj(), coupling,
                overlap, densities, optimize=True,
            ),
        )
        derivatives += device_profiler.call(
            "nonlocal_analytic_density_response_right",
            lambda: cp.einsum(
                "kzamp,ab,ckzbmq,dkqp->dc", overlap.conj(), coupling,
                response, densities, optimize=True,
            ),
        )
    scale = 1.0/(nkpts*float(cell.vol))
    values *= scale
    derivatives *= scale
    derivatives[:, :3] -= values[:, None]
    result = cp.zeros((ndensity, 3, 3), dtype=cp.float64)
    result[:, 0, 0] = derivatives[:, 0].real
    result[:, 1, 1] = derivatives[:, 1].real
    result[:, 2, 2] = derivatives[:, 2].real
    result[:, 1, 2] = result[:, 2, 1] = derivatives[:, 3].real
    result[:, 0, 2] = result[:, 2, 0] = derivatives[:, 4].real
    result[:, 0, 1] = result[:, 1, 0] = derivatives[:, 5].real
    work["estimated_explicit_live_workspace_bytes"] = max(
        work["estimated_explicit_live_workspace_bytes"],
        retained_channel_bytes+int(values.nbytes+derivatives.nbytes+result.nbytes),
    )
    return result, work


def _augmented_cartesian_moment_cell(module, cell):
    """Return an ``l+1`` shell cell and maps for Cartesian AO first moments."""

    angular = np.asarray(cell._bas[:, module.gto.ANG_OF], dtype=int)
    max_supported = int(getattr(module.ft_ao, "LMAX", 4))
    if angular.size and int(angular.max())+1 > max_supported:
        raise NotImplementedError(
            "analytic GPU nonlocal PP strain requires l+1 <= "
            f"gpu4pyscf.pbc.df.ft_ao.LMAX={max_supported}; "
            f"encountered AO angular momentum l={int(angular.max())}"
        )
    moment_cell = cell.copy(deep=True)
    moment_cell.cart = True
    moment_cell._bas[:, module.gto.ANG_OF] += 1
    original_loc = np.asarray(cell.ao_loc_nr(cart=True), dtype=int)
    moment_loc = np.asarray(moment_cell.ao_loc_nr(cart=True), dtype=int)
    nao = int(original_loc[-1])
    indices = np.empty((3, nao), dtype=np.int64)
    scales = np.empty(nao, dtype=float)
    for shell_id, l_value in enumerate(angular):
        l_value = int(l_value)
        original_exponents = _cartesian_exponents(l_value)
        moment_exponents = _cartesian_exponents(l_value+1)
        moment_lookup = {
            exponent: index for index, exponent in enumerate(moment_exponents)
        }
        original_count = len(original_exponents)
        moment_count = len(moment_exponents)
        contractions = int(cell._bas[shell_id, module.gto.NCTR_OF])
        scale = _gpu_ft_shell_scale(l_value)/_gpu_ft_shell_scale(l_value+1)
        for contraction in range(contractions):
            original_start = original_loc[shell_id]+contraction*original_count
            moment_start = moment_loc[shell_id]+contraction*moment_count
            for local_id, exponent in enumerate(original_exponents):
                ao_id = original_start+local_id
                scales[ao_id] = scale
                for axis in range(3):
                    raised = list(exponent)
                    raised[axis] += 1
                    indices[axis, ao_id] = (
                        moment_start+moment_lookup[tuple(raised)]
                    )
    return moment_cell, indices, scales


def _cartesian_exponents(angular_momentum):
    """PySCF Cartesian AO exponent ordering for one shell."""

    result = []
    for x_power in range(int(angular_momentum), -1, -1):
        for y_power in range(int(angular_momentum)-x_power, -1, -1):
            result.append((
                x_power, y_power,
                int(angular_momentum)-x_power-y_power,
            ))
    return tuple(result)


def _gpu_ft_shell_scale(angular_momentum):
    """Normalization applied internally by GPU4PySCF's FT-AO kernel."""

    angular_momentum = int(angular_momentum)
    if angular_momentum < 2:
        return ((2*angular_momentum+1)/(4*np.pi))**0.5
    return 1.0


def _has_nonlocal_projector(cell, atom):
    symbol = cell.atom_symbol(atom)
    if symbol not in cell._pseudo:
        return False
    return any(projector[1] > 0 for projector in cell._pseudo[symbol][5:])


def _nonlocal_projector_template(
    module, fakemol, ptr, gplusk, radial_g, pseudo,
):
    """Build atom-independent radial/angular projectors for one element."""

    blocks = []
    layout = []
    end = 0
    for angular, projector in enumerate(pseudo[5:]):
        radius, count, coupling = projector
        if count <= 0:
            continue
        fakemol._bas[0, module.gto.ANG_OF] = angular
        fakemol._env[ptr+3] = 0.5*radius**2
        fakemol._env[ptr+4] = radius**(angular+1.5)*np.pi**1.25
        solid_harmonic = fakemol.eval_gto("GTOval", gplusk)
        multiplicity = 2*angular+1
        block = np.empty(
            (count, multiplicity, len(radial_g)), dtype=np.complex128
        )
        for order in range(count):
            qvalue = module.pseudo.pp._qli(
                radial_g*radius, angular, order
            )
            block[order] = solid_harmonic.T*qvalue
        flattened = block.reshape(count*multiplicity, len(radial_g))
        blocks.append(flattened)
        start = end
        end += len(flattened)
        layout.append((
            start, end, count, multiplicity, np.asarray(coupling)
        ))
    if not blocks:
        return np.empty((0, len(radial_g)), dtype=np.complex128), ()
    return np.concatenate(blocks, axis=0), tuple(layout)


def _finite_diff_symmetric_cells(module, cell, terms, displacement):
    """Build a +/- cell pair for one symmetric Voigt strain coordinate."""

    lattice = np.asarray(cell.lattice_vectors(), dtype=float)
    coordinates = np.asarray(cell.atom_coords(), dtype=float)
    if not module.gto.mole.is_au(cell.unit):
        lattice *= module.lib.param.BOHR
        coordinates *= module.lib.param.BOHR

    def strained(sign):
        deformation = np.eye(3)
        for row, column, weight in terms:
            deformation[row, column] += sign*displacement*weight
        result = cell.set_geom_(
            coordinates.dot(deformation.T), inplace=False
        )
        result.a = lattice.dot(deformation.T)
        if cell.space_group_symmetry:
            result.build(False, False)
        return result

    return strained(1.0), strained(-1.0)


def _tensor_to_voigt(tensor):
    value = cp.asarray(tensor)
    if value.shape != (3, 3):
        raise ValueError(f"strain tensor must have shape (3, 3); got {value.shape}")
    return cp.stack((
        value[0, 0], value[1, 1], value[2, 2],
        0.5*(value[1, 2]+value[2, 1]),
        0.5*(value[0, 2]+value[2, 0]),
        0.5*(value[0, 1]+value[1, 0]),
    )).real


__all__ = [
    "ContractedPseudopotentialStrain",
    "contracted_gpu4pyscf_pp_strain",
]
