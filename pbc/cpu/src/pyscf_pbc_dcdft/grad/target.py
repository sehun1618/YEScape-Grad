"""Fixed-density target-DFA derivative terms for restricted DC-DFT.

The explicit two-electron target contribution consists of the Hartree
derivative and the semilocal XC derivative evaluated at the HF AO density.
LDA/GGA use PySCF's periodic row-derivative implementation.  MGGA adds the
kinetic-energy-density response that upstream CPU PySCF currently omits.
Multiplying the atom-owned row contraction by two restores bra and ket terms.
"""

from __future__ import annotations

import numpy as np
from pyscf.grad import rks as molecular_rks_grad
from pyscf.pbc.dft import numint as pbc_numint
from pyscf.pbc.grad import krks as pyscf_krks_grad


def target_fock_kpts(context) -> np.ndarray:
    """Return target-DFA Fock matrices at the frozen HF density."""

    target = context.target_ks
    hcore = np.asarray(target.get_hcore())
    veff = np.asarray(target.get_veff(context.cell, context.source_mf.make_rdm1()))
    if hcore.ndim == 2:
        hcore = hcore[None]
    if veff.ndim == 2:
        veff = veff[None]
    return hcore + veff


def target_energy_weighted_density(context, target_fock) -> np.ndarray:
    """Build ``2 C_occ F_target,oo C_occ^H`` at every k-point."""

    output = []
    for coeff, occupation, fock in zip(
        context.mo_coeff,
        context.mo_occ,
        np.asarray(target_fock),
        strict=True,
    ):
        occupied = occupation > 0
        occupied_coeff = coeff[:, occupied]
        fock_oo = occupied_coeff.conj().T @ fock @ occupied_coeff
        output.append(2.0 * occupied_coeff @ fock_oo @ occupied_coeff.conj().T)
    return np.asarray(output)


def target_xc_row_derivative(context) -> np.ndarray:
    """Return AO-row derivatives of the target semilocal ``V_xc``.

    The output has shape ``(3,nkpts,nao,nao)``.  A ``KNumInt`` object is used
    even at Gamma so that the derivative code and the density both retain an
    explicit one-element k-point axis.  At ``Nk=1`` this is algebraically
    identical to the Gamma ``NumInt`` energy evaluation.
    """

    target = context.target_ks
    if target._numint.libxc.is_hybrid_xc(target.xc):
        raise NotImplementedError(
            "target derivative supports pure semilocal LDA/GGA/MGGA only"
        )
    grids = target.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)
    k_numint = pbc_numint.KNumInt(context.kpts)
    xctype = k_numint._xc_type(target.xc)
    if xctype == "MGGA":
        return _target_mgga_xc_row_derivative(context, k_numint)
    vxc = pyscf_krks_grad.get_vxc(
        k_numint,
        context.cell,
        grids,
        target.xc,
        context.density,
        context.kpts,
        max_memory=context.dc_mf.max_memory,
        verbose=context.dc_mf.verbose,
    )
    return np.asarray(vxc)


def _target_mgga_xc_row_derivative(context, k_numint) -> np.ndarray:
    r"""Return the MGGA AO-row derivative including the \(\tau\) channel.

    PySCF defines

    .. math::

        \tau(\mathbf r)=\frac12\sum_{k\mu\nu}D^k_{\mu\nu}
        \nabla\phi^*_{\mu k}(\mathbf r)\cdot
        \nabla\phi_{\nu k}(\mathbf r).

    Consequently the libxc weight associated with ``rho[4]`` receives one
    additional factor ``1/2``.  ``_tau_grad_dot_`` then contracts first- and
    second-AO derivatives into the nuclear AO-row response.  SCAN/r2SCAN do
    not require a density-Laplacian channel in PySCF's convention.
    """

    cell = context.cell
    grids = context.target_ks.grids
    make_rho, nset, nao = k_numint._gen_rho_evaluator(
        cell, context.density, hermi=1
    )
    ao_loc = cell.ao_loc_nr()
    output = np.zeros(
        (3, nset, context.nkpts, nao, nao),
        dtype=context.density.dtype,
    )
    for ao_k1, ao_k2, mask, weight, _coords in k_numint.block_loop(
        cell,
        grids,
        nao,
        2,
        context.kpts,
        None,
        context.dc_mf.max_memory,
    ):
        ao_k1 = np.asarray(ao_k1)
        ao_k2 = np.asarray(ao_k2)
        for set_id in range(nset):
            rho = make_rho(set_id, ao_k2[:, :4], mask, "MGGA")
            vxc = k_numint.eval_xc_eff(
                context.target_ks.xc,
                rho,
                deriv=1,
                xctype="MGGA",
                spin=0,
            )[1]
            weighted = vxc * weight
            weighted[0] *= 0.5
            weighted[4] *= 0.5
            for k_index in range(context.nkpts):
                molecular_rks_grad._gga_grad_sum_(
                    output[:, set_id, k_index],
                    cell,
                    ao_k1[k_index],
                    weighted[:4],
                    mask,
                    ao_loc,
                )
                molecular_rks_grad._tau_grad_dot_(
                    output[:, set_id, k_index],
                    cell,
                    ao_k1[k_index],
                    weighted[4],
                    mask,
                    ao_loc,
                    True,
                )
    if nset != 1:
        raise NotImplementedError("restricted DC-DFT expects one density set")
    return -output[:, 0]


def target_potential_row_derivative(context) -> np.ndarray:
    """Return FFTDF AO-row derivatives of target ``J + V_xc``.

    GDF does not expose ``get_j_e1``.  Its gradient route contracts the full
    ``dJ/dR`` matrices generated from ``dCDERI`` and calls
    :func:`target_xc_row_derivative` only for the semilocal XC term.
    """

    vxc = target_xc_row_derivative(context)
    coulomb = context.dc_mf.with_df.get_j_e1(
        context.density,
        context.kpts,
    )
    return vxc + np.asarray(coulomb)


def contract_target_potential(
    context,
    row_derivative: np.ndarray,
) -> np.ndarray:
    """Contract target J/XC row derivatives into ``(natm,3)`` gradients."""

    output = np.zeros((context.cell.natm, 3), dtype=float)
    ao_slices = np.asarray(context.cell.aoslice_by_atom(), dtype=int)
    for atom_id in range(context.cell.natm):
        p0, p1 = ao_slices[atom_id, 2:]
        # Factor two adds the derivative on the conjugate AO side.  Equal
        # k-point weights give the final 1/Nk normalization.
        output[atom_id] = (
            2.0
            * np.einsum(
                "xkij,kji->x",
                row_derivative[:, :, p0:p1, :],
                context.density[:, :, p0:p1],
                optimize=True,
            ).real
            / context.nkpts
        )
    return output
