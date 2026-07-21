# pyscf-pbc-dcdft

PySCF-style CPU package for restricted periodic DC-DFT calculations.

The active snapshot supports fixed-density PBE, SCAN, and r2SCAN energy and
fixed-cell analytic nuclear gradients with FFTDF, GDF, and RSDF at Gamma and
sampled k-points. Full-energy central finite differences validate each promoted
route. Unsupported states or backends fail explicitly.

Published D2C-D4 parameters for HF-DFT can be added explicitly:

```python
mf = dft.DCRKS(cell, xc="PBE").set_dispersion("d2c")
energy = mf.kernel()                       # E_DC-PBE + E_D2C-D4
gradient = mf.nuc_grad_method().kernel()  # includes dE_D2C-D4/dR
```

This optional route requires `dftd4>=4.2`, available through the
`dispersion` package extra. It is disabled by default.

Runnable examples:

```text
examples/03_fftdf_cpu_energy_gradient.py
examples/04_gdf_cpu_energy_gradient.py
examples/05_custom_xyz_cell_basis_pseudo.py
examples/06_analytic_vs_fd_gradient.py
examples/07_cutoff_mesh_convergence.py
examples/08_kmesh_convergence.py
examples/09_save_energy_gradient_npy.py
examples/10_mgga_scan_r2scan.py
examples/11_rsdf_cpu_energy_gradient.py
examples/12_d2c_d4_energy_gradient.py
```

Energy is reported in Hartree and the returned nuclear gradient is
$dE/d\mathbf R$ in Hartree/Bohr, not force.
