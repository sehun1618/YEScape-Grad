# S0/S1: Stress Convention And FD Oracle

The first stress implementation is intentionally independent of the analytic
gradient packages. It validates energy derivatives under homogeneous strain
and does not claim production efficiency.

For PySCF row-vector lattices, the package applies

$$
a' = a(I + \epsilon)^T,
$$

then reconstructs Cartesian atom positions from unchanged fractional
coordinates. Off-diagonal Voigt coordinates are engineering shear, so an
`xy` perturbation of magnitude $q$ applies
$\epsilon_{xy}=\epsilon_{yx}=q/2$. This makes
$\partial E/\partial q=\Omega\sigma_{xy}$ without an extra factor in the
reported physical stress.

Each central-FD pair reruns source HF and target DC-DFT energy. The production
validation policy uses the same integer mesh for the plus and minus cells.
Allowing PySCF to recalculate a mesh from `ke_cutoff` is available only as a
diagnostic because an integer mesh jump can dominate the numerical derivative.

The S1 public scope is restricted Gamma/GDF with uniform target integration.
RSDF, sampled k-points, r2SCAN and GPU are deliberately rejected until their
respective stages in `stress/Goal.md`.
