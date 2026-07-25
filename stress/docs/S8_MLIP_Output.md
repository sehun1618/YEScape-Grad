# S8 MLIP Energy, Force, Stress, and Virial Output

`cpu_energy_force_stress`와 `gpu_energy_force_stress`는 이미 수렴한 하나의 DC-DFT
method만 받는다. 내부에서 source HF SCF를 다시 실행하지 않으며 nuclear gradient에서
구한 adjoint Z-vector를 analytic stress에 전달한다.

## Stored quantities

- energy: Hartree
- positions/lattice: Bohr
- force: Hartree/Bohr, $F_A=-\partial E/\partial R_A$
- tensile-positive stress: Hartree/Bohr$^3$
- virial: Hartree, $W=-\Omega\sigma$
- cell volume and SHA-256 state fingerprint

Fingerprint에는 lattice, atom coordinates, k-points, converged density, target XC와 DF
backend가 들어간다. 서로 다른 electronic state의 energy/force/stress를 실수로 한 frame에
섞었는지 추적하기 위한 provenance 값이지 물리적 observable은 아니다.

JSON은 위 native atomic units를 보존한다. extxyz는 position/lattice를 Angstrom, energy를
eV, force를 eV/Angstrom으로 변환한다. Stress는 표준 reader마다 virial/stress 부호 해석이
다를 수 있으므로 `dcdft_stress_tensile_eV_per_A3`라는 명시적 field로 저장하고,
`stress_convention=tensile_positive`를 함께 쓴다.

## Same-state reuse

Energy는 이미 수렴한 method의 `e_tot`을 사용한다. Gradient가 target Fock/AX를 통해 구한
Z-vector는 stress에 그대로 전달된다. Stress는 cell derivative primitive만 새로 만들고
Z-vector equation을 다시 풀지 않는다. GPU H2 smoke에서 공유 전후 stress 차이는
$6.44\times10^{-17}$ Eh/Bohr$^3$였고 target-Fock/Z 단계는 0.080 s에서 0.00083 s로
줄었다.

