# gpu4pyscf-pbc-dcdft

`pyscf-pbc-dcdft`의 restricted periodic DC-DFT GPU extension이다. PBE, SCAN,
r2SCAN target에 대해 FFTDF, GDF, RSDF의 Gamma/sampled-k-point energy와 fixed-cell
analytic nuclear gradient를 제공한다.

```python
from gpu4pyscf_pbc_dcdft import df, dft

mf = dft.DCKRKS(cell, kpts=kpts, xc="R2SCAN")
mf.with_df = df.RSDF(cell, kpts=kpts, auxbasis="gth-szv")
mf.conv_tol = 1.0e-11
mf.conv_tol_grad = 1.0e-9

energy = mf.kernel()                         # Hartree
gradient = mf.nuc_grad_method().kernel()    # Hartree/Bohr
```

FFTDF는 기본 provider이며 GDF/RSDF는 각각 `density_fit()`과
`rs_density_fit()`으로 선택할 수 있다. GPU route는 CPU DC-DFT 계산으로 조용히
fallback하지 않는다.

D2C-D4를 추가하려면 energy kernel 전에 다음 opt-in을 사용한다.

```python
mf.set_dispersion("d2c")
```

전자구조와 DC response는 계속 GPU 경로를 사용하고, 작은 geometry-only DFT-D4
energy/gradient 호출만 `dftd4` host library가 담당한다. 이 옵션은 기본적으로 꺼져
있으며 `dftd4>=4.2`가 필요하다.

Source HF는 GDF/RSDF를 유지하면서 fixed-density target DFA의 XC만 multigrid로
바꾸려면 다음 explicit opt-in을 사용한다.

```python
mf = dft.DCKRKS(cell, kpts=kpts, xc="R2SCAN").rs_density_fit("gth-szv")
mf.set_target_integrator("multigrid_xc")
```

이 mode에서 source HF J/K, target J, hcore/PP와 source Z-vector response는 선택한
GDF/RSDF가 계속 소유하고 target XC energy/Fock/gradient만 GPU
`multigrid_v2`가 담당한다. `multigrid_j_xc`도 제공하지만 target J representation까지
reciprocal multigrid로 바꾸므로 finite auxiliary GDF/RSDF와 같은 에너지를 보장하지
않는 diagnostic opt-in이다. 두 mode 모두 Gamma/sampled-k PBE/r2SCAN analytic-FD로
검증됐으며 CPU나 uniform NumInt로 silent fallback하지 않는다.

GDF의 기본 `setup_mode="native"`는 GPU4PySCF CUDA builder로 central CDERI를 만들고,
`get_j_e1`/`get_k_e1` 및 DC-DFT bilinear Z response를 source-contracted GPU derivative
kernel로 계산한다. Bilinear Z response의 기본 `gdf_bilinear_mode="one_pass"`는 Gamma와
sampled-k에서 SR/LR/metric/auxiliary/Ewald derivative source를 한 traversal로 합친다.
Exact polarization reference가 필요하면 gradient object에
`gdf_bilinear_mode="two_pass"`를 지정한다. CPU-identical full-$dCDERI$ 검증 경로가
필요하면 명시적으로 다음을 사용한다.

성분별 target-J와 Z-bilinear 값을 따로 보존할 필요가 없는 production timing에서는
`gdf_bilinear_mode="one_pass_fused"`를 선택할 수 있다. 이 mode는 target Coulomb
derivative와 signed Z-bilinear source를 SR/LR/metric/auxiliary/Ewald kernel 호출 전에
합쳐 한 traversal로 계산한다. 최종 gradient는 동일하지만 두 항은 사후 분리할 수
없으므로 결과의 `contributions.coupled`에 합을 저장하고 metadata의
`target_coulomb_component_resolved`를 `false`로 기록한다.

```python
mf.with_df = df.GDF(
    cell,
    kpts=kpts,
    auxbasis="gth-szv",
    setup_mode="reference",
)
```

Native GDF에서는 local pseudopotential part 1의 short-range/long-range nuclear
derivative도 source-contracted CUDA kernel로 계산하므로 CPU `_ft_ip_ovlp` full-H1
tensor를 만들지 않는다. Cell/AuxCell metadata, mapped-host CDERI bookkeeping,
local pseudopotential part 2 H1 primitive, sampled-k metric derivative의 upstream host
primitive와 Ewald nuclear term은 explicit host boundary다. CPU DC-DFT
energy/gradient로 silent fallback하지 않는다.

RSDF production gradient는 source와 metric response를 full atom-resolved
`dCDERI/dJ/dK`로 만들지 않는다. Target-J와 restricted-HF Z-bilinear의 CDERI adjoint를
먼저 더한 뒤 native short-range/reciprocal source derivative와 reciprocal
metric/Cholesky adjoint에 한 번만 수축한다. Metadata의
`response_mode="native_source_native_metric_fused_adjoint"`와
`gradient_partition="explicit_response_coupled"`가 이 경로를 표시한다. Central RSDF
CDERI setup과 production PySCF convention을 보존하는 exact screened short-range metric
`ip1/ip2`는 명시적인 host boundary다. 성분별 진단은 low-level builder의
`observable_mode="resolved"`를 사용한다.

RSDF central factor의 기본값은 정확도와 현재 end-to-end 성능이 고정된
`central_setup="reference"`다. Host memory가 제약일 때는 다음 opt-in을 사용할 수
있다.

```python
mf.rs_density_fit(
    "gth-szv",
    central_setup="exact_gpu_source",
)
```

이 mode는 PySCF로 source/metric parameter와 작은 screened two-center metric만
계산하고, 큰 three-center source, reciprocal metric 및 Cholesky/CDERI 변환을 GPU에서
수행한다. Gamma와 sampled-k에서 CPU central factor와 `1e-9` 이내로 검증되며,
upstream source-builder working memory는 bounded batch policy를 사용한다. 현재
Water64/SZ evidence에서는 host RSS는 감소했지만 5% end-to-end promotion gate를
통과하지 않았으므로 자동 선택하지 않는다.

Public examples:

```text
examples/00_fftdf_gpu_energy_gradient.py
examples/01_fftdf_gpu_kpoint.py
examples/02_gdf_gpu_energy_gradient.py
examples/03_gdf_gpu_kpoint.py
examples/04_cpu_gpu_parity.py
examples/05_gpu_timing_breakdown.py
examples/06_device_and_fallback_report.py
examples/07_mgga_scan_r2scan.py
examples/08_rsdf_gpu_energy_gradient.py
examples/09_gdf_target_multigrid.py
```

자세한 ownership과 equation mapping은 workspace의 `docs/implementation`을 참조한다.
