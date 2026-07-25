# S7 GPU contracted response optimization (2026-07-22)

## 목적과 범위

Water8 baseline에서 analytic stress 시간의 대부분은 완성된 stress tensor를 만드는 데
필요하지 않은 큰 중간 행렬을 생성하는 데 사용됐다.

- GDF: host PP 1565.84 s, host DF 651.13 s
- RSDF: host PP 1575.75 s, host DF 121.26 s

Numerical mesh, basis, pseudopotential, DF representation과 Z-vector tolerance는 바꾸지
않았다. 대신 DC Lagrangian이 실제로 요구하는 density-contracted scalar만 직접 만든다.

## PP fused contraction

기존 GPU4PySCF 호출은 source/adjoint density 각각에 대해 XC-only와 XC+PP를 계산했다.
따라서 5,177,717-point mesh에서 MGGA $\rho$, $\nabla\rho$, $\tau$와 strain response를
네 번 만들었다.

새 경로는 다음처럼 동작한다.

1. Source target XC는 공식 GPU4PySCF MGGA 경로로 한 번 계산한다.
2. PP에는 scalar $\rho$와 $d\rho/d\epsilon$만 필요하므로 gradient/$\tau$ channel을
   만들지 않는다.
3. AO와 AO-strain을 grid block마다 한 번 평가한다.
4. 같은 AO block을 source density와 adjoint transition density에 함께 수축한다.
5. Local reciprocal PP와 nonlocal projector response를 각 density에 수축한다.

PP 단계는 GDF에서 173.53 s에서 108.13 s, RSDF에서 171.83 s에서 108.52 s로 각각
1.60배와 1.58배 빨라졌다. Fused 전후 전체 stress 최대 차이는 GDF
$2.03\times10^{-16}$, RSDF $2.54\times10^{-16}$ Eh/Bohr$^3$다.

### 후속 PP factor 최적화

- Source와 adjoint density가 같은 strained-cell nonlocal projector를 공유한다.
- 최종 stress가 요구하는 symmetric Voigt 6성분만 직접 미분해 9개 cell 방향을
  6개로 줄였다.
- 3D GTH local potential은 $G^2$의 명시적 함수이므로
  $dG^2/d\epsilon_{xy}=-2G_xG_y$ chain rule로 analytic derivative를 만든다.
- 같은 원소의 radial/angular projector는 strained cell과 k-point마다 한 번만 만들고,
  원자별 structure factor만 적용한다.
- Fractional atom coordinate와 reciprocal mesh index가 strain 동안 고정되므로 nonlocal
  structure factor는 모든 $+/-$ cell에서 재사용한다.

최종 paired run에서 PP 단계는 GDF 18.72 s, RSDF 17.88 s까지 감소했다.
GDF 세부 시간은 source XC 5.83 s, AO-density strain 2.05 s, local setup 1.30 s,
nonlocal projector 9.43 s다. RSDF도 같은 PP implementation을 사용한다.

## Fixed-density contracted DF

완전 analytic 기본 경로는 6개 strain의 2c/3c derivative와 Cholesky response를 만든 뒤
$dJ$, $dK$를 density와 수축한다. 새 opt-in 경로는 다음 두 scalar만 직접 미분한다.

$$
E_J[D]=\frac{1}{2N_k}\sum_k\operatorname{Tr}[D_kJ_k[D]],
$$

$$
L_{Z,\mathrm{RHF}}[T,D]
=\frac{2}{N_k}\sum_k\operatorname{Tr}
\left[T_k\left(J_k[D]-\frac12K_k[D]\right)\right].
$$

각 symmetric strain의 $+\delta/-\delta$ cell에서 같은 AO density를 유지하고 GDF/RSDF
provider와 J/K만 재구성한다. 총 12개 provider build가 필요하지만 source HF, target
DFA, target Fock와 Z-vector는 다시 계산하지 않는다. Water8 DF 단계는 GDF
652.70 s에서 4.95 s, RSDF 122.67 s에서 4.71 s로 줄었다.

H2에서 기존 analytic DF와 비교한 최대 stress 차이는 GDF
$2.13\times10^{-13}$, RSDF $6.85\times10^{-14}$ Eh/Bohr$^3$였다.

## Water8 결과

조건은 Water8, 24 atoms/48 AO, 10 Angstrom cubic cell, SZV-GTH,
GTH-SCAN, DC-r2SCAN, Gamma, 400 Ha, mesh 173x173x173이다.

| Route | GDF stress | RSDF stress |
|---|---:|---:|
| Fully analytic baseline | 2226.20 s | 1705.86 s |
| GPU PP + analytic DF | 833.96 s | 305.46 s |
| GPU PP + contracted DF | 179.61 s | 178.33 s |
| Fused GPU PP + contracted DF | **25.09 s** | **23.72 s** |
| Baseline 대비 speedup | **88.73x** | **71.92x** |

Selected xx/xy energy FD 최대 오차는 최종 GDF
$9.33\times10^{-11}$, RSDF $6.74\times10^{-11}$ Eh/Bohr$^3$다.

| Memory metric | Baseline GDF | Final GDF | Baseline RSDF | Final RSDF |
|---|---:|---:|---:|---:|
| Host peak RSS | 12.64 GiB | 4.77 GiB | 12.63 GiB | 4.80 GiB |
| Peak CuPy used | 0.81 GiB | 0.70 GiB | 0.56 GiB | 0.81 GiB |
| Peak CuPy reserved | 1.36 GiB | 1.66 GiB | 1.37 GiB | 1.66 GiB |

Global device-used peak는 동시 production process의 변동을 포함하므로 위 표에는
process-local CuPy 값만 사용했다.

## 제한과 기본값

- 두 route 모두 explicit opt-in이며 fully analytic 기본 경로를 바꾸지 않는다.
- PP factor와 DF scalar derivative에 낮은 수준의 centered cell displacement가 있다.
  Report는 `production=false`와 `low_level_numerical_components`를 기록한다.
- PP route는 GTH-SCAN Water8에서 FD를 통과했다. GTH-HF H2에서는 host analytic PP와
  유의미한 차이가 있어 pseudopotential family 일반화가 완료되지 않았다.
- Restricted closed-shell Water8의 Gamma 및 2x1x1 sampled-k selected-FD gate를
  통과했다. 더 큰 k-mesh와 pseudopotential family 일반화 전에는 release default로
  승격하지 않는다.

Raw evidence:

- `benchmark/development/results/h2_contracted_df_validation_2026-07-22`
- `benchmark/development/results/water8_optimization_evaluation_2026-07-22`
- `benchmark/development/results/water8_gpu_pp_optimization_2026-07-22`
- `benchmark/development/results/water8_gpu_pp_df_optimization_2026-07-22`
- `benchmark/development/results/water8_gpu_pp_df_fused_2026-07-22`
- `benchmark/development/results/water8_gpu_pp_df_fused_final_2026-07-22`
- `benchmark/development/results/water8_gpu_pp_df_streaming_gdf_2026-07-22`
- `benchmark/development/results/water8_gpu_pp_df_streaming_rsdf_2026-07-22`
- `benchmark/development/results/water8_gpu_pp_df_final_v2_2026-07-22`

## Dedicated GPU 반복 측정

다른 CUDA process가 없는 RTX PRO 6000 Blackwell Server Edition에서 FD를 끄고 backend별
3회 반복했다.

| Metric | GDF median (range) | RSDF median (range) |
|---|---:|---:|
| Source HF + target | 1.84 s (1.82--1.85) | 1.84 s (1.79--1.85) |
| Analytic stress | **16.18 s** (16.09--16.19) | **15.70 s** (15.64--15.78) |
| End-to-end workflow | **18.18 s** (18.11--18.20) | **17.70 s** (17.66--17.73) |
| PP response | 13.56 s | 13.44 s |
| Nonlocal PP | 7.87 s | 7.78 s |
| DF response | 2.31 s | 1.94 s |
| Host peak RSS | 4.80 GiB | 4.80 GiB |
| CuPy pool used | 0.62 GiB | 0.62 GiB |

정확성 판정은 FD가 포함된
`water8_gpu_pp_df_final_v2_2026-07-22`에서 수행했으며 GDF/RSDF 모두 통과했다.
전용-GPU timing raw data와 median summary는
`water8_gpu_dedicated_timing_2026-07-22`에 있다.

## Sampled-k validation

같은 Water8 조건에서 k-mesh만 2x1x1로 바꾸고 selected xx energy FD를 수행했다.

| Backend | Source state | Analytic stress | Selected FD | Max abs. error |
|---|---:|---:|---:|---:|
| GDF | 6.61 s | 52.28 s | 17.78 s | $6.91\times10^{-11}$ |
| RSDF | 6.68 s | 52.61 s | 18.23 s | $6.58\times10^{-11}$ |

두 backend 모두 tolerance $10^{-8}$ Eh/Bohr$^3$를 통과했다. Sampled-k AO/XC는
GPU4PySCF `krks_stress`, GTH primitive는 공통 `rks_stress` helper를 사용한다.
Raw evidence는
`benchmark/development/results/water8_gpu_sampled_k_validation_2026-07-22`에 있다.
