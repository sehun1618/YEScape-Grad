# S7 GPU Hybrid Analytic Stress

## Status

S7의 production 경로는 `host_integral_device_response`라는 명시적 owner contract를
사용한다. 전체 CPU stress driver를 호출하거나 CPU SCF로 되돌아가지는 않지만,
PySCF/GPU4PySCF가 아직 제공하지 않는 homogeneous-strain integral primitive는 host에서
만든다. GPU4PySCF와 마찬가지로 host integral engine과 device electronic-response를
결합하는 구조이며, 결과 metadata가 이 경계를 숨기지 않는다.

## Ownership

GPU가 소유하는 항은 다음과 같다.

- source HF state와 density
- target DFA Fock 및 energy-weighted density
- matrix-free HF response와 adjoint Z-vector
- PBE/r2SCAN XC stress
- complex metric Cholesky와 Cholesky derivative
- sampled-k GDF/RSDF J/K strain contraction
- DC-DFT Lagrangian 조립과 최종 stress contraction

Host analytic primitive 경계는 다음과 같다.

- overlap 및 kinetic homogeneous-strain integral
- GTH local/nonlocal pseudopotential strain matrix
- Ewald derivative
- GDF/RSDF 2-center metric 및 3-center source strain integral

이 경계는 metadata에 owner와 worker 수를 기록한다. CPU target energy, CPU Z-vector 또는
완성된 CPU stress tensor를 가져오는 silent fallback은 허용하지 않는다.

## Single-pass Voigt device response

중앙 CDERI factor는 J/K 전체에 필요하므로 device에 유지한다. Host source/metric
derivative는 ordered Cartesian 9성분을 계속 보존하지 않고
`[xx, yy, zz, yz, xz, xy]` 대칭 Voigt 6성분으로 즉시 압축한다. 따라서 retained host
derivative array는 같은 dtype/shape 기준으로 $6/9$, 즉 33.3% 줄어든다.

각 Voigt 성분에서는 canonical k-pair를 정확히 한 번 순회한다.

1. 해당 pair의 source/metric derivative만 device로 전송한다.
2. complex Cholesky response로 $dC$를 한 번 만든다.
3. 같은 $dC$로 auxiliary Coulomb response, $dJ$와 $dK$를 모두 수축한다.
4. off-diagonal pair의 $dC$는 즉시 해제하고 reverse pair는 conjugate AO transpose로
   복원한다.

이전 cache-off 경로는 diagonal $dC$를 auxiliary response, $dJ$, $dK$에서 최대 세 번
다시 만들었다. 새 경로의 factor derivative 횟수는 항상
$6N_{\mathrm{canonical\ pair}}$다. Gamma에서는 6회이며 이전 cache-on 9회 또는
cache-off 27회보다 작다. 별도의 diagonal derivative cache는 더 이상 필요하지 않다.
기존 `df_diagonal_cache_bytes` 입력은 queue 호환을 위해 받지만 계산 선택에는 영향을
주지 않으며 metadata의 `df_diagonal_cache_enabled`는 항상 `false`다.

## Single-adjoint strain response

DC-DFT orbital-response 항의 기존 구현은 overlap strain 6성분으로 만든 density 묶음에
HF AX를 적용했다. HF response bilinear form의 self-adjoint identity

$$
\langle T_Z, G[M_x] \rangle = \langle G[T_Z], M_x \rangle
$$

를 사용하면 같은 값을 다음처럼 계산할 수 있다.

1. converged Z-vector로 transition density $T_Z$를 한 번 만든다.
2. $G[T_Z]$를 한 번만 계산한다.
3. 여섯 hcore/overlap strain source를 이 고정 adjoint state와 저비용 trace로 수축한다.

따라서 물리식이나 Z-vector solution은 바뀌지 않으면서 AX 입력 density 수가 6개에서
1개로 줄어든다. Synthetic complex Hermitian test에서 기존 explicit source contraction과
$10^{-10}$ 아래로 일치하며, 실제 Gamma GDF/RSDF stress도 CPU analytic reference와 각각
$2.31\times10^{-12}$, $5.04\times10^{-13}$ Eh/Bohr$^3$로 일치한다.

## Host pair parallelism

각 canonical k-pair의 2c/3c primitive는 서로 독립이다. Pair마다 deep-copied
`cell/auxcell`을 사용해 PySCF의 mutable integral context를 공유하지 않고 최대 4개의
host worker로 병렬 생성한다. Worker 수를 k-pair 수만큼 무제한 늘리지 않는 이유는 각
pair가 큰 source/metric derivative를 보유하며, OpenMP integral thread와 중첩될 수 있기
때문이다.

2x1x1 H2 test에서 이 변경은 다음과 같이 측정되었다.

| Backend | host DF primitive, 이전 | host DF primitive, 현재 | stress 전체, 이전 | stress 전체, 현재 |
| --- | ---: | ---: | ---: | ---: |
| GDF/PBE | 11.50 s | 4.40 s | 14.72 s | 7.89 s |
| RSDF/PBE | 2.59 s | 1.46 s | 5.70 s | 4.96 s |

One-electron/PP/Ewald 네 작업의 추가 thread 병렬화는 PP 한 작업이 대부분의 시간을
차지하고 내부 OpenMP와 경쟁해 유의미한 이득이 없었다. 따라서 이 병렬화는 기본 경로에
남기지 않았다.

## Bounded host reciprocal response

초기 구현은 long-range 3-center source에서 `(3,3,nG,nao,nao)` AO-pair strain을 먼저
만든 뒤 auxiliary basis에 수축했다. Water20/SZV/500 Ha의 실제 입력은
`mesh=(239,239,239)`, $N_G=13,651,919$, $N_{AO}=120$이므로 이 배열 하나의 이론적
크기만 약 26.4 TiB다. 실제 실행은 RSS 약 474 GiB와 swap 약 30 GiB에 도달한 뒤 report를
쓰지 못하고 종료됐다.

현재 경로는 G-vector를 bounded block으로 나누고 AO pair, center derivative와 Cartesian
moment를 만든 뒤 9개 strain 성분을 하나씩 auxiliary Fourier factor에 즉시 수축한다.
Production에서는 audit용 pair/auxiliary/kernel decomposition 사본도 보존하지 않는다.
기본 working-set budget은 약 8 GiB이며 위 Water20 입력에서는 G-vector 2,310개로
결정된다. 반환해야 하는 최종 source derivative는 약 0.23 GiB다.

GTH local-part2는 `atom x 3 x 3 x nG` 배열과 전체 structure factor를 만들지 않고 원자
항을 G block 안에서 즉시 합한다. Nonlocal projector도 structure factor를 block별로
생성한다. Mesh, range-separation parameter와 precision은 바꾸지 않는 순수 contraction
순서 최적화다.

## Fused Fourier source kernel

Long-range 3-center source의 이전 block kernel은 모든 lattice image에서 Fourier overlap,
center-i derivative, center-j derivative와 coordinate moment를 네 번의 integral driver로
각각 평가했다. 앞의 세 tensor는 동일한 shell pair, Gaussian primitive recurrence와 G block을
공유한다. Stress 전용 C extension은 이를 7-component callback
`[overlap, ip_i(3), ip_j(3)]`으로 합친다. Coordinate moment만 별도 호출하므로 image/block당
driver 호출은 4회에서 2회로 줄어든다.

Lattice image마다 생성하는 translated `Mole`, concatenated basis, shell slice와 phase도
G block에 의존하지 않는다. Pair 시작 시 한 번 만든 immutable context를 모든 reciprocal
block에서 재사용한다. Small complex test에서 기존 separate callback과 overlap/center-i는
bitwise 같고 center-j 최대 차이는 $1.75\times10^{-18}$이다. 동일 process의 blocked
long-range source에서 separate 3.53 s, fused median 1.97 s로 1.79배 빨랐으며 최종 source
derivative 차이는 $1.11\times10^{-16}$이다.

PP local-part1은 source vbar와 GTH pseudo-G0 보정이 같은 G=0 overlap strain을 요청한다.
이 overlap object를 한 번 만들어 두 경로에 전달한다. 작은 H2에서는 전체 시간이 거의
변하지 않지만 중복 integral은 제거되고 sampled-k에서도 같은 contract를 사용한다.

## Bounded host-pair pipeline

Sampled-k 계산은 $N_k(N_k+1)/2$ canonical pair를 갖는다. 모든 pair의 raw Cartesian
9-component source/metric을 먼저 dict에 쌓지 않고, 최대 `host_workers`개 future만 submit한다.
완료된 pair는 main thread가 즉시 Voigt 6성분으로 압축하고 GPU factorization한 뒤 raw
payload를 해제한다. 따라서 raw pair memory는 전체 pair 수가 아니라 bounded worker window에
비례한다. GPU preflight는 host tensor를 만든 뒤가 아니라 `naux`, `nao`, pair 수로 먼저
수행한다.

2x1x1 sampled-k PBE end-to-end parity는 GDF $2.88\times10^{-12}$, RSDF
$1.16\times10^{-12}$ Eh/Bohr$^3$다. Future 완료 순서가 pair ordering과 달라도 canonical
pair key로 authoritative CDERI를 연결하므로 결과 순서에 의존하지 않는다.

Gamma/complex-k source, metric, PP 및 RSDF를 포함한 CPU stress test와 GPU response,
queue test가 있다. Full-block, forced 17-point block, compact mode, six-component J/K 및
explicit/adjoint orbital-response parity를 각각 고정한다.

## Correctness evidence

Distorted H2, GTH-SZV, 2x1x1 k mesh에서 현재 GPU/CPU stress 최대 차이는 다음과 같다.

| Backend/XC | CPU-GPU maximum difference |
| --- | ---: |
| GDF/PBE | $2.88\times10^{-12}$ Eh/Bohr$^3$ |
| RSDF/PBE | $1.16\times10^{-12}$ Eh/Bohr$^3$ |
| GDF/r2SCAN | $2.88\times10^{-12}$ Eh/Bohr$^3$ |

Evidence는 `stress/validation/results/2026-07-21_s7_kpoint_*_gpu_parity.json`에 있다.
Synthetic complex-k test는 component streaming 결과가 CPU tensor algebra와 일치함을
검사한다.

## Release gates

- Gamma/sampled-k PBE/r2SCAN GDF/RSDF CPU-GPU parity가 통과했다.
- Single-pass Voigt route는 diagonal response cache 없이 동작한다.
- Preflight memory estimator가 central factor, one-pair derivative workspace와 J/K output을
  포함한 deterministic memory diagnostic을 제공한다.
- Gamma GDF/PBE 3회 median에서 source state와 analytic stress 합계 8.58 s가 full GPU
  energy FD 17.93 s보다 짧았다.
- Host integral-engine boundary를 production contract로 채택했으며 native GPU strain
  integral은 향후 성능 확장이지 correctness blocker가 아니다.
