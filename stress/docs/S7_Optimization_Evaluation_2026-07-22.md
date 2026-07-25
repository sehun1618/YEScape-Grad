# S7 optimization evaluation (2026-07-22)

## 결론

현재 GPU hybrid analytic stress 구현은 작은 독립 검증계에서 정확성 gate를 통과했다.
PBE/r2SCAN, Gamma/2x1x1 sampled-k, GDF/RSDF의 CPU-GPU 최대 차이는
$5.04\times10^{-13}$--$2.88\times10^{-12}$ Ha/Bohr$^3$이고, Gamma PBE의 full
six-component energy finite difference와의 최대 차이는 GDF
$6.45\times10^{-11}$, RSDF $5.72\times10^{-11}$ Ha/Bohr$^3$이다.

성능상 GPU J/K contraction은 더 이상 주 병목이 아니다. 현재 작은 H2 측정에서 GDF는
analytic stress 시간의 약 70--73%가 host DF strain primitive 생성이고, RSDF는 host
PP/one-electron primitive와 host DF primitive가 대부분을 차지한다. 따라서 다음 성능
작업의 우선순위는 GPU kernel 추가보다 host reciprocal integral과 PP source의 호출 수,
blocking 및 병렬 효율이다.

## 측정 조건과 제한

- System: H2, GTH-SZV/GTH-HF, cubic 8 Bohr cell
- Gamma mesh: 15x15x15
- Sampled-k: 2x1x1, mesh 11x11x11
- Source SCF/target state와 analytic stress 시간을 분리해 측정
- Full FD strain step: $2.5\times10^{-4}$
- 이 평가 중 다른 Python process가 GPU를 100% 사용하고 있었다. 아래 시간은 기능별
  비중과 현재 부하에서의 참고값이며 clean release median이 아니다.
- 작은 H2 정확도와 메모리 결과는 Water20/64의 확장성 또는 peak memory를 보증하지 않는다.

## 정확성

| Target | Sampling | Backend | CPU-GPU max abs. (Ha/Bohr$^3$) |
|---|---|---:|---:|
| PBE | Gamma | GDF | $2.31\times10^{-12}$ |
| PBE | Gamma | RSDF | $5.04\times10^{-13}$ |
| PBE | 2x1x1 | GDF | $2.88\times10^{-12}$ |
| PBE | 2x1x1 | RSDF | $1.16\times10^{-12}$ |
| r2SCAN | Gamma | GDF | $2.31\times10^{-12}$ |
| r2SCAN | Gamma | RSDF | $5.05\times10^{-13}$ |
| r2SCAN | 2x1x1 | GDF | $2.88\times10^{-12}$ |
| r2SCAN | 2x1x1 | RSDF | $1.17\times10^{-12}$ |

Full GPU energy FD는 CPU stress 복사본이 아닌 strained-cell energy 재계산이므로 더 강한
독립 검증이다.

| Backend | Analytic-FD max abs. (Ha/Bohr$^3$) | Analytic stress (s) | Full FD (s) |
|---|---:|---:|---:|
| GDF | $6.45\times10^{-11}$ | 5.273 | 45.040 |
| RSDF | $5.72\times10^{-11}$ | 2.054 | 31.578 |

## 현재 부하에서의 성능

| Target | Sampling | Backend | Stress total (s) | Host PP (s) | Host DF (s) | Device DF (s) |
|---|---|---:|---:|---:|---:|---:|
| PBE | Gamma | GDF | 5.196 | 1.237 | 3.712 | 0.029 |
| PBE | Gamma | RSDF | 1.853 | 1.238 | 0.323 | 0.047 |
| PBE | 2x1x1 | GDF | 8.090 | 1.860 | 5.880 | 0.060 |
| PBE | 2x1x1 | RSDF | 3.552 | 1.865 | 1.313 | 0.095 |
| r2SCAN | Gamma | GDF | 5.272 | 1.244 | 3.710 | 0.040 |
| r2SCAN | Gamma | RSDF | 1.803 | 1.227 | 0.320 | 0.037 |
| r2SCAN | 2x1x1 | GDF | 7.850 | 1.870 | 5.690 | 0.046 |
| r2SCAN | 2x1x1 | RSDF | 4.223 | 1.879 | 2.073 | 0.060 |

Gamma PBE에서 full FD는 analytic stress 자체보다 GDF 8.54배, RSDF 15.38배 느렸다.
Source state 생성까지 포함해도 analytic end-to-end는 full FD보다 각각 4.93배와 5.27배
빠르다. 이 비율도 현재 GPU contention의 영향을 받으므로 절대적인 production speedup으로
해석하지 않는다.

## 최적화별 평가

1. Symmetric Voigt layout은 retained derivative를 9에서 6성분으로 줄여 해당 배열
   memory를 33.3% 줄였다.
2. Single-pass DF contraction은 canonical k-pair당 $dC$ 평가를 6회로 고정했다. 이전
   diagonal-cache 경로의 9회보다 33.3%, cache 없이 재계산하던 27회보다 77.8% 적다.
3. Single-adjoint HF response는 여섯 strain density에 AX를 각각 적용하던 작업을 Z density
   한 번의 AX와 여섯 저비용 trace contraction으로 바꿨다. AX operator 호출은 6회에서
   1회로 줄었다.
4. Fused Fourier callback은 image/G block당 integral driver 호출을 4회에서 2회로 줄였다.
   동일 process microbenchmark에서 long-range source는 3.529 s에서 median 1.967 s로
   1.79배 빨라졌고 derivative 최대 차이는 $1.11\times10^{-16}$이었다.
5. Reciprocal image context 재사용만 분리해 측정하면 4.082 s에서 3.715 s로 1.10배
   빨라졌고 결과는 bitwise 동일했다.
6. Bounded producer-consumer는 sampled-k raw pair 대기 개수를 전체 pair 수에서
   `host_workers` 이하로 제한한다. 이는 주로 peak host memory 개선이며 작은 H2 timing에는
   뚜렷한 speedup을 요구하지 않는다.

## 메모리 판단

- DF retained strain payload는 Cartesian 9에서 Voigt 6으로 감소했다.
- Diagonal $dC$ cache는 제거되어 입력 크기에 비례하는 별도 cache가 없다.
- Raw source/metric pair는 완료 즉시 압축/분해/해제되므로 raw 9성분과 compact 6성분의
  전체 사본이 동시에 누적되지 않는다.
- Allocation 전 estimator가 central factor, component workspace, J/K workspace와 free
  VRAM을 비교해 OOM 전에 중단할 수 있다.
- 이번 H2 run에서는 OOM 위험이 없었다. 다만 `nvidia-smi` global free-memory 변화는 다른
  GPU process의 allocation까지 포함하고, observation 사이에 생성 후 해제된 temporary를
  놓칠 수 있으므로 이를 package 단독 peak VRAM 수치로 채택하지 않는다.

## 판정과 다음 gate

- **작은 계 정확성:** PASS
- **Gamma/sampled-k, PBE/r2SCAN, GDF/RSDF 기능 범위:** PASS
- **FD 대비 analytic 효율:** PASS
- **구조적 host/device memory 감소:** PASS
- **독점 GPU에서의 3회 timing median:** PENDING
- **Water20/64 SZ/DZ peak RSS/VRAM과 정확성:** PENDING

Water8/SZV/r2SCAN/400 Ha selected-FD gate는 정확성과 memory safety를 통과했지만 analytic
stress performance는 통과하지 못했다. GDF 37.10분, RSDF 28.43분이며 공통 host PP
primitive가 각각 26분 이상을 사용했다. 자세한 결과는
`S7_Water8_Evaluation_2026-07-22.md`에 둔다.

다음 release 성능 수치는 GPU가 비어 있을 때 같은 input을 3회 반복하고 median을 사용한다.
큰 계는 먼저 stress-only, FD-off로 실행하며 RSS/VRAM preflight와 stage별 progress를 확인한
뒤 selected FD를 별도 job으로 수행한다.
