# S7 Water8 evaluation (2026-07-22)

> 이 문서는 최적화 전 baseline을 보존한다. 후속 contracted GPU 최적화 결과는
> `S7_GPU_Contracted_Response_2026-07-22.md`에 있으며, 최종 Water8 stress 시간은
> GDF 25.09 s, RSDF 23.72 s다.

## Scope

현재 GPU hybrid analytic stress를 H2보다 큰 Water8에서 검증했다.

- 24 atoms, 48 AO
- 10 Angstrom cubic cell
- SZV-GTH / GTH-SCAN pseudopotential
- r2SCAN target, Gamma point
- 400 Ha, uniform mesh 173x173x173 (5,177,717 points)
- GDF and RSDF
- Independent selected energy FD: xx and xy, strain step $2.5\times10^{-4}$
- 64 host threads, matrix-free Z-vector

GPU에는 다른 production process가 함께 있었으나 stress process의 package-local CuPy pool과
process memory는 안전 범위였다. 두 case 모두 정상 완료했다.

## Accuracy

| Backend | Analytic xx (Ha/Bohr$^3$) | FD xx (Ha/Bohr$^3$) | Selected max abs. error |
|---|---:|---:|---:|
| GDF | $-4.0514549997\times10^{-4}$ | $-4.0514556879\times10^{-4}$ | $6.88\times10^{-11}$ |
| RSDF | $-4.0510687811\times10^{-4}$ | $-4.0510694710\times10^{-4}$ | $6.90\times10^{-11}$ |

두 case 모두 configured tolerance $10^{-5}$ Ha/Bohr$^3$를 큰 여유로 통과했다. Max error의
상대 크기는 xx stress에 대해 약 $1.7\times10^{-7}$이다. Shear xy는 analytic에서
$3\times10^{-17}$ 수준이고 FD에서 약 $10^{-13}$ 수준으로 모두 사실상 0이다.

GDF와 RSDF는 서로 다른 DF approximation이므로 두 backend의 절대 energy/stress가 완전히
같을 필요는 없다. 이 input에서 energy 차이는 $2.05\times10^{-5}$ Ha, xx stress 차이는
$3.86\times10^{-8}$ Ha/Bohr$^3$이며, 각 backend는 자기 energy FD와 독립적으로 일치한다.

## Performance

| Stage | GDF | RSDF |
|---|---:|---:|
| Source HF + target energy | 4.65 s | 5.26 s |
| Target Fock + Z-vector | 0.96 s | 0.99 s |
| Host one-electron/PP/Ewald | 1565.84 s | 1575.75 s |
| Host DF primitives | 651.13 s | 121.26 s |
| Device DF factorization/J/K | 0.060 s | 0.065 s |
| XC Lagrangian assembly | 8.19 s | 7.78 s |
| **Analytic stress total** | **2226.20 s (37.10 min)** | **1705.86 s (28.43 min)** |
| Selected two-component FD | 13.80 s | 16.43 s |

RSDF analytic stress는 GDF보다 23.4% 짧고 1.31배 빠르다. 차이는 PP가 아니라 DF host
primitive에서 온다. GDF DF primitive는 651.1 s, RSDF는 121.3 s다.

그러나 현 Water8 구현의 중요한 결론은 analytic stress가 아직 FD보다 빠르지 않다는 것이다.
Analytic stress / selected FD ratio는 GDF 161.3, RSDF 103.8이다. Selected FD는 두 strain
component만 포함하므로 full six-component FD와 직접 같은 작업량은 아니지만, 이 차이는
단순 component 수로 설명할 수 없을 만큼 크다.

### Bottleneck shares

- GDF: host PP 70.3%, host DF 29.2%
- RSDF: host PP 92.4%, host DF 7.1%
- Device DF response: 두 backend 모두 analytic stress의 0.01% 미만

따라서 추가 GPU J/K kernel은 end-to-end 시간을 거의 바꾸지 않는다. 최우선 작업은 GDF와
RSDF가 공유하는 PP strain primitive이며, 그 다음이 GDF reciprocal DF primitive다.

## Memory

| Metric | GDF | RSDF |
|---|---:|---:|
| Host peak RSS | 12.64 GiB | 12.63 GiB |
| Peak CuPy pool reserved | 1.36 GiB | 1.37 GiB |
| CuPy pool reserved increase during stress | 0.54 GiB | 0.54 GiB |

OOM은 발생하지 않았다. Global device free-memory 변화는 동시에 실행된 production process를
포함하므로 package 단독 peak로 사용하지 않았다. Package-local pool과 `nvidia-smi`의
stress process 항목은 약 1.4--1.8 GiB 범위였다.

## Verdict

- Water8 selected-FD accuracy: **PASS**
- Water8 host/device memory safety: **PASS**
- Water8 analytic-stress performance: **FAIL / optimization required**
- Primary blocker: common GTH PP strain host primitive
- Secondary blocker: GDF reciprocal DF host primitive

다음 최적화는 numerical parameter를 바꾸지 않고 다음 순서로 진행한다.

1. PP local-part2/nonlocal의 atom, G-block 및 Fourier AO-pair 중복 호출 profile
2. GDF/RSDF 공통 one-electron/PP/Ewald primitive의 same-cell cache/reuse contract
3. PP strain의 fused callback 또는 density-contracted direct route
4. GDF reciprocal source/metric의 block별 C-driver 호출 수와 AO-pair materialization 축소
5. 같은 Water8 queue를 release gate로 재실행해 accuracy를 유지한 wall-time 감소 확인

Raw evidence는
`benchmark/development/results/water8_optimization_evaluation_2026-07-22`에 있다.
