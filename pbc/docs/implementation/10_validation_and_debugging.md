# Validation과 debugging

## Full-energy FD의 정의

Nuclear component $(A,x)$의 reference는

$$
g_{Ax}^{\mathrm{FD}}(h)
=\frac{E(\mathbf R+h\mathbf e_{Ax})-E(\mathbf R-h\mathbf e_{Ax})}{2h}
$$

이다. 각 displaced geometry에서 source HF SCF와 target energy를 모두 새로 계산한다.
Matrix-FD 또는 frozen-density component FD를 full gradient evidence로 사용하지 않는다.

## 권장 validation matrix

| Axis | Required cases |
| --- | --- |
| Geometry | H2, H2O, distorted H2O |
| Basis | GTH-SZV, GTH-DZVP 수준 |
| Sampling | Gamma, sampled k-mesh |
| DF | FFTDF, GDF, RSDF |
| XC | PBE, SCAN, r2SCAN |
| Device | CPU, GPU |

모든 조합을 매 unit test에서 실행할 필요는 없지만 completion matrix는 각 축을 실제로
cover해야 한다.

## Error metrics

$$
\Delta_{\max}=\max_{Ax}|g_{Ax}-g_{Ax}^{\mathrm{FD}}|,
\qquad
\Delta_{\mathrm{trans}}=
\left\|\sum_A\mathbf g_A\right\|_2.
$$

초기 target은 analytic--FD $10^{-5}$ Ha/Bohr, CPU--GPU gradient $10^{-7}$ Ha/Bohr,
translation norm $10^{-6}$ Ha/Bohr다. Threshold는 converged FD step과 numerical
representation에서 적용한다.

## Debugging order

1. Energy metadata가 동일한지 비교한다: cell, basis, pseudo, k-points, DF, mesh.
2. Source HF convergence와 density parity를 확인한다.
3. Target energy component $E_1$, $E_H$, $E_{xc}$를 비교한다.
4. Target Fock VO RHS와 Z residual을 확인한다.
5. One-electron, $J/K$, XC, Pulay, nuclear contribution을 분리한다.
6. GDF/RSDF이면 central CDERI value parity를 먼저 확인한다.
7. Source-only/metric-only CDERI derivative를 검사한다.
8. 마지막에 full gradient와 translation sum을 확인한다.

## 자주 보이는 pattern

| Symptom | Likely cause |
| --- | --- |
| Gamma만 통과 | conjugation, reverse k-pair, $1/N_k$ |
| GDF size 증가 시 error 증가 | auxiliary metric/center response |
| RSDF value parity 실패 | source/metric $\omega$ 또는 mesh 혼용 |
| MGGA만 error 증가 | $\tau$ factor, AO second derivative, grid aliasing |
| Cutoff에 따라 FD 불연속 | plus/minus geometry의 resolved mesh 차이 |
| Z residual 작고 FD 불일치 | explicit derivative 또는 Pulay 누락 |
| CPU/GPU 결과만 같고 timing 이상 | silent CPU fallback 또는 host boundary 오분류 |

## Result evidence

Validation JSON은 input, resolved numerical settings, analytic/FD arrays, error norm,
SCF/Z residual, timing과 software version을 포함한다. Gradient는 NPY로 저장하고 JSON에는
unit, shape, dtype, path와 checksum을 기록할 수 있다.

## Completion과 regression의 차이

Unit test는 algebra와 API regression을 빠르게 검사한다. Completion evidence는 full
Cartesian FD, CPU/GPU parity, convergence와 clean-wheel execution을 함께 요구한다. 한
작은 molecule의 한 component가 맞는 것은 route completion이 아니다.
