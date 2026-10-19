# DC-Forge logger와 결과 요약

PySCF logger의 stdout/verbose를 그대로 사용한다. 별도 출력 객체나 파일
형식은 추가하지 않았다. 공개 workflow kernel이 성공하면 energy 요약,
gradient kernel이 성공하면 gradient 표와 진단을 출력한다.
CPU/GPU, 분자계 RHF/UHF와 PBC RHF/KRHF 경로에 같은 요약 함수를 연결했다.

- Method: HF-r2SCAN처럼 density 생성 방법–평가 functional 순서.
  현재 source는 HF만 지원한다. PBE-r2SCAN 계산을 새로 구현한 것은 아니다.
- Spin treatment: Restricted 또는 Unrestricted. 분자의 spin 값이 아니라
  실제 workflow 종류를 표시한다. singlet UHF도 Unrestricted이다.
- Source energy: source HF 총에너지.
- Target energy: HF density에서 평가한 dispersion 제외 target 총에너지.
- Dispersion energy: 선택한 D2C 보정. 미사용이면 not applied.
- Total energy: dispersion 포함 최종값. 에너지 단위는 Ha.
- Gradient: dispersion을 포함한 dE/dR, Ha/Bohr. Force는 그 음수이다.

## 출력 위치와 수준

Mole/Cell 생성 시 `output='calculation.log', verbose=4`를 설정하면 로그 파일을
사용한다. 또는 계산 객체의 stdout을 열린 파일 객체로 지정한다. gradient
객체는 생성 시 base의 출력 설정을 상속하므로 설정 변경은 생성 전에 한다.

- verbose 0: 결과 요약 없음.
- verbose 3 (NOTE): energy/gradient workflow 경과시간, 에너지, gradient 표, source 수렴, Z-vector residual (max abs),
  잔차 판정, gradient 최대값/RMS/합.
- verbose 4 (INFO): backend/provider, grid/auxiliary response, solver와 tolerance.

기존 PySCF SCF 진행 로그와 경고는 유지한다. 일반 print()의 목적지는 바꾸지
않는다. energy_tot()와 kernel_hf()/kernel_uhf()에는 별도 요약을 추가하지 않는다.
따라서 저수준 에너지 평가를 반복해도 결과 요약이 중복되지 않는다.

## 진단 해석

SCF 수렴과 Z residual 판정은 별개이다.
Z residual은 반환된 Z를 원래 응답 방정식에 다시 대입해 계산한 잔차이며,
solver 내부의 추정치가 아니다. 잔차 한도는 molecular의
cphf_residual_warn_tol 또는 PBC의 cphf_residual_tol이다. 잔차나 한도가 없으면
NOT CHECKED이며 PASS로 추정하지 않는다. WARNING은 계산 정책을 변경하지 않는다.
전체 원자 결과만 Sum of gradients로 표시한다. atmlst 부분집합은
Selected-atom gradient sum으로 표시하며 병진 대칭성 판정에 사용하지 않는다.
작은 병진합 자체가 작은 force 오차를 보장하지 않는다.

요약은 기존 결과만 읽는다. 새로운 Fock/SCF/Z 계산, GPU 동기화/배열 복사,
파일 저장은 하지 않는다. 단계별 timing 통합은 이번 범위에 없다.

## 계산 시간

공개 energy 및 gradient kernel이 성공하면 verbose >= 3에서 다음 항목을 출력한다.

```text
Energy workflow wall time = ... s
Gradient workflow wall time = ... s
Total wall time = ... s
```

Energy 시간에는 해당 호출에서 수행한 source SCF와 target energy 평가가 포함된다.
run_hf=False이면 source SCF를 다시 수행하지 않으므로 재사용 검증과 target 평가 등의
경과시간이다. Gradient 시간에는 해당 kernel 안의 작업만 포함된다. 두 시간을
SCF/CPHF 등 세부 단계의 순수 계산 시간으로 해석하지 않는다.

Total은 같은 base 객체의 가장 최근 성공한 energy kernel **시작부터 이번 gradient
kernel 종료까지 실제로 흐른 시간**이다. 단계별 시간의 합이 아니며, 두 호출 사이의
대기시간도 포함한다. Gradient를 반복하면 같은 energy 시작 시점부터 새 종료 시점까지
측정하므로 이전 gradient 호출과 그 사이의 시간도 포함된다. Energy를 gradient 내부에서
실행하면 gradient 시작 시점부터 측정한다. Energy kernel 측정 기록이 없는
checkpoint 복원/gradient-only 계산은 이번 gradient 호출의 경과시간을 표시한다.
새 energy 호출이 실패하면 이전 energy 시간은 재사용하지 않는다.
이 값에는 energy kernel 호출 전의 Python 시작·import·구조 생성 시간과 gradient 종료
후의 파일 저장 시간은 포함하지 않는다. 프로그램 전체 실행시간이나 여러 구조의 누적시간은 아니다.

단위는 초이며, monotonic perf_counter로 측정한 host 경과시간이다.
기존 코드의 동기화는 유지하되 계측을 위한 GPU 동기화는 추가하지 않는다.
따라서 GPU의 CUDA-event timing이나 개별 kernel 실행시간과는 다르다.
verbose 0~2에서는 출력하지 않으며, 실패한 호출을 성공 시간처럼 출력하지 않는다.
PySCF의 전역 TIMER_LEVEL 및 기존 상세 타이밍 출력은 변경하지 않는다.
추가 검증: CPU/GPU 분자계와 PBC를 포함한 logger 테스트 30개 통과.
반복 호출 시 시작 시간이 초기화되는지, verbose 0~2에서 출력이 없는지,
실패 시 성공 타이밍을 남기지 않는지도 확인했다.

## 검증

공통 구현은 `dcforge/tools/logger.py`에 둔다. 별도 출력 예제 파일은 두지 않는다.

conda gpu에서 다음 출력 검증을 실행한다(GPU 검증 포함).

```bash
OMP_PROC_BIND=FALSE OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python -m pytest -q tests/unit/test_logger.py
```

2026-10-19: 22개 통과. CPU/GPU conventional/DF RHF/UHF, singlet UHF 표기,
CPU/GPU FFTDF Gamma 및 2-k-point energy/gradient 요약을 실제 계산으로 확인했다.
verbose=0, 파일/StringIO 출력, 부분 원자 합, 잔차 누락/초과/NaN 판정,
dispersion 에너지 표시는 별도 검증했다. 출력 켜기/끄기 비교에서도 분자계
gradient가 1e-10 Ha/Bohr 이내에서 일치했다.
기존 CPU/unit 회귀도 `tests/unit tests/cpu -m 'not gpu and not slow'`로
실행하여 150개 통과(104개 제외)를 확인했다.

테스트 구성 중 H 원자의 자동 HF1e 경로는 기존 UHFDFT 타입 제한에 걸렸다.
또 STO-3G H2의 GPU DF-UHF gradient는 출력 hook 이전 upstream derivative
buffer assertion으로 실패했다. 계산 backend 문제를 이번 출력 작업에서
수정하지 않았으며, singlet UHF 출력 검증은 물 분자로 수행했다.
