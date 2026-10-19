# CPU 분자계 grid-response backend

## 선택 방법

```python
from pyscf import lib
lib.num_threads(32)
mf_grad = mf.nuc_grad_method()
mf_grad.grid_response = True
mf_grad.grid_response_backend = 'becke_omp'  # 기본값: 'pyscf_default'
gradient = mf_grad.kernel()
```

두 backend는 같은 물리 항을 계산한다. `grid_response`는 grid 이동 항의
포함 여부이고, `grid_response_backend`는 그 계산 구현을 선택한다.
`pyscf_default`는 grid 구현만 뜻한다. DC-Forge의 Z-vector 정책을 바꾸지 않는다.
요청 backend, 실제 사용 backend와 PySCF thread 수는 INFO 로그에 남는다.
grid response가 꺼져 있거나 target이 HF이면 실제 사용값은 `inactive`이다.
DF의 기존 grid_response/auxbasis_response 필수 조건은 그대로 유지한다.

## 설치

일반 패키지 설치 시 선택적 C/OpenMP 확장도 빌드된다. 현재 소스 트리에서는:

```bash
conda activate gpu
python setup.py build_ext --inplace
python examples/06-grid-response-backend.py
```

C compiler와 OpenMP가 필요하다. 계산 중 컴파일하지 않는다.
확장이 없어도 기본 backend는 동작하며, becke_omp를 명시하면 오류로 알려준다.
Linux/GCC 경로를 검증했으며 다른 compiler/platform은 별도 검증이 필요하다.
thread 수는 `pyscf.lib.num_threads()`를 사용한다. CPU affinity는 변경하지 않는다.
여러 OpenMP runtime이 로드된 환경의 binding은 성능을 제한할 수 있다.
성능 비교 시 실제 affinity와 runtime을 확인하고 필요하면 실행 전에
`OMP_PROC_BIND=FALSE`를 설정한다. 정확성 테스트의 2-thread 설정은 성능 권장값이 아니다.

## 지원 범위와 구현

- CPU 분자계 RHF/UHF, conventional/DF. GPU와 PBC는 변경하지 않는다.
- original_becke와 radii_adjust None/Treutler/Becke를 지원한다.
- 다른 partition/radii 함수는 최적화 경로에서 명시적으로 거부한다.
- FP64 계산이다. grid level, pruning, radial/angular grid와 적분 정확도를 낮추지 않는다.
- C kernel은 각 grid point에서 원자쌍을 두 번 순회하여 partition weight와
  그 핵 좌표 미분을 직접 수축한다. 큰 원자쌍×grid 미분 배열을 만들지 않는다.
  원자별 출력 weight-derivative 배열과 AO integration 메모리는 여전히 필요하다.
- PySCF의 XC/AO integration과 DF J/K/auxiliary derivative 본문은 그대로 호출한다.
  호출마다 private 함수 globals만 연결한다. 전역 monkeypatch는 없다.
- 네 driver에서 원래 PySCF get_veff provider를 명시해 DF 다중상속의
  메서드 선택이 conventional 경로로 바뀌지 않도록 했다.

PySCF 2.14.0에 검증한 연결이다. PySCF 업그레이드 시 get_veff,
get_vxc_full_response, grids_response_cc의 호출 관계 및 배열 tag를 재검증해야 한다.
PySCF 자체의 C weight derivative 구현도 있으나, 기존 실험에서는 다른 순회
방식의 비용을 확인했다. 전역적인 알고리즘 신규성을 주장하지 않는다.

## 검증

```bash
conda activate gpu
OMP_PROC_BIND=FALSE OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python -m pytest -q -s tests/unit/test_grid_response_backend.py
```

검증 범위: radii adjustment 3종, 1/2-thread weight parity, moving-grid weight FD,
RHF/UHF × conventional/DF × PBE/r2SCAN의 total/skeleton/nonvariational parity,
DF .aux 및 .exc1_grid tag, atom-subset, selected energy FD, 예외 및 비활성 경로.
기존 실험의 lindep=1e-24 설정은 가져오지 않는다. 두 backend 모두 현재
DC-Forge CPHF 정책을 사용한다.

이 승격 검증은 정확성 검증이다. 기존 실험의 caffeine 가속 수치를 현재
코드의 end-to-end 속도로 간주하지 않는다. 큰 분자의 성능은 별도 재측정해야 한다.
예시: `examples/06-grid-response-backend.py`.

### 2026-10-19 승격 검증 결과

- 환경: conda gpu, PySCF 2.14.0, 현재 #3334 backport 적용 소스.
- 새 backend 테스트 15개를 포함한 CPU/unit 회귀: **138 passed, 94 deselected**.
  명령: `python -m pytest -q tests/unit tests/cpu -m 'not gpu and not slow'`.
- 8개 full-gradient 비교의 최대 차이: 약 `1.23e-11 Ha/Bohr`
  (`6.32e-10 eV/Å`). STO-3G 물/RHF 및 OH/UHF, grid level 1,
  PBE/r2SCAN, conventional/DF. 작은 차이의 마지막 자릿수는 반복별로 변한다.
- DF 물/PBE, grid level 3, 핵 변위 `1e-4 Bohr`의 selected energy FD 오차:
  `2.87e-9 Ha/Bohr` (`1.48e-7 eV/Å`).
- 원래 PySCF 함수 객체 보존, 예외 뒤 재호출, concurrent grid 호출,
  extension 미설치 상황에서 기본 경로 사용/명시적 최적화 오류를 검사했다.
- 실제 두 OpenMP runtime(Conda libgomp, PySCF wheel libgomp)이 함께 로드된다.
  정확성 검증은 OMP_PROC_BIND=FALSE에서 했다. 이 구성의 대규모 scaling은
  아직 재측정하지 않았다.
- commit/push 및 기본값 변경은 하지 않았다.
