# DC-Forge examples

예제는 PySCF처럼 **한 파일이 하나의 계산 주제를 완결**하도록 구성한다. Energy와
gradient처럼 바로 이어지는 단계는 같은 case에 두고, conventional과 density fitting은
한 파일 안의 독립된 `Case 1`, `Case 2` 코드 블록으로 적는다. 사용자가 실제 계산 순서를
그대로 읽을 수 있도록 backend selector, `argparse` driver와 계산 helper는 두지 않는다.

## Molecular

| Example | 계산 흐름 |
|---|---|
| `01-rhf_dft.py` | CPU PBE@RHF conventional case와 DF case의 완전한 계산 코드 |
| `02-uhf_dft.py` | CPU open-shell PBE@UHF conventional/DF case |
| `03-gpu-rhf_dft.py` | GPU PBE@RHF conventional/DF case |
| `04-gpu-uhf_dft.py` | GPU PBE@UHF conventional/DF case |
| `05-xc-and-grid.py` | 한 RHF density에서 LDA/GGA/MGGA/hybrid/RSH target과 grid level 비교 |

각 RHF/UHF 파일은 source HF를 먼저 수렴하고

```python
dm_hf = mf.mf_hf.make_rdm1()
energy = mf.energy_tot(dm=dm_hf)
```

와 같은 supplied-density target energy를 평가한다. `RHFDFT.kernel()` 또는
`UHFDFT.kernel()`은 이 흐름을 묶은 driver이며 target RKS/UKS SCF를 수행하지 않는다.
DF 변형에서 auxiliary basis를 생략하면 PySCF의 matching JK-fit/even-tempered 규칙을
그대로 사용한다.

Repository root에서 예를 들어 다음처럼 실행한다.

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  conda run --no-capture-output -n gpu python examples/01-rhf_dft.py

PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  conda run --no-capture-output -n gpu python examples/03-gpu-rhf_dft.py

PYTHONPATH=src conda run --no-capture-output -n gpu \
  python examples/05-xc-and-grid.py
```

Molecular 기본 구조는 `structures/water1.xyz`와 `structures/oh.xyz`, basis는
`aug-cc-pvtz`, PBE quadrature는 `grids.level=4`다. 출력 gradient는
`g = dE/dR`, force는 `F = -g`이며 단위는 모두 `Ha/Bohr`다.

`grid_response=True`와 DF의 `auxbasis_response=True`는 성능 knob가 아니라 complete
analytic gradient의 구성 조건이다. NLC/VV10, dispersion, fractional occupation과
ROHF/ROKS는 현재 예제 범위가 아니다.

## Periodic

PBC 예제는 [`pbc/README.md`](pbc/README.md)에 정리했다. 현재 범위는 fixed 3D cell,
real Gamma, restricted RHF source, GTH pseudopotential, FFTDF/GDF/RSDF와
PBE/R2SCAN@RHF energy 및 analytic nuclear gradient다. Stress와 cell derivative는
포함하지 않는다.

## Examples와 validation의 경계

`examples`는 공개 API와 파라미터의 의미를 사람이 읽고 수정하며 학습하는 코드다.
다음 항목은 예제 파일을 늘리지 않고 `tools/tests` 또는 `tools/validation`이 담당한다.

- CPU/GPU parity와 all-3N finite difference
- scanner reset 이후 provider/CDERI/auxcell 상태 검사
- peak VRAM, allocator cache와 block-size 성능 sweep
- native C/CUDA endpoint oracle와 large-system benchmark
- failure tolerance와 release gate
