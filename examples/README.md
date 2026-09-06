# DC-Forge examples

각 예시는 PySCF처럼 한 파일 안에서 하나의 계산 주제를 완결한다. 분자계는
RHF/UHF, PBC는 3D closed-shell GTH Cell의 FFTDF/GDF를 다룬다. ROHF는
포함하지 않는다. GPU 예시는 GPU4PySCF 1.8.1을 사용한다.

| Example | 계산 흐름 |
|---|---|
| `01-rhf_dft.py` | PBE@RHF conventional/DF energy와 analytic gradient |
| `02-uhf_dft.py` | open-shell PBE@UHF conventional/DF energy와 analytic gradient |
| `03-xc-and-grid.py` | 하나의 RHF density에서 target XC와 grid level 비교 |
| `04-uhf-checkpoint.py` | UHF state를 저장·검증·복원한 뒤 SCF 재실행 없이 HF-DFT gradient 계산 |
| `05-d2c.py` | target XC에 맞는 D2C-D4 energy/gradient를 선택적으로 합산 |
| `gpu/01-rhf_dft.py` | GPU PBE@RHF conventional/DF energy와 analytic gradient |
| `gpu/02-uhf_dft.py` | GPU open-shell PBE@UHF conventional/DF energy와 analytic gradient |
| `gpu/03-d2c.py` | GPU HF-DFT에 D2C-D4 energy/gradient를 선택적으로 합산 |
| `pbc/cpu/01-rhf_dft.py` | CPU real-Gamma FFTDF/GDF energy와 analytic gradient |
| `pbc/cpu/02-multigrid.py` | CPU real-Gamma target MultiGrid 조합 |
| `pbc/cpu/03-kpts-rhf_dft.py` | CPU full-BZ sampled-k GDF energy와 analytic gradient |
| `pbc/cpu/04-kpts-r2scan.py` | CPU sampled-k tau-MGGA energy와 gradient |
| `pbc/cpu/05-hf-state.py` | CPU Gamma GDF source 저장 후 별도 프로세스에서 energy/gradient 복원 |
| `pbc/cpu/06-kpts-hf-state.py` | CPU sampled-k FFTDF source 저장 후 별도 프로세스에서 복원 |
| `pbc/gpu/01-rhf_dft.py` | GPU real-Gamma FFTDF/GDF energy와 analytic gradient |
| `pbc/gpu/02-multigrid.py` | GPU real-Gamma target MultiGrid 조합 |
| `pbc/gpu/03-kpts-rhf_dft.py` | GPU sampled-k GDF + XC-only MultiGrid energy/gradient; FFTDF 전환법 |
| `pbc/gpu/04-gdf-acceleration.py` | GPU real-Gamma GDF의 기본 가속 정책과 FP64 reference/근사 profile 경계 |
| `pbc/gpu/05-hf-state.py` | GPU Gamma GDF source 저장 후 별도 프로세스에서 energy/gradient 복원 |
| `pbc/gpu/06-kpts-hf-state.py` | GPU sampled-k FFTDF source 저장 후 별도 프로세스에서 복원 |

Repository root에서 다음처럼 실행한다.

```bash
PYTHONPATH=. conda run --no-capture-output -n gpu \
  python examples/01-rhf_dft.py

PYTHONPATH=. conda run --no-capture-output -n gpu \
  python examples/02-uhf_dft.py

PYTHONPATH=. conda run --no-capture-output -n gpu \
  python examples/03-xc-and-grid.py

PYTHONPATH=. conda run --no-capture-output -n gpu \
  python examples/04-uhf-checkpoint.py

PYTHONPATH=. conda run --no-capture-output -n gpu \
  python examples/05-d2c.py

PYTHONPATH=. conda run --no-capture-output -n gpu \
  python examples/gpu/01-rhf_dft.py

PYTHONPATH=. conda run --no-capture-output -n gpu \
  python examples/gpu/02-uhf_dft.py

PYTHONPATH=. conda run --no-capture-output -n gpu \
  python examples/gpu/03-d2c.py

PYTHONPATH=. conda run --no-capture-output -n gpu \
  python examples/pbc/gpu/03-kpts-rhf_dft.py

PYTHONPATH=. conda run --no-capture-output -n gpu \
  python examples/pbc/gpu/04-gdf-acceleration.py

PYTHONPATH=. conda run --no-capture-output -n gpu \
  python examples/pbc/gpu/05-hf-state.py source /tmp/gpu-pbc-hf.chk
PYTHONPATH=. conda run --no-capture-output -n gpu \
  python examples/pbc/gpu/05-hf-state.py gradient /tmp/gpu-pbc-hf.chk
```

CPU/GPU `RHFDFT.kernel()`과 `UHFDFT.kernel()`은 source HF를 수렴한 뒤 그 density에서
target DFT energy를 평가하며 target RKS/UKS SCF는 수행하지 않는다. DF case에서
auxiliary basis를 생략하면 PySCF의 matching JK-fit/even-tempered 규칙을 사용한다.

출력의 gradient는 `dE/dR`, force는 `-dE/dR`이며 단위는 모두 Ha/Bohr이다.
NLC/VV10, fractional occupation과 ROHF/ROKS는 현재 예제 범위가 아니다.
Dispersion은 기본적으로 꺼져 있으며, `set_dispersion('d2c')`를 명시한 경우에만
target XC에 대응하는 출판된 D2C-D4 parameter를 사용한다. PySCF의 일반 `mf.disp`
경로는 D2C와 다른 parameter를 쓸 수 있으므로 계속 거부한다.

`04-uhf-checkpoint.py`는 PySCF 표준 SCF checkpoint에 DF provider와 convergence
flag가 저장되지 않는다는 점을 명시한다. 따라서 DF 여부를 먼저 재구성하고 density,
MO orthogonality, electron count, source energy와 canonical Fock residual을 검사한 뒤에만
`kernel(run_hf=False)`를 호출한다. 실제 파일은 임시 디렉터리에 만들어져 예시 종료 시
자동으로 삭제된다.

PBC의 `05-hf-state.py`와 `06-kpts-hf-state.py`는 같은 파일을 `source`와
`gradient` 모드로 각각 실행한다. Checkpoint에는 canonical MO, density, Cell,
k-point와 FFTDF/GDF 표현 지문이 저장된다. CDERI 및 GPU 내부 helper는 portable
state가 아니므로 새 프로세스에서 다시 만들지만 source HF SCF는 다시 돌지 않는다.

GPU real-Gamma GDF의 `fp64`, `mixed`, `fp32_build` profile은 CDERI 정보
정밀도와 계산 산술 정밀도를 서로 다르게 조절한다. 전체 단계표, 완전한 FP64
reference 설정과 persistent-FP32 profile의 주의사항은
`docs/pbc_gdf_mixed_precision.md`에 정리되어 있다.
