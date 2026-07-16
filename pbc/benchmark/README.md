# PBC DC-DFT Benchmark

이 directory는 독립 package의 public API만 사용하여 restricted PBC DC-DFT
energy와 analytic gradient의 계산 시간 및 정확성을 비교한다.

지원하는 조합은 다음과 같다.

- device: `cpu`, `gpu`
- DF backend: `fftdf`, `gdf`, `rsdf`
- target DFA: `PBE`, `SCAN`, `R2SCAN`
- sampling: Gamma-point 및 일반 `kmesh`
- accuracy reference: 선택적으로 실행하는 full-energy central FD

출력은 force가 아니라 energy gradient이며 단위는 `Ha/Bohr`이다. FD는 각 변위
구조에서 source RHF/KRHF와 target-DFA energy를 처음부터 다시 계산하므로, analytic
gradient와 같은 device 및 DF backend의 전체 energy derivative를 검증한다.

## 1. Queue 실행

`gpu` conda environment에서 다음과 같이 실행한다.

```bash
cd pbc/benchmark
conda run -n gpu python run_benchmark_queue.py \
  --queue example_benchmark_queue.json
```

완료된 `report.json`이 있으면 기본적으로 skip한다. 완료된 case까지 다시 계산할
때만 `--overwrite`를 사용한다.

```bash
conda run -n gpu python run_benchmark_queue.py \
  --queue example_benchmark_queue.json \
  --overwrite
```

확장된 case 이름에 특정 문자열이 들어간 job만 실행하려면 `--only`를 사용한다.

```bash
conda run -n gpu python run_benchmark_queue.py \
  --queue example_benchmark_queue.json \
  --only cpu_fftdf
```

## 2. Queue 형식

`defaults`는 모든 job에 공통으로 적용되고, 각 `jobs` 항목은 필요한 값만
덮어쓴다. `devices`와 `df_backends`를 여러 개 주면 Cartesian product로 확장된다.

```json
{
  "output_root": "results/my_benchmark",
  "defaults": {
    "atom_file": "structures/h2.xyz",
    "atom_unit": "Angstrom",
    "lattice_vectors": [[6, 0, 0], [0, 6, 0], [0, 0, 6]],
    "lattice_unit": "Bohr",
    "basis": "gth-szv",
    "pseudo": "gth-hf",
    "auxbasis": "gth-szv",
    "gdf_setup_mode": "native",
    "ke_cutoff_ha": 20.0,
    "kmesh": [1, 1, 1],
    "xc": "PBE",
    "num_threads": 64,
    "run_fd": false
  },
  "jobs": [
    {
      "name": "all_routes",
      "devices": ["cpu", "gpu"],
      "df_backends": ["fftdf", "gdf", "rsdf"]
    },
    {
      "name": "one_component_fd",
      "device": "gpu",
      "df_backend": "fftdf",
      "run_fd": true,
      "fd_step_bohr": 0.0001,
      "fd_atoms": [0],
      "fd_axes": ["z"]
    }
  ]
}
```

`atom_file`은 queue JSON 위치를 기준으로 해석한다. 대신 다음처럼 inline atom을
사용할 수도 있다.

```json
"atom": [
  ["H", [0.0, 0.0, -0.7]],
  ["H", [0.0, 0.0, 0.7]]
],
"atom_unit": "Bohr"
```

현재 active package 범위에 맞춰 `charge=0`, `spin=0`인 restricted closed-shell
계산만 허용한다. grid는 `ke_cutoff_ha` 또는 명시적 `mesh` 중 하나만 지정한다.

GPU GDF에서 `gdf_setup_mode`는 다음 두 값을 받는다.

- `native` (default): GPU4PySCF CUDA CDERI와 source-contracted derivative kernel
- `reference`: CPU-identical canonical CDERI/full-$dCDERI$ 검증 경로

성능 측정에는 `native`를 사용하고, equation-level debugging이나 과거 result 재현에만
`reference`를 사용한다. CPU GDF에서는 이 option을 무시한다.

Native GPU GDF nuclear response에는 다음 bounded-memory controls를 선택적으로 사용할 수
있다.

- `z_nuclear_batch_memory_fraction`: 현재 free GPU memory 중 batch working-set estimate에
  사용할 비율. Default는 `0.05`이며 범위는 `(0, 1]`이다.
- `z_nuclear_batch_max_components`: memory-derived batch size에 적용하는 optional positive
  upper bound. 생략하거나 `null`이면 cap을 적용하지 않는다.

일반 계산에서는 default policy를 사용한다. 작은-batch parity를 검증하거나 다른 GPU
process와 memory를 공유할 때만 더 작은 값/cap을 지정한다. `report.json`의
`gradient_metadata.z_nuclear_response_batch_plan`에는 preflight가,
`z_nuclear_response_batch_execution`에는 실제 rebatch와 observed memory high-water가
저장된다. OOM은 batch를 절반으로 줄여 같은 component부터 재시도하며 CPU fallback으로
바뀌지 않는다.

Gamma native GDF reciprocal derivative에는 별도의 자동 memory planner가 항상 적용된다.
사용자가 block size를 추측해서 지정할 필요는 없다. `report.json`의
`gradient_metadata.gdf_reciprocal_memory`에는 다음 정보가 저장된다.

- 계획 당시 free/total GPU memory와 실제 계산에 허용한 byte budget
- zero-order 및 derivative live-tensor estimate와 최초 reciprocal block size
- 실제 처리한 block 수, 최소/최대 block size와 전체 reciprocal-vector 수
- OOM 때문에 동일 interval을 줄여 다시 계산한 횟수
- SR/LR/metric/auxiliary/Ewald phase별 CUDA event timing
- Gamma derivative에 사용한 SR/LR 분할값 `short_range_omega`와 scale

Block 결과는 해당 interval 계산이 성공한 뒤에만 commit된다. 따라서 OOM 재시도에서도
reciprocal vector를 빠뜨리거나 두 번 더하지 않으며, block size 1도 할당할 수 없으면
CPU fallback 대신 명시적 memory error로 종료한다.

GPU RSDF production gradient도 `gradient_metadata.df_response_profile`에 별도 bounded
response 정보를 기록한다. `response_mode=source_contracted_component_stream`이면 full
`dCDERI/dJ/dK`를 저장하지 않았다는 뜻이다. `full_derivative_equivalent_bytes`는 피한
full tensor 크기, `retained_device_derivative_bytes`는 실제 보존량,
`device_peak_explicit_working_bytes`는 component stream의 명시적 workspace estimate다.
`initial_component_batch_size`, `component_batch_size`, `component_rebatch_count`로 OOM
축소 이력도 확인할 수 있다.

## 3. FD 선택

- `run_fd: false`: analytic energy/gradient timing만 계산하고 accuracy field는 `null`
- `run_fd: true`: 선택한 Cartesian component에 대해 central FD 계산
- `fd_atoms: "all"` 또는 atom index list
- `fd_axes: "all"` 또는 `x`, `y`, `z`의 list
- `fd_step_bohr`: nuclear displacement, 단위 `Bohr`

FD를 일부 component에만 수행하면 NPY array의 미계산 component는 `NaN`이며,
`fd_mask.npy`가 실제 비교 component를 표시한다. error metric도 mask가 참인
component만 사용한다.

전체 선택은 문자열 `"all"`을 권장하지만 기존 queue와의 호환성을 위해
`["all"]`도 같은 의미로 허용한다.

## 4. 결과

기본 출력 위치는 queue 파일 아래 `results/<queue-name>/`이다. 각 case에는 다음
파일이 생성된다.

```text
000_h2_timing_matrix_cpu_fftdf/
|-- run_input.json
|-- command.txt
|-- stdout.txt
|-- stderr.txt
|-- progress.json
|-- report.json
`-- arrays/
    |-- analytic_gradient_ha_per_bohr.npy
    |-- fd_gradient_ha_per_bohr.npy                 # FD를 켠 경우
    |-- analytic_minus_fd_gradient_ha_per_bohr.npy # FD를 켠 경우
    `-- fd_mask.npy                                # FD를 켠 경우
```

queue root의 `summary.csv`는 device/DF별 핵심 timing과 FD error를 한 표로 모은다.
`queue_manifest.json`은 case 상태와 report 경로를 기록한다.

### 실행 중 progress 확인

새로 시작하는 계산은 각 case의 `progress.json`을 atomic write로 갱신한다. 따라서
계산 중인 파일을 읽어도 반쯤 쓰인 JSON을 볼 일이 없다. `updated_at_kst`는
`Asia/Seoul`의 ISO 8601 시각(`+09:00`)으로 저장한다. 한 case의 현재 stage는
다음처럼 확인한다.

```bash
python show_progress.py \
  results/my_queue/000_water_gpu_gdf
```

queue 아래의 모든 case를 5초마다 추적하려면 다음 명령을 사용한다.

```bash
python show_progress.py results/my_queue --watch 5
```

대표 stage는 `source_hf_scf`, `target_dfa_energy`, `zvector_matrix`,
`zvector_dense_solve`, `gradient_one_electron`, DF별 J/K derivative,
`z_nuclear_response`, `finite_difference`이다. 반복 횟수를 아는 stage에는
`completed/total`, percentage 및 단순 선형 ETA도 함께 표시한다. 특히 현재의 dense
Z-vector route는 Hessian column 생성 개수와 예상 dense matrix 크기를 기록하므로,
큰 system에서 계산이 멈춘 것처럼 보이는 구간을 구분할 수 있다.

이 기능을 추가하기 전에 이미 시작된 Python process에는 callback을 소급 삽입할 수
없다. 이 경우 `show_progress.py`는 동일 output directory로 실행 중인 benchmark
process를 찾아 PID, elapsed time, CPU 사용률을 보여주지만 정확한 내부 stage는 알 수
없다. 다음 실행부터는 `progress.json`의 상세 stage가 표시된다.

`report.json`에서 timing의 의미는 다음과 같다.

- `setup_wall_s`: central cell, k-points 및 method object 구성
- `energy_wall_s`: central source HF SCF와 target-DFA energy 평가
- `analytic_gradient_wall_s`: central analytic gradient만의 시간
- `analytic_total_wall_s`: `energy_wall_s + analytic_gradient_wall_s`
- `workflow_without_fd_wall_s`: setup을 포함한 analytic workflow
- `fd_wall_s`: 모든 선택된 `+h/-h` energy 계산
- `workflow_wall_s`: setup, 계산 및 NPY array 저장을 포함한 전체 workflow 시간
- `method_internal_s`: package가 제공하는 energy 단계별 timing
- `gradient_internal_s`: package가 제공하는 gradient 단계별 timing

GPU 외부 wall time은 각 구간 전후에 CUDA stream synchronization을 수행한 값이다.
따라서 비동기 kernel launch 시간만 재는 오류를 피한다.

`device_warmup`은 default `true`이며, GPU에서는 작은 CuPy reduction으로 CUDA context와
기본 kernel을 초기화한 뒤 memory pool의 unused block을 비우고 scientific timing을
시작한다. Warm-up 시간은 `device_warmup_wall_s`로 별도 저장되어
`analytic_gradient_wall_s`에 포함되지 않는다.

`memory` section은 `after_device_warmup`, `after_input_setup`, `after_energy`,
`after_analytic_gradient` 및 FD를 켠 경우 `after_fd`의 stage-boundary sample을 보존한다.
Host는 process `ru_maxrss`, GPU는 free/total device bytes와 CuPy pool used/reserved bytes를
기록한다. 이는 exact asynchronous kernel peak가 아니라 **observed high-water**이므로,
짧은 transient allocation 분석에는 Nsight/CUDA profiler가 별도로 필요하다.

## 5. Single case 실행

Queue를 사용하지 않고 하나의 merged input JSON을 직접 실행할 수도 있다.

```bash
conda run -n gpu python dcdft_benchmark.py \
  --input my_case.json \
  --output-dir results/my_case
```

CPU/GPU timing을 공정하게 비교하려면 같은 geometry, basis, pseudo, auxiliary
basis, `ke_cutoff_ha`/`mesh`, kmesh, convergence threshold를 사용하고, GPU warm-up 및
system load의 영향을 확인하기 위해 독립 반복 case를 두는 것이 좋다.
