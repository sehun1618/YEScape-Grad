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

완료된 `report.json`의 case fingerprint가 현재 effective input 및 source tree와
일치할 때만 기본적으로 skip한다. Basis, pseudo, cutoff/mesh, k-point, XC, XYZ 내용
또는 package source가 바뀌면 같은 output root와 job 이름이어도 기존 artifact를
archive하고 새로 실행한다. 동일 fingerprint의 완료 case까지 다시 계산할 때만
`--overwrite`를 사용한다.

```bash
conda run -n gpu python run_benchmark_queue.py \
  --queue example_benchmark_queue.json \
  --overwrite
```

각 실행 attempt에는 `run_identity`가 부여되며 `attempt.json`, `progress.json`,
`report.json`에 같은 값이 기록된다. Queue runner는 현재 attempt와 identity가 다른
report를 결과로 받아들이지 않는다. 같은 case를 다시 실행하면 기존 root artifact는
삭제하지 않고 `attempts/<시각>_<attempt-id>/`로 함께 이동한다.

각 case에는 다음 내용을 합친 SHA256 fingerprint도 기록된다.

- defaults와 job을 합친 effective numerical configuration
- XYZ 파일의 실제 byte content와 lattice 설정
- Git commit, source-only dirty state 및 CPU/GPU/benchmark source-tree digest

Fingerprint는 `run_input.json`, `attempt.json`, `progress.json`, `report.json` 및
`queue_manifest.json`에서 확인할 수 있다. 동시에 실행된 두 queue가 같은 case를
archive하거나 덮어쓰지 않도록 `.queue.lock`을 child process lifetime 전체에 유지한다.

외부 종료로 `progress.json`만 `running`에 남더라도 다음 queue 실행에서 기록된 PID와
Linux process start time을 확인한다. 같은 process가 더 이상 없으면 이전 attempt를
`interrupted`로 바꾸고 새 attempt를 시작한다. 같은 host에서 실제 process가 살아
있다면 중복 실행하지 않고 `skipped_running`으로 기록한다.

확장된 case 이름에 특정 문자열이 들어간 job만 실행하려면 `--only`를 사용한다.

```bash
conda run -n gpu python run_benchmark_queue.py \
  --queue example_benchmark_queue.json \
  --only cpu_fftdf
```

### RSDF release benchmark

현재 RSDF production 경로의 반복 timing과 선택 FD 검증은 다음 queue로 실행한다.

```bash
cd pbc/benchmark
conda run -n gpu python run_benchmark_queue.py \
  --queue rsdf_r1_release_gate.json \
  --output-root results/rsdf_user_run
```

이 queue는 Water8/Gamma/r2SCAN/SZV-GTH 조건에서 GPU RSDF analytic gradient를 독립
process로 세 번 측정하고, 마지막 job에서 atom 0의 z component만 central FD로
검증한다. `--output-root`를 생략하면 queue에 고정된 기존 release evidence directory를
사용하므로 완료된 case는 skip된다. 새 측정에는 위 예시처럼 별도 output root를 지정한다.

```bash
conda run -n gpu python run_benchmark_queue.py \
  --queue rsdf_r1_release_gate.json \
  --output-root results/my_rsdf_benchmark
```

다른 구조를 사용하려면 queue를 복사해 `atom_file`, `lattice_vectors`, `basis`,
`pseudo`, `auxbasis`, `ke_cutoff_ha`, `kmesh`, `xc`를 바꾼다. 핵심 route 선택은
`"device": "gpu"`, `"df_backend": "rsdf"`이다. CPU/GPU를 함께 비교하려면 한
job에 `"devices": ["cpu", "gpu"]`를 사용한다.

GPU RSDF의 central factor 생성 경로는 `rsdf_central_setup`으로 선택한다.

- `reference` (현재 default): PySCF가 central two-/three-center CDERI를 모두 만들고
  canonical factor를 GPU로 올린다.
- `exact_gpu_source`: PySCF는 independent source/metric parameter와 작은 screened
  two-center metric만 담당한다. 큰 three-center source와 long-range metric,
  Cholesky/CDERI 변환은 GPU4PySCF/CuPy에서 수행한다.

두 경로는 같은 source/metric omega와 mesh를 사용하며 Gamma 및 sampled-k에서
CPU CDERI와 `1e-9` 이내 parity를 만족해야 한다. 새 경로를 비교 측정할 때는 다음을
job 또는 defaults에 추가한다.

```json
"rsdf_central_setup": "exact_gpu_source"
```

## 2. Queue 형식

`defaults`는 모든 job에 공통으로 적용되고, 각 `jobs` 항목은 필요한 값만
덮어쓴다. `devices`, `df_backends`, `target_integrators`를 여러 개 주면 Cartesian
product로 확장된다.

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
    "rsdf_central_setup": "reference",
    "target_integrator": "uniform",
    "ke_cutoff_ha": 20.0,
    "kmesh": [1, 1, 1],
    "xc": "PBE",
    "num_threads": 64,
    "track_vram_peak": true,
    "vram_sample_interval_s": 0.05,
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
      "fd_axes": ["z"],
      "fd_max_abs_tolerance_ha_per_bohr": 1e-5
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

### Target multigrid

`df_backend`는 source HF의 J/K와 source response backend를 정한다. Target DFA의
수치적분은 별도 `target_integrator`로 고른다.

- `uniform` (default): selected DF target J + GPU uniform-grid XC
- `multigrid_xc`: selected DF target J + GPU multigrid XC
- `multigrid_j_xc`: GPU multigrid combined target J+XC

Source HF density, exchange, Z-vector Hessian과 pseudopotential/hcore owner는 세 mode에서
모두 선택한 GDF/RSDF로 유지된다. Multigrid mode에서만 optional
`target_multigrid_mesh: [NX, NY, NZ]`를 지정할 수 있다. CPU는 현재 `uniform`만
허용하며 요청을 조용히 다른 route로 바꾸지 않는다.

Small-system analytic/FD gate는 `multigrid_correctness_gate.json`, Water8 3회 반복 및
제한된 Water20 timing/memory 비교는 `multigrid_performance_gate.json`에 있다.
`report.json`의 `gradient_metadata.target_integration`에는 J/XC/PP owner, mesh, task
level, energy/gradient integration wall time 및 energy task-cache 재사용 여부가 저장된다.

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

GPU RSDF production gradient도 `gradient_metadata.df_response_profile`에 ownership과
memory 정보를 기록한다. Default
`response_mode=native_source_native_metric_fused_adjoint`는 target-J와 Z-bilinear
adjoint를 먼저 합친 뒤 source와 reciprocal metric derivative에 한 번만 직접 수축했다는
뜻이다. `observable_mode=fused`, `observable_count=1`이며 성분별 진단 경로는
`build_rsdf_contracted_response(..., observable_mode="resolved")`로 보존된다. Exact screened
short-range metric `ip1/ip2`는 host boundary로 metadata에 명시된다.
`full_derivative_equivalent_bytes`는 피한 full tensor 크기,
`retained_device_derivative_bytes`는 실제 보존량,
`device_peak_explicit_working_bytes`는 명시적 workspace estimate다. Reference 또는 mixed
`component_stream` mode에서만 `initial_component_batch_size`, `component_batch_size`,
`component_rebatch_count`가 OOM 축소 이력을 나타내며 native-only mode에서는 `null`이다.
Source 병목 분석에는 `gradient_internal_s`의
`rsdf_device_native_source_{setup,short_range,reciprocal}`과
`rsdf_device_native_metric_derivative`를 사용한다. 이 값은 CUDA event를 동기화해 측정한
exclusive device phase이며, 합산 wall time과 host setup time은 별도로 해석한다.

## 3. FD 선택

- `run_fd: false`: analytic energy/gradient timing만 계산하고 accuracy field는 `null`
- `run_fd: true`: 선택한 Cartesian component에 대해 central FD 계산
- `fd_atoms: "all"` 또는 atom index list
- `fd_axes: "all"` 또는 `x`, `y`, `z`의 list
- `fd_step_bohr`: nuclear displacement, 단위 `Bohr`
- `fd_max_abs_tolerance_ha_per_bohr`: optional selected-FD pass/fail threshold
- `net_gradient_tolerance_ha_per_bohr`: optional translation-sum norm threshold

FD를 일부 component에만 수행하면 NPY array의 미계산 component는 `NaN`이며,
`fd_mask.npy`가 실제 비교 component를 표시한다. error metric도 mask가 참인
component만 사용한다.

전체 선택은 문자열 `"all"`을 권장하지만 기존 queue와의 호환성을 위해
`["all"]`도 같은 의미로 허용한다.

`run_fd=true`는 FD 숫자를 계산한다는 뜻이고, tolerance를 함께 지정해야 자동
accuracy gate가 활성화된다. Report 상태는 다음처럼 분리된다.

- `execution_status`: `complete`, `failed`, `interrupted`
- `accuracy_status`: `not_requested`, `pass`, `fail`

따라서 timing-only job은 `execution_status=complete`,
`accuracy_status=not_requested`가 정상이다. Queue exit code는 execution failure뿐
아니라 명시적으로 요청한 accuracy gate의 failure도 반영한다.

## 4. 결과

기본 출력 위치는 queue 파일 아래 `results/<queue-name>/`이다. 각 case에는 다음
파일이 생성된다.

```text
000_h2_timing_matrix_cpu_fftdf/
|-- run_input.json
|-- command.txt
|-- stdout.txt
|-- stderr.txt
|-- attempt.json
|-- progress.json
|-- report.json
|-- attempts/                 # 재실행 전 attempt 보존; 존재할 때만 생성
`-- arrays/
    |-- analytic_gradient_ha_per_bohr.npy
    |-- fd_gradient_ha_per_bohr.npy                 # FD를 켠 경우
    |-- analytic_minus_fd_gradient_ha_per_bohr.npy # FD를 켠 경우
    `-- fd_mask.npy                                # FD를 켠 경우
```

queue root의 `summary.csv`는 device/DF별 핵심 timing과 FD error를 한 표로 모은다.
`queue_manifest.json`은 case 상태와 report 경로를 기록한다.

`report.json`에는 다음 release evidence도 함께 저장된다.

- `gradient_summary`: 원자별 analytic gradient의 합 vector와 norm
- `evidence`: case fingerprint, XYZ digest, Git commit과 source-tree digest
- `software_provenance`: Python, NumPy, PySCF, GPU4PySCF, CuPy 및 CUDA version
- `accuracy_gate`: 요청된 threshold별 observed value와 pass/fail

FD displacement energy helper는 energy와 timing만 반환한다. 이전 displacement의
method/provider/CDERI object를 다음 displacement 동안 보존하지 않는다.

### Translation-sum과 egg-box 진단

원자별 gradient 합이 0이 아닐 때는 먼저 analytic 항 누락과 finite-grid egg-box를
구분해야 한다. 다음 진단은 구조 전체를 uniform grid의 1/4 step만큼 옮긴 위치에서
analytic translation sum과 total-energy rigid-translation central FD를 비교하고,
one-electron, target J/XC, Pulay, nuclear 및 response channel별 합을 저장한다.

```bash
conda run -n gpu python diagnostics/diagnose_rigid_translation_eggbox.py \
  --input results/my_queue/000_case/run_input.json \
  --axis x \
  --center-phase 0.25 \
  --slope-step-bohr 1e-4 \
  --ke-cutoff-ha 400 \
  --target-integrator multigrid_xc \
  --output results/diagnostics/my_translation_probe.json
```

- analytic sum과 rigid-energy FD가 맞으면 계산된 finite-grid energy 자체의 egg-box
  derivative다.
- 두 값이 맞지 않으면 analytic derivative 누락 또는 energy/gradient route 불일치를
  먼저 조사해야 한다.
- `component_translation_sums_ha_per_bohr`는 drift를 어느 항이 소유하는지 보여준다.
- 한 phase에서 합이 우연히 0일 수 있으므로 `center-phase=0`만으로 판단하지 않는다.

현재 Water8 r2SCAN 결과에서는 drift가 거의 전부 `target_xc`이고 GDF와 RSDF가 같은
값을 보였다. 상세 수치와 production policy는
`../TRANSLATION_SUM_DIAGNOSIS_2026-07-17.md`에 있다. Production gradient에서 원자별로
평균 합력을 빼는 보정은 computed energy의 미분이 아니므로 사용하지 않는다.

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

GPU benchmark는 추가로 계산 동안 기본 0.05초마다 VRAM을 sampling한다. Queue에서

```json
"track_vram_peak": true,
"vram_sample_interval_s": 0.05
```

를 설정할 수 있으며, GPU에서는 tracking이 default로 켜져 있다. `report.json`의
`memory.vram_peak_tracking`에는 다음 값이 저장된다.

- `peak_observed_gpu_device_used_bytes`: `total - free`로 관측한 device-wide VRAM high-water
- `peak_observed_gpu_device_used_delta_from_baseline_bytes`: warm-up 직후 시작점 대비 증가량
- `peak_observed_cupy_pool_used_bytes`와 `...reserved_bytes`: 현재 benchmark process의 CuPy pool high-water
- `stage_peaks`: SCF, target energy, Z-vector, one-electron/DF response, FD 등 progress stage별 high-water
- `sample_count`, `sample_interval_s`, `sampling_errors`: sampling 품질과 상태

`progress.json`에도 현재/peak VRAM이 갱신되며 `show_progress.py`는
`vram=current/peak GiB`를 표시한다. `summary.csv`에는 sampled device peak, baseline 증가량,
CuPy reserved peak와 sample 수가 별도 column으로 들어간다.

이 polling 값은 stage-boundary sample보다 순간 peak를 훨씬 잘 포착하지만 CUDA allocation이
sampling interval보다 짧으면 놓칠 수 있다. 또한 device-used 값에는 같은 GPU를 사용하는
다른 process의 allocation도 포함된다. 따라서 재현 가능한 측정에서는 해당 GPU를 단독으로
사용하고, kernel 단위의 정확한 peak가 필요할 때는 Nsight Systems/Compute를 사용한다.

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
