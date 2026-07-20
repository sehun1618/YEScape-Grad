# Theory Document Build

`PBC_DC_DFT_Gradient_Theory.tex`는 active 범위인 restricted closed-shell PBC
DC-DFT nuclear analytic gradient를 설명하는 연속형 교재 원본이다. 포함 범위는
FFTDF, GDF, RSDF, PBE, SCAN/r2SCAN, Gamma-point와 일반 k-point, target
multigrid ownership, matrix-free Z-vector와 bounded-memory CPU/GPU algorithm
mapping이다.

동일한 수식 label과 chapter 구조를 사용하는 English edition은 다음 파일로 제공한다.

```text
PBC_DC_DFT_Gradient_Theory_EN.tex
PBC_DC_DFT_Gradient_Theory_EN.pdf
english/chapters/*.tex
```

## Build

Korean text와 수식을 안정적으로 처리할 수 있는 XeLaTeX 계열 engine을 사용한다.
Repository root에서 Tectonic을 사용할 경우 다음 명령으로 clean build할 수 있다.

```bash
cd docs/theory
tectonic -X compile PBC_DC_DFT_Gradient_Theory.tex --keep-intermediates
tectonic -X compile PBC_DC_DFT_Gradient_Theory.tex --keep-intermediates
tectonic -X compile PBC_DC_DFT_Gradient_Theory.tex --keep-intermediates
```

일반 XeLaTeX distribution에서는 다음 명령을 세 번 실행한다.

```bash
xelatex -interaction=nonstopmode -halt-on-error PBC_DC_DFT_Gradient_Theory.tex
```

English edition은 main filename만 바꾸어 같은 방식으로 build한다.

```bash
tectonic -X compile PBC_DC_DFT_Gradient_Theory_EN.tex --keep-intermediates
xelatex -interaction=nonstopmode -halt-on-error PBC_DC_DFT_Gradient_Theory_EN.tex
```

반복 build는 table of contents와 cross-reference를 고정하기 위해 필요하다. 배포에
필요한 산출물은 `PBC_DC_DFT_Gradient_Theory.pdf`이며, `.aux`, `.log`, `.out`, `.toc`
파일은 재생성 가능한 intermediate artifact다.

## Acceptance Check

- PDF는 A4 연속형 문서여야 하며 발표 slide 형식이 아니어야 한다.
- `undefined references`, `Overfull`, `Underfull` warning이 없어야 한다.
- Korean text, equation, table, equation number와 hyperlink가 정상 렌더링되어야 한다.
- 구현 함수 연결은 `../implementation/Equation_Code_Map.md`와 함께 확인한다.
- active 범위 밖의 multigrid PP, stress/cell derivative, larger-basis 확장 및
  open-shell/UHF는 구현 대상으로 기술하지 않는다.
