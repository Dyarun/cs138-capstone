# CS 138 Capstone: The Philippine Educational Crisis, from a Numerical Perspective

Model: a factor matrix M maps a group's state x at one PISA cycle to its state y at the next, M x = y.
M_G is fitted on international data, and M_PH = M_G + ΔM is calibrated on Philippine data.

## Repository map

```
config/variables.json      THE file to edit: paths, cycles, PISA column names, codes, thresholds, solver settings
obj/factor_dict.json       which factors enter the state vector (0.3): 19 state factors
obj/group_dict.json        derived variables + demographic groups (0.5): 12 fit cells + 14 descriptive
obj/transition_pairs.json  which cycle pairs give (x, y) columns: global 2018→22, 22→25; PH the same
imple/config.py            loader: col(), raw_file(), available_cycles(), state_factors(), groups(), transition_pairs()
imple/svd.py               1.1 SVD (numpy), rank, condition number, pinv, truncation                  [done]
imple/lsq.py               1.2 normal eq. / QR / SVD least squares (numpy+scipy), method comparison   [done]
imple/tikhonov.py          1.3 Tikhonov with prior M0, GCV / L-curve choice of λ          [left out for now]
imple/perturbation.py      2.2 ΔM solver: list of candidate methods + stubs                       [skeleton]
imple/effectiveness.py     1.5 effectiveness of M_G: analysis ideas + stubs                      [skeleton]
imple/compare.py           2.4 M_G vs M_PH comparison: analysis ideas + stubs                    [skeleton]
imple/report.py            Phase 1-2 figures (conditioning experiment, design spectra)              [done]
docs/perturbation.md       2.1 derivation skeleton
prep/load.py               0.1/0.2 read .sav, merge school file, rename to aliases, PH split, cache      [done]
prep/clean.py              0.4 missing codes, encode, PV means, verification log, missing report, scaler [done]
prep/groups.py             0.5 derived variables (ESCS_PCTL, LOCATION), group masks, sizes            [done]
prep/vectors.py            0.6 representative vectors (+ CLT se), X and Y, coverage, column weights   [done]
prep/report.py             Phase 0 tables and figure for the notebook                                 [done]
docs/decisions.md          0.7 data-handling decisions note
pipeline.py                glue: phase0() → phase1() → phase2(); phase 1-2 fits are skeletons
tests/run_all.py           runs every module's synthetic self-test
tests/synthetic_pisa.py    small FAKE PISA files (real names/layout) for testing before the data is in
tests/test_phase0.py       Phase 0 checks: missing codes, quartiles, partition, scaler, vectors, X/Y
data/raw/                  PISA files (git-ignored); names set in config paths.raw_files
data/processed/, results/  written by the pipeline
```

## How the pieces connect

```
config/variables.json ──► imple/config.py ◄── obj/*.json
                               │ (names, paths, codes, settings for every module)
 Phase 0  prep/load ─► prep/clean ─► prep/groups ─► prep/vectors ─► X_g, Y_g, X_ph, Y_ph
 Phase 1  imple/svd ─► imple/lsq ─► imple/tikhonov ─► M_G  (+ relative_error per column)
 Phase 2  imple/perturbation (uses svd + tikhonov) ─► ΔM, M_PH = M_G + ΔM
```

- Every module imports names from `imple.config`, never hard-coded strings. A renamed PISA column means one edit in `config/variables.json`.
- `available_cycles(dataset)` returns the years in `cycles.years` whose file is set and exists on disk. The pipeline loops over those, so adding a dataset only means filling in its path.
- The state vector order is the order of the `factors` with role `state` and `include: true`. That count is n, and M is n × n.
- One scaler is fitted on the global sample of `cycles.basis` and reused for every cycle and for PH, so M_G and M_PH are in the same units.

## Getting the data

Download the SPSS student and school files from the OECD PISA database and place them as

```
data/raw/intl/2018/CY07_MSU_STU_QQQ.sav     data/raw/intl/2018/CY07_MSU_SCH_QQQ.sav
data/raw/intl/2022/CY08MSP_STU_QQQ.SAV      data/raw/intl/2022/CY08MSP_SCH_QQQ.SAV
```

or anywhere else and set `CS138_RAW_DIR` (e.g. a Google Drive folder in Colab). The Philippine sample is taken from
the international files, so no separate PH download is needed. 2025: add the file names under `paths.raw_files.intl.2025`.

No data yet? `from pipeline import use_synthetic_data; use_synthetic_data()` points every phase at small fake files with
the same layout, so code can be written and tested now. Numbers from synthetic data are not findings.

## Colab

```python
!git clone https://github.com/Dyarun/cs138-capstone.git
%cd cs138-capstone
!pip install -q pyreadstat
from google.colab import drive; drive.mount('/content/drive')
import os; os.environ["CS138_RAW_DIR"] = "/content/drive/MyDrive/<folder with intl/2018/...>"
from pipeline import phase0
data = phase0()
```

## Implementation order (each step has a self-test or a clear output)

1. `imple/svd.py` → `python -m imple.svd`
2. `imple/lsq.py` (building blocks, then the three solvers, then `fit_factor_matrix`) → `python -m imple.lsq`
3. `imple/tikhonov.py` → `python -m imple.tikhonov`
4. `imple/perturbation.py` → `python -m imple.perturbation`
5. Fill `obj/factor_dict.json` and `obj/group_dict.json` from the templates.
6. `prep/load.py` → load one cycle and check `check_columns()` returns `[]`
7. `prep/clean.py` → inspect `missing_report`
8. `prep/groups.py` → inspect `group_sizes`, write sizes into `group_dict.json`
9. `prep/vectors.py` → check the shapes of X and Y (n × N)
10. `pipeline.py` runs end to end; build the notebook around it.

Steps 1–4 use synthetic matrices only, so they can be done in parallel with steps 5–9.

## Config lines still to add

These are optional. The code falls back to the defaults shown.

```json
"solver": {
  "tikhonov": { "lambda_grid": [-6, 2, 50], "method": "gcv" }
}
```

`lambda_grid` means `[log10 min, log10 max, count]`. `method` is either `gcv` or `lcurve`.

## Data folder layout (suggested)

File names in `paths.raw_files` are relative to `paths.raw_dir` and may include subfolders:

```
data/raw/intl/2018/CY07_MSU_STU_QQQ.sav    → "student": "intl/2018/CY07_MSU_STU_QQQ.sav"
data/raw/phl/2022/...                       → "student": "phl/2022/..."
```

## Notebook layout (preliminary report / final paper)

1. Title, members, contributions
2. Proposal TL;DR and revised phases
3. Config and dictionaries: display `factor_dict` and `group_dict` as tables
4. Phase 0: `data = phase0()`, then the missing report, group sizes and X/Y shapes
5. Phase 1: self-test output, solver comparison (accuracy against condition number), `g = phase1(data)`, effectiveness table and plot
6. Phase 2: derivation (markdown), `ph = phase2(data, g)`, ΔM heatmap, singular values of M_G against M_PH
7. Stuff tried and to try, questions for the instructor, LLM declaration

## Later phases (not scaffolded yet)

| Module | Deliverable |
|---|---|
| `imple/diagnose.py` | 3.1–3.4 `diagnose(M)`, noise injection, bootstrap with replicate weights |
| `imple/baselines.py` | 3.5 multiple linear regression, Markov chain, Bland–Altman |
| `imple/eigen.py` | 4.1 power and inverse power method |
| `imple/qr_gs.py` | 4.2 Gram–Schmidt / QR on factor columns |
| `imple/pcr.py` | 4.3 principal component regression |
| `imple/weights.py` | 4.4–4.6 category weight analysis, ranking |
| `imple/pde.py` | 5.2–5.5 finite-difference advection–diffusion solver |
