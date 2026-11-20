# Data-handling decisions (deliverable 0.7)

*Status 20 Nov 2026. Each decision names the file or setting that implements it, so the code and this note stay in step.*

## 1. What x and y are
PISA is cross-sectional, so no student is observed twice. We follow **groups** instead: a pseudo-panel (Deaton, 1985).

- A **state vector** describes one group of students in one country and one cycle. Its entries are the group's weighted means of the 19 state factors (§2).
- **x** is a group's state in one cycle and **y** is the same group's state in the next cycle (2018 → 2022, 2022 → 2025). So **M x = y** is a one-cycle step of the group's educational trajectory.
- x and y live in the same space, so **M is square (n × n)**. Phase 4 needs this for the eigen-analysis and the power method.
- Each column of **X** and **Y** is one (cycle pair, country, group) cell. The fit groups are the 12 cells *ESCS quartile × school location* (§5). With all PISA countries this gives several hundred columns for M_G, and up to 24 for the Philippines.
- Implemented in `obj/transition_pairs.json` and `prep/vectors.build_xy`.

## 2. The state factors (n = 19)
| Category | Factors |
|---|---|
| Outcome | MATH, READ, SCIE |
| Socioeconomic | HOMEPOS, HISEI, PAREDINT |
| Regional | SH_RURAL, SH_CITY (share of students in rural or city schools) |
| School | SH_PUBLIC, STRATIO, EDUSHORT, STAFFSHORT |
| Individual | SH_REPEAT, DURECEC, SH_LANG_OTHER |
| Attitudinal / climate | BELONG, BULLIED, TEACHSUP, DISCLIM |

- **ESCS is left out of the state on purpose.** It defines the groups, and it is a composite of HOMEPOS, HISEI and PAREDINT, so including it would make the rows of X nearly collinear.
- **IMMIG is left out** because the Philippine share is close to zero, which would give a constant row.
- The categories feed the Phase 4 weight analysis (NLA hypothesis).
- Source: `obj/factor_dict.json`.

## 3. Weights and plausible values
- Every mean, SD, percentile and scaler uses the **final student weight** `W_FSTUWT`.
- The **80 replicate weights** are not loaded by default (`weights.load_replicates = false`) to save memory. Phase 3 switches them on for BRR standard errors.
- **Achievement** is the mean of the 10 plausible values, divided by 100 (`plausible_values.policy = mean_of_pvs`). This gives an unbiased group mean but understates its variance. Rubin's rules over the PVs are deferred to Phase 3.
- **Group uncertainty (CLT):** each group mean comes with se = weighted SD / √n_eff, where n_eff is Kish's effective sample size. This ignores school clustering, so it is a lower bound.

## 4. Missing data
- **Missing codes.** Codes declared in the SPSS files become NaN when they are read. As a safety net, `prep.clean.drop_missing_codes` also does two things:
  - A categorical value outside the codes listed in the config becomes NaN. A blanket "5 to 9 means missing" rule would wrongly delete `SC001Q01TA = 5`, which means "large city".
  - An index value of 9995 or more becomes NaN.
- **Factors.** A state factor is dropped if more than 30% of its values are missing in any (dataset, cycle) used (`preprocessing.max_missing_rate_per_factor`). The same n-dimensional space must exist for every column.
- **Group means.** Each mean uses the students who have that factor (available-case, weighted).
- **Cells.** A cell must have at least 30 students in both cycles (`countries.sample_threshold`) and no missing factor, or it does not become a column. `xy_coverage.csv` records every dropped cell and the reason.

## 5. Groups
- **Fit cells (`fit: true`):** within-country ESCS quartile (weighted percentile rank) × school location (rural < 3,000 people, town, city > 100,000). These 12 cells **partition** each country's students whose location is known, so no student counts twice in a fit. They put both hypothesis dimensions, socioeconomic and regional, into the columns of X.
- **Descriptive groups (`fit: false`):** ALL, the ESCS quartiles, the three locations, public/private, female/male, and test language/other language at home. Their vectors are computed for the effectiveness tables (1.5, 2.4) and for Phases 4 and 5. They are not fitted.
- Source: `obj/group_dict.json`.

## 6. Scaling
- **One scaler.** A weighted z-score is fitted on the **global sample of the 2018 cycle** and reused for every cycle and for the Philippines (`prep.clean.fit_scaler`). So M_G and M_PH act on the same units, and a drop in a score between cycles stays visible as a change in z.
- **STRATIO** is clipped at its global weighted 1st and 99th percentiles before scaling. The bounds are fitted once.
- The model has **no intercept** (M x = y). The data is centred on the global 2018 average, so the intercept is small but not zero. *Open:* add a constant entry to the state (homogeneous coordinates) if Phase 1 residuals show a systematic offset.

## 7. Which countries go into which fit
- **M_G** is fitted on all countries except the Philippines (`countries.global_exclude = ["PHL"]`). This keeps the Philippine data independent: it enters only through ΔM.
- **Philippine rows** are taken from the international files (where CNT is PHL), unless separate PH files are set in the config.
- **Column weighting in the fits** is `country_equal`: each country has the same total weight, split over its cells by population. This follows the OECD-average convention, so the largest systems do not dominate M_G.

## 8. Pooled or per-group M_G (one M for all (x, y) pairs, not one per pair)
- **Why one M for many pairs.** M has n² = 361 unknowns, and one (x, y) pair gives only n = 19 equations. A matrix per pair would therefore have infinitely many exact solutions and mean nothing. M_G is fitted to **all** columns together, M_G X = Y (every cycle pair × country × fit cell). It is the one matrix that best maps every group's state to its next state in the least-squares sense.
- **Primary:** one **pooled** M_G over all fit cells. Phase 2 calibrates a single M_PH = M_G + ΔM on the Philippine cells.
- **Secondary (Phase 4.6):** per-quartile or per-location M_G, each fitted on its 3 or 4 cells. The Philippine data is too thin per cell (1–2 columns) for a per-cell M_PH.

## 9. Cycle comparability and known caveats
- **2018 → 2022 spans COVID-19.** The Philippines had one of the longest school closures in the world, and ΔM from this pair absorbs that shock.
- **Changed indices:**
  - BULLIED was rebuilt (2018 `BEINGBULLIED`).
  - TEACHSUP and DISCLIM refer to test-language lessons in 2018 and to mathematics lessons in 2022.
  - `SC001Q01TA` gained a sixth "megacity" category in 2022, mapped to "city".
- **2025 cycle:**
  - Only the **school** questionnaire is in the repo (`datasets/PISA-2025.sav`: 25,133 schools, 208 of them Philippine). The cycle is used once the 2025 **student** file is added.
  - In the 2025 public-use file, STRATIO exists only as an international quartile (`STRATIO_Q`), so it is set to null for 2025.
- **Region (Phase 5):** `PH_REGION` (17 regions) and `PH_ISLAND_GROUP` (NCR, Luzon, Visayas, Mindanao) are mapped from the 2025 STRATUM labels. The 2018 and 2022 strata use other codes, so their maps still need to be added from those cycles' labels.

## 10. Outputs (paths.processed_dir)
- `clean/clean_<dataset>_<cycle>.parquet` and `clean_<dataset>_all.parquet`: cleaned, encoded, **unscaled** student tables (deliverables 0.1 and 0.2).
- `verification_log.csv`: student counts against the technical reports, duplicate ids, weights and value ranges.
- `missing_report.csv`, `scaler.json`, `group_sizes.csv`, `representative_vectors.parquet`, `xy.npz`, `meta_global.csv`, `meta_ph.csv`, `xy_coverage.csv`.
