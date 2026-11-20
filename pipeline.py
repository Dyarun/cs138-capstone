"""Runs Phases 0 to 2 in order. The notebook calls these three functions, one per section.

    from pipeline import phase0, phase1, phase2
    data = phase0()                 # cleaned tables, groups, X/Y matrices
    g = phase1(data)                # conditioning, solver comparison; M_G (skeleton)
    ph = phase2(data, g)            # PH design and residual; dM, M_PH (skeleton)

Every name, path and threshold comes from config/variables.json via imple.config.
Outputs are written under config paths.processed_dir and paths.results_dir.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from imple.config import (INTL, PHL, basis_year, factors, path,
                          state_factors, transition_pairs)
from imple.lsq import compare_methods
from imple.svd import summary
from prep.clean import (active_state, apply_scaler, drop_missing_codes, encode, fit_scaler,
                        missing_report, verification_log, write_back_missing)
from prep.groups import derive, fit_group_ids, group_masks, group_sizes, write_back_sizes
from prep.load import load_datasets
from prep.vectors import (build_xy, column_weights, loaded_pairs, representative_vectors,
                          xy_coverage)


def phase0(record: bool = False) -> dict:
    """Data prep (deliverables 0.1 - 0.6):
    load -> missing codes -> encode -> derive -> verify -> scale -> groups -> representative vectors -> X, Y.

    record=True also writes missing % (factor_dict) and group sizes (group_dict) back into obj/*.json.
    Outputs under paths.processed_dir:
      clean/clean_<dataset>_<cycle>.parquet, clean/clean_<dataset>_all.parquet   cleaned, encoded, UNSCALED
      verification_log.csv, missing_report.csv, scaler.json, group_sizes.csv,
      representative_vectors.parquet/.csv, xy.npz, meta_global.csv, meta_ph.csv, xy_coverage.csv
    """
    if not state_factors():
        raise RuntimeError("obj/factor_dict.json has no included 'state' factors yet.")
    intl, phl = load_datasets()
    print("cycles loaded:", {INTL: sorted(intl["CYCLE"].unique().tolist()),
                             PHL: sorted(phl["CYCLE"].unique().tolist())})

    # 0.4 clean + encode (identical steps for both datasets)
    intl, phl = (derive(encode(drop_missing_codes(d), factors())) for d in (intl, phl))

    # 0.1 / 0.2 verification and 0.3 missingness, on unscaled values
    log = verification_log(intl, phl)
    report = missing_report(pd.concat([intl, phl], ignore_index=True), factors())
    state = active_state(report)

    out = path("processed_dir")
    (out / "clean").mkdir(parents=True, exist_ok=True)
    for ds, d in ((INTL, intl), (PHL, phl)):
        for cyc, part in d.groupby("CYCLE"):
            part.to_parquet(out / "clean" / f"clean_{ds}_{cyc}.parquet", index=False)
        d.to_parquet(out / "clean" / f"clean_{ds}_all.parquet", index=False)

    # one scaler, fitted on the global sample of the basis cycle, reused everywhere
    basis = intl[intl["CYCLE"] == basis_year()]
    if basis.empty:
        first = int(intl["CYCLE"].min())
        warnings.warn(f"basis cycle {basis_year()} not loaded; scaling on {first} instead")
        basis = intl[intl["CYCLE"] == first]
    scaler = fit_scaler(basis, [f for f in state_factors() if f["alias"] in state])
    intl, phl = apply_scaler(intl, scaler), apply_scaler(phl, scaler)

    # 0.5 groups and 0.6 representative vectors
    masks_intl, masks_ph = group_masks(intl), group_masks(phl)
    sizes = pd.concat([group_sizes(intl, masks_intl), group_sizes(phl, masks_ph)], ignore_index=True)
    reps = pd.concat([representative_vectors(intl, masks_intl, state),
                      representative_vectors(phl, masks_ph, state)], ignore_index=True)

    fit_ids = fit_group_ids()
    pairs_g, pairs_ph = transition_pairs("global"), transition_pairs("philippines")
    reps_g, reps_ph = reps[reps["DATASET"] == INTL], reps[reps["DATASET"] == PHL]
    X_g, Y_g, meta_g = build_xy(reps_g, pairs_g, fit_ids, state)
    X_ph, Y_ph, meta_ph = build_xy(reps_ph, pairs_ph, fit_ids, state)
    coverage = pd.concat([xy_coverage(reps_g, pairs_g, fit_ids, state),
                          xy_coverage(reps_ph, pairs_ph, fit_ids, state)], ignore_index=True)
    print("transition pairs used:", loaded_pairs(reps_g, pairs_g) + loaded_pairs(reps_ph, pairs_ph))
    print(f"state dimension n = {len(state)};  X_g {X_g.shape},  X_ph {X_ph.shape}")

    log.to_csv(out / "verification_log.csv", index=False)
    report.to_csv(out / "missing_report.csv", index=False)
    (out / "scaler.json").write_text(json.dumps(scaler, indent=1))
    sizes.to_csv(out / "group_sizes.csv", index=False)
    reps.to_parquet(out / "representative_vectors.parquet", index=False)
    reps.to_csv(out / "representative_vectors.csv", index=False)
    np.savez(out / "xy.npz", X_g=X_g, Y_g=Y_g, X_ph=X_ph, Y_ph=Y_ph, state=np.array(state))
    meta_g.to_csv(out / "meta_global.csv", index=False)
    meta_ph.to_csv(out / "meta_ph.csv", index=False)
    coverage[["pair", "DATASET", "COUNTRY", "GROUP", "keep", "reason"]].to_csv(out / "xy_coverage.csv", index=False)
    if record:
        write_back_missing(report)
        write_back_sizes(sizes)
    return dict(X_g=X_g, Y_g=Y_g, meta_g=meta_g, X_ph=X_ph, Y_ph=Y_ph, meta_ph=meta_ph,
                state=state, reps=reps, scaler=scaler, report=report, log=log, sizes=sizes,
                coverage=coverage)


def use_synthetic_data(root: str | None = None, seed: int = 138) -> str:
    """Point the pipeline at a small FAKE PISA dataset (same file layout and column names),
    so every phase can be run and tested before the real files are available.
    Sets CS138_RAW_DIR / CS138_PROCESSED_DIR / CS138_RESULTS_DIR. Results are NOT real findings."""
    import os
    import tempfile
    from tests.synthetic_pisa import make_fake_pisa

    root = root or tempfile.mkdtemp(prefix="cs138_synth_")
    make_fake_pisa(os.path.join(root, "raw"), seed=seed)
    os.environ["CS138_RAW_DIR"] = os.path.join(root, "raw")
    os.environ["CS138_PROCESSED_DIR"] = os.path.join(root, "processed")
    os.environ["CS138_RESULTS_DIR"] = os.path.join(root, "results")
    print("SYNTHETIC data in", root, "- numbers are for testing only, not findings")
    return root


def phase1(data: dict) -> dict:
    """Phase 1: global factor matrix M_G.  Steps 1-2 run; steps 3-4 are a SKELETON.

    Pipeline idea
      1. Conditioning of the design (1.1): SVD of X_g^T (N x n). kappa and rank say how well the
         data can separate the effects of the 19 factors (near-collinear factors -> large kappa).
      2. Solver comparison (1.2): normal / QR / SVD on the same weighted problem, against numpy.
      3. M_G (1.4) = fit_factor_matrix(X_g, Y_g, method="qr", weights=column_weights(meta_g)).x
         Decide and record: method (QR unless rank deficient -> SVD), column weighting
         (config preprocessing.column_weighting), intercept or not (decisions note 6),
         pooled fit over both cycle pairs vs one M_G per pair (Phase 3.4 consistency check).
         Save M_G with metadata: N, pairs, weighting, method, residual, kappa.
      4. Effectiveness (1.5): imple.effectiveness (ideas listed there).
    """
    X, Y = data["X_g"], data["Y_g"]
    w = column_weights(data["meta_g"])
    design = summary(X.T)                                   # 1.1
    comparison, candidates = compare_methods(X, Y, w)       # 1.2

    # 1.4 TODO: choose and save M_G (see docstring step 3)
    M_G = None
    # _save("M_G", M_G=M_G, state=np.array(data["state"]), ...)

    # 1.5 TODO: effectiveness, e.g.
    # eff = imple.effectiveness.column_errors(M_G, X, Y, data["meta_g"])
    effectiveness = None
    return dict(M_G=M_G, design=design, comparison=comparison, candidates=candidates,
                effectiveness=effectiveness)


def phase2(data: dict, g: dict) -> dict:
    """Phase 2: Philippine factor matrix M_PH = M_G + dM.  Step 1 runs; steps 2-4 are a SKELETON.

    Pipeline idea
      1. PH design (2.1 input): SVD of X_ph^T. N_PH vs n tells whether dM X = R is under- or
         over-determined; kappa(X_ph) enters the error bound in docs/perturbation.md (c).
      2. Residual of the global matrix on PH data: R = Y_ph - M_G X_ph (needs M_G from Phase 1).
      3. dM (2.2): one primary solver from the list in imple/perturbation.py
         (minimum-norm, TSVD, low-rank, structured, ...), then M_PH = M_G + dM (2.3).
         Report residual, ||dM||_F / ||M_G||_F, Weyl bound, leave-one-cell-out error. Save.
      4. Comparison (2.4): imple.compare (ideas listed there).
    """
    X, Y = data["X_ph"], data["Y_ph"]
    design = summary(X.T)                                   # 2.1 input
    n, N = X.shape
    print(f"PH system dM X = R: {n}x{n} = {n * n} unknowns, {n * N} equations "
          f"({'under' if N < n else 'over'}determined), rank(X_ph) = {design['numerical_rank']}")

    R = None if g.get("M_G") is None else Y - g["M_G"] @ X   # 2.1 residual
    # 2.2 / 2.3 TODO: dM = <solver>(R, X); M_PH = g["M_G"] + dM; _save("M_PH", ...)
    dM = M_PH = None
    # 2.4 TODO: imple.compare.perturbation_size(g["M_G"], dM), category_blocks(...), ...
    return dict(design=design, R=R, dM=dM, M_PH=M_PH)


def _save(name: str, **arrays) -> None:
    out = path("results_dir")
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{name}.npz", **arrays)


if __name__ == "__main__":
    d = phase0()
    g = phase1(d)
    phase2(d, g)
