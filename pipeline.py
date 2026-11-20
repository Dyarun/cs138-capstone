"""Runs Phases 0 to 2 in order. The notebook calls these three functions, one per section.

    from pipeline import phase0, phase1, phase2
    data = phase0()                 # cleaned tables, groups, X/Y matrices
    g = phase1(data)                # M_G and its effectiveness
    ph = phase2(data, g)            # M_PH, dM and the comparison numbers

Every name, path and threshold comes from config/variables.json via imple.config.
Outputs are written under config paths.processed_dir and paths.results_dir.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from imple.config import (INTL, PHL, available_cycles, basis_year, cfg, factors, path,
                          state_factors, transition_pairs)
from imple.lsq import fit_factor_matrix, relative_error
from imple.perturbation import calibrate
from imple.svd import summary
from imple.tikhonov import choose_lambda, fit_factor_matrix_tikhonov
from prep.clean import apply_scaler, encode, fit_scaler, missing_report
from prep.groups import derive, group_masks, group_sizes
from prep.load import filter_countries, load_all
from prep.vectors import build_xy, column_weights, representative_vectors


def phase0() -> dict:
    """Data prep: load -> encode -> scale -> groups -> representative vectors -> X, Y."""
    if not state_factors():
        raise RuntimeError("obj/factor_dict.json has no included 'state' factors yet.")
    intl = filter_countries(load_all(INTL), "global")
    phl = filter_countries(load_all(PHL), "philippines")
    print("cycles used:", {INTL: available_cycles(INTL), PHL: available_cycles(PHL)})

    intl, phl = encode(intl, factors()), encode(phl, factors())
    report = pd.concat([missing_report(intl, factors()), missing_report(phl, factors())])

    scaler = fit_scaler(intl[intl["CYCLE"] == basis_year()], state_factors())   # one scaler for both
    intl, phl = apply_scaler(intl, scaler), apply_scaler(phl, scaler)

    intl, phl = derive(intl), derive(phl)
    masks_intl, masks_ph = group_masks(intl), group_masks(phl)
    sizes = pd.concat([group_sizes(intl, masks_intl), group_sizes(phl, masks_ph)])

    reps = pd.concat([representative_vectors(intl, masks_intl), representative_vectors(phl, masks_ph)])
    X_g, Y_g, meta_g = build_xy(reps, transition_pairs("global"))
    X_ph, Y_ph, meta_ph = build_xy(reps, transition_pairs("philippines"))

    out = path("processed_dir")
    out.mkdir(parents=True, exist_ok=True)
    reps.to_parquet(out / "representative_vectors.parquet")
    report.to_csv(out / "missing_report.csv", index=False)
    sizes.to_csv(out / "group_sizes.csv", index=False)
    np.savez(out / "xy.npz", X_g=X_g, Y_g=Y_g, X_ph=X_ph, Y_ph=Y_ph)
    meta_g.to_csv(out / "meta_global.csv", index=False)
    meta_ph.to_csv(out / "meta_ph.csv", index=False)
    return dict(X_g=X_g, Y_g=Y_g, meta_g=meta_g, X_ph=X_ph, Y_ph=Y_ph, meta_ph=meta_ph,
                reps=reps, scaler=scaler, report=report, sizes=sizes)


def phase1(data: dict) -> dict:
    """Global factor matrix M_G: plain least squares (all three methods) and Tikhonov."""
    X, Y = data["X_g"], data["Y_g"]
    w = column_weights(data["meta_g"])
    plain = {m: fit_factor_matrix(X, Y, method=m, weights=w) for m in ("normal", "qr", "svd")}
    lam = choose_lambda(X.T, Y.T)
    reg = fit_factor_matrix_tikhonov(X, Y, lam, M0=np.zeros((Y.shape[0], X.shape[0])), weights=w)
    M_G = reg.x
    effectiveness = data["meta_g"].assign(rel_error=relative_error(M_G, X, Y))
    _save("M_G", M_G=M_G, lam=lam)
    return dict(M_G=M_G, lam=lam, plain=plain, effectiveness=effectiveness,
                X_summary=summary(X), M_summary=summary(M_G))


def phase2(data: dict, g: dict) -> dict:
    """Philippine factor matrix M_PH = M_G + dM, and the comparison with M_G."""
    X, Y = data["X_ph"], data["Y_ph"]
    cal = calibrate(g["M_G"], X, Y, weights=column_weights(data["meta_ph"]))
    effectiveness = data["meta_ph"].assign(
        rel_error_MG=relative_error(g["M_G"], X, Y),
        rel_error_MPH=relative_error(cal.M_PH, X, Y),
    )
    _save("M_PH", M_PH=cal.M_PH, dM=cal.dM, lam=cal.lam)
    return dict(cal=cal, effectiveness=effectiveness,
                sv_G=summary(g["M_G"])["singular_values"], sv_PH=summary(cal.M_PH)["singular_values"])


def _save(name: str, **arrays) -> None:
    out = path("results_dir")
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{name}.npz", **arrays)


if __name__ == "__main__":
    d = phase0()
    g = phase1(d)
    phase2(d, g)
