"""Phase 0 self-test on synthetic PISA files: python -m tests.test_phase0 (from the repo root).

Checks the properties every later phase relies on, not just that the code runs.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _self_test(verbose: bool = True) -> bool:
    saved = {k: os.environ.get(k) for k in ("CS138_RAW_DIR", "CS138_PROCESSED_DIR", "CS138_RESULTS_DIR")}
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))

    try:
        import warnings
        from imple.config import PHL, cfg, state_factors
        from pipeline import phase0, use_synthetic_data
        from prep.clean import drop_missing_codes, encode, weighted_mean_sd
        from prep.groups import fit_group_ids, group_masks, derive
        from prep.load import load_datasets
        from prep.vectors import column_weights

        use_synthetic_data(tempfile.mkdtemp(prefix="cs138_test_"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = phase0()
            intl_raw, phl_raw = load_datasets()

        state = d["state"]
        check("PH rows split off the international file", (phl_raw["COUNTRY"] == "PHL").all()
              and not (intl_raw["COUNTRY"] == "PHL").any())
        cleaned = drop_missing_codes(intl_raw)
        check("planted 9999 index codes removed", cleaned["BELONG"].max() < 9995,
              f"raw max {intl_raw['BELONG'].max():.0f}")
        check("planted code 9 in SC001Q01TA removed", not (cleaned["SCHOOL_COMMUNITY"] == 9).any())
        check("renamed variables mapped to one alias (BEINGBULLIED 2018 -> BULLIED)",
              intl_raw.groupby("CYCLE")["BULLIED"].apply(lambda s: s.notna().mean() > 0.9).all())

        # quartiles: within each country-cycle, each ESCS quartile holds ~25% of the weight
        enc = derive(encode(cleaned, __import__("imple.config", fromlist=["factors"]).factors()))
        enc["_q"] = np.ceil(enc["ESCS_PCTL"] * 4).clip(1, 4)
        share = enc.groupby(["COUNTRY", "CYCLE", "_q"])["WEIGHT"].sum() / enc.groupby(["COUNTRY", "CYCLE"])["WEIGHT"].sum()
        check("ESCS quartiles hold ~25% of each country's weight", share.between(0.2, 0.3).all(),
              f"range {share.min():.3f}-{share.max():.3f}")

        # fit cells partition the students with a known location
        masks = group_masks(enc)
        fit = fit_group_ids()
        cover = sum(masks[g].astype(int) for g in fit)
        known = enc["LOCATION"].notna() & enc["ESCS_PCTL"].notna()
        check("fit cells partition students with known ESCS and location",
              (cover[known] == 1).all() and (cover[~known] == 0).all(), f"{len(fit)} cells")

        # scaling: weighted mean 0 / sd 1 on the global basis cycle
        sc = d["scaler"]
        from prep.clean import apply_scaler
        b = apply_scaler(enc[enc["CYCLE"] == cfg("cycles.basis")], sc)
        m, s = weighted_mean_sd(b["MATH"], b["WEIGHT"])
        check("one scaler: global basis MATH has weighted mean 0, sd 1", abs(m) < 1e-9 and abs(s - 1) < 1e-9,
              f"mean {m:.2e}, sd {s:.6f}")
        check("clip bounds fitted once (STRATIO)", sc["STRATIO"]["clip"] is not None)

        # representative vector = weighted mean, recomputed by hand for one cell
        reps = d["reps"]
        r = reps[(reps["DATASET"] == PHL) & (reps["GROUP"] == "ALL")].iloc[0]
        ph = apply_scaler(derive(encode(drop_missing_codes(phl_raw), __import__("imple.config", fromlist=["factors"]).factors())), sc)
        cell = ph[ph["CYCLE"] == r["CYCLE"]]
        v = cell["BELONG"].notna()
        hand = np.average(cell.loc[v, "BELONG"], weights=cell.loc[v, "WEIGHT"])
        check("representative vector equals the hand-computed weighted mean", np.isclose(hand, r["BELONG"]),
              f"{hand:.6f} vs {r['BELONG']:.6f}")
        check("standard errors positive and small", (reps[[f"{a}_se" for a in state]].stack() >= 0).all()
              and r["MATH_se"] < 0.1, f"PH MATH se {r['MATH_se']:.4f}")

        # X, Y
        for tag in ("g", "ph"):
            X, Y, meta = d[f"X_{tag}"], d[f"Y_{tag}"], d[f"meta_{tag}"]
            check(f"X_{tag}, Y_{tag} are n x N with no NaN, meta has N rows",
                  X.shape == Y.shape and X.shape[0] == len(state) and X.shape[1] == len(meta)
                  and not np.isnan(X).any() and not np.isnan(Y).any(), f"{X.shape}")
        mg = d["meta_g"]
        check("only fit cells enter X", mg["GROUP"].isin(fit).all())
        check("PHL kept out of the global fit", not (mg["COUNTRY"] == "PHL").any())
        w = column_weights(mg, "country_equal")
        tot = pd.Series(w).groupby([mg["pair"].values, mg["COUNTRY"].values]).sum()
        check("country_equal weights: every country-pair has the same total", np.allclose(tot, tot.iloc[0]))
        check("n equals the number of included state factors", len(state) == len(state_factors()))
    except NotImplementedError:
        raise
    except Exception as e:  # report instead of crashing run_all
        import traceback
        traceback.print_exc()
        check("phase 0 ran without errors", False, repr(e))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return ok


if __name__ == "__main__":
    sys.exit(0 if _self_test() else 1)
