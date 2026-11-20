"""0.1 - 0.4 Clean, encode and scale student-level factors.

Order (same for intl and phl, so M_G and M_PH share units):
    df = drop_missing_codes(df)                    # PISA missing codes -> NaN (safety net after pyreadstat)
    df = encode(df, factors())                     # factor_dict 'encode' rules -> one column per alias
    report = missing_report(df, factors())         # missing %, n valid -> fills factor_dict
    log = verification_log(intl, phl)              # row counts, ranges, weights (deliverable 0.1 / 0.2)
    scaler = fit_scaler(global_basis_df, state_factors())   # basis = config cycles.basis, global scope
    df = apply_scaler(df, scaler)                  # reuse the SAME scaler for every cycle and for PH

The scaler is {alias: {"mean", "sd", "clip"}}. Clipping bounds (factor_dict
encode.clip_quantiles) are also fitted on the global basis sample, so a
clipped factor is cut at the same value in every cycle and in the PH data.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from imple.config import INTL, PHL, cfg, path, resolve, state_factors, write_json

INDEX_MISSING_FLOOR = 9995      # PISA index codes 9995-9999 and 9999995-9999999


# ------------------------------------------------------------------ missing codes
def _coded_aliases() -> dict[str, set]:
    """{alias: set of valid codes} for every categorical alias listed in config 'codes'."""
    out = {}
    for alias, groups_ in cfg("codes").items():
        if alias.startswith("_") or not isinstance(groups_, dict):
            continue
        out[alias] = {v for vals in groups_.values() for v in vals}
    return out


def drop_missing_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Set PISA missing codes to NaN.

    pyreadstat already converts declared user-missing values, so this is a
    safety net for files where codes are not declared:
      * categorical aliases with config 'codes': any value outside the listed codes -> NaN
        (a blanket "5..9 is missing" rule would wrongly delete SC001Q01TA = 5, 'large city');
      * every other numeric variable column: values >= 9995 -> NaN.
    Ids, weights and the bookkeeping columns are never touched.
    """
    df = df.copy()
    protected = {"COUNTRY", "SCHOOL_ID", "STUDENT_ID", "STRATUM", "CYCLE", "DATASET", "WEIGHT"}
    coded = _coded_aliases()
    for c in df.columns:
        if c in protected or c.startswith("W_FSTURWT") or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if c in coded:
            df.loc[~df[c].isin(coded[c]) & df[c].notna(), c] = np.nan
        else:
            df.loc[df[c] >= INDEX_MISSING_FLOOR, c] = np.nan
    return df


# ------------------------------------------------------------------ encoding
def pv_mean(df: pd.DataFrame, source_alias: str, divide_by: float) -> pd.Series:
    """Mean of the plausible-value columns <source_alias>_1..n, divided by preprocessing.pv_scale_divisor.
    Return all-NaN if the cycle has no PVs for this domain (e.g. PROB in some cycles)."""
    cols = [c for c in df.columns if c.startswith(f"{source_alias}_") and c.rsplit("_", 1)[1].isdigit()]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    return df[cols].mean(axis=1, skipna=False) / float(divide_by)


def encode(df: pd.DataFrame, factor_list: list[dict]) -> pd.DataFrame:
    """Add one column named f['alias'] per factor according to f['encode']['type']:
       as_is    -> copy (clip_quantiles is applied later, by the scaler, with global bounds)
       pv_mean  -> pv_mean(...)
       binary   -> isin(resolve(true_codes)) as 0/1, NaN kept NaN
       category -> label from resolve(codes)
    A factor whose source column is absent in a cycle becomes all-NaN (shows up in missing_report).
    """
    df = df.copy()
    for f in factor_list:
        alias, src, enc = f["alias"], f["source_alias"], f["encode"]
        kind = enc["type"]
        if kind == "pv_mean":
            df[alias] = pv_mean(df, src, resolve(enc.get("divide_by", 1.0)))
            continue
        if src not in df.columns:
            df[alias] = np.nan
            continue
        s = df[src]
        if kind == "as_is":
            df[alias] = pd.to_numeric(s, errors="coerce") if alias != src else s
        elif kind == "binary":
            true_codes = resolve(enc["true_codes"])
            df[alias] = np.where(s.isna(), np.nan, s.isin(true_codes).astype(float))
        elif kind == "category":
            mapping = resolve(enc["codes"])
            lab = pd.Series(pd.NA, index=df.index, dtype="object")
            for label, vals in mapping.items():
                lab[s.isin(vals)] = label
            df[alias] = lab
        else:
            raise ValueError(f"{alias}: unknown encode type '{kind}'")
    return df


# ------------------------------------------------------------------ reports
def missing_report(df: pd.DataFrame, factor_list: list[dict]) -> pd.DataFrame:
    """Per alias, per (DATASET, CYCLE): missing %, n valid. Flag > preprocessing.max_missing_rate_per_factor.

    missing_pct is unweighted; missing_pct_w uses the final student weight
    (the share of the represented population with no value)."""
    limit = cfg("preprocessing.max_missing_rate_per_factor", 0.3)
    rows = []
    if df.empty:
        return pd.DataFrame(columns=["DATASET", "CYCLE", "alias", "role", "n", "n_valid",
                                     "missing_pct", "missing_pct_w", "above_limit"])
    for (ds, cyc), part in df.groupby(["DATASET", "CYCLE"]):
        w = part["WEIGHT"] if "WEIGHT" in part else pd.Series(1.0, index=part.index)
        for f in factor_list:
            a = f["alias"]
            valid = part[a].notna() if a in part else pd.Series(False, index=part.index)
            miss = 1 - valid.mean()
            miss_w = 1 - w[valid].sum() / w.sum()
            rows.append(dict(DATASET=ds, CYCLE=cyc, alias=a, role=f["role"], n=len(part),
                             n_valid=int(valid.sum()), missing_pct=round(100 * miss, 2),
                             missing_pct_w=round(100 * miss_w, 2), above_limit=bool(miss > limit)))
    return pd.DataFrame(rows)


def active_state(report: pd.DataFrame) -> list[str]:
    """State aliases kept in the model, in factor_dict order.

    Policy (config preprocessing.missing_policy): a state factor is dropped if its
    missing rate exceeds the limit in ANY (dataset, cycle) that is used, because
    the same n-dimensional state space must exist for every column of X and Y."""
    bad = set(report.loc[report["above_limit"] & (report["role"] == "state"), "alias"])
    if bad:
        warnings.warn(f"state factors dropped for missingness above the limit: {sorted(bad)}")
    return [f["alias"] for f in state_factors() if f["alias"] not in bad]


def _range_for(f: dict) -> list | None:
    rc = cfg("preprocessing.range_checks", {})
    if f["alias"] in rc:
        return rc[f["alias"]]
    if f["encode"]["type"] == "pv_mean":
        div = resolve(f["encode"].get("divide_by", 1.0))
        lo, hi = rc.get("pv_raw", [0, 1000])
        return [lo / div, hi / div]
    if f["type"] == "binary":
        return rc.get("share", [0, 1])
    if f["type"] == "continuous":
        return rc.get("index")
    return None


def verification_log(intl: pd.DataFrame, phl: pd.DataFrame) -> pd.DataFrame:
    """Checks for deliverables 0.1 / 0.2, one row per check: scope, CYCLE, check, value, expected, ok.

    * student counts against config verification.students (the international
      file includes PHL, so the 'intl' check adds the two tables back together);
    * number of countries, PH schools, duplicate student ids;
    * positive final weights;
    * value ranges of every encoded state factor (config preprocessing.range_checks).
    Run on the ENCODED, UNSCALED tables.
    """
    tol = cfg("verification.tolerance", 0.01)
    expected = cfg("verification.students", {})
    rows = []

    def add(scope, cyc, check, value, exp, ok):
        rows.append(dict(scope=scope, CYCLE=cyc, check=check, value=value, expected=exp, ok=bool(ok)))

    both = pd.concat([intl, phl], ignore_index=True)
    for cyc in sorted(both["CYCLE"].unique()):
        allc = both[both["CYCLE"] == cyc]
        ph = phl[phl["CYCLE"] == cyc]
        for scope, part in ((INTL, allc), (PHL, ph)):
            exp = expected.get(scope, {}).get(str(cyc))
            n = len(part)
            add(scope, cyc, "students", n, exp, exp is None or abs(n - exp) <= tol * exp)
            if "STUDENT_ID" in part:
                d = int(part.duplicated(["COUNTRY", "STUDENT_ID"]).sum())
                add(scope, cyc, "duplicate student ids", d, 0, d == 0)
            if "WEIGHT" in part:
                bad = int((~(part["WEIGHT"] > 0)).sum())
                add(scope, cyc, "non-positive or missing weights", bad, 0, bad == 0)
        add(INTL, cyc, "countries (incl. PHL)", int(allc["COUNTRY"].nunique()), None, True)
        if "SCHOOL_ID" in ph:
            add(PHL, cyc, "PH schools", int(ph["SCHOOL_ID"].nunique()), None, True)
        for f in state_factors():
            rng = _range_for(f)
            if rng is None or f["alias"] not in allc:
                continue
            v = allc[f["alias"]].dropna()
            out = int(((v < rng[0]) | (v > rng[1])).sum())
            add("both", cyc, f"range {f['alias']} in {rng}", out, 0, out == 0)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ scaling
def weighted_mean_sd(x: pd.Series, w: pd.Series) -> tuple[float, float]:
    """Weighted mean and standard deviation, ignoring NaN."""
    x = pd.to_numeric(x, errors="coerce")
    ok = x.notna() & w.notna()
    if not ok.any():
        return float("nan"), float("nan")
    xv, wv = x[ok].to_numpy(float), w[ok].to_numpy(float)
    m = np.average(xv, weights=wv)
    sd = np.sqrt(np.average((xv - m) ** 2, weights=wv))
    return float(m), float(sd)


def _weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    order = np.argsort(x)
    cw = np.cumsum(w[order])
    return float(x[order][np.searchsorted(cw, q * cw[-1])])


def fit_scaler(df_basis: pd.DataFrame, factor_list: list[dict]) -> dict[str, dict]:
    """{alias: {"mean", "sd", "clip"}} from the global sample of the basis cycle, weighted by WEIGHT.
    clip = [lo, hi] from encode.clip_quantiles (weighted quantiles), or None."""
    w = df_basis["WEIGHT"]
    out = {}
    for f in factor_list:
        a = f["alias"]
        x = pd.to_numeric(df_basis[a], errors="coerce")
        clip = None
        qs = f["encode"].get("clip_quantiles")
        if qs:
            ok = x.notna() & w.notna()
            xv, wv = x[ok].to_numpy(float), w[ok].to_numpy(float)
            clip = [_weighted_quantile(xv, wv, qs[0]), _weighted_quantile(xv, wv, qs[1])]
            x = x.clip(*clip)
        m, sd = weighted_mean_sd(x, w)
        if not np.isfinite(sd) or sd == 0:
            warnings.warn(f"{a}: zero or undefined SD in the basis sample; left unscaled (sd = 1)")
            sd = 1.0
        out[a] = dict(mean=m, sd=sd, clip=clip)
    return out


def apply_scaler(df: pd.DataFrame, scaler: dict[str, dict]) -> pd.DataFrame:
    """z = (clip(value) - mean) / sd for every alias in the scaler."""
    df = df.copy()
    for a, s in scaler.items():
        x = pd.to_numeric(df[a], errors="coerce")
        if s.get("clip"):
            x = x.clip(*s["clip"])
        df[a] = (x - s["mean"]) / s["sd"]
    return df


def unscale(values, alias: str, scaler: dict[str, dict]):
    """Back to original units (for tables in the paper)."""
    s = scaler[alias]
    return np.asarray(values) * s["sd"] + s["mean"]


# ------------------------------------------------------------------ write-back
def write_back_missing(report: pd.DataFrame) -> None:
    """Record missing_pct and n_valid per 'dataset_cycle' in obj/factor_dict.json (deliverable 0.3)."""
    from imple.config import reload

    p = path("factor_dict")
    d = json.loads(p.read_text(encoding="utf-8"))
    for f in d.get("factors", []):
        r = report[report["alias"] == f["alias"]]
        if r.empty:
            continue
        key = r["DATASET"].astype(str) + "_" + r["CYCLE"].astype(str)
        f["missing_pct"] = dict(zip(key, r["missing_pct"].astype(float)))
        f["n_valid"] = dict(zip(key, r["n_valid"].astype(int)))
    write_json(p, d)
    reload()
