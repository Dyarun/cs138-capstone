"""0.6 Representative vectors and the X, Y matrices for M X = Y.

Flow:
    reps = representative_vectors(df, masks)             # one row per (DATASET, CYCLE, COUNTRY, GROUP)
    X, Y, meta = build_xy(reps[reps.DATASET == "intl"], transition_pairs("global"), fit_group_ids())
    X_ph, Y_ph, meta_ph = build_xy(reps[reps.DATASET == "phl"], transition_pairs("philippines"), fit_group_ids())

Each column j of X / Y is one (pair, country, group): x_j from the pair's 'from'
cycle, y_j from its 'to' cycle. Rows follow imple.config.state_factors() order
(or the `state` list passed in, after factors dropped for missingness).

Uncertainty (CLT): for every factor the group mean comes with a standard error
    se = weighted SD / sqrt(n_eff),   n_eff = (sum w)^2 / sum w^2   (Kish effective size).
This ignores the clustered sample design, so it understates the true error;
Phase 3 replaces it with replicate-weight (BRR) errors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from imple.config import cfg, state_factors

KEYS = ["DATASET", "CYCLE", "COUNTRY"]


def _state(state: list[str] | None) -> list[str]:
    return list(state) if state is not None else [f["alias"] for f in state_factors()]


def representative_vectors(df: pd.DataFrame, masks: dict[str, pd.Series],
                           state: list[str] | None = None) -> pd.DataFrame:
    """For every group mask and (DATASET, CYCLE, COUNTRY): weighted mean of each state alias
    (weighted_share for binaries is the same formula on 0/1), plus n, weight_sum, n_eff and a
    spread column <alias>_se per alias (weighted SD / sqrt(n_eff), for the CLT-based uncertainty).

    Missing values are skipped per factor: each mean uses the students who have that factor."""
    state = _state(state)
    frames = []
    for gid, m in masks.items():
        sub = df.loc[m, KEYS + ["WEIGHT"] + state]
        if sub.empty:
            continue
        w = sub["WEIGHT"].to_numpy(float)
        parts = {"n": sub.groupby(KEYS).size(), "weight_sum": sub.groupby(KEYS)["WEIGHT"].sum()}
        parts["n_eff"] = parts["weight_sum"] ** 2 / (sub.assign(_w2=w ** 2).groupby(KEYS)["_w2"].sum())
        tmp = sub[KEYS].copy()
        for a in state:
            x = sub[a].to_numpy(float)
            ok = ~np.isnan(x)
            tmp[f"{a}__w"] = np.where(ok, w, 0.0)
            tmp[f"{a}__wx"] = np.where(ok, w * x, 0.0)
            tmp[f"{a}__wxx"] = np.where(ok, w * x * x, 0.0)
            tmp[f"{a}__ww"] = np.where(ok, w * w, 0.0)
        s = tmp.groupby(KEYS).sum()
        for a in state:
            sw = s[f"{a}__w"]
            mean = s[f"{a}__wx"] / sw.where(sw > 0)
            var = (s[f"{a}__wxx"] / sw.where(sw > 0) - mean ** 2).clip(lower=0)
            neff = sw ** 2 / s[f"{a}__ww"].where(s[f"{a}__ww"] > 0)
            parts[a] = mean
            parts[f"{a}_se"] = np.sqrt(var / neff)
        t = pd.DataFrame(parts).reset_index()
        t.insert(3, "GROUP", gid)
        frames.append(t)
    cols = KEYS + ["GROUP", "n", "weight_sum", "n_eff"] + state + [f"{a}_se" for a in state]
    if not frames:
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)[cols]


def xy_coverage(reps: pd.DataFrame, pairs: list[dict], group_ids: list[str] | None = None,
                state: list[str] | None = None) -> pd.DataFrame:
    """Every candidate (pair, country, group) cell with keep = True/False and the reason it was dropped.
    Pairs whose cycles are not loaded at all are skipped silently (see loaded_pairs).

    A cell is kept only if the group exists in both cycles, has at least
    countries.sample_threshold students in both, and has no missing factor."""
    state = _state(state)
    thr = cfg("countries.sample_threshold", 30)
    if group_ids is not None:
        reps = reps[reps["GROUP"].isin(group_ids)]
    on = ["DATASET", "COUNTRY", "GROUP"]
    loaded = set(reps["CYCLE"].unique())
    frames = []
    for p in pairs:
        if p["from"] not in loaded or p["to"] not in loaded:
            continue                                  # cycle not loaded (e.g. 2025 file not yet set)
        a = reps[reps["CYCLE"] == p["from"]]
        b = reps[reps["CYCLE"] == p["to"]]
        mg = a.merge(b, on=on, how="outer", suffixes=("_x", "_y"), indicator=True)
        reason = pd.Series("", index=mg.index, dtype=object)
        reason[mg["_merge"] == "left_only"] = "missing in 'to' cycle"
        reason[mg["_merge"] == "right_only"] = "missing in 'from' cycle"
        small = (mg["n_x"] < thr) | (mg["n_y"] < thr)
        reason[(reason == "") & small] = f"fewer than {thr} students"
        nan = mg[[f"{s}_x" for s in state] + [f"{s}_y" for s in state]].isna().any(axis=1)
        reason[(reason == "") & nan] = "missing factor value"
        mg["pair"], mg["years"], mg["reason"] = p["id"], p.get("years"), reason
        mg["keep"] = reason == ""
        frames.append(mg.drop(columns="_merge"))
    if not frames:
        return pd.DataFrame(columns=["pair", "years"] + on + ["keep", "reason"])
    return pd.concat(frames, ignore_index=True)


def loaded_pairs(reps: pd.DataFrame, pairs: list[dict]) -> list[str]:
    """Ids of the pairs whose two cycles are both present in `reps`."""
    loaded = set(reps["CYCLE"].unique())
    return [p["id"] for p in pairs if p["from"] in loaded and p["to"] in loaded]


def build_xy(reps: pd.DataFrame, pairs: list[dict], group_ids: list[str] | None = None,
             state: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Stack columns over pairs x countries x groups.
    Keep a column only if both cycles exist, n >= countries.sample_threshold in both, and no NaN
    (see xy_coverage for the dropped cells and why).
    meta: one row per column (pair id, years, COUNTRY, GROUP, n_from, n_to, weight) so results
    can be traced back. weight = mean of the two cycles' weighted populations."""
    state = _state(state)
    cov = xy_coverage(reps, pairs, group_ids, state)
    kept = cov[cov["keep"]].reset_index(drop=True)
    X = kept[[f"{s}_x" for s in state]].to_numpy(float).T
    Y = kept[[f"{s}_y" for s in state]].to_numpy(float).T
    meta = pd.DataFrame({
        "pair": kept["pair"], "years": kept["years"], "DATASET": kept["DATASET"],
        "COUNTRY": kept["COUNTRY"], "GROUP": kept["GROUP"],
        "n_from": kept["n_x"].astype(int) if len(kept) else [], "n_to": kept["n_y"].astype(int) if len(kept) else [],
        "weight": (kept["weight_sum_x"] + kept["weight_sum_y"]) / 2,
    })
    return X.reshape(len(state), -1), Y.reshape(len(state), -1), meta


def column_weights(meta: pd.DataFrame, scheme: str | None = None) -> np.ndarray:
    """Per-column weights for the fits, normalised to mean 1.

    scheme (default config preprocessing.column_weighting):
      'country_equal' every (pair, country) gets the same total weight, split over its
                      cells by population (the OECD-average convention);
      'population'    weighted population of the cell (populous countries dominate);
      'sqrt'          square root of the population;
      'equal'         1 for every cell."""
    scheme = scheme or cfg("preprocessing.column_weighting", "country_equal")
    if len(meta) == 0:
        return np.zeros(0)
    pop = meta["weight"].to_numpy(float)
    if scheme == "population":
        w = pop
    elif scheme == "sqrt":
        w = np.sqrt(pop)
    elif scheme == "equal":
        w = np.ones(len(meta))
    elif scheme == "country_equal":
        tot = meta.groupby(["pair", "COUNTRY"])["weight"].transform("sum").to_numpy(float)
        w = pop / tot
    else:
        raise ValueError(f"unknown column weighting '{scheme}'")
    return w / w.mean()
