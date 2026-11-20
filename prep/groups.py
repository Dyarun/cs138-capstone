"""0.5 Demographic groups from obj/group_dict.json.

Flow:
    df = derive(df)                       # adds the 'derived' columns (e.g. ESCS_PCTL, LOCATION)
    masks = group_masks(df)               # {group id: boolean Series}
    sizes = group_sizes(df, masks)        # n students and weighted population per (DATASET, CYCLE, COUNTRY)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from imple.config import cfg, resolve, load_obj, groups, path, write_json


def weighted_percentile_rank(values: pd.Series, weights: pd.Series) -> pd.Series:
    """Rank in [0, 1]: cumulative weight share of students at or below each value (NaN stays NaN).

    Tied values get the same rank (the share at or below the tie), so a
    quartile cut never splits students with identical ESCS."""
    out = pd.Series(np.nan, index=values.index)
    ok = values.notna() & weights.notna() & (weights > 0)
    if not ok.any():
        return out
    v, w = values[ok].to_numpy(float), weights[ok].to_numpy(float)
    order = np.argsort(v, kind="mergesort")
    vs, cw = v[order], np.cumsum(w[order])
    # last index of each run of equal values -> share at or below that value
    last = np.searchsorted(vs, vs, side="right") - 1
    rank_sorted = cw[last] / cw[-1]
    r = np.empty_like(rank_sorted)
    r[order] = rank_sorted
    out[ok] = r
    return out


def derive(df: pd.DataFrame) -> pd.DataFrame:
    """Add each entry of group_dict['derived'] that is not 'enabled': false.
       weighted_percentile_rank -> per groupby(entry['within']) (aliases COUNTRY, CYCLE)
       map_codes                -> label for each code list in entry['map'] (resolve references)
    """
    df = df.copy()
    for name, d in load_obj("group_dict").get("derived", {}).items():
        if not d.get("enabled", True):
            continue
        src = d["source"]
        if src not in df.columns:
            df[name] = np.nan
            continue
        if d["method"] == "weighted_percentile_rank":
            within = d.get("within") or []
            if within:
                parts = [weighted_percentile_rank(g[src], g["WEIGHT"])
                         for _, g in df.groupby(within, sort=False)]
                df[name] = pd.concat(parts).reindex(df.index)
            else:
                df[name] = weighted_percentile_rank(df[src], df["WEIGHT"])
        elif d["method"] == "map_codes":
            lab = pd.Series(pd.NA, index=df.index, dtype="object")
            for label, vals in d["map"].items():
                vals = resolve(vals)
                lab[df[src].isin(vals if isinstance(vals, list) else [vals])] = label
            df[name] = lab
        else:
            raise ValueError(f"derived '{name}': unknown method '{d['method']}'")
    return df


def condition_mask(df: pd.DataFrame, cond: dict) -> pd.Series:
    """One {var, op, value} condition -> boolean Series. ops: in, ==, !=, between.

    between [lo, hi] means lo < v <= hi, except that lo = 0 is inclusive, so
    consecutive quantile bands ([0, .25], [.25, .5], ...) never overlap.
    Missing values never satisfy a condition (also not '!=')."""
    var, op, value = cond["var"], cond["op"], resolve(cond["value"])
    if var not in df.columns:
        return pd.Series(False, index=df.index)
    s = df[var]
    if op == "in":
        m = s.isin(value if isinstance(value, list) else [value])
    elif op == "==":
        m = s == value
    elif op == "!=":
        m = (s != value) & s.notna()
    elif op == "between":
        lo, hi = value
        m = ((s > lo) | ((lo == 0) & (s >= lo))) & (s <= hi)
    else:
        raise ValueError(f"unknown op '{op}'")
    return m.fillna(False).astype(bool)


def group_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """AND of condition_mask over each group's rule; empty rule = everyone."""
    out = {}
    for g in groups():
        m = pd.Series(True, index=df.index)
        for cond in g["rule"]:
            m &= condition_mask(df, cond)
        out[g["id"]] = m
    return out


def fit_group_ids() -> list[str]:
    """Ids of the groups with fit: true (the columns of X and Y in Phases 1 and 2)."""
    return [g["id"] for g in groups() if g.get("fit", False)]


def group_sizes(df: pd.DataFrame, masks: dict[str, pd.Series]) -> pd.DataFrame:
    """Long table: DATASET, CYCLE, COUNTRY, GROUP, n, weight_sum. Used to fill size_intl / size_ph
    and to drop cells below countries.sample_threshold (column below_threshold)."""
    thr = cfg("countries.sample_threshold", 30)
    keys = ["DATASET", "CYCLE", "COUNTRY"]
    frames = []
    for gid, m in masks.items():
        sub = df.loc[m, keys + ["WEIGHT"]]
        if sub.empty:
            continue
        t = sub.groupby(keys).agg(n=("WEIGHT", "size"), weight_sum=("WEIGHT", "sum")).reset_index()
        t.insert(3, "GROUP", gid)
        frames.append(t)
    if not frames:
        return pd.DataFrame(columns=keys + ["GROUP", "n", "weight_sum", "below_threshold"])
    out = pd.concat(frames, ignore_index=True)
    out["below_threshold"] = out["n"] < thr
    return out


def write_back_sizes(sizes: pd.DataFrame) -> None:
    """Record per group: total students and weighted population per cycle, for intl and PH
    (size_intl / size_ph in obj/group_dict.json, deliverable 0.5)."""
    from imple.config import reload, INTL, PHL

    p = path("group_dict")
    d = json.loads(p.read_text(encoding="utf-8"))
    tot = sizes.groupby(["GROUP", "DATASET", "CYCLE"])[["n", "weight_sum"]].sum().reset_index()
    for g in d.get("groups", []):
        for ds, field in ((INTL, "size_intl"), (PHL, "size_ph")):
            r = tot[(tot["GROUP"] == g["id"]) & (tot["DATASET"] == ds)]
            g[field] = {str(int(c)): {"n": int(n), "population": round(float(w))}
                        for c, n, w in zip(r["CYCLE"], r["n"], r["weight_sum"])} or None
    write_json(p, d)
    reload()
