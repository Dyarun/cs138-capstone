"""Loader for config/variables.json, the one file to edit when names change.

Every other module asks this one for column names, paths and settings, so a
renamed PISA variable or a moved data file is a one-line change in
config/variables.json.

Usage
-----
>>> from imple.config import cfg, col, raw_file, load_obj
>>> col("BULLIED", 2018)                    # -> 'BEINGBULLIED'
>>> col("PV_MATH", 2022, pv=3)              # -> 'PV3MATH'
>>> raw_file("phl", 2022, "student")        # -> Path to the PH 2022 student file (or None)
>>> available_cycles("intl")                # -> years in cycles.years whose file is set and on disk

Set the environment variable CS138_CONFIG to use a different config file
(for example a copy with local paths) without editing the repository one.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "variables.json"

INTL = "intl"   # dataset keys under paths.raw_files
PHL = "phl"


@lru_cache(maxsize=None)
def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def cfg(dotted: str | None = None, default: Any = None) -> Any:
    """Return the whole config, or one entry by dotted path (e.g. 'solver.svd.tol')."""
    data = _load(os.environ.get("CS138_CONFIG", str(DEFAULT_CONFIG)))
    if dotted is None:
        return data
    node: Any = data
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def resolve(ref: Any) -> Any:
    """Turn a 'codes.X.y'-style string reference (used in obj/*.json) into its config value."""
    if isinstance(ref, str) and ref.split(".", 1)[0] in cfg():
        value = cfg(ref)
        if value is not None:
            return value
    return ref


# ------------------------------------------------------------------ paths and cycles
def path(key: str) -> Path:
    """Absolute path for an entry under 'paths' (relative entries are resolved from the repo root)."""
    p = Path(cfg(f"paths.{key}"))
    return p if p.is_absolute() else REPO_ROOT / p


def datasets() -> list[str]:
    """Dataset keys under paths.raw_files, e.g. ['intl', 'phl']."""
    return list(cfg("paths.raw_files"))


def years() -> list[int]:
    """Cycles the pipeline uses (cycles.years)."""
    return [int(y) for y in cfg("cycles.years")]


def basis_year() -> int:
    """Cycle whose global sample defines the z-score scaling (cycles.basis)."""
    return int(cfg("cycles.basis"))


def raw_file(dataset: str, cycle: int | str, kind: str = "student") -> Path | None:
    """Path of a raw file, or None if it is not set in the config.

    dataset: 'intl' or 'phl'. kind: 'student', 'school', 'math', 'science',
    'reading' or 'problem_solving' (any key under paths.raw_files.<dataset>.<cycle>).
    File names are relative to paths.raw_dir and may include subfolders,
    e.g. "intl/2018/CY07_MSU_STU_QQQ.sav".
    """
    name = cfg(f"paths.raw_files.{dataset}.{cycle}.{kind}")
    return None if name is None else path("raw_dir") / name


def available_cycles(dataset: str, kind: str = "student") -> list[int]:
    """Cycles in cycles.years whose file is set in the config AND exists on disk.

    The pipeline loops over these, so a missing dataset is skipped instead of
    raising an error, and nothing else needs to be switched on when a file is added.
    """
    out = []
    for y in years():
        f = raw_file(dataset, y, kind)
        if f is not None and f.exists():
            out.append(y)
    return out


# ------------------------------------------------------------------ column names
def col(alias: str, cycle: int | str, pv: int | None = None) -> str | None:
    """Raw PISA column name for an alias in a given cycle.

    Aliases are the keys of 'columns' (variables) and 'keys' (ids). Returns
    None when the variable is not collected in that cycle. Plausible-value
    patterns such as 'PV{i}MATH' are filled in with `pv`.
    """
    keys = cfg("keys")
    if alias in keys:
        return keys[alias]
    entry = cfg("columns").get(alias)
    if entry is None:
        raise KeyError(f"Unknown alias '{alias}'. Add it to config/variables.json -> columns.")
    name = entry.get(str(cycle))
    if name is not None and "{i}" in name:
        if pv is None:
            raise ValueError(f"'{alias}' is a plausible-value column; pass pv=1..n.")
        name = name.format(i=pv)
    return name


def pv_columns(alias: str, cycle: int | str) -> list[str]:
    """All plausible-value columns of an outcome alias (e.g. PV1MATH..PV10MATH); [] if absent that cycle."""
    if cfg("columns").get(alias, {}).get(str(cycle)) is None:
        return []
    n = cfg(f"plausible_values.n.{cycle}")
    return [col(alias, cycle, pv=i) for i in range(1, n + 1)]


def replicate_weights() -> list[str]:
    w = cfg("weights")
    return [w["replicate_pattern"].format(i=i) for i in range(1, w["n_replicates"] + 1)]


def aliases(source: str | None = None) -> list[str]:
    """Variable aliases, optionally only those from the 'student' or 'school' file."""
    return [a for a, e in cfg("columns").items() if source is None or e["source"] == source]


def columns_for(cycle: int | str, source: str) -> dict[str, str]:
    """{raw column name: alias} for one file, ready for DataFrame.rename(columns=...).

    Plausible-value aliases expand into one entry per PV (PV_MATH_1, ...)."""
    out: dict[str, str] = {}
    for alias in aliases(source):
        name = cfg("columns")[alias].get(str(cycle))
        if name is None:
            continue
        if "{i}" in name:
            for i, c in enumerate(pv_columns(alias, cycle), start=1):
                out[c] = f"{alias}_{i}"
        else:
            out[name] = alias
    return out


def usecols(cycle: int | str, source: str) -> list[str]:
    """Raw columns to read from a file: ids, weights (student file) and the mapped variables."""
    cols = [cfg("keys.COUNTRY"), cfg("keys.SCHOOL_ID")]
    if source == "student":
        cols += [cfg("keys.STUDENT_ID"), cfg("keys.STRATUM"), cfg("weights.final")]
        cols += replicate_weights()
    cols += list(columns_for(cycle, source))
    return list(dict.fromkeys(cols))


def check_columns(df_columns, cycle: int | str, source: str) -> list[str]:
    """Columns the config expects that are missing from a loaded file (empty list = all good)."""
    present = set(df_columns)
    return [c for c in usecols(cycle, source) if c not in present]


def codes(alias: str) -> dict[str, list]:
    """Category codes for a categorical alias, e.g. codes('SCHOOL_TYPE')['public']."""
    return cfg(f"codes.{alias}")


# ------------------------------------------------------------------ obj/*.json
def load_obj(name: str) -> dict:
    """Load obj/<name>.json through the path in the config (factor_dict, group_dict, transition_pairs)."""
    return _load(str(path(name)))


def factors(role: str | None = None, include_only: bool = True) -> list[dict]:
    """Factor entries from factor_dict.json (the '_template' is never returned)."""
    out = load_obj("factor_dict").get("factors", [])
    return [f for f in out
            if (role is None or f["role"] == role) and (f.get("include", True) or not include_only)]


def state_factors() -> list[dict]:
    """Included factors with role 'state', in vector order. Their count is the dimension n of M."""
    return factors("state")


def groups() -> list[dict]:
    """Group entries from group_dict.json (the '_template' is never returned)."""
    return load_obj("group_dict").get("groups", [])


def transition_pairs(scope: str) -> list[dict]:
    """Enabled pairs for 'global' or 'philippines' whose two cycles are both in cycles.years."""
    ys = set(years())
    return [p for p in load_obj("transition_pairs")[scope]
            if p.get("enabled", True) and p["from"] in ys and p["to"] in ys]


def reload() -> None:
    """Clear the cache after editing a JSON file in a running notebook."""
    _load.cache_clear()
