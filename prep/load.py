"""0.1 / 0.2 Load raw PISA files into one tidy student table per (dataset, cycle).

Flow for one cycle:
    student = read_sav(raw_file(ds, y, "student"), usecols(y, "student"))
    school  = read_sav(raw_file(ds, y, "school"),  usecols(y, "school"))
    df = merge_school(student, school)             # COUNTRY + SCHOOL_ID join, adds school variables
    df = to_aliases(df, y)                         # raw names -> aliases from config
    df["CYCLE"], df["DATASET"] = y, ds

Only cycles returned by imple.config.available_cycles(ds) are loaded, so a
file that is not set in the config (or not on disk) is skipped.

The Philippine sample is already inside the international files. When
paths.raw_files.phl.<cycle> is null, load_datasets() splits it off the
international table instead of reading the (very large) file twice.
"""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import pandas as pd

from imple.config import (INTL, PHL, available_cycles, cfg, check_columns, columns_for, path,
                          raw_file, usecols)


def read_sav(path_: Path, columns: list[str]) -> pd.DataFrame:
    """Read only `columns` from an SPSS file with pyreadstat (declared user-missing codes become NaN).

    The file's column list is read first (metadataonly=True); requested columns
    that the file does not have are reported with a warning and skipped, so one
    renamed variable does not stop the whole load. Large files are read with
    several processes.
    """
    import pyreadstat

    _, meta = pyreadstat.read_sav(str(path_), metadataonly=True)
    present = set(meta.column_names)
    missing = [c for c in columns if c not in present]
    if missing:
        warnings.warn(f"{Path(path_).name}: {len(missing)} expected column(s) not in file: {missing}")
    wanted = [c for c in columns if c in present]
    kwargs = dict(usecols=wanted, disable_datetime_conversion=True)
    if Path(path_).stat().st_size > 200 * 2**20:
        df, _ = pyreadstat.read_file_multiprocessing(pyreadstat.read_sav, str(path_), **kwargs)
    else:
        df, _ = pyreadstat.read_sav(str(path_), **kwargs)
    return df


def merge_school(student: pd.DataFrame, school: pd.DataFrame | None) -> pd.DataFrame:
    """Left-join school variables onto students by the raw COUNTRY + SCHOOL_ID keys.

    Run before to_aliases, so both tables still use raw PISA names."""
    if school is None or school.empty:
        return student
    keys = [cfg("keys.COUNTRY"), cfg("keys.SCHOOL_ID")]
    school = school.drop_duplicates(subset=keys)
    extra = [c for c in school.columns if c not in keys and c not in student.columns]
    out = student.merge(school[keys + extra], on=keys, how="left", validate="many_to_one")
    assert len(out) == len(student), "school merge changed the number of students"
    return out


def to_aliases(df: pd.DataFrame, cycle: int) -> pd.DataFrame:
    """Rename raw columns to aliases: columns_for(cycle, 'student') | columns_for(cycle, 'school'),
    plus keys (CNT -> COUNTRY, ...) and the final weight -> 'WEIGHT'. Replicate weights keep their names."""
    mapping = {**columns_for(cycle, "student"), **columns_for(cycle, "school")}
    mapping.update({raw: alias for alias, raw in cfg("keys").items()})
    mapping[cfg("weights.final")] = "WEIGHT"
    return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})


def _cache_file(dataset: str, cycle: int) -> Path:
    sig = repr((usecols(cycle, "student"), usecols(cycle, "school"),
                str(raw_file(dataset, cycle, "student")), str(raw_file(dataset, cycle, "school"))))
    h = hashlib.md5(sig.encode()).hexdigest()[:8]
    return path("processed_dir") / "cache" / f"raw_{dataset}_{cycle}_{h}.parquet"


def load_cycle(dataset: str, cycle: int) -> pd.DataFrame:
    """Load, merge and rename one cycle (flow in module docstring).

    The renamed table is cached as parquet under processed_dir/cache (config
    preprocessing.cache_raw); the cache key changes whenever the requested
    columns or file names change, so a stale cache is never used.
    """
    cache = _cache_file(dataset, cycle)
    if cfg("preprocessing.cache_raw", True) and cache.exists():
        return pd.read_parquet(cache)

    student = read_sav(raw_file(dataset, cycle, "student"), usecols(cycle, "student"))
    sch_file = raw_file(dataset, cycle, "school")
    school = read_sav(sch_file, usecols(cycle, "school")) if sch_file and sch_file.exists() else None
    if school is None:
        warnings.warn(f"{dataset} {cycle}: no school file, school variables will be missing")
    df = to_aliases(merge_school(student, school), cycle)
    df["CYCLE"] = int(cycle)
    df["DATASET"] = dataset
    for c in ("COUNTRY", "STRATUM"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    if cfg("preprocessing.cache_raw", True):
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
    return df


def filter_countries(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    """scope 'global': countries.global_include ('all' or list) minus countries.global_exclude.
    scope 'philippines': COUNTRY == countries.philippines."""
    if df.empty:
        return df
    c = df["COUNTRY"]
    if scope == "philippines":
        keep = c == cfg("countries.philippines")
    elif scope == "global":
        inc = cfg("countries.global_include", "all")
        keep = pd.Series(True, index=df.index) if inc == "all" else c.isin(inc)
        keep &= ~c.isin(cfg("countries.global_exclude", []))
    else:
        raise ValueError(f"unknown scope '{scope}'")
    return df.loc[keep].reset_index(drop=True)


def load_all(dataset: str) -> pd.DataFrame:
    """Concatenate load_cycle over available_cycles(dataset) (empty table if none)."""
    frames = [load_cycle(dataset, y) for y in available_cycles(dataset)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(intl, phl) tables, already filtered to the 'global' and 'philippines' scopes.

    Cycles with their own PH file use it; the other cycles take the PH rows
    from the international file (config countries.ph_source)."""
    intl_all = load_all(INTL)
    if intl_all.empty:
        raise FileNotFoundError(
            "No international PISA file found. Put the files under paths.raw_dir "
            f"({path('raw_dir')}) or set CS138_RAW_DIR; names are in config paths.raw_files.")
    phl_own = load_all(PHL)
    own_cycles = set(phl_own["CYCLE"]) if not phl_own.empty else set()
    from_intl = filter_countries(intl_all[~intl_all["CYCLE"].isin(own_cycles)], "philippines")
    phl = filter_countries(pd.concat([phl_own, from_intl.assign(DATASET=PHL)], ignore_index=True),
                           "philippines")
    intl = filter_countries(intl_all, "global")
    return intl, phl


def stratum_labels(cycle: int, country: str | None = None) -> pd.Series:
    """Value labels of the STRATUM variable in the student file, to fill the PH_REGION
    map in obj/group_dict.json. Keeps codes starting with `country` (default PHL)."""
    import pyreadstat

    country = country or cfg("countries.philippines")
    f = raw_file(PHL, cycle, "student")
    f = f if f is not None and f.exists() else raw_file(INTL, cycle, "student")
    _, meta = pyreadstat.read_sav(str(f), metadataonly=True)
    labels = meta.variable_value_labels.get(cfg("keys.STRATUM"), {})
    s = pd.Series(labels, name="label")
    return s[s.index.astype(str).str.startswith(country)]


def column_checks(cycle: int, dataset: str = INTL) -> dict[str, list[str]]:
    """Expected columns missing from each raw file of one cycle ([] = all good)."""
    import pyreadstat

    out = {}
    for kind in ("student", "school"):
        f = raw_file(dataset, cycle, kind)
        if f is None or not f.exists():
            out[kind] = ["<file not found>"]
            continue
        _, meta = pyreadstat.read_sav(str(f), metadataonly=True)
        out[kind] = check_columns(meta.column_names, cycle, kind)
    return out
