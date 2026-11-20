"""0.1 / 0.2 Load raw PISA files into one tidy student table per (dataset, cycle).

Flow for one cycle:
    student = read_sav(raw_file(ds, y, "student"), usecols(y, "student"))
    school  = read_sav(raw_file(ds, y, "school"),  usecols(y, "school"))
    df = merge_school(student, school)             # SCHOOL_ID join, adds school aliases
    df = to_aliases(df, y)                         # raw names -> aliases from config
    df["CYCLE"], df["DATASET"] = y, ds

Only cycles returned by imple.config.available_cycles(ds) are loaded, so a
file that is not set in the config (or not on disk) is skipped.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from imple.config import (available_cycles, cfg, check_columns, columns_for, raw_file,
                          usecols, INTL, PHL)


def read_sav(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read only `columns` from an SPSS file with pyreadstat (user-missing codes become NaN).
    Read the file's column list first (metadataonly=True), warn about check_columns() misses,
    and request only the columns that exist."""
    raise NotImplementedError


def merge_school(student: pd.DataFrame, school: pd.DataFrame) -> pd.DataFrame:
    """Left-join school variables onto students by keys COUNTRY + SCHOOL_ID."""
    raise NotImplementedError


def to_aliases(df: pd.DataFrame, cycle: int) -> pd.DataFrame:
    """Rename raw columns to aliases: columns_for(cycle, 'student') | columns_for(cycle, 'school'),
    plus keys (CNT -> COUNTRY, ...) and the final weight -> 'WEIGHT'."""
    raise NotImplementedError


def load_cycle(dataset: str, cycle: int) -> pd.DataFrame:
    """Load, merge and rename one cycle (flow in module docstring)."""
    raise NotImplementedError


def filter_countries(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    """scope 'global': countries.global_include ('all' or list) minus countries.global_exclude.
    scope 'philippines': COUNTRY == countries.philippines."""
    raise NotImplementedError


def load_all(dataset: str) -> pd.DataFrame:
    """Concatenate load_cycle over available_cycles(dataset)."""
    raise NotImplementedError
