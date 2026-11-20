"""0.1 - 0.4 Clean, encode and scale student-level factors.

Order (same for intl and phl, so M_G and M_PH share units):
    df = drop_missing_codes(df)                    # only if read with user_missing=True
    df = encode(df, factors())                     # factor_dict 'encode' rules -> one column per alias
    report = missing_report(df, factors())         # missing %, n valid -> fills factor_dict
    scaler = fit_scaler(global_basis_df, state_factors())   # basis = config cycles.basis, global scope
    df = apply_scaler(df, scaler)                  # reuse the SAME scaler for every cycle and for PH
"""
from __future__ import annotations

import pandas as pd

from imple.config import cfg, resolve, factors, state_factors


def drop_missing_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Set every value listed in config missing_codes to NaN (needed only with user_missing=True)."""
    raise NotImplementedError


def pv_mean(df: pd.DataFrame, source_alias: str, divide_by: float) -> pd.Series:
    """Mean of the plausible-value columns <source_alias>_1..n, divided by preprocessing.pv_scale_divisor.
    Return all-NaN if the cycle has no PVs for this domain (e.g. PROB in some cycles)."""
    raise NotImplementedError


def encode(df: pd.DataFrame, factor_list: list[dict]) -> pd.DataFrame:
    """Add one column named f['alias'] per factor according to f['encode']['type']:
       as_is    -> copy (optional clip_quantiles)
       pv_mean  -> pv_mean(...)
       binary   -> isin(resolve(true_codes)) as 0/1, NaN kept NaN
       category -> label from resolve(codes)
    """
    raise NotImplementedError


def missing_report(df: pd.DataFrame, factor_list: list[dict]) -> pd.DataFrame:
    """Per alias, per (DATASET, CYCLE): missing %, n valid. Flag > preprocessing.max_missing_rate_per_factor."""
    raise NotImplementedError


def weighted_mean_sd(x: pd.Series, w: pd.Series) -> tuple[float, float]:
    """Weighted mean and standard deviation, ignoring NaN."""
    raise NotImplementedError


def fit_scaler(df_basis: pd.DataFrame, factor_list: list[dict]) -> dict[str, tuple[float, float]]:
    """{alias: (mean, sd)} from the global sample of the basis cycle, weighted by WEIGHT."""
    raise NotImplementedError


def apply_scaler(df: pd.DataFrame, scaler: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """z = (value - mean) / sd for every alias in the scaler."""
    raise NotImplementedError
