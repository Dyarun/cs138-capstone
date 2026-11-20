"""0.5 Demographic groups from obj/group_dict.json.

Flow:
    df = derive(df)                       # adds the 'derived' columns (e.g. ESCS_PCTL, LOCATION)
    masks = group_masks(df)               # {group id: boolean Series}
    sizes = group_sizes(df, masks)        # n students and weighted population per (DATASET, CYCLE, COUNTRY)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from imple.config import cfg, resolve, load_obj, groups


def weighted_percentile_rank(values: pd.Series, weights: pd.Series) -> pd.Series:
    """Rank in [0, 1]: cumulative weight share of students at or below each value (NaN stays NaN)."""
    raise NotImplementedError


def derive(df: pd.DataFrame) -> pd.DataFrame:
    """Add each entry of group_dict['derived'] that is not 'enabled': false.
       weighted_percentile_rank -> per groupby(entry['within']) (aliases COUNTRY, CYCLE)
       map_codes                -> label for each code list in entry['map'] (resolve references)
    """
    raise NotImplementedError


def condition_mask(df: pd.DataFrame, cond: dict) -> pd.Series:
    """One {var, op, value} condition -> boolean Series. ops: in, ==, !=, between (inclusive)."""
    raise NotImplementedError


def group_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """AND of condition_mask over each group's rule; empty rule = everyone."""
    raise NotImplementedError


def group_sizes(df: pd.DataFrame, masks: dict[str, pd.Series]) -> pd.DataFrame:
    """Long table: DATASET, CYCLE, COUNTRY, GROUP, n, weight_sum. Used to fill size_intl / size_ph
    and to drop cells below countries.sample_threshold."""
    raise NotImplementedError
