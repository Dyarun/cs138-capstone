"""0.6 Representative vectors and the X, Y matrices for M X = Y.

Flow:
    reps = representative_vectors(df, masks)             # one row per (DATASET, CYCLE, COUNTRY, GROUP)
    X, Y, meta = build_xy(reps, transition_pairs("global"))        # Phase 1 input
    X_ph, Y_ph, meta_ph = build_xy(reps, transition_pairs("philippines"))  # Phase 2 input

Each column j of X / Y is one (pair, country, group): x_j from the pair's 'from'
cycle, y_j from its 'to' cycle. Rows follow imple.config.state_factors() order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from imple.config import cfg, state_factors


def representative_vectors(df: pd.DataFrame, masks: dict[str, pd.Series]) -> pd.DataFrame:
    """For every group mask and (DATASET, CYCLE, COUNTRY): weighted mean of each state alias
    (weighted_share for binaries is the same formula on 0/1), plus n, weight_sum and a spread
    column per alias (weighted SD / sqrt(n), for the CLT-based uncertainty)."""
    raise NotImplementedError


def build_xy(reps: pd.DataFrame, pairs: list[dict]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Stack columns over pairs x countries x groups.
    Keep a column only if both cycles exist, n >= countries.sample_threshold in both, and no NaN.
    meta: one row per column (pair id, COUNTRY, GROUP, weight) so results can be traced back."""
    raise NotImplementedError


def column_weights(meta: pd.DataFrame, scheme: str = "population") -> np.ndarray:
    """Per-column weights for the fits: 'population' (weight_sum), 'equal', or 'sqrt'."""
    raise NotImplementedError
