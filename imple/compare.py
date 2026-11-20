"""2.4 International vs Philippine factor matrices: comparison and analysis. SKELETON.

Inputs: M_G, M_PH = M_G + dM, the state names and categories (obj/factor_dict.json),
X_ph, Y_ph and the PH representative vectors.

Ideas, roughly in the order they answer "how is the Philippines different?"
---------------------------------------------------------------------------
1. Size of the perturbation
       rho_F = ||dM||_F / ||M_G||_F,   rho_2 = ||dM||_2 / ||M_G||_2
   Small rho (< ~0.1): the PH system is the global system plus a correction, so the
   first-order theory of 2.1 applies. Large rho: the PH system is structurally different
   (say so; the linearisation is then only indicative).

2. Entrywise differences
   Heatmap of dM (rows = output factor at t+1, columns = input factor at t), same colour
   scale as M_G, rows/columns grouped by category. Also the relative difference
   dM_ij / (|M_G,ij| + eps) masked where |M_G,ij| is tiny.
   Read column j as "how much more/less factor j pushes every outcome in the PH".

3. Category blocks (feeds the NLA hypothesis and Phase 4.4)
   Partition rows/columns by category (outcome, socioeconomic, regional, school, individual,
   attitudinal); table of block norms ||M[out, cat]||_F for M_G and M_PH and their ratio.
   The key row block is OUTCOME: compare the socioeconomic + regional columns against the
   individual columns. Hypothesis predicts the PH ratio is larger than the global one.

4. Largest differences
   Top-k entries of |dM| with factor names and sign: "SH_RURAL -> MATH: global -0.05, PH -0.20".
   Attach uncertainty later (Phase 3.4 bootstrap); for now flag entries supported by few PH columns.

5. Spectra
   Singular values of M_G and M_PH on one log plot; Weyl check |s_i(M_PH) - s_i(M_G)| <= ||dM||_2.
   Condition numbers of both. Eigenvalues of both on the complex plane (spectral radius > 1 means
   the one-cycle map amplifies; < 1 contracts towards a steady state) - a preview of Phase 4.1.

6. Effectiveness on PH data (same tables as 1.5)
   Per PH cell: err(M_G) vs err(M_PH). Report how much dM reduces the error, and whether the
   reduction is spread over all cells or comes from one or two (overfitting warning when
   N_PH < n: with few PH columns, M_PH can fit them almost exactly; compare with the
   leave-one-cell-out error before claiming improvement).

7. Narrative check
   Push the PH all-students x through both matrices: M_G x answers "where would PH students be
   next cycle if their system worked like the world average?", M_PH x "where the PH system takes
   them". The gap per factor is the most readable single result for the paper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def perturbation_size(M_G, dM) -> dict:
    """Idea 1: rho_F, rho_2, ||dM||_2."""
    raise NotImplementedError


def category_blocks(M, state: list[str]) -> pd.DataFrame:
    """Idea 3: Frobenius norm of each (output category, input category) block of M."""
    raise NotImplementedError


def top_differences(M_G, M_PH, state: list[str], k: int = 15) -> pd.DataFrame:
    """Idea 4: the k largest |dM_ij| with names, M_G and M_PH values."""
    raise NotImplementedError


def spectra(M_G, M_PH) -> pd.DataFrame:
    """Idea 5: singular values (and eigenvalues) of both, with the Weyl check."""
    raise NotImplementedError


def plot_difference_heatmap(M_G, M_PH, state: list[str]):
    """Idea 2: side-by-side heatmaps M_G, M_PH, dM with category separators."""
    raise NotImplementedError


def counterfactual(M_G, M_PH, x: np.ndarray, state: list[str]) -> pd.DataFrame:
    """Idea 7: M_G x vs M_PH x for one representative vector."""
    raise NotImplementedError
