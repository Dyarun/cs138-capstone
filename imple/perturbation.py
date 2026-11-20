"""2.2 Perturbation system solver: calibrate M_G into M_PH = M_G + dM. METHOD LIST + STUBS.

The derivation skeleton (deliverable 2.1) is in docs/perturbation.md and the notebook.

The system.  With R = Y_PH - M_G X_PH (what the global matrix fails to explain for PH),
the calibration solves
        dM X_PH = R              X_PH: n x N_PH,  dM: n x n,  R: n x N_PH
i.e. n least-squares problems  X_PH^T dm_i = r_i  (one per row i of dM) sharing one
design matrix. n = 19 and N_PH <= 12 cells per cycle pair, so the system is usually
UNDERDETERMINED (n^2 = 361 unknowns, n * N_PH equations): it has infinitely many exact
solutions and the solver choice is really a choice of which one to prefer.

Candidate solvers (pick one primary + one or two for comparison; justify in 2.1)
--------------------------------------------------------------------------------
 1. Minimum-norm (pseudoinverse)      dM = R X_PH^+                     imple.svd.pinv
    The smallest dM that explains the PH data exactly (first-order solution).
    Exact fit -> overfits sampling noise when N_PH <= n. Baseline / derivation (a).
 2. QR least squares                   X_PH^T dM^T = R^T                 imple.lsq (method 'qr')
    Only when N_PH > n and X_PH has full row rank (e.g. after adding 2025 pairs).
 3. Truncated SVD (TSVD)               keep the k largest s_i of X_PH    imple.svd
    Drops directions the PH data barely measures; k by the discrepancy principle
    (residual ~ CLT noise level) or leave-one-cell-out.
 4. Tikhonov / ridge toward M_G        dM = R X^T (X X^T + lam I)^-1     (Phase 1.3, optional)
    = min ||(M_G + dM) X - Y||^2 + lam ||dM||^2: shrinks M_PH to M_G where PH data is thin.
    Same as a Gaussian prior dM ~ N(0, tau^2) with lam = sigma^2 / tau^2 (Bayesian reading).
 5. Low-rank dM (reduced-rank)          rank(dM) <= r                    truncated SVD of solution 1
    "The PH differs from the world along r directions"; r = 1 or 2 is very interpretable.
 6. Structured dM (column subset)       dM nonzero only in chosen columns  imple.lsq on a smaller system
    e.g. only socioeconomic + regional columns may differ -> a direct test of the NLA hypothesis
    (compare the fit with and without the restriction). L1 / LASSO (ISTA) is the sparse variant.
 7. Multiplicative calibration          M_PH = D_r M_G D_c (diagonal D)   alternating least squares
    2n parameters instead of n^2: "each factor's effect is scaled up or down in the PH".
 8. Iterative with early stopping       LSQR / CGLS / Landweber          scipy.sparse.linalg.lsqr
    The number of iterations acts as the regulariser (semi-convergence).
 9. Full refit on PH only               M_PH = Y X_PH^+ (no M_G)          comparison only
    Shows why calibration is needed: with N_PH < n it ignores everything the world data says.

Checks every solver should report: residual ||M_PH X - Y||, size ||dM||_F / ||M_G||_F,
Weyl bound ||dM||_2, and leave-one-cell-out error (the honest accuracy when N_PH is small).

Uses: imple.svd, imple.lsq.  Used by: pipeline.phase2, Phase 3 stability tests.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

try:
    from .config import cfg
    from .svd import svd, pinv, condition_number
    from .tikhonov import fit_factor_matrix_tikhonov, choose_lambda
except ImportError:  # python imple/perturbation.py
    from config import cfg
    from svd import svd, pinv, condition_number
    from tikhonov import fit_factor_matrix_tikhonov, choose_lambda


class Calibration(NamedTuple):
    M_PH: np.ndarray
    dM: np.ndarray
    lam: float
    residual_norm: float        # ||M_PH X - Y||_F
    relative_size: float        # ||dM||_F / ||M_G||_F
    weyl_bound: float           # ||dM||_2, bound on every singular value shift


def residual(M_G, X_PH, Y_PH) -> np.ndarray:
    """R = Y_PH - M_G X_PH: what the global matrix fails to explain for PH."""
    raise NotImplementedError


def first_order_delta(M_G, X_PH, Y_PH) -> np.ndarray:
    """Unregularised minimum-norm dM = R X^+ (derivation (a)); uses imple.svd.pinv."""
    raise NotImplementedError


def solve_delta_m(M_G, X_PH, Y_PH, lam: float | None = None, weights=None) -> tuple[np.ndarray, float]:
    """Regularised dM (derivation (b)). If lam is None choose it with imple.tikhonov.choose_lambda
    on the residual system. Return (dM, lam)."""
    raise NotImplementedError


def weyl_bound(dM) -> float:
    """||dM||_2 = largest singular value of dM (use imple.svd)."""
    raise NotImplementedError


def calibrate(M_G, X_PH, Y_PH, lam: float | None = None, weights=None) -> Calibration:
    """Full Phase 2 step: dM, M_PH = M_G + dM, residual, relative size and Weyl bound."""
    raise NotImplementedError


# --------------------------------------------------------------------------- tests
def _self_test(verbose: bool = True) -> bool:
    """Plant a known dM and check it is recovered (over-determined) and that the
    regularised solution shrinks toward M_G when PH data is scarce (under-determined)."""
    rng = np.random.default_rng(cfg("solver.random_seed", 138))
    n = 8
    M_G = rng.standard_normal((n, n))
    dM_true = 0.1 * rng.standard_normal((n, n))
    X_many = rng.standard_normal((n, 100))
    X_few = rng.standard_normal((n, 3))
    results = {}
    dM = first_order_delta(M_G, X_many, (M_G + dM_true) @ X_many)
    results["recover dM, 100 cols"] = np.linalg.norm(dM - dM_true) < 1e-8
    small, _ = solve_delta_m(M_G, X_few, (M_G + dM_true) @ X_few, lam=1.0)
    results["few cols: ||dM|| < truth"] = np.linalg.norm(small) < np.linalg.norm(dM_true)
    cal = calibrate(M_G, X_many, (M_G + dM_true) @ X_many, lam=1e-10)
    s_shift = np.abs(np.linalg.svd(cal.M_PH, compute_uv=False) - np.linalg.svd(M_G, compute_uv=False))
    results["Weyl bound holds"] = np.all(s_shift <= cal.weyl_bound + 1e-12)
    for name, ok in results.items():
        if verbose:
            print(f"{'PASS' if ok else 'FAIL'}  {name}")
    return all(results.values())


if __name__ == "__main__":
    raise SystemExit(0 if _self_test() else 1)
