"""2.2 Perturbation solver: calibrate M_G into M_PH = M_G + dM.

Derivation (deliverable 2.1, written in the notebook):
  (a) First order. (M + dM)(x + dx) = y + dy, drop dM dx:
          dM x = dy - M dx = r
      Minimum-norm solution for one pair: dM = r x^T / ||x||^2.
      For many PH columns: dM X = R with R = Y_PH - M_G X_PH, min-norm dM = R X^+.
  (b) Regularised. min ||(M_G + dM) X - Y||^2 + lam ||dM||^2
      = Tikhonov on the residual R with prior 0
      = Tikhonov on Y with prior M_G (same answer, imple.tikhonov).
      This is what keeps the fit stable when PH has few group columns (N_PH < n).
  (c) Bounds. Weyl: |sigma_i(M_PH) - sigma_i(M_G)| <= ||dM||_2.
      Sensitivity of the solve to noise in X: relative error <~ kappa(X) * relative noise.

Uses: imple.svd, imple.tikhonov.  Used by: pipeline.phase2, Phase 3 stability tests.
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
