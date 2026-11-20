"""1.3 Tikhonov regularisation built on our SVD.

Problem:  min_x ||A x - b||^2 + lam ||x - x0||^2        (x0 = prior, default 0)
Solution via A = U S V^T and r = b - A x0:
          x = x0 + V diag( s_i / (s_i^2 + lam) ) U^T r
lam = 0 reduces to the SVD least squares solution (imple.lsq.solve_svd).

Matrix form used by the model (one row of M per column of b):
          min_M ||M X - Y||_F^2 + lam ||M - M0||_F^2
          -> A = X^T, b = Y^T, x0 = M0^T, then transpose back.
Phase 2 calls this with M0 = M_G (shrink M_PH toward the global matrix).

Choosing lam (pick one, report it in the decisions note):
  GCV(lam)  = N ||A x_lam - b||^2 / (N - sum_i f_i)^2,  f_i = s_i^2 / (s_i^2 + lam)
  L-curve   = plot log||A x_lam - b|| against log||x_lam - x0||, take the corner
  CV        = leave-one-country-out error (matches transition_pairs validation)

Uses: imple.svd.svd.  Used by: imple.perturbation, pipeline.phase1/phase2.
Config lines to add (optional, defaults below): solver.tikhonov.lambda_grid, solver.tikhonov.method.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

try:
    from .config import cfg
    from .svd import svd
except ImportError:  # python imple/tikhonov.py
    from config import cfg
    from svd import svd


class TikhonovResult(NamedTuple):
    x: np.ndarray            # solution; for fit_factor_matrix_tikhonov this is M
    lam: float
    residual_norm: float     # ||A x - b|| (Frobenius for matrices)
    deviation_norm: float    # ||x - x0||
    effective_dof: float     # sum_i s_i^2 / (s_i^2 + lam)


def lambda_grid() -> np.ndarray:
    """Candidate lambdas: config solver.tikhonov.lambda_grid = [log10_min, log10_max, count], default 1e-6..1e2."""
    lo, hi, k = cfg("solver.tikhonov.lambda_grid", [-6, 2, 50])
    return np.logspace(lo, hi, int(k))


def tikhonov(A, b, lam: float, x0=None, weights=None) -> TikhonovResult:
    """Solve the regularised problem above. b may have several columns; weights scale rows by sqrt(w)."""
    # TODO 1: apply sqrt(weights) to rows of A and b
    # TODO 2: r = b - A x0 ; svd(A) ; filter factors s/(s^2 + lam)
    # TODO 3: x = x0 + V (filtered U^T r) ; fill the result tuple
    raise NotImplementedError


def fit_factor_matrix_tikhonov(X, Y, lam: float, M0=None, weights=None) -> TikhonovResult:
    """M minimising ||M X - Y||^2 + lam ||M - M0||^2. X: (n_x, N), Y: (n_y, N), M0: (n_y, n_x)."""
    # TODO: tikhonov(X.T, Y.T, lam, x0=M0.T, weights) and transpose x
    raise NotImplementedError


def gcv_score(A, b, lam: float, x0=None) -> float:
    """Generalised cross-validation score for one lambda (formula in module docstring)."""
    raise NotImplementedError


def l_curve(A, b, lams, x0=None) -> tuple[np.ndarray, np.ndarray]:
    """(residual norms, deviation norms) over lams, for plotting and corner detection."""
    raise NotImplementedError


def choose_lambda(A, b, x0=None, method: str | None = None, lams=None) -> float:
    """Best lambda over lambda_grid() by 'gcv' (default config solver.tikhonov.method) or 'lcurve'."""
    method = method or cfg("solver.tikhonov.method", "gcv")
    raise NotImplementedError


# --------------------------------------------------------------------------- tests
def _self_test(verbose: bool = True) -> bool:
    """lam = 0 matches least squares; lam -> large returns x0; matrix form recovers a known M."""
    rng = np.random.default_rng(cfg("solver.random_seed", 138))
    A, b = rng.standard_normal((40, 6)), rng.standard_normal(40)
    x0 = rng.standard_normal(6)
    checks = {
        "lam=0 equals lstsq": np.linalg.norm(tikhonov(A, b, 0.0).x - np.linalg.lstsq(A, b, rcond=None)[0]),
        "lam=1e12 returns x0": np.linalg.norm(tikhonov(A, b, 1e12, x0=x0).x - x0),
    }
    ref = np.linalg.solve(A.T @ A + 0.5 * np.eye(6), A.T @ b + 0.5 * x0)   # normal-equation form
    checks["lam=0.5 vs closed form"] = np.linalg.norm(tikhonov(A, b, 0.5, x0=x0).x - ref)
    M = rng.standard_normal((5, 5))
    X = rng.standard_normal((5, 200))
    checks["matrix form, M0 = M"] = np.linalg.norm(fit_factor_matrix_tikhonov(X, M @ X, 3.0, M0=M).x - M)
    ok_all = True
    for name, err in checks.items():
        ok = err < 1e-8
        ok_all &= ok
        if verbose:
            print(f"{'PASS' if ok else 'FAIL'}  {name:<26} err={err:.1e}")
    return ok_all


if __name__ == "__main__":
    raise SystemExit(0 if _self_test() else 1)
