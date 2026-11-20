"""1.2 Least squares (own implementations) and the factor-matrix fit M X = Y.

Three methods, compared in the report on accuracy versus conditioning:
  normal - Cholesky on A^T A. Fast; squares the condition number.
  qr     - Householder QR. Error ~ kappa(A). Default. For m < n use QR of A^T (min-norm).
  svd    - pseudoinverse from imple.svd. Robust to rank deficiency; min-norm solution.

Weights (survey weights / group populations): scale row i of A and b by sqrt(w_i).

Model fit: M X = Y  <=>  X^T M^T = Y^T, so M = lstsq(X^T, Y^T).x.T
(one factorisation of X^T serves every row of M).

Used by: imple.tikhonov (as the lambda = 0 case), pipeline.phase1, Phase 3 baselines.
Settings: config solver.lsq_default_method, solver.qr_rank_warning_rtol.
Check: `python -m imple.lsq` (compares against numpy.linalg.lstsq).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

try:
    from .config import cfg
    from .svd import svd, numerical_rank, condition_number
except ImportError:  # python imple/lsq.py
    from config import cfg
    from svd import svd, numerical_rank, condition_number

METHODS = ("normal", "qr", "svd")


class LstsqResult(NamedTuple):
    x: np.ndarray            # solution (n,) or (n, k); for fit_factor_matrix this is M
    residual_norm: float     # ||A x - b|| (Frobenius for several right-hand sides), unweighted
    rank: int
    cond: float
    method: str


# --------------------------------------------------------------- building blocks
def forward_sub(L, b):
    """Solve L y = b, L lower triangular (b may be a matrix)."""
    raise NotImplementedError


def back_sub(R, b):
    """Solve R x = b, R upper triangular (b may be a matrix)."""
    raise NotImplementedError


def cholesky(S):
    """L with S = L L^T. Raise np.linalg.LinAlgError if a pivot <= 0 (A^T A singular)."""
    raise NotImplementedError


def householder_qr(A):
    """Householder QR of A (m >= n). Return (vs, R): list of n reflector vectors, n x n upper R.

    Do not form Q; apply it with apply_qt / apply_q.
    Reflector k: v = x + sign(x_0)||x|| e_1 on A[k:, k], normalised; A[k:, k:] -= 2 v (v^T A[k:, k:]).
    """
    raise NotImplementedError


def apply_qt(vs, b):
    """Q^T b from stored reflectors (apply them in order k = 0..n-1)."""
    raise NotImplementedError


def apply_q(vs, y, m: int):
    """Q y (pad y to m rows, apply reflectors in reverse order)."""
    raise NotImplementedError


# --------------------------------------------------------------- solvers
def solve_normal(A, b):
    """x from (A^T A) x = A^T b via cholesky + forward_sub + back_sub. Require m >= n."""
    raise NotImplementedError


def solve_qr(A, b):
    """m >= n: x = R^{-1} (Q^T b)[:n].  m < n: A^T = QR, x = Q (R^{-T} b) (min-norm).
    Warn if min|R_ii| <= rtol * max|R_ii| (rank-deficient)."""
    raise NotImplementedError


def solve_svd(A, b, rtol: float | None = None):
    """x = V_k S_k^{-1} U_k^T b with k = numerical rank."""
    raise NotImplementedError


def lstsq(A, b, method: str | None = None, weights=None) -> LstsqResult:
    """Dispatcher: apply sqrt(weights) to rows, call the chosen solver, fill LstsqResult
    (residual on the unweighted system; rank and cond from svd of the weighted A)."""
    method = method or cfg("solver.lsq_default_method", "qr")
    raise NotImplementedError


def fit_factor_matrix(X, Y, method: str | None = None, weights=None) -> LstsqResult:
    """M (n_y x n_x) minimising ||M X - Y||_F. X: (n_x, N), Y: (n_y, N), weights: (N,).
    Implement as lstsq(X.T, Y.T, ...) and transpose the solution."""
    raise NotImplementedError


def relative_error(M, X, Y) -> np.ndarray:
    """Per-column ||M x_j - y_j|| / ||y_j|| (deliverable 1.5 effectiveness)."""
    raise NotImplementedError


# --------------------------------------------------------------------------- tests
def _self_test(verbose: bool = True) -> bool:
    rng = np.random.default_rng(cfg("solver.random_seed", 138))
    ok_all = True

    def check(name, err, limit):
        nonlocal ok_all
        ok = err < limit
        ok_all &= ok
        if verbose:
            print(f"{'PASS' if ok else 'FAIL'}  {name:<36} rel_err={err:.1e}")

    def rel(a, b):
        return np.linalg.norm(a - b) / np.linalg.norm(b)

    A, b = rng.standard_normal((50, 8)), rng.standard_normal(50)
    ref = np.linalg.lstsq(A, b, rcond=None)[0]
    for m in METHODS:
        check(f"overdetermined 50x8 [{m}]", rel(lstsq(A, b, m).x, ref), 1e-10)

    A, b = rng.standard_normal((5, 12)), rng.standard_normal(5)
    ref = np.linalg.lstsq(A, b, rcond=None)[0]
    for m in ("qr", "svd"):
        check(f"underdetermined 5x12 [{m}]", rel(lstsq(A, b, m).x, ref), 1e-10)

    A = rng.standard_normal((30, 4)) @ rng.standard_normal((4, 10))
    b = rng.standard_normal(30)
    check("rank-4 30x10 [svd]", rel(lstsq(A, b, "svd").x, np.linalg.lstsq(A, b, rcond=None)[0]), 1e-8)

    A, b, w = rng.standard_normal((40, 5)), rng.standard_normal(40), rng.uniform(0.5, 3.0, 40)
    ref = np.linalg.lstsq(A * np.sqrt(w)[:, None], b * np.sqrt(w), rcond=None)[0]
    check("weighted 40x5 [qr]", rel(lstsq(A, b, "qr", weights=w).x, ref), 1e-10)

    M_true = rng.standard_normal((20, 20)) / np.sqrt(20)
    X = rng.standard_normal((20, 300))
    for m in METHODS:
        check(f"recover M 20x20 [{m}]", rel(fit_factor_matrix(X, M_true @ X, m).x, M_true), 1e-10)
    return ok_all


if __name__ == "__main__":
    raise SystemExit(0 if _self_test() else 1)
