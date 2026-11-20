"""1.2 Least squares (numpy / scipy) and the factor-matrix fit M X = Y.

Three methods, compared in the report on accuracy versus conditioning:
  normal - Cholesky on A^T A (scipy.linalg.cho_factor). Fast; SQUARES the condition number,
           so the error grows like kappa(A)^2 * eps. Needs m >= n and full column rank.
  qr     - Householder QR (numpy.linalg.qr) + triangular solve. Error ~ kappa(A) * eps. Default.
           For m < n: QR of A^T gives the minimum-norm solution.
  svd    - pseudoinverse (imple.svd.pinv). Handles rank deficiency; minimum-norm solution.
Reference: numpy.linalg.lstsq (LAPACK gelsd), used only to check the three above.

Weights (group populations): scale row i of A and b by sqrt(w_i), i.e. minimise sum_i w_i r_i^2.

Model fit: M X = Y  <=>  X^T M^T = Y^T, so M = lstsq(X^T, Y^T).x.T
(one factorisation of X^T serves every row of M: n independent least-squares
problems, one per output factor, sharing the same design matrix X^T).

Used by: pipeline.phase1/phase2, Phase 3 baselines.
Settings: config solver.lsq_default_method, solver.qr_rank_warning_rtol.
Check: `python -m imple.lsq`.
"""
from __future__ import annotations

import time
import warnings
from typing import NamedTuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

try:
    from .config import cfg
    from .svd import svd, numerical_rank, condition_number, pinv
except ImportError:  # python imple/lsq.py
    from config import cfg
    from svd import svd, numerical_rank, condition_number, pinv

METHODS = ("normal", "qr", "svd")


class LstsqResult(NamedTuple):
    x: np.ndarray            # solution (n,) or (n, k); for fit_factor_matrix this is M
    residual_norm: float     # ||A x - b|| (Frobenius for several right-hand sides), unweighted
    rank: int
    cond: float
    method: str


# --------------------------------------------------------------- solvers
def solve_normal(A, b):
    """x from (A^T A) x = A^T b via Cholesky. Requires m >= n and full column rank."""
    m, n = A.shape
    if m < n:
        raise ValueError("normal equations need m >= n (A^T A is singular otherwise); use 'qr' or 'svd'")
    c = cho_factor(A.T @ A)
    return cho_solve(c, A.T @ b)


def solve_qr(A, b):
    """m >= n: A = QR, x = R^{-1} Q^T b.  m < n: A^T = QR, x = Q R^{-T} b (minimum norm).
    Warns if min|R_ii| <= rtol * max|R_ii| (rank deficient: use 'svd')."""
    m, n = A.shape
    Q, R = np.linalg.qr(A if m >= n else A.T)
    d = np.abs(np.diag(R))
    rtol = cfg("solver.qr_rank_warning_rtol", 1e-10)
    if d.size and d.min() <= rtol * d.max():
        warnings.warn("QR: R is (nearly) singular, the system is rank deficient; prefer method='svd'")
    if m >= n:
        return solve_triangular(R, Q.T @ b, lower=False)
    return Q @ solve_triangular(R, b, trans="T", lower=False)


def solve_svd(A, b, rtol: float | None = None):
    """x = V_k S_k^{-1} U_k^T b with k = numerical rank (minimum-norm least squares)."""
    return pinv(A, rtol) @ b


SOLVERS = {"normal": solve_normal, "qr": solve_qr, "svd": solve_svd,
           "numpy": lambda A, b: np.linalg.lstsq(A, b, rcond=None)[0]}


def lstsq(A, b, method: str | None = None, weights=None) -> LstsqResult:
    """Apply sqrt(weights) to the rows, call the chosen solver, and report the residual on the
    unweighted system plus the rank and condition number of the weighted A."""
    method = method or cfg("solver.lsq_default_method", "qr")
    A, b = np.asarray(A, dtype=float), np.asarray(b, dtype=float)
    if weights is not None:
        sw = np.sqrt(np.asarray(weights, dtype=float))
        Aw, bw = A * sw[:, None], (b * sw if b.ndim == 1 else b * sw[:, None])
    else:
        Aw, bw = A, b
    x = SOLVERS[method](Aw, bw)
    s = svd(Aw).s
    return LstsqResult(x, float(np.linalg.norm(A @ x - b)), numerical_rank(s, Aw.shape),
                       condition_number(s), method)


def fit_factor_matrix(X, Y, method: str | None = None, weights=None) -> LstsqResult:
    """M (n_y x n_x) minimising sum_j w_j ||M x_j - y_j||^2. X: (n_x, N), Y: (n_y, N), weights: (N,)."""
    r = lstsq(np.asarray(X).T, np.asarray(Y).T, method, weights)
    return r._replace(x=r.x.T)


def relative_error(M, X, Y) -> np.ndarray:
    """Per-column ||M x_j - y_j|| / ||y_j|| (building block for deliverable 1.5)."""
    num = np.linalg.norm(M @ X - Y, axis=0)
    den = np.linalg.norm(Y, axis=0)
    return num / np.where(den > 0, den, np.nan)


# --------------------------------------------------------------- comparisons for the report
def compare_methods(X, Y, weights=None, methods=METHODS):
    """Fit M X = Y with every method; table of residual, distance to numpy's solution, time.
    Returns (pandas.DataFrame, {method: M})."""
    import pandas as pd

    ref = fit_factor_matrix(X, Y, "numpy", weights).x
    rows, Ms = [], {}
    for m in methods:
        t0 = time.perf_counter()
        try:
            r = fit_factor_matrix(X, Y, m, weights)
        except (ValueError, np.linalg.LinAlgError) as e:
            rows.append(dict(method=m, error=str(e)))
            continue
        Ms[m] = r.x
        rows.append(dict(method=m, residual_norm=r.residual_norm,
                         rel_diff_vs_numpy=np.linalg.norm(r.x - ref) / max(np.linalg.norm(ref), 1e-300),
                         rank=r.rank, cond_XT=r.cond, seconds=time.perf_counter() - t0))
    return pd.DataFrame(rows), Ms


def conditioning_experiment(kappas=None, m: int = 200, n: int = 19, seed: int | None = None):
    """Accuracy of each method against the condition number of A (zero-residual problems
    A x = b with known x). Shows the kappa^2 * eps error of the normal equations against
    kappa * eps for QR/SVD. Returns a long DataFrame: kappa, method, rel_error."""
    import pandas as pd

    rng = np.random.default_rng(cfg("solver.random_seed", 138) if seed is None else seed)
    kappas = np.logspace(0, 14, 15) if kappas is None else kappas
    U, _ = np.linalg.qr(rng.standard_normal((m, n)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    x_true = rng.standard_normal(n)
    rows = []
    for k in kappas:
        A = (U * np.logspace(0, -np.log10(k), n)) @ V.T
        b = A @ x_true
        for meth in METHODS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    x = SOLVERS[meth](A, b)
                    err = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
                except (ValueError, np.linalg.LinAlgError):
                    err = np.nan                       # Cholesky failed: A^T A numerically singular
            rows.append(dict(kappa=k, method=meth, rel_error=err))
    return pd.DataFrame(rows)


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

    ce = conditioning_experiment(kappas=[1e2, 1e10])
    e = ce.set_index(["kappa", "method"])["rel_error"]
    worse = np.isnan(e[(1e10, "normal")]) or e[(1e10, "normal")] > 100 * e[(1e10, "qr")]
    check("normal eq. lose ~kappa^2 at kappa=1e10", 0.0 if worse else 1.0, 0.5)
    return ok_all


if __name__ == "__main__":
    raise SystemExit(0 if _self_test() else 1)
