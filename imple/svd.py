"""1.1 Singular Value Decomposition (numpy) and the quantities built on it.

The decomposition itself is numpy.linalg.svd (LAPACK gesdd: Householder
bidiagonalisation followed by divide-and-conquer). This module adds what the
project needs on top of it, with one rank rule used everywhere:

    A = U diag(s) V^T                      thin SVD, s descending
    numerical rank  = #{ s_i > rtol * s_max },  rtol = max(m, n) * eps  (numpy's default)
    condition number kappa_2(A) = s_max / s_min  (inf if A is rank deficient)
    pseudoinverse    A^+ = V_k diag(1/s_1..1/s_k) U_k^T,  k = numerical rank
    best rank-k      A_k = U_k diag(s_1..s_k) V_k^T,  ||A - A_k||_2 = s_{k+1}  (Eckart-Young)

Used by: imple.lsq.solve_svd, imple.perturbation, Phase 3 conditioning, Phase 4.3 truncation.
Settings: config solver.rank_rtol.
Check: `python -m imple.svd`.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

try:
    from .config import cfg
except ImportError:  # python imple/svd.py
    from config import cfg


class SVDResult(NamedTuple):
    U: np.ndarray        # m x k, orthonormal columns (k = min(m, n))
    s: np.ndarray        # k singular values, descending
    Vt: np.ndarray       # k x n, orthonormal rows
    sweeps: int          # kept for interface compatibility (0: LAPACK does not report sweeps)
    converged: bool      # True unless LAPACK raised LinAlgError


def svd(A, tol: float | None = None, max_sweeps: int | None = None) -> SVDResult:
    """Thin SVD A = U diag(s) Vt (numpy.linalg.svd, full_matrices=False).
    tol and max_sweeps are accepted for interface compatibility and ignored."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    try:
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        return SVDResult(U, s, Vt, 0, True)
    except np.linalg.LinAlgError:
        k = min(A.shape)
        return SVDResult(np.full((A.shape[0], k), np.nan), np.full(k, np.nan),
                         np.full((k, A.shape[1]), np.nan), 0, False)


def _rtol(shape, rtol: float | None) -> float:
    if rtol is None:
        rtol = cfg("solver.rank_rtol")
    if rtol is None:
        rtol = (max(shape) if shape is not None else 1) * np.finfo(float).eps
    return rtol


def numerical_rank(s, shape=None, rtol: float | None = None) -> int:
    """Count of s_i > rtol * s_max (rtol default: config solver.rank_rtol, else max(m, n) * eps)."""
    s = np.asarray(s, dtype=float)
    if s.size == 0 or s.max() == 0:
        return 0
    return int(np.sum(s > _rtol(shape, rtol) * s.max()))


def condition_number(s) -> float:
    """s_max / s_min, inf if s_min == 0."""
    s = np.asarray(s, dtype=float)
    if s.size == 0 or s.min() == 0:
        return float("inf")
    return float(s.max() / s.min())


def pinv(A, rtol: float | None = None) -> np.ndarray:
    """Moore-Penrose pseudoinverse V S^+ U^T, dropping s_i below the rank cutoff."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    r = svd(A)
    k = numerical_rank(r.s, A.shape, rtol)
    return (r.Vt[:k].T / r.s[:k]) @ r.U[:, :k].T


def truncated(A, k: int) -> np.ndarray:
    """Best rank-k approximation U_k S_k V_k^T (Phase 4.3)."""
    r = svd(A)
    return (r.U[:, :k] * r.s[:k]) @ r.Vt[:k]


def summary(A) -> dict:
    """{'shape', 'singular_values', 'condition_number', 'numerical_rank', 'sweeps', 'converged'} for reports."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    r = svd(A)
    return dict(shape=A.shape, singular_values=r.s, condition_number=condition_number(r.s),
                numerical_rank=numerical_rank(r.s, A.shape), sweeps=r.sweeps, converged=r.converged)


# --------------------------------------------------------------------------- tests
def _self_test(verbose: bool = True) -> bool:
    """Properties on tall, wide, square, rank-deficient, Hilbert, 1x1 and zero matrices:
    reconstruction, orthonormality, rank, the four Penrose conditions, and Eckart-Young.
    The tolerance grows with the condition number of the kept singular values (Hilbert 8x8:
    kappa ~ 1.5e10, so pinv is only accurate to ~kappa * eps ~ 1e-6, which is expected)."""
    rng = np.random.default_rng(cfg("solver.random_seed", 138))
    cases = {
        "tall 30x20": rng.standard_normal((30, 20)),
        "wide 6x15": rng.standard_normal((6, 15)),
        "square 20x20": rng.standard_normal((20, 20)),
        "rank-3 12x7": rng.standard_normal((12, 3)) @ rng.standard_normal((3, 7)),
        "Hilbert 8x8": 1.0 / (np.arange(8)[:, None] + np.arange(8)[None, :] + 1.0),
        "1x1": np.array([[3.0]]),
        "zero 4x3": np.zeros((4, 3)),
    }
    ok_all = True
    for name, A in cases.items():
        r = svd(A)
        k = min(A.shape)
        P = pinv(A)
        nA = max(1.0, np.linalg.norm(A))
        errs = [
            np.linalg.norm(A - (r.U * r.s) @ r.Vt) / nA,                     # reconstruction
            np.linalg.norm(r.U.T @ r.U - np.eye(k)) if r.s.max() > 0 else 0,  # U orthonormal
            np.linalg.norm(r.Vt @ r.Vt.T - np.eye(k)),                        # V orthonormal
            np.linalg.norm(A @ P @ A - A) / nA,                               # Penrose 1
            np.linalg.norm(P @ A @ P - P) / max(1.0, np.linalg.norm(P)),      # Penrose 2
            np.linalg.norm((A @ P).T - A @ P),                                # Penrose 3
            np.linalg.norm((P @ A).T - P @ A),                                # Penrose 4
        ]
        rk = numerical_rank(r.s, A.shape)
        kappa_eff = r.s[0] / r.s[rk - 1] if rk > 0 else 1.0       # condition of the kept part
        limit = max(1e-8, 1e-14 * kappa_eff)                       # backward-stable: err ~ kappa * eps
        ok = max(errs) < limit and numerical_rank(r.s, A.shape) == np.linalg.matrix_rank(A)
        if k > 1 and r.s.max() > 0:                                           # Eckart-Young, k = 1
            ok &= abs(np.linalg.norm(A - truncated(A, 1), 2) - r.s[1]) < 1e-10 * max(1, r.s[0])
        ok_all &= ok
        if verbose:
            print(f"{'PASS' if ok else 'FAIL'}  {name:<13} max_err={max(errs):.1e}  "
                  f"rank={numerical_rank(r.s, A.shape)}  cond={condition_number(r.s):.2e}")
    return ok_all


if __name__ == "__main__":
    raise SystemExit(0 if _self_test() else 1)
