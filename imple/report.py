"""Figures for the Phase 1 and 2 sections of the notebook (one call per cell)."""
from __future__ import annotations

import numpy as np


def plot_conditioning(df):
    """Relative error of each least-squares method against kappa(A) (imple.lsq.conditioning_experiment),
    with the eps*kappa and eps*kappa^2 reference lines."""
    import matplotlib.pyplot as plt

    eps = np.finfo(float).eps
    fig, ax = plt.subplots(figsize=(7, 4.5))
    style = {"normal": ("#c0392b", "o"), "qr": ("#3b5b92", "s"), "svd": ("#27ae60", "^")}
    for m, part in df.groupby("method"):
        c, mk = style.get(m, ("grey", "x"))
        ax.loglog(part["kappa"], part["rel_error"], marker=mk, color=c, label=m)
    k = np.sort(df["kappa"].unique())
    ax.loglog(k, eps * k, "k:", lw=1, label=r"$\varepsilon\,\kappa$")
    ax.loglog(k, np.minimum(eps * k ** 2, 1), "k--", lw=1, label=r"$\varepsilon\,\kappa^2$")
    ax.set_xlabel(r"condition number $\kappa(A)$")
    ax.set_ylabel(r"relative error $\|\hat x - x\| / \|x\|$")
    ax.set_title("Least-squares accuracy against conditioning\n(missing normal-equation points: Cholesky failed)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_design_spectrum(data: dict):
    """Singular values of the global and PH design matrices X^T (normalised by s_max),
    with the numerical-rank cutoff. Small trailing values = factor combinations the data barely separates."""
    import matplotlib.pyplot as plt
    from .svd import summary

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, label, c in (("X_g", "global", "#3b5b92"), ("X_ph", "Philippines", "#c0392b")):
        X = data[key]
        if X.size == 0:
            continue
        sm = summary(X.T)
        s = sm["singular_values"]
        ax.semilogy(np.arange(1, len(s) + 1), s / s[0], "o-", color=c,
                    label=f"{label}: N={X.shape[1]}, rank={sm['numerical_rank']}, "
                          f"$\\kappa$={sm['condition_number']:.1e}")
    ax.axhline(max(data["X_g"].shape) * np.finfo(float).eps, color="grey", ls=":", lw=1, label="rank cutoff")
    ax.set_xlabel("index i")
    ax.set_ylabel(r"$s_i / s_1$")
    ax.set_title(r"Singular values of the design matrices $X^T$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
