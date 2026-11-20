"""Tables and figures for the Phase 0 section of the notebook (one call per cell).

    from prep.report import factor_table, group_table, phase0_tables, plot_phase0
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from imple.config import INTL, PHL, factors, groups, state_factors


def factor_table(all_roles: bool = False) -> pd.DataFrame:
    """Deliverable 0.3 as a table (state factors only unless all_roles)."""
    rows = factors(None if all_roles else "state", include_only=not all_roles)
    t = pd.DataFrame(rows)
    t["encode"] = t["encode"].map(lambda e: e["type"])
    cols = ["alias", "label", "category", "role", "type", "source_alias", "encode", "higher_is",
            "available_in_ph", "include", "notes"]
    t = t[cols]
    t.index = range(1, len(t) + 1)
    return t


def group_table() -> pd.DataFrame:
    """Deliverable 0.5 as a table, with the rule written out."""
    def rule_text(rule):
        if not rule:
            return "everyone"
        return " AND ".join(f"{c['var']} {c['op']} {c['value']}" for c in rule)
    t = pd.DataFrame([dict(id=g["id"], label=g["label"], fit=g.get("fit", False), rule=rule_text(g["rule"]),
                           rationale=g["rationale"]) for g in groups()])
    t.index = range(1, len(t) + 1)
    return t


def phase0_tables(data: dict) -> dict[str, pd.DataFrame]:
    """The tables the Phase 0 section shows: verification failures, missingness of state
    factors, group sizes (fit cells) and X/Y coverage."""
    rep = data["report"]
    miss = (rep[rep["role"] == "state"]
            .pivot_table(index="alias", columns=["DATASET", "CYCLE"], values="missing_pct_w")
            .reindex([f["alias"] for f in state_factors()]))
    sizes = data["sizes"]
    fit_ids = [g["id"] for g in groups() if g.get("fit")]
    size_t = (sizes[sizes["GROUP"].isin(fit_ids)]
              .assign(scope=lambda s: np.where(s["DATASET"] == PHL, "PHL", "global (median)"))
              .groupby(["GROUP", "scope", "CYCLE"])["n"].median().unstack(["scope", "CYCLE"])
              .reindex(fit_ids))
    cov = data["coverage"]
    cov_t = cov.groupby(["pair", "reason"]).size().rename("cells").reset_index()
    cov_t["reason"] = cov_t["reason"].replace("", "kept")
    return dict(verification=data["log"], verification_failures=data["log"][~data["log"]["ok"]],
                missing_pct_weighted=miss.round(1), fit_cell_sizes=size_t, xy_coverage=cov_t)


def plot_phase0(data: dict, cycle: int | None = None):
    """Two panels: (a) PH vs global representative vectors of the ALL group (z-scores, so 0 =
    global basis-cycle average); (b) mathematics score by ESCS quartile, PH vs global median."""
    import matplotlib.pyplot as plt

    reps, state = data["reps"], data["state"]
    cycle = cycle or int(reps["CYCLE"].min())
    r = reps[reps["CYCLE"] == cycle]
    allg = r[r["GROUP"] == "ALL"]
    ph = allg[allg["DATASET"] == PHL][state].mean()
    ph_se = allg[allg["DATASET"] == PHL][[f"{a}_se" for a in state]].mean().to_numpy()
    glob = allg[allg["DATASET"] == INTL][state]
    lo, mid, hi = glob.quantile(0.25), glob.median(), glob.quantile(0.75)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw=dict(width_ratios=[1.6, 1]))
    y = np.arange(len(state))[::-1]
    ax[0].barh(y, hi - lo, left=lo, color="#c9d3e6", label="global countries, middle 50%")
    ax[0].scatter(mid, y, color="#3b5b92", s=18, zorder=3, label="global median country")
    ax[0].errorbar(ph, y, xerr=1.96 * ph_se, fmt="o", color="#c0392b", ms=5, zorder=4, label="Philippines (95% CI)")
    ax[0].axvline(0, color="grey", lw=0.6)
    ax[0].set_yticks(y, state)
    ax[0].set_xlabel("z-score (0 = global average in the basis cycle)")
    ax[0].set_title(f"(a) Representative vector, all students, {cycle}")
    ax[0].legend(loc="lower right", fontsize=8)

    qs = [f"ESCS_Q{i}" for i in range(1, 5)]
    for cyc, ls in zip(sorted(reps["CYCLE"].unique()), ("-", "--", ":")):
        rc = reps[(reps["CYCLE"] == cyc) & reps["GROUP"].isin(qs)]
        g = rc[rc["DATASET"] == INTL].groupby("GROUP")["MATH"].median().reindex(qs)
        p = rc[rc["DATASET"] == PHL].set_index("GROUP")["MATH"].reindex(qs)
        ax[1].plot(range(1, 5), g, ls, color="#3b5b92", marker="o", label=f"global median {cyc}")
        ax[1].plot(range(1, 5), p, ls, color="#c0392b", marker="o", label=f"Philippines {cyc}")
    ax[1].set_xticks(range(1, 5), ["Q1\n(lowest)", "Q2", "Q3", "Q4\n(highest)"])
    ax[1].set_xlabel("ESCS quartile within country")
    ax[1].set_ylabel("MATH (z-score)")
    ax[1].set_title("(b) Mathematics by socioeconomic quartile")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    return fig
