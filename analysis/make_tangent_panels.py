#!/usr/bin/env python
"""
Tangent-space diagnostics as four separate single-column figures (IEEE column width 3.5 in).

    python make_tangent_panels.py <dir with tangent_*.csv and svals/> --outdir .

Writes
  fig_tangent_alignment.pdf  alignment A after training vs the 2-design level
  fig_tangent_gradient.pdf   squared gradient norms, circuit vs loss
  fig_tangent_spectrum.pdf   singular-value spectrum of the tangent vectors
  fig_tangent_rank.pdf       metric rank vs parameter count
Same data and statistics as make_tangent_figure.py; only the layout and the plot style change
(see ieee_style.py). No titles: each figure's caption carries the title.
"""

import argparse
import os
import sys
import importlib.util

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ieee_style as st                                                    # noqa: E402

_spec = importlib.util.spec_from_file_location("mtf", os.path.join(HERE, "make_tangent_figure.py"))
mtf = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(mtf)
ORDER, f = mtf.ORDER, mtf.f

NAME = {"overhead_adaptive": "Overhead-MNIST", "sat6": "SAT-6", "so2sat_adaptive": "So2Sat"}
OFF = {"rho01": -0.16, "rho1": 0.16}
COL = {"rho01": st.BLUE, "rho1": st.RED}
MK = {"rho01": "o", "rho1": "s"}
RHO = {"rho01": r"$\rho_q=0.1$", "rho1": r"$\rho_q=1$"}
W, H = st.COL_W, 2.15


def cfg_label(rows, cfg):
    """Tick label 'dataset / gate, n_q' with the gate and register read from the run records."""
    rs = rows.get((cfg, "rho01"), [])
    if not rs:
        return NAME[cfg]
    return f"{NAME[cfg]}\n{rs[0]['gate']}, $n_q={int(float(rs[0]['n_qubits']))}$"


def groups(ax, rows):
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([cfg_label(rows, c) for c in ORDER])
    ax.set_xlim(-0.5, len(ORDER) - 0.5)
    ax.tick_params(axis="x", which="both", length=0)


def center_spread(ys, log):
    """Mean with +-SD (linear axis) or median with the 5-95% range (log axis)."""
    ys = np.asarray(ys, float); ys = ys[np.isfinite(ys)]
    if not ys.size:
        return None
    if log:
        c = np.median(ys); lo, hi = np.percentile(ys, [5, 95])
    else:
        c = ys.mean(); lo, hi = c - ys.std(ddof=1), c + ys.std(ddof=1)
    return c, lo, hi


def point(ax, x, ys, color, marker, log=False, filled=True, ms=4.5):
    cs = center_spread(ys, log)
    if cs is None:
        return
    c, lo, hi = cs
    ax.errorbar(x, c, yerr=[[c - lo], [hi - c]], fmt=marker, color=color, markersize=ms,
                markerfacecolor=color if filled else "white", markeredgecolor=color, markeredgewidth=0.8,
                elinewidth=0.8, capsize=2.5, capthick=0.8, zorder=4)


def handle(color, marker, label, filled=True):
    return Line2D([], [], marker=marker, color=color, linestyle="", markersize=4.5,
                  markerfacecolor=color if filled else "white", markeredgewidth=0.8, label=label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rows = mtf.load(a.dir)
    svals = mtf.load_svals(a.dir)
    st.apply()

    # ---------------------------------------------------------------- alignment
    fig, ax = plt.subplots(figsize=(W, H))
    for i, cfg in enumerate(ORDER):
        for s in ("rho01", "rho1"):
            rs = rows.get((cfg, s), [])
            if rs:
                point(ax, i + OFF[s], [f(r, "A_final") for r in rs], COL[s], MK[s])
        rs = rows.get((cfg, "rho01"), [])
        if rs:
            deff = np.mean([f(r, "deff_final") for r in rs]); D = int(float(rs[0]["hilbert_dim"]))
            ax.hlines(deff / (2 * (D - 1)), i - 0.34, i + 0.34, color=st.BLACK, linestyle=(0, (4, 2)), linewidth=1.0)
    groups(ax, rows)
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel(r"Alignment $A$")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    hd = [handle(COL[s], MK[s], RHO[s]) for s in ("rho01", "rho1")]
    hd.append(Line2D([], [], color=st.BLACK, linestyle=(0, (4, 2)), linewidth=1.0, label="2-design level"))
    st.legend(ax, handles=hd, loc="center", bbox_to_anchor=(0.5, 0.46), ncol=3, columnspacing=1.2, handlelength=1.8)
    st.save(fig, a.outdir, "fig_tangent_alignment")

    # ---------------------------------------------------------------- gradient norms
    fig, ax = plt.subplots(figsize=(W, H))
    for i, cfg in enumerate(ORDER):
        for s in ("rho01", "rho1"):
            rs = rows.get((cfg, s), [])
            if rs:
                point(ax, i + OFF[s], [f(r, "grad2_final") for r in rs], COL[s], MK[s], log=True, filled=False)
                point(ax, i + OFF[s], [f(r, "task_grad2_final") for r in rs], COL[s], MK[s], log=True)
    groups(ax, rows)
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1e1)
    st.log_minor(ax.yaxis)
    ax.set_ylabel("Squared gradient norm")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    hd = [handle(COL["rho01"], MK["rho01"], r"$\|\nabla C\|^2$, " + RHO["rho01"], filled=False),
          handle(COL["rho01"], MK["rho01"], r"$\|\partial L/\partial\theta_q\|^2$, " + RHO["rho01"]),
          handle(COL["rho1"], MK["rho1"], r"$\|\nabla C\|^2$, " + RHO["rho1"], filled=False),
          handle(COL["rho1"], MK["rho1"], r"$\|\partial L/\partial\theta_q\|^2$, " + RHO["rho1"])]
    st.legend(ax, handles=hd, loc="lower center", ncol=2, columnspacing=1.2)
    st.save(fig, a.outdir, "fig_tangent_gradient")

    # ---------------------------------------------------------------- spectrum
    fig, ax = plt.subplots(figsize=(W, H))
    spec_c = {"overhead_adaptive": st.BLACK, "sat6": st.BLUE, "so2sat_adaptive": st.RED}
    spec_ls = {"overhead_adaptive": "-", "sat6": (0, (5, 1.5)), "so2sat_adaptive": (0, (4, 1.5, 1, 1.5))}
    floor = 1e-4
    ranks = []
    for cfg in ORDER:
        if cfg not in svals:
            continue
        sp = svals[cfg]
        med = np.median(sp, axis=0); lo = np.percentile(sp, 5, axis=0); hi = np.percentile(sp, 95, axis=0)
        rank = int(np.sum(med > 1e-8 * med[0])); ranks.append(rank)
        idx = np.arange(1, rank + 1)
        ax.fill_between(idx, np.clip(lo[:rank], floor, None), hi[:rank], color=spec_c[cfg], alpha=0.15, linewidth=0)
        ax.plot(idx, med[:rank], color=spec_c[cfg], linestyle=spec_ls[cfg], linewidth=1.1, label=NAME[cfg], zorder=3)
        ax.vlines(rank, floor, med[rank - 1], color=spec_c[cfg], linestyle=(0, (1, 1.5)), linewidth=0.8, zorder=2)
    L_max = max(sp.shape[1] for sp in svals.values())
    ax.set_yscale("log")
    ax.set_ylim(floor, 3)
    st.log_minor(ax.yaxis)
    ax.set_xlim(0, L_max + 2)
    ax.set_xticks(sorted(set([0, 25, 50] + ranks)))
    ax.set_xlabel("Index of tangent direction (sorted)")
    ax.set_ylabel("Singular value")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    st.legend(ax, loc="lower left", handlelength=2.4)
    st.save(fig, a.outdir, "fig_tangent_spectrum")

    # ---------------------------------------------------------------- rank vs parameter count
    # One horizontal bar per configuration, split into independent, merged and unused angles.
    fig, ax = plt.subplots(figsize=(W, 1.6))
    fill = {"live": "#bdbdbd", "merged": "white", "dead": "white"}
    hatch = {"live": None, "merged": "/////", "dead": "xxxxx"}
    box = dict(boxstyle="square,pad=0.12", facecolor="white", edgecolor="none")
    Lmax = 0
    for i, cfg in enumerate(ORDER):
        rs = rows.get((cfg, "rho01"), [])
        if not rs:
            continue
        L = int(float(rs[0]["n_params"])); nq = rs[0]["n_qubits"]; gate = rs[0]["gate"]
        _, pred, dead, rym, xxm = mtf._an.deff_predicted(nq, rs[0].get("n_reup", 2), gate)
        deff = int(round(np.mean([f(r, "deff_final") for r in rs])))
        assert deff == pred, f"{cfg}: measured rank {deff} differs from the gate-algebra value {pred}"
        merged = rym + xxm
        assert deff + merged + dead == L, f"{cfg}: segments do not add up to L"
        Lmax = max(Lmax, L)
        h = 0.56
        left = 0
        for key, val in (("live", deff), ("merged", merged), ("dead", dead)):
            if not val:
                continue
            ax.barh(i, val, height=h, left=left, facecolor=fill[key], hatch=hatch[key], edgecolor=st.BLACK,
                    linewidth=st.LW, zorder=3)
            ax.text(left + val / 2, i, f"{val}", ha="center", va="center", fontsize=7.5, zorder=4,
                    bbox=None if key == "live" else box)
            left += val
        ax.text(L + 2.5, i, f"$L={L}$", ha="left", va="center", fontsize=7.5)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([cfg_label(rows, c) for c in ORDER], linespacing=1.15)
    ax.set_ylim(len(ORDER) - 0.5, -0.5)
    ax.tick_params(axis="y", which="both", length=0)
    ax.set_xlim(0, Lmax + 26)
    ax.set_xlabel("Number of circuit parameters")
    hd = [Patch(facecolor=fill["live"], edgecolor=st.BLACK, linewidth=st.LW, label=r"Independent ($d_{\mathrm{eff}}$)"),
          Patch(facecolor="white", hatch=hatch["merged"], edgecolor=st.BLACK, linewidth=st.LW, label="Merged"),
          Patch(facecolor="white", hatch=hatch["dead"], edgecolor=st.BLACK, linewidth=st.LW, label="Unused")]
    ax.legend(handles=hd, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
                   columnspacing=1.4, handlelength=1.6, borderaxespad=0.3)
    st.save(fig, a.outdir, "fig_tangent_rank")


if __name__ == "__main__":
    main()
