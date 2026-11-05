#!/usr/bin/env python
"""
Tangent-space diagnostics as four separate single-column figures (IEEE column width 3.5 in).

    python make_tangent_panels.py <dir with tangent_*.csv and svals/> --outdir .

Writes
  fig_tangent_spectrum.pdf   singular-value spectrum of the tangent vectors (old panel a)
  fig_tangent_alignment.pdf  alignment A after training vs the 2-design prediction (old panel b)
  fig_tangent_gradient.pdf   squared gradient norms, circuit vs loss (old panel c)
  fig_tangent_rank.pdf       metric rank vs parameter count (old panel d)
Same data, same statistics and colors as make_tangent_figure.py; only the layout changes.
Panel titles are left out: each figure's caption carries the title.
"""

import argparse
import os
import importlib.util

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("mtf", os.path.join(HERE, "make_tangent_figure.py"))
mtf = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(mtf)
BLUE, ORANGE, GREY, INK, LINE = mtf.BLUE, mtf.ORANGE, mtf.GREY, mtf.INK, mtf.LINE
ORDER, LABEL, f = mtf.ORDER, mtf.LABEL, mtf.f
OFF = {"rho01": -0.17, "rho1": 0.17}
COL = {"rho01": BLUE, "rho1": ORANGE}
MK = {"rho01": "o", "rho1": "s"}
W, H = 3.5, 2.45                     # inches: one IEEE column


def style(ax):
    ax.grid(axis="y", color=LINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("black"); ax.spines[sp].set_linewidth(1.1)


def groups(ax, right=0.3):
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([LABEL[c] for c in ORDER])
    ax.set_xlim(-0.6, len(ORDER) - right)


def summary(ax, x, ys, color, marker, log=False, ms=6.0, z=4):
    """Mean with +-SD whiskers (linear) or median with 5-95% range (log)."""
    ys = np.asarray(ys, float); ys = ys[np.isfinite(ys)]
    if not ys.size:
        return
    if log:
        c = np.median(ys); lo, hi = np.percentile(ys, [5, 95])
    else:
        c = ys.mean(); lo, hi = c - ys.std(ddof=1), c + ys.std(ddof=1)
    ax.errorbar(x, c, yerr=[[c - lo], [hi - c]], fmt=marker, color=color, markersize=ms, markeredgecolor="white",
                markeredgewidth=0.8, elinewidth=1.1, capsize=3, capthick=1.1, zorder=z)


def rho_handles():
    return [Line2D([], [], marker="o", color=BLUE, linestyle="", markersize=5, label=r"$\rho_q=0.1$ (paper schedule)"),
            Line2D([], [], marker="s", color=ORANGE, linestyle="", markersize=5, label=r"$\rho_q=1$ (shared learning rate)")]


def save(fig, outdir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("wrote", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()
    rows = mtf.load(a.dir)
    svals = mtf.load_svals(a.dir)
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif", "font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
                         "axes.edgecolor": "black", "axes.linewidth": 1.1, "text.color": INK, "axes.labelcolor": INK,
                         "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 1.0, "ytick.major.width": 1.0})

    # ---------------------------------------------------------------- spectrum
    fig, ax = plt.subplots(figsize=(W, H))
    spec_c = {"overhead_adaptive": "#4a3aa7", "sat6": "#1baf7a", "so2sat_adaptive": "#c98500"}
    spec_lab = {"overhead_adaptive": "Overhead (CX, 8 qb)", "sat6": "SAT-6 (XX, 10 qb)", "so2sat_adaptive": "So2Sat (CRY, 10 qb)"}
    floor = 1e-4
    for cfg in ORDER:
        if cfg not in svals:
            continue
        sp = svals[cfg]
        med = np.median(sp, axis=0); lo = np.percentile(sp, 5, axis=0); hi = np.percentile(sp, 95, axis=0)
        idx = np.arange(1, med.size + 1)
        rank = int(np.sum(med > 1e-8 * med[0]))
        medc, loc, hic = (np.clip(v, floor, None) for v in (med, lo, hi))
        ax.fill_between(idx, loc, hic, color=spec_c[cfg], alpha=0.18, linewidth=0)
        ax.plot(idx, medc, color=spec_c[cfg], linewidth=1.5, label=spec_lab[cfg], zorder=3)
        ax.vlines(rank + 0.5, floor, medc[rank - 1], color=spec_c[cfg], linestyle=(0, (2, 2)), linewidth=0.9)
        ax.text(rank + 2.5, floor * 1.6, f"{rank}", color=spec_c[cfg], fontsize=7.5, fontweight="bold", ha="left", va="bottom")
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(floor * 0.7, 3)
    ax.set_xlim(0, 146)
    ax.set_xticks([0, 25, 50, 72, 110, 128, 144])
    ax.set_xlabel("tangent direction index (sorted)")
    ax.set_ylabel(r"singular value of $[\,|D_l\psi\rangle\,]$")
    style(ax)
    ax.legend(frameon=False, loc="lower left", handlelength=1.2, borderaxespad=0.2)
    save(fig, a.outdir, "fig_tangent_spectrum")

    # ---------------------------------------------------------------- alignment
    fig, ax = plt.subplots(figsize=(W, H))
    for i, cfg in enumerate(ORDER):
        for s in ("rho01", "rho1"):
            rs = rows.get((cfg, s), [])
            if rs:
                summary(ax, i + OFF[s], [f(r, "A_final") for r in rs], COL[s], MK[s])
        rs = rows.get((cfg, "rho01"), [])
        if rs:
            deff = np.mean([f(r, "deff_final") for r in rs]); D = int(float(rs[0]["hilbert_dim"]))
            ax.hlines(deff / (2 * (D - 1)), i - 0.42, i + 0.42, color=INK, linestyle=(0, (3, 2)), linewidth=1.0, zorder=5)
    groups(ax)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(r"alignment $A(\theta)$")
    style(ax)
    hd = rho_handles() + [Line2D([], [], color=INK, linestyle=(0, (3, 2)), label=r"2-design level $d_{\mathrm{eff}}/2(D-1)$")]
    ax.legend(handles=hd, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2,
              handletextpad=0.4, columnspacing=1.0)
    save(fig, a.outdir, "fig_tangent_alignment")

    # ---------------------------------------------------------------- gradient norms
    fig, ax = plt.subplots(figsize=(W, H))
    for i, cfg in enumerate(ORDER):
        for s in ("rho01", "rho1"):
            rs = rows.get((cfg, s), [])
            if rs:
                summary(ax, i + OFF[s], [f(r, "grad2_final") for r in rs], GREY, MK[s], log=True, ms=5.0, z=3)
                summary(ax, i + OFF[s], [f(r, "task_grad2_final") for r in rs], COL[s], MK[s], log=True)
    groups(ax)
    ax.set_yscale("log")
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(3e-4, 4)
    ax.set_ylabel("squared gradient norm")
    ax.text(-0.55, 1.6, r"circuit  $\|\nabla C\|^2$ (grey)", fontsize=7.5, color="black", ha="left", va="bottom")
    ax.text(-0.55, 0.0005, r"loss  $\|\partial L/\partial\theta_q\|^2$ (color)", fontsize=7.5, color=INK, ha="left", va="bottom")
    style(ax)
    ax.legend(handles=rho_handles(), frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2,
              handletextpad=0.4, columnspacing=1.0)
    save(fig, a.outdir, "fig_tangent_gradient")

    # ---------------------------------------------------------------- rank vs parameter count
    # One horizontal bar per configuration, split into independent, merged and unused angles.
    fig, ax = plt.subplots(figsize=(W, 1.75))
    live_c, merge_c, dead_c = "#2a78d6", "#a9c7ec", "#d9d8d4"
    ylab = {"overhead_adaptive": "Overhead\n(CX, 8 qb)", "sat6": "SAT-6\n(XX, 10 qb)", "so2sat_adaptive": "So2Sat\n(CRY, 10 qb)"}
    for i, cfg in enumerate(ORDER):
        rs = rows.get((cfg, "rho01"), [])
        if not rs:
            continue
        L = int(float(rs[0]["n_params"])); nq = rs[0]["n_qubits"]; gate = rs[0]["gate"]
        _, pred, dead, rym, xxm = mtf._an.deff_predicted(nq, rs[0].get("n_reup", 2), gate)
        deff = int(round(np.mean([f(r, "deff_final") for r in rs])))
        merged = rym + xxm
        y, h = i, 0.58
        ax.barh(y, deff, height=h, color=live_c, zorder=3)
        ax.barh(y, merged, height=h, left=deff, color=merge_c, zorder=3)
        ax.barh(y, dead, height=h, left=deff + merged, color=dead_c, zorder=3)
        ax.text(deff / 2, y, f"$d_{{\\mathrm{{eff}}}}={deff}$ (pred. {pred})", ha="center", va="center", fontsize=7.5, color="white")
        ax.text(deff + merged / 2, y, f"{merged}", ha="center", va="center", fontsize=7.5)
        if dead:
            ax.text(deff + merged + dead / 2, y, f"{dead}", ha="center", va="center", fontsize=7.5)
        ax.text(L + 2, y, f"$L={L}$", ha="left", va="center", fontsize=7.5)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([ylab[c] for c in ORDER])
    ax.set_ylim(len(ORDER) - 0.5, -0.5)
    ax.set_xlim(0, 168)
    ax.set_xlabel("quantum parameters")
    ax.grid(axis="x", color=LINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("black"); ax.spines[sp].set_linewidth(1.1)
    hd = [Patch(color=live_c, label="independent"), Patch(color=merge_c, label="merged by gate identities"),
          Patch(color=dead_c, label="unused (CX, CZ)")]
    ax.legend(handles=hd, frameon=False, loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=3,
              handletextpad=0.4, columnspacing=0.9, handlelength=1.2)
    save(fig, a.outdir, "fig_tangent_rank")


if __name__ == "__main__":
    main()
