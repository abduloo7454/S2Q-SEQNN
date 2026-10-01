#!/usr/bin/env python
"""
Figure: tangent-space diagnostics of the S2Q-SEQNN quantum branch.

    python make_tangent_figure.py <dir with tangent_*.csv> --out fig_tangent

Four panels, one point per seed where applicable:
  (a) relative parameter displacement ||theta_f - theta_0|| / ||theta_0||
  (b) alignment A after training, against the 2-design prediction d_eff / 2(D-1)
  (c) squared gradient norm: per-Pauli ||dC||^2 vs the loss gradient reaching
      the circuit through the head, ||dL/dtheta_q||^2 (log scale)
  (d) parameter accounting: measured rank d_eff against the parameter count,
      split into disconnected (CX/CZ) angles and algebraically merged angles
Series: rho_q = 0.1 (the paper's schedule) and rho_q = 1 (shared learning rate).
"""

import argparse
import csv
import glob
import math
import os
import importlib.util
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BLUE, ORANGE, GREY, INK, MUTED, LINE = "#2a78d6", "#eb6834", "#9a9893", "#0b0b0b", "#52514e", "#e6e5e2"
LABEL = {"overhead_adaptive": "Overhead\n(CX, 8 qb)", "sat6": "SAT-6\n(XX, 10 qb)", "so2sat_adaptive": "So2Sat\n(CRY, 10 qb)"}
ORDER = ["overhead_adaptive", "sat6", "so2sat_adaptive"]

_spec = importlib.util.spec_from_file_location("an", os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze_tangent.py"))
_an = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_an)


def load_svals(d):
    """Mean singular-value spectrum of the tangent Jacobian per config: (seeds x alphas) averaged."""
    out = {}
    for cfg in ORDER:
        files = sorted(glob.glob(os.path.join(d, "svals", f"{cfg}__seed*.npz")))
        if not files:
            continue
        specs = []
        for fpath in files:
            z = np.load(fpath)
            sv = np.sort(z["svals_final"], axis=1)[:, ::-1]      # alphas x L, descending
            specs.append(sv)
        specs = np.concatenate(specs, axis=0)                     # (seeds*alphas) x L
        out[cfg] = specs
    return out


def load(d):
    rows = defaultdict(list)
    for f in glob.glob(os.path.join(d, "tangent_*.csv")):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                series = "rho1" if r["campaign"] == "schedule" else "rho01"
                rows[(r["config"], series)].append(r)
    return rows


def f(r, k):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return math.nan


def strip(ax, x, ys, color, marker, jitter=0.05, ms=4.6, z=3, alpha=0.95):
    rng = np.random.default_rng(0)
    ys = np.asarray(ys, float)
    xj = x + rng.uniform(-jitter, jitter, len(ys))
    ax.scatter(xj, ys, color=color, marker=marker, s=ms ** 2, edgecolor="white", linewidths=0.8, zorder=z, alpha=alpha)


def mean_bar(ax, x, ys, color, w=0.2):
    ys = np.asarray(ys, float); ys = ys[np.isfinite(ys)]
    if ys.size:
        ax.hlines(ys.mean(), x - w, x + w, color=color, linewidth=2.2, zorder=4)


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=8.5, pad=6)
    ax.grid(axis="y", color=LINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):            # inner panel: only the x and y axis lines remain
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("black"); ax.spines[sp].set_linewidth(1.3)
    ax.set_facecolor("white")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", default="fig_tangent")
    a = ap.parse_args()
    rows = load(a.dir)
    svals = load_svals(a.dir)

    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "axes.edgecolor": "black", "axes.linewidth": 1.3, "xtick.color": "black",
                         "ytick.color": "black", "text.color": INK, "axes.labelcolor": INK,
                         "xtick.major.size": 3, "ytick.major.size": 3,
                         "xtick.major.width": 1.1, "ytick.major.width": 1.1,
                         "xtick.minor.width": 0.8, "ytick.minor.width": 0.8})
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.7))
    (ax_a, ax_b), (ax_c, ax_d) = axes
    off = {"rho01": -0.17, "rho1": 0.17}
    col = {"rho01": BLUE, "rho1": ORANGE}
    mk = {"rho01": "o", "rho1": "s"}

    def summary(ax, x, ys, color, marker, log=False, ms=6.5, z=4):
        """One marker per group: mean with +-SD whiskers (linear) or median with 5-95% range (log)."""
        ys = np.asarray(ys, float); ys = ys[np.isfinite(ys)]
        if not ys.size:
            return
        if log:
            c = np.median(ys); lo, hi = np.percentile(ys, [5, 95])
        else:
            c = ys.mean(); lo, hi = c - ys.std(ddof=1), c + ys.std(ddof=1)
        ax.errorbar(x, c, yerr=[[c - lo], [hi - c]], fmt=marker, color=color, markersize=ms, markeredgecolor="white",
                    markeredgewidth=0.8, elinewidth=1.2, capsize=3, capthick=1.2, zorder=z)

    for i, cfg in enumerate(ORDER):
        for s in ("rho01", "rho1"):
            rs = rows.get((cfg, s), [])
            if not rs:
                continue
            x = i + off[s]
            summary(ax_b, x, [f(r, "A_final") for r in rs], col[s], mk[s])
            summary(ax_c, x, [f(r, "grad2_final") for r in rs], GREY, mk[s], log=True, ms=5.5, z=3)
            summary(ax_c, x, [f(r, "task_grad2_final") for r in rs], col[s], mk[s], log=True)
        # 2-design reference for alignment: d_eff / 2(D-1)
        rs = rows.get((cfg, "rho01"), [])
        if rs:
            deff = np.mean([f(r, "deff_final") for r in rs]); D = int(float(rs[0]["hilbert_dim"]))
            ax_b.hlines(deff / (2 * (D - 1)), i - 0.42, i + 0.42, color=INK, linestyle=(0, (3, 2)), linewidth=1.0, zorder=5)

    # (d) parameter accounting
    live_c, merge_c, dead_c = "#2a78d6", "#a9c7ec", "#d9d8d4"
    for i, cfg in enumerate(ORDER):
        rs = rows.get((cfg, "rho01"), [])
        if not rs:
            continue
        L = int(float(rs[0]["n_params"])); nq = rs[0]["n_qubits"]; gate = rs[0]["gate"]
        _, pred, dead, rym, xxm = _an.deff_predicted(nq, rs[0].get("n_reup", 2), gate)
        deff = np.mean([f(r, "deff_final") for r in rs])
        merged = rym + xxm
        ax_d.bar(i, deff, width=0.55, color=live_c, zorder=3)
        ax_d.bar(i, merged, width=0.55, bottom=deff, color=merge_c, zorder=3)
        ax_d.bar(i, dead, width=0.55, bottom=deff + merged, color=dead_c, zorder=3)
        ax_d.text(i, L + 3, f"$L={L}$", ha="center", va="bottom", fontsize=7.5, color=INK)
        ax_d.text(i, deff / 2, f"$d_{{\\mathrm{{eff}}}}={int(round(deff))}$\npred. {pred}", ha="center", va="center",
                  fontsize=6.3, color="white")
        if dead:
            ax_d.text(i, deff + merged + dead / 2, f"{dead} unused", ha="center", va="center", fontsize=6.8, color="black")
        ax_d.text(i, deff + merged / 2, f"{merged} merged", ha="center", va="center", fontsize=6.8, color="black")

    for ax in (ax_b, ax_c, ax_d):
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([LABEL[c] for c in ORDER])
        ax.set_xlim(-0.6, len(ORDER) - 0.3)

    # (a) tangent-space spectrum
    ax_a.clear()
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
        medc = np.clip(med, floor, None); loc = np.clip(lo, floor, None); hic = np.clip(hi, floor, None)
        ax_a.fill_between(idx, loc, hic, color=spec_c[cfg], alpha=0.18, linewidth=0)
        ax_a.plot(idx, medc, color=spec_c[cfg], linewidth=1.6, label=spec_lab[cfg], zorder=3)
        ax_a.vlines(rank + 0.5, floor, medc[rank - 1], color=spec_c[cfg], linestyle=(0, (2, 2)), linewidth=0.9)
        ax_a.text(rank + 2.5, floor * 1.6, f"{rank}", color=spec_c[cfg], fontsize=7, fontweight="bold", ha="left", va="bottom")
    ax_a.set_yscale("log")
    ax_a.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax_a.set_ylim(floor * 0.7, 3)
    ax_a.set_xlim(0, 146)
    ax_a.set_xticks([0, 25, 50, 72, 110, 128, 144])
    ax_a.set_xlabel("tangent direction index (sorted)")
    ax_a.set_ylabel(r"singular value of $[\,|D_l\psi\rangle\,]$")
    style(ax_a, r"(a) Tangent-space spectrum (trained circuit)")
    ax_a.legend(frameon=False, fontsize=6.4, loc="lower left", handlelength=1.2, borderaxespad=0.2)

    style(ax_b, "(b) Alignment $A(\\theta)$ after training (mean ± SD)")
    ax_b.set_ylim(0, 1.05)
    ax_b.set_ylabel(r"alignment $A(\theta)$")
    ax_b.text(2.42, 0.10, "2-design\nprediction", fontsize=7, color="black", ha="right", va="bottom")

    style(ax_c, "(c) Squared gradient norm (median, 5–95%)")
    ax_c.set_yscale("log")
    ax_c.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax_c.set_ylim(3e-4, 4)
    ax_c.set_ylabel("squared gradient norm")
    ax_c.text(-0.55, 1.6, r"circuit  $\|\nabla C\|^2$", fontsize=7.5, color="black", ha="left", va="bottom")
    ax_c.text(-0.55, 0.0006, r"loss  $\|\partial L/\partial\theta_q\|^2$", fontsize=7, color=INK, ha="left", va="bottom")

    style(ax_d, r"(d) Rank of the metric vs. parameter count")
    ax_d.set_ylim(0, 165)
    ax_d.set_ylabel("parameters")
    ax_d.set_xlim(-0.6, len(ORDER) - 0.4)

    handles = [Line2D([], [], marker="o", color=BLUE, linestyle="", markersize=5, label=r"$\rho_q=0.1$ (paper schedule)"),
               Line2D([], [], marker="s", color=ORANGE, linestyle="", markersize=5, label=r"$\rho_q=1$ (shared learning rate)"),
               Line2D([], [], color=INK, linestyle=(0, (3, 2)), label=r"$d_{\mathrm{eff}}/2(D-1)$, 2-design prediction of $A$"),
               Patch(color=live_c, label="connected, independent"),
               Patch(color=merge_c, label="merged by gate identities"),
               Patch(color=dead_c, label="unused by CX / CZ")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, -0.01), handletextpad=0.5, columnspacing=1.4)
    fig.tight_layout(rect=(0.01, 0.07, 0.99, 0.99), h_pad=3.4, w_pad=3.2)
    # one box around each complete subfigure (title + axes + tick and axis labels)
    from matplotlib.patches import FancyBboxPatch
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    cols = {ax_a: (0.008, 0.494), ax_b: (0.506, 0.992), ax_c: (0.008, 0.494), ax_d: (0.506, 0.992)}
    bbs = {ax: ax.get_tightbbox(rend).transformed(fig.transFigure.inverted()) for ax in (ax_a, ax_b, ax_c, ax_d)}
    py = 0.018
    top = (min(bbs[ax_a].y0, bbs[ax_b].y0) - py, max(bbs[ax_a].y1, bbs[ax_b].y1) + py)
    bot = (min(bbs[ax_c].y0, bbs[ax_d].y0) - py, max(bbs[ax_c].y1, bbs[ax_d].y1) + py)
    h = max(top[1] - top[0], bot[1] - bot[0])            # identical box height for all four
    rows = {ax_a: (top[1] - h, h), ax_b: (top[1] - h, h), ax_c: (bot[0], h), ax_d: (bot[0], h)}
    for ax in (ax_a, ax_b, ax_c, ax_d):
        x0, x1 = cols[ax]; y0, hh = rows[ax]
        fig.patches.append(FancyBboxPatch((x0, y0), x1 - x0, hh,
                                          boxstyle="round,pad=0,rounding_size=0.008", transform=fig.transFigure,
                                          fill=False, edgecolor=MUTED, linewidth=0.8, zorder=0.5, clip_on=False))
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=1000, bbox_inches="tight", pad_inches=0.06)
    print("wrote", a.out + ".pdf/.png")


if __name__ == "__main__":
    main()
