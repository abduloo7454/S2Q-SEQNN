#!/usr/bin/env python
"""
Fig. 1(b): the headline result as a forest plot.

Every interval is recomputed from the run log with the same functions that
build Table 5 (build_tables.py: paired t-test, 95% CI on the seed-paired
difference, Holm over the full family), so the figure and the table cannot
disagree. Nothing is typed by hand.

    # from the repository root
    python analysis/make_fig1_forest.py --runs results/all_runs.csv \
        --out figures/fig1_forest
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_tables as bt  # noqa: E402

BLUE, INK, MUTED, LINE, GREY = "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e2", "#8c8b86"
DATASETS = [("overhead", "Overhead-MNIST"), ("sat6", "SAT-6"), ("so2sat", "So2Sat LCZ42")]
# quantum-branch controls first, classical-interface controls last
ROWS = [("classical_only", "No circuit"),
        ("mlp_replace", "Trained MLP"),
        ("frozen_mlp", "Frozen MLP"),
        ("frozen_rff", "Frozen random features"),
        ("frozen_quantum", "Frozen circuit"),
        ("no_iqp", "No IQP encoding"),
        ("no_reupload", "No re-uploading"),
        ("fixed_z", "Fixed-$Z$ read-out"),
        ("quantum_only", "No fusion (read-out only)")]
N_QUANTUM = 7  # rows 0..6 concern the quantum branch; 7..8 the classical interface


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default="figures/fig1_forest")
    ap.add_argument("--xmax", type=float, default=5.0,
                    help="clip the x-axis here; clipped intervals get an arrow")
    a = ap.parse_args()

    by = bt.load_runs(a.runs)
    _, tests = bt.table_isolation(by, [])
    p_holm = bt.holm([t["p_t"] for *_, t in tests])
    res = {(ds, ab): (t, p) for (ds, ab, _, t), p in zip(tests, p_holm)}

    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8,
                         "axes.edgecolor": "black", "axes.linewidth": 1.3,
                         "xtick.color": "black", "ytick.color": "black",
                         "text.color": INK, "axes.labelcolor": INK,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35), sharey=True, sharex=True)
    y = np.arange(len(ROWS))[::-1]

    for ax, (ds, name) in zip(axes, DATASETS):
        # shade the classical-interface rows so the two groups read apart
        ax.axhspan(-0.5, len(ROWS) - N_QUANTUM - 0.5, color="#f3f2ef", zorder=0, lw=0)
        ax.plot([0, 0], [-0.6, len(ROWS) - 0.4], color=INK, lw=0.8, zorder=2)
        lo_all = []
        for yi, (ab, _) in zip(y, ROWS):
            t, p = res[(ds, ab)]
            sig = p < 0.05
            c = BLUE if sig else GREY
            lo, hi = t["ci_lo"], t["ci_hi"]
            lo_all.append(lo)
            hi_c = min(hi, a.xmax)
            ax.plot([lo, hi_c], [yi, yi], color=c, lw=2, solid_capstyle="round", zorder=3)
            if hi > a.xmax:  # clipped: say so
                ax.annotate("", xy=(a.xmax, yi), xytext=(a.xmax - 0.35, yi),
                            arrowprops=dict(arrowstyle="-|>", color=c, lw=1.2), zorder=3)
            ax.plot(min(t["delta"], a.xmax), yi, "o", ms=5.2, zorder=4,
                    mfc=c if sig else "white", mec=c, mew=1.3)
            if sig:
                ax.text(hi_c + 0.08, yi, "*", color=INK, fontsize=9, va="center", ha="left")
        ax.set_title(name, loc="left", fontsize=8.5, pad=5)
        ax.grid(axis="x", color=LINE, lw=0.6, zorder=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", length=3, width=1.1, pad=2)
        ax.set_xlabel(r"$\Delta$ = full $-$ control (pp)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([lab for _, lab in ROWS], fontsize=7.6)
    axes[0].set_ylim(-0.6, len(ROWS) - 0.4)
    lo_min = min(t["ci_lo"] for (_, _, _, t) in tests)
    axes[0].set_xlim(np.floor(lo_min) - 0.2, a.xmax + 0.45)
    # the three frozen-type controls are the paper's test; set their labels in bold
    for lab in axes[0].get_yticklabels():
        if lab.get_text() in ("No circuit", "Frozen random features", "Frozen circuit"):
            lab.set_fontweight("bold")

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=GREY, marker="o", mfc="white", mec=GREY, lw=2, ms=5,
                      label="not significant after Holm correction"),
               Line2D([], [], color=BLUE, marker="o", mfc=BLUE, mec=BLUE, lw=2, ms=5,
                      label=r"$p_{\mathrm{Holm}}<0.05$ (*)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.55, -0.07))
    fig.tight_layout(w_pad=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=1000 if ext == "png" else None,
                    bbox_inches="tight", pad_inches=0.04)
    print("wrote", a.out + ".pdf/.png from", sum(len(v) for v in by.values()), "runs,",
          len(tests), "tests")


if __name__ == "__main__":
    main()
