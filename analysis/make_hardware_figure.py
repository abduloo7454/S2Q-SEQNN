#!/usr/bin/env python
"""
Figure: the S2Q-SEQNN quantum branch on real hardware, trained circuit vs frozen twin.

    python make_hardware_figure.py run_trained run_frozen --out fig_hardware

Each argument is a directory holding hw_features.csv and hw_results.json as written
by hw_collect.py. The second directory is optional; with one directory the figure
drops the frozen series.

Three panels:
  (a) hardware read-out against the exact simulator read-out, every image and wire,
      with the fitted shrinkage slope -- the contraction toward zero that
      depolarising noise produces
  (b) that slope per wire, so a single bad qubit is visible rather than averaged away
  (c) subset accuracy under three read-outs: exact simulator, shot noise alone,
      and hardware, with Wilson intervals

Every number is read from the two files. Nothing is typed in by hand.
Style follows make_tangent_figure.py so the figure sits beside the others.
"""

import argparse
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import pandas as pd

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, LINE = "#0b0b0b", "#52514e", "#e6e5e2"


def wilson(k, n, z=1.96):
    """Wilson score interval; k successes of n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load(run_dir):
    feat = pd.read_csv(os.path.join(run_dir, "hw_features.csv"))
    res = json.load(open(os.path.join(run_dir, "hw_results.json")))
    n_q = int(res["n_qubits"])
    cols = [c for c in feat.columns if c.startswith("q_sim_")]
    idx = sorted(int(c.split("_")[-1]) for c in cols)
    sim = feat[[f"q_sim_{i}" for i in idx]].to_numpy()
    hw = feat[[f"q_hw_{i}" for i in idx]].to_numpy()
    return dict(feat=feat, res=res, n_q=n_q, sim=sim, hw=hw,
                wire=np.array([i % n_q for i in idx]))


def tidy(ax, title):
    """Plain L-shaped axes. No panel box: a frame drawn around the axes rect
    runs straight through the outermost tick labels, which sit centred on the
    axis limits."""
    ax.set_title(title, loc="left", fontsize=8.5, pad=6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(True)
        ax.spines[sp].set_color(MUTED)
        ax.spines[sp].set_linewidth(0.7)
    ax.tick_params(length=2.5, width=0.7, color=MUTED, pad=2.5)
    ax.set_facecolor("white")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run_trained [run_frozen]")
    ap.add_argument("--labels", nargs="+",
                    default=["trained circuit", "frozen circuit", "frozen, other qubits"])
    ap.add_argument("--keys", nargs="+", default=["trained", "frozen", "frozenalt"],
                    help="macro-name stems, one per run: \\hw<key>slope and friends. "
                         "Letters only -- LaTeX command names cannot contain digits.")
    ap.add_argument("--out", default="fig_hardware")
    ap.add_argument("--dpi", type=int, default=1000,
                    help="raster resolution; 1000 matches the other figures, 500 is the floor")
    a = ap.parse_args()

    runs = [load(r) for r in a.runs]
    labels = a.labels[:len(runs)]
    colors = [BLUE, ORANGE, GREEN][:len(runs)]
    markers = ["o", "s", "^"][:len(runs)]

    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "text.color": INK, "axes.labelcolor": INK,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(7.16, 2.55))

    # ---- (a) hardware vs exact read-out -------------------------------------
    for r, lab, col in zip(runs, labels, colors):
        x, y = r["sim"].ravel(), r["hw"].ravel()
        ax_a.scatter(x, y, s=3.0, color=col, alpha=0.22, linewidths=0, zorder=3)
        s, b0 = np.polyfit(x, y, 1)
        xs = np.array([x.min(), x.max()])
        ax_a.plot(xs, s * xs + b0, color=col, linewidth=1.4, zorder=4)
        r["slope_all"], r["corr_all"] = s, float(np.corrcoef(x, y)[0, 1])
    lim = 1.02
    ax_a.plot([-lim, lim], [-lim, lim], color=INK, linestyle=(0, (3, 2)),
              linewidth=0.9, zorder=2)
    ax_a.set_xlim(-lim, lim); ax_a.set_ylim(-lim, lim)
    ax_a.set_xlabel(r"exact simulator $\langle P_w\rangle$")
    ax_a.set_ylabel(r"hardware $\langle P_w\rangle$")
    ax_a.set_aspect("equal", adjustable="box")
    txt = "\n".join(
        rf"{lab}: slope {r['slope_all']:.2f}, $\rho$ {r['corr_all']:.2f}"
        for r, lab in zip(runs, labels))
    ax_a.text(0.97, 0.03, txt, transform=ax_a.transAxes, fontsize=5.8,
              va="bottom", ha="right", color=INK,
              bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    tidy(ax_a, "(a) read-out fidelity")

    # ---- (b) shrinkage slope per wire ---------------------------------------
    n_q = runs[0]["n_q"]
    w = 0.8 / len(runs)
    for k, (r, lab, col) in enumerate(zip(runs, labels, colors)):
        slopes = []
        for q in range(n_q):
            m = r["wire"] == q
            x, y = r["sim"][:, m].ravel(), r["hw"][:, m].ravel()
            slopes.append(np.polyfit(x, y, 1)[0] if len(x) > 2 else np.nan)
        pos = np.arange(n_q) + (k - (len(runs) - 1) / 2) * (0.62 / len(runs))
        ax_b.plot(pos, slopes, marker=markers[k], markersize=4.2,
                  markeredgecolor="white", markeredgewidth=0.7, linestyle="",
                  color=col, zorder=3, label=lab)
        r["slopes"] = np.array(slopes)
    ax_b.axhline(1.0, color=INK, linestyle=(0, (3, 2)), linewidth=0.9, zorder=2)
    ax_b.text(n_q - 1.0, 1.008, "noiseless", fontsize=6.2, color=INK,
              va="bottom", ha="right")
    ax_b.set_xticks(np.arange(n_q))
    ax_b.set_xlabel("qubit")
    ax_b.set_ylabel("shrinkage slope")
    allsl = np.concatenate([r["slopes"] for r in runs])
    lo_b = min(0.75, float(np.nanmin(allsl)) - 0.05)
    ax_b.set_ylim(lo_b, 1.06)
    ax_b.grid(axis="y", color=LINE, linewidth=0.6, zorder=0)
    ax_b.set_axisbelow(True)
    tidy(ax_b, "(b) attenuation per wire")

    # ---- (c) subset accuracy under three read-outs --------------------------
    groups = ["exact\nsimulator", "shot\nnoise", "hardware"]
    gw = 0.8 / len(runs)
    for k, (r, lab, col) in enumerate(zip(runs, labels, colors)):
        res, n = r["res"], int(r["res"]["n_images"])
        vals = [res["acc_exact_sim"], res["acc_shot_noise_mean"], res["acc_hardware"]]
        lo, hi = [], []
        for j, v in enumerate(vals):
            if j == 1:                      # shot noise: SD over sampled replicas
                sd = res["acc_shot_noise_sd"]
                lo.append(v - sd); hi.append(v + sd)
            else:                           # Wilson on n images
                a0, b0 = wilson(round(v * n), n)
                lo.append(a0); hi.append(b0)
        pos = np.arange(3) + (k - (len(runs) - 1) / 2) * gw
        ax_c.bar(pos, np.array(vals) * 100, width=gw * 0.92, color=col, zorder=3)
        ax_c.errorbar(pos, np.array(vals) * 100,
                      yerr=[100 * (np.array(vals) - np.array(lo)),
                            100 * (np.array(hi) - np.array(vals))],
                      fmt="none", ecolor=INK, elinewidth=0.8, capsize=2, zorder=4)
        # Three narrow bars leave no room for horizontal labels, so they are
        # set upright above the error bar instead of being dropped.
        rot = 0 if len(runs) <= 2 else 90
        for p, v, h in zip(pos, vals, hi):
            ax_c.text(p, 100 * h + (1.4 if rot == 0 else 1.8), f"{100*v:.1f}",
                      ha="center", va="bottom", rotation=rot,
                      fontsize=6.2 if rot == 0 else 5.8, color=INK)
    ax_c.set_xticks(np.arange(3)); ax_c.set_xticklabels(groups, fontsize=6.8)
    ax_c.set_ylabel("subset accuracy (%)")
    ax_c.set_ylim(0, 108 if len(runs) <= 2 else 118)
    ax_c.grid(axis="y", color=LINE, linewidth=0.6, zorder=0)
    ax_c.set_axisbelow(True)
    tidy(ax_c, "(c) accuracy on the subset")

    # ---- legend and caption line -------------------------------------------
    res0 = runs[0]["res"]
    twoq = res0.get("twoq_gates_per_circuit") or [None, None]
    twoq_s = f"{twoq[0]}" if twoq[0] == twoq[1] else f"{twoq[0]}--{twoq[1]}"
    handles = [Line2D([], [], color=c, marker=m, linestyle="", markersize=5,
                      markeredgecolor="white", markeredgewidth=0.7, label=l)
               for c, m, l in zip(colors, markers, labels)]
    handles.append(Line2D([], [], color=INK, linestyle=(0, (3, 2)),
                          label="noiseless reference"))
    fig.legend(handles=handles, loc="lower center", ncol=min(3, len(handles)),
               frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.085))

    fig.tight_layout(rect=(0, 0.02, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=a.dpi, bbox_inches="tight", pad_inches=0.06)
    print(f"wrote {a.out}.pdf and {a.out}.png at {a.dpi} dpi")

    # ---- macros, so the text never hard-codes a hardware number -------------
    lines = ["% generated by make_hardware_figure.py; do not edit"]
    keys = a.keys[:len(runs)]
    for r, key in zip(runs, keys):
        res = r["res"]
        for name, val, fmt in [
            (f"hw{key}slope", r["slope_all"], "{:.2f}"),
            (f"hw{key}corr", r["corr_all"], "{:.2f}"),
            (f"hw{key}accsim", 100 * res["acc_exact_sim"], "{:.1f}"),
            (f"hw{key}accshot", 100 * res["acc_shot_noise_mean"], "{:.1f}"),
            (f"hw{key}accshotsd", 100 * res["acc_shot_noise_sd"], "{:.1f}"),
            (f"hw{key}acchw", 100 * res["acc_hardware"], "{:.1f}"),
            (f"hw{key}agree", 100 * res["prediction_agreement_hw_vs_sim"], "{:.1f}"),
        ]:
            lines.append(f"\\providecommand{{\\{name}}}{{}}"
                         f"\\renewcommand{{\\{name}}}{{{fmt.format(val)}}}")
    # run-level facts, so the prose never hard-codes a device number either
    d = res0.get("depth") or [None, None]
    depth_s = f"{d[0]}" if d[0] == d[1] else f"{d[0]}--{d[1]}"
    allsl2 = np.concatenate([r["slopes"] for r in runs])
    extra = [
        ("hwbackend", "\\texttt{" + res0["backend"].replace("_", "\\_") + "}"),
        ("hwtwoq", twoq_s),
        ("hwdepth", depth_s),
        ("hwshots", str(res0["shots"])),
        ("hwnimages", str(res0["n_images"])),
        ("hwnqubits", str(res0["n_qubits"])),
        ("hwslopemin", f"{float(np.nanmin(allsl2)):.2f}"),
        ("hwslopemax", f"{float(np.nanmax(allsl2)):.2f}"),
    ]
    for r, key in zip(runs, keys):
        extra.append((f"hw{key}mae", f"{r['res']['readout_mae']:.3f}"))
        extra.append((f"hw{key}qpu", str(r['res'].get('qpu_usage', ''))))
        extra.append((f"hw{key}phys",
                      ", ".join(str(q) for q in (r['res'].get('physical_qubits') or []))))
    for k, r in list(enumerate(runs))[1:]:
        d = np.abs(runs[0]["slopes"] - r["slopes"])
        extra.append((f"hwwiredeltamax{keys[k]}", f"{float(np.nanmax(d)):.3f}"))
        extra.append((f"hwwiredeltamean{keys[k]}", f"{float(np.nanmean(d)):.3f}"))
    for name, val in extra:
        lines.append(f"\\providecommand{{\\{name}}}{{}}"
                     f"\\renewcommand{{\\{name}}}{{{val}}}")
    with open(f"{a.out}_numbers.tex", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {a.out}_numbers.tex")


if __name__ == "__main__":
    main()
