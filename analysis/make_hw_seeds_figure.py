#!/usr/bin/env python3
"""
Hardware figure over all three seeds (Overhead-MNIST Adaptive on ibm_kingston).

    python make_hw_seeds_figure.py --root CODE/hardware --out fig_hardware_seeds

Reads hw_features.csv and hw_results.json of each seed's trained circuit and frozen
twin (seed 42: run_trained / run_frozen_pinned; 43, 44: run_s4x_trained / _frozen).
All six runs use the same physical qubits.

  (a) hardware against exact read-out, every image, wire, and seed, with one pooled fit
      per circuit
  (b) contraction slope per wire: mean over the three seeds, bars span min to max
  (c) whole-register slope per seed, trained against frozen, with prediction agreement

Writes fig_hardware_seeds.pdf/.png and fig_hardware_seeds_numbers.tex. Nothing typed by hand.
"""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, LINE = "#0b0b0b", "#52514e", "#e6e5e2"
PAIRS = [(42, "run_trained", "run_frozen_pinned"), (43, "run_s43_trained", "run_s43_frozen"),
         (44, "run_s44_trained", "run_s44_frozen")]


def load(d):
    f = pd.read_csv(os.path.join(d, "hw_features.csv"))
    r = json.load(open(os.path.join(d, "hw_results.json")))
    n = int(r["n_qubits"])
    idx = sorted(int(c.split("_")[-1]) for c in f.columns if c.startswith("q_sim_"))
    return dict(res=r, n=n, sim=f[[f"q_sim_{i}" for i in idx]].to_numpy(),
                hw=f[[f"q_hw_{i}" for i in idx]].to_numpy(), wire=np.array([i % n for i in idx]))


def wire_slopes(r):
    return np.array([np.polyfit(r["sim"][:, r["wire"] == q].ravel(), r["hw"][:, r["wire"] == q].ravel(), 1)[0]
                     for q in range(r["n"])])


def macro(k, v):
    return f"\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{v}}}\n"


def tidy(ax, title):
    ax.set_title(title, loc="left", fontsize=8, pad=5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=3, width=1.1, color="black", pad=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="fig_hardware_seeds")
    a = ap.parse_args()
    runs = {(s, k): load(os.path.join(a.root, d)) for s, t, f in PAIRS for k, d in (("trained", t), ("frozen", f))}
    phys = {tuple(r["res"]["physical_qubits"]) for r in runs.values()}
    assert len(phys) == 1, f"runs use different qubits: {phys}"
    for (s, k), r in runs.items():
        assert int(r["res"]["seed"]) == s, (s, k)
    seeds = [p[0] for p in PAIRS]
    n = runs[(42, "trained")]["n"]

    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 7.5, "axes.edgecolor": "black", "axes.linewidth": 1.3,
                         "xtick.color": "black", "ytick.color": "black", "text.color": INK, "axes.labelcolor": INK,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    fig, (A, B, C) = plt.subplots(1, 3, figsize=(7.16, 2.5), gridspec_kw=dict(width_ratios=[1, 1.25, 0.9], wspace=0.42))
    col = {"trained": BLUE, "frozen": ORANGE}
    pooled = {}
    for k in ("trained", "frozen"):
        x = np.concatenate([runs[(s, k)]["sim"].ravel() for s in seeds])
        y = np.concatenate([runs[(s, k)]["hw"].ravel() for s in seeds])
        A.scatter(x, y, s=2.2, color=col[k], alpha=0.14, linewidths=0, zorder=3)
        sl, b0 = np.polyfit(x, y, 1)
        xs = np.array([-1, 1])
        A.plot(xs, sl * xs + b0, color=col[k], lw=1.4, zorder=4)
        pooled[k] = (sl, float(np.corrcoef(x, y)[0, 1]))
    A.plot([-1.02, 1.02], [-1.02, 1.02], color=INK, ls=(0, (3, 2)), lw=0.9, zorder=2)
    A.set_xlim(-1.02, 1.02); A.set_ylim(-1.02, 1.02); A.set_aspect("equal", adjustable="box")
    A.set_xlabel(r"exact simulator $\langle P_w\rangle$"); A.set_ylabel(r"hardware $\langle P_w\rangle$")
    A.text(0.97, 0.03, "\n".join(f"{k}: slope {pooled[k][0]:.2f}, $\\rho$ {pooled[k][1]:.2f}" for k in ("trained", "frozen")),
           transform=A.transAxes, fontsize=6.5, ha="right", va="bottom",
           bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    tidy(A, "(a) read-out, three seeds")

    ws = {k: np.array([wire_slopes(runs[(s, k)]) for s in seeds]) for k in ("trained", "frozen")}
    for j, (k, mk) in enumerate((("trained", "o"), ("frozen", "s"))):
        m, lo, hi = ws[k].mean(0), ws[k].min(0), ws[k].max(0)
        pos = np.arange(n) + (j - 0.5) * 0.3
        B.errorbar(pos, m, yerr=[m - lo, hi - m], fmt=mk, color=col[k], ms=4, mec="white", mew=0.6,
                   elinewidth=0.9, capsize=0, zorder=3)
    B.axhline(1, color=INK, ls=(0, (3, 2)), lw=0.9)
    B.set_xticks(range(n)); B.set_xlabel("wire"); B.set_ylabel("contraction slope")
    B.set_ylim(0, 1.08); B.grid(axis="y", color=LINE, lw=0.5); B.set_axisbelow(True)
    tidy(B, "(b) per wire, mean and range")

    for k, mk in (("trained", "o"), ("frozen", "s")):
        v = [runs[(s, k)]["res"]["readout_shrinkage_slope"] for s in seeds]
        C.plot(seeds, v, marker=mk, color=col[k], ms=4.5, mec="white", mew=0.6, lw=1.2, zorder=3)
    for s in seeds:
        agr = [100 * runs[(s, k)]["res"]["prediction_agreement_hw_vs_sim"] for k in ("trained", "frozen")]
        C.text(s, 0.605, f"{agr[0]:.0f}/{agr[1]:.0f}%", ha="center", va="bottom", fontsize=6.8, color="black")
    C.text(0.03, 0.025, "label agreement (trained/frozen)", transform=C.transAxes, fontsize=5.9, color="black", va="bottom")
    C.set_xticks(seeds); C.set_xlim(41.5, 44.5); C.set_ylim(0.58, 0.86)
    C.set_xlabel("seed"); C.set_ylabel("whole-register slope")
    C.grid(axis="y", color=LINE, lw=0.5); C.set_axisbelow(True)
    tidy(C, "(c) per seed")

    hd = [Line2D([], [], color=col[k], marker=m, ls="-", ms=4.5, mec="white", mew=0.6, label=f"{k} circuit")
          for k, m in (("trained", "o"), ("frozen", "s"))]
    hd.append(Line2D([], [], color=INK, ls=(0, (3, 2)), label="noiseless"))
    fig.legend(handles=hd, loc="lower center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.25)
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=1000, bbox_inches="tight", pad_inches=0.05)

    allw = np.concatenate([ws["trained"].ravel(), ws["frozen"].ravel()])
    spread = np.concatenate([ws[k].max(0) - ws[k].min(0) for k in ws])
    mac = ["% generated by make_hw_seeds_figure.py; do not edit\n",
           macro("hwpooledslopetrained", f"{pooled['trained'][0]:.2f}"), macro("hwpooledslopefrozen", f"{pooled['frozen'][0]:.2f}"),
           macro("hwpooledcorrtrained", f"{pooled['trained'][1]:.2f}"), macro("hwpooledcorrfrozen", f"{pooled['frozen'][1]:.2f}"),
           macro("hwwireslopemin", f"{allw.min():.2f}"), macro("hwwireslopemax", f"{allw.max():.2f}"),
           macro("hwwireseedspreadmax", f"{spread.max():.2f}"),
           macro("hwseedsphys", ", ".join(str(q) for q in phys.pop()))]
    open(f"{a.out}_numbers.tex", "w").write("".join(mac))
    print("".join(mac))


if __name__ == "__main__":
    main()
