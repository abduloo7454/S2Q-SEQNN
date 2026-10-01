#!/usr/bin/env python
"""Fig. 2: trainable parameters vs test accuracy (mean, seed SD), regenerated from the ten-seed run log.

    python make_param_figure.py --runs evidence/panther/all_runs.csv --out fig_param_accuracy

Uses the same campaigns as Table III (main, classical, seqnn, qccnn) so the two agree by construction.
ViT (~16k parameters) is off the shared axis and is annotated at the right edge instead of plotted.
"""
import argparse, csv, importlib.util, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_spec = importlib.util.spec_from_file_location("bt", os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_tables.py"))
bt = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(bt)

INK, MUTED, LINE = "#0b0b0b", "#52514e", "#e6e5e2"
# marker, colour, size, zorder
STYLE = {
    "S$^{2}$Q-SEQNN-Adaptive": ("*", "#4a3aa7", 13, 6),
    "S$^{2}$Q-SEQNN-Shared":   ("P", "#eb6834", 9, 6),
    "S$^{2}$Q-SEQNN-Both":     ("*", "#4a3aa7", 13, 6),
    "SEQNN ($w{=}16$)": ("^", "#2a78d6", 8, 5),
    "SEQNN ($w{=}10$)":  ("v", "#2a78d6", 8, 5),
    "QC-CNN":       ("D", "#1baf7a", 6, 5),
    "DenseNet":     ("o", "#7a7a76", 6, 4),
    "VGG":          ("s", "#7a7a76", 6, 4),
    "ResNet":       ("p", "#7a7a76", 7, 4),
    "EfficientNet": ("h", "#7a7a76", 7, 4),
    "Swin":         ("X", "#7a7a76", 7, 4),
    "ViT":          ("x", "#7a7a76", 7, 4),
}
ORDER = list(STYLE)


def entries(by, ds):
    out = []
    for (camp, c) in by:
        if camp in ("main", "classical", "seqnn", "qccnn") and (c.startswith(ds) or c.endswith("_" + ds)):
            r0 = next(iter(by[(camp, c)].values())); d = bt.acc(by, (camp, c))
            if camp == "main":
                nm = f"S$^{{2}}$Q-SEQNN-{r0['variant'].capitalize()}"
            elif camp == "seqnn":
                nm = "SEQNN ($w{=}16$)" if "w16" in c else "SEQNN ($w{=}10$)"
            elif camp == "qccnn":
                nm = "QC-CNN"
            else:
                nm = r0["variant"].capitalize().replace("Densenet", "DenseNet").replace("Efficientnet", "EfficientNet").replace("Resnet", "ResNet").replace("Vgg", "VGG").replace("Vit", "ViT")
            m, s, n = bt.ms(d)
            out.append((nm, int(float(r0["total_params"])), m, s, n))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", required=True); ap.add_argument("--out", default="fig_param_accuracy")
    a = ap.parse_args()
    by = bt.load_runs(a.runs)
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "axes.edgecolor": "black", "axes.linewidth": 1.3, "xtick.color": "black", "ytick.color": "black",
                         "xtick.major.width": 1.1, "ytick.major.width": 1.1,
                         "text.color": INK, "axes.labelcolor": INK, "xtick.major.size": 3, "ytick.major.size": 3})
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.55))
    seen = {}
    for ax, ds in zip(axes, ("overhead", "sat6", "so2sat")):
        rows = entries(by, ds)
        xmax = 3700
        vit = [r for r in rows if r[0] == "ViT"]
        for nm, p, m, s, n in rows:
            mk, col, ms_, z = STYLE[nm]
            if nm == "ViT":
                continue
            ax.errorbar(p, m, yerr=s, fmt="none", ecolor=col, elinewidth=0.8, capsize=0, alpha=0.7, zorder=z - 1)
            ax.plot(p, m, marker=mk, color=col, markersize=ms_, linestyle="", markeredgecolor="white", markeredgewidth=0.6, zorder=z)
            seen[nm] = (mk, col, ms_)
        ys = [r[2] for r in rows if r[0] != "ViT"]; sd = [r[3] for r in rows if r[0] != "ViT"]
        lo, hi = min(y - e for y, e in zip(ys, sd)), max(y + e for y, e in zip(ys, sd))
        ax.set_ylim(lo - 2, hi + 2); ax.set_xlim(500, xmax)
        if vit:
            nm, p, m, s, n = vit[0]
            ax.annotate(f"ViT: {p/1000:.1f}k params,\n{m:.1f}$\\pm${s:.1f}%  $\\rightarrow$", xy=(xmax - 30, lo - 1), ha="right", va="bottom", fontsize=7, color="black")
        ax.set_title(bt.DS_NAME[ds], fontsize=9, loc="left", pad=5)
        ax.set_xlabel("trainable parameters"); ax.grid(color=LINE, linewidth=0.6, zorder=0); ax.set_axisbelow(True)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.set_xticks([1000, 2000, 3000])
    axes[0].set_ylabel("test accuracy (%)")
    handles = [Line2D([], [], marker=STYLE[n][0], color=STYLE[n][1], markersize=STYLE[n][2] * 0.8, linestyle="", markeredgecolor="white", markeredgewidth=0.5,
                      label=n.replace("S$^{2}$Q-SEQNN-Both", "S$^{2}$Q-SEQNN (SAT-6: Both)"))
               for n in ORDER if n in seen and n != "S$^{2}$Q-SEQNN-Both"]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02), handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=1.6)
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=1000, bbox_inches="tight")
    for ds in ("overhead", "sat6", "so2sat"):
        print(ds, sorted(entries(by, ds), key=lambda r: -r[2]))


if __name__ == "__main__":
    main()
