#!/usr/bin/env python3
"""
Register-size figure and numbers for the TQE revision (So2Sat Adaptive, CRY, xyz read-out).

    python make_geom_figure.py --geom-dir results/geom \
        --scaling-stats out/scaling_stats.csv --out-dir out

Inputs
    geom/geom_nq*_seed*.csv   tangent diagnostics of trained checkpoints (s2q_geom.py --trained)
    geom/bp_gpu_*.csv         barren-plateau scan at random parameters (s2q_geom.py --bp-scan)
    scaling_stats.csv         paired accuracy tests per n_q (make_scaling.py)
Outputs
    fig_register.pdf/.png     (a) gradient scan, (b) alignment vs 2-design level, (c) accuracy delta
    geom_numbers.tex          macros for the text
    tab_geom.tex              supplement table, one row per n_q
    bp_fit.csv                fitted decay rates with bootstrap CIs
"""
import argparse, glob, math, os, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, LINE, BAND, GREY = "#0b0b0b", "#52514e", "#e6e5e2", "#f3f2ef", "#8c8b86"
MAIN_GATE = "CRY"


def macro(k, v):
    k = re.sub(r"[^A-Za-z]", "", k)
    return f"\\providecommand{{\\{k}}}{{}}\\renewcommand{{\\{k}}}{{{v}}}\n"


def sci(x, nd=1):
    m, e = f"{x:.{nd}e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def bp_tables(files, B=2000, seed=0):
    d = pd.concat([pd.read_csv(f) for f in files])
    d["g2"] = d.grad_global ** 2
    rng = np.random.default_rng(seed)
    per = d.groupby(["gate", "n"]).agg(local=("grad_local_mean2", "mean"), glob=("g2", "mean"),
                                        rows=("sample", "size")).reset_index()
    fits = []
    ps = d.groupby(["gate", "n", "sample"]).agg(loc=("grad_local_mean2", "mean"), glo=("g2", "mean")).reset_index()
    for gate, g in ps.groupby("gate"):
        ns = np.array(sorted(g.n.unique()), float)
        L = [g[g.n == n]["loc"].to_numpy() for n in ns]
        Gl = [g[g.n == n]["glo"].to_numpy() for n in ns]
        m = ns >= 10

        def stat(loc, glo):
            sg = np.polyfit(ns, np.log2(glo), 1)[0]
            sl = np.polyfit(ns[m], np.log2(loc[m] * ns[m]), 1)[0]
            sr = np.polyfit(ns[m], np.log2(loc[m]), 1)[0]
            return sg, sl, sr
        est = stat(np.array([x.mean() for x in L]), np.array([x.mean() for x in Gl]))
        boots = []
        for _ in range(B):
            idx = [rng.integers(0, len(x), len(x)) for x in L]
            boots.append(stat(np.array([x[i].mean() for x, i in zip(L, idx)]),
                              np.array([x[i].mean() for x, i in zip(Gl, idx)])))
        lo, hi = np.percentile(np.array(boots), [2.5, 97.5], axis=0)
        fits.append(dict(gate=gate, nmin=int(ns.min()), nmax=int(ns.max()), glob_slope=est[0], glob_lo=lo[0], glob_hi=hi[0],
                         nloc_slope=est[1], nloc_lo=lo[1], nloc_hi=hi[1], loc_slope=est[2], loc_lo=lo[2], loc_hi=hi[2]))
    return per, pd.DataFrame(fits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom-dir", required=True)
    ap.add_argument("--scaling-stats", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    G = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(a.geom_dir, "geom_nq*_seed*.csv")))])
    cnt = G.groupby("n_qubits").size()
    assert (cnt == 10).all(), f"need 10 seeds per n_q: {cnt.to_dict()}"
    NQg = sorted(G.n_qubits.unique())
    per, fits = bp_tables(sorted(glob.glob(os.path.join(a.geom_dir, "bp_*.csv"))), B=a.boot)
    fits.to_csv(os.path.join(a.out_dir, "bp_fit.csv"), index=False)
    S = pd.read_csv(a.scaling_stats)

    # ------------------------------------------------------------ figure
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.edgecolor": "black",
                         "axes.linewidth": 1.3, "xtick.color": "black", "ytick.color": "black", "text.color": INK,
                         "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 1.1,
                         "ytick.major.width": 1.1, "xtick.minor.width": 0.8, "ytick.minor.width": 0.8,
                         "axes.labelcolor": INK, "figure.facecolor": "white", "savefig.facecolor": "white"})
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.45), gridspec_kw=dict(wspace=0.4, width_ratios=[1.15, 1, 1]))
    nq_lo, nq_hi = int(S.nq.min()), int(S.nq.max())

    # (a) gradient scan
    A0 = ax[0]
    A0.axvspan(nq_lo - 0.5, nq_hi + 0.5, color=BAND, lw=0, zorder=0)
    for gate in sorted(per.gate.unique()):
        g = per[per.gate == gate].sort_values("n")
        main = gate == MAIN_GATE
        A0.plot(g.n, g.local, color=BLUE if main else GREY, lw=1.7 if main else 0.8, alpha=1 if main else 0.8,
                zorder=4 if main else 2)
        A0.plot(g.n, g.glob, color=INK if main else GREY, lw=1.5 if main else 0.8, ls="--", alpha=1 if main else 0.8,
                zorder=4 if main else 2)
    g = per[per.gate == MAIN_GATE].sort_values("n")
    n0, y0 = g.n.iloc[0], g.glob.iloc[0]
    nn = np.array([g.n.min(), g.n.max()])
    A0.plot(nn, y0 * 2.0 ** (-(nn - n0)), color=MUTED, lw=0.7, ls=":", zorder=1)
    A0.set_yscale("log")
    A0.set_xlabel("Qubits $n$")
    A0.set_ylabel(r"$\mathrm{E}_{\theta}[(\partial_k C)^2]$")
    A0.set_title("(a) Gradients at random $\\theta$", loc="left", fontsize=8, pad=4)
    hd0 = [Line2D([], [], color=BLUE, lw=1.7, label="local read-out (model)"),
           Line2D([], [], color=INK, lw=1.5, ls="--", label="global $Z^{\\otimes n}$"),
           Line2D([], [], color=MUTED, lw=0.8, ls=":", label="$2^{-n}$"),
           Line2D([], [], color=GREY, lw=0.8, label="CX, XX entanglers")]
    A0.legend(handles=hd0, loc="lower left", frameon=False, fontsize=6.8, handlelength=1.8, borderaxespad=0.1)
    A0.text((nq_lo + nq_hi) / 2, 0.97, "trained", transform=A0.get_xaxis_transform(), ha="center", va="top",
            fontsize=7, color="black")

    # (b) alignment vs 2-design level
    A1 = ax[1]
    agg = G.groupby("n_qubits").agg(Af=("A_final", "mean"), Afs=("A_final", "std"), Ai=("A_init", "mean"),
                                    Ais=("A_init", "std"), A2=("A_2design_final", "mean")).reset_index()
    A1.errorbar(agg.n_qubits - 0.2, agg.Ai, yerr=agg.Ais, fmt="o", color=BLUE, mfc="white", ms=3.8, mew=1,
                elinewidth=0.9, capsize=0, zorder=3)
    A1.errorbar(agg.n_qubits + 0.2, agg.Af, yerr=agg.Afs, fmt="o", color=BLUE, ms=3.8, mec="white", mew=0.5,
                elinewidth=0.9, capsize=0, zorder=4)
    A1.plot(agg.n_qubits, agg.A2, color=MUTED, ls="--", lw=1, marker="s", ms=3, mfc="white", zorder=3)
    A1.set_yscale("log")
    A1.set_xticks(NQg)
    A1.set_xlabel("Qubits $n_q$")
    A1.set_ylabel("Alignment $A$")
    A1.set_title("(b) Trained checkpoints", loc="left", fontsize=8, pad=4)
    A1.text(agg.n_qubits.iloc[0], agg.A2.min() * 1.2, "2-design level", color="black", ha="left", va="bottom", fontsize=7)
    A1.text(agg.n_qubits.iloc[-1], agg.Af.iloc[-1] * 0.62, "init (open), trained", color=BLUE, ha="right", va="top", fontsize=7)
    A1.set_ylim(agg.A2.min() / 2.5, 1.6)

    # (c) accuracy delta
    A2 = ax[2]
    A2.axhspan(-0.5, 0.5, color=BAND, lw=0, zorder=0)
    A2.axhline(0, color=INK, lw=0.8, zorder=1)
    off = {"frozen_quantum": -0.12, "frozen_rff": 0.0, "classical_only": 0.12}
    col = {"frozen_quantum": ORANGE, "frozen_rff": AQUA, "classical_only": YELLOW}
    mk = {"frozen_quantum": "s", "frozen_rff": "D", "classical_only": "^"}
    lab = {"frozen_quantum": "Frozen circuit", "frozen_rff": "Frozen RFF", "classical_only": "No circuit"}
    for arm in ("frozen_quantum", "frozen_rff", "classical_only"):
        g = S[S.arm == arm].sort_values("nq")
        x = g.nq + off[arm]
        A2.plot(x, g.delta, color=col[arm], lw=1.1, zorder=3)
        A2.errorbar(x, g.delta, yerr=[g.delta - g.ci_lo, g.ci_hi - g.delta], fmt=mk[arm], color=col[arm], ms=3.6,
                    mec="white", mew=0.5, elinewidth=0.9, capsize=0, zorder=4)
    A2.set_xticks(sorted(S.nq.unique()))
    A2.set_xlabel("Qubits $n_q$")
    A2.set_ylabel(r"$\Delta$ = trained $-$ control (pp)")
    A2.set_title("(c) Accuracy, 95% CI", loc="left", fontsize=8, pad=4)
    hd = [Line2D([], [], color=col[k], marker=mk[k], ms=3.6, mec="white", mew=0.5, lw=1.1, label=lab[k])
          for k in ("frozen_quantum", "frozen_rff", "classical_only")]
    A2.legend(handles=hd, loc="upper left", frameon=False, fontsize=7, handlelength=1.6, borderaxespad=0.2,
              ncol=1, bbox_to_anchor=(0.0, 1.0))
    ylo, yhi = A2.get_ylim(); A2.set_ylim(min(ylo, -0.8), max(yhi, 1.75))
    for axx in ax:
        for s_ in ("top", "right"):
            axx.spines[s_].set_visible(False)
        axx.grid(axis="y", color=LINE, lw=0.5, zorder=0)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.9, bottom=0.18)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out_dir, f"fig_register.{ext}"), dpi=1000)

    # ------------------------------------------------------------ macros
    mac = ["% generated by make_geom_figure.py; do not edit\n"]
    gm = per[per.gate == MAIN_GATE].sort_values("n")
    f_main = fits[fits.gate == MAIN_GATE].iloc[0]
    nloc = per.assign(nl=per.n * per.local)
    nloc10 = nloc[nloc.n >= 10]
    mac += [macro("bpnmin", str(int(per.n.min()))), macro("bpnmax", str(int(per.n.max()))),
            macro("bpngates", str(per.gate.nunique())),
            macro("bpglobslope", f"{f_main.glob_slope:.2f}"),
            macro("bpglobslopelo", f"{fits.glob_lo.min():.2f}"), macro("bpglobslopehi", f"{fits.glob_hi.max():.2f}"),
            macro("bpglobfall", f"{gm.glob.iloc[0] / gm.glob.iloc[-1]:.0e}".replace("e+0", "e").replace("e+", "e")),
            macro("bpglobfallorders", str(int(math.floor(math.log10(gm.glob.iloc[0] / gm.glob.iloc[-1]))))),
            macro("bplocfall", f"{gm.local.iloc[0] / gm.local.iloc[-1]:.0f}"),
            macro("bpnlocslope", f"{f_main.nloc_slope:+.3f}"),
            macro("bpnlocslopelo", f"{fits.nloc_lo.min():+.3f}"), macro("bpnlocslopehi", f"{fits.nloc_hi.max():+.3f}"),
            macro("bpnlocmin", f"{nloc10.nl.min():.3f}"), macro("bpnlocmax", f"{nloc10.nl.max():.3f}"),
            macro("bplocatmax", "$" + sci(gm.local.iloc[-1]) + "$"), macro("bpglobatmax", "$" + sci(gm.glob.iloc[-1]) + "$")]
    agg["ratio"] = agg.Af / agg.A2
    dd = G.groupby("n_qubits").agg(L=("L", "first"), deff=("deff_final", "mean"), deffsd=("deff_final", "std"),
                                   g2=("grad2_final", "mean"), g2s=("grad2_final", "std"),
                                   tg=("task_grad2_final", "mean"), tgs=("task_grad2_final", "std"),
                                   disp=("disp_rel", "mean"), disps=("disp_rel", "std"),
                                   chi=("chi2_final", "mean")).reset_index()
    assert (dd.deffsd.fillna(0) == 0).all(), "d_eff varies across seeds"
    rule = (dd.L - dd.deff == 2 * (dd.n_qubits - 2)).all()
    assert rule, "d_eff no longer equals L - 2(n_q - 2); the text states this rule"
    mac += [macro("geomnqmin", str(int(min(NQg)))), macro("geomnqmax", str(int(max(NQg)))),
            macro("geomnnq", str(len(NQg))),
            macro("geomAmin", f"{min(agg.Af.min(), agg.Ai.min()):.2f}"), macro("geomAmax", f"{max(agg.Af.max(), agg.Ai.max()):.2f}"),
            macro("geomAtwodmax", f"{agg.A2.iloc[-1]:.4f}"), macro("geomAtwodmin", f"{agg.A2.iloc[0]:.3f}"),
            macro("geomratiomin", f"{agg.ratio.iloc[0]:.0f}"), macro("geomratiomax", f"{agg.ratio.iloc[-1]:.0f}"),
            macro("geomgradmin", f"{dd.g2.min():.2f}"), macro("geomgradmax", f"{dd.g2.max():.2f}"),
            macro("geomdispmin", f"{dd.disp.min():.3f}"), macro("geomdispmax", f"{dd.disp.max():.3f}"),
            macro("geomdeffrule", "true" if rule else "false"),
            macro("geomdeffmax", str(int(dd.deff.iloc[-1]))), macro("geomLmax", str(int(dd.L.iloc[-1])))]
    open(os.path.join(a.out_dir, "geom_numbers.tex"), "w").write("".join(mac))

    # ------------------------------------------------------------ supplement table
    rows = []
    for _, r in dd.merge(agg, on="n_qubits").iterrows():
        rows.append(f"{int(r.n_qubits)} & {int(r.L)} & {int(r.deff)} & {r.Ai:.3f}$\\pm${r.Ais:.3f} & "
                    f"{r.Af:.3f}$\\pm${r.Afs:.3f} & {r.A2:.4f} & {r.g2:.2f}$\\pm${r.g2s:.2f} & "
                    f"{r.tg:.3f}$\\pm${r.tgs:.3f} & {r.disp:.3f}$\\pm${r.disps:.3f} \\\\")
    tab = ("\\begin{tabular}{@{}rrrccccccc@{}}\n\\toprule\n"
           "$n_q$ & $L$ & $d_{\\mathrm{eff}}$ & $A_0$ & $A_f$ & $d_{\\mathrm{eff}}/[2(D{-}1)]$ & $\\|\\nabla C\\|^2$ & "
           "$\\|\\partial\\mathcal{L}/\\partial\\theta_q\\|^2$ & $\\|\\Delta\\theta\\|/\\|\\theta_0\\|$ \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(a.out_dir, "tab_geom.tex"), "w").write(tab)
    fr = [f"{r.gate} & {r.nmin}--{r.nmax} & ${r.glob_slope:+.2f}$ & $[{r.glob_lo:+.2f},\\,{r.glob_hi:+.2f}]$ & "
          f"${r.loc_slope:+.3f}$ & ${r.nloc_slope:+.3f}$ & $[{r.nloc_lo:+.3f},\\,{r.nloc_hi:+.3f}]$ \\\\" for r in fits.itertuples()]
    ftab = ("\\begin{tabular}{@{}lcrcrrc@{}}\n\\toprule\n"
            "Entangler & $n$ & \\multicolumn{2}{c}{Global $Z^{\\otimes n}$} & Local & \\multicolumn{2}{c}{$n\\times$ local ($n\\ge10$)} \\\\\n"
            "\\cmidrule(lr){3-4}\\cmidrule(l){6-7}\n & & slope & 95\\% CI & slope & slope & 95\\% CI \\\\\n\\midrule\n"
            + "\n".join(fr) + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(a.out_dir, "tab_bpfit.tex"), "w").write(ftab)
    pd.set_option("display.width", 200)
    print(fits.round(3).to_string(index=False)); print(dd.round(4).to_string(index=False)); print("".join(mac))


if __name__ == "__main__":
    main()
