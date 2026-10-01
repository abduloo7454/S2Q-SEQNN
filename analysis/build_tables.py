#!/usr/bin/env python
"""
Generate every table and inline number of the S2Q-SEQNN manuscript from the
raw run log, so that nothing in the paper is typed by hand.

    python build_tables.py --runs results/all_runs.csv \
        --tangent-dir results/tangent --out tables/

Writes tables/*.tex (one \\begin{tabular} block each, to be \\input) and
tables/numbers.tex (\\newcommand macros for inline use). Statistics:
mean +- sample SD over seeds; paired tests are seed-paired t-tests with a
95% CI on the mean difference, Cohen's d_z, a Wilcoxon signed-rank test, and
Holm correction over the whole family of ablation tests.
"""

import argparse
import csv
import glob
import math
import os
import re
from collections import defaultdict

import numpy as np

try:
    from scipy import stats
except ImportError:  # pragma: no cover
    stats = None

DS_NAME = {"overhead": "Overhead-MNIST", "sat6": "SAT-6", "so2sat": "So2Sat LCZ42"}
FULL = {"overhead": "overhead_adaptive", "sat6": "sat6", "so2sat": "so2sat_adaptive"}
ABL_ORDER = [("classical_only", "Classical only"),
             ("mlp_replace", "Trained MLP"),
             ("frozen_mlp", "Frozen MLP"),
             ("frozen_rff", "Frozen RFF"),
             ("frozen_quantum", "Frozen circuit"),
             ("no_iqp", "No IQP"),
             ("no_reupload", "No re-upload"),
             ("fixed_z", "Fixed $Z$"),
             ("quantum_only", "No fusion")]
GATES = ["ZZ", "CZ", "CX", "XY", "CRZ", "CRY", "XX", "YY"]
BASES = ["X", "Y", "Z", "XYZ"]


# ----------------------------------------------------------------- helpers
def load_runs(path):
    by = defaultdict(dict)   # (campaign, config) -> seed -> row
    for r in csv.DictReader(open(path)):
        by[(r["campaign"], r["config"])][int(r["seed"])] = r
    return by


def acc(by, key, field="test_acc"):
    return {s: float(r[field]) for s, r in by[key].items()} if key in by else {}


def ms(d):
    v = np.array(list(d.values()), float)
    return (v.mean(), v.std(ddof=1) if v.size > 1 else float("nan"), v.size)


def fmt(d, bold=False):
    if not d:
        return "--"
    m, s, n = ms(d)
    t = f"{m:.2f}$\\pm${s:.2f}"
    return f"\\textbf{{{t}}}" if bold else t


def paired(a, b):
    """a, b: seed->acc. Returns dict with delta=(a-b) stats."""
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds])
    n = len(d)
    out = dict(n=n, delta=d.mean(), sd=d.std(ddof=1) if n > 1 else float("nan"))
    se = out["sd"] / math.sqrt(n) if n > 1 else float("nan")
    if stats is not None and n > 1:
        tcrit = stats.t.ppf(0.975, n - 1)
        out["ci_lo"], out["ci_hi"] = d.mean() - tcrit * se, d.mean() + tcrit * se
        t = d.mean() / se if se > 0 else float("nan")
        out["t"] = t
        out["p_t"] = float(2 * (1 - stats.t.cdf(abs(t), n - 1))) if np.isfinite(t) else float("nan")
        try:
            out["p_w"] = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        except Exception:
            out["p_w"] = float("nan")
    else:
        out.update(ci_lo=float("nan"), ci_hi=float("nan"), t=float("nan"), p_t=float("nan"), p_w=float("nan"))
    out["dz"] = d.mean() / out["sd"] if out["sd"] else float("nan")
    return out


def holm(pvals):
    p = np.array(pvals, float)
    idx = np.argsort(p)
    adj = np.empty_like(p)
    prev = 0.0
    m = len(p)
    for rank, i in enumerate(idx):
        v = (m - rank) * p[i]
        prev = max(prev, v)
        adj[i] = min(prev, 1.0)
    return adj


def dead_params(gate, n_qubits, n_reup=2):
    n = int(n_qubits)
    kernels = ((n // 2) + (n - 1) // 2) * n_reup
    return 2 * kernels if gate in ("CX", "CZ") else 0


def pnum(p):
    if not np.isfinite(p):
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def macro(name, value):
    name = re.sub(r"[^A-Za-z]", "", name)
    return f"\\providecommand{{\\{name}}}{{}}\\renewcommand{{\\{name}}}{{{value}}}\n"


# ----------------------------------------------------------------- tables
def table_main(by, macros):
    rows = []
    for ds in ("overhead", "sat6", "so2sat"):
        cfgs = [c for (camp, c) in by if camp == "main" and c.startswith(ds)]
        for c in sorted(cfgs):
            r0 = next(iter(by[("main", c)].values()))
            gate, basis, nq = r0["gate"], r0["basis"].upper(), r0["n_qubits"]
            tot = int(float(r0["total_params"]))
            dead = dead_params(gate, nq)
            variant = {"adaptive": "Adaptive", "shared": "Shared", "both": "Both"}[r0["variant"]]
            te = acc(by, ("main", c)); va = acc(by, ("main", c), "val_acc")
            rows.append(f"{DS_NAME[ds]} & {variant} & {r0['frontend'].capitalize()} & {gate}/{basis} & {nq} & "
                        f"{tot} & {tot - dead} & {fmt(te)} \\\\")
            m, s, n = ms(te)
            macros.append(macro(f"acc{ds}{r0['variant']}", f"{m:.2f}"))
            macros.append(macro(f"sd{ds}{r0['variant']}", f"{s:.2f}"))
            macros.append(macro(f"params{ds}{r0['variant']}", str(tot)))
            macros.append(macro(f"liveparams{ds}{r0['variant']}", str(tot - dead)))
            macros.append(macro(f"deadparams{ds}{r0['variant']}", str(dead)))
    head = ("\\begin{tabular}{llllrrrc}\n\\toprule\nDataset & Variant & Frontend & Gate/Basis & Qubits & "
            "Params & Live & Test (\\%) \\\\\n\\midrule\n")
    return head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"


def table_isolation(by, macros):
    present = [(a, lab) for a, lab in ABL_ORDER if any(k[1].endswith("__" + a) for k in by)]
    cols = "l" + "c" * (1 + len(present))
    head = (f"\\begin{{tabular}}{{{cols}}}\n\\toprule\nDataset & Full model & "
            + " & ".join(lab for _, lab in present) + " \\\\\n\\midrule\n")
    rows, tests = [], []
    for ds in ("overhead", "sat6", "so2sat"):
        full = acc(by, ("main", FULL[ds]))
        cells = [fmt(full)]
        for a, lab in present:
            camp = "fusion" if a == "quantum_only" else "isolation"
            key = (camp, f"{FULL[ds]}__{a}")
            d = acc(by, key)
            cells.append(fmt(d))
            if d:
                tests.append((ds, a, lab, paired(full, d)))
        rows.append(f"{DS_NAME[ds]} & " + " & ".join(cells) + " \\\\")
    tab = head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    return tab, tests


IFACE = {"fixed_z", "quantum_only"}   # controls that change the classical interface, not the quantum branch


def _m(x):
    """signed number in math mode, so negatives get a true minus sign"""
    return f"${x:+.2f}$"


def table_significance(tests, macros):
    """Main-text table (single column): Delta and Holm-corrected p per dataset.
    Returns (matrix, full); `full` is the supplementary table with CI, d_z, p_t, p_W."""
    ph = holm([t["p_t"] for _, _, _, t in tests])
    cell = {(ds, a): (lab, t, p) for (ds, a, lab, t), p in zip(tests, ph)}
    dss = [d for d in DS_NAME if any(k[0] == d for k in cell)]
    order = []
    for _, a, lab, _ in tests:
        if (a, lab) not in order:
            order.append((a, lab))
    # ---- matrix for the main text
    rows = []
    for a, lab in order:
        if a in IFACE and "\\midrule" not in rows:
            rows.append("\\midrule")
        cs = []
        for ds in dss:
            _, t, p = cell[(ds, a)]
            d, q = _m(t["delta"]), pnum(p)
            if p < 0.05:
                d, q = f"$\\mathbf{{{t['delta']:+.2f}}}$", f"\\textbf{{{q}}}"
            cs += [d, q]
        rows.append(f"{lab} & " + " & ".join(cs) + " \\\\")
    n = len(dss)
    head = (f"\\begin{{tabular}}{{@{{}}l{'rr' * n}@{{}}}}\n\\toprule\n & "
            + " & ".join(f"\\multicolumn{{2}}{{c}}{{{DS_NAME[d]}}}" for d in dss) + " \\\\\n"
            + "".join(f"\\cmidrule({'l' if i == n - 1 else 'lr'}){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(n))
            + "\nControl" + " & $\\Delta$ & $p_{\\mathrm{Holm}}$" * n + " \\\\\n\\midrule\n")
    matrix = head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
    # ---- full statistics for the supplement
    rows, last = [], None
    for (ds, a, lab, t), p in zip(tests, ph):
        if ds != last:
            rows.append(("\\midrule\n" if last is None else "\\addlinespace[3pt]\n")
                        + f"\\multicolumn{{8}}{{@{{}}l}}{{\\textbf{{{DS_NAME[ds]}}}}} \\\\")
            last = ds
        q = pnum(p)
        if p < 0.05:
            q = f"\\textbf{{{q}}}"
        name = lab + ("$^{\\dagger}$" if a in IFACE else "")
        rows.append(f"{name} & {_m(t['delta'])} & $[{t['ci_lo']:.2f}$ & ${t['ci_hi']:.2f}]$ & {_m(t['dz'])} & "
                    f"{pnum(t['p_t'])} & {pnum(t['p_w'])} & {q} \\\\")
    full = ("\\begin{tabular}{@{}lr@{\\hspace{6pt}}r@{,\\;}lrrrr@{}}\n\\toprule\n"
            "Control & $\\Delta$ (pp) & \\multicolumn{2}{c}{95\\% CI} & $d_z$ & $p_t$ & $p_W$ & $p_{\\mathrm{Holm}}$ \\\\\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    for (ds, a, lab, t), p in zip(tests, ph):
        macros.append(macro(f"delta{ds}{a}", f"{t['delta']:+.2f}"))
        macros.append(macro(f"pholm{ds}{a}", pnum(p).replace("$<$", "<")))
    macros.append(macro("nsigtests", str(int(np.sum(ph < 0.05)))))
    macros.append(macro("ntests", str(len(tests))))
    return matrix, full


def table_sweep(by, campaign, prefix, options, label, macros):
    """gate / basis sweeps: rows = configs, cols = options, default from main."""
    head = (f"\\begin{{tabular}}{{l{'c' * len(options)}}}\n\\toprule\nConfiguration & "
            + " & ".join(options) + " \\\\\n\\midrule\n")
    rows = []
    for (camp, c) in sorted(k for k in by if k[0] == "main"):
        r0 = next(iter(by[(camp, c)].values()))
        default = r0["gate"] if campaign == "gate" else r0["basis"].upper()
        ds = c.split("_")[0]
        vals = {}
        for opt in options:
            key = ("main", c) if opt == default else (campaign, f"{c}__{prefix}{opt}")
            vals[opt] = acc(by, key)
        best = max((o for o in options if vals[o]), key=lambda o: ms(vals[o])[0])
        cells = [fmt(vals[o], bold=(o == best)) + ("$^{\\dagger}$" if o == default else "") for o in options]
        name = f"{DS_NAME[ds]} {r0['variant'].capitalize()}"
        rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
        # spread
        means = [ms(vals[o])[0] for o in options if vals[o]]
        macros.append(macro(f"{campaign}spread{ds}{r0['variant']}", f"{max(means) - min(means):.2f}"))
    note = f"\\multicolumn{{{len(options) + 1}}}{{l}}{{\\footnotesize $^{{\\dagger}}$ configuration selected by the diagnostic pipeline (main-campaign run); bold marks the best mean.}} \\\\\n"
    return head + "\n".join(rows) + "\n\\midrule\n" + note + "\\bottomrule\n\\end{tabular}\n"


def table_baselines(by, macros):
    """Methods x datasets. One row per method, test accuracy per dataset, and the
    parameter range across the three datasets. Bold marks the best mean per dataset."""
    dss = ("overhead", "sat6", "so2sat")
    cell = defaultdict(dict)   # method -> ds -> (runs dict, params)
    kind = {}
    for (camp, c) in by:
        if camp not in ("main", "classical", "seqnn", "qccnn"):
            continue
        ds = next((d for d in dss if c.startswith(d) or c.endswith("_" + d)), None)
        if ds is None:
            continue
        r0 = next(iter(by[(camp, c)].values()))
        p = int(float(r0["total_params"]))
        d = acc(by, (camp, c))
        if camp == "main":
            v = r0["variant"]
            names = ["S$^{2}$Q-SEQNN-Adaptive", "S$^{2}$Q-SEQNN-Shared"] if v == "both" \
                else [f"S$^{{2}}$Q-SEQNN-{v.capitalize()}"]
        elif camp == "seqnn":
            names = ["SEQNN ($w{=}16$)" if "w16" in c else "SEQNN ($w{=}10$)"]
        elif camp == "qccnn":
            names = ["QC-CNN"]
        else:
            names = [r0["variant"].capitalize().replace("Densenet", "DenseNet").replace("Efficientnet", "EfficientNet")
                     .replace("Resnet", "ResNet").replace("Vgg", "VGG").replace("Vit", "ViT")]
        for nm in names:
            cell[nm][ds] = (d, p, camp == "main" and r0["variant"] == "both")
            kind[nm] = camp
        for nm in (["S$^{2}$Q-SEQNN-Both"] if camp == "main" and r0["variant"] == "both" else names):
            slug = re.sub(r"[^A-Za-z]", "", nm)
            if camp == "seqnn":                       # LaTeX names cannot hold the digits of w
                slug += "sixteen" if "w16" in c else "ten"
            macros.append(macro(f"base{ds}{slug}", f"{ms(d)[0]:.2f}"))
    best = {ds: max(ms(v[ds][0])[0] for v in cell.values() if ds in v) for ds in dss}
    groups = [("main",), ("seqnn", "qccnn"), ("classical",)]
    rows = []
    for gi, g in enumerate(groups):
        members = [nm for nm in cell if kind[nm] in g]
        members.sort(key=lambda nm: (0 if "Adaptive" in nm else 1 if "Shared" in nm else 2,
                                     -np.mean([ms(cell[nm][d][0])[0] for d in cell[nm]])))
        if gi:
            rows.append("\\midrule")
        for nm in members:
            ps = [cell[nm][d][1] for d in dss if d in cell[nm]]
            prange = f"{min(ps)}" if min(ps) == max(ps) else f"{min(ps)}--{max(ps)}"
            cs = []
            for d in dss:
                if d not in cell[nm]:
                    cs.append("--"); continue
                runs, _, shared_model = cell[nm][d]
                t = fmt(runs, bold=abs(ms(runs)[0] - best[d]) < 1e-9)
                cs.append(t + ("$^{\\ddagger}$" if shared_model else ""))
            rows.append(f"{nm} & {prange} & " + " & ".join(cs) + " \\\\")
    head = ("\\begin{tabular}{@{}lrccc@{}}\n\\toprule\nMethod & Params & "
            + " & ".join(DS_NAME[d] for d in dss) + " \\\\\n\\midrule\n")
    return head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"


def table_schedule(by, macros):
    head = ("\\begin{tabular}{lcc}\n\\toprule\nConfiguration & Two-stage ($\\rho_q{=}0.1$) & "
            "Shared rate ($\\rho_q{=}1$) \\\\\n\\midrule\n")
    rows = []
    for (camp, c) in sorted(k for k in by if k[0] == "main"):
        r0 = next(iter(by[(camp, c)].values())); ds = c.split("_")[0]
        full = acc(by, ("main", c)); sh = acc(by, ("schedule", f"{c}__sharedLR"))
        rows.append(f"{DS_NAME[ds]} {r0['variant'].capitalize()} & {fmt(full)} & {fmt(sh)} \\\\")
        if sh:
            t = paired(full, sh)
            macros.append(macro(f"sched{ds}{r0['variant']}", f"{t['delta']:+.2f}"))
    return head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"


def table_sensitivity(by, macros):
    head = ("\\begin{tabular}{lccccccc}\n\\toprule\nConfiguration & Default & $n_q{=}10$ & $n_q{=}12$ & $R{=}1$ & $R{=}3$ & "
            "$\\varepsilon{=}0$ & $\\varepsilon{=}0.10$ \\\\\n\\midrule\n")
    rows = []
    for (camp, c) in sorted(k for k in by if k[0] == "main"):
        r0 = next(iter(by[(camp, c)].values())); ds = c.split("_")[0]
        cells = [fmt(acc(by, ("main", c)))]
        for camp2, suf in (("sens_nq", "nq10"), ("sens_nq", "nq12"), ("sens_R", "R1"), ("sens_R", "R3"),
                           ("sens_eps", "eps0.00"), ("sens_eps", "eps0.10")):
            cells.append(fmt(acc(by, (camp2, f"{c}__{suf}"))))
        rows.append(f"{DS_NAME[ds]} {r0['variant'].capitalize()} & " + " & ".join(cells) + " \\\\")
    return head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"


def table_tangent(tangent_dir, macros):
    rows_by = defaultdict(list)
    for f in glob.glob(os.path.join(tangent_dir, "tangent_*.csv")):
        for r in csv.DictReader(open(f)):
            rows_by[(r["config"], r["campaign"])].append(r)
    if not rows_by:
        return "", ""

    def col(rs, k):
        v = np.array([float(r[k]) for r in rs if r.get(k) not in (None, "", "nan")])
        return v

    def cell(v, fm="{:.3f}"):
        if v.size == 0:
            return "--"
        return (fm + "$\\pm$" + fm).format(v.mean(), v.std(ddof=1) if v.size > 1 else 0.0)

    head_full = ("\\begin{tabular}{llrrrccccc}\n\\toprule\nConfiguration & $\\rho_q$ & $L$ & $d_{\\mathrm{eff}}$ & pred. & "
            "$\\|\\Delta\\theta\\|/\\|\\theta_0\\|$ & $A_0$ & $A_f$ & $\\|\\nabla C\\|^2$ & $\\|\\partial\\mathcal{L}/\\partial\\theta_q\\|^2$ \\\\\n\\midrule\n")
    rows, rows_full = [], []
    import importlib.util
    spec = importlib.util.spec_from_file_location("an", os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze_tangent.py"))
    an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)
    for cfg in ("overhead_adaptive", "sat6", "so2sat_adaptive"):
        ds = cfg.split("_")[0]
        for camp, rho in (("main", "0.1"), ("schedule", "1")):
            rs = rows_by.get((cfg, camp), [])
            if not rs:
                continue
            L = int(float(rs[0]["n_params"])); nq = rs[0]["n_qubits"]; gate = rs[0]["gate"]
            _, dp, dead, rym, xxm = an.deff_predicted(nq, rs[0].get("n_reup", 2), gate)
            deff = col(rs, "deff_final")
            rows_full.append(f"{DS_NAME[ds]} ({gate}, {nq} qb) & {rho} & {L} & {deff.mean():.0f} & {dp} & "
                        f"{cell(col(rs, 'disp_rel'))} & {cell(col(rs, 'A_init'))} & {cell(col(rs, 'A_final'))} & "
                        f"{cell(col(rs, 'grad2_final'), '{:.2f}')} & {cell(col(rs, 'task_grad2_final'), '{:.3f}')} \\\\")
            rows.append(f"{DS_NAME[ds]} ({gate}, {nq} qb) & {rho} & {L} & {deff.mean():.0f} & {dp} & "
                        f"{cell(col(rs, 'disp_rel'))} & {cell(col(rs, 'A_final'))} & "
                        f"{cell(col(rs, 'grad2_final'), '{:.2f}')} & {cell(col(rs, 'task_grad2_final'), '{:.3f}')} \\\\")
            tag = f"{ds}rho{'one' if rho == '1' else 'tenth'}"
            macros.append(macro(f"disp{tag}", f"{col(rs, 'disp_rel').mean():.3f}"))
            macros.append(macro(f"align{tag}", f"{col(rs, 'A_final').mean():.2f}"))
            macros.append(macro(f"deff{ds}", f"{deff.mean():.0f}"))
            macros.append(macro(f"deffpred{ds}", str(dp)))
            macros.append(macro(f"nseedstangent{tag}", str(len(rs))))
    head = ("\\begin{tabular}{llrrrcccc}\n\\toprule\nConfiguration & $\\rho_q$ & $L$ & $d_{\\mathrm{eff}}$ & pred. & "
            "$\\|\\Delta\\theta\\|/\\|\\theta_0\\|$ & $A_f$ & $\\|\\nabla C\\|^2$ & $\\|\\partial\\mathcal{L}/\\partial\\theta_q\\|^2$ \\\\\n\\midrule\n")
    tail = "\n\\bottomrule\n\\end{tabular}\n"
    return head + "\n".join(rows) + tail, head_full + "\n".join(rows_full) + tail



# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--tangent-dir", default="")
    ap.add_argument("--out", default="tables")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    by = load_runs(a.runs)
    macros = [f"% generated by build_tables.py from {os.path.basename(a.runs)}; do not edit\n"]
    n_runs = sum(len(v) for v in by.values())
    macros.append(macro("nruns", str(n_runs)))
    macros.append(macro("nconfigs", str(len(by))))

    out = {}
    out["tab_main.tex"] = table_main(by, macros)
    out["tab_isolation.tex"], tests = table_isolation(by, macros)
    out["tab_significance.tex"], out["tab_significance_full.tex"] = table_significance(tests, macros)
    out["tab_gate.tex"] = table_sweep(by, "gate", "gate", GATES, "gate", macros)
    out["tab_basis.tex"] = table_sweep(by, "basis", "basis", BASES, "basis", macros)
    out["tab_baselines.tex"] = table_baselines(by, macros)
    out["tab_schedule.tex"] = table_schedule(by, macros)
    # adaptive vs shared, seed-paired
    for ds in ("overhead", "so2sat"):
        t = paired(acc(by, ("main", f"{ds}_adaptive")), acc(by, ("main", f"{ds}_shared")))
        macros.append(macro(f"advsh{ds}delta", f"{t['delta']:+.2f}"))
        macros.append(macro(f"advsh{ds}cilo", f"{t['ci_lo']:+.2f}"))
        macros.append(macro(f"advsh{ds}cihi", f"{t['ci_hi']:+.2f}"))
        macros.append(macro(f"advsh{ds}p", pnum(t["p_t"]).replace("$<$", "<")))
        macros.append(macro(f"advsh{ds}dz", f"{t['dz']:+.2f}"))
    # margins of the best S2Q variant over reference baselines
    for ds in ("overhead", "sat6", "so2sat"):
        ours = max(ms(acc(by, ("main", c)))[0] for (camp, c) in by if camp == "main" and c.startswith(ds))
        for camp, cfg, tag in (("classical", f"densenet_{ds}", "densenet"), ("seqnn", f"seqnn_w10_{ds}", "seqnnmatched"),
                               ("seqnn", f"seqnn_w16_{ds}", "seqnnfaithful"), ("qccnn", f"qccnn_{ds}", "qccnn")):
            d = acc(by, (camp, cfg))
            if d:
                macros.append(macro(f"margin{ds}{tag}", f"{ours - ms(d)[0]:+.2f}"))
        macros.append(macro(f"best{ds}", f"{ours:.2f}"))
    out["tab_sensitivity.tex"] = table_sensitivity(by, macros)
    if a.tangent_dir:
        t, t_full = table_tangent(a.tangent_dir, macros)
        if t:
            out["tab_tangent.tex"] = t
            out["tab_tangent_full.tex"] = t_full
    for name, body in out.items():
        with open(os.path.join(a.out, name), "w") as fh:
            fh.write(body)
    with open(os.path.join(a.out, "numbers.tex"), "w") as fh:
        fh.writelines(macros)
    print("wrote", ", ".join(sorted(out)), "and numbers.tex", f"({len(macros)} macros) from {n_runs} runs")
    # console summary of the family of tests
    ph = holm([t["p_t"] for _, _, _, t in tests])
    print("\npaired tests (full - ablation), Holm over", len(tests), "tests:")
    for (ds, a_, lab, t), p in zip(tests, ph):
        print(f"  {ds:9s} {a_:16s} d={t['delta']:+.2f} [{t['ci_lo']:+.2f},{t['ci_hi']:+.2f}] p_holm={p:.3f}{' *' if p < .05 else ''}")


if __name__ == "__main__":
    main()
