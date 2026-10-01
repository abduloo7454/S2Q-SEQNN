#!/usr/bin/env python
"""
Aggregate the tangent-space diagnostics CSVs into one per-config table.

    python analyze_tangent.py results/tangent/*.csv --out tangent_summary.csv

Every number is a mean +- sd over the seeds present in the input files; the
input rows come straight from s2q_tangent_diagnostics.py, nothing is typed.
"""

import argparse
import csv
import glob
import math
import sys
from collections import defaultdict

import numpy as np


def read_rows(paths):
    rows = []
    for p in paths:
        for f in glob.glob(p):
            with open(f) as fh:
                for r in csv.DictReader(fh):
                    r["_file"] = f
                    rows.append(r)
    return rows


def deff_predicted(n_qubits, n_reup, gate):
    """Rank of the QCNN block predicted from gate algebra alone.

    L = 8 params x (n-1) pairs x R rounds. Removed:
      * CX / CZ take no angle: w[2], w[5] of every kernel are disconnected;
      * an odd-sweep kernel starts with RY on both wires right after the even
        sweep ended with RY on the same wires, so those pairs merge (2 per
        odd kernel);
      * IsingXX commutes with the RX layer between the two entanglers, so a
        kernel's two XX angles add into one (1 per kernel). IsingZZ likewise
        commutes with nothing in between (RX does not), so no merge there.
    """
    n, R = int(n_qubits), int(n_reup)
    even = (n // 2)
    odd = (n - 1) // 2
    kernels = (even + odd) * R
    L = 8 * (n - 1) * R
    dead = 2 * kernels if gate in ("CX", "CZ", "SWAP", "ISWAP") else 0
    ry_merge = 2 * odd * R
    xx_merge = kernels if gate == "XX" else 0
    return L, L - dead - ry_merge - xx_merge, dead, ry_merge, xx_merge


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--out", default="tangent_summary.csv")
    args = ap.parse_args()
    rows = read_rows(args.csvs)
    if not rows:
        sys.exit("no rows")

    groups = defaultdict(list)
    for r in rows:
        key = (r["config"], r["campaign"], r.get("suffix", ""), r["gate"], r["basis"], r["n_qubits"], r["n_params"])
        groups[key].append(r)

    metrics = ["disp_rel", "A_init", "A_final", "deff_init", "deff_final", "chi2_init", "chi2_final",
               "grad2_init", "grad2_final", "natgrad2_init", "natgrad2_final",
               "task_grad2_init", "task_grad2_final", "loss_init", "loss_final",
               "head_w_q", "head_w_alpha", "head_w_skip", "prop1_relerr_final",
               "q_grad2_pp_init", "q_grad2_pp_final", "cls_grad2_pp_init", "cls_grad2_pp_final"]
    derived = ["atten_init", "atten_final", "head_q_share"]

    out = []
    for key, rs in sorted(groups.items()):
        config, campaign, suffix, gate, basis, nq, L = key
        Lp, dp, dead, rym, xxm = deff_predicted(nq, rs[0].get("n_reup", 2), gate)
        rec = {"config": config, "campaign": campaign, "gate": gate, "basis": basis,
               "n_qubits": nq, "n_params": L, "hilbert_dim": 2 ** int(nq),
               "tangent_dim_bound": 2 * (2 ** int(nq) - 1), "n_seeds": len(rs),
               "seeds": " ".join(sorted(r["seed"] for r in rs)),
               "deff_predicted": dp, "dead_params": dead, "ry_merges": rym, "xx_merges": xxm}
        for m in metrics:
            vals = np.array([fnum(r.get(m)) for r in rs])
            vals = vals[~np.isnan(vals)]
            rec[f"{m}_mean"] = float(vals.mean()) if vals.size else math.nan
            rec[f"{m}_sd"] = float(vals.std(ddof=1)) if vals.size > 1 else math.nan
        # attenuation: how much of the per-Pauli gradient survives the head
        for tag in ("init", "final"):
            a = np.array([fnum(r.get(f"task_grad2_{tag}")) / fnum(r.get(f"grad2_{tag}")) for r in rs])
            a = a[np.isfinite(a)]
            rec[f"atten_{tag}_mean"] = float(a.mean()) if a.size else math.nan
            rec[f"atten_{tag}_sd"] = float(a.std(ddof=1)) if a.size > 1 else math.nan
        # per-parameter gradient of the circuit relative to the classical parameters
        for tag in ("init", "final"):
            r_ = np.array([fnum(r.get(f"q_grad2_pp_{tag}")) / fnum(r.get(f"cls_grad2_pp_{tag}")) for r in rs])
            r_ = r_[np.isfinite(r_)]
            rec[f"q_over_cls_pp_{tag}_mean"] = float(r_.mean()) if r_.size else math.nan
            rec[f"q_over_cls_pp_{tag}_sd"] = float(r_.std(ddof=1)) if r_.size > 1 else math.nan
        # share of first-layer weight norm on the quantum stream
        sh = []
        for r in rs:
            q, al, sk = fnum(r["head_w_q"]), fnum(r["head_w_alpha"]), fnum(r["head_w_skip"])
            tot = q * q + al * al + sk * sk
            if np.isfinite(tot) and tot > 0:
                sh.append(q * q / tot)
        sh = np.array(sh)
        rec["head_q_share_mean"] = float(sh.mean()) if sh.size else math.nan
        rec["head_q_share_sd"] = float(sh.std(ddof=1)) if sh.size > 1 else math.nan
        out.append(rec)

    fields = list(out[0].keys())
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out)

    print(f"{'config':20s} {'camp':9s} {'n':>2s} {'disp_rel':>9s} {'A_init':>7s} {'A_fin':>7s} "
          f"{'d_eff=pred':>10s} {'chi2':>6s} {'grad2':>7s} {'task_g2':>9s} {'atten':>9s} {'q-share':>8s}")
    for r in out:
        print(f"{r['config']:20s} {r['campaign']:9s} {r['n_seeds']:>2d} "
              f"{r['disp_rel_mean']:9.4f} {r['A_init_mean']:7.3f} {r['A_final_mean']:7.3f} "
              f"{r['deff_final_mean']:6.1f}={r['deff_predicted']:<4d}{r['chi2_final_mean']:6.3f} {r['grad2_final_mean']:7.3f} "
              f"{r['task_grad2_final_mean']:9.2e} {r['atten_final_mean']:9.2e} {r['head_q_share_mean']:8.3f} {r['q_over_cls_pp_final_mean']:9.2e}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
