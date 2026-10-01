#!/usr/bin/env python3
"""
Generate the run manifest for the full S2Q-SEQNN campaign on Panther.

Every row is one independent job: (campaign, dataset, variant, config-name,
seed, script, flags, output-tag). slurm/run_one.sh executes one row,
and slurm/run_packed.sh runs many rows per job. collect_results.py later gathers every row's output.

The base configurations below are copied from the files that produced the
published numbers, not from the paper text:
  Adaptive configs : the per-dataset flags of the original training runs
  Shared configs   : per-run CSVs of the three-seed Shared runs, read back
                     in this session (frontend/gate/basis/n_qubits/shapelet_*)
Change a base config only if you also intend to change the paper.

Campaigns
---------
  main        the 5 published configurations
  isolation   6 controls x 3 Adaptive configs
  gate        8 entanglers x 5 configs, best skipped
  basis       4 read-outs  x 5 configs, best skipped
  fusion      quantum_only head on all 5 configs
  schedule    shared learning rate (rho_q = 1) x 5
  sens_nq     n_qubits in {10, 12} on Overhead Adaptive
  sens_R      re-upload rounds in {1, 3} x 5 configs
  sens_eps    label smoothing in {0.00, 0.10} x 5 configs
  seqnn       faithful (w16) and matched (w10) SEQNN, 3 datasets
  qccnn       QC-CNN, 3 datasets
  classical   6 compact classical baselines x 3 datasets

Usage
-----
    python tools/make_manifest.py                      # all campaigns, seeds 42-51
    python tools/make_manifest.py --campaigns main gate
    python tools/make_manifest.py --seeds 42 43 44 --out manifest_pilot.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

SEEDS_DEFAULT = list(range(42, 52))

# ---------------------------------------------------------------------------
# Base configurations that produced the published tables.
# key -> (dataset, variant, flags)
# ---------------------------------------------------------------------------
BASE = {
    "overhead_adaptive": ("overhead", "adaptive", [
        "--dataset", "overhead", "--overhead-mode", "seqnn5",
        "--frontend-mode", "conv", "--gate-type", "CX", "--measure-bases", "x",
        "--n-qubits", "8", "--cls-hidden", "10", "--conv-width", "4",
    ]),
    "overhead_shared": ("overhead", "shared", [
        "--dataset", "overhead", "--overhead-mode", "seqnn5",
        "--frontend-mode", "shapelet", "--gate-type", "XY", "--measure-bases", "z",
        "--n-qubits", "10", "--cls-hidden", "10",
        "--shapelet-count", "6", "--shapelet-length", "1", "--shapelet-stats", "compact4",
    ]),
    # On SAT-6 the selector picks the shapelet frontend, so Adaptive == Shared.
    "sat6": ("sat6", "both", [
        "--dataset", "sat6",
        "--frontend-mode", "shapelet", "--gate-type", "XX", "--measure-bases", "x",
        "--n-qubits", "10", "--cls-hidden", "10",
        "--shapelet-count", "6", "--shapelet-length", "4", "--shapelet-stats", "full6",
    ]),
    "so2sat_adaptive": ("so2sat", "adaptive", [
        "--dataset", "so2sat",
        "--frontend-mode", "global", "--gate-type", "CRY", "--measure-bases", "xyz",
        "--n-qubits", "10", "--cls-hidden", "10",
    ]),
    "so2sat_shared": ("so2sat", "shared", [
        "--dataset", "so2sat",
        "--frontend-mode", "shapelet", "--gate-type", "CZ", "--measure-bases", "x",
        "--n-qubits", "10", "--cls-hidden", "10",
        "--shapelet-count", "6", "--shapelet-length", "4", "--shapelet-stats", "compact4",
    ]),
}
ADAPTIVE_KEYS = ["overhead_adaptive", "sat6", "so2sat_adaptive"]   # isolation study scope
ALL_KEYS = list(BASE.keys())

GATES = ["ZZ", "CZ", "CX", "XY", "CRZ", "CRY", "XX", "YY"]
BASES = ["x", "y", "z", "xyz"]
ISOLATION = ["classical_only", "mlp_replace", "frozen_quantum", "no_iqp", "no_reupload", "fixed_z"]
CLASSICAL = ["resnet", "vgg", "densenet", "efficientnet", "vit", "swin"]
DATASETS = ["overhead", "sat6", "so2sat"]


def _flag_value(flags: list[str], name: str) -> str | None:
    if name in flags:
        return flags[flags.index(name) + 1]
    return None


def _with_flag(flags: list[str], name: str, value: str) -> list[str]:
    f = list(flags)
    if name in f:
        f[f.index(name) + 1] = value
    else:
        f += [name, value]
    return f


def rows_for(campaigns: list[str], seeds: list[int]) -> list[dict]:
    rows: list[dict] = []

    def add(campaign, key, config, flags, seed, script="sq_seqnn_fast.py", extra=None):
        ds, var, _ = BASE[key] if key in BASE else (key, "-", None)
        tag = f"{campaign}__{config}__seed{seed}"
        rows.append(dict(
            campaign=campaign, dataset=ds, variant=var, config=config, seed=seed,
            script=script, tag=tag,
            flags=" ".join(flags + (extra or [])),
        ))

    for seed in seeds:
        if "main" in campaigns:
            for k in ALL_KEYS:
                add("main", k, k, BASE[k][2] + ["--ablation", "none"], seed)

        if "isolation" in campaigns:
            for k in ADAPTIVE_KEYS:
                for ab in ISOLATION:
                    add("isolation", k, f"{k}__{ab}", BASE[k][2] + ["--ablation", ab], seed)

        if "gate" in campaigns:
            for k in ALL_KEYS:
                best = _flag_value(BASE[k][2], "--gate-type")
                for g in GATES:
                    if g == best:
                        continue          # identical to main; do not run twice
                    add("gate", k, f"{k}__gate{g}", _with_flag(BASE[k][2], "--gate-type", g), seed)

        if "basis" in campaigns:
            for k in ALL_KEYS:
                best = _flag_value(BASE[k][2], "--measure-bases")
                for b in BASES:
                    if b == best:
                        continue
                    add("basis", k, f"{k}__basis{b.upper()}", _with_flag(BASE[k][2], "--measure-bases", b), seed)

        if "fusion" in campaigns:
            for k in ALL_KEYS:
                add("fusion", k, f"{k}__quantum_only", BASE[k][2] + ["--ablation", "quantum_only"], seed)

        if "schedule" in campaigns:
            for k in ALL_KEYS:
                add("schedule", k, f"{k}__sharedLR", BASE[k][2] + ["--lr-q-factor", "1.0"], seed)

        if "sens_nq" in campaigns:
            k = "overhead_adaptive"
            for nq in ["10", "12"]:
                add("sens_nq", k, f"{k}__nq{nq}", _with_flag(BASE[k][2], "--n-qubits", nq), seed)

        if "sens_R" in campaigns:
            for k in ALL_KEYS:
                for R in ["1", "3"]:
                    add("sens_R", k, f"{k}__R{R}", BASE[k][2] + ["--n-reupload", R], seed)

        if "sens_eps" in campaigns:
            for k in ALL_KEYS:
                for eps in ["0.00", "0.10"]:
                    add("sens_eps", k, f"{k}__eps{eps}", BASE[k][2] + ["--label-smooth", eps], seed)

        if "seqnn" in campaigns:
            for ds in DATASETS:
                for w in ["16", "10"]:
                    rows.append(dict(
                        campaign="seqnn", dataset=ds, variant=f"w{w}", config=f"seqnn_w{w}_{ds}", seed=seed,
                        script="run_baselines.py", tag=f"seqnn__seqnn_w{w}_{ds}__seed{seed}",
                        flags=f"--datasets {ds} --baseline seqnn --seqnn-widths {w} --seeds {seed}",
                    ))

        if "qccnn" in campaigns:
            for ds in DATASETS:
                rows.append(dict(
                    campaign="qccnn", dataset=ds, variant="-", config=f"qccnn_{ds}", seed=seed,
                    script="run_baselines.py", tag=f"qccnn__qccnn_{ds}__seed{seed}",
                    flags=f"--datasets {ds} --baseline qccnn --seeds {seed}",
                ))

        if "classical" in campaigns:
            for ds in DATASETS:
                for m in CLASSICAL:
                    rows.append(dict(
                        campaign="classical", dataset=ds, variant=m, config=f"{m}_{ds}", seed=seed,
                        script="heavy_baselines.py", tag=f"classical__{m}_{ds}__seed{seed}",
                        flags=f"--dataset {ds} --model {m} --runs 1 --seed {seed} --epochs 80",
                    ))
    return rows


ALL_CAMPAIGNS = ["main", "isolation", "gate", "basis", "fusion", "schedule",
                 "sens_nq", "sens_R", "sens_eps", "seqnn", "qccnn", "classical"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaigns", nargs="+", default=ALL_CAMPAIGNS, choices=ALL_CAMPAIGNS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS_DEFAULT)
    ap.add_argument("--out", default="manifest.csv")
    args = ap.parse_args()

    rows = rows_for(args.campaigns, args.seeds)
    for i, r in enumerate(rows):
        r["task_id"] = i
    fields = ["task_id", "campaign", "dataset", "variant", "config", "seed", "script", "tag", "flags"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    by = {}
    for r in rows:
        by[r["campaign"]] = by.get(r["campaign"], 0) + 1
    print(f"wrote {len(rows)} tasks to {args.out}")
    for c in ALL_CAMPAIGNS:
        if c in by:
            print(f"  {c:10s} {by[c]:5d}")
    print(f"\nsbatch --array=0-9 slurm/run_packed.sh {args.out}")


if __name__ == "__main__":
    main()
