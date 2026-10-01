#!/usr/bin/env python3
"""
Collect every finished run of the Panther campaign into one table.

Reads the manifest to know what *should* exist, walks the output folders to
find what *does* exist, joins the two, and writes:

    results/all_runs.csv               one row per finished run, all campaigns
    results/by_campaign/<campaign>.csv the same rows split by campaign
    results/completeness.csv           expected / found / missing per campaign
    results/missing_task_ids.txt       task ids to resubmit, one per line

Nothing here is typed in by hand: every accuracy comes from a CSV that the
training script wrote.

Usage
-----
    python tools/collect_results.py manifest.csv
    # then, to resubmit only what is missing:
    for t in $(cat results/missing_task_ids.txt); do bash slurm/run_one.sh manifest.csv $t; done
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

BUNDLE = Path(__file__).resolve().parent.parent
SRC = BUNDLE / "src"
RES = BUNDLE / "results"


def _pct(x):
    """Accuracies are stored as fractions by sq_seqnn_fast (0.86) and as
    percents by the baseline runners (92.3). The merged column is mixed, so
    normalise element-wise: anything <= 1.5 is a fraction. No model here
    scores under 1.5 %, so the rule is unambiguous."""
    x = pd.to_numeric(x, errors="coerce")
    return x.where(x > 1.5, x * 100.0)


def gather_main_model(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sub = manifest[manifest.script == "sq_seqnn_fast.py"]
    for _, m in sub.iterrows():
        mode = "seqnn5" if m.dataset == "overhead" else "default"
        f = SRC / "sq_seqnn_outputs" / f"{m.dataset}_{mode}_{m.tag}" / f"results_{m.dataset}.csv"
        if not f.exists():
            continue
        r = pd.read_csv(f).iloc[0].to_dict()
        rows.append(dict(
            task_id=m.task_id, campaign=m.campaign, dataset=m.dataset, variant=m.variant,
            config=m.config, seed=int(m.seed), tag=m.tag,
            test_acc=r.get("accuracy"), bal_test_acc=r.get("balanced_accuracy"),
            val_acc=r.get("best_val_acc"), train_acc=r.get("train_accuracy", r.get("train_acc")),
            total_params=r.get("total_params"), quantum_params=r.get("quantum_params"),
            n_qubits=r.get("n_qubits"), gate=r.get("gate_type"), basis=r.get("measure_bases"),
            frontend=r.get("frontend_mode"), ablation=r.get("ablation", "none"),
            source_file=str(f.relative_to(BUNDLE)),
        ))
    return pd.DataFrame(rows)


def gather_seqnn_qccnn(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sub = manifest[manifest.script == "run_baselines.py"]
    for _, m in sub.iterrows():
        d = RES / "baselines" / m.tag
        files = list(d.glob("*_runs.csv")) if d.is_dir() else []
        if not files:
            continue
        df = pd.read_csv(files[0])
        df = df[df.seed == int(m.seed)]
        if df.empty:
            continue
        r = df.iloc[0].to_dict()
        rows.append(dict(
            task_id=m.task_id, campaign=m.campaign, dataset=m.dataset, variant=m.variant,
            config=m.config, seed=int(m.seed), tag=m.tag,
            test_acc=r.get("test_acc"), bal_test_acc=r.get("bal_test_acc"),
            val_acc=r.get("val_acc"), train_acc=r.get("train_acc"),
            total_params=r.get("n_params"), quantum_params=None, n_qubits=None,
            gate=None, basis=None, frontend=None, ablation="none",
            source_file=str(files[0].relative_to(BUNDLE)),
        ))
    return pd.DataFrame(rows)


def gather_classical(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sub = manifest[manifest.script == "heavy_baselines.py"]
    for _, m in sub.iterrows():
        f = RES / "classical" / m.tag / "full" / "heavy_baseline_runs.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        if "seed" in df.columns:
            df = df[df.seed == int(m.seed)]
        if df.empty:
            continue
        r = df.iloc[0].to_dict()
        # column names in heavy_baselines vary slightly; take what is present
        def pick(*names):
            for n in names:
                if n in r and pd.notna(r[n]):
                    return r[n]
            return None
        rows.append(dict(
            task_id=m.task_id, campaign=m.campaign, dataset=m.dataset, variant=m.variant,
            config=m.config, seed=int(m.seed), tag=m.tag,
            test_acc=pick("test_acc", "test_accuracy", "accuracy"),
            bal_test_acc=pick("bal_test_acc", "balanced_accuracy", "test_balanced_accuracy"),
            val_acc=pick("val_acc", "best_val_acc", "val_accuracy"),
            train_acc=pick("train_acc", "train_accuracy"),
            total_params=pick("n_params", "params", "total_params"),
            quantum_params=None, n_qubits=None, gate=None, basis=None, frontend=None, ablation="none",
            source_file=str(f.relative_to(BUNDLE)),
        ))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    parts = [gather_main_model(manifest), gather_seqnn_qccnn(manifest), gather_classical(manifest)]
    found = pd.concat([p for p in parts if not p.empty], ignore_index=True) if any(not p.empty for p in parts) else pd.DataFrame()

    RES.mkdir(exist_ok=True)
    (RES / "by_campaign").mkdir(exist_ok=True)

    if found.empty:
        print("No finished runs found yet.")
    else:
        for c in ("test_acc", "bal_test_acc", "val_acc", "train_acc"):
            if c in found.columns:
                found[c] = _pct(found[c])
        found = found.sort_values(["campaign", "dataset", "config", "seed"]).reset_index(drop=True)
        found.to_csv(RES / "all_runs.csv", index=False)
        for c, g in found.groupby("campaign"):
            g.to_csv(RES / "by_campaign" / f"{c}.csv", index=False)

    # completeness
    exp = manifest.groupby("campaign").size().rename("expected")
    got = found.groupby("campaign").size().rename("found") if not found.empty else pd.Series(dtype=int, name="found")
    comp = pd.concat([exp, got], axis=1).fillna(0).astype(int)
    comp["missing"] = comp["expected"] - comp["found"]
    comp.to_csv(RES / "completeness.csv")

    done_ids = set(found.task_id.astype(int)) if not found.empty else set()
    missing = sorted(set(manifest.task_id.astype(int)) - done_ids)
    (RES / "missing_task_ids.txt").write_text("\n".join(map(str, missing)) + ("\n" if missing else ""))

    print(f"finished runs: {len(found)} / {len(manifest)}")
    print(comp.to_string())
    if missing:
        print(f"\n{len(missing)} tasks missing -> results/missing_task_ids.txt")
        print("resubmit with:")
        print(f"  for t in $(cat results/missing_task_ids.txt); do bash slurm/run_one.sh {args.manifest} $t; done")
    else:
        print("\nAll tasks complete. results/all_runs.csv is the single source for every table.")


if __name__ == "__main__":
    main()
