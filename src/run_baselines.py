"""
run_baselines.py

SEQNN and QC-CNN baselines under the EXACT S2Q-SEQNN data pipeline.

This runner imports load_data, balanced_loader, and plain_loader directly
from sq_seqnn_fast.py, so the splits, normalisation, augmentation, and
weighted sampler are identical to the ablation runs. The seed is
injected into CONFIG["SEED"] before every split, matching how the ablation
runner drives sq_seqnn_fast.py per seed.

Usage
-----
Run everything (three datasets, both baselines, seeds 42-51):
    python run_baselines.py

Common variations:
    python run_baselines.py --datasets overhead
    python run_baselines.py --datasets sat6 so2sat --baseline qccnn
    python run_baselines.py --seeds 42 43 44 45 46          (five-seed pilot)
    python run_baselines.py --seqnn-widths 10               (matched only)

Outputs
-------
reviewer_ablations/baselines/<tag>_runs.csv   per-seed rows
plus a printed summary with mean, std, and 95 percent CI half-width,
in the same format as the ablation table.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from seqnn_qccnn_baselines import (
    QCCNN,
    SEQNN,
    TrainConfig,
    count_params,
    evaluate,
    evaluate_balanced,
    train_one,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "reviewer_ablations" / "baselines"

# Same per-dataset flags as the ablation runs, so the imported
# module configures itself identically to the ablation runs.
DATASET_FLAGS = {
    "overhead": [
        "--dataset", "overhead", "--overhead-mode", "seqnn5",
        "--frontend-mode", "conv", "--gate-type", "CX", "--measure-bases", "x",
        "--n-qubits", "8", "--cls-hidden", "10", "--conv-width", "4",
    ],
    "sat6": [
        "--dataset", "sat6",
        "--frontend-mode", "shapelet", "--gate-type", "XX", "--measure-bases", "x",
        "--n-qubits", "10", "--cls-hidden", "10",
        "--shapelet-count", "6", "--shapelet-length", "4", "--shapelet-stats", "full6",
    ],
    "so2sat": [
        "--dataset", "so2sat",
        "--frontend-mode", "global", "--gate-type", "CRY", "--measure-bases", "xyz",
        "--n-qubits", "10", "--cls-hidden", "10",
    ],
}

DATASET_INFO = {
    "overhead": {"in_channels": 1, "num_classes": 5},
    "sat6": {"in_channels": 4, "num_classes": 6},
    "so2sat": {"in_channels": 4, "num_classes": 5},
}


def import_pipeline(dataset: str):
    """Import (or re-import) sq_seqnn_fast configured for one dataset.

    The module parses sys.argv at import time, so the dataset flags are
    injected there, mirroring how the ablation runs launch it.
    """
    argv_backup = sys.argv
    sys.argv = ["sq_seqnn_fast.py"] + DATASET_FLAGS[dataset] + [
        "--seed", "42", "--run-tag", "baseline_probe",
    ]
    try:
        if "sq_seqnn_fast" in sys.modules:
            sqf = importlib.reload(sys.modules["sq_seqnn_fast"])
        else:
            import sq_seqnn_fast as sqf  # noqa: PLC0415
    finally:
        sys.argv = argv_backup
    return sqf


def loaders_for_seed(sqf, dataset: str, seed: int, batch_size: int):
    """Seed-matched loaders through the repo's own pipeline functions."""
    sqf.CONFIG["SEED"] = seed
    sqf.set_seed(seed)
    tr_x, tr_y, val_x, val_y, te_x, te_y = sqf.load_data(dataset)
    train = sqf.balanced_loader(tr_x, tr_y, batch_size, aug=True)
    # Clean pass over the training set (no augmentation, no resampling)
    # for reporting train accuracy comparably to the main-model tables.
    train_eval = sqf.plain_loader(tr_x, tr_y, batch_size)
    val = sqf.plain_loader(val_x, val_y, batch_size)
    test = sqf.plain_loader(te_x, te_y, batch_size)
    return train, train_eval, val, test


def run_seeds_for(model_fn, sqf, dataset, cfg, seeds, tag, out_dir=OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{tag}_runs.csv"

    accs = []
    t0 = time.time()
    # Truncate and write a header once so a re-run overwrites the previous
    # results for this tag instead of appending duplicate rows 
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["tag", "seed", "train_acc", "val_acc",
                                "test_acc", "bal_test_acc", "n_params"])
    for seed in seeds:
        train, train_eval, val, test = loaders_for_seed(
            sqf, dataset, seed, cfg.batch_size)
        torch.manual_seed(seed)          # model init after data build
        model = model_fn()
        n_params = count_params(model)
        model, val_acc = train_one(model, train, val, cfg)
        train_acc = evaluate(model, train_eval)
        acc = evaluate(model, test)
        bal_acc = evaluate_balanced(model, test)
        accs.append(acc)
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([tag, seed, f"{train_acc:.4f}",
                                    f"{val_acc:.4f}", f"{acc:.4f}",
                                    f"{bal_acc:.4f}", n_params])
        print(f"  [{tag}] seed {seed}: train {train_acc:.2f}  val {val_acc:.2f}"
              f"  test {acc:.2f}  bal {bal_acc:.2f} ({n_params} params)")

    arr = np.asarray(accs)
    mean = arr.mean()
    std = arr.std(ddof=1) if len(arr) > 1 else float("nan")
    ci95 = 1.96 * std / math.sqrt(len(arr)) if len(arr) > 1 else float("nan")
    print(f"  [{tag}] {mean:.2f} +/- {std:.2f} (CI95 half {ci95:.2f})")
    return {"tag": tag, "n_seeds": len(arr), "mean": mean, "std": std,
            "ci95_half": ci95, "minutes": (time.time() - t0) / 60.0}


def main():
    ap = argparse.ArgumentParser(
        description="SEQNN and QC-CNN baselines under the exact S2Q-SEQNN splits."
    )
    ap.add_argument("--datasets", nargs="+",
                    default=list(DATASET_INFO.keys()),
                    choices=list(DATASET_INFO.keys()))
    ap.add_argument("--baseline", default="both",
                    choices=["seqnn", "qccnn", "both"])
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51])
    ap.add_argument("--seqnn-widths", type=int, nargs="+", default=[16, 10],
                    help="16 is faithful, 10 is parameter-matched, "
                         "default runs both")
    ap.add_argument("--qccnn-hidden", type=int, default=16,
                    help="Hidden width of the QC-CNN classifier head "
                         "(16 is the faithful configuration; larger values "
                         "give parameter-matched variants).")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--out-dir", default=str(OUT_DIR),
                    help="Output directory for <tag>_runs.csv files. Use a "
                         "separate directory to avoid overwriting existing "
                         "campaign results (CSVs are truncated on start).")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = SCRIPT_DIR / out_dir

    summaries = []
    for ds in args.datasets:
        info = DATASET_INFO[ds]
        cfg = TrainConfig(epochs=args.epochs)
        print(f"\n########## Dataset: {ds} ##########")
        sqf = import_pipeline(ds)

        jobs = []
        want = ["seqnn", "qccnn"] if args.baseline == "both" else [args.baseline]
        if "seqnn" in want:
            for w in args.seqnn_widths:
                jobs.append((
                    f"seqnn_w{w}_{ds}",
                    lambda w=w: SEQNN(info["in_channels"],
                                      info["num_classes"], width=w),
                ))
        if "qccnn" in want:
            h = args.qccnn_hidden
            qccnn_tag = f"qccnn_{ds}" if h == 16 else f"qccnn_h{h}_{ds}"
            jobs.append((
                qccnn_tag,
                lambda h=h: QCCNN(info["in_channels"], info["num_classes"],
                                  hidden=h),
            ))

        for tag, model_fn in jobs:
            probe = model_fn()
            print(f"\n=== {tag}: {count_params(probe)} parameters, "
                  f"seeds {args.seeds[0]}-{args.seeds[-1]} ===")
            del probe
            summaries.append(
                run_seeds_for(model_fn, sqf, ds, cfg, args.seeds, tag,
                              out_dir=out_dir)
            )

    print("\n===== Final summary across all runs =====")
    for s in summaries:
        print(f"{s['tag']:>24s}: {s['mean']:6.2f} +/- {s['std']:5.2f} "
              f"(CI95 half {s['ci95_half']:.2f}, {s['minutes']:.0f} min)")


if __name__ == "__main__":
    main()
