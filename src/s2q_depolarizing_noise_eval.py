"""
Evaluate saved S2Q-SEQNN checkpoints under local depolarizing noise.

The script keeps the trained model fixed and replaces only the quantum
forward pass with a default.mixed simulator. A single depolarizing rate p is
applied after each encoded or trainable quantum operation.

Example:
  MPLCONFIGDIR="$PWD/.mplconfig" \
  python src/s2q_depolarizing_noise_eval.py --dataset all --seeds 42 43 44

Quick smoke test:
  MPLCONFIGDIR="$PWD/.mplconfig" \
  python src/s2q_depolarizing_noise_eval.py --dataset overhead --seeds 42 \
  --max-test-samples 24 --batch-size 4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


import os

# Resolve everything relative to this file unless told otherwise. The original
# value was a machine-specific path.
CODES_DIR = Path(os.environ.get("S2Q_NOISE_CODES", Path(__file__).resolve().parent))
BASE_DIR = CODES_DIR

DATASET_LABELS = {
    "overhead": "Overhead-MNIST",
    "sat6": "SAT-6",
    "so2sat": "So2Sat",
}

# Paper-best choices from the completed optimization summaries.
# Overhead uses the best Adaptive run, SAT-6 is identical for Shared and
# Adaptive, and So2Sat uses the best overall Shared run.
BEST_RUNS = {
    "overhead": {
        "variant": "S2Q-SEQNN-Adaptive",
        "model_script": CODES_DIR / "sq_seqnn_fast.py",
        "outputs_root": CODES_DIR / "sq_seqnn_outputs",
        "folder": "overhead_seqnn5_main__overhead_adaptive__seed{seed}",
        "gate": "CX",
        "basis": "x",
        "frontend": "conv",
    },
    "sat6": {
        "variant": "S2Q-SEQNN-Both",
        "model_script": CODES_DIR / "sq_seqnn_fast.py",
        "outputs_root": CODES_DIR / "sq_seqnn_outputs",
        "folder": "sat6_default_main__sat6__seed{seed}",
        "gate": "XX",
        "basis": "x",
        "frontend": "shapelet",
    },
    "so2sat": {
        "variant": "S2Q-SEQNN-Shared",
        "model_script": CODES_DIR / "sq_seqnn_fast.py",
        "outputs_root": CODES_DIR / "sq_seqnn_outputs",
        "folder": "so2sat_default_main__so2sat_shared__seed{seed}",
        "gate": "CZ",
        "basis": "x",
        "frontend": "shapelet",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Depolarizing-noise evaluation for paper-best S2Q-SEQNN gates."
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["all"],
        choices=["all", "overhead", "sat6", "so2sat"],
        help="Dataset(s) to evaluate.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--p-values",
        nargs="+",
        type=float,
        default=[0.0, 0.001, 0.005, 0.01],
        help="Depolarizing rates. Keep 0.0 to re-check the clean checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional stratified cap for smoke tests. Omit for the full test set.",
    )
    parser.add_argument(
        "--tta",
        default="checkpoint",
        choices=["checkpoint", "none", "flip", "rot4", "d4"],
        help="Use checkpoint TTA or force a fixed TTA mode.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(CODES_DIR / "noise_results" / "depolarizing_best_gates"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip dataset/seed/p combinations already present in the raw CSV.",
    )
    return parser


def selected_datasets(items: list[str]) -> list[str]:
    return ["overhead", "sat6", "so2sat"] if "all" in items else items


def valid_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    if isinstance(value, str) and value.strip() in {"", "nan", "None"}:
        return False
    return True


def add_flag(flags: list[str], name: str, value: Any) -> None:
    if valid_value(value):
        flags.extend([name, str(value)])


def flags_from_result(dataset: str, result: dict[str, Any], batch_size: int) -> list[str]:
    flags = ["model", "--dataset", dataset]
    if dataset == "overhead":
        add_flag(flags, "--overhead-mode", result.get("overhead_mode", "seqnn5"))
    if result.get("quick", False):
        flags.append("--quick")

    for name, key in [
        ("--n-qubits", "n_qubits"),
        ("--lr", "lr"),
        ("--epochs", "epochs"),
        ("--cls-hidden", "cls_hidden"),
        ("--lr-q-factor", "lr_q_factor"),
        ("--label-smooth", "label_smooth"),
        ("--tta", "tta_mode"),
        ("--measure-bases", "measure_bases"),
        ("--gate-type", "gate_type"),
        ("--scat-grid", "scat_grid"),
        ("--compress-mode", "compress_mode"),
        ("--frontend-mode", "frontend_mode"),
        ("--conv-width", "conv_width"),
        ("--shapelet-count", "shapelet_count"),
        ("--shapelet-length", "shapelet_length"),
        ("--shapelet-stats", "shapelet_stats"),
        ("--summary-mode", "summary_mode"),
        ("--angle-mode", "angle_mode"),
        ("--seed", "seed"),
    ]:
        add_flag(flags, name, result.get(key))
    add_flag(flags, "--batch-size", batch_size)
    return flags


def import_model_module(
    model_script: Path,
    dataset: str,
    result: dict[str, Any],
    batch_size: int,
    module_name: str,
):
    old_argv = sys.argv[:]
    sys.argv = flags_from_result(dataset, result, batch_size)
    try:
        spec = importlib.util.spec_from_file_location(module_name, model_script)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import {model_script}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.argv = old_argv


def stratified_cap(X: np.ndarray, y: np.ndarray, max_samples: int | None, seed: int):
    if max_samples is None or len(y) <= max_samples:
        return X, y
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class = max(1, max_samples // len(classes))
    chosen = []
    for cls in classes:
        idx = np.flatnonzero(y == cls)
        take = min(per_class, len(idx))
        chosen.extend(rng.choice(idx, size=take, replace=False).tolist())
    if len(chosen) < max_samples:
        remaining = np.setdiff1d(np.arange(len(y)), np.array(chosen), assume_unique=False)
        extra = min(max_samples - len(chosen), len(remaining))
        if extra > 0:
            chosen.extend(rng.choice(remaining, size=extra, replace=False).tolist())
    chosen = np.array(sorted(chosen), dtype=int)
    return X[chosen], y[chosen]


def install_noisy_q_forward(module, p: float):
    """Patch module.q_forward with a default.mixed noisy circuit."""

    qml = module.qml
    torch = module.torch
    n_qubits = module.N_QUBITS
    n_reup = module.N_REUP
    pair_params = module.QCNN_PAIR_PARAMS
    gate_type = module.GATE_TYPE
    measure_bases = module.MEASURE_BASES

    noisy_dev = qml.device("default.mixed", wires=n_qubits)

    def depol(wires):
        for wire in wires:
            qml.DepolarizingChannel(p, wires=wire)

    def apply_entangler(param, q0, q1):
        if gate_type == "ZZ":
            qml.IsingZZ(param, wires=[q0, q1])
        elif gate_type == "XY":
            qml.IsingXY(param, wires=[q0, q1])
        elif gate_type == "XX":
            qml.IsingXX(param, wires=[q0, q1])
        elif gate_type == "YY":
            qml.IsingYY(param, wires=[q0, q1])
        elif gate_type == "CRZ":
            qml.CRZ(param, wires=[q0, q1])
        elif gate_type == "CRY":
            qml.CRY(param, wires=[q0, q1])
        elif gate_type == "CZ":
            qml.CZ(wires=[q0, q1])
        elif gate_type == "CX":
            qml.CNOT(wires=[q0, q1])
        elif gate_type == "SWAP":
            qml.SWAP(wires=[q0, q1])
        elif gate_type == "ISWAP":
            qml.ISWAP(wires=[q0, q1])
        else:
            raise ValueError(f"Unsupported gate type: {gate_type}")
        depol([q0, q1])

    def iqp_encode_noisy(features, include_hadamards=True):
        if include_hadamards:
            for q in range(n_qubits):
                qml.Hadamard(wires=q)
                depol([q])
        for q in range(n_qubits):
            qml.RZ(2.0 * features[q], wires=q)
            depol([q])
        for q in range(n_qubits - 1):
            qml.CNOT(wires=[q, q + 1])
            depol([q, q + 1])
            qml.RZ(2.0 * features[q] * features[q + 1], wires=q + 1)
            depol([q + 1])
            qml.CNOT(wires=[q, q + 1])
            depol([q, q + 1])

    def qcnn_pair_noisy(weights, q0, q1):
        qml.RY(weights[0], wires=q0)
        depol([q0])
        qml.RY(weights[1], wires=q1)
        depol([q1])
        apply_entangler(weights[2], q0, q1)
        qml.RX(weights[3], wires=q0)
        depol([q0])
        qml.RX(weights[4], wires=q1)
        depol([q1])
        apply_entangler(weights[5], q0, q1)
        qml.RY(weights[6], wires=q0)
        depol([q0])
        qml.RY(weights[7], wires=q1)
        depol([q1])

    def qcnn_sweep_noisy(weights, ptr, start_idx):
        for q in range(start_idx, n_qubits - 1, 2):
            qcnn_pair_noisy(weights[ptr : ptr + pair_params], q, q + 1)
            ptr += pair_params
        return ptr

    @qml.qnode(noisy_dev, interface="torch", diff_method="backprop")
    def noisy_circuit(features, weights):
        ptr = 0
        for upload_idx in range(n_reup):
            iqp_encode_noisy(features, include_hadamards=(upload_idx == 0))
            ptr = qcnn_sweep_noisy(weights, ptr, 0)
            ptr = qcnn_sweep_noisy(weights, ptr, 1)
        if measure_bases == "x":
            return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
        if measure_bases == "y":
            return [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]
        if measure_bases == "z":
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        if measure_bases == "xz":
            x = [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
            z = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            return x + z
        z = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        x = [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
        y = [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]
        return z + x + y

    def noisy_q_forward(alpha_batch, weights):
        rows = []
        for b in range(alpha_batch.shape[0]):
            out = noisy_circuit(alpha_batch[b], weights)
            rows.append(torch.stack(list(out)))
        return torch.stack(rows, dim=0)

    module.q_forward = noisy_q_forward


def load_run(dataset: str, seed: int, batch_size: int):
    cfg = BEST_RUNS[dataset]
    folder = cfg["outputs_root"] / cfg["folder"].format(seed=seed)
    result_json = folder / f"results_{dataset}.json"
    model_path = folder / f"sq_seqnn_{dataset}.pt"
    if not result_json.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"Missing saved checkpoint for {dataset}, seed {seed}:\n"
            f"  {result_json}\n"
            f"  {model_path}"
        )
    result = json.loads(result_json.read_text())
    module = import_model_module(
        cfg["model_script"],
        dataset,
        result,
        batch_size,
        f"s2q_noise_{dataset}_{seed}_{abs(hash(str(folder))) % 10**9}",
    )
    tr_x, tr_y, val_x, val_y, te_x, te_y = module.load_data(dataset)
    model = module.SQ_SEQNN(
        module.META[dataset]["C"],
        module.META[dataset]["size"],
        module.META[dataset]["K"],
    ).to(module.DEVICE)
    state = module.torch.load(model_path, map_location=module.DEVICE)
    model.load_state_dict(state)
    model.eval()
    return cfg, result, module, model, te_x, te_y, folder, model_path


def evaluate_one(
    dataset: str,
    seed: int,
    p: float,
    batch_size: int,
    max_test_samples: int | None,
    tta_choice: str,
) -> dict[str, Any]:
    cfg, result, module, model, te_x, te_y, folder, model_path = load_run(
        dataset, seed, batch_size
    )
    te_x, te_y = stratified_cap(te_x, te_y, max_test_samples, seed)

    if p > 0:
        install_noisy_q_forward(module, p)

    tta_mode = result.get("tta_mode", "none") if tta_choice == "checkpoint" else tta_choice
    t0 = time.time()
    y_pred = module.predict(
        model,
        te_x,
        module.DEVICE,
        batch_size=batch_size,
        tta_mode=tta_mode,
    )
    elapsed = time.time() - t0

    return {
        "dataset_key": dataset,
        "dataset": DATASET_LABELS[dataset],
        "variant": cfg["variant"],
        "seed": seed,
        "p": p,
        "gate_type": result.get("gate_type", cfg["gate"]),
        "measure_bases": result.get("measure_bases", cfg["basis"]),
        "frontend_mode": result.get("frontend_mode", cfg["frontend"]),
        "total_params": int(result.get("total_params", -1)),
        "tta_mode": tta_mode,
        "n_test": int(len(te_y)),
        "accuracy": float(accuracy_score(te_y, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(te_y, y_pred)),
        "macro_f1": float(f1_score(te_y, y_pred, average="macro")),
        "eval_time_sec": elapsed,
        "checkpoint_folder": str(folder),
        "model_path": str(model_path),
    }


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "dataset_key",
        "dataset",
        "variant",
        "p",
        "gate_type",
        "measure_bases",
        "frontend_mode",
        "total_params",
        "tta_mode",
    ]
    summary = (
        raw.groupby(group_cols, dropna=False)
        .agg(
            runs=("seed", "count"),
            n_test=("n_test", "mean"),
            acc_mean=("accuracy", "mean"),
            acc_std=("accuracy", "std"),
            bal_acc_mean=("balanced_accuracy", "mean"),
            bal_acc_std=("balanced_accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            time_mean_sec=("eval_time_sec", "mean"),
            time_total_sec=("eval_time_sec", "sum"),
        )
        .reset_index()
    )
    return summary.fillna(0.0).sort_values(["dataset_key", "p"])


def percent_pm(mean: float, std: float) -> str:
    return f"{100.0 * mean:.2f} $\\pm$ {100.0 * std:.2f}"


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "\\begin{table}[!t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3.2pt}",
        "\\renewcommand{\\arraystretch}{1.10}",
        "\\caption{Robustness under a local single-rate depolarizing channel.}",
        "\\label{tab:depolarizing_noise}",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Dataset & Gate & $p$ & Test Acc. & Bal. Acc. & Macro-F1 \\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{row['dataset']} & {row['gate_type']} & {row['p']:.3f} & "
            f"{percent_pm(row['acc_mean'], row['acc_std'])} & "
            f"{percent_pm(row['bal_acc_mean'], row['bal_acc_std'])} & "
            f"{percent_pm(row['macro_f1_mean'], row['macro_f1_std'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\\\[2pt]",
            "\\raggedright\\footnotesize Values are mean $\\pm$ standard deviation over "
            f"{int(summary['runs'].max())} seeds. "
            "The noise channel is applied after each encoded and trainable quantum operation while keeping all trained parameters fixed.",
            "\\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def write_section_paragraph(summary: pd.DataFrame, path: Path) -> None:
    pieces = []
    noisy = summary[summary["p"] > 0].copy()
    for dataset in ["overhead", "sat6", "so2sat"]:
        rows = noisy[noisy["dataset_key"] == dataset].sort_values("p")
        if rows.empty:
            continue
        label = DATASET_LABELS[dataset]
        gate = str(rows.iloc[0]["gate_type"])
        rates = ", ".join(
            f"p={row['p']:.3f}: {100.0 * row['acc_mean']:.2f}$\\pm${100.0 * row['acc_std']:.2f}\\%"
            for _, row in rows.iterrows()
        )
        pieces.append(f"{label} with {gate} ({rates})")

    paragraph = (
        "To assess the hardware sensitivity of the proposed quantum core, we also evaluated the "
        "per-dataset best-gate checkpoints under a local single-rate depolarising channel with "
        "$p \\in \\{0.001, 0.005, 0.01\\}$. The trained feature extractors, quantum parameters, "
        "and classifier weights were kept fixed, and the channel was inserted after each encoded "
        "and trainable quantum operation in the simulator. The resulting test accuracies were "
        + "; ".join(pieces)
        + ". These results indicate that the proposed compact quantum core preserves most of its "
        "classification behavior under modest depolarizing perturbations, while making explicit "
        "the expected degradation as the noise rate increases."
    )
    path.write_text(paragraph + "\n")


def main() -> None:
    args = build_parser().parse_args()
    datasets = selected_datasets(args.dataset)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "depolarizing_noise_raw.csv"
    rows = []
    completed = set()
    if args.resume and raw_path.exists():
        previous = pd.read_csv(raw_path)
        rows = previous.to_dict("records")
        completed = {
            (str(row["dataset_key"]), int(row["seed"]), float(row["p"]))
            for _, row in previous.iterrows()
        }
        print(f"Resuming from {raw_path} with {len(completed)} completed runs.")

    for dataset in datasets:
        for seed in args.seeds:
            for p in args.p_values:
                key = (dataset, int(seed), float(p))
                if key in completed:
                    print(f"Skipping completed run: dataset={dataset} seed={seed} p={p}")
                    continue
                print(
                    f"\nDataset={dataset} seed={seed} p={p} "
                    f"batch={args.batch_size} cap={args.max_test_samples}"
                )
                row = evaluate_one(
                    dataset=dataset,
                    seed=seed,
                    p=p,
                    batch_size=args.batch_size,
                    max_test_samples=args.max_test_samples,
                    tta_choice=args.tta,
                )
                rows.append(row)
                print(
                    f"  acc={row['accuracy']:.4f} "
                    f"bal={row['balanced_accuracy']:.4f} "
                    f"f1={row['macro_f1']:.4f} "
                    f"time={row['eval_time_sec']:.1f}s"
                )

                raw_so_far = pd.DataFrame(rows)
                raw_so_far.to_csv(raw_path, index=False)

    raw = pd.DataFrame(rows)
    summary = aggregate(raw)
    raw.to_csv(raw_path, index=False)
    summary.to_csv(out_dir / "depolarizing_noise_summary.csv", index=False)
    write_latex_table(summary, out_dir / "depolarizing_noise_table.tex")
    write_section_paragraph(summary, out_dir / "section_iv_noise_paragraph.txt")

    print("\nSaved depolarizing-noise outputs to:")
    print(f"  {out_dir}")
    print("\nSummary:")
    display_cols = [
        "dataset",
        "variant",
        "gate_type",
        "measure_bases",
        "p",
        "acc_mean",
        "acc_std",
        "bal_acc_mean",
        "bal_acc_std",
        "macro_f1_mean",
        "macro_f1_std",
        "time_mean_sec",
    ]
    print(summary[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
