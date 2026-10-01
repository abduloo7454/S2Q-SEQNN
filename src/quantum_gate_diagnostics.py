"""
Quantum gate diagnostics for SQ-SEQNN.

This script compares candidate QCNN entangling gates using circuit-level
diagnostics:
  1. Expressibility via fidelity-distribution divergence from Haar random states.
  2. Entanglement capability via Meyer-Wallach global entanglement and central-cut
     von Neumann entropy.
  3. Barren-plateau indicators via gradient norm/variance/near-zero fraction.
  4. Parameter sensitivity via response/state change under small perturbations.
  5. Runtime for every diagnostic and every gate.

It does not train the classifier. Classification gate ablation results can be
merged with --classification-summary when available.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    import pennylane as qml
except Exception as exc:  # pragma: no cover - environment guidance
    raise RuntimeError(
        "Could not import PennyLane. Use the project virtual environment:\n"
        "  python src/quantum_gate_diagnostics.py"
    ) from exc


DEFAULT_GATES = ["ZZ", "CZ", "CX", "XY", "CRZ", "CRY", "XX", "YY"]
PARAMETRIC_ENTANGLERS = {"ZZ", "XX", "YY", "XY", "CRZ", "CRY"}
SCRIPT_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Circuit-level gate diagnostics for SQ-SEQNN.")
    parser.add_argument("--gates", nargs="+", default=DEFAULT_GATES)
    parser.add_argument("--n-qubits", type=int, default=10)
    parser.add_argument("--n-reup", type=int, default=2)
    parser.add_argument("--samples-state", type=int, default=64)
    parser.add_argument("--samples-grad", type=int, default=24)
    parser.add_argument("--bins", type=int, default=75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-threshold", type=float, default=1e-6)
    parser.add_argument("--perturb-scale", type=float, default=1e-2)
    parser.add_argument(
        "--classification-summary",
        default="",
        help="Optional gate_summary.csv/json from classification ablation to merge by gate_type.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(SCRIPT_DIR / "gate_diagnostics"),
    )
    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def q_param_count(n_qubits: int, n_reup: int) -> int:
    return (n_qubits - 1) * 8 * n_reup


def active_params_per_pair(gate: str) -> int:
    # Six trainable single-qubit rotations are always active. The two entangler
    # angles are active only for parameterized two-qubit gates.
    return 8 if gate in PARAMETRIC_ENTANGLERS else 6


def apply_entangler(gate: str, param: torch.Tensor, q0: int, q1: int) -> None:
    if gate == "ZZ":
        qml.IsingZZ(param, wires=[q0, q1])
    elif gate == "XX":
        qml.IsingXX(param, wires=[q0, q1])
    elif gate == "YY":
        qml.IsingYY(param, wires=[q0, q1])
    elif gate == "XY":
        qml.IsingXY(param, wires=[q0, q1])
    elif gate == "CRZ":
        qml.CRZ(param, wires=[q0, q1])
    elif gate == "CRY":
        qml.CRY(param, wires=[q0, q1])
    elif gate == "CZ":
        qml.CZ(wires=[q0, q1])
    elif gate == "CX":
        qml.CNOT(wires=[q0, q1])
    else:
        raise ValueError(f"Unsupported gate: {gate}")


def iqp_encode(features: torch.Tensor, n_qubits: int, include_hadamards: bool) -> None:
    if include_hadamards:
        for q in range(n_qubits):
            qml.Hadamard(wires=q)
    for q in range(n_qubits):
        qml.RZ(2.0 * features[q], wires=q)
    for q in range(n_qubits - 1):
        qml.CNOT(wires=[q, q + 1])
        qml.RZ(2.0 * features[q] * features[q + 1], wires=q + 1)
        qml.CNOT(wires=[q, q + 1])


def qcnn_pair(gate: str, weights: torch.Tensor, ptr: int, q0: int, q1: int) -> int:
    w = weights[ptr : ptr + 8]
    qml.RY(w[0], wires=q0)
    qml.RY(w[1], wires=q1)
    apply_entangler(gate, w[2], q0, q1)
    qml.RX(w[3], wires=q0)
    qml.RX(w[4], wires=q1)
    apply_entangler(gate, w[5], q0, q1)
    qml.RY(w[6], wires=q0)
    qml.RY(w[7], wires=q1)
    return ptr + 8


def qcnn_sweep(gate: str, weights: torch.Tensor, ptr: int, n_qubits: int, start_idx: int) -> int:
    for q in range(start_idx, n_qubits - 1, 2):
        ptr = qcnn_pair(gate, weights, ptr, q, q + 1)
    return ptr


def build_qnodes(gate: str, n_qubits: int, n_reup: int):
    dev_state = qml.device("default.qubit", wires=n_qubits)
    dev_exp = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev_state, interface="torch", diff_method="backprop")
    def state_circuit(features, weights):
        ptr = 0
        for upload_idx in range(n_reup):
            iqp_encode(features, n_qubits, include_hadamards=(upload_idx == 0))
            ptr = qcnn_sweep(gate, weights, ptr, n_qubits, 0)
            ptr = qcnn_sweep(gate, weights, ptr, n_qubits, 1)
        return qml.state()

    @qml.qnode(dev_exp, interface="torch", diff_method="backprop")
    def expval_circuit(features, weights):
        ptr = 0
        for upload_idx in range(n_reup):
            iqp_encode(features, n_qubits, include_hadamards=(upload_idx == 0))
            ptr = qcnn_sweep(gate, weights, ptr, n_qubits, 0)
            ptr = qcnn_sweep(gate, weights, ptr, n_qubits, 1)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return state_circuit, expval_circuit


def random_features(n_qubits: int) -> torch.Tensor:
    return torch.rand(n_qubits, dtype=torch.float64) * math.pi


def random_weights(n_params: int, requires_grad: bool = False) -> torch.Tensor:
    weights = torch.rand(n_params, dtype=torch.float64) * (2.0 * math.pi)
    weights.requires_grad_(requires_grad)
    return weights


def pairwise_fidelities(states: np.ndarray) -> np.ndarray:
    fidels = []
    for i in range(len(states)):
        overlaps = np.abs(states[i + 1 :] @ np.conjugate(states[i])) ** 2
        if len(overlaps):
            fidels.append(overlaps)
    if not fidels:
        return np.array([], dtype=np.float64)
    return np.concatenate(fidels).astype(np.float64)


def haar_bin_probs(n_qubits: int, bins: np.ndarray) -> np.ndarray:
    dim = 2**n_qubits
    # Integral of (D - 1)(1 - F)^(D - 2) over [a, b].
    probs = []
    for a, b in zip(bins[:-1], bins[1:]):
        probs.append((1.0 - a) ** (dim - 1) - (1.0 - b) ** (dim - 1))
    probs = np.asarray(probs, dtype=np.float64)
    return probs / probs.sum()


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64) + eps
    q = np.asarray(q, dtype=np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def meyer_wallach_entanglement(state: np.ndarray, n_qubits: int) -> float:
    psi = state.reshape([2] * n_qubits)
    purities = []
    for q in range(n_qubits):
        moved = np.moveaxis(psi, q, 0).reshape(2, -1)
        rho = moved @ np.conjugate(moved.T)
        purities.append(np.real(np.trace(rho @ rho)))
    return float(2.0 * (1.0 - np.mean(purities)))


def central_entropy(state: np.ndarray, n_qubits: int) -> float:
    cut = n_qubits // 2
    psi = state.reshape(2**cut, 2 ** (n_qubits - cut))
    rho = psi @ np.conjugate(psi.T)
    eigvals = np.linalg.eigvalsh(rho).clip(min=1e-12)
    return float(-np.sum(eigvals * np.log2(eigvals)))


def collect_states(state_circuit, n_qubits: int, n_params: int, n_samples: int) -> np.ndarray:
    states = []
    with torch.no_grad():
        for _ in range(n_samples):
            features = random_features(n_qubits)
            weights = random_weights(n_params)
            state = state_circuit(features, weights)
            states.append(state.detach().cpu().numpy())
    return np.asarray(states)


def expressibility_metrics(states: np.ndarray, n_qubits: int, n_bins: int) -> dict:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    fidels = pairwise_fidelities(states)
    hist, _ = np.histogram(fidels, bins=bins, density=False)
    model_probs = hist.astype(np.float64) / max(hist.sum(), 1)
    haar_probs = haar_bin_probs(n_qubits, bins)
    return {
        "fidelity_mean": float(np.mean(fidels)),
        "fidelity_std": float(np.std(fidels)),
        "expressibility_js": js_divergence(model_probs, haar_probs),
    }


def entanglement_metrics(states: np.ndarray, n_qubits: int) -> dict:
    mw = np.array([meyer_wallach_entanglement(s, n_qubits) for s in states])
    entropy = np.array([central_entropy(s, n_qubits) for s in states])
    return {
        "meyer_wallach_mean": float(mw.mean()),
        "meyer_wallach_std": float(mw.std()),
        "central_entropy_mean": float(entropy.mean()),
        "central_entropy_std": float(entropy.std()),
    }


def as_tensor_vector(values) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values
    return torch.stack(list(values))


def gradient_metrics(expval_circuit, n_qubits: int, n_params: int, n_samples: int, threshold: float) -> dict:
    grad_rows = []
    for _ in range(n_samples):
        features = random_features(n_qubits)
        weights = random_weights(n_params, requires_grad=True)
        vals = as_tensor_vector(expval_circuit(features, weights))
        loss = vals.square().mean()
        loss.backward()
        grad = weights.grad.detach().cpu().numpy().astype(np.float64)
        grad_rows.append(grad)
    grads = np.stack(grad_rows)
    abs_grads = np.abs(grads)
    norms = np.linalg.norm(grads, axis=1)
    return {
        "grad_abs_mean": float(abs_grads.mean()),
        "grad_abs_median": float(np.median(abs_grads)),
        "grad_variance": float(grads.var()),
        "grad_norm_mean": float(norms.mean()),
        "grad_norm_std": float(norms.std()),
        "near_zero_grad_frac": float((abs_grads < threshold).mean()),
    }


def sensitivity_metrics(
    state_circuit,
    expval_circuit,
    n_qubits: int,
    n_params: int,
    n_samples: int,
    perturb_scale: float,
) -> dict:
    response_sens = []
    state_sens = []
    with torch.no_grad():
        for _ in range(n_samples):
            features = random_features(n_qubits)
            weights = random_weights(n_params)
            delta = torch.randn(n_params, dtype=torch.float64) * perturb_scale
            perturbed = weights + delta
            y0 = as_tensor_vector(expval_circuit(features, weights)).detach()
            y1 = as_tensor_vector(expval_circuit(features, perturbed)).detach()
            s0 = state_circuit(features, weights).detach().cpu().numpy()
            s1 = state_circuit(features, perturbed).detach().cpu().numpy()
            delta_norm = float(torch.linalg.norm(delta).item()) + 1e-12
            response_sens.append(float(torch.linalg.norm(y1 - y0).item()) / delta_norm)
            fidelity = float(np.abs(np.vdot(s0, s1)) ** 2)
            state_sens.append((1.0 - fidelity) / delta_norm)
    return {
        "response_sensitivity_mean": float(np.mean(response_sens)),
        "response_sensitivity_std": float(np.std(response_sens)),
        "state_infidelity_sensitivity_mean": float(np.mean(state_sens)),
        "state_infidelity_sensitivity_std": float(np.std(state_sens)),
    }


def load_classification_summary(path: str) -> pd.DataFrame | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Classification summary not found: {p}")
    if p.suffix.lower() == ".json":
        df = pd.read_json(p)
    else:
        df = pd.read_csv(p)
    if "gate_type" not in df.columns:
        raise ValueError("Classification summary must contain a gate_type column.")
    return df


def run_gate(gate: str, args) -> dict:
    gate = gate.upper()
    n_params = q_param_count(args.n_qubits, args.n_reup)
    state_circuit, expval_circuit = build_qnodes(gate, args.n_qubits, args.n_reup)
    row = {
        "gate_type": gate,
        "n_qubits": args.n_qubits,
        "n_reup": args.n_reup,
        "qcnn_params_counted": n_params,
        "active_params_per_pair": active_params_per_pair(gate),
        "active_param_ratio": active_params_per_pair(gate) / 8.0,
    }

    t0 = time.perf_counter()
    states = collect_states(state_circuit, args.n_qubits, n_params, args.samples_state)
    row.update(expressibility_metrics(states, args.n_qubits, args.bins))
    row.update(entanglement_metrics(states, args.n_qubits))
    row["state_diagnostics_time_sec"] = round(time.perf_counter() - t0, 4)

    t1 = time.perf_counter()
    row.update(
        gradient_metrics(
            expval_circuit,
            args.n_qubits,
            n_params,
            args.samples_grad,
            args.grad_threshold,
        )
    )
    row["gradient_time_sec"] = round(time.perf_counter() - t1, 4)

    t2 = time.perf_counter()
    row.update(
        sensitivity_metrics(
            state_circuit,
            expval_circuit,
            args.n_qubits,
            n_params,
            args.samples_grad,
            args.perturb_scale,
        )
    )
    row["sensitivity_time_sec"] = round(time.perf_counter() - t2, 4)
    row["total_gate_time_sec"] = round(time.perf_counter() - t0, 4)
    return row


def add_rank_columns(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()
    ranked["expressibility_rank"] = ranked["expressibility_js"].rank(method="min", ascending=True)
    ranked["entanglement_rank"] = ranked["meyer_wallach_mean"].rank(method="min", ascending=False)
    ranked["gradient_rank"] = ranked["grad_norm_mean"].rank(method="min", ascending=False)
    ranked["stability_rank"] = ranked["near_zero_grad_frac"].rank(method="min", ascending=True)
    ranked["diagnostic_rank_mean"] = ranked[
        ["expressibility_rank", "entanglement_rank", "gradient_rank", "stability_rank"]
    ].mean(axis=1)
    ranked = ranked.sort_values(
        by=["diagnostic_rank_mean", "expressibility_js", "near_zero_grad_frac"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    ranked.insert(0, "diagnostic_rank", range(1, len(ranked) + 1))
    return ranked


def merge_classification_datasets(diagnostics_df: pd.DataFrame, cls_df: pd.DataFrame) -> pd.DataFrame:
    """Return dataset-level classification rows with diagnostics attached."""
    diag_cols = [
        c for c in diagnostics_df.columns
        if c not in {"diagnostic_rank"}
    ]
    return cls_df.merge(diagnostics_df[diag_cols], on="gate_type", how="left")


def aggregate_classification_by_gate(cls_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        c for c in [
            "mean_accuracy",
            "std_accuracy",
            "mean_balanced_accuracy",
            "std_balanced_accuracy",
            "mean_params",
        ]
        if c in cls_df.columns
    ]
    if not metric_cols:
        return cls_df[["gate_type"]].drop_duplicates().copy()
    return cls_df.groupby("gate_type", as_index=False)[metric_cols].mean()


def save_outputs(
    diagnostics_df: pd.DataFrame,
    args,
    total_time: float,
    classification_df: pd.DataFrame | None = None,
) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = diagnostics_df.copy()
    df.to_csv(out_dir / "gate_diagnostics_summary.csv", index=False)
    with open(out_dir / "gate_diagnostics_summary.json", "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)

    if classification_df is not None:
        merged_dataset = merge_classification_datasets(df, classification_df)
        merged_dataset.to_csv(out_dir / "gate_diagnostics_with_classification_by_dataset.csv", index=False)
        with open(out_dir / "gate_diagnostics_with_classification_by_dataset.json", "w") as f:
            json.dump(merged_dataset.to_dict(orient="records"), f, indent=2)

        cls_gate = aggregate_classification_by_gate(classification_df)
        merged_gate = df.merge(cls_gate, on="gate_type", how="left")
        merged_gate.to_csv(out_dir / "gate_diagnostics_with_classification_by_gate.csv", index=False)
        with open(out_dir / "gate_diagnostics_with_classification_by_gate.json", "w") as f:
            json.dump(merged_gate.to_dict(orient="records"), f, indent=2)

    metadata = {
        "gates": [g.upper() for g in args.gates],
        "n_qubits": args.n_qubits,
        "n_reup": args.n_reup,
        "samples_state": args.samples_state,
        "samples_grad": args.samples_grad,
        "bins": args.bins,
        "seed": args.seed,
        "grad_threshold": args.grad_threshold,
        "perturb_scale": args.perturb_scale,
        "classification_summary": args.classification_summary,
        "total_time_sec": round(total_time, 4),
    }
    with open(out_dir / "gate_diagnostics_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    start = time.perf_counter()
    rows = []
    for gate in args.gates:
        print("\n" + "=" * 72)
        print(f"Gate diagnostics: {gate.upper()}")
        print("=" * 72)
        gate_start = time.perf_counter()
        row = run_gate(gate, args)
        rows.append(row)
        print(
            f"Done {gate.upper()} in {time.perf_counter() - gate_start:.2f}s | "
            f"JS={row['expressibility_js']:.5f} | "
            f"MW={row['meyer_wallach_mean']:.4f} | "
            f"grad_norm={row['grad_norm_mean']:.3e}"
        )

    df = pd.DataFrame(rows)
    cls = load_classification_summary(args.classification_summary)
    df = add_rank_columns(df)
    total_time = time.perf_counter() - start
    save_outputs(df, args, total_time, classification_df=cls)

    print("\n" + "=" * 72)
    print("Saved gate diagnostics")
    print("=" * 72)
    print(f"Output folder: {args.out_dir}")
    print(f"Total time: {total_time:.2f}s")
    print(df[["diagnostic_rank", "gate_type", "expressibility_js", "meyer_wallach_mean", "grad_norm_mean", "near_zero_grad_frac", "total_gate_time_sec"]].to_string(index=False))


if __name__ == "__main__":
    main()
