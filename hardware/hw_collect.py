#!/usr/bin/env python
"""Step C (local machine): fetch the hardware counts, rebuild the quantum read-out, push it
through the trained head and write every number the paper paragraph needs.

    python hw_collect.py --job hw_job.json --status      # queue position / state only
    python hw_collect.py --job hw_job.json               # fetch, evaluate, write outputs

Outputs (next to hw_job.json):
    hw_counts.csv        one row per (image, basis): counts and <P_w> per wire
    hw_features.csv      one row per image: hardware read-out, exact read-out, label, predictions
    hw_results.json      the three accuracies (exact simulator / shot-noise-only / hardware)
                         plus agreement statistics and QPU usage
    hw_paragraph.tex     the paragraph + Table S row, numbers filled in by this script
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from hw_circuit import expectations_from_counts


def head_predict(z, h):
    """numpy copy of the trained head: LayerNorm -> Linear -> GELU -> Linear."""
    mu = h.mean(1, keepdims=True)
    var = h.var(1, keepdims=True)
    hn = (h - mu) / np.sqrt(var + float(z["head_ln_eps"])) * z["head_ln_w"] + z["head_ln_b"]
    a = hn @ z["head_w1"].T + z["head_b1"]
    a = 0.5 * a * (1.0 + np.vectorize(math.erf)(a / math.sqrt(2.0)))
    return (a @ z["head_w2"].T + z["head_b2"]).argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="hw_job.json")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    info = json.load(open(a.job))
    out_dir = Path(a.job).resolve().parent

    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService()
    job = service.job(info["job_id"])
    st = job.status()
    print(f"job {info['job_id']} on {info['backend']}: {st}")
    if a.status or str(st) not in ("DONE", "JobStatus.DONE"):
        try:
            print("queue position:", job.queue_position())
        except Exception:
            pass
        if not a.status:
            print("not finished yet")
        return

    z = np.load(info["package"], allow_pickle=False)
    n_q = int(z["n_qubits"])
    bl = info["bases"]
    n = info["n_images"]
    alpha, summary, q_sim, y, pred_sim = z["alpha"][:n], z["summary"][:n], z["q_sim"][:n], z["y"][:n], z["pred_sim"][:n]

    # sanity: the numpy head must reproduce the PennyLane/torch predictions on the exact read-out
    rep = head_predict(z, np.concatenate([q_sim, alpha, summary], 1))
    assert (rep == pred_sim).all(), "numpy head disagrees with the torch head"

    res = job.result()
    q_hw = np.zeros_like(q_sim)
    rows = []
    for k, (i, b) in enumerate(info["index"]):
        pub = res[k]
        bits = pub.data.c if hasattr(pub.data, "c") else next(iter(pub.data.values()))
        counts = bits.get_counts()
        e = expectations_from_counts(counts, n_q)
        j = bl.index(b)
        q_hw[i, j * n_q:(j + 1) * n_q] = e
        rows.append({"image": i, "basis": b, "shots": sum(counts.values()), **{f"e{w}": e[w] for w in range(n_q)},
                     "counts": json.dumps(counts, separators=(",", ":"))})
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_dir / "hw_counts.csv", index=False)

    pred_hw = head_predict(z, np.concatenate([q_hw, alpha, summary], 1))
    acc_hw = float((pred_hw == y).mean())
    acc_sim = float((pred_sim == y).mean())
    shot_acc = z["shot_acc"]
    agree = float((pred_hw == pred_sim).mean())
    err = q_hw - q_sim
    feat = pd.DataFrame({"image": np.arange(n), "test_idx": z["test_idx"][:n], "y": y, "pred_sim": pred_sim, "pred_hw": pred_hw})
    for c in range(q_sim.shape[1]):
        feat[f"q_sim_{c}"] = q_sim[:, c]
        feat[f"q_hw_{c}"] = q_hw[:, c]
    feat.to_csv(out_dir / "hw_features.csv", index=False)

    usage = None
    try:
        usage = job.usage()
    except Exception:
        try:
            usage = job.metrics().get("usage", {}).get("quantum_seconds")
        except Exception:
            pass

    # per-class accuracies, for the reader who wants to see where the drop lands
    classes = np.unique(y)
    per_class = {int(c): {"n": int((y == c).sum()), "sim": float((pred_sim[y == c] == c).mean()),
                          "hw": float((pred_hw[y == c] == c).mean())} for c in classes}

    results = {
        "backend": info["backend"], "job_id": info["job_id"], "n_images": n, "shots": info["shots"],
        "bases": bl, "n_qubits": n_q, "gate": str(z["gate"]), "dataset": str(z["dataset"]), "seed": int(z["seed"]),
        "mitigation": info.get("mitigation"), "physical_qubits": info.get("physical_qubits"),
        "twoq_gates_per_circuit": [info.get("twoq_min"), info.get("twoq_max")],
        "depth": [info.get("depth_min"), info.get("depth_max")],
        "acc_exact_sim": acc_sim,
        "acc_shot_noise_mean": float(shot_acc.mean()), "acc_shot_noise_sd": float(shot_acc.std(ddof=1)),
        "shot_repeats": int(len(shot_acc)),
        "acc_hardware": acc_hw, "prediction_agreement_hw_vs_sim": agree,
        "readout_mae": float(np.abs(err).mean()), "readout_rmse": float(np.sqrt((err ** 2).mean())),
        "readout_corr": float(np.corrcoef(q_hw.ravel(), q_sim.ravel())[0, 1]),
        "readout_shrinkage_slope": float(np.polyfit(q_sim.ravel(), q_hw.ravel(), 1)[0]),
        "qpu_usage": usage, "per_class": per_class,
    }
    json.dump(results, open(out_dir / "hw_results.json", "w"), indent=2)

    pct = lambda v: f"{100 * v:.1f}"
    twoq = f"{info.get('twoq_min')}--{info.get('twoq_max')}" if info.get("twoq_min") != info.get("twoq_max") else f"{info.get('twoq_min')}"
    paragraph = rf"""% generated by hw_collect.py from {info['job_id']} -- do not edit numbers by hand
\paragraph{{Hardware check}}
To see whether the read-out survives a real device, we ran the trained {str(z['dataset'])} circuit (seed {int(z['seed'])}, {n_q} qubits, {str(z['gate'])} entanglers, {twoq} native two-qubit gates after compilation) on \texttt{{{info['backend']}}} for a stratified subset of {n} test images, with all trained weights fixed and test-time augmentation off, at {info['shots']} shots per circuit{' with dynamical decoupling and Pauli twirling' if info.get('mitigation') else ''}. The classical head then received the hardware expectation values in place of the simulator values. Accuracy on the subset was {pct(acc_sim)}\% with the exact simulator, {pct(shot_acc.mean())}$\pm${pct(shot_acc.std(ddof=1))}\% with shot noise alone ({len(shot_acc)} sampled replicas), and {pct(acc_hw)}\% on hardware; the hardware prediction agreed with the simulator prediction on {pct(agree)}\% of the images. The hardware read-out tracks the exact one with a correlation of {results['readout_corr']:.2f} and a shrinkage slope of {results['readout_shrinkage_slope']:.2f}, which is the contraction toward zero that depolarising noise produces. This is a proof of concept on one seed and one device, not a deployment claim.

% Table S row
{str(z['dataset'])} & \texttt{{{info['backend']}}} & {n} & {info['shots']} & {pct(acc_sim)} & {pct(shot_acc.mean())} $\pm$ {pct(shot_acc.std(ddof=1))} & {pct(acc_hw)} & {pct(agree)} \\
"""
    (out_dir / "hw_paragraph.tex").write_text(paragraph)
    print(json.dumps({k: v for k, v in results.items() if k != "per_class"}, indent=2))
    print("per class:", per_class)
    print("wrote hw_counts.csv, hw_features.csv, hw_results.json, hw_paragraph.tex in", out_dir)


if __name__ == "__main__":
    main()
