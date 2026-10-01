#!/usr/bin/env python
"""Step A of the IBM hardware check: export everything the quantum layer needs
from one trained S2Q-SEQNN checkpoint, without touching the trained weights.

Runs wherever the repo, the data and the checkpoint are (cluster node
or a laptop). Needs the repo's sq_seqnn_fast.py, the run directory
(results_<ds>.json + sq_seqnn_<ds>.pt) and S2Q_DATA_ROOT.

    python hw_export.py --src src \
        --run-dir src/sq_seqnn_outputs/overhead_seqnn5_main__overhead_adaptive__seed42 \
        --n-samples 100 --shots 4000 --out hw_package.npz

What it writes (hw_package.npz):
    alpha      (N, n_q)     encoding angles of the N chosen test images
    q_w        (P,)         trained circuit parameters
    summary    (N, d)       classical skip features the head also receives
    q_sim      (N, n_out)   exact simulator read-out (PennyLane, noiseless)
    y          (N,)         labels
    pred_sim   (N,)         head prediction from q_sim (TTA off)
    head_*                  LayerNorm + Linear + GELU + Linear weights
    shot_acc   (R,)         accuracy of R shot-noise-only replicas (Aer-free,
                            sampled from the exact X-basis distribution)
    plus config strings (gate, bases, n_qubits, n_reup, ablation, angle_mode)
and hw_export_summary.json with the reference accuracies.

TTA is switched off for the subset because each TTA view would be a separate
hardware circuit; the full-test accuracy with the checkpoint's TTA is also
recorded so the reader can see both.
"""
import argparse
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np


def import_model(src, meta):
    ds = {"Overhead MNIST": "overhead", "SAT-6": "sat6", "So2Sat LCZ42": "so2sat"}[meta["dataset"]]
    argv = ["sq_seqnn_fast.py", "--dataset", ds, "--seed", str(meta["seed"]),
            "--n-qubits", str(meta["n_qubits"]),
            "--gate-type", meta["gate_type"], "--measure-bases", meta["measure_bases"],
            "--frontend-mode", meta["frontend_mode"], "--ablation", meta["ablation"],
            "--compress-mode", meta["compress_mode"], "--summary-mode", meta["summary_mode"],
            "--angle-mode", meta["angle_mode"], "--cls-hidden", str(meta["cls_hidden"]),
            "--scat-grid", str(meta["scat_grid"]), "--lr", str(meta["lr"]),
            "--lr-q-factor", str(meta["lr_q_factor"]), "--label-smooth", str(meta["label_smooth"]),
            "--tta", "none"]
    if ds == "overhead":
        argv += ["--overhead-mode", meta["overhead_mode"] or "seqnn5"]
    if meta.get("conv_width") is not None:
        argv += ["--conv-width", str(meta["conv_width"])]
    if meta.get("shapelet_count") is not None:
        argv += ["--shapelet-count", str(meta["shapelet_count"]),
                 "--shapelet-length", str(meta["shapelet_length"])]
        if meta.get("shapelet_stats"):
            argv += ["--shapelet-stats", str(meta["shapelet_stats"])]
    src_text = (Path(src) / "sq_seqnn_fast.py").read_text()
    if "--n-reupload" in src_text and int(meta.get("n_reuploading", 2)) != 2:
        argv += ["--n-reupload", str(meta["n_reuploading"])]
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    old = sys.argv
    sys.argv = argv
    try:
        sys.modules.pop("sq_seqnn_fast", None)
        m = importlib.import_module("sq_seqnn_fast")
    finally:
        sys.argv = old
    return m, ds


def stratified_subset(y, n, seed):
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    per = n // len(classes)
    idx = []
    for c in classes:
        pool = np.flatnonzero(y == c)
        idx.extend(rng.choice(pool, size=min(per, len(pool)), replace=False).tolist())
    rest = np.setdiff1d(np.arange(len(y)), np.array(idx))
    if len(idx) < n:
        idx.extend(rng.choice(rest, size=n - len(idx), replace=False).tolist())
    return np.array(sorted(idx))


def bases_list(measure_bases):
    return {"x": ["x"], "y": ["y"], "z": ["z"], "xz": ["x", "z"], "xyz": ["z", "x", "y"]}[measure_bases]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder containing sq_seqnn_fast.py")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--shot-repeats", type=int, default=20)
    ap.add_argument("--out", default="hw_package.npz")
    a = ap.parse_args()

    run_dir = Path(a.run_dir)
    meta = json.load(open(next(run_dir.glob("results_*.json"))))
    m, ds = import_model(a.src, meta)
    import torch
    qml = m.qml

    _, _, _, _, te_x, te_y = m.load_data(ds)
    model = m.SQ_SEQNN(m.META[ds]["C"], m.META[ds]["size"], m.META[ds]["K"])
    sd = torch.load(run_dir / f"sq_seqnn_{ds}.pt", map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    n_q, n_out = m.N_QUBITS, m.N_Q_OUT

    # reference: full test set, checkpoint TTA (should reproduce the run log)
    full_pred_tta = m.predict(model, te_x, torch.device("cpu"), tta_mode=meta["tta_mode"])
    full_acc_tta = float((full_pred_tta == te_y).mean())
    full_pred = m.predict(model, te_x, torch.device("cpu"), tta_mode="none")
    full_acc = float((full_pred == te_y).mean())
    print(f"full test: acc={full_acc_tta:.4f} (tta={meta['tta_mode']}, run log {meta['accuracy']:.4f})  "
          f"acc={full_acc:.4f} (tta=none)")

    idx = stratified_subset(te_y, a.n_samples, a.subset_seed)
    x = torch.as_tensor(te_x[idx], dtype=torch.float32)
    y = te_y[idx]

    # capture the fused vector h = [q | alpha | summary] at the head input
    captured = {}
    hook = model.clf.register_forward_hook(lambda mod, inp, out: captured.__setitem__("h", inp[0].detach().clone()))
    with torch.no_grad():
        logits = model(x)
    hook.remove()
    h = captured["h"].numpy().astype(np.float64)
    q_sim, alpha, summary = h[:, :n_out], h[:, n_out:n_out + n_q], h[:, n_out + n_q:]
    pred_sim = logits.argmax(1).numpy()
    sub_acc = float((pred_sim == y).mean())
    print(f"subset n={len(y)}: exact-simulator acc={sub_acc:.4f} (tta=none)")

    # shot-noise-only replicas: sample bitstrings from the exact rotated-basis distribution
    theta = sd["q_w"].detach().cpu().numpy().astype(np.float64)
    bl = bases_list(m.MEASURE_BASES)
    dev = qml.device("default.qubit", wires=n_q)

    def rotate(basis):
        for q in range(n_q):
            if basis == "x":
                qml.Hadamard(wires=q)
            elif basis == "y":
                qml.adjoint(qml.S)(wires=q)
                qml.Hadamard(wires=q)

    @qml.qnode(dev)
    def probs(features, weights, basis):
        ptr = 0
        for r in range(m.N_REUP):
            if not (m.ABLATION == "no_reupload" and r > 0):
                m._encode(features, first_round=(r == 0))
            ptr = m.qcnn_sweep(weights, ptr, 0)
            ptr = m.qcnn_sweep(weights, ptr, 1)
        rotate(basis)
        return qml.probs(wires=range(n_q))

    bits = ((np.arange(2 ** n_q)[:, None] >> np.arange(n_q)[None, ::-1]) & 1)  # row k: bitstring of basis state k, wire 0 first
    rng = np.random.RandomState(a.subset_seed + 1)
    q_shot = np.zeros((a.shot_repeats, len(y), n_out))
    for i in range(len(y)):
        for j, b in enumerate(bl):
            p = np.asarray(probs(alpha[i], theta, b), dtype=np.float64)
            p = p / p.sum()
            exact = 1.0 - 2.0 * (p[:, None] * bits).sum(0)
            assert np.allclose(exact, q_sim[i, j * n_q:(j + 1) * n_q], atol=1e-5), "basis bookkeeping mismatch"
            counts = rng.multinomial(a.shots, p, size=a.shot_repeats)
            q_shot[:, i, j * n_q:(j + 1) * n_q] = 1.0 - 2.0 * (counts @ bits) / a.shots
    shot_acc = []
    with torch.no_grad():
        for r in range(a.shot_repeats):
            hr = torch.as_tensor(np.concatenate([q_shot[r], alpha, summary], 1), dtype=torch.float32)
            shot_acc.append(float((model.clf(hr).argmax(1).numpy() == y).mean()))
    shot_acc = np.array(shot_acc)
    print(f"shot-noise-only ({a.shots} shots, {a.shot_repeats} replicas): acc={shot_acc.mean():.4f} +- {shot_acc.std(ddof=1):.4f}")

    clf = model.clf
    np.savez(a.out,
             alpha=alpha, q_w=theta, summary=summary, q_sim=q_sim, y=y, pred_sim=pred_sim, test_idx=idx,
             q_shot=q_shot, shot_acc=shot_acc,
             head_ln_w=clf[0].weight.detach().numpy(), head_ln_b=clf[0].bias.detach().numpy(),
             head_ln_eps=np.array(clf[0].eps),
             head_w1=clf[1].weight.detach().numpy(), head_b1=clf[1].bias.detach().numpy(),
             head_w2=clf[4].weight.detach().numpy(), head_b2=clf[4].bias.detach().numpy(),
             gate=m.GATE_TYPE, bases=m.MEASURE_BASES, n_qubits=n_q, n_reup=m.N_REUP,
             ablation=m.ABLATION, angle_mode=m.ANGLE_MODE, dataset=ds, seed=meta["seed"], shots=a.shots)
    summary_json = {
        "run_dir": str(run_dir), "dataset": ds, "seed": meta["seed"], "gate": m.GATE_TYPE,
        "bases": m.MEASURE_BASES, "n_qubits": n_q, "n_reup": m.N_REUP, "total_params": meta["total_params"],
        "full_test_n": int(len(te_y)), "full_acc_tta": full_acc_tta, "tta_mode": meta["tta_mode"],
        "run_log_acc": meta["accuracy"], "full_acc_no_tta": full_acc,
        "subset_n": int(len(y)), "subset_seed": a.subset_seed, "subset_acc_exact": sub_acc,
        "shots": a.shots, "shot_repeats": a.shot_repeats,
        "subset_acc_shot_mean": float(shot_acc.mean()), "subset_acc_shot_sd": float(shot_acc.std(ddof=1)),
    }
    json.dump(summary_json, open(Path(a.out).with_name("hw_export_summary.json"), "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
