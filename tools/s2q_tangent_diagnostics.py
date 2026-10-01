#!/usr/bin/env python
"""
Tangent-space diagnostics for the S2Q-SEQNN quantum branch.

For every selected run directory under src/sq_seqnn_outputs/ this script
loads the trained quantum parameters theta_final (state_dict key "q_w"),
finds theta_init from the frozen_quantum twin run with the same seed (in
frozen_quantum runs q_w never trains, and the seed fixes the draw, so its
q_w is exactly the initial point of the full run), and evaluates at both
points, averaged over a set of encoding-angle vectors alpha:

    A        alignment   ||P_T chi_perp||^2 / ||chi_perp||^2       (Prop. 1)
    chi2     residual    ||chi_perp||^2 = <O^2> - <O>^2
    d_eff    rank of the Fubini-Study metric g
    ||dC||^2 Euclidean gradient norm of C = <O>
    ||dC||_g natural-gradient norm, computed two ways (Prop. 1 check)

O runs over the single-qubit Paulis the run actually measures (its
measure_bases), and the reported values are the mean over those observables
and over alpha. The relative displacement ||theta_final - theta_init|| /
||theta_init|| is reported per run.

Angles alpha come either from real validation images pushed through the
run's own trained frontend (--alpha-source data, default, needs the data
root) or from the uniform encoding domain (--alpha-source random).

Run on Panther from the bundle root:

    python s2q_tangent_diagnostics.py --bundle . --selftest
    python s2q_tangent_diagnostics.py --bundle . \
        --configs overhead_adaptive,sat6,so2sat_adaptive --seeds 42-51 \
        --n-alpha 32 --out tangent_diagnostics.csv

The circuit is re-implemented here in plain PennyLane/NumPy so the Jacobian
can be taken without torch. --selftest compares it against _circuit() in
src/sq_seqnn_fast.py on random inputs and refuses to continue if they differ.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    import pennylane as qml
except ImportError as exc:  # pragma: no cover
    sys.exit(f"PennyLane is required: {exc}")

EPS_FD = 1e-6


# --------------------------------------------------------------------------
# Circuit (mirrors sq_seqnn_fast.py: iqp_encode, qcnn_pair, qcnn_sweep,
# _encode, _circuit). Keep these in sync with the training script.
# --------------------------------------------------------------------------

def _apply_entangler(gate_type, param, q0, q1):
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


def make_circuit_ops(n_qubits, n_reup, gate_type, ablation):
    """Return a function ops(features, weights) that queues the S2Q circuit."""
    pair_params = 8

    def iqp_encode(f, include_hadamards):
        if include_hadamards:
            for q in range(n_qubits):
                qml.Hadamard(wires=q)
        for q in range(n_qubits):
            qml.RZ(2.0 * f[q], wires=q)
        for q in range(n_qubits - 1):
            qml.CNOT(wires=[q, q + 1])
            qml.RZ(2.0 * f[q] * f[q + 1], wires=q + 1)
            qml.CNOT(wires=[q, q + 1])

    def encode(f, first_round):
        if ablation == "no_iqp":
            for q in range(n_qubits):
                qml.RY(2.0 * f[q], wires=q)
        else:
            iqp_encode(f, include_hadamards=first_round)

    def qcnn_pair(w, q0, q1):
        qml.RY(w[0], wires=q0)
        qml.RY(w[1], wires=q1)
        _apply_entangler(gate_type, w[2], q0, q1)
        qml.RX(w[3], wires=q0)
        qml.RX(w[4], wires=q1)
        _apply_entangler(gate_type, w[5], q0, q1)
        qml.RY(w[6], wires=q0)
        qml.RY(w[7], wires=q1)

    def qcnn_sweep(weights, ptr, start_idx):
        for q in range(start_idx, n_qubits - 1, 2):
            qcnn_pair(weights[ptr:ptr + pair_params], q, q + 1)
            ptr += pair_params
        return ptr

    def ops(features, weights):
        ptr = 0
        for upload_idx in range(n_reup):
            if not (ablation == "no_reupload" and upload_idx > 0):
                encode(features, first_round=(upload_idx == 0))
            ptr = qcnn_sweep(weights, ptr, 0)
            ptr = qcnn_sweep(weights, ptr, 1)

    return ops


def observables_for(measure_bases, n_qubits):
    """Single-qubit Paulis the run measures, in the order _circuit returns."""
    P = {"x": qml.PauliX, "y": qml.PauliY, "z": qml.PauliZ}
    if measure_bases in ("x", "y", "z"):
        order = [measure_bases]
    elif measure_bases == "xz":
        order = ["x", "z"]
    else:  # "xyz" -> z, x, y (matches _circuit)
        order = ["z", "x", "y"]
    obs = []
    for b in order:
        for i in range(n_qubits):
            obs.append((f"{b.upper()}{i}", P[b](i)))
    return obs


def build_state_fn(n_qubits, n_reup, gate_type, ablation):
    dev = qml.device("default.qubit", wires=n_qubits)
    ops = make_circuit_ops(n_qubits, n_reup, gate_type, ablation)

    @qml.qnode(dev)
    def state_fn(features, weights):
        ops(features, weights)
        return qml.state()

    return state_fn


# --------------------------------------------------------------------------
# Tangent-space linear algebra (same conventions as tangent_space_diagnostics.py)
# --------------------------------------------------------------------------

def state_jacobian(state_fn, alpha, theta):
    psi0 = np.asarray(state_fn(alpha, theta), dtype=complex)
    d, L = psi0.shape[0], theta.shape[0]
    J = np.zeros((d, L), dtype=complex)
    for l in range(L):
        tp = theta.copy(); tp[l] += EPS_FD
        tm = theta.copy(); tm[l] -= EPS_FD
        J[:, l] = (np.asarray(state_fn(alpha, tp)) - np.asarray(state_fn(alpha, tm))) / (2.0 * EPS_FD)
    return psi0, J


def covariant(psi, J):
    return J - np.outer(psi, psi.conj() @ J)


def real_embed(V):
    return np.vstack([V.real, V.imag])


def projector_and_rank(Jr, rcond):
    U, s, _ = np.linalg.svd(Jr, full_matrices=False)
    r = int(np.sum(s > rcond * s.max())) if s.size else 0
    Ur = U[:, :r]
    return Ur, s, r


def diagnostics_at(state_fn, alpha, theta, obs_mats, rcond):
    """All quantities at one (alpha, theta), averaged over the observables."""
    psi, J = state_jacobian(state_fn, alpha, theta)
    D = covariant(psi, J)                    # covariant tangent vectors |D_l psi>
    Dr = real_embed(D)                       # 2d x L real embedding
    Ur, svals, deff = projector_and_rank(Dr, rcond)
    g = Dr.T @ Dr                            # Fubini-Study metric (real, L x L)
    g_pinv = np.linalg.pinv(g, rcond=rcond)
    out = {"A": [], "chi2": [], "grad2": [], "natgrad2_prop1": [], "natgrad2_direct": []}
    for O in obs_mats:
        Opsi = O @ psi
        expO = float(np.real(psi.conj() @ Opsi))
        chi = Opsi - expO * psi                # |chi_perp>
        chi2 = float(np.real(chi.conj() @ chi))
        chi_r = real_embed(chi.reshape(-1, 1)).ravel()
        proj2 = float(np.sum((Ur.T @ chi_r) ** 2))
        A = proj2 / chi2 if chi2 > 1e-14 else 0.0
        # dC/dtheta_l = 2 Re <D_l psi | chi_perp>
        grad = 2.0 * np.real(D.conj().T @ chi)
        grad2 = float(grad @ grad)
        natgrad2_direct = float(grad @ g_pinv @ grad)
        out["A"].append(A)
        out["chi2"].append(chi2)
        out["grad2"].append(grad2)
        out["natgrad2_prop1"].append(4.0 * A * chi2)
        out["natgrad2_direct"].append(natgrad2_direct)
    res = {k: float(np.mean(v)) for k, v in out.items()}
    res["deff"] = deff
    res["svals"] = svals
    return res


# --------------------------------------------------------------------------
# Run discovery and parameter loading
# --------------------------------------------------------------------------

def parse_seeds(spec):
    seeds = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            seeds.extend(range(int(a), int(b) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


def find_run_dir(outputs_root, ds, config, seed, campaign, suffix=""):
    """Directory name pattern used by the Panther campaign."""
    prefix = "overhead_seqnn5" if ds == "overhead" else f"{ds}_default"
    name = f"{prefix}_{campaign}__{config}{suffix}__seed{seed}"
    p = outputs_root / name
    return p if p.is_dir() else None


def load_run(run_dir, ds):
    import torch
    meta = json.load(open(run_dir / f"results_{ds}.json"))
    sd = torch.load(run_dir / f"sq_seqnn_{ds}.pt", map_location="cpu")
    if "q_w" not in sd:
        raise KeyError(f"{run_dir}: no q_w in state_dict (keys: {list(sd)[:6]}...)")
    theta = sd["q_w"].detach().cpu().numpy().astype(float)
    return meta, theta, sd


def dataset_of(config):
    return config.split("_")[0]


# --------------------------------------------------------------------------
# alpha sources
# --------------------------------------------------------------------------

def alphas_random(n_alpha, n_qubits, angle_mode, rng):
    if angle_mode == "signed":
        return rng.uniform(-math.pi / 2, math.pi / 2, size=(n_alpha, n_qubits))
    return rng.uniform(0.0, math.pi, size=(n_alpha, n_qubits))


def alphas_from_data(bundle, run_dir, meta, sd, n_alpha, seed):
    """Push validation images through the run's trained frontend to get alpha.

    Imports src/sq_seqnn_fast.py with argv rebuilt from results_<ds>.json,
    so every global (frontend mode, gate, basis, qubits, ...) matches the run.
    """
    import importlib
    import torch

    ds = {"Overhead MNIST": "overhead", "SAT-6": "sat6", "So2Sat LCZ42": "so2sat"}[meta["dataset"]]
    argv = ["sq_seqnn_fast.py", "--dataset", ds, "--seed", str(meta["seed"]),
            "--n-qubits", str(meta["n_qubits"]), "--n-reupload", str(meta["n_reuploading"]),
            "--gate-type", meta["gate_type"], "--measure-bases", meta["measure_bases"],
            "--frontend-mode", meta["frontend_mode"], "--ablation", meta["ablation"],
            "--compress-mode", meta["compress_mode"], "--summary-mode", meta["summary_mode"],
            "--angle-mode", meta["angle_mode"], "--cls-hidden", str(meta["cls_hidden"]),
            "--scat-grid", str(meta["scat_grid"]), "--lr", str(meta["lr"]),
            "--lr-q-factor", str(meta["lr_q_factor"]), "--label-smooth", str(meta["label_smooth"])]
    if ds == "overhead":
        argv += ["--overhead-mode", meta["overhead_mode"] or "seqnn5"]
    if meta.get("conv_width") is not None:
        argv += ["--conv-width", str(meta["conv_width"])]
    if meta.get("shapelet_count") is not None:
        argv += ["--shapelet-count", str(meta["shapelet_count"]),
                 "--shapelet-length", str(meta["shapelet_length"])]
        if meta.get("shapelet_stats"):
            argv += ["--shapelet-stats", str(meta["shapelet_stats"])]

    src = str(bundle / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    old_argv = sys.argv
    sys.argv = argv
    try:
        if "sq_seqnn_fast" in sys.modules:
            del sys.modules["sq_seqnn_fast"]
        m = importlib.import_module("sq_seqnn_fast")
    finally:
        sys.argv = old_argv

    _, _, val_x, val_y, _, _ = m.load_data(ds)
    model = m.SQ_SEQNN(m.META[ds]["C"], m.META[ds]["size"], m.META[ds]["K"])
    model.load_state_dict(sd)
    model.eval()
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(val_x), size=min(n_alpha, len(val_x)), replace=False)
    x = torch.as_tensor(np.asarray(val_x)[idx], dtype=torch.float32)
    with torch.no_grad():
        if model.frontend_mode in ("global", "shapelet"):
            s = model.scat(x)
            alpha = m.to_angles(model.compress(s))
        else:
            tokens = model.scat(x)
            pooled, weights = model.region_pool(tokens)
            if m.COMPRESS_MODE == "flat":
                alpha = m.to_angles(model.compress(tokens.flatten(1)))
            else:
                alpha_tokens = model.compress(tokens)
                alpha = m.to_angles((alpha_tokens * weights.unsqueeze(-1)).sum(dim=1))
    y = torch.as_tensor(np.asarray(val_y)[idx], dtype=torch.long)
    return alpha.cpu().numpy().astype(float), m, model, x, y


def task_gradient_and_head_norms(m, model, x, y, theta, label_smooth):
    """||dL/dq_w||^2 of the real training loss through the full model at theta,
    plus the Frobenius norms of the first classifier layer's column blocks for
    the three fused streams [q | alpha | skip]. Says how much of the per-Pauli
    gradient actually reaches the circuit through the head."""
    import torch
    import torch.nn as nn
    model.eval()
    with torch.no_grad():
        model.q_w.copy_(torch.as_tensor(theta, dtype=model.q_w.dtype))
    model.q_w.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    crit = nn.CrossEntropyLoss(label_smoothing=float(label_smooth))
    loss = crit(model(x), y)
    loss.backward()
    g = model.q_w.grad.detach().cpu().numpy().astype(float)
    # classical reference: mean squared gradient per parameter over every
    # trainable parameter that is not the circuit, on the same batch
    cls_sq, cls_n = 0.0, 0
    for name, prm in model.named_parameters():
        if name == "q_w" or prm.grad is None:
            continue
        cls_sq += float((prm.grad.detach() ** 2).sum().item())
        cls_n += prm.numel()
    W = model.clf[1].weight.detach().cpu().numpy()
    nq_out = m.N_Q_OUT if m.ABLATION not in ("classical_only",) else 0
    nq = m.N_QUBITS
    Wq, Wa, Ws = W[:, :nq_out], W[:, nq_out:nq_out + nq], W[:, nq_out + nq:]
    fro = lambda A: float(np.sqrt(np.sum(A ** 2))) if A.size else 0.0
    return dict(task_grad2=float(g @ g), loss=float(loss.item()),
                q_grad2_pp=float(g @ g) / max(g.size, 1),
                cls_grad2_pp=cls_sq / max(cls_n, 1), cls_n_params=cls_n,
                head_w_q=fro(Wq), head_w_alpha=fro(Wa), head_w_skip=fro(Ws))


# --------------------------------------------------------------------------
# Self-test against the training script's own circuit
# --------------------------------------------------------------------------

def selftest(bundle):
    import importlib
    import torch
    src = str(bundle / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    rng = np.random.default_rng(0)
    worst = 0.0
    for ds, nq, gate, basis, abl in [("overhead", 8, "CX", "x", "none"),
                                      ("sat6", 10, "XX", "x", "none"),
                                      ("so2sat", 10, "CRY", "xyz", "none"),
                                      ("sat6", 10, "XX", "x", "no_reupload"),
                                      ("sat6", 10, "XX", "x", "no_iqp")]:
        old = sys.argv
        sys.argv = ["sq_seqnn_fast.py", "--dataset", ds, "--n-qubits", str(nq),
                    "--gate-type", gate, "--measure-bases", basis, "--ablation", abl]
        try:
            if "sq_seqnn_fast" in sys.modules:
                del sys.modules["sq_seqnn_fast"]
            m = importlib.import_module("sq_seqnn_fast")
        finally:
            sys.argv = old
        n_reup = m.N_REUP
        L = m.N_Q_PARAMS
        state_fn = build_state_fn(nq, n_reup, gate, abl)
        obs = observables_for(basis, nq)
        for _ in range(3):
            alpha = rng.uniform(0, math.pi, nq)
            theta = rng.uniform(0, 2 * math.pi, L)
            ref = torch.stack(m._circuit(torch.tensor(alpha, dtype=torch.float32),
                                         torch.tensor(theta, dtype=torch.float32))).numpy()
            psi = np.asarray(state_fn(alpha, theta))
            mine = np.array([np.real(psi.conj() @ (qml.matrix(o, wire_order=range(nq)) @ psi)) for _, o in obs])
            worst = max(worst, float(np.max(np.abs(ref - mine))))
        print(f"  selftest {ds:8s} nq={nq} gate={gate} basis={basis} ablation={abl}: max|diff| so far {worst:.2e}")
    ok = worst < 1e-4   # float32 in the torch path vs float64 here
    print("SELFTEST", "PASSED" if ok else "FAILED", f"(max abs diff {worst:.2e})")
    return ok


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="Panther bundle root (contains src/)")
    ap.add_argument("--configs", default="overhead_adaptive,sat6,so2sat_adaptive")
    ap.add_argument("--seeds", default="42-51")
    ap.add_argument("--campaign", default="main", help="campaign of the trained run")
    ap.add_argument("--suffix", default="", help="config suffix in the run dir, e.g. __sharedLR")
    ap.add_argument("--n-alpha", type=int, default=32)
    ap.add_argument("--alpha-source", choices=["data", "random"], default="data")
    ap.add_argument("--rcond", type=float, default=1e-8, help="relative SV cut for d_eff")
    ap.add_argument("--out", default="tangent_diagnostics.csv")
    ap.add_argument("--svals-dir", default="tangent_svals", help="where per-run singular values go")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    bundle = Path(args.bundle).expanduser().resolve()
    outputs_root = bundle / "src" / "sq_seqnn_outputs"
    if not outputs_root.is_dir():
        sys.exit(f"not found: {outputs_root}")

    if args.selftest:
        if not selftest(bundle):
            sys.exit(1)
        if args.configs == "" :
            return

    Path(args.svals_dir).mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    fields = ["config", "dataset", "seed", "campaign", "gate", "basis", "frontend", "n_qubits",
              "n_reup", "n_params", "hilbert_dim", "alpha_source", "n_alpha", "rcond",
              "theta_init_source", "disp_abs", "disp_rel",
              "A_init", "A_final", "chi2_init", "chi2_final", "deff_init", "deff_final",
              "grad2_init", "grad2_final", "natgrad2_init", "natgrad2_final",
              "prop1_relerr_init", "prop1_relerr_final",
              "task_grad2_init", "task_grad2_final", "loss_init", "loss_final",
              "q_grad2_pp_init", "q_grad2_pp_final", "cls_grad2_pp_init", "cls_grad2_pp_final", "cls_n_params",
              "head_w_q", "head_w_alpha", "head_w_skip", "seconds"]
    done = set()
    if args.skip_existing and Path(args.out).exists():
        with open(args.out) as f:
            for row in csv.DictReader(f):
                done.add((row["config"], int(row["seed"])))
    write_header = not Path(args.out).exists() or os.path.getsize(args.out) == 0
    fout = open(args.out, "a", newline="")
    writer = csv.DictWriter(fout, fieldnames=fields)
    if write_header:
        writer.writeheader()

    rng = np.random.default_rng(1234)
    for config in configs:
        ds = dataset_of(config)
        for seed in seeds:
            if (config, seed) in done:
                continue
            t0 = time.time()
            run_dir = find_run_dir(outputs_root, ds, config, seed, args.campaign, args.suffix)
            if run_dir is None:
                print(f"[skip] no run dir for {config} seed {seed}")
                continue
            meta, theta_final, sd = load_run(run_dir, ds)
            nq, n_reup = int(meta["n_qubits"]), int(meta["n_reuploading"])
            gate, basis, abl = meta["gate_type"], meta["measure_bases"], meta["ablation"]

            # theta_init from the frozen_quantum twin (same seed, same config)
            twin = find_run_dir(outputs_root, ds, config, seed, "isolation", "__frozen_quantum")
            if twin is not None:
                _, theta_init, _ = load_run(twin, ds)
                init_src = "frozen_quantum_twin"
            else:
                theta_init = None
                init_src = "none"
            if theta_init is not None and theta_init.shape != theta_final.shape:
                print(f"[warn] {config} seed {seed}: twin shape {theta_init.shape} != {theta_final.shape}")
                theta_init, init_src = None, "none"

            # alpha
            task = {"task_grad2_init": float("nan"), "task_grad2_final": float("nan"),
                    "loss_init": float("nan"), "loss_final": float("nan"),
                    "q_grad2_pp_init": float("nan"), "q_grad2_pp_final": float("nan"),
                    "cls_grad2_pp_init": float("nan"), "cls_grad2_pp_final": float("nan"),
                    "cls_n_params": float("nan"),
                    "head_w_q": float("nan"), "head_w_alpha": float("nan"), "head_w_skip": float("nan")}
            if args.alpha_source == "data":
                try:
                    alphas, m, model, xb, yb = alphas_from_data(bundle, run_dir, meta, sd, args.n_alpha, seed)
                    a_src = "data"
                    try:
                        tf = task_gradient_and_head_norms(m, model, xb, yb, theta_final, meta["label_smooth"])
                        task.update(task_grad2_final=tf["task_grad2"], loss_final=tf["loss"],
                                    q_grad2_pp_final=tf["q_grad2_pp"], cls_grad2_pp_final=tf["cls_grad2_pp"],
                                    cls_n_params=tf["cls_n_params"],
                                    head_w_q=tf["head_w_q"], head_w_alpha=tf["head_w_alpha"],
                                    head_w_skip=tf["head_w_skip"])
                        if theta_init is not None:
                            ti = task_gradient_and_head_norms(m, model, xb, yb, theta_init, meta["label_smooth"])
                            task.update(task_grad2_init=ti["task_grad2"], loss_init=ti["loss"],
                                        q_grad2_pp_init=ti["q_grad2_pp"], cls_grad2_pp_init=ti["cls_grad2_pp"])
                    except Exception as exc:
                        print(f"[warn] task gradient failed for {config} seed {seed}: {exc}")
                except Exception as exc:
                    print(f"[warn] data alphas failed for {config} seed {seed}: {exc}; using random")
                    alphas = alphas_random(args.n_alpha, nq, meta["angle_mode"], rng)
                    a_src = "random_fallback"
            else:
                alphas = alphas_random(args.n_alpha, nq, meta["angle_mode"], rng)
                a_src = "random"

            state_fn = build_state_fn(nq, n_reup, gate, abl)
            obs_mats = [qml.matrix(o, wire_order=range(nq)) for _, o in observables_for(basis, nq)]

            def average(theta):
                acc = {}
                svals = []
                for a in alphas:
                    r = diagnostics_at(state_fn, a, theta, obs_mats, args.rcond)
                    svals.append(r.pop("svals"))
                    for k, v in r.items():
                        acc.setdefault(k, []).append(v)
                mean = {k: float(np.mean(v)) for k, v in acc.items()}
                mean["prop1_relerr"] = float(np.mean([
                    abs(p - d) / max(abs(d), 1e-12)
                    for p, d in zip(acc["natgrad2_prop1"], acc["natgrad2_direct"])]))
                return mean, np.array(svals)

            fin, sv_fin = average(theta_final)
            if theta_init is not None:
                ini, sv_ini = average(theta_init)
                disp_abs = float(np.linalg.norm(theta_final - theta_init))
                disp_rel = disp_abs / float(np.linalg.norm(theta_init))
            else:
                ini = {k: float("nan") for k in fin}
                sv_ini = np.array([])
                disp_abs = disp_rel = float("nan")

            np.savez_compressed(Path(args.svals_dir) / f"{config}__seed{seed}.npz",
                                svals_final=sv_fin, svals_init=sv_ini, alphas=alphas,
                                theta_final=theta_final,
                                theta_init=theta_init if theta_init is not None else np.array([]))

            row = dict(config=config, dataset=ds, seed=seed, campaign=args.campaign, gate=gate,
                       basis=basis, frontend=meta["frontend_mode"], n_qubits=nq, n_reup=n_reup,
                       n_params=theta_final.shape[0], hilbert_dim=2 ** nq, alpha_source=a_src,
                       n_alpha=len(alphas), rcond=args.rcond, theta_init_source=init_src,
                       disp_abs=disp_abs, disp_rel=disp_rel,
                       A_init=ini["A"], A_final=fin["A"], chi2_init=ini["chi2"], chi2_final=fin["chi2"],
                       deff_init=ini["deff"], deff_final=fin["deff"],
                       grad2_init=ini["grad2"], grad2_final=fin["grad2"],
                       natgrad2_init=ini["natgrad2_direct"], natgrad2_final=fin["natgrad2_direct"],
                       prop1_relerr_init=ini["prop1_relerr"], prop1_relerr_final=fin["prop1_relerr"],
                       seconds=round(time.time() - t0, 1), **task)
            writer.writerow(row)
            fout.flush()
            print(f"{config:18s} seed {seed}: disp_rel={disp_rel:.4f}  "
                  f"A {ini['A']:.4f}->{fin['A']:.4f}  d_eff {ini['deff']:.0f}->{fin['deff']:.0f}  "
                  f"chi2 {ini['chi2']:.3f}->{fin['chi2']:.3f}  "
                  f"taskgrad2 {task['task_grad2_init']:.2e}->{task['task_grad2_final']:.2e}  "
                  f"headW q/a/s {task['head_w_q']:.2f}/{task['head_w_alpha']:.2f}/{task['head_w_skip']:.2f}  "
                  f"q/cls grad2 per param {task['q_grad2_pp_final']:.2e}/{task['cls_grad2_pp_final']:.2e}  "
                  f"({row['seconds']}s)", flush=True)
    fout.close()
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
