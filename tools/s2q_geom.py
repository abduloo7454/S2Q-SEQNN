#!/usr/bin/env python3
"""
Geometry of the S2Q-SEQNN quantum branch across register sizes (TQE revision).

A plain-NumPy statevector simulator of the exact circuit used in training
(IQP encoding with re-uploading + QCNN kernels), so that 8-18 qubits run on a
CPU without dense observable matrices. Three modes:

  --selftest   compare this simulator with the PennyLane reference circuit in
               s2q_tangent_diagnostics.py (itself checked against the training
               script) on random inputs, every gate, 4-10 qubits. Refuses to
               continue above 1e-9.

  --bp-scan    barren-plateau scan at random parameters: for n = 4..N and
               theta ~ U[0, 2pi) (the training initialization), alpha ~ U[0, pi),
               Var_theta[dC/dtheta_k] for (i) the local read-out the model uses
               (single-qubit Paulis, averaged) and (ii) a global Z^{(x)n} cost
               as a positive control that must show a plateau. Pure NumPy.

  --trained    tangent diagnostics on trained checkpoints of the scaling
               campaign (and n_q = 10 from the main campaign), as in the paper:
               alignment A at init and after training, the 2-design level
               d_eff / [2(D-1)], rank d_eff, ||chi_perp||^2, ||grad C||^2, the
               task gradient reaching the circuit, and parameter displacement.
               Needs the checkpoints, torch and the data root (Panther).

    python s2q_geom.py --selftest --bundle .
    python s2q_geom.py --bp-scan --gates CRY,CX,XX --nmin 4 --nmax 16 --samples 200 --out bp_scan.csv
    python s2q_geom.py --bp-scan --backend torch --device cuda --nmin 4 --nmax 26 --out bp_gpu.csv
    python s2q_geom.py --trained --bundle . --nq 12 --seeds 42-51 --out geom_nq12.csv
"""
import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ gates
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
PAULI = {"x": X, "y": Y, "z": Z}


def RX(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def RY(t):
    c, s = math.cos(t / 2), math.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def RZ(t):
    return np.array([[np.exp(-0.5j * t), 0], [0, np.exp(0.5j * t)]], dtype=complex)


def two_q(gate, p):
    """4x4 matrix in PennyLane's convention, basis |q0 q1>."""
    c, s = math.cos(p / 2), math.sin(p / 2)
    if gate == "ZZ":
        e, f = np.exp(-0.5j * p), np.exp(0.5j * p)
        return np.diag([e, f, f, e]).astype(complex)
    if gate == "XX":
        return np.array([[c, 0, 0, -1j * s], [0, c, -1j * s, 0], [0, -1j * s, c, 0], [-1j * s, 0, 0, c]], dtype=complex)
    if gate == "YY":
        return np.array([[c, 0, 0, 1j * s], [0, c, -1j * s, 0], [0, -1j * s, c, 0], [1j * s, 0, 0, c]], dtype=complex)
    if gate == "XY":
        return np.array([[1, 0, 0, 0], [0, c, 1j * s, 0], [0, 1j * s, c, 0], [0, 0, 0, 1]], dtype=complex)
    if gate == "CRZ":
        return np.diag([1, 1, np.exp(-0.5j * p), np.exp(0.5j * p)]).astype(complex)
    if gate == "CRY":
        M = np.eye(4, dtype=complex); M[2:, 2:] = RY(p); return M
    if gate == "CZ":
        return np.diag([1, 1, 1, -1]).astype(complex)
    if gate == "CX":
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
    if gate == "SWAP":
        return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
    if gate == "ISWAP":
        return np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex)
    raise ValueError(gate)


CNOT = two_q("CX", 0.0)
PARAM_GATES = {"ZZ", "XX", "YY", "XY", "CRZ", "CRY"}


def ap1(psi, U, q):
    psi = np.tensordot(U, psi, axes=([1], [q]))
    return np.moveaxis(psi, 0, q)


def ap2(psi, U, q0, q1):
    psi = np.tensordot(U.reshape(2, 2, 2, 2), psi, axes=([2, 3], [q0, q1]))
    return np.moveaxis(psi, [0, 1], [q0, q1])


# ------------------------------------------------------------------ circuit
def n_params(n, R=2):
    return 8 * R * (len(range(0, n - 1, 2)) + len(range(1, n - 1, 2)))


def state(alpha, theta, n, gate, R=2):
    """Mirrors make_circuit_ops() in s2q_tangent_diagnostics.py (ablation none)."""
    psi = np.zeros((2,) * n, dtype=complex); psi[(0,) * n] = 1.0
    ptr = 0
    for r in range(R):
        if r == 0:
            for q in range(n):
                psi = ap1(psi, H, q)
        for q in range(n):
            psi = ap1(psi, RZ(2.0 * alpha[q]), q)
        for q in range(n - 1):
            psi = ap2(psi, CNOT, q, q + 1)
            psi = ap1(psi, RZ(2.0 * alpha[q] * alpha[q + 1]), q + 1)
            psi = ap2(psi, CNOT, q, q + 1)
        for start in (0, 1):
            for q in range(start, n - 1, 2):
                w = theta[ptr:ptr + 8]; ptr += 8
                psi = ap1(psi, RY(w[0]), q); psi = ap1(psi, RY(w[1]), q + 1)
                psi = ap2(psi, two_q(gate, w[2]), q, q + 1)
                psi = ap1(psi, RX(w[3]), q); psi = ap1(psi, RX(w[4]), q + 1)
                psi = ap2(psi, two_q(gate, w[5]), q, q + 1)
                psi = ap1(psi, RY(w[6]), q); psi = ap1(psi, RY(w[7]), q + 1)
    assert ptr == len(theta), (ptr, len(theta))
    return psi


def local_expvals(psi, n, bases):
    """<P_q> for P in bases (order z, x, y for 'xyz', as _circuit returns)."""
    order = {"x": "x", "y": "y", "z": "z", "xz": "xz"}.get(bases, "zxy")
    flat = psi.reshape(-1)
    out = []
    for b in order:
        for q in range(n):
            out.append(float(np.real(np.vdot(flat, ap1(psi, PAULI[b], q).reshape(-1)))))
    return np.array(out)


def global_z(psi, n):
    p = np.abs(psi) ** 2
    for q in range(n):
        sh = [1] * n; sh[q] = 2
        p = p * np.array([1.0, -1.0]).reshape(sh)
    return float(p.sum())


def apply_pauli(psi, n, b, q):
    return ap1(psi, PAULI[b], q)


# ------------------------------------------------------------------ selftest
def selftest(bundle):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import s2q_tangent_diagnostics as ref   # PennyLane reference, checked against training
    if bundle is not None:
        if not ref.selftest(Path(bundle).expanduser().resolve()):
            print("reference selftest against the training script FAILED"); return False
    rng = np.random.default_rng(7)
    worst_state = worst_obs = 0.0
    for gate in ("CX", "CZ", "CRY", "CRZ", "XX", "YY", "ZZ", "XY"):
        for n in (4, 5, 8, 10):
            fn = ref.build_state_fn(n, 2, gate, "none")
            for _ in range(3):
                a = rng.uniform(0, math.pi, n); t = rng.uniform(0, 2 * math.pi, n_params(n))
                p_ref = np.asarray(fn(a, t)); p_me = state(a, t, n, gate).reshape(-1)
                worst_state = max(worst_state, float(np.max(np.abs(p_ref - p_me))))
                obs = ref.observables_for("xyz", n)
                e_ref = np.array([np.real(p_ref.conj() @ (ref.qml.matrix(o, wire_order=range(n)) @ p_ref)) for _, o in obs])
                e_me = local_expvals(p_me.reshape((2,) * n), n, "xyz")
                worst_obs = max(worst_obs, float(np.max(np.abs(e_ref - e_me))))
        print(f"  {gate:4s}: max|state diff| {worst_state:.1e}  max|<P> diff| {worst_obs:.1e}", flush=True)
    ok = worst_state < 1e-9 and worst_obs < 1e-9
    print("GEOM SELFTEST", "PASSED" if ok else "FAILED")
    return ok


# ------------------------------------------------------------------ torch backend (BP scan on GPU)
class TorchSim:
    """Same circuit as state(), on a torch device (complex128). Used only by --bp-scan;
    checked against the NumPy path at start-up and refused above 1e-10."""

    def __init__(self, device):
        import torch
        self.t = torch
        self.dev = torch.device(device)
        self.cache = {}

    def m(self, M):
        return self.t.as_tensor(M, dtype=self.t.complex128, device=self.dev)

    def ap1(self, psi, U, q):
        return self.t.movedim(self.t.tensordot(self.m(U), psi, dims=([1], [q])), 0, q)

    def ap2(self, psi, U, q0, q1):
        out = self.t.tensordot(self.m(U).reshape(2, 2, 2, 2), psi, dims=([2, 3], [q0, q1]))
        return self.t.movedim(out, (0, 1), (q0, q1))

    def state(self, alpha, theta, n, gate, R=2):
        t = self.t
        psi = t.zeros((2,) * n, dtype=t.complex128, device=self.dev); psi[(0,) * n] = 1.0
        ptr = 0
        for r in range(R):
            if r == 0:
                for q in range(n):
                    psi = self.ap1(psi, H, q)
            for q in range(n):
                psi = self.ap1(psi, RZ(2.0 * alpha[q]), q)
            for q in range(n - 1):
                psi = self.ap2(psi, CNOT, q, q + 1)
                psi = self.ap1(psi, RZ(2.0 * alpha[q] * alpha[q + 1]), q + 1)
                psi = self.ap2(psi, CNOT, q, q + 1)
            for start in (0, 1):
                for q in range(start, n - 1, 2):
                    w = theta[ptr:ptr + 8]; ptr += 8
                    psi = self.ap1(psi, RY(w[0]), q); psi = self.ap1(psi, RY(w[1]), q + 1)
                    psi = self.ap2(psi, two_q(gate, w[2]), q, q + 1)
                    psi = self.ap1(psi, RX(w[3]), q); psi = self.ap1(psi, RX(w[4]), q + 1)
                    psi = self.ap2(psi, two_q(gate, w[5]), q, q + 1)
                    psi = self.ap1(psi, RY(w[6]), q); psi = self.ap1(psi, RY(w[7]), q + 1)
        return psi

    def local_expvals(self, psi, n, bases):
        order = {"x": "x", "y": "y", "z": "z", "xz": "xz"}.get(bases, "zxy")
        flat = psi.reshape(-1)
        out = []
        for b in order:
            for q in range(n):
                out.append(float(self.t.vdot(flat, self.ap1(psi, PAULI[b], q).reshape(-1)).real))
        return np.array(out)

    def global_z(self, psi, n):
        p = (psi.abs() ** 2)
        sign = self.t.tensor([1.0, -1.0], dtype=p.dtype, device=self.dev)
        for q in range(n):
            sh = [1] * n; sh[q] = 2
            p = p * sign.reshape(sh)
        return float(p.sum())

    def check(self, gates):
        rng = np.random.default_rng(11); worst = 0.0
        for gate in gates:
            for n in (4, 7, 10):
                a = rng.uniform(0, math.pi, n); th = rng.uniform(0, 2 * math.pi, n_params(n))
                ref = state(a, th, n, gate)
                me = self.state(a, th, n, gate)
                worst = max(worst, float(np.max(np.abs(ref.reshape(-1) - me.reshape(-1).cpu().numpy()))),
                            float(np.max(np.abs(local_expvals(ref, n, "xyz") - self.local_expvals(me, n, "xyz")))),
                            abs(global_z(ref, n) - self.global_z(me, n)))
        print(f"torch backend vs NumPy: max diff {worst:.1e}", flush=True)
        if worst > 1e-10:
            sys.exit("torch backend disagrees with the NumPy simulator; refusing to run")


class NumpySim:
    state = staticmethod(state)
    local_expvals = staticmethod(local_expvals)
    global_z = staticmethod(global_z)


# ------------------------------------------------------------------ BP scan
def bp_scan(gates, nmin, nmax, samples, n_idx, bases, out, seed, sim=None):
    sim = sim or NumpySim()
    rng = np.random.default_rng(seed)
    new = not Path(out).exists()
    have = set()
    if not new:
        import collections
        cnt = collections.Counter()
        with open(out) as fh:
            for r in csv.DictReader(fh):
                cnt[(r["gate"], int(r["n"]))] += 1
        have = {k for k, v in cnt.items() if v >= samples * min(n_idx, n_params(k[1]))}
    f = open(out, "a", newline=""); w = csv.writer(f)
    if new:
        w.writerow(["gate", "n", "L", "k", "sample", "grad_local_mean2", "grad_global"])
    eps = 1e-5
    for gate in gates:
        for n in range(nmin, nmax + 1):
            L = n_params(n)
            if (gate, n) in have:
                print(f"  {gate} n={n}: already in {out}", flush=True); continue
            idx = np.unique(np.linspace(0, L - 1, min(n_idx, L)).round().astype(int))
            t0 = time.time()
            for s in range(samples):
                a = rng.uniform(0, math.pi, n); th = rng.uniform(0, 2 * math.pi, L)
                for k in idx:
                    tp = th.copy(); tp[k] += eps; tm = th.copy(); tm[k] -= eps
                    sp = sim.state(a, tp, n, gate)
                    lp, gp = sim.local_expvals(sp, n, bases), sim.global_z(sp, n); del sp
                    sm = sim.state(a, tm, n, gate)
                    lm, gm = sim.local_expvals(sm, n, bases), sim.global_z(sm, n); del sm
                    gl = (lp - lm) / (2 * eps)
                    gg = (gp - gm) / (2 * eps)
                    w.writerow([gate, n, L, int(k), s, float(np.mean(gl ** 2)), gg])
            f.flush()
            print(f"  {gate} n={n:2d} L={L:3d}: {samples} samples x {len(idx)} params in {time.time() - t0:.0f}s", flush=True)
    f.close()


# ------------------------------------------------------------------ trained checkpoints
def jac(alpha, theta, n, gate, eps=1e-6):
    psi = state(alpha, theta, n, gate).reshape(-1)
    J = np.empty((psi.size, theta.size), dtype=complex)
    for l in range(theta.size):
        tp = theta.copy(); tp[l] += eps; tm = theta.copy(); tm[l] -= eps
        J[:, l] = (state(alpha, tp, n, gate) - state(alpha, tm, n, gate)).reshape(-1) / (2 * eps)
    return psi, J


def diag_at(alpha, theta, n, gate, bases, rcond):
    psi, J = jac(alpha, theta, n, gate)
    D = J - np.outer(psi, psi.conj() @ J)
    Dr = np.vstack([D.real, D.imag])
    U, s, _ = np.linalg.svd(Dr, full_matrices=False)
    r = int(np.sum(s > rcond * s.max())); Ur = U[:, :r]
    order = {"x": "x", "y": "y", "z": "z", "xz": "xz"}.get(bases, "zxy")
    A, chi2, g2 = [], [], []
    ps = psi.reshape((2,) * n)
    for b in order:
        for q in range(n):
            Op = apply_pauli(ps, n, b, q).reshape(-1)
            e = float(np.real(np.vdot(psi, Op)))
            chi = Op - e * psi
            c2 = float(np.real(np.vdot(chi, chi)))
            cr = np.concatenate([chi.real, chi.imag])
            A.append(float(np.sum((Ur.T @ cr) ** 2)) / c2 if c2 > 1e-14 else 0.0)
            chi2.append(c2)
            g = 2.0 * np.real(D.conj().T @ chi); g2.append(float(g @ g))
    return dict(A=float(np.mean(A)), chi2=float(np.mean(chi2)), grad2=float(np.mean(g2)), deff=r, smax=float(s.max()))


def run_dirs(outputs, nq, seed):
    if nq == 10:
        tr = outputs / f"so2sat_default_main__so2sat_adaptive__seed{seed}"
        tw = outputs / f"so2sat_default_isolation__so2sat_adaptive__frozen_quantum__seed{seed}"
    else:
        tr = outputs / f"so2sat_default_scaling__so2sat_adaptive__nq{nq}__none__seed{seed}"
        tw = outputs / f"so2sat_default_scaling__so2sat_adaptive__nq{nq}__frozen_quantum__seed{seed}"
    return tr, tw


def trained(bundle, nqs, seeds, n_alpha, rcond, out):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import s2q_tangent_diagnostics as ref
    bundle = Path(bundle).expanduser().resolve()
    outputs = bundle / "src" / "sq_seqnn_outputs"
    fields = ["n_qubits", "seed", "gate", "basis", "L", "D", "n_alpha", "disp_rel",
              "A_init", "A_final", "A_2design_final", "deff_init", "deff_final", "chi2_init", "chi2_final",
              "grad2_init", "grad2_final", "task_grad2_init", "task_grad2_final",
              "head_w_q", "head_w_alpha", "head_w_skip", "seconds"]
    done = set()
    if Path(out).exists():
        with open(out) as f:
            done = {(int(r["n_qubits"]), int(r["seed"])) for r in csv.DictReader(f)}
    new = not Path(out).exists()
    f = open(out, "a", newline=""); w = csv.DictWriter(f, fieldnames=fields)
    if new:
        w.writeheader()
    for nq in nqs:
        for seed in seeds:
            if (nq, seed) in done:
                continue
            t0 = time.time()
            tr, tw = run_dirs(outputs, nq, seed)
            if not tr.is_dir() or not tw.is_dir():
                print(f"[skip] nq={nq} seed={seed}: missing {tr.name if not tr.is_dir() else tw.name}", flush=True)
                continue
            meta, th_f, sd = ref.load_run(tr, "so2sat")
            _, th_i, _ = ref.load_run(tw, "so2sat")
            n, gate, basis = int(meta["n_qubits"]), meta["gate_type"], meta["measure_bases"]
            assert n == nq and th_f.shape == th_i.shape == (n_params(n),), (n, th_f.shape, th_i.shape)
            alphas, m, model, xb, yb = ref.alphas_from_data(bundle, tr, meta, sd, n_alpha, seed)
            tf = ref.task_gradient_and_head_norms(m, model, xb, yb, th_f, meta["label_smooth"])
            ti = ref.task_gradient_and_head_norms(m, model, xb, yb, th_i, meta["label_smooth"])
            res = {}
            for tag, th in (("init", th_i), ("final", th_f)):
                rows = [diag_at(a, th, n, gate, basis, rcond) for a in alphas]
                res[tag] = {k: float(np.mean([r[k] for r in rows])) for k in ("A", "chi2", "grad2", "deff")}
            D = 2 ** n
            row = dict(n_qubits=n, seed=seed, gate=gate, basis=basis, L=n_params(n), D=D, n_alpha=len(alphas),
                       disp_rel=float(np.linalg.norm(th_f - th_i) / np.linalg.norm(th_i)),
                       A_init=res["init"]["A"], A_final=res["final"]["A"],
                       A_2design_final=res["final"]["deff"] / (2 * (D - 1)),
                       deff_init=res["init"]["deff"], deff_final=res["final"]["deff"],
                       chi2_init=res["init"]["chi2"], chi2_final=res["final"]["chi2"],
                       grad2_init=res["init"]["grad2"], grad2_final=res["final"]["grad2"],
                       task_grad2_init=ti["task_grad2"], task_grad2_final=tf["task_grad2"],
                       head_w_q=tf["head_w_q"], head_w_alpha=tf["head_w_alpha"], head_w_skip=tf["head_w_skip"],
                       seconds=round(time.time() - t0, 1))
            w.writerow(row); f.flush()
            print(f"nq={n} seed={seed}: A {row['A_init']:.3f}->{row['A_final']:.3f} (2-design {row['A_2design_final']:.2e}) "
                  f"d_eff {row['deff_final']:.0f}/{row['L']}  grad2 {row['grad2_final']:.3f}  "
                  f"disp {row['disp_rel']:.3f}  ({row['seconds']}s)", flush=True)
    f.close()


def parse_list(spec):
    out = []
    for part in str(spec).split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bp-scan", action="store_true")
    ap.add_argument("--trained", action="store_true")
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--gates", default="CRY,CX,XX")
    ap.add_argument("--nmin", type=int, default=4); ap.add_argument("--nmax", type=int, default=16)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--n-idx", type=int, default=16, help="parameter indices per sample (evenly spaced)")
    ap.add_argument("--bases", default="xyz")
    ap.add_argument("--nq", default="10,12-15"); ap.add_argument("--seeds", default="42-51")
    ap.add_argument("--n-alpha", type=int, default=32); ap.add_argument("--rcond", type=float, default=1e-8)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    ap.add_argument("--device", default="cuda", help="torch backend only")
    ap.add_argument("--out", default="geom.csv")
    a = ap.parse_args()
    if a.selftest and not selftest(a.bundle):
        sys.exit(1)
    if a.bp_scan:
        gates = [g.strip() for g in a.gates.split(",")]
        sim = None
        if a.backend == "torch":
            sim = TorchSim(a.device); sim.check(gates)
        bp_scan(gates, a.nmin, a.nmax, a.samples, a.n_idx, a.bases, a.out, a.seed, sim)
    if a.trained:
        trained(a.bundle, parse_list(a.nq), parse_list(a.seeds), a.n_alpha, a.rcond, a.out)


if __name__ == "__main__":
    main()
