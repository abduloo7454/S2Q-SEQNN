"""Qiskit translation of the S2Q-SEQNN quantum branch (mirrors sq_seqnn_fast.py).

Gate conventions are identical in PennyLane and Qiskit for H, RX, RY, RZ, CX,
CZ, SWAP, RZZ (IsingZZ), RXX, RYY, CRZ, CRY, iSWAP. PennyLane's IsingXY(phi)
is exp(i phi/4 (XX+YY)), which is Qiskit's XXPlusYYGate(-phi) up to the
same convention (both are excitation-preserving rotations); it is checked
numerically by hw_submit.py --check before anything is submitted.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate


def _entangler(qc, gate, param, q0, q1):
    if gate == "ZZ":
        qc.rzz(param, q0, q1)
    elif gate == "XY":
        qc.append(XXPlusYYGate(-param), [q0, q1])
    elif gate == "XX":
        qc.rxx(param, q0, q1)
    elif gate == "YY":
        qc.ryy(param, q0, q1)
    elif gate == "CRZ":
        qc.crz(param, q0, q1)
    elif gate == "CRY":
        qc.cry(param, q0, q1)
    elif gate == "CZ":
        qc.cz(q0, q1)
    elif gate == "CX":
        qc.cx(q0, q1)
    elif gate == "SWAP":
        qc.swap(q0, q1)
    elif gate == "ISWAP":
        qc.iswap(q0, q1)
    else:
        raise ValueError(gate)


def build_circuit(alpha, theta, n_q, n_reup, gate, basis, ablation="none", measure=True):
    """One image, one read-out basis. alpha: (n_q,), theta: (P,)."""
    qc = QuantumCircuit(n_q, n_q if measure else 0)
    ptr = 0
    for r in range(n_reup):
        if not (ablation == "no_reupload" and r > 0):
            if ablation == "no_iqp":
                for q in range(n_q):
                    qc.ry(2.0 * alpha[q], q)
            else:
                if r == 0:
                    qc.h(range(n_q))
                for q in range(n_q):
                    qc.rz(2.0 * alpha[q], q)
                for q in range(n_q - 1):
                    qc.cx(q, q + 1)
                    qc.rz(2.0 * alpha[q] * alpha[q + 1], q + 1)
                    qc.cx(q, q + 1)
        for start in (0, 1):
            for q in range(start, n_q - 1, 2):
                w = theta[ptr:ptr + 8]
                ptr += 8
                qc.ry(w[0], q)
                qc.ry(w[1], q + 1)
                _entangler(qc, gate, w[2], q, q + 1)
                qc.rx(w[3], q)
                qc.rx(w[4], q + 1)
                _entangler(qc, gate, w[5], q, q + 1)
                qc.ry(w[6], q)
                qc.ry(w[7], q + 1)
    # rotate the read-out basis onto Z
    if basis == "x":
        qc.h(range(n_q))
    elif basis == "y":
        qc.sdg(range(n_q))
        qc.h(range(n_q))
    if measure:
        qc.measure(range(n_q), range(n_q))
    return qc


def bases_list(measure_bases):
    return {"x": ["x"], "y": ["y"], "z": ["z"], "xz": ["x", "z"], "xyz": ["z", "x", "y"]}[measure_bases]


def expectations_from_counts(counts, n_q):
    """<Z_w> per wire from a Qiskit counts dict (little-endian bitstrings)."""
    tot = sum(counts.values())
    ones = np.zeros(n_q)
    for bitstr, c in counts.items():
        s = bitstr.replace(" ", "")
        for w in range(n_q):
            if s[-1 - w] == "1":
                ones[w] += c
    return 1.0 - 2.0 * ones / tot
