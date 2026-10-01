#!/usr/bin/env python
"""Step B (local machine): build the hardware circuits from hw_package.npz, verify them
against the PennyLane read-out stored in the package, and submit one Sampler
job to an IBM Quantum backend.

    python hw_submit.py --package hw_package.npz --check            # local check only, nothing submitted
    python hw_submit.py --package hw_package.npz --submit           # least-busy backend
    python hw_submit.py --package hw_package.npz --submit --backend ibm_torino

Save the account once (local machine):
    python -c "from qiskit_ibm_runtime import QiskitRuntimeService as S; S.save_account(channel='ibm_quantum_platform', token='<API KEY>', instance='<CRN>', set_as_default=True, overwrite=True)"

The job id is written to hw_job.json; hw_collect.py reads it.
"""
import argparse
import json
import time

import numpy as np
from qiskit.quantum_info import Statevector

from hw_circuit import build_circuit, bases_list


def load_package(path):
    z = np.load(path, allow_pickle=False)
    cfg = dict(gate=str(z["gate"]), bases=str(z["bases"]), n_q=int(z["n_qubits"]), n_reup=int(z["n_reup"]),
               ablation=str(z["ablation"]), shots=int(z["shots"]))
    return z, cfg


def check_translation(z, cfg, n_check=8, tol=1e-4):
    """Exact statevector read-out of the Qiskit circuits must equal q_sim from PennyLane."""
    alpha, theta, q_sim = z["alpha"], z["q_w"], z["q_sim"]
    n_q = cfg["n_q"]
    worst = 0.0
    for i in range(min(n_check, len(alpha))):
        for j, b in enumerate(bases_list(cfg["bases"])):
            qc = build_circuit(alpha[i], theta, n_q, cfg["n_reup"], cfg["gate"], b, cfg["ablation"], measure=False)
            sv = Statevector(qc)
            probs = sv.probabilities()
            ez = np.array([1.0 - 2.0 * sum(p for k, p in enumerate(probs) if (k >> w) & 1) for w in range(n_q)])
            worst = max(worst, float(np.abs(ez - q_sim[i, j * n_q:(j + 1) * n_q]).max()))
    print(f"translation check on {min(n_check, len(alpha))} images: max |Qiskit - PennyLane| = {worst:.2e}")
    if worst > tol:
        raise SystemExit("translation mismatch; do not submit")
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default="hw_package.npz")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--backend", default="")
    ap.add_argument("--shots", type=int, default=0, help="override the package's shot count")
    ap.add_argument("--n-images", type=int, default=0, help="use only the first n images (0 = all)")
    ap.add_argument("--no-mitigation", action="store_true", help="switch off dynamical decoupling and twirling")
    ap.add_argument("--initial-layout", default="",
                    help="comma-separated physical qubits to pin the logical wires to, "
                         "e.g. 1,2,3,16,23,24,25,37. Use the SAME value for two circuits "
                         "that will be compared, or their read-out fidelities differ by "
                         "qubit quality as well as by circuit.")
    ap.add_argument("--out", default="hw_job.json")
    a = ap.parse_args()

    z, cfg = load_package(a.package)
    shots = a.shots or cfg["shots"]
    n = a.n_images or len(z["alpha"])
    bl = bases_list(cfg["bases"])
    print(f"package: {cfg}  images={n}  circuits={n * len(bl)}  shots={shots}")

    worst = check_translation(z, cfg)
    if not a.submit:
        return

    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    service = QiskitRuntimeService()
    if a.backend:
        backend = service.backend(a.backend)
    else:
        backend = service.least_busy(operational=True, simulator=False, min_num_qubits=cfg["n_q"])
    print(f"backend: {backend.name}  qubits={backend.num_qubits}  pending jobs={backend.status().pending_jobs}")

    circuits, index = [], []
    for i in range(n):
        for b in bl:
            circuits.append(build_circuit(z["alpha"][i], z["q_w"], cfg["n_q"], cfg["n_reup"], cfg["gate"], b, cfg["ablation"]))
            index.append((int(i), b))
    layout = [int(q) for q in a.initial_layout.split(",") if q.strip()] or None
    if layout is not None:
        if len(layout) != cfg["n_q"]:
            raise SystemExit(f"--initial-layout needs {cfg['n_q']} qubits, got {len(layout)}")
        print(f"pinning initial layout: {layout}")
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend,
                                      seed_transpiler=0, initial_layout=layout)
    isa = pm.run(circuits)
    twoq = [sum(1 for inst in c.data if inst.operation.num_qubits == 2) for c in isa]
    depth = [c.depth() for c in isa]
    phys = isa[0].layout.final_index_layout() if isa[0].layout is not None else None
    print(f"transpiled: two-qubit gates per circuit {min(twoq)}-{max(twoq)}, depth {min(depth)}-{max(depth)}, "
          f"physical qubits {phys}")

    sampler = Sampler(mode=backend)
    sampler.options.default_shots = shots
    if not a.no_mitigation:
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XpXm"
        sampler.options.twirling.enable_gates = True
        sampler.options.twirling.enable_measure = True
        sampler.options.twirling.num_randomizations = "auto"
    job = sampler.run(isa)
    info = {"job_id": job.job_id(), "backend": backend.name, "shots": shots, "n_images": n, "bases": bl,
            "index": index, "package": a.package, "translation_max_abs_diff": worst,
            "twoq_min": min(twoq), "twoq_max": max(twoq), "depth_min": min(depth), "depth_max": max(depth),
            "physical_qubits": phys,
            "mitigation": (not a.no_mitigation), "initial_layout_requested": layout, "submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(info, open(a.out, "w"), indent=2)
    print(f"submitted job {job.job_id()} on {backend.name}; wrote {a.out}")
    print("check status with:  python hw_collect.py --job hw_job.json --status")


if __name__ == "__main__":
    main()
