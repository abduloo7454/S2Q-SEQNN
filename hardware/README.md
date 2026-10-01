# Hardware pipeline (IBM Quantum)

Nothing is trained on the device. A trained checkpoint is exported once, the quantum branch
is rebuilt in Qiskit and checked against the PennyLane model on the exact statevector, and
only the Pauli read-out is replaced by estimates from hardware shots. The frontend,
compressor, circuit parameters, and head keep their trained values.

| Step | Script | Environment | Output |
|---|---|---|---|
| A. Export | `hw_export.py` | training env (`environment.yml`), data and checkpoint | `hw_package.npz`: 100 stratified test images, their angles, the circuit parameters, the exact read-out, and the head |
| B. Check and submit | `hw_submit.py` | `requirements-hardware.txt` | `hw_job.json`: backend, job ID, physical qubits |
| C. Collect | `hw_collect.py` | `requirements-hardware.txt` | `hw_counts.csv`, `hw_features.csv`, `hw_results.json` |

`hw_circuit.py` builds the circuit and is shared by B and C.

```bash
# A (from the repository root)
python hardware/hw_export.py --src src \
    --run-dir src/sq_seqnn_outputs/overhead_seqnn5_main__overhead_adaptive__seed42 \
    --n-samples 100 --shots 4000 --out hw_package.npz

# B: save your own IBM Quantum account once, then check locally and submit
python -c "from qiskit_ibm_runtime import QiskitRuntimeService as S; S.save_account(channel='ibm_quantum_platform', token='<API KEY>', instance='<CRN>', set_as_default=True)"
python hardware/hw_submit.py --package hw_package.npz --check
python hardware/hw_submit.py --package hw_package.npz --submit --backend ibm_kingston \
    --initial-layout 1,2,3,16,23,24,25,37

# C
python hardware/hw_collect.py --job hw_job.json --status
python hardware/hw_collect.py --job hw_job.json
```

`--check` compares the Qiskit circuit with the PennyLane read-out stored in the package
(tolerance 1e-4) and submits nothing. Pin the layout with `--initial-layout` whenever two
circuits will be compared; a free compiler changes the placement between submissions, and
the per-wire contraction depends on it (Fig. S1 of the supplementary material).

## Released runs (`results/hardware/`)

All runs use `ibm_kingston`, 100 test images of Overhead-MNIST, 4000 shots, dynamical
decoupling and Pauli twirling.

| Folder | Seed | Circuit | Physical qubits |
|---|---|---|---|
| `run_trained` | 42 | trained | 1, 2, 3, 16, 23, 24, 25, 37 |
| `run_frozen_pinned` | 42 | frozen twin | same |
| `run_frozen_rep` | 42 | frozen twin, repeat | same |
| `run_frozen` | 42 | frozen twin | 0, 1, 2, 3, 16, 23, 24, 25 |
| `run_s43_trained`, `run_s43_frozen` | 43 | trained, frozen | 1, 2, 3, 16, 23, 24, 25, 37 |
| `run_s44_trained`, `run_s44_frozen` | 44 | trained, frozen | 1, 2, 3, 16, 23, 24, 25, 37 |

Each folder holds the exported package, the job record with its IBM job ID, the raw counts,
the per-image read-out, and the summary used by the paper.
