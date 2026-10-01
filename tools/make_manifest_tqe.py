#!/usr/bin/env python3
"""
Manifest for the two TQE-revision campaigns. Writes manifest_tqe.csv in the
same format as manifest.csv, so slurm/run_packed.sh and slurm/run_packed_gpu.sh
can run it unchanged. Task ids start at 0; it is a separate file, so nothing
collides with the original 1380-run manifest.

  nofusion : head fed the branch read-out alone, for every branch control.
             Trained no-fusion already exists (campaign "fusion", quantum_only).
             3 configs x 4 arms x 10 seeds = 120 runs, CPU, 10/8 qubits as in the paper.
  scaling  : So2Sat Adaptive at n_q = 8..15, four arms, ten seeds, all on GPU
             (so every point of the curve shares one device).
             8 x 4 x 10 = 320 runs.

    python tools/make_manifest_tqe.py            # writes manifest_tqe.csv
    python tools/make_manifest_tqe.py --only scaling --nq 8 9 10 11 --device cpu --out manifest_scale_cpu.csv
    python tools/make_manifest_tqe.py --only scaling --nq 12 13 14 15 --out manifest_scale_gpu.csv

  teacher  : positive control on So2Sat Adaptive. Labels come from a fixed teacher of the
             student's own form (--teacher far|near, see sq_seqnn_fast.py), no augmentation.
             Arms: trained circuit, trained at rho_q = 1, frozen circuit, frozen RFF, no circuit.
             Pilot: seeds 42-44, both teachers -> 2 x 5 x 3 = 30 runs (CPU).
             Full:  one teacher, ten seeds       -> 5 x 10 = 50 runs.

    python tools/make_manifest_tqe.py --only teacher --out manifest_teacher_pilot.csv
    python tools/make_manifest_tqe.py --only teacher --teachers near --seeds 42 43 44 45 46 47 48 49 50 51 \\
        --campaign teacher --out manifest_teacher.csv
"""
import argparse
import csv

SEEDS = list(range(42, 52))
BASE = {  # identical to the "main" rows of manifest.csv
    "overhead_adaptive": ("overhead", "adaptive", "--dataset overhead --overhead-mode seqnn5 --frontend-mode conv --gate-type CX --measure-bases x --n-qubits 8 --cls-hidden 10 --conv-width 4"),
    "sat6": ("sat6", "both", "--dataset sat6 --frontend-mode shapelet --gate-type XX --measure-bases x --n-qubits 10 --cls-hidden 10 --shapelet-count 6 --shapelet-length 4 --shapelet-stats full6"),
    "so2sat_adaptive": ("so2sat", "adaptive", "--dataset so2sat --frontend-mode global --gate-type CRY --measure-bases xyz --n-qubits 10 --cls-hidden 10"),
}
NOFUSION_ARMS = ["frozen_quantum", "frozen_rff", "mlp_replace", "frozen_mlp"]
SCALING_ARMS = ["none", "frozen_quantum", "frozen_rff", "classical_only"]
TEACHER_ARMS = [("none", ""), ("none_rhoq1", " --lr-q-factor 1"), ("frozen_quantum", ""),
                ("frozen_rff", ""), ("classical_only", "")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="manifest_tqe.csv")
    ap.add_argument("--only", choices=["nofusion", "scaling", "teacher"], default=None)
    ap.add_argument("--teachers", nargs="*", default=["far", "near"], choices=["far", "near"])
    ap.add_argument("--teacher-radius", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44], help="teacher only")
    ap.add_argument("--campaign", default="teacher_pilot", help="teacher only")
    ap.add_argument("--nq", type=int, nargs="*", default=list(range(8, 16)))
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                    help="scaling only: cuda adds --device cuda; cpu keeps the original CPU path")
    a = ap.parse_args()
    rows = []

    def add(campaign, ds, variant, config, seed, flags):
        tag = f"{campaign}__{config}__seed{seed}"
        rows.append(dict(task_id=len(rows), campaign=campaign, dataset=ds, variant=variant, config=config,
                         seed=seed, script="sq_seqnn_fast.py", tag=tag, flags=flags))

    if a.only in (None, "nofusion"):
        for cfg, (ds, var, flags) in BASE.items():
            for arm in NOFUSION_ARMS:
                for s in SEEDS:
                    add("nofusion", ds, var, f"{cfg}__nofusion_{arm}", s, f"{flags} --ablation {arm} --no-fusion")
    if a.only in (None, "scaling"):
        ds, var, flags = BASE["so2sat_adaptive"]
        for nq in a.nq:
            f = flags.replace("--n-qubits 10", f"--n-qubits {nq}")
            for arm in SCALING_ARMS:
                for s in SEEDS:
                    dev = " --device cuda" if a.device == "cuda" else ""
                    add("scaling", ds, var, f"so2sat_adaptive__nq{nq}__{arm}", s, f"{f} --ablation {arm}{dev}")

    if a.only == "teacher":                     # never part of the default manifest
        ds, var, flags = BASE["so2sat_adaptive"]
        for te in a.teachers:
            tf = f" --teacher {te} --no-aug" + (f" --teacher-radius {a.teacher_radius}" if te == "near" else "")
            for arm, extra in TEACHER_ARMS:
                abl = "none" if arm.startswith("none") else arm
                for s in a.seeds:
                    add(a.campaign, ds, var, f"so2sat_adaptive__teacher_{te}__{arm}", s,
                        f"{flags} --ablation {abl}{extra}{tf}")

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    by = {}
    for r in rows:
        by[r["campaign"]] = by.get(r["campaign"], []) + [r["task_id"]]
    print(f"wrote {a.out}: {len(rows)} runs")
    for c, ids in by.items():
        print(f"  {c:9s} {len(ids):4d} runs   task ids {min(ids)}-{max(ids)}")


if __name__ == "__main__":
    main()
