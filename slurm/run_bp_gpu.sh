#!/bin/bash -l
# RUNS ON: Panther H200, submit from the login node:
#     cd <repository root>
#     sbatch slurm/run_bp_gpu.sh
# Barren-plateau scan at random parameters (circuit only, no training, no data):
# gradient^2 of the model's local read-out and of a global Z..Z cost vs n_q.
# Part 1: n = 4..22, 100 samples x 8 parameters.  Part 2: n = 23..28, 50 x 4.
# Resumable: a (gate, n) block already complete in the CSV is skipped.
#SBATCH -J s2q-bp
#SBATCH -p gpu-H200
##SBATCH -A <your_account>        # site-specific; enable and edit
##SBATCH --qos <your_qos>         # site-specific; enable and edit
#SBATCH --gres=gpu:H200_141GB:1
#SBATCH -c 4
#SBATCH --mem 64000MB
#SBATCH -t 72:00:00
#SBATCH -o logs/bp-%j.out
#SBATCH -e logs/bp-%j.err
set -uo pipefail
BUNDLE="${S2Q_BUNDLE:-${SLURM_SUBMIT_DIR:-$PWD}}"; cd "$BUNDLE"; mkdir -p logs results/geom
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate s2q || { echo "cannot activate s2q"; exit 10; }
nvidia-smi --query-gpu=name,memory.total --format=csv
GATES="${S2Q_GATES:-CRY,CX,XX}"
python3 tools/s2q_geom.py --bp-scan --backend torch --device cuda --gates "$GATES" \
    --nmin 4 --nmax "${S2Q_NMAX1:-22}" --samples 100 --n-idx 8 --out results/geom/bp_gpu_a.csv
python3 tools/s2q_geom.py --bp-scan --backend torch --device cuda --gates "$GATES" \
    --nmin $(( ${S2Q_NMAX1:-22} + 1 )) --nmax "${S2Q_NMAX2:-28}" --samples 50 --n-idx 4 --out results/geom/bp_gpu_b.csv
echo "== bp scan finished $(date)"
