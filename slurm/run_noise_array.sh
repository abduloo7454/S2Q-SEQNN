#!/bin/bash -l
# ---------------------------------------------------------------------------
# Slurm job; submit from the repository root (login nodes often kill long python processes).
#
# Ten-seed depolarising-noise sweep: 120 runs = 3 datasets x 10 seeds (42-51)
# x 4 rates (0.0, 0.001, 0.005, 0.010).
#
# Packed, not a 120-task array: this account is capped at 10 concurrent JOBS
# (QOS gpulimit, MaxJobsPU=10) but not at cores, so we submit 10 jobs of 16
# cores and run 16 evaluations at a time inside each -- the same shape as
# run_packed.sh. Shard rule: job A takes every combination whose index % 10 == A.
#
#     sbatch --array=0-9 slurm/run_noise_array.sh
#     python tools/merge_noise.py                    # when all 10 finish
#
# Resumable: a combination whose raw CSV already exists is skipped, so
# re-submitting the same array only runs what is missing.
# ---------------------------------------------------------------------------
#SBATCH -J s2q-noise10
#SBATCH -p cpu-all
#SBATCH -c 16
#SBATCH --mem 96000MB
#SBATCH -t 24:00:00
#SBATCH -o logs/noise10-%A_%a.out
#SBATCH -e logs/noise10-%A_%a.err

set -uo pipefail

BUNDLE="${S2Q_BUNDLE:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repository root
cd "$BUNDLE"
mkdir -p logs results/noise

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NSHARD="${SLURM_ARRAY_TASK_COUNT:-10}"
PAR="${S2Q_PAR:-${SLURM_CPUS_PER_TASK:-16}}"

export S2Q_CONDA_ENV="${S2Q_CONDA_ENV:-s2q}"
export S2Q_DATA_ROOT="${S2Q_DATA_ROOT:-$BUNDLE/data}"
export TMPDIR="${S2Q_TMPDIR:-$BUNDLE/tmp}"
export S2Q_THREADS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$TMPDIR"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "$S2Q_CONDA_ENV" || { echo "cannot activate $S2Q_CONDA_ENV"; exit 10; }

echo "== noise shard $SHARD/$NSHARD  host $(hostname)  parallel $PAR  $(date)"

# combination index -> dataset, seed, p.  No here-docs: /tmp may be full.
COMBOS="$(python3 -c '
import sys
shard, n = int(sys.argv[1]), int(sys.argv[2])
ds = ["overhead", "sat6", "so2sat"]
seeds = list(range(42, 52))
pv = ["0.0", "0.001", "0.005", "0.010"]
i = 0
for d in ds:
    for s in seeds:
        for p in pv:
            if i % n == shard:
                print(f"{d} {s} {p}")
            i += 1
' "$SHARD" "$NSHARD")"
COUNT="$(printf '%s\n' "$COMBOS" | grep -c . || true)"
echo "== $COUNT combinations in this shard"

# so2sat is the slowest (~8.6 h per combination at the full test set);
# overhead is ~0.6 h. With 16 running at once a shard is dominated by so2sat.
printf '%s\n' "$COMBOS" | grep . | \
  xargs -P "$PAR" -I{} bash -c '
    set -- {}
    D=$1; S=$2; P=$3
    OUT="'"$BUNDLE"'/results/noise/${D}_seed${S}_p${P}"
    if [ -f "$OUT/depolarizing_noise_raw.csv" ]; then
        echo "   already done: $D seed $S p $P"; exit 0
    fi
    mkdir -p "$OUT"
    export MPLCONFIGDIR="'"$BUNDLE"'/.mplconfig/noise_${D}_${S}_${P}"
    mkdir -p "$MPLCONFIGDIR"
    cd "'"$BUNDLE"'"
    python3 -u src/s2q_depolarizing_noise_eval.py \
        --dataset "$D" --seeds "$S" --p-values "$P" \
        --batch-size 8 --tta checkpoint --out-dir "$OUT" \
        > "'"$BUNDLE"'/logs/noise_${D}_${S}_${P}.log" 2>&1 \
      || echo "FAILED $D seed $S p $P"
  '

echo "== noise shard $SHARD finished  $(date)"
