#!/bin/bash -l
# RUNS ON: Panther, as a slurm job. Submit from the login node:
#     cd <repository root>
#     S2Q_NQ="10,12,13,14,15" sbatch --array=0-3 slurm/run_geom.sh
# Tangent diagnostics on the So2Sat scaling checkpoints, one process per (n_q, seed),
# each writing its own CSV under results/geom/. Resumable: finished pairs are skipped.
#SBATCH -J s2q-geom
#SBATCH -p cpu-all
#SBATCH -c 16
#SBATCH --mem 64000MB
#SBATCH -t 24:00:00
#SBATCH -o logs/geom-%A_%a.out
#SBATCH -e logs/geom-%A_%a.err
set -uo pipefail
BUNDLE="${S2Q_BUNDLE:-${SLURM_SUBMIT_DIR:-$PWD}}"; cd "$BUNDLE"; mkdir -p logs results/geom
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NSHARD="${S2Q_NSHARD:-${SLURM_ARRAY_TASK_COUNT:-4}}"
PAR="${S2Q_PAR:-${SLURM_CPUS_PER_TASK:-16}}"
NQ="${S2Q_NQ:-10,12,13,14,15}"
export S2Q_DATA_ROOT="${S2Q_DATA_ROOT:-$BUNDLE/data}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TMPDIR="$BUNDLE/tmp"; mkdir -p "$TMPDIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate s2q || { echo "cannot activate s2q"; exit 10; }
echo "== geom shard $SHARD/$NSHARD host $(hostname) par $PAR nq $NQ $(date)"
PAIRS="$(python3 -c '
import sys
shard, n, spec = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
i = 0
for q in [int(x) for x in spec.split(",")]:
    for s in range(42, 52):
        if i % n == shard: print(q, s)
        i += 1
' "$SHARD" "$NSHARD" "$NQ")"
printf '%s\n' "$PAIRS" | grep . | xargs -P "$PAR" -I{} bash -c '
  set -- {}; Q=$1; S=$2
  OUT="'"$BUNDLE"'/results/geom/geom_nq${Q}_seed${S}.csv"
  if [ -s "$OUT" ] && [ "$(wc -l < "$OUT")" -ge 2 ]; then echo "   done: nq $Q seed $S"; exit 0; fi
  rm -f "$OUT"
  python3 -u tools/s2q_geom.py --trained --bundle "'"$BUNDLE"'" --nq "$Q" --seeds "$S" --out "$OUT" \
     > "'"$BUNDLE"'/logs/geom_nq${Q}_seed${S}.log" 2>&1 || echo "FAILED nq $Q seed $S"
'
echo "== geom shard $SHARD finished $(date)"
