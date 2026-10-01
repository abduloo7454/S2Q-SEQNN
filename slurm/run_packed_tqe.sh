#!/bin/bash -l
# ---------------------------------------------------------------------------
#  Packed runner for the TQE no-fusion campaign (CPU, same settings as the
#  paper runs). Logs go to logs/<manifest name>/ so they never overwrite the
#  logs of the original campaign.
#
#      python tools/make_manifest_tqe.py --only nofusion --out manifest_nofusion.csv
#      sbatch --array=0-9 slurm/run_packed_tqe.sh manifest_nofusion.csv
#
#  Packed campaign runner.
#
#  Panther caps this account at 10 concurrent JOBS (QOS gpulimit, MaxJobsPU=10)
#  but not at cores. So instead of one job per run, we submit 10 jobs, each
#  holding 16 cores and running 16 runs at once, single-threaded. That is 160
#  concurrent runs instead of 10.
#
#  Submit:
#      sbatch --array=0-9 slurm/run_packed.sh manifest.csv
#
#  Shard rule: job A takes every manifest row whose task_id % 10 == A. Rows are
#  interleaved across campaigns, so each shard gets a similar mix of cheap and
#  expensive runs.
#
#  Resumable and idempotent: a run whose result file exists exits immediately,
#  so re-submitting the same array only executes what is missing.
# ---------------------------------------------------------------------------
#SBATCH -J s2q-nofus
#SBATCH -p cpu-all
#SBATCH -c 16
#SBATCH --mem 96000MB
#SBATCH -t 24:00:00
#SBATCH -o logs/pack-%A_%a.out
#SBATCH -e logs/pack-%A_%a.err

set -uo pipefail

MANIFEST="${1:?usage: sbatch --array=0-9 slurm/run_packed.sh manifest.csv}"
BUNDLE="${S2Q_BUNDLE:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repository root
cd "$BUNDLE"
mkdir -p logs results
LOGDIR="$BUNDLE/logs/$(basename "$MANIFEST" .csv)"
mkdir -p "$LOGDIR"

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
NSHARD="${SLURM_ARRAY_TASK_COUNT:-10}"
PAR="${S2Q_PAR:-${SLURM_CPUS_PER_TASK:-16}}"

export S2Q_CONDA_ENV="${S2Q_CONDA_ENV:-s2q}"
export S2Q_DATA_ROOT="${S2Q_DATA_ROOT:-$BUNDLE/data}"
export S2Q_THREADS=1
# Some compute nodes have a full local /tmp, which breaks here-documents and
# any library that writes a scratch file. Keep temp inside the bundle.
export TMPDIR="${S2Q_TMPDIR:-$BUNDLE/tmp}"
export S2Q_TMPDIR="$TMPDIR"
mkdir -p "$TMPDIR"

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "$S2Q_CONDA_ENV" || { echo "cannot activate $S2Q_CONDA_ENV"; exit 10; }
export S2Q_ENV_READY=1

echo "== shard $SHARD/$NSHARD  host $(hostname)  parallel $PAR  $(date)"
python3 -c "import pennylane, torch; print('   pennylane', pennylane.__version__, '| torch', torch.__version__)"

IDS="$(python3 -c '
import csv, sys
path, shard, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ids = [int(r["task_id"]) for r in csv.DictReader(open(path))]
print("\n".join(str(i) for i in ids if i % n == shard))
' "$MANIFEST" "$SHARD" "$NSHARD")"
COUNT="$(printf '%s\n' "$IDS" | grep -c . || true)"
echo "== $COUNT tasks in this shard"

printf '%s\n' "$IDS" | grep . | \
  xargs -I{} -P "$PAR" bash -c \
    'bash "$0/slurm/run_one.sh" "$1" "{}" > "$2/task_{}.log" 2>&1 || echo "FAILED task {}"' \
    "$BUNDLE" "$MANIFEST" "$LOGDIR"

echo "== shard $SHARD finished  $(date)"
