#!/bin/bash
# ---------------------------------------------------------------------------
#  Run ONE manifest row. No SBATCH directives: this is called either directly
#  or, many at a time, by slurm/run_packed.sh.
#
#      bash slurm/run_one.sh manifest.csv 137
#
#  Resumable: exits immediately if the result file already exists.
# ---------------------------------------------------------------------------
set -uo pipefail

MANIFEST="${1:?usage: run_one.sh manifest.csv TASK_ID}"
TASK_ID="${2:?usage: run_one.sh manifest.csv TASK_ID}"
BUNDLE="${S2Q_BUNDLE:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repository root
cd "$BUNDLE"

CONDA_ENV="${S2Q_CONDA_ENV:-s2q}"
export S2Q_DATA_ROOT="${S2Q_DATA_ROOT:-$BUNDLE/data}"

if [ -z "${S2Q_ENV_READY:-}" ]; then
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
    else
        source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi
    conda activate "$CONDA_ENV" || exit 10
fi

T="${S2Q_THREADS:-1}"
export OMP_NUM_THREADS="$T" MKL_NUM_THREADS="$T" OPENBLAS_NUM_THREADS="$T" NUMEXPR_NUM_THREADS="$T"
export MPLCONFIGDIR="$BUNDLE/.mplconfig/$TASK_ID"
# Some compute nodes have a full local /tmp. Keep every temp file in the
# bundle instead, where there is room.
export TMPDIR="${S2Q_TMPDIR:-$BUNDLE/tmp}"
mkdir -p "$MPLCONFIGDIR" "$TMPDIR" logs results

# No here-document anywhere in this script: a here-doc needs a writable
# $TMPDIR, and that is exactly what fails on a node with a full /tmp.
ROW="$(python3 -c '
import csv, sys
path, tid = sys.argv[1], int(sys.argv[2])
for r in csv.DictReader(open(path)):
    if int(r["task_id"]) == tid:
        print("\t".join([r["campaign"], r["dataset"], r["config"], r["seed"],
                         r["script"], r["tag"], r["flags"]]))
        break
' "$MANIFEST" "$TASK_ID")"
# A here-string ("<<<") also spills to a temp file in bash, so use process
# substitution instead. Nothing in this script touches $TMPDIR now.
IFS=$'\t' read -r CAMPAIGN DATASET CONFIG SEED SCRIPT TAG FLAGS < <(printf '%s\n' "$ROW")
[ -n "${TAG:-}" ] || { echo "task $TASK_ID not found in $MANIFEST"; exit 2; }

echo "== task $TASK_ID  [$CAMPAIGN] $CONFIG seed $SEED  on $(hostname)  $(date)"
echo "   script: $SCRIPT"
echo "   flags : $FLAGS"

cd "$BUNDLE/src"
case "$SCRIPT" in

  sq_seqnn_fast.py)
    MODE="default"; [ "$DATASET" = "overhead" ] && MODE="seqnn5"
    OUT="sq_seqnn_outputs/${DATASET}_${MODE}_${TAG}/results_${DATASET}.csv"
    if [ -f "$OUT" ]; then echo "   already done: $OUT"; exit 0; fi
    # shellcheck disable=SC2086
    python3 sq_seqnn_fast.py $FLAGS --seed "$SEED" --run-tag "$TAG" || exit 3
    [ -f "$OUT" ] || { echo "   ERROR: expected $OUT not written"; exit 3; }
    ;;

  run_baselines.py)
    OUTDIR="$BUNDLE/results/baselines/$TAG"
    if ls "$OUTDIR"/*_runs.csv >/dev/null 2>&1; then echo "   already done: $OUTDIR"; exit 0; fi
    # shellcheck disable=SC2086
    python3 run_baselines.py $FLAGS --epochs 80 --out-dir "$OUTDIR" || exit 3
    ;;

  heavy_baselines.py)
    OUTDIR="$BUNDLE/results/classical/$TAG"
    if [ -f "$OUTDIR/full/heavy_baseline_runs.csv" ]; then echo "   already done: $OUTDIR"; exit 0; fi
    # shellcheck disable=SC2086
    python3 heavy_baselines.py $FLAGS --out-dir "$OUTDIR" || exit 3
    ;;

  *)
    echo "unknown script: $SCRIPT"; exit 4 ;;
esac

echo "== done task $TASK_ID  $(date)"
