#!/bin/bash -l
# ---------------------------------------------------------------------------
#  Tangent-space diagnostics for the S2Q-SEQNN quantum branch.
#
#  Submit from the bundle root:
#
#      sbatch slurm/run_tangent.sh selftest        # circuit check + 1 seed, ~10 min
#      sbatch slurm/run_tangent.sh full            # 3 configs x 10 seeds, array of 3
#
#  Header copied from slurm/run_packed.sh. The login node kills python, so
#  nothing here is meant to run interactively.
# ---------------------------------------------------------------------------
#SBATCH -J s2q-tangent
#SBATCH -p cpu-all
#SBATCH -c 8
#SBATCH --mem 32000MB
#SBATCH -t 06:00:00
#SBATCH -a 0-2
#SBATCH -o logs/tangent-%A_%a.out
#SBATCH -e logs/tangent-%A_%a.err
set -uo pipefail

MODE="${1:-full}"
BUNDLE="${S2Q_BUNDLE:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repository root
cd "$BUNDLE"
mkdir -p logs results/tangent

# Direct interpreter path: conda's shell hooks get killed on the login node
# and are not needed on a compute node either.
PY="$HOME/miniconda3/envs/s2q/bin/python"
export S2Q_DATA_ROOT="${S2Q_DATA_ROOT:-$HOME/s2q_data}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLCONFIGDIR="$BUNDLE/tmp/mpl"
mkdir -p "$MPLCONFIGDIR"

CONFIGS=(overhead_adaptive sat6 so2sat_adaptive)
TASK="${SLURM_ARRAY_TASK_ID:-0}"
CFG="${CONFIGS[$TASK]}"

echo "host=$(hostname) mode=$MODE task=$TASK cfg=$CFG python=$PY"
"$PY" -c "import numpy, torch, pennylane; print('numpy', numpy.__version__, 'torch', torch.__version__, 'pennylane', pennylane.__version__)" || exit 20

if [ "$MODE" = "selftest" ]; then
    # Only array task 0 does the self-test; the others exit at once.
    [ "$TASK" != "0" ] && exit 0
    "$PY" s2q_tangent_diagnostics.py --bundle "$BUNDLE" --selftest --configs "" || exit 30
    echo "--- one-seed timing run (sat6, seed 42, 8 alphas) ---"
    "$PY" s2q_tangent_diagnostics.py --bundle "$BUNDLE" \
        --configs sat6 --seeds 42 --n-alpha 8 \
        --out results/tangent/test_diag.csv --svals-dir results/tangent/test_svals
    exit $?
fi

# full: one config per array task, all ten seeds, resumable
"$PY" s2q_tangent_diagnostics.py --bundle "$BUNDLE" \
    --configs "$CFG" --seeds 42-51 --n-alpha 32 \
    --out "results/tangent/tangent_${CFG}.csv" \
    --svals-dir results/tangent/svals \
    --skip-existing
