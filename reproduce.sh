#!/usr/bin/env bash
# Regenerate every table, figure, and inline number of the paper and its supplement
# from the released run logs in results/. No training, no GPU, no quantum hardware:
# about two minutes on a laptop with the environment of environment.yml.
#
#     bash reproduce.sh            # writes everything to out/
#
# Each line names the paper item it produces. Every script refuses to run if an
# input is incomplete (wrong seed count, missing arm, failed reproduction check).
set -euo pipefail
cd "$(dirname "$0")"
R=results; O=out; A=analysis
mkdir -p $O
export MPLCONFIGDIR="$PWD/.mplconfig"

# Tables III, IV, V, S1, S2, S4, S6, S7, S8 and the inline numbers they feed
python $A/build_tables.py --runs $R/all_runs.csv --tangent-dir $R/tangent --out $O/tables
# Fig. 3 (parameters vs accuracy) and Fig. 4 (paired differences of every control)
python $A/make_param_figure.py --runs $R/all_runs.csv --out $O/fig_param_accuracy
python $A/make_fig1_forest.py --runs $R/all_runs.csv --out $O/fig_forest
# Fig. 5 and Tables VI, S9 (tangent-space diagnostics of the trained circuits)
python $A/analyze_tangent.py "$R/tangent/tangent_*.csv" --out $O/tangent_summary.csv
python $A/make_tangent_figure.py $R/tangent --out $O/fig_tangent
# Tables S10, S11 (equivalence tests, controls without fusion)
python $A/make_equivalence.py $R/all_runs.csv $R/tqe/nofusion/all_runs_tqe.csv --out-dir $O/equivalence
# Table S15 (minimum detectable effect of the seed-paired design)
python $A/make_power.py $R/all_runs.csv --out-dir $O/power
# Table S12 (register sizes 8 to 15)
python $A/make_scaling.py $R/all_runs.csv $R/tqe/scale_cpu/all_runs_tqe.csv \
    $R/tqe/scale_gpu_a/all_runs_tqe.csv $R/tqe/scale_gpu_b/all_runs_tqe.csv --out-dir $O/scaling
# Fig. 6 and Tables S13, S14 (barren-plateau scan to 28 qubits, diagnostics at 8 to 15 qubits)
python $A/make_geom_figure.py --geom-dir $R/geom --scaling-stats $O/scaling/scaling_stats.csv --out-dir $O/geom
# Table S5 (depolarizing noise, ten seeds)
python $A/make_noise_table.py $R/noise/noise_raw_10seed.csv $R/all_runs.csv --out-dir $O/noise
# Table VII and Fig. 7 (ibm_kingston, three seeds); Fig. S1 (qubit placement).
# The seed-42 run also gives the gate counts, depth, QPU time, and slope quoted in Sections IV-C and V-E.
python $A/make_hw_seeds.py --root $R/hardware --out-dir $O/hardware
python $A/make_hw_seeds_figure.py --root $R/hardware --out $O/hardware/fig_hardware_seeds
( cd $R/hardware && python ../../$A/make_hardware_figure.py run_trained run_frozen_pinned run_frozen \
    --out ../../$O/hardware/fig_hardware_seed42 )
( cd $R/hardware && python ../../$A/make_hardware_figure.py run_frozen_pinned run_frozen_rep run_frozen \
    --labels "frozen, qubits A" "frozen, qubits A repeated" "frozen, qubits B" \
    --keys fza fzarepeat fzb --out ../../$O/hardware/fig_hardware_layout )
# Positive control (labels from a circuit teacher), Section V-B and Table S16,
# with the same rho_q = 1 contrast on the real labels
python $A/make_teacher.py $R/teacher/all_runs_tqe.csv --real-runs $R/all_runs.csv --out-dir $O/teacher

echo "done: every output is in $O/"
