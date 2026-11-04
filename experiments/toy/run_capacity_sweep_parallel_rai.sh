#!/usr/bin/env bash
#
# Parallel capacity sweep.
#
# Runs the capacity-sweep configs (see README) several at a time as
# independent processes. Each config is CPU-bound and tiny, so config-level
# parallelism gives near-linear speedup up to your core count.
#
#   JOBS=8 bash run_capacity_sweep_parallel_rai.sh     # run 8 configs at once
#   bash run_capacity_sweep_parallel_rai.sh            # defaults to all cores

set -euo pipefail

SCRIPT="relu10_further_exp.py"

# ---- knobs -------------------------------------------------------------------
BASE_WIDTH=4
WIDE_WIDTH=5
NUM_HIDDEN=19
N_RUNS="${N_RUNS:-1000}"   # e.g. N_RUNS=4 EPOCHS=3 for a smoke test
EPOCHS="${EPOCHS:-250}"
JOBS="${JOBS:-$(nproc)}"   # how many configs to run concurrently
# ------------------------------------------------------------------------------

# One math thread per worker, so N processes don't each spawn N threads and
# thrash the cores. This is what makes the parallelism actually scale.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

FGIS=("NONE")
TARGETS=("absfunc4")

mkdir -p logs   # per-config console logs (results go to results/, see relu10_further_exp.py)

# Build the staircase width configs.
WIDTH_CONFIGS=()
for ((k = 0; k <= NUM_HIDDEN; k++)); do
    parts=()
    for ((i = 0; i < NUM_HIDDEN; i++)); do
        if ((i < k)); then parts+=("$WIDE_WIDTH"); else parts+=("$BASE_WIDTH"); fi
    done
    IFS=','; WIDTH_CONFIGS+=("${parts[*]}"); unset IFS
done

# Assemble the full job list as "fgi|target|widths" specs.
JOBLIST=()
for fgi in "${FGIS[@]}"; do
    for target in "${TARGETS[@]}"; do
        for widths in "${WIDTH_CONFIGS[@]}"; do
            JOBLIST+=("$fgi|$target|$widths")
        done
    done
done

# What each worker does. Console output goes to a per-config file in logs/ so
# parallel jobs don't interleave on your terminal.
run_one() {
    local spec="$1"
    IFS='|' read -r fgi target widths <<< "$spec"
    local tag="fgi_${fgi}_target_${target}_widths_${widths//,/-}_rai"
    echo ">> start $tag"
    python3 "$SCRIPT" \
        --fgi "$fgi" \
        --target_func "$target" \
        --widths "$widths" \
        --n_runs "$N_RUNS" \
        --rai \
        --num_epochs "$EPOCHS" > "logs/${tag}.log" 2>&1
    echo "<< done  $tag"
}
export -f run_one
export SCRIPT N_RUNS EPOCHS

echo "Running ${#JOBLIST[@]} configs, $JOBS at a time (1 thread each)."
echo

# xargs -P is the process pool: it keeps $JOBS workers busy until the list is done.
printf '%s\n' "${JOBLIST[@]}" | xargs -P "$JOBS" -I {} bash -c 'run_one "$@"' _ {}

echo
echo "Sweep complete. Per-config console logs are in logs/."
