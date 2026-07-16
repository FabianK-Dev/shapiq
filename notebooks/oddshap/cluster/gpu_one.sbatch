#!/bin/bash
#SBATCH -p NvidiaAll
#SBATCH --time=1:30:00
#SBATCH -J odd_gpu1
# ONE deep-learning instance (short job) so the scheduler can backfill it onto an idle
# NvidiaAll node's free window instead of queueing behind higher-priority multi-hour jobs.
# Submitted by submit_gpu_waves.sh with VF, VARIANT, INST, and optional EXPERIMENTS.
export TMPDIR=/tmp
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/oddshap_reproduction" || exit 1
uv run --no-sync python notebooks/oddshap/cluster/train_gpu.py \
    --vf "$VF" --variant "$VARIANT" --start "$INST" --end "$((INST + 1))" \
    ${EXPERIMENTS:+--experiments $EXPERIMENTS} \
    | tee -a "notebooks/oddshap/data/gpu_${VF}_${VARIANT}.log"
