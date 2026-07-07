#!/bin/bash
#SBATCH -p NvidiaAll
#SBATCH --time=6:00:00
#SBATCH -J odd_gpu
# One array task per 3-instance batch of a deep-learning value function + variant.
# Covers Table 1/3, Figure 2, Figure 4/11 for ViT16 / DistilBERT (exact Shapley GT).
# Submitted by submit_all.sh with VF and VARIANT exported; array bounds sized to the
# QOS submit cap (10 tasks x 3 instances = 30).
export TMPDIR=/tmp
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/oddshap_reproduction" || exit 1
STRIDE=3
start=$(( SLURM_ARRAY_TASK_ID * STRIDE ))
end=$(( start + STRIDE ))
uv run --no-sync python -m reproduction.cluster.train_gpu \
    --vf "$VF" --variant "$VARIANT" --start "$start" --end "$end" \
    | tee -a "reproduction/data/gpu_${VF}_${VARIANT}.log"
