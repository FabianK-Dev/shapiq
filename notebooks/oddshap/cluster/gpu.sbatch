#!/bin/bash
#SBATCH -p NvidiaAll
#SBATCH --time=6:00:00
#SBATCH -J odd_gpu
# One array task per 3-instance batch of a deep-learning value function + variant.
# Covers Table 1/3, Figure 2, Figure 5 (runtime) and Figure 4/11 for ViT16 / DistilBERT
# with exact Shapley ground truth. NvidiaAll nodes expose their GPU by partition (no
# --gres and no schedulable --mem on this cluster); train_gpu.py asserts CUDA is visible
# and fails loudly otherwise. Submitted by submit_all.sh with VF and VARIANT exported.
export TMPDIR=/tmp
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/oddshap_reproduction" || exit 1
STRIDE=3
start=$(( SLURM_ARRAY_TASK_ID * STRIDE ))
end=$(( start + STRIDE ))
uv run --no-sync python notebooks/oddshap/cluster/train_gpu.py \
    --vf "$VF" --variant "$VARIANT" --start "$start" --end "$end" \
    | tee -a "notebooks/oddshap/data/gpu_${VF}_${VARIANT}.log"
