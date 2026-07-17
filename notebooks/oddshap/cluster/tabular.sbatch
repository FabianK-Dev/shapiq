#!/bin/bash
#SBATCH -p All
#SBATCH --cpus-per-task=14
#SBATCH --time=10:00:00
#SBATCH -J odd_tab
# One node per (value function, variant): all 14 cores via loky. Covers Table 1/3,
# Figure 2, Figure 4/11 (eta), Figure 5 (runtime) for one tabular value function.
# Submitted by submit_all.sh with VF and VARIANT exported.
export TMPDIR=/tmp
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/oddshap_reproduction" || exit 1
uv run --no-sync python notebooks/oddshap/cluster/train_tabular.py \
    --vf "$VF" --variant "$VARIANT" --instances 30 --jobs "${SLURM_CPUS_PER_TASK:-14}" \
    --out notebooks/oddshap/data
