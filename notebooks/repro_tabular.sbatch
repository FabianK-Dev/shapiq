#!/bin/bash
#SBATCH -p All
#SBATCH --cpus-per-task=32
#SBATCH --time=4:00:00
#SBATCH -J repro_tab
# Full-scale tabular reproduction (Table 1 + Figure 2 + Figure 4) against the
# canonical OddSHAP, parallelised across all allocated cores. Submit with:
#   sbatch -o "$HOME/oddshap_repro/repro_tab.out" notebooks/repro_tabular.sbatch
export TMPDIR=/tmp
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/oddshap_repro" || exit 1
uv run --no-sync python notebooks/repro_tabular.py --experiment all --instances 30 --jobs "${SLURM_CPUS_PER_TASK:-16}"
