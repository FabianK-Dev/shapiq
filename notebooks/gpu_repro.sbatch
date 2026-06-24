#!/bin/bash
#SBATCH -p NvidiaAll
#SBATCH --time=4:00:00
# One Slurm array task per instance (Table 1 + Figure 4 eta; Figure 2 uses the paper's
# extracted curves). Submit with:
#   sbatch -J vit16      -o "$HOME/oddshap_repro/gpu_out/vit16_%a.out"      --export=ALL,VF=vit16      --array=0-29%15 notebooks/gpu_repro.sbatch
#   sbatch -J distilbert -o "$HOME/oddshap_repro/gpu_out/distilbert_%a.out" --export=ALL,VF=distilbert --array=0-29%15 notebooks/gpu_repro.sbatch
export TMPDIR=/tmp
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/oddshap_repro" || exit 1
i="$SLURM_ARRAY_TASK_ID"
uv run --no-sync python notebooks/gpu_repro.py --vf "$VF" --experiment table1eta --start "$i" --end "$((i + 1))"
