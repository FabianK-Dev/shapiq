#!/bin/bash
#SBATCH -p NvidiaAll
#SBATCH --time=4:00:00
# One Slurm array task per instance. Submit with:
#   sbatch -J vit16      -o .../gpu_out/vit16_%a.out      --export=ALL,VF=vit16      --array=0-29%15 notebooks/gpu_repro.sbatch
#   sbatch -J distilbert -o .../gpu_out/distilbert_%a.out --export=ALL,VF=distilbert --array=0-29%15 notebooks/gpu_repro.sbatch
export TMPDIR=/tmp
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/oddshap_repro" || exit 1
i="$SLURM_ARRAY_TASK_ID"
uv run --no-sync python notebooks/gpu_repro.py --vf "$VF" --start "$i" --end "$((i + 1))"
