#!/bin/bash
#SBATCH -p NvidiaAll
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH -J oddshap_full
# Execute the complete self-contained reproduction notebook at paper scale on a GPU node.
# The CPU value functions parallelise across the allocated cores (joblib); the
# deep-learning value functions (ViT16, DistilBERT) run on the GPU. Every figure is
# produced by the notebook's own cells. Submit with:
#   sbatch -o "$HOME/oddshap_repro/full_nb.out" notebooks/run_full_notebook.sbatch
export TMPDIR=/tmp
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/oddshap_repro" || exit 1

export ODDSHAP_FULL_SCALE=1
export ODDSHAP_GPU_VF=1
export ODDSHAP_DEVICE=0
export ODDSHAP_JOBS="${SLURM_CPUS_PER_TASK:-16}"

# regenerate the .ipynb from the .py, then execute it in place (outputs are committed)
uv run --no-sync jupytext --to notebook --output notebooks/oddshap_reproduction_full.ipynb \
    notebooks/oddshap_reproduction_full.py
uv run --no-sync jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=40000 notebooks/oddshap_reproduction_full.ipynb
echo "DONE_FULL_NOTEBOOK"
