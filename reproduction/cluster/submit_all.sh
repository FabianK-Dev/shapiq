#!/bin/bash
# Orchestrate the full OddSHAP paper reproduction on the cluster, for both OddSHAP
# variants, maximizing utilization within the QOS submit cap (MaxSubmitJobs=30).
#
# Six tabular value functions x {v522_merged, v560_improved} = 12 single-node CPU jobs;
# two GPU value functions x 2 variants = 4 array jobs (10 tasks each). Tabular and GPU
# run on disjoint partitions (All / NvidiaAll), so they proceed concurrently.
#
# Usage:  bash reproduction/cluster/submit_all.sh [v522_merged|v560_improved|both]
set -euo pipefail
cd "$HOME/oddshap_reproduction"
mkdir -p reproduction/data logs

VARIANTS="${1:-both}"
[ "$VARIANTS" = "both" ] && VARIANTS="v522_merged v560_improved"
TAB_VFS="cancer realestate corrgroups60 independentlinear60 nhanes crime"
GPU_VFS="vit16 distilbert"

for VARIANT in $VARIANTS; do
  echo "=== submitting variant $VARIANT ==="
  for VF in $TAB_VFS; do
    sbatch -J "tab_${VF}_${VARIANT}" -o "logs/tab_${VF}_${VARIANT}.out" \
      --export=ALL,VF="$VF",VARIANT="$VARIANT" reproduction/cluster/tabular.sbatch
  done
  for VF in $GPU_VFS; do
    sbatch -J "gpu_${VF}_${VARIANT}" -o "logs/gpu_${VF}_${VARIANT}_%a.out" \
      --export=ALL,VF="$VF",VARIANT="$VARIANT" --array=0-9%15 reproduction/cluster/gpu.sbatch
  done
done
echo "=== queue ==="
squeue -u "$USER" -o "%.12i %.22j %.8T %.10M %R" | head -40
