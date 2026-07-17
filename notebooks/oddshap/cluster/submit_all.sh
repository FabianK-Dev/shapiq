#!/bin/bash
# Submit the full OddSHAP paper reproduction for ONE OddSHAP variant, staying within the
# cluster's QOS limits. Slurm counts every array element individually toward
# MaxSubmitJobs=30, so a single variant is the safe unit:
#   6 tabular single-node jobs + 2 GPU arrays x 10 tasks = 6 + 20 = 26 <= 30.
# Submitting BOTH variants at once (52 records) would exceed the cap and, with set -e,
# abort mid-submission — so this script deliberately does one variant per invocation.
#
# Usage:
#   bash notebooks/oddshap/cluster/submit_all.sh v522_merged
#   # wait for that batch to drain, then:
#   bash notebooks/oddshap/cluster/submit_all.sh v560_improved
set -uo pipefail
cd "$HOME/oddshap_reproduction"
mkdir -p notebooks/oddshap/data logs

VARIANT="${1:?usage: submit_all.sh <v522_merged|v560_improved>}"
TAB_VFS="cancer realestate corrgroups60 independentlinear60 nhanes crime"
GPU_VFS="vit16 distilbert"

# guard: refuse to submit if it would blow the submit cap
QUEUED=$(squeue -u "$USER" -h -r 2>/dev/null | wc -l)
WANT=26
if [ "$((QUEUED + WANT))" -gt 30 ]; then
  echo "ABORT: $QUEUED jobs already queued + $WANT new > MaxSubmitJobs=30. Wait for the current batch to drain." >&2
  exit 1
fi

echo "=== submitting variant $VARIANT ($WANT jobs) ==="
for VF in $TAB_VFS; do
  sbatch -J "tab_${VF}_${VARIANT}" -o "logs/tab_${VF}_${VARIANT}.out" \
    --export=ALL,VF="$VF",VARIANT="$VARIANT" notebooks/oddshap/cluster/tabular.sbatch \
    || echo "WARN: tabular $VF submit failed" >&2
done
for VF in $GPU_VFS; do
  sbatch -J "gpu_${VF}_${VARIANT}" -o "logs/gpu_${VF}_${VARIANT}_%a.out" \
    --export=ALL,VF="$VF",VARIANT="$VARIANT" --array=0-9 notebooks/oddshap/cluster/gpu.sbatch \
    || echo "WARN: gpu $VF submit failed" >&2
done
echo "=== queue ==="
squeue -u "$USER" -o "%.12i %.24j %.8T %.10M %R" | head -40
