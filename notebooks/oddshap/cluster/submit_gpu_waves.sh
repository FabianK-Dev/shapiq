#!/bin/bash
# Submit the deep-learning value functions as many SHORT per-instance jobs, kept flowing in
# waves so the total queued job count never exceeds the QOS submit cap. Short jobs backfill
# onto idle NvidiaAll windows instead of waiting ~days behind higher-priority long jobs.
#
#   bash notebooks/oddshap/cluster/submit_gpu_waves.sh v522_merged          # all experiments
#   EXPERIMENTS="table1 eta" bash .../submit_gpu_waves.sh v522_merged  # a fast subset
#
# Runs in the foreground and blocks until every instance is submitted; launch it detached:
#   nohup bash notebooks/oddshap/cluster/submit_gpu_waves.sh v522_merged > logs/gpu_waves.out 2>&1 &
set -uo pipefail
cd "$HOME/oddshap_reproduction"
mkdir -p logs notebooks/oddshap/data
export EXPERIMENTS="${EXPERIMENTS:-}"

VARIANT="${1:?usage: submit_gpu_waves.sh <variant> (env EXPERIMENTS optional)}"
GPU_VFS="vit16 distilbert"
N_INST=30
CAP=28                 # keep 2 slots of head-room under MaxSubmitJobs=30
POLL=45                # seconds between queue checks

queued() { squeue -u "$USER" -h -r 2>/dev/null | wc -l; }

echo "=== GPU waves · variant=$VARIANT · experiments=${EXPERIMENTS:-all} ==="
submitted=0
for VF in $GPU_VFS; do
  for INST in $(seq 0 $((N_INST - 1))); do
    while [ "$(queued)" -ge "$CAP" ]; do
      echo "$(date '+%H:%M:%S')  queue full ($(queued)/$CAP) — waiting…"; sleep "$POLL"
    done
    sbatch -J "gpu1_${VF}_${INST}_${VARIANT}" -o "logs/gpu1_${VF}_${INST}_${VARIANT}.out" \
      --export=ALL,VF="$VF",VARIANT="$VARIANT",INST="$INST",EXPERIMENTS="$EXPERIMENTS" \
      notebooks/oddshap/cluster/gpu_one.sbatch >/dev/null \
      && submitted=$((submitted + 1)) \
      && echo "$(date '+%H:%M:%S')  submitted $VF inst=$INST  (total $submitted, queue $(queued))" \
      || echo "WARN: submit failed $VF inst=$INST"
    sleep 2
  done
done
echo "=== all $submitted per-instance GPU jobs submitted for $VARIANT ==="
