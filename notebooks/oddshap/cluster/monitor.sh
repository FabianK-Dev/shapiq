#!/bin/bash
# Live training dashboard for the OddSHAP reproduction runs.
#
#   one snapshot:   bash notebooks/oddshap/cluster/monitor.sh
#   live (refresh):  bash notebooks/oddshap/cluster/monitor.sh watch        (Ctrl-C to stop)
#   pick a variant:  VARIANT=v560_improved bash notebooks/oddshap/cluster/monitor.sh
#
# From your laptop:
#   ssh krater20 'bash ~/oddshap_reproduction/notebooks/oddshap/cluster/monitor.sh'
#   ssh -t krater20 'bash ~/oddshap_reproduction/notebooks/oddshap/cluster/monitor.sh watch'
export PATH="$HOME/.local/bin:$PATH"
REPO="$HOME/oddshap_reproduction"
DATA="$REPO/notebooks/oddshap/data"
LOGS="$REPO/logs"
VARIANT="${VARIANT:-v522_merged}"
TAB_VFS="cancer realestate corrgroups60 independentlinear60 nhanes crime"
GPU_VFS="vit16 distilbert"

snapshot() {
  echo "=============================================================="
  echo " OddSHAP training monitor · variant=$VARIANT · $(date '+%H:%M:%S')"
  echo "=============================================================="

  # --- queue ---
  local run pend
  run=$(squeue -u "$USER" -h -t RUNNING -r 2>/dev/null | wc -l)
  pend=$(squeue -u "$USER" -h -t PENDING -r 2>/dev/null | wc -l)
  echo "QUEUE   running=$run  pending=$pend  (QOS: 15 concurrent / 30 submit)"
  squeue -u "$USER" -o "  %.10i %.24j %.8T %.6M %R" 2>/dev/null | head -14
  echo ""

  # --- tabular progress: which experiment CSVs exist per VF ---
  echo "TABULAR  (T1=Table1  F2=Figure2  ETA=Figure4/11  RT=runtime)"
  printf "  %-20s %-4s %-4s %-5s %-4s\n" "value function" "T1" "F2" "ETA" "RT"
  for vf in $TAB_VFS; do
    t=$([ -f "$DATA/table1_${vf}_${VARIANT}.csv" ] && echo "OK" || echo "..")
    f=$([ -f "$DATA/fig2_${vf}_${VARIANT}.csv" ]   && echo "OK" || echo "..")
    e=$([ "$vf" = realestate ] && echo "n/a" || { [ -f "$DATA/eta_${vf}_${VARIANT}.csv" ] && echo "OK" || echo ".."; })
    r=$([ -f "$DATA/runtime_${vf}_${VARIANT}.csv" ] && echo "OK" || echo "..")
    printf "  %-20s %-4s %-4s %-5s %-4s\n" "$vf" "$t" "$f" "$e" "$r"
  done
  echo ""

  # --- GPU progress: instances finished per VF (INSTANCE_DONE lines in the logs) ---
  echo "GPU      instances finished / 30"
  for vf in $GPU_VFS; do
    local done part
    done=$(grep -h "^INSTANCE_DONE $vf $VARIANT" $LOGS/gpu_${vf}_${VARIANT}_*.out 2>/dev/null | wc -l)
    part=$(grep -h "^PARTIAL_T1 $vf " $LOGS/gpu_${vf}_${VARIANT}_*.out 2>/dev/null | wc -l)
    printf "  %-12s %2s/30   (%s Table-1 result lines so far)\n" "$vf" "$done" "$part"
  done
  echo ""

  # --- health: WARN / tracebacks in any log ---
  local warns
  warns=$(grep -rhiE "WARN|Traceback|Error:" $LOGS/*.out "$REPO"/logs/*.out 2>/dev/null | grep -viE "HF_TOKEN|Deprecation|filterwarnings" | tail -4)
  if [ -n "$warns" ]; then
    echo "WARN / ERRORS (recent):"; echo "$warns" | sed 's/^/  /'
  else
    echo "HEALTH  no WARN/errors in logs"
  fi
}

if [ "${1:-}" = "watch" ]; then
  while true; do clear; snapshot; echo ""; echo "(refresh 20s · Ctrl-C to stop)"; sleep 20; done
else
  snapshot
fi
