#!/bin/bash
# One-shot deploy of the reproduction onto a single AutoDL-style GPU box (RTX 3080 Ti,
# torch pre-installed). Idempotent — safe to re-run. Run it ON the machine, or push it
# via the fleet deployer. Assumes miniconda python with a working CUDA torch already
# present (AutoDL images ship one that matches the driver — we do NOT touch torch).
set -e
export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com          # China-friendly HuggingFace mirror
[ -f /etc/network_turbo ] && source /etc/network_turbo 2>/dev/null || true

REPO=/root/autodl-tmp/oddshap_reproduction
BRANCH=wu/oddshap-reproduction

# --- code: clone/update; if GitHub is unreachable, keep whatever is already there ---
mkdir -p /root/autodl-tmp
if [ ! -d "$REPO/.git" ]; then
  git clone -q https://github.com/FabianK-Dev/shapiq.git "$REPO" || {
    echo "WARN: git clone failed (network) — code must be pushed via SFTP"; }
fi
cd "$REPO" 2>/dev/null && git fetch origin -q && git checkout -q "$BRANCH" \
  && git reset --hard origin/"$BRANCH" -q && echo "code HEAD $(git rev-parse --short HEAD)" \
  || echo "WARN: git update skipped (offline) — using on-disk code"

# --- deps: keep the pre-installed CUDA torch, add only what the reproduction needs ---
python -c "import torch;print('torch',torch.__version__)" >/dev/null 2>&1 || { echo "FATAL: no torch"; exit 1; }
TORCH_V=$(python -c "import torch;print(torch.__version__.split('+')[0])")
printf "torch==%s\n" "$TORCH_V" > /tmp/pin.txt
python -c "import torchvision" 2>/dev/null && \
  printf "torchvision==%s\n" "$(python -c 'import torchvision;print(torchvision.__version__.split("+")[0])')" >> /tmp/pin.txt
# transformers 4.44 loads DistilBERT with torch<2.6 (newer versions require torch>=2.6)
pip install -q -e . -c /tmp/pin.txt
pip install -q "transformers==4.44.2" xgboost lightgbm scikit-learn joblib pandas openpyxl -c /tmp/pin.txt

# --- verify GPU + models reachable ---
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available"
print("cuda OK:", torch.cuda.get_device_name(0))
import shapiq, shapiq_games            # noqa: F401
print("shapiq OK")
PY
echo "DEPLOY_OK $(hostname)"
