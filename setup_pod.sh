#!/usr/bin/env bash
# setup_pod.sh -- system deps + Python env for DRACO. Run ONCE on a fresh RunPod.
# Phases 1-3 need only Python deps. Julia/PySR are Phase-4-only and install themselves
# on first `import pysr` (modern PySR manages its own Julia via juliacall) -- so we do NOT
# hand-install Julia here (a system Julia can shadow PySR's managed one).
set -euo pipefail

echo ">>> 1. System packages..."
apt-get update -y
apt-get install -y git curl wget build-essential python3-venv zip unzip

echo ">>> 2. Python virtual environment at /workspace/venv ..."
python3 -m venv /workspace/venv
# shellcheck disable=SC1091
source /workspace/venv/bin/activate

echo ">>> 3. pip + requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo ">>> 4. Verify the GPU is visible to torch (L4 should print CUDA: True)..."
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

echo "============================================================"
echo "Setup complete. Every NEW shell must re-activate the venv:"
echo "    source /workspace/venv/bin/activate"
echo "Then export your W&B key (or the run will hang at wandb.init):"
echo "    export WANDB_API_KEY='your_key_here'   # or: wandb login"
echo ""
echo "Phase 4 only: the first 'import pysr' will download + build the Julia"
echo "backend (a few minutes). Phases 1-3 do not need it."
echo "============================================================"