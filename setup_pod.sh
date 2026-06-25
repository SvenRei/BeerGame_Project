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

echo ">>> 4. Verify the GPU is visible to torch..."
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

echo ">>> 5. Authenticating with Weights & Biases..."
# Check if the key was passed via RunPod Environment Variables
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WANDB_API_KEY is not set as an environment variable."
    # -s hides the input so your key doesn't get printed in the terminal logs
    read -s -p "Please paste your W&B API key here (input will be hidden): " USER_WANDB_KEY
    echo ""
    if [ -n "$USER_WANDB_KEY" ]; then
        wandb login "$USER_WANDB_KEY"
    else
        echo "WARNING: No key provided. You will need to run 'wandb login' manually before training."
    fi
else
    echo "Found WANDB_API_KEY in environment variables. Logging in..."
    wandb login "$WANDB_API_KEY"
fi

echo "============================================================"
echo "Setup complete! The persistent .netrc file has been generated."
echo "Every NEW shell window simply needs to re-activate the venv:"
echo "    source /workspace/venv/bin/activate"
echo "You are ready to launch your sweeps!"
echo "============================================================"