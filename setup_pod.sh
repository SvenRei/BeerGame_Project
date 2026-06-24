#!/usr/bin/env bash
# setup_pod.sh -- Installs all system dependencies, Julia, and Python packages for DRACO.

set -euo pipefail

echo ">>> 1. Updating system packages..."
apt-get update -y
apt-get install -y git curl wget build-essential python3-venv

echo ">>> 2. Installing Julia (Required for PySR symbolic distillation)..."
# PySR requires a local Julia installation to perform the symbolic regression search
wget https://julialang-s3.julialang.org/bin/linux/x64/1.9/julia-1.9.3-linux-x86_64.tar.gz
tar zxvf julia-1.9.3-linux-x86_64.tar.gz -C /usr/local --strip-components=1
rm julia-1.9.3-linux-x86_64.tar.gz

echo ">>> 3. Setting up Python virtual environment..."
python3 -m venv /workspace/venv
source /workspace/venv/bin/activate

echo ">>> 4. Upgrading pip and installing requirements..."
pip install --upgrade pip
# Installs Torch with CUDA 12.1 wheels first, then RL & math packages
pip install -r requirements.txt

echo ">>> 5. Installing PySR backend via Julia..."
python -c "import pysr; pysr.install()"

echo "============================================================"
echo "Setup Complete!"
echo "To activate the environment, run: source /workspace/venv/bin/activate"
echo "Don't forget to export your W&B API key before starting the sweep:"
echo "export WANDB_API_KEY='your_key_here'"
echo "============================================================"