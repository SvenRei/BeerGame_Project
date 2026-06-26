#!/usr/bin/env bash
# setup_pod.sh -- Clean automated system setup for DRACO v4

# Exit immediately if an unhandled command fails
set -euo pipefail

echo "============================================================"
echo ">>> 1. Updating System Packages & Installing Core Utilities"
echo "============================================================"
apt-get update -y
apt-get install -y git curl wget build-essential python3-venv zip unzip

echo "============================================================"
echo ">>> 2. Configuring Persistent Python Virtual Environment"
echo "============================================================"
# Create the virtual environment in the persistent workspace
if [ ! -d "/workspace/venv" ]; then
    python3 -m venv /workspace/venv
    echo "[INFO] Created new virtual environment at /workspace/venv"
else
    echo "[INFO] Existing virtual environment found at /workspace/venv. Skipping creation."
fi

# Activate the venv for the remainder of this setup script execution
# shellcheck disable=SC1091
source /workspace/venv/bin/activate

echo "============================================================"
echo ">>> 3. Automating Future Terminal Logins (~/.bashrc)"
echo "============================================================"
# Permanently add venv activation to the shell startup configuration
if ! grep -q "source /workspace/venv/bin/activate" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# Automatically activate DRACO venv" >> ~/.bashrc
    echo "source /workspace/venv/bin/activate" >> ~/.bashrc
    echo "[INFO] Added automatic venv activation to ~/.bashrc"
else
    echo "[INFO] venv activation already configured in ~/.bashrc"
fi

echo "============================================================"
echo ">>> 4. Upgrading Package Installer & Installing Requirements"
echo "============================================================"
pip install --upgrade pip

# Using --no-cache-dir to completely bypass RunPod RAM spikes
if [ -f "requirements.txt" ]; then
    echo "[INFO] Installing dependencies from requirements.txt..."
    pip install --no-cache-dir -r requirements.txt
else
    echo "[ERROR] requirements.txt not found in the current directory!"
    echo "Make sure you are running this script from inside your repository folder."
    exit 1
fi

echo "============================================================"
echo ">>> 5. Executing Sanity Check & Hardware Validation"
echo "============================================================"
python -c "import torch; print('-> PyTorch Version:', torch.__version__); print('-> CUDA Available:', torch.cuda.is_available()); print('-> Target Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo "============================================================"
echo ">>> SETUP COMPLETE! <<<"
echo "============================================================"
echo "1. To automatically enter your environment right now, run:"
echo "       exec bash"
echo ""
echo "2. Once inside your environment, log into WandB manually:"
echo "       wandb login"
echo "============================================================"