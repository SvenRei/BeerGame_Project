#!/usr/bin/env bash
# setup_pod.sh -- Master automated environment installation for DRACO v4

set -euo pipefail

echo "============================================================"
echo ">>> 1. Updating Linux Packages & Core Utilities"
echo "============================================================"
apt-get update -y
apt-get install -y git curl wget build-essential python3-venv zip unzip

echo "============================================================"
echo ">>> 2. Creating Persistent Virtual Environment"
echo "============================================================"
if [ ! -d "/workspace/venv" ]; then
    python3 -m venv /workspace/venv
    echo "[INFO] Created virtual environment at /workspace/venv"
else
    echo "[INFO] Existing virtual environment found. Refreshing."
fi

echo "============================================================"
echo ">>> 3. Automating Future Logins (~/.bashrc)"
echo "============================================================"
if ! grep -q "source /workspace/venv/bin/activate" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# Automatically activate DRACO project environment" >> ~/.bashrc
    echo "source /workspace/venv/bin/activate" >> ~/.bashrc
    echo "[INFO] Added automatic venv activation to ~/.bashrc"
fi

# Activate explicitly for the rest of the script
# shellcheck disable=SC1091
source /workspace/venv/bin/activate

echo "============================================================"
echo ">>> 4. Upgrading Package Installer & Downloading Requirements"
echo "============================================================"
/workspace/venv/bin/pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    echo "[INFO] Installing project dependencies from requirements.txt..."
    # --no-cache-dir prevents RunPod out-of-memory container crashes
    /workspace/venv/bin/pip install --no-cache-dir -r requirements.txt
else
    echo "[ERROR] requirements.txt not found!"
    exit 1
fi

echo "============================================================"
echo ">>> 5. Hardware Validation"
echo "============================================================"
/workspace/venv/bin/python -c "import torch; print('-> PyTorch Version:', torch.__version__); print('-> CUDA Available:', torch.cuda.is_available()); print('-> Target GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo "============================================================"
echo ">>> SUCCESS: SETUP COMPLETE! <<<"
echo "============================================================"