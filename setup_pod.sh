
set -euo pipefail

echo "============================================================"
echo ">>> 1. Installing System Utilities"
echo "============================================================"
apt-get update -y
apt-get install -y git curl wget build-essential python3-venv zip unzip htop

echo "============================================================"
echo ">>> 2. Creating Persistent Virtual Environment"
echo "============================================================"
if [ ! -d "/workspace/venv" ]; then
    python3 -m venv /workspace/venv
    echo "[INFO] Created virtual environment at /workspace/venv"
fi

if ! grep -q "source /workspace/venv/bin/activate" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# Automatically activate DRACO project environment" >> ~/.bashrc
    echo "source /workspace/venv/bin/activate" >> ~/.bashrc
fi

# Explicitly activate for this setup session
# shellcheck disable=SC1091
source /workspace/venv/bin/activate

echo "============================================================"
echo ">>> 3. Upgrading Installer & Forcing Binary Requirements"
echo "============================================================"
/workspace/venv/bin/pip install --upgrade pip

# Force the correct CUDA 12.1 torch wheels to match the L40S cloud runtime
echo "[INFO] Downloading pre-compiled PyTorch binaries..."
/workspace/venv/bin/pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

if [ -f "requirements.txt" ]; then
    echo "[INFO] Installing remaining dependencies using pre-compiled binaries..."
    # --prefer-binary prevents packages like sympy/pysr from compiling from source for hours
    /workspace/venv/bin/pip install --no-cache-dir --prefer-binary -r requirements.txt
else
    echo "[ERROR] requirements.txt not found!"
    exit 1
fi

echo "============================================================"
echo ">>> 4. Hardware Validation"
echo "============================================================"
/workspace/venv/bin/python -c "import torch; print('-> PyTorch Version:', torch.__version__); print('-> CUDA Available:', torch.cuda.is_available()); print('-> Target GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo "============================================================"
echo ">>> SUCCESS: SETUP COMPLETE! Run 'exec bash' to begin. <<<"
echo "============================================================"