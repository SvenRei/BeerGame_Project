#!/usr/bin/env bash
# setup_local.sh -- one-shot LOCAL test bootstrap (run in Git Bash on Windows, or sh on macOS/Linux).
#
# Your system Python has no packages, so this creates a project-local .venv, installs the deps, and
# runs the unit tests + smoke tests. Two tiers:
#
#   bash setup_local.sh          # FAST: numpy/scipy/pettingzoo/gymnasium/pytest -> env + scripts smoke
#                                #       (no torch; the agent/training/eval stages auto-skip)
#   bash setup_local.sh --full   # ALSO install torch (CPU) + wandb/hydra -> the FULL smoke incl. agent
#
# The fast tier is seconds-to-a-minute. --full downloads torch (~200 MB) and runs 8 tiny trainings
# (a few minutes on CPU). Re-running reuses the existing .venv. Add .venv/ (and weights_draco/,
# results/smoke_c1/) to .gitignore so smoke artifacts don't show up in git.
set -euo pipefail
cd "$(dirname "$0")"
FULL=0; [ "${1:-}" = "--full" ] && FULL=1

PYBIN="${PYTHON:-python}"; command -v "$PYBIN" >/dev/null 2>&1 || PYBIN=python3
echo ">> bootstrapping with $("$PYBIN" --version 2>&1)"

[ -d ".venv" ] || "$PYBIN" -m venv .venv
VPY=".venv/Scripts/python.exe"; [ -f "$VPY" ] || VPY=".venv/bin/python"   # Windows vs POSIX layout
"$VPY" -m pip install -q -U pip

echo ">> installing lightweight deps (numpy scipy pettingzoo gymnasium pytest) ..."
"$VPY" -m pip install -q numpy scipy pettingzoo gymnasium pytest

if [ "$FULL" = "1" ]; then
  echo ">> installing torch (CPU) + wandb/hydra/omegaconf for the agent/training stages ..."
  "$VPY" -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
  "$VPY" -m pip install -q wandb hydra-core omegaconf
fi

echo ">> running smoke tests with the venv interpreter ..."
PY="$VPY" bash smoke_test.sh
