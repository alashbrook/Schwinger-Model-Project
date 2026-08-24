#!/bin/bash
# One-time environment build.  RUN THIS ON THE LOGIN NODE (compute nodes have no
# outbound network on klone, so nothing can be downloaded from inside a job).
#
#   bash hpc/setup_env.sh
#
# Installs miniforge into $CONDA_ROOT (default ~/miniforge3) and creates the
# `thermal` env with the pinned versions from requirements.txt.
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
ENV_NAME="${ENV_NAME:-thermal}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$CONDA_ROOT" ]; then
  echo "installing miniforge into $CONDA_ROOT ..."
  curl -fsSL -o /tmp/miniforge.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash /tmp/miniforge.sh -b -p "$CONDA_ROOT"
  rm -f /tmp/miniforge.sh
fi

source "$CONDA_ROOT/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.12
fi
conda activate "$ENV_NAME"

python -m pip install --upgrade pip
python -m pip install -r "$HERE/requirements.txt"

echo
echo "checking the environment reproduces the notebook ..."
python "$HERE/run_relax.py" --selftest

cat <<MSG

environment ready.
  CONDA_ROOT=$CONDA_ROOT  ENV_NAME=$ENV_NAME

optional accelerators for L = 16 (see HYAK_README.md):
  pip install sparse-dot-mkl      # threaded sparse products on a CPU node
  pip install cupy-cuda12x        # GPU nodes; check the CUDA version with nvidia-smi
MSG
