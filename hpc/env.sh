# Activate the project environment on klone.  Sourced by every SLURM script.
# Created once by hpc/setup_env.sh -- see HYAK_README.md.
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniforge3}"
ENV_NAME="${ENV_NAME:-thermal}"

if [ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
  echo "no conda at $CONDA_ROOT -- run hpc/setup_env.sh first (or set CONDA_ROOT)" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
