#!/bin/bash
#SBATCH --job-name=schwinger-relax
#SBATCH --account=CHANGEME            # <-- run `hyakalloc` and put your account here
#SBATCH --partition=compute           # <-- and the partition it lists (e.g. compute, ckpt)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --array=0                     # 0 -> L=12, 1 -> L=14, 2 -> L=16 (see --L-list)
#SBATCH --output=logs/relax_%A_%a.out
#SBATCH --error=logs/relax_%A_%a.out
#SBATCH --requeue                     # so a preempted ckpt job comes back and resumes
#
# One system size per array task.  The sizes cost VERY different amounts (see
# HYAK_README.md), so submit them separately rather than as one array with one
# --time; submit.sh does that for you.
#
#   sbatch --array=0 relax.slurm                  # L = 12
#   sbatch --array=1 --time=72:00:00 relax.slurm  # L = 14
#
set -euo pipefail

REPO="${REPO:-$PWD}"                  # override if you sbatch from somewhere else
RESULTS="${RESULTS:-$REPO/hpc/results}"
BACKEND="${BACKEND:-scipy}"           # scipy | mkl | gpu

mkdir -p logs "$RESULTS"

# Threads: the scipy backend is single-threaded in the sparse products but LAPACK
# still threads the fidelity's eigh/svd, so give BLAS the cores we asked for.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

source "$REPO/hpc/env.sh"             # activates the conda env; see HYAK_README.md

echo "host=$(hostname)  task=${SLURM_ARRAY_TASK_ID:-none}  backend=$BACKEND"
echo "cpus=${SLURM_CPUS_PER_TASK:-?}  mem=${SLURM_MEM_PER_NODE:-?}M"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# --ckpt-every writes the trajectory to $RESULTS every 30 min.  If this job hits its
# wall-clock limit or is preempted, resubmitting the SAME command picks up where it
# stopped instead of restarting -- which is what makes L = 14 and 16 survivable.
srun python -u "$REPO/hpc/run_relax.py" \
    --L-list 12 14 16 \
    --out "$RESULTS" \
    --backend "$BACKEND" \
    --ckpt-every 1800

echo "done: $(date)"
