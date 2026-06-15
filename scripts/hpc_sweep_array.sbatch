#!/bin/bash
# Example SLURM job array for a chunked, resumable SparseGF2 sweep.
# One array task per chunk (here 2 pictures x 4 system sizes = 8). Each task
# runs on its own node and fills its allocated cores; finished chunks are
# skipped, so re-queuing after a timeout costs nothing already computed.
#
#   mkdir -p logs && sbatch scripts/hpc_sweep_array.sbatch
#
# Create logs/ before submitting: SLURM opens the --output file below before the
# job body runs, so the directory must already exist. Adapt the resource lines
# and the module/venv activation to your cluster.

#SBATCH --job-name=sparsegf2-sweep
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=logs/sweep_%A_%a.out

set -euo pipefail

# --- environment (adapt to your cluster) ---
module load python/3.12 || true
source .venv/bin/activate

# --- headless settings ---
export SPARSEGF2_PROGRESS=0          # silence progress bars in batch logs
export STUDY_OUT="runs/hpc_study"    # where the resumable chunks are written
mkdir -p logs "$STUDY_OUT"

# n_jobs=-1 inside the script resolves to SLURM_CPUS_PER_TASK, so the sweep
# fills exactly the allocation above.
srun python scripts/hpc_sweep_array.py
