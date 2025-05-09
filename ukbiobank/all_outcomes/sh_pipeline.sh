#!/bin/bash
#SBATCH --job-name=rfb_pipeline
#SBATCH --output=job_%A_%a.out
#SBATCH --error=job_%A_%a.err
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --time=00:07:00
#SBATCH --mem=16G

# Load modules (modify if necessary)
module load gcc/9.2.0
module load graphviz/3.0.0

# Set PYTHONUNBUFFERED to ensure immediate flushing of print statements
export PYTHONUNBUFFERED=1

# Use the array task ID directly
MYVAR=${SLURM_ARRAY_TASK_ID}

python rfb_pipeline.py --outcome $MYVAR