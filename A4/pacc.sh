#!/bin/bash
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --time=00:59:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1

# Load modules (modify if necessary)
module load gcc/9.2.0
module load graphviz/3.0.0

# conda init bash
source ~/.bashrc
conda activate pymc_env

# Set PYTHONUNBUFFERED to ensure immediate flushing of print statements
export PYTHONUNBUFFERED=1

# Get the visit value from the array task ID
VISITS=(6 12 18 24 30 36 42 48 54 60 66 72 78 84 90 96 102 108 114 701 997 999)
VISIT=${VISITS[$SLURM_ARRAY_TASK_ID]}

# PACC.raw or PACC are the only options
python pacc_experiments.py --pacc_col PACC --visit $VISIT