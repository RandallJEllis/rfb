#!/bin/bash
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --exclude=compute-f-17-[09-16]
#SBATCH --time=12:00:00
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

# Convert SLURM_TIMELIMIT from minutes to HH:MM:SS and print
# time_limit_in_minutes="$SLURM_JOB_END_TIME"
# echo "Projected end time for this job is: $time_limit_in_minutes"

# echo "Running experiment with experiment: $experiment and metric: $metric"
python t2e.py --predictor "$predictor" --fold "$fold"

# running a single experiment
# predictor=$1
# fold=$2
# echo "Running experiment with experiment: $1 and metric: $2"
# python t2e.py --predictor "$predictor" --fold "$fold"


