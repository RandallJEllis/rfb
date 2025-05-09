#!/bin/bash
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --exclude=compute-f-17-[09-16]
#SBATCH --time=12:00:00
#SBATCH --mem=16G
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
python timetoevent_experiments.py --modality "$modality" --experiment "$experiment" --metric "$metric" --model "$model" --region_index "$region_index" --age_cutoff "$age_cutoff" --predict_alzheimers_only "$predict_alzheimers_only"

# running a single experiment
# experiment=$1
# metric=$2
# echo "Running experiment with experiment: $1 and metric: $2"
# python ml_experiments.py --experiment "$experiment" --metric "$metric" --age_cutoff 65



