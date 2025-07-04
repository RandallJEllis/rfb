#!/bin/bash
#SBATCH --job-name=cognitive_tests
#SBATCH --output=job_%J.out
#SBATCH --error=job_%J.err
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --exclude=compute-f-17-[09-16]
#SBATCH --time=4:00:00
#SBATCH --mem=4G

# Load modules (modify if necessary)
module load gcc/9.2.0
module load graphviz/3.0.0

# conda init bash
source ~/.bashrc
conda activate pymc_env

# Set PYTHONUNBUFFERED to ensure immediate flushing of print statements
export PYTHONUNBUFFERED=1

echo "Running experiment with experiment: modality: $modality, experiment: $experiment, and metric: $metric, age cutoff $age_cutoff"
python ml_65_and_older.py --modality "$modality" --experiment "$experiment" --metric "$metric" --age_cutoff "$age_cutoff"

# running a single experiment
#modality=$1
#experiment=$2
#metric=$3
#echo "Running experiment with experiment: $1 and metric: $2"
#echo "Running experiment with experiment: modality: $modality, experiment: $experiment, and metric: $metric"
#python ml_65_and_older.py --modality "$modality" --experiment "$experiment" --metric "$metric" #--age_cutoff 65


