#!/bin/bash
#SBATCH --job-name=rfb_pipeline
#SBATCH --output=job_%J.out
#SBATCH --error=job_%J.err
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --exclude=compute-f-17-[09-16]
#SBATCH --time=3:00:00
#SBATCH --mem=8G

# Load modules (modify if necessary)
module load gcc/9.2.0
module load graphviz/3.0.0

# Set PYTHONUNBUFFERED to ensure immediate flushing of print statements
export PYTHONUNBUFFERED=1

python rfb_pipeline.py --task_id $MYVAR
