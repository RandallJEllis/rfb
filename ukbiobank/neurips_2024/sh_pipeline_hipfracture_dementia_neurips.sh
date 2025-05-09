#!/bin/bash
#SBATCH --job-name=rfb_pipeline
#SBATCH --output=job_%J.out
#SBATCH --error=job_%J.err
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --exclude=compute-f-17-[09-16]
#SBATCH --time=2:00:00
#SBATCH --mem=16G

# Load modules (modify if necessary)
module load gcc/9.2.0
module load graphviz/3.0.0

# conda init bash
source ~/.bashrc
conda activate pymc_env

# Set PYTHONUNBUFFERED to ensure immediate flushing of print statements
export PYTHONUNBUFFERED=1

python rfb.py --task_id $MYVAR --outcome hip_fracture --output_path '../tidy_data/demographics/hip_fracture/' --data_path '../../proj_idp/tidy_data/'
