#!/bin/bash

# Define the strings for experiment and metric
experiments=('demographics' 'ptau217' 'demographics_ptau217' 'demographics_ptau217_no_apoe' 'demographics_no_apoe') 


# Nested loops to iterate over the strings
for predictor in "${experiments[@]}"; do
    for fold_index in {0..9}; do
        
        job_name="${predictor}_${fold_index}"
        output_dir="${predictor}/"
        error_dir="${predictor}/"

        echo "Running script on modality $predictor,\
        Experiment: $predictor,\
        Predict Alzheimer's only: $predict_alzheimers_only"

        sbatch --job-name="$job_name"_%J \
                --output="$output_dir"job_%J_$job_name.out \
                --error="$error_dir"job_%J_$job_name.err \
                --export=predictor="$predictor",fold="$fold" sh_t2e.sh
    done
done
