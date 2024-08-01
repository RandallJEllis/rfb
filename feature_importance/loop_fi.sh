#!/bin/bash

# Define the strings for experiment and metric
# modalities=("cognitive_tests" "neuroimaging" "proteomics")
modalities=("cognitive_tests" "proteomics")

experiments=("all_demographics" "modality_only" "demographics_and_modality")
# metrics=("roc_auc" "f3" "ap")
metrics=("log_loss")

age_cutoffs=(0 65)

# Nested loops to iterate over the strings
for modality in "${modalities[@]}"; do
    for experiment in "${experiments[@]}"; do
        for metric in "${metrics[@]}"; do
            for age_cutoff in "${age_cutoffs[@]}"; do
                for region_index in {0..9}; do
                    echo "Running script with modality: $modality, experiment: $experiment, and metric: $metric, age cutoff $age_cutoff, region index $region_index"

                    # Set the partition and time based on the modality and experiment
                    if [[ "$modality" == "neuroimaging" ]]; then
                        job_name=neuroimaging
                        mem="64G"
                        
                    elif [[ "$modality" == "proteomics" ]]; then
                        job_name=proteomics
                        mem="24G"
                        
                    elif [[ "$modality" == "cognitive_tests" ]]; then
                        job_name=cognitive_tests
                        mem="4G"
                    fi

                    # Replace the following line with the command you want to run
                    # sbatch --partition="$partition" --time="$time" --export=experiment="$experiment",metric="$metric" sh_ml_experiments.sh
                    sbatch --job-name="$job_name" --mem="$mem" --export=modality="$modality",experiment="$experiment",metric="$metric",age_cutoff="$age_cutoff",region_index="$region_index" sh_fi.sh
                    # sbatch --export=experiment="$experiment",metric="$metric" sh_ml_experiments.sh
                done
            done
        done
    done
done