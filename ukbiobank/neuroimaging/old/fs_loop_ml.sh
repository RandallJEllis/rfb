#!/bin/bash

# Define the strings for experiment and metric
# experiments=("modality_only" "demographics_and_modality")
experiments=("demographics_modality_lancet2024")
# metrics=("roc_auc" "f3" "ap")
metrics=("log_loss")

age_cutoffs=(0 65)

# Nested loops to iterate over the strings
for experiment in "${experiments[@]}"; do
    for metric in "${metrics[@]}"; do
        for age_cutoff in "${age_cutoffs[@]}"; do
            for region_index in {1..9}; do
                echo "Running feature selection with experiment: $experiment and metric: $metric and age cutoff: $age_cutoff and region index: $region_index"

                partition="short"
                time="0:08:00"
                mem="16G"

                if [[ $age_cutoff -eq 65 ]]; then
                    mem="8G"
                fi

                # Replace the following line with the command you want to run
                # sbatch --partition="$partition" --time="$time" --mem="$mem" --export=experiment="$experiment",metric="$metric" fs_sh_ml_experiments.sh
                sbatch --partition="$partition" --time="$time" --mem="$mem" --export=experiment="$experiment",metric="$metric",age_cutoff="$age_cutoff",region_index="$region_index" fs_sh_ml_experiments.sh
            done
        done
    done
done

