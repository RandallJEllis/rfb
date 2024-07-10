#!/bin/bash

# Define the strings for experiment and metric
# experiments=("age_only" "all_demographics" "proteins_only" "demographics_and_proteins")
experiments=("proteins_only" "demographics_and_proteins")

# metrics=("roc_auc" "f3" "ap")
metrics=("roc_auc")

# Nested loops to iterate over the strings
for experiment in "${experiments[@]}"; do
    for metric in "${metrics[@]}"; do
        for region_index in {0..9}; do
            echo "Running script with experiment: $experiment and metric: $metric"
            # Set the partition and time based on the experiment
            if [[ $experiment == "proteins_only" || $experiment == "demographics_and_proteins" ]]; then
                partition="short"
                time="4:00:00"
            else
                partition="short"
                time="0:45:00"
            fi

            # Replace the following line with the command you want to run
            sbatch --partition="$partition" --time="$time" --export=experiment="$experiment",metric="$metric",region_index="$region_index" sh_ml_experiments.sh
            # sbatch --export=experiment="$experiment",metric="$metric" sh_ml_experiments.sh
        done
    done
done
