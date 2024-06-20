#!/bin/bash

# Define the strings for experiment and metric
experiments=("age_only" "all_demographics" "idps_only" "demographics_and_idps")
# metrics=("roc_auc" "f3")
metrics=("ap")
# Nested loops to iterate over the strings
for experiment in "${experiments[@]}"; do
    for metric in "${metrics[@]}"; do
        echo "Running script with experiment: $experiment and metric: $metric"
        # Set the partition and time based on the experiment
        if [[ $experiment == "idps_only" || $experiment == "demographics_and_idps" ]]; then
            partition="medium"
            time="18:00:00"
            mem="128G"
        else
            partition="short"
            time="3:00:00"
            mem="64G"
        fi

        # Replace the following line with the command you want to run
        sbatch --partition="$partition" --time="$time" --mem="$mem" --export=experiment="$experiment",metric="$metric" sh_ml_experiments.sh
        # sbatch --export=experiment="$experiment",metric="$metric" sh_ml_experiments.sh

    done
done

