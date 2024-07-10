#!/bin/bash

# Define the strings for experiment and metric
experiments=("cognitive_tests_only" "demographics_and_cognitive_tests")
metrics=("roc_auc" "f3" "ap")

# Nested loops to iterate over the strings
for experiment in "${experiments[@]}"; do
    for metric in "${metrics[@]}"; do
        echo "Running script with experiment: $experiment and metric: $metric"
        # Set the partition and time based on the experiment
        # if [[ $experiment == "idps_only" || $experiment == "demographics_and_idps" ]]; then
        #     partition="medium"
        #     time="18:00:00"
        # else
        #     partition="short"
        #     time="3:00:00"
        # fi

        # Replace the following line with the command you want to run
        # sbatch --partition="$partition" --time="$time" --export=experiment="$experiment",metric="$metric" sh_ml_experiments.sh
        sbatch --export=experiment="$experiment",metric="$metric" fs_sh_ml_experiments.sh
        # sbatch --export=experiment="$experiment",metric="$metric" fs_sh_ml_experiments.sh

    done
done

