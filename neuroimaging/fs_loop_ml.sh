#!/bin/bash

# Define the strings for experiment and metric
experiments=("modality_only" "demographics_and_modality")
# metrics=("roc_auc" "f3" "ap")
metrics=("roc_auc")

# Nested loops to iterate over the strings
for experiment in "${experiments[@]}"; do
    for metric in "${metrics[@]}"; do
        for region_index in {0..9}; do
            echo "Running script with experiment: $experiment and metric: $metric"
            #Set the partition and time based on the experiment
            if [[ $experiment == "modality_only" || $experiment == "demographics_and_modality" ]]; then
                partition="short"
                time="1:00:00"
                mem="24G"
            # else
            #     partition="short"
            #     time="3:00:00"
            #     mem="64G"
            fi

            # Replace the following line with the command you want to run
            # sbatch --partition="$partition" --time="$time" --mem="$mem" --export=experiment="$experiment",metric="$metric" fs_sh_ml_experiments.sh
            sbatch --partition="$partition" --time="$time" --export=experiment="$experiment",metric="$metric",region_index="$region_index" fs_sh_ml_experiments.sh
        done
    done
done

