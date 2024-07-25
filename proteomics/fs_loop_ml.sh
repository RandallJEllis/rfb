#!/bin/bash

# Define the strings for experiment and metric
experiments=("modality_only" "demographics_and_modality")
# metrics=("roc_auc" "f3" "ap")
metrics=("log_loss")

age_cutoffs=(0 65)

# Nested loops to iterate over the strings
for experiment in "${experiments[@]}"; do
    for metric in "${metrics[@]}"; do
        for age_cutoff in "${age_cutoffs[@]}"; do
            for region_index in {0..9}; do
            # for region_index in {6..7}; do
                echo "Running feature selection with experiment: $experiment and metric: $metric and age cutoff: $age_cutoff and region index: $region_index"

                # Set the partition and time based on the experiment
                partition="short"
                time="0:30:00"

                # set time if region_index is 6 and the experiment is demographics_and_modality
                if [[ ($region_index -eq 6 && $experiment == "demographics_and_modality" && $age_cutoff -eq 0) || ($region_index -eq 7 && $experiment == "modality_only" && $age_cutoff -eq 0) ]]; then
                    time="1:00:00"
                # else
                #     continue
                fi

                # Replace the following line with the command you want to run
                sbatch --partition="$partition" --time="$time" --export=experiment="$experiment",metric="$metric",age_cutoff="$age_cutoff",region_index="$region_index" fs_sh_ml_experiments.sh
            done
        done
    done
done

