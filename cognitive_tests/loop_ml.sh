#!/bin/bash

# Define the strings for experiment and metric
experiments=("age_only" "all_demographics" "modality_only" "demographics_and_modality")
# experiments=("demographics_and_modality")

# metrics=("roc_auc" "f3" "ap")
metrics=("log_loss")

age_cutoffs=(0 65)
# age_cutoffs=(65)

# Nested loops to iterate over the strings
for experiment in "${experiments[@]}"; do
    for metric in "${metrics[@]}"; do
        for age_cutoff in "${age_cutoffs[@]}"; do
            for region_index in {0..9}; do
            # for region_index in 4; do

                # Set the partition and time based on the experiment
                echo "Running script with experiment: $experiment and metric: $metric and age cutoff: $age_cutoff and region index: $region_index"
                # Set the partition and time based on the experiment
                if [[ $experiment == "modality_only" || $experiment == "demographics_and_modality" ]]; then
                    partition="short"
                    if [[ $age_cutoff -eq 0 ]]; then    
                        # time="3:30:00"
                        time="0:15:00"
                    elif [[ $age_cutoff -eq 65 ]]; then
                        time="0:05:00"
                    fi
                else
                    partition="short"
                    # if [[ $age_cutoff -eq 0 ]]; then    
                    time="0:05:00"
                    # elif [[ $age_cutoff -eq 65 ]]; then
                        # time="0:30:00"
                    # fi
                fi

                # Replace the following line with the command you want to run
                sbatch --partition="$partition" --time="$time" --export=experiment="$experiment",metric="$metric",age_cutoff="$age_cutoff",region_index="$region_index" sh_ml_experiments.sh
            done
        done
    done
done

