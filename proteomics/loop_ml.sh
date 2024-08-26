#!/bin/bash

# Define the strings for experiment and metric
# experiments=("age_only" "all_demographics" "age_and_lancet2024" "demographics_and_lancet2024" "modality_only" "demographics_and_modality" "demographics_modality_lancet2024")
# experiments=("age_and_lancet2024" "age_sex_lancet2024" "demographics_and_lancet2024" "demographics_modality_lancet2024")
experiments=("age_sex_lancet2024")

# metrics=("roc_auc" "f3" "ap")
metrics=("log_loss")

age_cutoffs=(0 65)
# age_cutoffs=(0)

# Nested loops to iterate over the strings
for experiment in "${experiments[@]}"; do
    for metric in "${metrics[@]}"; do
        for age_cutoff in "${age_cutoffs[@]}"; do
            for region_index in {0..9}; do
                echo "Running script with experiment: $experiment and metric: $metric and age cutoff: $age_cutoff and region index: $region_index"
                # Set the partition and time based on the experiment
                if [[ $experiment == "modality_only" || $experiment == "demographics_and_modality" || $experiment == "demographics_modality_lancet2024" ]]; then

                    if [[ $age_cutoff -eq 0 ]]; then
                        # partition="medium"
                        # time="15:00:00"

                        partition="short"
                        time="3:30:00"
                    elif [[ $age_cutoff -eq 65 ]]; then
                        partition="short"
                        # time="5:00:00"
                        time="1:45:00"
                    fi

                else
                    partition="short"
                    # time="0:45:00"
                    time="0:30:00"
                fi

                # Replace the following line with the command you want to run
                sbatch --partition="$partition" --time="$time" --export=experiment="$experiment",metric="$metric",age_cutoff="$age_cutoff",region_index="$region_index" sh_ml_experiments.sh
                # sbatch --export=experiment="$experiment",metric="$metric" sh_ml_experiments.sh
            done
        done
    done
done
