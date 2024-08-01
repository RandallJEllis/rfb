#!/bin/bash

# Define the strings for experiment and metric
modalities=("cognitive_tests" "neuroimaging" "proteomics")
# modalities=("proteomics")

experiments=("age_only" "all_demographics" "modality_only" "demographics_and_modality")
# metrics=("roc_auc" "f3" "ap")
metrics=("log_loss")

age_cutoffs=(0 65)

# Nested loops to iterate over the strings
for modality in "${modalities[@]}"; do
    for experiment in "${experiments[@]}"; do
        for metric in "${metrics[@]}"; do
            for age_cutoff in "${age_cutoffs[@]}"; do
                echo "Running script with modality: $modality, experiment: $experiment, and metric: $metric"

                # Set the partition and time based on the modality and experiment
                if [[ "$modality" == "neuroimaging" ]]; then
                    job_name=neuroimaging
                    partition="short"
                    mem="64G"
                    if [[ $experiment == "modality_only" || $experiment == "demographics_and_modality" ]]; then
                        time="6:00:00"
                    else
                        time="0:45:00"
                    fi
                elif [[ "$modality" == "proteomics" ]]; then
                    job_name=proteomics
                    mem="24G"
                    if [[ $experiment == "modality_only" || $experiment == "demographics_and_modality" ]]; then
                        partition="short"
                        time="6:00:00"
                    else
                        partition="short"
                        time="0:45:00"
                    fi
                elif [[ "$modality" == "cognitive_tests" ]]; then
                    job_name=cognitive_tests
                    mem="4G"
                    partition="short"
                    if [[ $experiment == "modality_only" || $experiment == "demographics_and_modality" ]]; then
                        time="6:00:00"
                    else
                        time="0:45:00"
                    fi
                fi

                # Replace the following line with the command you want to run
                # sbatch --partition="$partition" --time="$time" --export=experiment="$experiment",metric="$metric" sh_ml_experiments.sh
                sbatch --job-name="$job_name" --partition="$partition" --time="$time" --mem="$mem" --export=modality="$modality",experiment="$experiment",metric="$metric",age_cutoff="$age_cutoff" sh_ml_experiments.sh
                # sbatch --export=experiment="$experiment",metric="$metric" sh_ml_experiments.sh
            done
        done
    done
done
