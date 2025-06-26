#!/bin/bash

# Define the strings for experiment and metric
# experiments=("age_only" "age_sex_lancet2024" "all_demographics" "age_and_lancet2024" "demographics_and_lancet2024")
experiments=("demographics_and_modality" "demographics_modality_lancet2024")
# "modality_only" 

# metrics=("roc_auc" "f3" "ap")
metrics=("log_loss")
# ("lrl1" "lgbm")
models=("lrl1")
age_cutoffs=(0 65)

# Nested loops to iterate over the strings
for experiment in "${experiments[@]}"; do
    for metric in "${metrics[@]}"; do
        for model in "${models[@]}"; do
            for age_cutoff in "${age_cutoffs[@]}"; do
                for region_index in {0..0}; do
                   
                    # Set the partition and time based on the experiment
                    if [[ $experiment == "modality_only" || $experiment == "demographics_and_modality" || $experiment == "demographics_modality_lancet2024" ]]; then

                        if [[ $age_cutoff -eq 0 ]]; then
                            mem="16G"
                            partition="short"
                            
                            if [[ "$model" == "lgbm" ]]; then
                                time="3:30:00"
                            else
                                partition="medium"
                                time="14:00:00"
                            fi
                            
                        elif [[ $age_cutoff -eq 65 ]]; then
                            mem="4G"
                            partition="short"

                            if [[ "$model" == "lgbm" ]]; then   
                                time="1:45:00"
                            else
                                time="4:30:00"                            
                            fi
                        fi 

                    else
                        partition="short"
                        time="0:12:00"
                        mem="4G"
                    fi

                    echo "Running script on partition $partition, with time limit $time. Experiment: $experiment, metric: $metric, model: $model, age cutoff: $age_cutoff, and region index: $region_index"
                    sbatch --partition="$partition" --time="$time" --mem="$mem" --export=time="$time",experiment="$experiment",metric="$metric",model="$model",age_cutoff="$age_cutoff",region_index="$region_index" sh_ml_experiments.sh
                    # sbatch --export=experiment="$experiment",metric="$metric" sh_ml_experiments.sh
                done
            done
        done
    done
done
