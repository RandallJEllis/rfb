#!/bin/bash

modality=$1
# Define the strings for experiment and metric
experiments=("modality_only")
# "age_only" "age_sex_lancet2024" "all_demographics" "demographics_and_lancet2024"  "demographics_and_modality" "demographics_modality_lancet2024")
# metrics=("roc_auc" "f3" "ap")
metrics=("log_loss")
# ("lrl1" "lgbm")
models=("lrl1")
age_cutoffs=(0)
#  65)

# Nested loops to iterate over the strings
for experiment in "${experiments[@]}"; do
    for metric in "${metrics[@]}"; do
        for model in "${models[@]}"; do
            for age_cutoff in "${age_cutoffs[@]}"; do
                for region_index in {0..0}; do
                    if [[ $experiment == "modality_only" || $experiment == "demographics_and_modality" || $experiment == "demographics_modality_lancet2024" ]]; then
                   
                        # Set the partition and time based on the modality and experiment
                        if [[ $modality == 'proteomics' ]]; then
                            mem="4G"
                            partition="short"

                            if [[ $age_cutoff -eq 0 ]]; then
                                mem="16G"                                
                                if [[ "$model" == "lgbm" ]]; then
                                    time="3:30:00"
                                else
                                    partition="medium"
                                    time="14:00:00"
                                fi
                                
                            elif [[ $age_cutoff -eq 65 ]]; then
                                
                                if [[ "$model" == "lgbm" ]]; then   
                                    time="1:45:00"
                                else
                                    time="4:30:00"                            
                                fi
                            fi 

                        elif [[ $modality == 'neuroimaging' ]]; then
                            mem="8G"
                            partition="short"

                            if [[ $age_cutoff -eq 0 ]]; then
                                mem="16G"                                
                                if [[ "$model" == "lgbm" ]]; then
                                    time="5:00:00"
                                else
                                    partition="medium"
                                    time="18:00:00"
                                fi
                                
                            elif [[ $age_cutoff -eq 65 ]]; then
                                
                                if [[ "$model" == "lgbm" ]]; then   
                                    time="2:30:00"
                                else
                                    time="7:00:00"                            
                                fi
                            fi 
                           
                        elif [[ $modality == 'cognitive_tests' ]]; then
                            mem="2G"
                            partition="short"

                            if [[ $age_cutoff -eq 0 ]]; then
                                mem="4G"    
                                if [[ "$model" == "lgbm" ]]; then
                                    time="0:30:00"
                                else
                                    time="2:00:00"
                                fi
                            elif [[ $age_cutoff -eq 65 ]]; then
                                if [[ "$model" == "lgbm" ]]; then
                                    time="0:15:00"
                                else
                                    time="1:00:00"
                                fi
                            fi
                        fi
                    
                    else
                        partition="short"
                        time="0:12:00"
                        mem="4G"
                    fi 

                    # current_time=$(date +"%Y-%m-%d_%H-%M-%S")
                    job_name="${experiment}_${model}_AgeCutoff${age_cutoff}_Region${region_index}"
                    output_path="${modality}/${job_name}"
                    error_path="${modality}/${job_name}"

                    echo "Running script on modality $modality, partition $partition, time limit $time. Experiment: $experiment, metric: $metric, model: $model, age cutoff: $age_cutoff, and region index: $region_index"
                    echo "Output path: $output_path"
                    echo "Error path: $error_path"
                    sbatch --job-name="$job_name"_%J \
                           --output="$output_path"_%J.out \
                           --error="$error_path"_%J.err \
                           --partition="$partition" \
                           --time="$time" \
                           --mem="$mem" \
                           --export=modality="$modality",experiment="$experiment",metric="$metric",model="$model",age_cutoff="$age_cutoff",region_index="$region_index"\
                        sh_ml_experiments.sh
                    # sbatch --export=experiment="$experiment",metric="$metric" sh_ml_experiments.sh
                done
            done
        done
    done
done
