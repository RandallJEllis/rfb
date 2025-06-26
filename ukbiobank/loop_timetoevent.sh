#!/bin/bash

modality=$1
predict_alzheimers_only=$2

# Define the strings for experiment and metric
experiments=("modality_only" "demographics_and_modality" "demographics_modality_lancet2024")

# "age_only" "age_sex_lancet2024" "all_demographics" "demographics_and_lancet2024"
# "modality_only" "demographics_and_modality" "demographics_modality_lancet2024"
# "fs_modality_only" "fs_demographics_and_modality" "fs_demographics_modality_lancet2024"

# metrics=("roc_auc" "f3" "ap")
metrics=("log_loss")
# ("lrl1" "lgbm")
models=("rsf")
age_cutoffs=(0 65)
# age_cutoffs=(0 65)

# Nested loops to iterate over the strings
for experiment in "${experiments[@]}"; do
    for metric in "${metrics[@]}"; do
        for model in "${models[@]}"; do
            for age_cutoff in "${age_cutoffs[@]}"; do
                for region_index in {0..9}; do
                    if [[ $experiment == *"modality"* ]]; then  
                        # if [[ $experiment == "modality_only" || $experiment == "demographics_and_modality" || $experiment == "demographics_modality_lancet2024" ]]; then

                        if [[ ! $experiment == *"fs_"* ]]; then
                   
                            # Set the partition and time based on the modality and experiment
                            if [[ $modality == 'proteomics' ]]; then
                                mem="20G"
                                partition="short"
                                time="4:00:00"

                                # if [[ $age_cutoff -eq 0 ]]; then
                                #     mem="16G"                                
                                    # if [[ "$model" == "lgbm" ]]; then
                                    # time="3:30:00"
                                    # else
                                        # partition="medium"
                                        # time="14:00:00"
                                    # fi
                                    
                                if [[ $age_cutoff -eq 65 ]]; then
                                    time="0:30:00"
                                #     if [[ "$model" == "lgbm" ]]; then   
                                #         time="1:45:00"
                                #     else
                                #         time="4:30:00"                            
                                #     fi
                                fi 

                            elif [[ $modality == 'neuroimaging' ]]; then
                                mem="20G"
                                partition="short"
                                time="5:00:00"

                                # if [[ $age_cutoff -eq 0 ]]; then
                                #     mem="20G"                                
                                #     if [[ "$model" == "lgbm" ]]; then
                                #         time="5:00:00"
                                #     else
                                #         time="12:00:00"
                                #     fi
                                    
                                if [[ $age_cutoff -eq 65 ]]; then
                                    time="1:00:00"
                                    # if [[ "$model" == "lgbm" ]]; then   
                                    #     time="2:30:00"
                                    # else
                                    #     time="7:00:00"                            
                                    # fi
                                fi 
                            
                            elif [[ $modality == 'cognitive_tests' ]]; then
                                mem="20G"
                                partition="short"
                                time="5:00:00"

                                # if [[ $age_cutoff -eq 0 ]]; then
                                #     mem="4G"    
                                #     if [[ "$model" == "lgbm" ]]; then
                                #         time="0:30:00"
                                #     else
                                #         time="2:00:00"
                                #     fi
                                if [[ $age_cutoff -eq 65 ]]; then
                                    time="1:00:00"
                                    # if [[ "$model" == "lgbm" ]]; then
                                    #     time="0:15:00"
                                    # else
                                    #     time="1:00:00"
                                    # fi
                                fi
                            fi

                        else
                            partition="short"
                            time="0:30:00"
                            mem="16G"
                        fi 
                    
                    # separate conditions for cognitive tests when experiment is not age only because it has 4x samples of proteomics and neuroimaging
                    # elif [[ $modality == 'cognitive_tests' && $experiment != "age_only" ]]; then
                    #     partition="short"
                    #     time="1:30:00"
                    #     mem="4G"
                    
                    
                    # elif [[ $experiment == "demographics_and_lancet2024" ]]; then
                    #     partition="short"
                    #     time="0:10:00"

                    #     # separate conditions for neuroimaging when experiment is demographics_and_lancet2024
                    #     if [[ $modality == 'neuroimaging' ]]; then
                    #         mem="28G"
                    #     elif [[ $modality == 'cognitive_tests' ]]; then
                    #         mem="16G"
                    #     else
                    #         mem="20G"
                    #     fi

                    else
                        partition="short"
                        time="0:15:00"
                        mem="20G"
                    fi 

                    # current_time=$(date +"%Y-%m-%d_%H-%M-%S")
                    job_name="${experiment}_${model}_AgeCutoff${age_cutoff}_Region${region_index}"
                    output_dir="${modality}/"
                    error_dir="${modality}/"

                    echo "Running script on modality $modality,\
                    partition $partition, time limit $time.\
                    Experiment: $experiment,\
                    metric: $metric,\
                    model: $model,\
                    age cutoff: $age_cutoff, and region index: $region_index.\
                    Predict Alzheimer's only: $predict_alzheimers_only"

                    sbatch --job-name="$job_name"_%J \
                           --output="$output_dir"job_%J_$job_name.out \
                           --error="$error_dir"job_%J_$job_name.err \
                           --partition="$partition" \
                           --time="$time" \
                           --mem="$mem" \
                           --export=modality="$modality",experiment="$experiment",metric="$metric",model="$model",age_cutoff="$age_cutoff",region_index="$region_index",predict_alzheimers_only="$predict_alzheimers_only" \
                        sh_timetoevent_experiments.sh
                    # sbatch --export=experiment="$experiment",metric="$metric" sh_ml_experiments.sh
                done
            done
        done
    done
done
