import sys
sys.path.append('./ukb_func')
from ml_utils import save_labels_probas
import df_utils
import ukb_utils
from utils import save_pickle
import f3
import dementia_utils
from ml_experiments import *

import pickle
import argparse
import os
import pandas as pd
import numpy as np
from flaml import AutoML
from flaml.automl.data import get_output_from_log
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import pickle

from pyarrow.parquet import ParquetFile
import pyarrow as pa 

from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_ipcw, concordance_index_censored, cumulative_dynamic_auc, integrated_brier_score

'''
Time to event experiments for:
Age alone
All demographics
Age + Sex + Lancet 2024
All demographics + Lancet 2024
Modality only
All demographics + modality
All demographics + Lancet 2024 + modality

Input arguments:
modality - proteomics, neuroimaging, cognitive_tests
experiment - age_only, all_demographics, age_sex_lancet, demographics_and_lancet2024, 
            modality_only, demographics_and_modality, demographics_modality_lancet2024,
            fs_modality_only, fs_demographics_and_modality, fs_demographics_modality_lancet2024
time_budget - number of seconds for AutoML training
metric - log_loss, roc_auc, or f3 (log_loss is used throughout the project)
model - lgbm or lrl1
age_cutoff - 0 or 65
region_index - index of the region to run the experiment on (or CV fold for neuroimaging)
'''

def main():

    # Parse the arguments
    data_modality, data_instance, experiment, metric, model, age_cutoff, region_index, alzheimers_only = parse_args()
    print(f'Running {experiment} experiment, modality {data_modality}, instance {data_instance}, region {region_index}, model {model}, {metric} as the metric, and an age cutoff of {age_cutoff} years. Predicting Alzheimer\'s only: {alzheimers_only}')
    
    # Load the datasets
    X, y, region_indices = load_datasets(data_modality, data_instance, alzheimers_only)
    
    data_path = '../../proj_idp/tidy_data/'
    demo = ukb_utils.load_demographics(data_path)
    df = X.merge(demo.loc[:, ['eid', f'53-{data_instance}.0']], on='eid')
    df[f'53-{data_instance}.0'] = pd.to_datetime(df[f'53-{data_instance}.0'])

    acd = pd.read_parquet(data_path + 'acd/allcausedementia.parquet')
    acd = dementia_utils.get_first_diagnosis(acd)

    controls = df.iloc[y == 0]
    cases = df.iloc[y == 1]

    cases = pd.merge(cases, acd.loc[:, ['eid', 'first_dx']], on='eid')

    controls.loc[:, 'first_dx'] = pd.Timestamp('2022-10-31')

    cases['time2event'] = (cases['first_dx'] - cases[f'53-{data_instance}.0']).dt.days
    controls['time2event'] = (controls['first_dx'] - controls[f'53-{data_instance}.0']).dt.days

    X = pd.concat([controls, cases]).reset_index(drop=True)
    X['label'] = [0]*len(controls) + [1]*len(cases)
    y = X.label
    region_indices = update_region_indices(X, data_instance)

    directory_path, original_results_directory_path = get_dir_path(data_modality, experiment, metric, model, alzheimers_only, survival=True)
    
    # subset data by age if there is an age cutoff
    if age_cutoff is not None:
        directory_path, original_results_directory_path,\
            X, y,\
                region_indices = setup_age_cutoff(directory_path, original_results_directory_path, X, y, age_cutoff, data_modality, data_instance)
    
    # check if output folder exists
    utils.check_folder_existence(directory_path)

    # set up experiment variables
    lancet_vars, continuous_lancet_vars = get_lancet_vars()    
    X = subset_experiment_vars(data_modality, data_instance, experiment, X, lancet_vars, survival=True)

    # set time budget based on experiment, data_modality, model, and age_cutoff
    # time_budget = get_time_budget(experiment, data_modality, model, age_cutoff)
    
    # get experiment-specific continuous variables if using lrl1
    # if model == 'lrl1':
    #     print('Scaling data for lrl1 classifier')
    #     continuous_cols = continuous_vars_for_scaling(data_modality, data_instance, experiment, continuous_lancet_vars, X)
    #     print('Done scaling data')
        
    # print experiment details
    # print(f'Running {experiment} experiment, region {region_index}, autoML time budget of {time_budget} seconds, {metric} as the metric, and an age cutoff of {age_cutoff} years')

    # set metric to f3 if using f3
    # if metric == 'f3':
    #     metric = f3.f3_metric

    print(f'Dimensionality of the dataset: {X.shape}')

    # Split the data into training and testing sets
    X_train, y_train, X_test, y_test, region = subset_train_test_data(X, y, data_modality, region_index, region_indices)


    # Define the data types for the structured array
    dtype = [('label', 'bool'), ('time2event', 'int16')]  # U10 for strings of length up to 10 characters

    # Convert the DataFrame to a NumPy structured array
    y_train = np.array(list(X_train.loc[:, ['label', 'time2event']].itertuples(index=False, name=None)), dtype=dtype)
    y_test = np.array(list(X_test.loc[:, ['label', 'time2event']].itertuples(index=False, name=None)), dtype=dtype)

    X_train = X_train.drop(columns=['label', 'time2event'])
    X_test = X_test.drop(columns=['label', 'time2event'])
    
    random_state = 20
    rsf = RandomSurvivalForest(
        n_estimators=10, min_samples_split=10, min_samples_leaf=15, n_jobs=-1, random_state=random_state, max_depth=5
    )
    rsf.fit(X_train, y_train)
    
    times = [x[1] for x in y_test]
    # lower, upper = min(times), max(times)
    lower, upper = np.percentile(times, [5, 95])
    times = np.arange(lower, upper)

    train_pred = rsf.predict(X_train)
    test_pred = rsf.predict(X_test)

    train_prob = np.vstack([fn(times) for fn in rsf.predict_survival_function(X_train)])
    test_prob = np.vstack([fn(times) for fn in rsf.predict_survival_function(X_test)])

    train_int_brier = integrated_brier_score(y_train, y_train, train_prob, times)        
    test_int_brier = integrated_brier_score(y_train, y_test, test_prob, times)

    train_auc = cumulative_dynamic_auc(y_train, y_train, train_pred, times)
    test_auc = cumulative_dynamic_auc(y_train, y_test, test_pred, times)

    train_ci_ipcw = concordance_index_ipcw(y_train, y_train, train_pred, tau=times[-1])
    test_ci_ipcw = concordance_index_ipcw(y_train, y_test, test_pred, tau=times[-1])

    train_ci_harrell = concordance_index_censored([x[0] for x in y_train], [x[1] for x in y_train], train_pred)
    test_ci_harrell = concordance_index_censored([x[0] for x in y_test], [x[1] for x in y_test], test_pred)

    print(f'Train integrated Brier score: {train_int_brier}')
    print(f'Test integrated Brier score: {test_int_brier}')
    print(f'Train C-index IPCW: {train_ci_ipcw}')
    print(f'Test C-index IPCW: {test_ci_ipcw}')
    print(f'Train C-index Harrell: {train_ci_harrell}')
    print(f'Test C-index Harrell: {test_ci_harrell}')
    print(f'Train AUC: {train_auc}')
    print(f'Test AUC: {test_auc}')

    save_labels_probas(directory_path, y_train, train_pred, y_test, test_pred,
                       other_file_info=f'_region_{region_index}', survival=True,
                       surv_model=rsf, train_surv_fn=train_prob, test_surv_fn=test_prob)

    save_pickle(f'{directory_path}/train_int_brier_{region_index}.pkl', train_int_brier)
    save_pickle(f'{directory_path}/test_int_brier_{region_index}.pkl', test_int_brier)
    save_pickle(f'{directory_path}/train_ci_ipcw_{region_index}.pkl', train_ci_ipcw)
    save_pickle(f'{directory_path}/test_ci_ipcw_{region_index}.pkl', test_ci_ipcw)
    save_pickle(f'{directory_path}/train_ci_harrell_{region_index}.pkl', train_ci_harrell)
    save_pickle(f'{directory_path}/test_ci_harrell_{region_index}.pkl', test_ci_harrell)
    save_pickle(f'{directory_path}/train_auc_{region_index}.pkl', train_auc)
    save_pickle(f'{directory_path}/test_auc_{region_index}.pkl', test_auc)
    
    
    # if model == 'lrl1':
    #     X_train, X_test = scale_continuous_vars(X_train, X_test, continuous_cols)
        
    # automl = AutoML()
    # automl_settings = settings_automl(experiment, time_budget, metric, model)
    # print(automl_settings)

    # if 'fs_' in experiment:
        # automl_settings["log_file_name"] = f'{original_results_directory_path}/results_log_{region_index}.json'
        # print(f'Retraining best model to do feature selection: {datetime.now().time()}')
        # automl.retrain_from_log(
                            # X_train=X_train, y_train=y_train, 
                            # **automl_settings
                            # )
        # print(f'Done retraining model: {datetime.now().time()}')
        
        # time_history, best_valid_loss_history,\
        #     valid_loss_history, config_history,\
        #         metric_history = get_output_from_log(filename=f'{original_results_directory_path}/results_log_{region_index}.json',
        #                                                 time_budget=time_budget)

        # train_labels_l, train_probas_l,\
        #     test_labels_l, test_probas_l,\
        #         train_res_l, test_res_l = iterative_fs_inference(automl, automl_settings, config_history,
        #                                                          X_train, y_train, X_test, y_test,
        #                                                          region_index, region)

        # save_fs_results(directory_path, train_labels_l, train_probas_l, test_labels_l, test_probas_l, train_res_l, test_res_l, region_index)
        
    # else:
        # automl_settings["log_file_name"] = f'{directory_path}/results_log_{region_index}.json'
        # print(f'Fitting model: {datetime.now().time()}')
        # automl.fit(X_train, y_train, **automl_settings)
        # print(f'Done fitting model: {datetime.now().time()}')

        # if experiment != 'age_only':
            # Save the feature importance
            # save_feature_importance(automl, directory_path, region_index)

        # Collect the results for train and test
        # save_results(directory_path, automl, X_train, y_train, X_test, y_test, region, region_index)

if __name__ == "__main__":
    main()
