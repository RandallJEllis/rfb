import sys
sys.path.append('../ukb_func')
import ml_utils
import df_utils
import ukb_utils
import utils
import f3

import pickle
import argparse
import os
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from flaml import AutoML

import sys
from datetime import datetime
import pickle

'''
Proteomics experiments for:
Age alone
All demographics
Age + Lancet 2024
All demographics + Lancet 2024
All proteins
All demographics + all proteins
All demographics + Lancet 2024 + all proteins

Input arguments:
experiment - age_only, all_demographics, proteins_only, demographics_and_proteins
time_budget - number of seconds for AutoML training
metric - roc_auc or f3
file_suffix - other information to add to the output directory path so that things don't get overwritten
'''

data_modality = 'proteomics'
data_instance = 0

def main():
    X = pd.read_parquet(f'../../tidy_data/dementia/{data_modality}/X.parquet')
    X = X.iloc[:,1:]
    y = np.load(f'../../tidy_data/dementia/{data_modality}/y.npy')
    region_indices = pickle.load(open(f'../../tidy_data/dementia/{data_modality}/region_cv_indices.pickle', 'rb'))

    # Create the argument parser
    parser = argparse.ArgumentParser(
        description='Run AutoML on chosen feature sets')

    # Add arguments
    # parser.add_argument('njobs', type=str, help='Number of cores')
    parser.add_argument('--experiment', type=str,
                        help='options: age_only, all_demographics, lancet2024, proteins_only, demographics_and_proteins')
    parser.add_argument('--time_budget', type=int, default=3600,
                        help='options: seconds to allow FLAML to optimize')
    parser.add_argument('--metric', type=str, default='roc_auc',
                        help='options: roc_auc, f3, ap')
    parser.add_argument('--age_cutoff', type=int, default=None,
                        help='age cutoff')
    parser.add_argument('--region_index', type=int, default=None,
                        help='region index')
    # parser.add_argument('--file_suffix', type=str, default='',
    #                     help='extra information to put in the filename')

    # Parse the arguments
    args = parser.parse_args()
    experiment = args.experiment
    time_budget = args.time_budget
    metric = args.metric
    age_cutoff = args.age_cutoff
    if age_cutoff == 0:
        age_cutoff = None
    # file_suffix = args.file_suffix
    region_index = args.region_index

    if region_index == None:
        print('NEED REGION INDEX')
        sys.exit()

    # Specify the directory path
    directory_path = f'../../results/dementia/{data_modality}/{experiment}/{metric}/'
 

    if age_cutoff is not None:
        over_age_idx = X[f'21003-{data_instance}.0'] >= age_cutoff
        X = X[over_age_idx].reset_index(drop=True)
        y = y[over_age_idx]
        
        # cv indices based on region
        region_lookup = pd.read_csv('../../metadata/coding10.tsv', sep='\t')
        region_indices = ukb_utils.group_assessment_center(
            X, data_instance, region_lookup)

        directory_path = f'{directory_path}/agecutoff_{age_cutoff}'

    # check if output folder exists
    utils.check_folder_existence(directory_path)

    lancet_vars = ['4700-0.0', '5901-0.0', '30780-0.0', 'head_injury', '22038-0.0', '20161-0.0', 'alcohol_consumption', 'hypertension', 'obesity', 
                    'diabetes', 'hearing_loss', 'depression', 'freq_friends_family_visit', '24012-0.0', '24018-0.0', '24019-0.0', '24006-0.0', 
                    '24015-0.0', '24011-0.0', '2020-0.0_-3.0', '2020-0.0_-1.0',
                    '2020-0.0_0.0', '2020-0.0_1.0', '2020-0.0_nan']

    if experiment == 'age_only':
        X = X.loc[:, [f'21003-{data_instance}.0']]
        time_budget = 10
    elif experiment == 'all_demographics':
        X = X.loc[:, df_utils.pull_columns_by_prefix(X, [f'21003-{data_instance}.0', '31-0.0', 'apoe', 'max_educ_complete', '845-0.0', '21000-0.0']).columns.tolist()]
        time_budget = 25
    elif experiment == 'age_and_lancet2024':
        X = X.loc[:, df_utils.pull_columns_by_prefix(X, [f'21003-{data_instance}.0', 'max_educ_complete', '845-0.0']).columns.tolist() + lancet_vars]
        time_budget = 50
    elif experiment == 'age_sex_lancet2024':
        X = X.loc[:, df_utils.pull_columns_by_prefix(X, [f'21003-{data_instance}.0', '31-0.0', 'max_educ_complete', '845-0.0', '21000-0.0']).columns.tolist() + \
            lancet_vars]
        time_budget = 75
    elif experiment == 'demographics_and_lancet2024':
        X = X.loc[:, df_utils.pull_columns_by_prefix(X, [f'21003-{data_instance}.0', '31-0.0', 'apoe', 'max_educ_complete', '845-0.0', '21000-0.0']).columns.tolist() + \
            lancet_vars]
        time_budget = 100
    elif experiment == 'modality_only':
        X = X.loc[:, df_utils.pull_columns_by_suffix(X, ['-0']).columns.tolist()]
        time_budget = 8500
    elif experiment == 'demographics_and_modality':
        X = X.loc[:, df_utils.pull_columns_by_prefix(X, [f'21003-{data_instance}.0', '31-0.0', 'apoe', 'max_educ_complete',\
                                                 '845-0.0', '21000-0.0']).columns.tolist() + df_utils.pull_columns_by_suffix(X, ['-0']).columns.tolist()]
        time_budget = 9000
    elif experiment == 'demographics_modality_lancet2024':
        X = X.loc[:, df_utils.pull_columns_by_prefix(X, [f'21003-{data_instance}.0', '31-0.0', 'apoe', 'max_educ_complete',\
                                                 '845-0.0', '21000-0.0']).columns.tolist() + lancet_vars + df_utils.pull_columns_by_suffix(X, ['-0']).columns.tolist()]
        time_budget = 9500

    if age_cutoff == 65:
        print('Modifying time budget by dividing by 2 for age cutoff of 65') 
        time_budget = time_budget/2

    print(f'Running {experiment} experiment, region {region_index}, autoML time budget of {time_budget} seconds, {metric} as the metric, and an age cutoff of {age_cutoff} years')

    if metric == 'roc_auc':
        pass
    elif metric == 'f3':
        metric = f3.f3_metric

    train_labels_l = []
    train_probas_l = []

    test_labels_l = []
    test_probas_l = []

    train_res_l = []
    test_res_l = []

    print(f'Dimensionality of the dataset: {X.shape}')

    region_list = list(region_indices.keys())
    # del region_list[1]
    
    i = region_index
    r = region_list[i]
    print(f'Starting region {i+1} of {len(region_list)}: {r}')


    # for i,r in enumerate(list(region_indices.keys())):
    current_time = datetime.now().time()
    print(f'{r}, {current_time}')
    indices = region_indices[r]
    X_test = X.iloc[indices, :]
    y_test = y[indices]

    # Create a mask to select all indices not in indices
    mask = np.ones(len(y), dtype=bool)
    mask[indices] = False

    # Step 4: Subset the main array using the mask
    X_train = X.iloc[mask, :]
    y_train = y[mask]
    print('made train and test split')

    automl = AutoML()
    # automl.fit(X_train, y_train, task="classification", time_budget=time_budget, metric=metric, n_jobs=-1, eval_method='cv', n_splits=10,
    #                 max_iter=None, early_stop=True, append_log=True, log_file_name=f'{directory_path}/results_log_{i}.json')
    
    automl.fit(X_train, y_train, task="classification", time_budget=time_budget, metric=metric,
                n_jobs=-1, eval_method='cv', n_splits=10, split_type='stratified',
                log_training_metric=True, early_stop=True,
                seed=239875, model_history=True, estimator_list=['lgbm'],
                log_file_name=f'{directory_path}/results_log_{i}.json')

    print('Done fitting model')

    if len(automl.feature_importances_) == 1:
        feature_names = np.array(automl.feature_names_in_)[np.argsort(abs(automl.feature_importances_[0]))[::-1]]
        fi = automl.feature_importances_[0][np.argsort(abs(automl.feature_importances_[0]))[::-1]]
    else:
        feature_names = np.array(automl.feature_names_in_)[np.argsort(abs(automl.feature_importances_))[::-1]]
        fi = automl.feature_importances_[np.argsort(abs(automl.feature_importances_))[::-1]]
        
    fi_df = pd.DataFrame({'feature': feature_names, 'importance': fi})
    fi_df.to_parquet(f'{directory_path}/feature_importance_region_{i}.parquet')

    series_automl = pd.Series([i, r, automl.best_estimator, automl.best_config], index=['region_index', 'region', 'model', 'hyperparams'])

    train_probas = automl.predict_proba(X_train)[:,1]

    # if metric == 'roc_auc':
    train_res, threshold = ml_utils.calc_results(y_train, train_probas, beta=1)
    # elif metric == f3.f3_metric:
    #     train_res, threshold = ml_utils.calc_results(y_train, train_probas, beta=3)

    train_res = pd.concat([series_automl, train_res])
    # train_res_l.append(train_res)
    train_labels_l.append(y_train)
    train_probas_l.append(train_probas)

    test_probas = automl.predict_proba(X_test)[:,1]

    # if metric == 'roc_auc':
    test_res = ml_utils.calc_results(y_test, test_probas, beta=1, threshold=threshold)
    # elif metric == f3.f3_metric:
    #     test_res = ml_utils.calc_results(y_test, test_probas, threshold=threshold, beta=3)

    test_res = pd.concat([series_automl, test_res])
    # test_res_l.append(test_res)
    test_labels_l.append(y_test)
    test_probas_l.append(test_probas)
        

    ml_utils.save_labels_probas(directory_path, train_labels_l, train_probas_l, test_labels_l, test_probas_l, other_file_info=f'_region_{i}')

    # train_df = pd.concat(train_res_l, axis=1).T
    train_df = pd.DataFrame(train_res).T
    # Specify the path to your CSV file
    file_path = f'{directory_path}/training_results.csv'

    if os.path.exists(file_path):
        # If file exists, read it into a DataFrame and append the new data
        existing_df = pd.read_csv(file_path)
        combined_df = pd.concat([existing_df, train_df])
        combined_df.to_csv(file_path, index=False)
    else:
        # If file does not exist, simply write the new data to a CSV
        train_df.to_csv(file_path, index=False)
    # train_df.to_csv(f'{directory_path}/training_results_region_{i}.csv')

    # test_df = pd.concat(test_res_l, axis=1).T
    test_df = pd.DataFrame(test_res).T
    # Specify the path to your CSV file
    file_path = f'{directory_path}/test_results.csv'

    if os.path.exists(file_path):
        # If file exists, read it into a DataFrame and append the new data
        existing_df = pd.read_csv(file_path)
        combined_df = pd.concat([existing_df, test_df])
        combined_df.to_csv(file_path, index=False)
    else:
        # If file does not exist, simply write the new data to a CSV
        test_df.to_csv(file_path, index=False)
    # test_df.to_csv(f'{directory_path}/test_results_region_{i}.csv')


    # plot_title = {'age_only': 'Age Only', 'all_demographics': 'All Demographics',
    #                 'proteins_only': 'All Proteins', 'demographics_and_proteins': 'All Demographics + Proteins'}
    # fig = plot_results.mean_roc_curve(true_labels_list=test_labels_l, predicted_probs_list=test_probas_l,
    #                         individual_label='Region fold', title=plot_title[experiment])
    # fig.savefig(f'{directory_path}/roc_curve.pdf')

if __name__ == "__main__":
    main()
