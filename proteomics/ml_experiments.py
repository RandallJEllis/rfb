import sys
sys.path.append('../ukb_func')
import icd
import ml_utils
import df_utils
import ukb_utils
import utils
import f3
import plot_results

import pickle
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, cross_validate, cross_val_score
from sklearn.metrics import confusion_matrix, roc_auc_score, fbeta_score, RocCurveDisplay, auc, roc_curve, precision_recall_fscore_support, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix, roc_auc_score, precision_recall_fscore_support, accuracy_score, balanced_accuracy_score
from lightgbm import LGBMClassifier
from flaml import AutoML

import sys
from datetime import datetime
import pickle
import matplotlib.pyplot as plt

'''
Proteomics experiments for:
Age alone
All demographics
All proteins
All demographics + all proteins

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
    y = np.load(f'../../tidy_data/dementia/{data_modality}/y.npy')
    region_indices = pickle.load(open(f'../../tidy_data/dementia/{data_modality}/region_cv_indices.pickle', 'rb'))

    # Create the argument parser
    parser = argparse.ArgumentParser(
        description='Run AutoML on chosen feature sets')

    # Add arguments
    # parser.add_argument('njobs', type=str, help='Number of cores')
    parser.add_argument('--experiment', type=str,
                        help='options: age_only, all_demographics, proteins_only, demographics_and_proteins')
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
        region_indices = ukb_utils.group_assessment_center(X, data_instance, region_lookup)

        directory_path = f'{directory_path}/agecutoff_{age_cutoff}'

    # check if output folder exists
    utils.check_folder_existence(directory_path)

    if experiment == 'age_only':
        X = X.loc[:, [f'21003-{data_instance}.0']]
        time_budget = 400
    elif experiment == 'all_demographics':
        X = X.loc[:, df_utils.pull_columns_by_prefix(X, [f'21003-{data_instance}.0', '31-0.0', 'apoe', 'max_educ_complete', '845-0.0', '21000-0.0']).columns.tolist()]
        time_budget = 500
    elif experiment == 'proteins_only':
        X = X.loc[:, df_utils.pull_columns_by_suffix(X, ['-0']).columns.tolist()]
        time_budget = 32000
    elif experiment == 'demographics_and_proteins':
        X = X.loc[:, df_utils.pull_columns_by_prefix(X, [f'21003-{data_instance}.0', '31-0.0', 'apoe', 'max_educ_complete', '845-0.0', '21000-0.0']).columns.tolist() + df_utils.pull_columns_by_suffix(X, ['-0']).columns.tolist()]
        time_budget = 32000

    if age_cutoff == 65:
        print('Modifying time budget by dividing by 3 for age cutoff of 65') 
        time_budget = time_budget/3

    print(f'Running {experiment} experiment, autoML time budget of {time_budget} seconds, {metric} as the metric, and an age cutoff of {age_cutoff} years')

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
    automl.fit(X_train, y_train, task="classification", time_budget=time_budget, metric=metric, n_jobs=-1,
                    max_iter=None, early_stop=True, append_log=True, log_file_name=f'{directory_path}/results_log_{i}.json')

    print('Done fitting model')

    series_automl = pd.Series([automl.best_estimator, automl.best_config], index=['model', 'hyperparams'])

    train_probas = automl.predict_proba(X_train)[:,1]

    # if metric == 'roc_auc':
    train_res, threshold = ml_utils.calc_results(metric, y_train, train_probas, beta=3)
    # elif metric == f3.f3_metric:
    #     train_res, threshold = ml_utils.calc_results(y_train, train_probas, beta=3)

    train_res = pd.concat([series_automl, train_res])
    train_res_l.append(train_res)
    train_labels_l.append(y_train)
    train_probas_l.append(train_probas)

    test_probas = automl.predict_proba(X_test)[:,1]

    # if metric == 'roc_auc':
    test_res = ml_utils.calc_results(metric, y_test, test_probas, beta=3, threshold=threshold)
    # elif metric == f3.f3_metric:
    #     test_res = ml_utils.calc_results(y_test, test_probas, threshold=threshold, beta=3)

    test_res = pd.concat([series_automl, test_res])
    test_res_l.append(test_res)
    test_labels_l.append(y_test)
    test_probas_l.append(test_probas)
        

    ml_utils.save_labels_probas(directory_path, train_labels_l, train_probas_l, test_labels_l, test_probas_l, other_file_info=f'_region_{i}')

    train_df = pd.concat(train_res_l, axis=1).T
    train_df.to_csv(f'{directory_path}/training_results_region_{i}.csv')

    test_df = pd.concat(test_res_l, axis=1).T
    test_df.to_csv(f'{directory_path}/test_results_region_{i}.csv')


    # plot_title = {'age_only': 'Age Only', 'all_demographics': 'All Demographics',
    #                 'proteins_only': 'All Proteins', 'demographics_and_proteins': 'All Demographics + Proteins'}
    # fig = plot_results.mean_roc_curve(true_labels_list=test_labels_l, predicted_probs_list=test_probas_l,
    #                         individual_label='Region fold', title=plot_title[experiment])
    # fig.savefig(f'{directory_path}/roc_curve.pdf')

if __name__ == "__main__":
    main()
