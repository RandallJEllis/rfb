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
from flaml.automl.data import get_output_from_log

import sys
from datetime import datetime
import pickle
import matplotlib.pyplot as plt

'''
IDP experiments for:
Age alone
All demographics
All idps
All demographics + all idps

Input arguments:
experiment - age_only, all_demographics, idps_only, demographics_and_idps
time_budget - number of seconds for AutoML training
metric - roc_auc or f3
file_suffix - other information to add to the output directory path so that things don't get overwritten
'''

data_modality = 'neuroimaging'
data_instance = 2

def main():
    """
    Runs the main function which performs feature selection experiments on the given dataset.
    
    This function reads the dataset from the specified parquet and numpy files, and loads the region indices from a pickle file.
    It then creates an argument parser to parse command line arguments.
    The arguments that can be passed are:
    - experiment: options are 'proteins_only' and 'demographics_and_proteins'
    - time_budget: the number of seconds to allow FLAML to optimize (default is 3600)
    - metric: the metric to use for evaluation ('roc_auc', 'f3', or 'ap') (default is 'roc_auc')
    - age_cutoff: the age cutoff to use (default is None)
    
    The function then checks if an age cutoff is provided and subsets the dataset accordingly.
    It also checks if the experiment is 'proteins_only' or 'demographics_and_proteins' and subsets the dataset accordingly.
    """

    X = pd.read_parquet(f'../../tidy_data/dementia/{data_modality}/X.parquet')
    X = X.iloc[:,1:]
    y = np.load(f'../../tidy_data/dementia/{data_modality}/y.npy')

    # Create the argument parser
    parser = argparse.ArgumentParser(
        description='Run AutoML on chosen feature sets')

    # Add arguments
    # parser.add_argument('njobs', type=str, help='Number of cores')
    parser.add_argument('--experiment', type=str,
                        help='options: age_only, all_demographics, idps_only, demographics_and_idps')
    parser.add_argument('--time_budget', type=int, default=3600,
                        help='options: seconds to allow FLAML to optimize')
    parser.add_argument('--metric', type=str, default='roc_auc',
                        help='metric to optimize during training')
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
    original_results_directory_path = f'../../results/dementia/{data_modality}/{experiment}/{metric}/'
    directory_path = f'../../results/dementia/{data_modality}/{experiment}/feature_selection/{metric}/'

    if age_cutoff is not None:
        over_age_idx = X[f'21003-{data_instance}.0'] >= age_cutoff
        X = X[over_age_idx].reset_index(drop=True)
        y = y[over_age_idx]
        
        # cv indices based on region
        # region_lookup = pd.read_csv('../../metadata/coding10.tsv', sep='\t')
        # region_indices = ukb_utils.group_assessment_center(X, region_lookup)

        original_results_directory_path = f'{original_results_directory_path}/agecutoff_{age_cutoff}'
        directory_path = f'{directory_path}/agecutoff_{age_cutoff}'

    # check if output folder exists
    utils.check_folder_existence(directory_path)

    lancet_vars = ['4700-0.0', '5901-0.0', '30780-0.0', 'head_injury', '22038-0.0', '20161-0.0', 'alcohol_consumption', 'hypertension', 'obesity', 
                    'diabetes', 'hearing_loss', 'depression', 'freq_friends_family_visit', '24012-0.0', '24018-0.0', '24019-0.0', '24006-0.0', 
                    '24015-0.0', '24011-0.0', '2020-0.0_-3.0', '2020-0.0_-1.0',
                    '2020-0.0_0.0', '2020-0.0_1.0', '2020-0.0_nan']
    
    idp_vars = pickle.load(open('../../tidy_data/dementia/neuroimaging/idp_variables.pkl', 'rb'))
    if experiment == 'modality_only':
        X = X.loc[:, idp_vars]
        time_budget = 9500
    elif experiment == 'demographics_and_modality':
        X = X.loc[:, df_utils.pull_columns_by_prefix(X, [f'21003-{data_instance}.0', '31-0.0', 'apoe', 'max_educ_complete', '845-0.0', '21000-0.0']).columns.tolist() + idp_vars]
        time_budget = 14000
    elif experiment == 'demographics_modality_lancet2024':
        X = X.loc[:, df_utils.pull_columns_by_prefix(X, [f'21003-{data_instance}.0', '31-0.0', 'apoe', 'max_educ_complete',\
                                                 '845-0.0', '21000-0.0']).columns.tolist() + lancet_vars + idp_vars]
        time_budget = 14500

    if age_cutoff == 65:
        print('Modifying time budget by dividing by 2 for age cutoff of 65') 
        time_budget = time_budget/2

    print(f'Running {experiment} experiment, autoML time budget of {time_budget} seconds, {metric} as the metric, and an age cutoff of {age_cutoff} years')

    if metric == 'f3':
        metric = f3.f3_metric


    print(f'Dimensionality of the dataset: {X.shape}')

    skf = StratifiedKFold(n_splits=10)

    
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        if i != region_index:
            continue
        print(f"Fold {i}:")
    
    # for i,r in enumerate(list(region_indices.keys())):
        current_time = datetime.now().time()
        print(f'{current_time}')
        # print(f'{r}, {current_time}')
        # indices = region_indices[r]
        X_test = X.iloc[test_index, :]
        y_test = y[test_index]

        # Create a mask to select all indices not in indices
        # mask = np.ones(len(y), dtype=bool)
        # mask[indices] = False

        # Step 4: Subset the main array using the mask
        X_train = X.iloc[train_index, :]
        y_train = y[train_index]
        print('made train and test split')

        automl = AutoML()
        # automl.fit(X_train, y_train, task="classification", time_budget=time_budget, metric=metric, n_jobs=-1,
        #                 max_iter=None, early_stop=True, append_log=True, log_file_name=f'{directory_path}/results_log_{i}.json')
        automl.retrain_from_log(log_file_name=f'{original_results_directory_path}/results_log_{i}.json',
                                X_train=X_train, y_train=y_train, task='classification', 
                                train_full=True, n_jobs=-1,
                                #eval_method='cv', n_splits=10, split_type='stratified',
                                train_best=True)

        time_history, best_valid_loss_history, valid_loss_history, config_history, metric_history = get_output_from_log(filename=f'{original_results_directory_path}/results_log_{i}.json', time_budget=time_budget)

        print('Done fitting model')

        train_labels_l = []
        train_probas_l = []

        test_labels_l = []
        test_probas_l = []

        train_res_l = []
        test_res_l = []

        if len(automl.feature_importances_) == 1:
            top_feature_names = np.array(automl.feature_names_in_)[np.argsort(abs(automl.feature_importances_[0]))[::-1]][:100]
        else:
            top_feature_names = np.array(automl.feature_names_in_)[np.argsort(abs(automl.feature_importances_))[::-1]][:100]

        tflist = []
        for j, tf in enumerate(top_feature_names):
            tflist.append(tf)
            current_time = datetime.now().time()
            print(f'Running top {j+1} features: {tflist}, {current_time}')

            X_train_sub = X_train.loc[:, tflist]
            X_test_sub = X_test.loc[:, tflist]

            automl = AutoML()
            automl.retrain_from_log(log_file_name=f'{original_results_directory_path}/results_log_{i}.json',
                                    X_train=X_train_sub, y_train=y_train, task='classification', 
                                    train_full=True, n_jobs=-1,
                                    #eval_method='cv', n_splits=10, split_type='stratified',
                                    train_best=True)

            current_time = datetime.now().time()
            print(f'Done fitting model for region {i+1} with top {j+1} variables. {current_time}')
        
            series_automl = pd.Series([config_history[-1]['Best Learner'], config_history[-1]['Best Hyper-parameters'], i, tflist], index=['model', 'hyperparams', 'region', 'features'])

            train_probas = automl.predict_proba(X_train_sub)[:,1]

            # if metric == 'roc_auc':
            train_res, threshold = ml_utils.calc_results(y_train, train_probas, beta=1)
            # elif metric == f3.f3_metric or metric == 'ap':
            #     train_res, threshold = ml_utils.calc_results(y_train, train_probas, beta=3)

            train_res = pd.concat([series_automl, train_res])
            train_res_l.append(train_res)
            
            if j == 0:
                train_labels_l.append(y_train)
            train_probas_l.append(train_probas)

            test_probas = automl.predict_proba(X_test_sub)[:,1]

            # if metric == 'roc_auc':
            test_res = ml_utils.calc_results(y_test, test_probas, beta=1, threshold=threshold)
                # test_res = ml_utils.calc_results(y_test, test_probas, threshold=threshold)
            # elif metric == f3.f3_metric or metric == 'ap':
            #     test_res = ml_utils.calc_results(y_test, test_probas, threshold=threshold, beta=3)

            test_res = pd.concat([series_automl, test_res])
            test_res_l.append(test_res)
            
            if j == 0:
                test_labels_l.append(y_test)
            test_probas_l.append(test_probas)
        

        ml_utils.save_labels_probas(directory_path, train_labels_l, train_probas_l, test_labels_l, test_probas_l, other_file_info=f'_region_{i}')

        train_df = pd.concat(train_res_l, axis=1).T
        train_df.to_csv(f'{directory_path}/training_results_region_{i}.csv')

        test_df = pd.concat(test_res_l, axis=1).T
        test_df.to_csv(f'{directory_path}/test_results_region_{i}.csv')

        break

        # plot_title = {'idps_only': 'FS IDPs', 'demographics_and_idps': 'FS Demographics + IDPs'}
        # fig = plot_results.mean_roc_curve(true_labels_list=test_labels_l, predicted_probs_list=test_probas_l,
        #                         individual_label='Top features:', title=plot_title[experiment])
        # fig.savefig(f'{directory_path}/roc_curve_region_{i}.pdf')

if __name__ == "__main__":
    main()
