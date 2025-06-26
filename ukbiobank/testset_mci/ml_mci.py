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


def main():

    # Create the argument parser
    parser = argparse.ArgumentParser(
        description='Run AutoML on chosen feature sets')

    # Add arguments
    # parser.add_argument('njobs', type=str, help='Number of cores')
    parser.add_argument('--modality', type=str,
                        help='options: cognitive_tests, neuroimaging, proteomics')
    parser.add_argument('--experiment', type=str,
                        help='options: age_only, all_demographics, modality_only, demographics_and_modality')
    parser.add_argument('--time_budget', type=int, default=3600,
                        help='options: seconds to allow FLAML to optimize')
    parser.add_argument('--metric', type=str, default='roc_auc',
                        help='options: roc_auc, f3, ap')
    parser.add_argument('--age_cutoff', type=int, default=None,
                        help='age cutoff')
    # parser.add_argument('--region_index', type=int, default=None,
    #                     help='region index')
    # parser.add_argument('--file_suffix', type=str, default='',
    #                     help='extra information to put in the filename')

    # Parse the arguments
    args = parser.parse_args()
    data_modality = args.modality
    experiment = args.experiment
    time_budget = args.time_budget
    metric = args.metric
    age_cutoff = args.age_cutoff
    # file_suffix = args.file_suffix
    # region_index = args.region_index

    # if region_index == None:
    #     print('NEED REGION INDEX')
    #     sys.exit()

    # Specify the directory path
    directory_path = f'../../results/dementia/{data_modality}/{experiment}/mci_test_set/{metric}/'

    # Load datasets
    X = pd.read_parquet(f'../../tidy_data/dementia/{data_modality}/X.parquet')
    y = np.load(f'../../tidy_data/dementia/{data_modality}/y.npy')
    mci = pd.read_parquet('../../tidy_data/dementia/mci/mci_data.parquet')

    if data_modality == 'neuroimaging':
        data_instance = 2
    else:
        data_instance = 0
        region_indices = pickle.load(open(f'../../tidy_data/dementia/{data_modality}/region_cv_indices.pickle', 'rb'))

    if age_cutoff is not None:
        over_age_idx = X[f'21003-{data_instance}.0'] >= age_cutoff
        X = X[over_age_idx].reset_index(drop=True)
        y = y[over_age_idx]
        
        # cv indices based on region (for not neuroimaging because it doesn't use regions)
        if data_modality != 'neuroimaging':
            region_lookup = pd.read_csv('../../metadata/coding10.tsv', sep='\t')
            region_indices = ukb_utils.group_assessment_center(X, data_instance, region_lookup)

        directory_path = f'{directory_path}/agecutoff_{age_cutoff}'

    # check if output folder exists
    utils.check_folder_existence(directory_path)

    if data_modality=='proteomics':
        proteomics_vars = df_utils.pull_columns_by_suffix(X, ['-0']).columns.tolist()
    elif data_modality=='neuroimaging':
        idp_vars = pickle.load(open('../../tidy_data/dementia/neuroimaging/idp_variables.pkl', 'rb'))
    elif data_modality=='cognitive_tests':
        cog_vars = pickle.load(open(f'../../tidy_data/dementia/{data_modality}/cognitive_columns.pkl', 'rb'))

    if experiment == 'age_only':
        X = X.loc[:, ['eid', f'21003-{data_instance}.0']]
        time_budget = 400
    elif experiment == 'all_demographics':
        X = X.loc[:, ['eid'] + df_utils.pull_columns_by_prefix(X, [f'21003-{data_instance}.0', '31-0.0', 'apoe', 'max_educ_complete', '845-0.0', '21000-0.0']).columns.tolist()]
        time_budget = 500
    elif experiment == 'modality_only':
        if data_modality=='proteomics':
            X = X.loc[:, ['eid'] + proteomics_vars]
        elif data_modality=='neuroimaging':
            X = X.loc[:, ['eid'] + idp_vars]
        elif data_modality=='cognitive_tests':
            X = X.loc[:, ['eid'] + cog_vars]
        time_budget = 9500
    elif experiment == 'demographics_and_modality':
        if data_modality=='proteomics':
            X = X.loc[:, ['eid'] + df_utils.pull_columns_by_prefix(X, [f'21003-{data_instance}.0', '31-0.0', 'apoe', 'max_educ_complete', '845-0.0', '21000-0.0']).columns.tolist() + proteomics_vars]
        elif data_modality=='neuroimaging':
            X = X.loc[:, ['eid'] + df_utils.pull_columns_by_prefix(X, [f'21003-{data_instance}.0', '31-0.0', 'apoe', 'max_educ_complete', '845-0.0', '21000-0.0']).columns.tolist() + idp_vars]
        elif data_modality=='cognitive_tests':
            X = X.loc[:, ['eid'] + df_utils.pull_columns_by_prefix(X, [f'21003-{data_instance}.0', '31-0.0', 'apoe', 'max_educ_complete', '845-0.0', '21000-0.0']).columns.tolist() + cog_vars]     
        time_budget = 14000

    if age_cutoff == 65:
        print('Modifying time budget by dividing by 3 for age cutoff of 65') 
        time_budget = time_budget/2

    print(f'Running {experiment} experiment, MCI test set, autoML time budget of {time_budget} seconds, {metric} as the metric, and an age cutoff of {age_cutoff} years')

    # if metric == 'roc_auc':
    #     pass
    # elif metric == 'f3':
    #     metric = f3.f3_metric

    train_labels_l = []
    train_probas_l = []

    test_labels_l = []
    test_probas_l = []

    train_res_l = []
    test_res_l = []

    print(f'Dimensionality of the dataset: {X.shape}')

    X_train = X[~X.eid.isin(mci.eid)]
    X_test = X[X.eid.isin(mci.eid)]
    y_train = y[X_train.index]
    y_test = y[X_test.index]

    X_train = X_train.drop(columns=['eid'])
    X_test = X_test.drop(columns=['eid'])

    current_time = datetime.now().time()
    print(f'{current_time}')
    print('made train and test split')

    automl = AutoML()
    automl.fit(X_train, y_train, task="classification", time_budget=time_budget, metric=metric, 
               n_jobs=-1, eval_method='cv', n_splits=10, split_type='stratified',
               log_training_metric=True, early_stop=True, 
               seed=239875, model_history=True, estimator_list=['lgbm'],
               log_file_name=f'{directory_path}/results_log.json')

    print('Done fitting model')

    series_automl = pd.Series([automl.best_estimator, automl.best_config], index=['model', 'hyperparams'])

    train_probas = automl.predict_proba(X_train)[:,1]

    # train_res, threshold = ml_utils.calc_results(metric, y_train, train_probas, youden=True, beta=1)

    # train_res = pd.concat([series_automl, train_res])
    # train_res_l.append(train_res)
    train_labels_l.append(y_train)
    train_probas_l.append(train_probas)

    test_probas = automl.predict_proba(X_test)[:,1]

    # if metric == 'roc_auc':
    # test_res = ml_utils.calc_results(metric, y_test, test_probas, beta=3, threshold=threshold)
    # elif metric == f3.f3_metric:
    #     test_res = ml_utils.calc_results(y_test, test_probas, threshold=threshold, beta=3)

    # test_res = pd.concat([series_automl, test_res])
    # test_res_l.append(test_res)
    test_labels_l.append(y_test)
    test_probas_l.append(test_probas)
        

    ml_utils.save_labels_probas(directory_path, train_labels_l, train_probas_l, test_labels_l, test_probas_l)#, other_file_info=f'_region_{i}')

    # train_df = pd.concat(train_res_l, axis=1).T
    # train_df.to_csv(f'{directory_path}/training_results.csv')

    # test_df = pd.concat(test_res_l, axis=1).T
    # test_df.to_csv(f'{directory_path}/test_results.csv')


    # plot_title = {'age_only': 'Age Only', 'all_demographics': 'All Demographics',
    #                 'proteins_only': 'All Proteins', 'demographics_and_proteins': 'All Demographics + Proteins'}
    # fig = plot_results.mean_roc_curve(true_labels_list=test_labels_l, predicted_probs_list=test_probas_l,
    #                         individual_label='Region fold', title=plot_title[experiment])
    # fig.savefig(f'{directory_path}/roc_curve.pdf')

if __name__ == "__main__":
    main()
