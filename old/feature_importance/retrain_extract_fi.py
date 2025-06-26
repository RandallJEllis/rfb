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
Input arguments:
data_modality - proteomics, neuroimaging, cogntive_tests
experiment - age_only, all_demographics, modality_only, demographics_and_modality
metric - log_loss, roc_auc, or f3
age_cutoff - 0, 65
region_index - 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
'''

# data_modality = 'cognitive_tests'
# data_instance = 0

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

    # Create the argument parser
    parser = argparse.ArgumentParser(
        description='Run AutoML on chosen feature sets')

    # Add arguments
    # parser.add_argument('njobs', type=str, help='Number of cores')
    parser.add_argument('--modality', type=str,
                        help='options: proteomics, neuroimaging, cognitive_tests')
    parser.add_argument('--experiment', type=str,
                        help='options: cognitive_tests_only, demographics_and_cognitive_tests')
    parser.add_argument('--metric', type=str, default='roc_auc',
                        help='options: roc_auc, f3')
    parser.add_argument('--age_cutoff', type=int, default=None,
                        help='age cutoff')
    parser.add_argument('--region_index', type=int, default=None,
                        help='region index')
    # parser.add_argument('--file_suffix', type=str, default='',
    #                     help='extra information to put in the filename')

    # Parse the arguments
    args = parser.parse_args()
    data_modality = args.modality
    experiment = args.experiment
    metric = args.metric
    age_cutoff = args.age_cutoff
    if age_cutoff == 0:
        age_cutoff = None
    # file_suffix = args.file_suffix
    region_index = args.region_index
    
    X = pd.read_parquet(f'../../tidy_data/dementia/{data_modality}/X.parquet')
    y = np.load(f'../../tidy_data/dementia/{data_modality}/y.npy')

    # Remove EID column
    X = X.iloc[:,1:]
    
    if data_modality == 'neuroimaging':
        data_instance = 2
        skf = StratifiedKFold(n_splits=10)
    else:
        data_instance = 0
        # neuroimaging doesn't use region_indices because it only had 4 regions
        region_indices = pickle.load(open(f'../../tidy_data/dementia/{data_modality}/region_cv_indices.pickle', 'rb'))
        region_list = list(region_indices.keys())
        
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
    
    if data_modality=='proteomics':
        modality_vars = df_utils.pull_columns_by_suffix(X, ['-0']).columns.tolist()
    elif data_modality=='neuroimaging':
        modality_vars = pickle.load(open('../../tidy_data/dementia/neuroimaging/idp_variables.pkl', 'rb'))
    elif data_modality=='cognitive_tests':
        modality_vars = pickle.load(open(f'../../tidy_data/dementia/cognitive_tests/cognitive_columns.pkl', 'rb'))

    if experiment == 'age_only':
        X = X.loc[:, [f'21003-{data_instance}.0']]
    elif experiment == 'all_demographics':
        X = X.loc[:, df_utils.pull_columns_by_prefix(X, [f'21003-{data_instance}.0', '31-0.0', 'apoe', 'max_educ_complete', '845-0.0', '21000-0.0']).columns.tolist()]
    elif experiment == 'modality_only':
        X = X.loc[:, modality_vars]
    elif experiment == 'demographics_and_modality':
        X = X.loc[:, df_utils.pull_columns_by_prefix(X, [f'21003-{data_instance}.0', '31-0.0', 'apoe', 'max_educ_complete', '845-0.0', '21000-0.0']).columns.tolist() + modality_vars]    

    print(f'Running {data_modality} on the {experiment} experiment, {metric} as the metric, and an age cutoff of {age_cutoff} years')

    # set the metric function if f3 is chosen
    if metric == 'f3':
        metric = f3.f3_metric


    print(f'Dimensionality of the dataset: {X.shape}')
    
    if data_modality == 'neuroimaging':
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            if i != region_index:
                continue
            else:
                # X_test = X.iloc[test_index, :]
                # y_test = y[test_index]

                X_train = X.iloc[train_index, :]
                y_train = y[train_index]
                break
    else:
        i = region_index
        r = region_list[i]
        
        indices = region_indices[r]
        # X_test = X.iloc[indices, :]
        # y_test = y[indices]

        # Create a mask to select all indices not in indices
        mask = np.ones(len(y), dtype=bool)
        mask[indices] = False

        # Step 4: Subset the main array using the mask
        X_train = X.iloc[mask, :]
        y_train = y[mask]
    
    print(f'Made train and test split for region {i}')
    current_time = datetime.now().time()
    print(f'Current Time = {current_time}')
    
    automl = AutoML()
    automl.retrain_from_log(log_file_name=f'{directory_path}/results_log_{i}.json',
                            X_train=X_train, y_train=y_train, task='classification', 
                            train_full=True, n_jobs=-1,
                            #eval_method='cv', n_splits=10, split_type='stratified',
                            train_best=True)

    print('Done fitting model')


    if len(automl.feature_importances_) == 1:
        feature_names = np.array(automl.feature_names_in_)[np.argsort(abs(automl.feature_importances_[0]))[::-1]]
        fi = automl.feature_importances_[0][np.argsort(abs(automl.feature_importances_[0]))[::-1]]
    else:
        feature_names = np.array(automl.feature_names_in_)[np.argsort(abs(automl.feature_importances_))[::-1]]
        fi = automl.feature_importances_[np.argsort(abs(automl.feature_importances_))[::-1]]
        
    fi_df = pd.DataFrame({'feature': feature_names, 'importance': fi})
    fi_df.to_parquet(f'{directory_path}/feature_importance_region_{i}.parquet')
    

if __name__ == "__main__":
    main()