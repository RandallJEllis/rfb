import sys
sys.path.append('../ukb_func')
import icd
import ml_utils
import df_utils
import dementia_utils
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
    directory_path = f'../../results/dementia/{data_modality}/{experiment}/survival/'

    # Load datasets
    X = pd.read_parquet(f'../../tidy_data/dementia/{data_modality}/X.parquet')
    y = np.load(f'../../tidy_data/dementia/{data_modality}/y.npy')

    data_path = '../../proj_idp/tidy_data/'
    demo = ukb_utils.load_demographics(data_path)
    df = X.merge(demo.loc[:, ['eid', f'53-{data_instance}.0']], on='eid')

    acd = pd.read_parquet(data_path + 'acd/allcausedementia.parquet')
    acd = dementia_utils.get_first_diagnosis(acd)

    controls = df.iloc[y == 0]
    cases = df.iloc[y == 1]
    cases = pd.merge(cases, acd.loc[:, ['eid', 'first_dx']], on='eid')
    controls.loc[:, 'first_dx'] = 'control'

    cases['time2event'] = cases['first_dx'] - cases[f'53-{data_instance}.0']
    controls['time2event'] = np.nan