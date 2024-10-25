import sys
sys.path.append('./ukb_func')
import ml_utils
import df_utils
import ukb_utils
import utils
import f3
import dementia_utils

import pickle
import argparse
import os
import pandas as pd
import numpy as np
from flaml import AutoML
from flaml.automl.data import get_output_from_log
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
import pickle

from pyarrow.parquet import ParquetFile
import pyarrow as pa 

'''
Machine learning experiments for:
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


def parse_args():
    """
    Parses command-line arguments for running AutoML experiments.
    Returns:
        tuple: A tuple containing the following elements:
            - data_modality (str): The chosen data modality. Options are 'proteomics', 'neuroimaging', 'cognitive_tests'.
            - experiment (str): The chosen experiment type. Options include 'age_only', 'all_demographics', 
              'age_sex_lancet2024', 'demographics_and_lancet2024', 'modality_only', 'demographics_and_modality', 
              'demographics_modality_lancet2024, 'fs_modality_only', 'fs_demographics_and_modality', 'fs_demographics_modality_lancet2024'.
            - time_budget (int): The time budget in seconds for FLAML to optimize.
            - metric (str): The evaluation metric to use. Options are 'log_loss', 'roc_auc', 'f3', 'ap'. Default is 'log_loss'.
            - model (str): The model to use. Options are 'lgbm', 'lrl1'. Default is 'lgbm'.
            - age_cutoff (int or None): The age cutoff value. Default is None.
            - region_index (int): The region index. If not provided, the function will print 'NEED REGION INDEX' and exit.
    Raises:
        SystemExit: If the region_index is not provided.
    """
     # Create the argument parser
    parser = argparse.ArgumentParser(
        description='Run AutoML on chosen feature sets')

    # Add arguments
    parser.add_argument('--modality', type=str, required=True,
                        help='options: proteomics, neuroimaging, cognitive_tests')
    parser.add_argument('--experiment', type=str, required=True,
                        help='options: age_only, all_demographics, \
                        age_sex_lancet2024, demographics_and_lancet2024, modality_only, \
                        demographics_and_modality, demographics_modality_lancet2024, \
                        fs_modality_only, fs_demographics_and_modality, fs_demographics_modality_lancet2024'
                        )
    parser.add_argument('--metric', type=str, default='log_loss',
                        help='options: log_loss, roc_auc, f3, ap')
    parser.add_argument('--model', type=str, default='lgbm',
                        help='options: lgbm, lrl1')
    parser.add_argument('--age_cutoff', type=int, default=None,
                        help='age cutoff')
    parser.add_argument('--region_index', type=int, required=True,
                        help='region index')
    parser.add_argument('--predict_alzheimers_only', type=int, default=0,
                        help='predict alzheimers outcomes only')

    # Parse the arguments
    args = parser.parse_args()
    data_modality = args.modality
    if data_modality == 'neuroimaging':
        data_instance = 2
    else:
        data_instance = 0
        
    experiment = args.experiment
    metric = args.metric
    model = args.model
    
    age_cutoff = args.age_cutoff
    if age_cutoff == 0:
        age_cutoff = None
    region_index = args.region_index
    
    if args.predict_alzheimers_only == 0:
        alzheimers_only = False
    elif args.predict_alzheimers_only == 1:
        alzheimers_only = True
    else:
        print('predict_alzheimers_only must be 0 or 1')
        sys.exit()        
        

    if region_index == None:
        print('NEED REGION INDEX')
        sys.exit()
        
    return data_modality, data_instance, experiment, metric, model, age_cutoff, region_index, alzheimers_only 
 
def update_region_indices(X, data_instance):
    """
    Updates the region indices for the given data instance.

    This function reads a region lookup table from a TSV file and uses it to 
    group the assessment center data in the provided dataset. It returns the 
    updated region indices.

    Args:
        X (pd.DataFrame): The dataset containing the data to be updated.
        data_instance (int): The specific instance of data to be processed.

    Returns:
        pd.GroupBy: A grouped dataframe containing the updated region indices.
    """
    region_lookup = pd.read_csv('../metadata/coding10.tsv', sep='\t')
    region_indices = ukb_utils.group_assessment_center(
            X, data_instance, region_lookup)
    return region_indices
        
def alzheimers_cases_only(X, y):
    """
    Filters the input dataset to include only Alzheimer's disease cases and controls.
    Parameters:
    X (pd.DataFrame): The input features dataframe containing participant data.
    y (np.ndarray or pd.Series): The target array indicating the presence or absence of dementia.
    Returns:
    tuple: A tuple containing:
        - X_new (pd.DataFrame): The filtered features dataframe with only Alzheimer's cases and controls.
        - y_new (np.ndarray): The filtered target array with only Alzheimer's cases and controls.
    """
    data_path = '../../proj_idp/tidy_data/'
    
    # the allcausedementia file is large, so we will import the first row to get the columns we need
    pf = ParquetFile(data_path + 'acd/allcausedementia.parquet') 
    first_rows = next(pf.iter_batches(batch_size = 2)) 
    acd = pa.Table.from_batches([first_rows]).to_pandas() 
    acd = df_utils.pull_columns_by_prefix(acd, ['eid', '42019', '42021', '42023', '42025',
                                '131037', '130837', '130839', '130841', '130843',
                                '42018', '42020', '42022', '42024', '131036',
                                '130836', '130838', '130840', '130842'])
    cols_import = acd.columns.tolist()
        
    acd = pd.read_parquet(data_path + 'acd/allcausedementia.parquet', columns=cols_import)
    acd = acd[acd.eid.isin(X.eid)]
        
    both_eid, _, _ = dementia_utils.pull_dementia_cases(acd, alzheimers_only=True)

    indices_keep = X.loc[(y == 0) | (X['eid'].isin(both_eid))].index
    X = X.loc[indices_keep]
    y = y[indices_keep]
    
    print('Alzheimer\'s cases:', sum(y))
    print('Controls:', len(y[y==0]))
    print('Data shape:', X.shape)
    
    return X, y

def load_datasets(data_modality, data_instance, alzheimers_only=False):
    """
    Load datasets for a given data modality.
    Parameters:
    data_modality (str): The modality of the data to load. Can be 'neuroimaging' or other modalities.
    Returns:
    tuple: A tuple containing:
        - X (pd.DataFrame): The feature dataset.
        - y (np.ndarray): The target labels.
        - data_instance (int): An identifier for the data instance (2 for 'neuroimaging', 0 for other modalities).
        - region_indices (list or None): Region indices for cross-validation if the modality is not 'neuroimaging', otherwise None.
    """
    X = pd.read_parquet(f'../tidy_data/dementia/{data_modality}/X.parquet')
    y = np.load(f'../tidy_data/dementia/{data_modality}/y.npy')

    if alzheimers_only:
        X, y = alzheimers_cases_only(X, y)
    
    # set data instance, import region indices if not neuroimaging, and set modality_vars
    if data_modality == 'neuroimaging':
        return X, y, None
    else:       
        region_indices = update_region_indices(X, data_instance)     
        # region_indices = pickle.load(
        #     open(f'../tidy_data/dementia/{data_modality}/region_cv_indices.pickle', 'rb')
        #     )
        return X, y, region_indices
  
def get_dir_path(data_modality, experiment, metric, model, alzheimers_only=False):
    """
    Get the directory path based on the specified parameters.

    Args:
        data_modality (str): The data modality.
        experiment (str): The experiment name.
        metric (str): The metric name.
        model (str): The model name.

    Returns:
        tuple: A tuple containing the directory path and the original results directory path.
    """
    if alzheimers_only:
        base_path = '../results/alzheimers'
    else:
        base_path = '../results/dementia'
    
    if 'fs_' in experiment: # running a feature selection experiment
        main_experiment = experiment[3:] # remove 'fs_' from experiment name
        directory_path = f'{base_path}/{data_modality}/{main_experiment}/feature_selection/{metric}/{model}/' 
        original_results_directory_path = f'{base_path}/{data_modality}/{main_experiment}/{metric}/{model}/' 
        
    else:
        directory_path = f'{base_path}/{data_modality}/{experiment}/{metric}/{model}/' 
        original_results_directory_path = None     
        
    return directory_path, original_results_directory_path
                  
def setup_age_cutoff(directory_path, original_results_directory_path, X, y, age_cutoff, data_modality, data_instance):
    """
    Filters the dataset based on an age cutoff and updates the directory path accordingly.
    Parameters:
    X (pd.DataFrame): The feature matrix containing the data.
    y (pd.Series): The target variable.
    age_cutoff (int): The age threshold to filter the data.
    data_modality (str): The type of data modality (e.g., 'neuroimaging').
    data_instance (int): The specific data instance identifier.
    Returns:
    tuple: A tuple containing the updated directory path, filtered feature matrix (X), 
           filtered target variable (y), and region indices (if applicable, otherwise None).
    """
    directory_path = f'{directory_path}agecutoff_{age_cutoff}'
    if original_results_directory_path is not None:
        original_results_directory_path = f'{original_results_directory_path}agecutoff_{age_cutoff}'
            
    over_age_idx = X[f'21003-{data_instance}.0'] >= age_cutoff
    X = X[over_age_idx].reset_index(drop=True)
    y = y[over_age_idx]
    
    # update region_indices if not neuroimaging
    if data_modality != 'neuroimaging':
        region_indices = update_region_indices(X, data_instance)
        return directory_path, original_results_directory_path, X, y, region_indices
    
    else:
        return directory_path, original_results_directory_path, X, y, None
 
def get_lancet_vars():
    """
    Returns two lists of variables related to a study.

    The first list, `lancet_vars`, contains a mix of categorical and continuous variables.
    The second list, `continuous_lancet_vars`, contains only continuous variables.

    Returns:
        tuple: A tuple containing two lists:
            - lancet_vars (list of str): A list of variable identifiers and names.
            - continuous_lancet_vars (list of str): A list of continuous variable identifiers.
    """
    lancet_vars = ['4700-0.0', '5901-0.0', '30780-0.0', 'head_injury', '22038-0.0', '20161-0.0',
                   'alcohol_consumption', 'hypertension', 'obesity', 'diabetes', 'hearing_loss',
                   'depression', 'freq_friends_family_visit', '24012-0.0', '24018-0.0',
                   '24019-0.0', '24006-0.0', '24015-0.0', '24011-0.0', '2020-0.0_-3.0',
                   '2020-0.0_-1.0', '2020-0.0_0.0', '2020-0.0_1.0', '2020-0.0_nan']
    continuous_lancet_vars = ['4700-0.0', '5901-0.0', '30780-0.0', '22038-0.0',
                                '20161-0.0','24012-0.0', '24018-0.0', '24019-0.0',
                                '24006-0.0', '24015-0.0', '24011-0.0']
    return lancet_vars, continuous_lancet_vars

def _get_experiment_vars(data_instance, X, lancet_vars):
    """
    Generates a dictionary of experiment variables based on the given data modality and data instance.

    Args:
        data_instance (str): The instance of the data (e.g., '0', '1').
        X (pd.DataFrame): The feature matrix containing the data.
        lancet_vars (list): A list of variables related to the Lancet 2024 study

    Returns:
        dict: A dictionary containing various sets of experiment variables:
            - 'age_only': List of columns related to age.
            - 'all_demographics': List of columns related to all demographics.
            - 'age_sex_lancet2024': List of columns related to age, sex, and additional variables from the Lancet 2024 study.
            - 'demographics_and_lancet2024': List of columns related to demographics and additional variables from the Lancet 2024 study.
            - 'modality_only': Dictionary with keys as modalities and values as lists of columns specific to each modality.
            - 'demographics_and_modality': Dictionary with keys as modalities and values as lists of columns combining demographics and modality-specific columns.
            - 'demographics_modality_lancet2024': Dictionary with keys as modalities and values as lists of columns combining demographics, modality-specific columns, and additional variables from the Lancet 2024 study.
    """
    experiment_vars = {
        'age_only': [f'21003-{data_instance}.0'],
        'all_demographics': df_utils.pull_columns_by_prefix(X, [f'21003-{data_instance}.0', '31-0.0', 'apoe',
                                                         'max_educ_complete', '845-0.0', '21000-0.0']).columns.tolist(),
        'age_sex_lancet2024': df_utils.pull_columns_by_prefix(X, [f'21003-{data_instance}.0', '31-0.0',
                                                         'max_educ_complete', '845-0.0',
                                                         '21000-0.0']).columns.tolist() + lancet_vars,
        'demographics_and_lancet2024': df_utils.pull_columns_by_prefix(X, [f'21003-{data_instance}.0', '31-0.0',
                                                         'apoe', 'max_educ_complete', '845-0.0',
                                                         '21000-0.0']).columns.tolist() + lancet_vars,
        'modality_only': {'proteomics': df_utils.pull_columns_by_suffix(X, ['-0']).columns.tolist(),\
                          'neuroimaging': pickle.load(open('../tidy_data/dementia/neuroimaging/idp_variables.pkl', 'rb')),\
                          'cognitive_tests': pickle.load(open(f'../tidy_data/dementia/cognitive_tests/cognitive_columns.pkl', 'rb'))}, 
    }
    experiment_vars['demographics_and_modality'] = {'proteomics': experiment_vars['all_demographics'] + experiment_vars['modality_only']['proteomics'],
                                                    'neuroimaging': experiment_vars['all_demographics'] + experiment_vars['modality_only']['neuroimaging'],
                                                    'cognitive_tests': experiment_vars['all_demographics'] + experiment_vars['modality_only']['cognitive_tests']} 
    experiment_vars['demographics_modality_lancet2024'] = {'proteomics': experiment_vars['demographics_and_modality']['proteomics'] + lancet_vars,
                                                           'neuroimaging': experiment_vars['demographics_and_modality']['neuroimaging'] + lancet_vars,
                                                           'cognitive_tests': experiment_vars['demographics_and_modality']['cognitive_tests'] + lancet_vars}
    experiment_vars['fs_modality_only'] = experiment_vars['modality_only']
    experiment_vars['fs_demographics_and_modality'] = experiment_vars['demographics_and_modality']
    experiment_vars['fs_demographics_modality_lancet2024'] = experiment_vars['demographics_modality_lancet2024']
    
    return experiment_vars

def subset_experiment_vars(data_modality, data_instance, experiment, X, lancet_vars, survival=False):
    """
    Subsets the feature matrix based on the experiment variables.

    Args:
        data_modality (str): The type of data modality (e.g., 'proteomics', 'neuroimaging', 'cognitive_tests').
        data_instance (int): The instance of the data (e.g., 0, 1, 2).
        experiment (str): The chosen experiment type.
        X (pd.DataFrame): The feature matrix containing the data.
        lancet_vars (list): A list of variables related to the Lancet 2024 study.
        survival (bool): Are you running survival analysis? Default is False.
        
    Returns:
        pd.DataFrame: The feature matrix with columns subset based on the experiment variables.
    """
    
    experiment_vars = _get_experiment_vars(data_instance, X, lancet_vars)
    
    if survival:
        t2e_features = ['time2event', 'label']
    else:
        t2e_features = []
        
    if experiment in experiment_vars:
        if isinstance(experiment_vars[experiment], dict):
            if data_modality in experiment_vars[experiment]:
                X = X.loc[:, experiment_vars[experiment][data_modality] + t2e_features]
            else:
                # output an error saying data_modality is not in experiment_vars
                print('Data modality not in experiment_vars')
                sys.exit()
        else:
            X = X.loc[:, experiment_vars[experiment] + t2e_features]
    else:
        # output an error saying experiment is not in experiment_vars
        print('Experiment not in experiment_vars')
        sys.exit()
    return X

def _flaml_time_budgets():
    """
    Returns a dictionary containing time budgets for various machine learning experiments.

    The time budgets are organized by different experimental setups and modalities. Each setup
    contains time budgets for different modalities such as 'proteomics', 'neuroimaging', and 
    'cognitive_tests'. The time budgets are further divided by machine learning models such as 
    'lgbm' and 'lrl1'.

    The structure of the returned dictionary is as follows:
    {
        'experiment_setup': {
            'modality': {
                'model': time_budget

    Example:
    {
            ...
        ...

    Returns:
        dict: A dictionary containing time budgets for various machine learning experiments.
    """
    time_budgets = {
        'age_only': {
            'proteomics': {
                'lgbm': 10,
                'lrl1': 10
            },
            'neuroimaging': {
                'lgbm': 10,
                'lrl1': 10
            },
            'cognitive_tests': {
                'lgbm': 30,
                'lrl1': 100
            }
        }, 
        'all_demographics': {
            'proteomics': {
                'lgbm': 25,
                'lrl1': 100
            },
            'neuroimaging': {
                'lgbm': 30,
                'lrl1': 500
            },
            'cognitive_tests': {
                'lgbm': 150,
                'lrl1': 2500
            }
        },
        'age_sex_lancet2024': {
            'proteomics': {
                'lgbm': 75,
                'lrl1': 500
            },
            'neuroimaging': {
                'lgbm': 75,
                'lrl1': 600
            },
            'cognitive_tests': {
                'lgbm': 325,
                'lrl1': 3000
            }
        },
        'demographics_and_lancet2024': {
            'proteomics': {
                'lgbm': 100,
                'lrl1': 800
            },
            'neuroimaging': {
                'lgbm': 100,
                'lrl1': 800
            },
            'cognitive_tests': {
                'lgbm': 350,
                'lrl1': 3500
            }
        },
        'modality_only': {
            'proteomics': {
                'lgbm': 8500,
                'lrl1': 35000
            },
            'neuroimaging': {
                'lgbm': 9500,
                'lrl1': 36000
            },
            'cognitive_tests': {
                'lgbm': 350,
                'lrl1': 4500
            }
        },
        'demographics_and_modality': {
            'proteomics': {
                'lgbm': 9000,
                'lrl1': 35000
            },
            'neuroimaging': {
                'lgbm': 14000,
                'lrl1': 38000
            },
            'cognitive_tests': {
                'lgbm': 400,
                'lrl1': 5500
            }
        },
        'demographics_modality_lancet2024': {
            'proteomics': {
                'lgbm': 9500,
                'lrl1': 35000
            },
            'neuroimaging': {
                'lgbm': 14500,
                'lrl1': 40000
            },
            'cognitive_tests': {
                'lgbm': 450,
                'lrl1': 6500
            }
        },
        'fs_modality_only': {
            'proteomics': {
                'lgbm': 8500,
                'lrl1': 35000
            },
            'neuroimaging': {
                'lgbm': 9500,
                'lrl1': 36000
            },
            'cognitive_tests': {
                'lgbm': 350,
                'lrl1': 4500
            }
        },
        'fs_demographics_and_modality': {
            'proteomics': {
                'lgbm': 9000,
                'lrl1': 35000
            },
            'neuroimaging': {
                'lgbm': 14000,
                'lrl1': 38000
            },
            'cognitive_tests': {
                'lgbm': 400,
                'lrl1': 5500
            }
        },
        'fs_demographics_modality_lancet2024': {
            'proteomics': {
                'lgbm': 9500,
                'lrl1': 35000
            },
            'neuroimaging': {
                'lgbm': 14500,
                'lrl1': 40000
            },
            'cognitive_tests': {
                'lgbm': 450,
                'lrl1': 6500
            }
        }
    }
    return time_budgets

def get_time_budget(experiment, data_modality, model, age_cutoff):
    """
    Returns the time budget for a given experiment, data modality, and model.

    Args:
        experiment (str): The chosen experiment type.
        data_modality (str): The type of data modality (e.g., 'proteomics', 'neuroimaging', 'cognitive_tests').
        model (str): The model to use. Options are 'lgbm' and 'lrl1'.
        time_budgets (dict): A dictionary containing time budgets for various machine learning experiments.

    Returns:
        int: The time budget in seconds.
    """
    
    time_budgets = _flaml_time_budgets()
    if experiment in time_budgets:
        if data_modality in time_budgets[experiment]:
            if model in time_budgets[experiment][data_modality]:
                time_budget = time_budgets[experiment][data_modality][model]
            else:
                # output an error saying model is not in time_budgets
                print('Model not in time_budgets')
                sys.exit()
        else:
            # output an error saying data_modality is not in time_budgets
            print('Data modality not in time_budgets')
            sys.exit()
    else:
        # output an error saying experiment is not in time_budgets
        print('Experiment not in time_budgets')
        sys.exit()
        
    # modify time budget for age cutoff of 65                        
    if age_cutoff == 65:
        print('Modifying time budget for age cutoff of 65') 
        if model == 'lgbm':
            time_budget = time_budget/2
        if model == 'lrl1':
            if data_modality == 'neuroimaging':
                time_budget = time_budget/2
            else:
                time_budget = time_budget/4
    return time_budget
        
def continuous_vars_for_scaling(data_modality, data_instance, experiment, continuous_lancet_vars, X):
    """
    Generates a dictionary of continuous columns for scaling based on the specified data modality, 
    data instance, and continuous Lancet variables.

    Parameters:
    data_modality (str): The modality of the data (e.g., 'proteomics', 'neuroimaging', 'cognitive_tests').
    data_instance (int): The instance of the data (e.g., 0, 1, 2).
    continuous_lancet_vars (list): A list of continuous Lancet variables to be included.

    Returns:
    dict: A dictionary where keys represent different configurations (e.g., 'age_only', 'all_demographics', 
          'age_sex_lancet2024', 'demographics_and_lancet2024', 'modality_only', 'demographics_and_modality', 
          'demographics_modality_lancet2024') and values are lists of column names or dictionaries of column names 
          for the specified configurations.
    """

    continuous_cols = {
        'age_only': [f'21003-{data_instance}.0'],
        'all_demographics': df_utils.pull_columns_by_prefix(X, [f'21003-{data_instance}.0', '845-0.0']).columns.tolist(),
        'age_sex_lancet2024': df_utils.pull_columns_by_prefix(X, [f'21003-{data_instance}.0', '845-0.0']).columns.tolist() + \
                                            continuous_lancet_vars,
        'demographics_and_lancet2024': df_utils.pull_columns_by_prefix(X, [f'21003-{data_instance}.0', '845-0.0']).columns.tolist() + \
                                            continuous_lancet_vars,
        'modality_only': { # SET THIS UP FOR ALL MODALITIES
            'proteomics': df_utils.pull_columns_by_suffix(X, ['-0']).columns.tolist(),
            'neuroimaging': pickle.load(open('../tidy_data/dementia/neuroimaging/idp_variables.pkl', 'rb')),
            'cognitive_tests': pickle.load(open(f'../tidy_data/dementia/cognitive_tests/cognitive_columns.pkl', 'rb'))
        },
    }
    continuous_cols['demographics_and_modality'] = {
        'proteomics': continuous_cols['all_demographics'] + continuous_cols['modality_only']['proteomics'],
        'neuroimaging': continuous_cols['all_demographics'] + continuous_cols['modality_only']['neuroimaging'],
        'cognitive_tests': continuous_cols['all_demographics'] + continuous_cols['modality_only']['cognitive_tests']
    }
    continuous_cols['demographics_modality_lancet2024'] = {
        'proteomics': continuous_cols['demographics_and_modality']['proteomics'] + continuous_lancet_vars,
        'neuroimaging': continuous_cols['demographics_and_modality']['neuroimaging'] + continuous_lancet_vars,
        'cognitive_tests': continuous_cols['demographics_and_modality']['cognitive_tests'] + continuous_lancet_vars
    }
    
    continuous_cols['fs_modality_only'] = continuous_cols['modality_only']
    continuous_cols['fs_demographics_and_modality'] = continuous_cols['demographics_and_modality']
    continuous_cols['fs_demographics_modality_lancet2024'] = continuous_cols['demographics_modality_lancet2024']
    
    if experiment in continuous_cols:
        if isinstance(continuous_cols[experiment], dict):
            continuous_cols = continuous_cols[experiment][data_modality]
        else:
            continuous_cols = continuous_cols[experiment]
    else:
        # output an error saying experiment is not in continuous_cols
        print('Experiment not in continuous_cols')
        sys.exit()
        
    return continuous_cols

def subset_train_test_data(X, y, data_modality, region_index, region_indices):
    """
    Splits the dataset into training and testing subsets based on the specified data modality and region index.
    Parameters:
    X (pd.DataFrame): The feature matrix.
    y (pd.Series or np.ndarray): The target vector.
    data_modality (str): The type of data modality, either 'neuroimaging' or other.
    region_index (int): The index of the region to be used for splitting the data.
    region_indices (dict): A dictionary where keys are region names and values are lists of indices corresponding to those regions.
    Returns:
    tuple: A tuple containing:
        - X_train (pd.DataFrame): The training feature matrix.
        - y_train (pd.Series or np.ndarray): The training target vector.
        - X_test (pd.DataFrame): The testing feature matrix.
        - y_test (pd.Series or np.ndarray): The testing target vector.
        - region (int or str): The region used for the split, either an integer (for 'neuroimaging') or a string (for other modalities).
    """
    
    if data_modality == 'neuroimaging':
        skf = StratifiedKFold(n_splits=10)

        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            
            if i != region_index:
                continue
            print(f"Fold {i}:")
        
            current_time = datetime.now().time()
            print(f'Fold {i}, {current_time}')

            X_test = X.iloc[test_index, :]
            y_test = y[test_index]

            X_train = X.iloc[train_index, :]
            y_train = y[train_index]
            
            # set region to i for neuroimaging
            region = i
    else:
        region_list = list(region_indices.keys())    
        region = region_list[region_index]
        print(f'Starting region {region_index+1} of {len(region_list)}: {region}')

        current_time = datetime.now().time()
        print(f'{region}, {current_time}')
        indices = region_indices[region]
        X_test = X.iloc[indices, :]
        y_test = y[indices]

        # Create a mask to select all indices not in indices
        mask = np.ones(len(y), dtype=bool)
        mask[indices] = False

        # Subset the main array using the mask
        X_train = X.iloc[mask, :]
        y_train = y[mask]
        
    print('Made train and test split')
    return X_train.reset_index(drop=True), y_train, X_test.reset_index(drop=True), y_test, region

def scale_continuous_vars(X_train, X_test, continuous_cols):
    """
    Scales the continuous variables in the training and test datasets using StandardScaler.
    Used only with lrl1 model.
    Parameters:
    X_train (pd.DataFrame): The training dataset.
    X_test (pd.DataFrame): The test dataset.
    continuous_cols (list of str): List of column names corresponding to continuous variables to be scaled.
    Returns:
    tuple: A tuple containing the scaled training and test datasets (X_train, X_test).
    """
    
    scaler = StandardScaler()

    # Fit and transform only the continuous columns
    scaler.fit(X_train[continuous_cols])
    X_train.loc[:, continuous_cols] = scaler.transform(X_train[continuous_cols])
    X_test.loc[:, continuous_cols] = scaler.transform(X_test[continuous_cols])
    
    return X_train, X_test
    
def settings_automl(experiment, time_budget, metric, model):
    """
    Generate settings for an AutoML classification task.
    Parameters:
    time_budget (int): The time budget for the AutoML process in seconds.
    metric (str): The evaluation metric to be used (e.g., 'log_loss' 'accuracy', 'f1').
    model (str): The model to be used in the AutoML process (e.g., 'lrl1').
    region_index (int): The index of the region for logging purposes.
    Returns:
    dict: A dictionary containing the settings for the AutoML process.
    """
    
    if 'fs_' in experiment:
        automl_settings = {
            "task": "classification", 
            "train_full": True,
            "n_jobs": -1,
            "train_best": True
        }
        
    else:
        automl_settings = {
        "task": "classification",
        "time_budget": time_budget,
        "metric": metric,
        "n_jobs": -1,
        "eval_method": 'cv',
        "n_splits": 10,
        "split_type": 'stratified',
        "early_stop": True,
        "log_training_metric": True,
        "model_history": True,
        "seed": 239875,
        "estimator_list": [model]
        }
    
        if model == 'lrl1':
            automl_settings['max_iter'] = 100000000
    return automl_settings

def get_top_features(automl):
    """
    Extracts the top features from an AutoML model.
    Parameters:
    automl (object): The AutoML model object.
    Returns:
    list: A list of the top features from the AutoML model.
    """
    if len(automl.feature_importances_) == 1:
        feature_names = np.array(automl.feature_names_in_)[np.argsort(abs(automl.feature_importances_[0]))[::-1]]
        fi = automl.feature_importances_[0][np.argsort(abs(automl.feature_importances_[0]))[::-1]]
    else:
        feature_names = np.array(automl.feature_names_in_)[np.argsort(abs(automl.feature_importances_))[::-1]]
        fi = automl.feature_importances_[np.argsort(abs(automl.feature_importances_))[::-1]]
        
    return feature_names, fi

def save_feature_importance(automl, directory_path, region_index):
    """
    Save the feature importance from an AutoML model to a parquet file.
    Parameters:
    automl (object): The AutoML model object that contains feature importances and feature names.
    directory_path (str): The directory path where the parquet file will be saved.
    region_index (int): The index of the region for which the feature importance is being saved.
    Returns:
    None
    The function extracts feature importances and their corresponding feature names from the AutoML model,
    sorts them in descending order of importance, and saves the result as a parquet file in the specified directory.
    """
    feature_names, fi = get_top_features(automl)
        
    fi_df = pd.DataFrame({'feature': feature_names, 'importance': fi})
    fi_df.to_parquet(f'{directory_path}/feature_importance_region_{region_index}.parquet')
    
def iterative_fs_inference(automl, automl_settings, config_history, X_train, y_train, X_test, y_test, region_index, region):
    """
    Perform iterative feature selection and inference using AutoML.

    Args:
        automl_settings (dict): Settings for AutoML.
        config_history (list): List of configuration history.
        X_train (pandas.DataFrame): Training data features.
        y_train (pandas.Series): Training data labels.
        X_test (pandas.DataFrame): Test data features.
        y_test (pandas.Series): Test data labels.
        directory_path (str): Path to the directory.
        region_index (int): Index of the region.
        region (str): Name of the region.

    Returns:
        tuple: A tuple containing the following:
            - train_labels_l (list): List of training data labels for each iteration.
            - train_probas_l (list): List of training data probabilities for each iteration.
            - test_labels_l (list): List of test data labels for each iteration.
            - test_probas_l (list): List of test data probabilities for each iteration.
            - train_res_l (list): List of training data results for each iteration.
            - test_res_l (list): List of test data results for each iteration.
    """
    train_labels_l = []
    train_probas_l = []

    test_labels_l = []
    test_probas_l = []

    train_res_l = []
    test_res_l = []

    top_feature_names, _ = get_top_features(automl)
    
    tflist = []
    for j, tf in enumerate(top_feature_names[:100]):
        tflist.append(tf)
        current_time = datetime.now().time()
        print(f'Running top {j+1} features: {tflist}, {current_time}')
        
        X_train_sub = X_train.loc[:, tflist]
        X_test_sub = X_test.loc[:, tflist]

        automl = AutoML()
        automl.retrain_from_log(
                                X_train=X_train_sub, y_train=y_train,
                                **automl_settings
                                )

        current_time = datetime.now().time()
        print(f'Done fitting model for region {region_index+1} with top {j+1} variables. {current_time}')

        series_automl = pd.Series([config_history[-1]['Best Learner'], 
                                   config_history[-1]['Best Hyper-parameters'],
                                   region_index, region, tflist],
                                  index=['model', 'hyperparams', 'region_index', 'region', 'features'])

        train_probas = automl.predict_proba(X_train_sub)[:,1]
        train_res, threshold = ml_utils.calc_results(y_train, train_probas, beta=1)
        train_res = pd.concat([series_automl, train_res])
        train_res_l.append(train_res)
        
        if j == 0:
            train_labels_l.append(y_train)
        train_probas_l.append(train_probas)

        test_probas = automl.predict_proba(X_test_sub)[:,1]
        test_res = ml_utils.calc_results(y_test, test_probas, beta=1, threshold=threshold)
        test_res = pd.concat([series_automl, test_res])
        test_res_l.append(test_res)
        
        if j == 0:
            test_labels_l.append(y_test)
        test_probas_l.append(test_probas)
        
    return train_labels_l, train_probas_l, test_labels_l, test_probas_l, train_res_l, test_res_l

def save_fs_results(directory_path, train_labels_l, train_probas_l, test_labels_l, test_probas_l, train_res_l, test_res_l, region_index):
    """
    Save the results of a machine learning experiment to files.

    Parameters:
    - directory_path (str): The path to the directory where the files will be saved.
    - train_labels_l (list): A list of training labels.
    - train_probas_l (list): A list of training probabilities.
    - test_labels_l (list): A list of test labels.
    - test_probas_l (list): A list of test probabilities.
    - train_res_l (list): A list of training results.
    - test_res_l (list): A list of test results.
    - region_index (int): The index of the region.

    Returns:
    None
    """

    ml_utils.save_labels_probas(directory_path, train_labels_l, train_probas_l, test_labels_l, test_probas_l, other_file_info=f'_region_{region_index}')

    train_df = pd.concat(train_res_l, axis=1).T
    train_df.to_csv(f'{directory_path}/training_results_region_{region_index}.csv')

    test_df = pd.concat(test_res_l, axis=1).T
    test_df.to_csv(f'{directory_path}/test_results_region_{region_index}.csv')

def save_results(directory_path, automl, X_train, y_train, X_test, y_test, region, region_index):
    """
    Save the results of an AutoML experiment to CSV files.
    Parameters:
    directory_path (str): The directory where the results will be saved.
    automl (object): The AutoML object that contains the trained model and its configurations.
    X_train (pd.DataFrame): The training feature set.
    y_train (pd.Series): The training labels.
    X_test (pd.DataFrame): The test feature set.
    y_test (pd.Series): The test labels.
    region (str): The region name or identifier.
    region_index (int): The index of the region.
    Returns:
    None
    This function performs the following steps:
    1. Initializes lists to store training and test labels and probabilities.
    2. Creates a pandas Series with the AutoML model and its best configuration.
    3. Predicts probabilities for the training set and calculates results.
    4. Appends the training labels and probabilities to their respective lists.
    5. Predicts probabilities for the test set and calculates results using the same threshold as the training set.
    6. Appends the test labels and probabilities to their respective lists.
    7. Saves the labels and probabilities to files.
    8. Saves the training results to a CSV file, appending to the file if it already exists.
    9. Saves the test results to a CSV file, appending to the file if it already exists.
    """
    # set up lists to store results
    train_labels_l = []
    train_probas_l = []

    test_labels_l = []
    test_probas_l = []
    
    series_automl = pd.Series([region_index, region, automl.best_estimator, automl.best_config],
                              index=['region_index', 'region', 'model', 'hyperparams'])

    train_probas = automl.predict_proba(X_train)[:,1]

    train_res, threshold = ml_utils.calc_results(y_train, train_probas, beta=1)
    train_res = pd.concat([series_automl, train_res])
    train_labels_l.append(y_train)
    train_probas_l.append(train_probas)

    test_probas = automl.predict_proba(X_test)[:,1]
    test_res = ml_utils.calc_results(y_test, test_probas, beta=1, threshold=threshold)

    test_res = pd.concat([series_automl, test_res])
    test_labels_l.append(y_test)
    test_probas_l.append(test_probas)
        

    ml_utils.save_labels_probas(directory_path, train_labels_l, train_probas_l,
                                test_labels_l, test_probas_l, other_file_info=f'_region_{region_index}')

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

def main():

    # Parse the arguments
    data_modality, data_instance, experiment, metric, model, age_cutoff, region_index, alzheimers_only = parse_args()
    print(f'Running {experiment} experiment, modality {data_modality}, instance {data_instance}, region {region_index}, model {model}, {metric} as the metric, and an age cutoff of {age_cutoff} years. Predicting Alzheimer\'s only: {alzheimers_only}')
    
    # Load the datasets
    X, y, region_indices = load_datasets(data_modality, data_instance, alzheimers_only)

    directory_path, original_results_directory_path = get_dir_path(data_modality, experiment, metric, model, alzheimers_only)
    
    # subset data by age if there is an age cutoff
    if age_cutoff is not None:
        directory_path, original_results_directory_path,\
            X, y,\
                region_indices = setup_age_cutoff(directory_path, original_results_directory_path, X, y, age_cutoff, data_modality, data_instance)
    
    # check if output folder exists
    utils.check_folder_existence(directory_path)

    # set up experiment variables
    lancet_vars, continuous_lancet_vars = get_lancet_vars()    
    X = subset_experiment_vars(data_modality, data_instance, experiment, X, lancet_vars)

    # set time budget based on experiment, data_modality, model, and age_cutoff
    time_budget = get_time_budget(experiment, data_modality, model, age_cutoff)
    
    # get experiment-specific continuous variables if using lrl1
    if model == 'lrl1':
        print('Scaling data for lrl1 classifier')
        continuous_cols = continuous_vars_for_scaling(data_modality, data_instance, experiment, continuous_lancet_vars, X)
        print('Done scaling data')
        
    # print experiment details
    print(f'Running {experiment} experiment, region {region_index}, autoML time budget of {time_budget} seconds, {metric} as the metric, and an age cutoff of {age_cutoff} years')

    # set metric to f3 if using f3
    if metric == 'f3':
        metric = f3.f3_metric

    print(f'Dimensionality of the dataset: {X.shape}')

    # Split the data into training and testing sets
    X_train, y_train, X_test, y_test, region = subset_train_test_data(X, y, data_modality, region_index, region_indices)
    if model == 'lrl1':
        X_train, X_test = scale_continuous_vars(X_train, X_test, continuous_cols)
        
    automl = AutoML()
    automl_settings = settings_automl(experiment, time_budget, metric, model)
    print(automl_settings)

    if 'fs_' in experiment:
        automl_settings["log_file_name"] = f'{original_results_directory_path}/results_log_{region_index}.json'
        print(f'Retraining best model to do feature selection: {datetime.now().time()}')
        automl.retrain_from_log(
                            X_train=X_train, y_train=y_train, 
                            **automl_settings
                            )
        print(f'Done retraining model: {datetime.now().time()}')
        
        time_history, best_valid_loss_history,\
            valid_loss_history, config_history,\
                metric_history = get_output_from_log(filename=f'{original_results_directory_path}/results_log_{region_index}.json',
                                                        time_budget=time_budget)

        train_labels_l, train_probas_l,\
            test_labels_l, test_probas_l,\
                train_res_l, test_res_l = iterative_fs_inference(automl, automl_settings, config_history,
                                                                 X_train, y_train, X_test, y_test,
                                                                 region_index, region)

        save_fs_results(directory_path, train_labels_l, train_probas_l, test_labels_l, test_probas_l, train_res_l, test_res_l, region_index)
        
    else:
        automl_settings["log_file_name"] = f'{directory_path}/results_log_{region_index}.json'
        print(f'Fitting model: {datetime.now().time()}')
        automl.fit(X_train, y_train, **automl_settings)
        print(f'Done fitting model: {datetime.now().time()}')

        if experiment != 'age_only':
            # Save the feature importance
            save_feature_importance(automl, directory_path, region_index)

        # Collect the results for train and test
        save_results(directory_path, automl, X_train, y_train, X_test, y_test, region, region_index)

if __name__ == "__main__":
    main()
