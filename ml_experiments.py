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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from datetime import datetime
import pickle

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
            modality_only, demographics_and_modality, demographics_modality_lancet2024
time_budget - number of seconds for AutoML training
metric - log_loss, roc_auc, or f3 (log_loss is used throughout the project)
model - lgbm or lrl1
age_cutoff - 0 or 65
region_index - index of the region to run the experiment on (or CV fold for neuroimaging)
'''


def main():

    # Create the argument parser
    parser = argparse.ArgumentParser(
        description='Run AutoML on chosen feature sets')

    # Add arguments
    parser.add_argument('--modality', type=str,
                        help='options: proteomics, neuroimaging, cognitive_tests')
    parser.add_argument('--experiment', type=str,
                        help='options: age_only, all_demographics, \
                        age_sex_lancet2024, demographics_and_lancet2024, modality_only, \
                        demographics_and_modality, demographics_modality_lancet2024')
    parser.add_argument('--time_budget', type=int, default=3600,
                        help='options: seconds to allow FLAML to optimize')
    parser.add_argument('--metric', type=str, default='roc_auc',
                        help='options: roc_auc, f3, ap')
    parser.add_argument('--model', type=str, default='lgbm',
                        help='options: lgbm, lrl1')
    parser.add_argument('--age_cutoff', type=int, default=None,
                        help='age cutoff')
    parser.add_argument('--region_index', type=int, default=None,
                        help='region index')

    # Parse the arguments
    args = parser.parse_args()
    data_modality = args.modality
    
    X = pd.read_parquet(f'../tidy_data/dementia/{data_modality}/X.parquet')
    X = X.iloc[:,1:]
    y = np.load(f'../../tidy_data/dementia/{data_modality}/y.npy')
    
    # set data instance, import region indices if not neuroimaging, and set modality_vars
    if data_modality == 'neuroimaging':
        data_instance = 2
    else:
        data_instance = 0
        region_indices = pickle.load(
            open(f'../tidy_data/dementia/{data_modality}/region_cv_indices.pickle', 'rb')
            )
        
    experiment = args.experiment
    time_budget = args.time_budget
    metric = args.metric
    model = args.model
    age_cutoff = args.age_cutoff
    if age_cutoff == 0:
        age_cutoff = None
    region_index = args.region_index

    if region_index == None:
        print('NEED REGION INDEX')
        sys.exit()

    # Specify the directory path
    directory_path = f'../results/dementia/{data_modality}/{experiment}/{metric}/{model}/'
 
    # subset data by age if there is an age cutoff
    if age_cutoff is not None:
        over_age_idx = X[f'21003-{data_instance}.0'] >= age_cutoff
        X = X[over_age_idx].reset_index(drop=True)
        y = y[over_age_idx]
        
        # update region_indices if not neuroimaging
        if data_modality != 'neuroimaging':
            region_lookup = pd.read_csv('../metadata/coding10.tsv', sep='\t')
            region_indices = ukb_utils.group_assessment_center(
                X, data_instance, region_lookup)

        directory_path = f'{directory_path}/agecutoff_{age_cutoff}'

    # check if output folder exists
    utils.check_folder_existence(directory_path)

    # set up experiment variables
    lancet_vars = ['4700-0.0', '5901-0.0', '30780-0.0', 'head_injury', '22038-0.0', '20161-0.0',
                   'alcohol_consumption', 'hypertension', 'obesity', 'diabetes', 'hearing_loss',
                   'depression', 'freq_friends_family_visit', '24012-0.0', '24018-0.0',
                   '24019-0.0', '24006-0.0', '24015-0.0', '24011-0.0', '2020-0.0_-3.0',
                   '2020-0.0_-1.0', '2020-0.0_0.0', '2020-0.0_1.0', '2020-0.0_nan']
    continuous_lancet_vars = ['4700-0.0', '5901-0.0', '30780-0.0', '22038-0.0',
                                '20161-0.0','24012-0.0', '24018-0.0', '24019-0.0',
                                '24006-0.0', '24015-0.0', '24011-0.0']
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
                          'cognitive_tests': pickle.load(open(f'../tidy_data/dementia/{data_modality}/cognitive_columns.pkl', 'rb'))}, 
    }
    experiment_vars['demographics_and_modality'] = {'proteomics': experiment_vars['all_demographics'] + experiment_vars['modality_only']['proteomics'],
                                                    'neuroimaging': experiment_vars['all_demographics'] + experiment_vars['modality_only']['neuroimaging'],
                                                    'cognitive_tests': experiment_vars['all_demographics'] + experiment_vars['modality_only']['cognitive_tests']} 
    experiment_vars['demographics_modality_lancet2024'] = {'proteomics': experiment_vars['demographics_and_modality']['proteomics'] + lancet_vars,
                                                           'neuroimaging': experiment_vars['demographics_and_modality']['neuroimaging'] + lancet_vars,
                                                           'cognitive_tests': experiment_vars['demographics_and_modality']['cognitive_tests'] + lancet_vars}

    time_budgets = {
        'age_only': {
            'proteomics': 10,
            'neuroimaging': 10,
            'cognitive_tests': 10
        }, 
        'all_demographics': {
            'proteomics': {
                'lgbm': 25,
                'lrl1': 500
            },
            'neuroimaging': {
                'lgbm': 25,
                'lrl1': 500
            },
            'cognitive_tests': {
                'lgbm': 25,
                'lrl1': 500
            }
        },
        'age_sex_lancet2024': {
            'proteomics': {
                'lgbm': 75,
                'lrl1': 700
            },
            'neuroimaging': {
                'lgbm': 75,
                'lrl1': 700
            },
            'cognitive_tests': {
                'lgbm': 75,
                'lrl1': 700
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
                'lgbm': 100,
                'lrl1': 800
            }
        }
    }

    # set up continuous columns for scaling if using lrl1
    continuous_cols = {
        'age_only': [f'21003-{data_instance}.0'],
        'all_demographics': df_utils.pull_columns_by_prefix(X, [f'21003-{data_instance}.0', '845-0.0']).columns.tolist(),
        'age_sex_lancet2024': df_utils.pull_columns_by_prefix(X, [f'21003-{data_instance}.0', '845-0.0']) + \
                                            continuous_lancet_vars,
        'demographics_and_lancet2024': df_utils.pull_columns_by_prefix(X, [f'21003-{data_instance}.0', '845-0.0']) + \
                                            continuous_lancet_vars,
        'modality_only': { # SET THIS UP FOR ALL MODALITIES
            'proteomics': df_utils.pull_columns_by_suffix(X, ['-0']).columns.tolist(),
            'neuroimaging': pickle.load(open('../tidy_data/dementia/neuroimaging/idp_variables.pkl', 'rb')),
            'cognitive_tests': pickle.load(open(f'../tidy_data/dementia/{data_modality}/cognitive_columns.pkl', 'rb'))
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

    # subset data based on experiment and data_modality
    if experiment in experiment_vars:
        if isinstance(experiment_vars[experiment], dict):
            if data_modality in experiment_vars[experiment]:
                X = X.loc[:, experiment_vars[experiment][data_modality]]
            else:
                # output an error saying data_modality is not in experiment_vars
                print('Data modality not in experiment_vars')
                sys.exit()
        else:
            X = X.loc[:, experiment_vars[experiment]]
    else:
        # output an error saying experiment is not in experiment_vars
        print('Experiment not in experiment_vars')
        sys.exit()
    
    # set time budget based on experiment, data_modality, and model    
    if experiment in time_budgets:
        time_budget = time_budgets[experiment][data_modality][model]
    else:
        # output an error saying experiment is not in time_budgets
        print('Experiment not in time_budgets')
        sys.exit()
    
    # set continuous columns for scaling if using lrl1
    if model == 'lrl1':    
        if experiment in continuous_cols:
            if isinstance(time_budgets[experiment], dict):
                continuous_cols = continuous_cols[experiment][data_modality]
            else:
                continuous_cols = continuous_cols[experiment]
        else:
            # output an error saying experiment is not in continuous_cols
            print('Experiment not in continuous_cols')
            sys.exit()
                
    # modify time budget for age cutoff of 65                        
    if age_cutoff == 65:
        print('Modifying time budget by dividing by 2 for age cutoff of 65') 
        if model == 'lgbm':
            time_budget = time_budget/2
        if model == 'lrl1':
            time_budget = time_budget/4

    # print experiment details
    print(f'Running {experiment} experiment, region {region_index}, autoML time budget of {time_budget} seconds, {metric} as the metric, and an age cutoff of {age_cutoff} years')

    # set metric to f3 if using f3
    if metric == 'roc_auc':
        pass
    elif metric == 'f3':
        metric = f3.f3_metric

    # set up lists to store results
    train_labels_l = []
    train_probas_l = []

    test_labels_l = []
    test_probas_l = []

    print(f'Dimensionality of the dataset: {X.shape}')

    # Split the data into training and testing sets
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

        # Step 4: Subset the main array using the mask
        X_train = X.iloc[mask, :]
        y_train = y[mask]
    print('made train and test split')

    automl = AutoML()
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
    "log_file_name": f'{directory_path}/results_log_{i}.json',
    "estimator_list": [model]
    }
    
    if model == 'lrl1':
        automl_settings['max_iter'] = 100000000
        
        print('Scaling data')
        # Initialize the StandardScaler
        scaler = StandardScaler()

        # Fit and transform only the continuous columns
        scaler.fit(X_train[continuous_cols])
        X_train[continuous_cols] = scaler.transform(X_train[continuous_cols])
        X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])
        print('Done scaling data')
    
    print(automl_settings)
    print(f'Fitting model: {datetime.now().time()}')
    automl.fit(X_train, y_train, **automl_settings)
    print(f'Done fitting model: {datetime.now().time()}')

    if len(automl.feature_importances_) == 1:
        feature_names = np.array(automl.feature_names_in_)[np.argsort(abs(automl.feature_importances_[0]))[::-1]]
        fi = automl.feature_importances_[0][np.argsort(abs(automl.feature_importances_[0]))[::-1]]
    else:
        feature_names = np.array(automl.feature_names_in_)[np.argsort(abs(automl.feature_importances_))[::-1]]
        fi = automl.feature_importances_[np.argsort(abs(automl.feature_importances_))[::-1]]
        
    fi_df = pd.DataFrame({'feature': feature_names, 'importance': fi})
    fi_df.to_parquet(f'{directory_path}/feature_importance_region_{region_index}.parquet')

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

if __name__ == "__main__":
    main()
