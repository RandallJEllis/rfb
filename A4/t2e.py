import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import sys
sys.path.append('../ukb_func')
from ml_utils import save_labels_probas, calc_results
import plot_results
import df_utils
import ukb_utils
from utils import save_pickle, check_folder_existence
from flaml import AutoML
import pickle
from datetime import datetime

from sklearn.metrics import RocCurveDisplay, roc_curve, auc, roc_auc_score, d2_absolute_error_score,\
    d2_pinball_score, d2_tweedie_score, explained_variance_score, max_error,\
        mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score,\
            mean_absolute_percentage_error, mean_poisson_deviance, mean_gamma_deviance, mean_tweedie_deviance,\
                mean_pinball_loss, root_mean_squared_error, root_mean_squared_log_error
from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import (
    f1_score, 
    matthews_corrcoef, 
    confusion_matrix, 
)
from itertools import product
import seaborn as sns

from pyarrow.parquet import ParquetFile
import pyarrow as pa 

from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_ipcw, concordance_index_censored, cumulative_dynamic_auc, integrated_brier_score

from sklearn import set_config
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sksurv.datasets import load_breast_cancer
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import brier_score
from lifelines import CoxTimeVaryingFitter

import matplotlib.pyplot as plt
import matplotlib as mpl
import ptitprince as pt

from lifelines import CoxTimeVaryingFitter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_curve, f1_score, matthews_corrcoef, 
                           confusion_matrix, roc_auc_score)
from lifelines.utils import concordance_index
import numpy as np
import pandas as pd
from itertools import product
from datetime import datetime

from scipy import stats

# Set global font weight to bold
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'  # For axis labels
mpl.rcParams['axes.titleweight'] = 'bold'  # For plot titles

def parse_args():
    predictors_l = ['demographics', 'ptau217', 'demographics_ptau217', 'demographics_ptau217_no_apoe', 
              'demographics_no_apoe']
    
    parser = argparse.ArgumentParser(description='Run time-to-event analysis.')
    parser.add_argument('--predictor', type=str,  choices=predictors_l,
                        help='Predictor to use for analysis')
    parser.add_argument('--fold', type=int, help='Fold number')

    # retrieve predictors
    predictors = parser.parse_args().predictor
    if predictors not in predictors_l:
        raise ValueError(f'Predictor {predictors} not recognized.')
    fold = parser.parse_args().fold

    return predictors, fold

def get_X(df, predictor, model=None, time_vary=False, binary_outcome=False):
    if model is None:
        # raise an error
        raise ValueError('Model is not specified.')
    else:
        if binary_outcome:
            directory_path = f'../../results/A4/CDR_functional_impairment/{predictor}/{model}/'
        else:
            directory_path = f'../../results/A4/CDR_functional_impairment/time_to_event/{predictor}/{model}/'
    check_folder_existence(directory_path)
        
    if predictor == 'ptau217':
        X = df[['ORRESRAW', 'label']]#, 'time_to_event']]
    elif predictor == 'demographics':
        X = df_utils.pull_columns_by_prefix(df, ['AGEYR', 'EDCCNTU', 'SEX',
            'APOEGN', 'label']) # 'RACE', 'ETHNIC', , 'time_to_event'])
    elif predictor == 'demographics_no_apoe':
        X = df_utils.pull_columns_by_prefix(df, ['AGEYR', 'EDCCNTU', 'SEX',
             'label'])#, 'time_to_event']) 'RACE', 'ETHNIC',
    elif predictor == 'demographics_ptau217':
        X = df_utils.pull_columns_by_prefix(df, ['ORRESRAW', 'AGEYR', 'EDCCNTU', 'SEX',
             'APOEGN', 'label'])#, 'time_to_event']) 'RACE', 'ETHNIC',
    elif predictor == 'demographics_ptau217_no_apoe':
        X = df_utils.pull_columns_by_prefix(df, ['ORRESRAW', 'AGEYR', 'EDCCNTU', 'SEX',
             'label'])#, 'time_to_event']) 'RACE', 'ETHNIC',
    else:
        raise ValueError(f'Predictor {predictor} not recognized.')
    
    if time_vary:
        X = pd.concat([df[['BID', 'start', 'stop']], X], axis=1)

    print(X.columns)
    
    return X, directory_path

def preprocess_cats(X):
    cat_cols_l = ['SEX', 'APOEGN'] # 'RACE', 'ETHNIC', 
    cat_cols = [col for col in X.columns if col in cat_cols_l]
    
    if len(cat_cols) > 0:
        encoder = OneHotEncoder(sparse_output=False)
        encoder.fit(X.loc[:, cat_cols])
        X_cat = encoder.transform(X.loc[:, cat_cols])
        X_cat = pd.DataFrame(
            X_cat, 
            columns=encoder.get_feature_names_out(cat_cols)
        )
        
        X = X.drop(columns=cat_cols)
        X = pd.concat([X, X_cat], axis=1)
    
    return X

def preprocess_xtrain_xtest(X_train, X_test, time_vary=False, binary_outcome=False):
    
    if binary_outcome:
        y_train = X_train['label']
        y_test = X_test['label']
    else:
        # Convert the DataFrame to a NumPy structured array
        dtype = [('label', 'bool'), ('stop', 'int16')]  # U10 for strings of length up to 10 characters
        y_train = np.array(list(X_train.loc[:, ['label', 'stop']].itertuples(index=False, name=None)), dtype=dtype)
        y_test = np.array(list(X_test.loc[:, ['label', 'stop']].itertuples(index=False, name=None)), dtype=dtype)

    if time_vary == False:
        X_train = X_train.drop(columns=['label', 'stop'])
        X_test = X_test.drop(columns=['label', 'stop'])

    if 'ORRESRAW' in X_train.columns:
        # # zscore age and education using StandardScaler
        # scaler = StandardScaler()
        # scaler.fit(X_train['ORRESRAW'].values.reshape(-1, 1))
        # X_train['ORRESRAW'] = scaler.transform(X_train['ORRESRAW'].values.reshape(-1, 1))
        # X_test['ORRESRAW'] = scaler.transform(X_test['ORRESRAW'].values.reshape(-1, 1))
        # boxcox transform ORRES
        X_train['ORRES_boxcox'], lambda_val = stats.boxcox(X_train.ORRES)
        X_test['ORRES_boxcox'] = stats.boxcox(X_test.ORRES, lmbda = lambda_val)

        # drop ORRESRAW 
        X_train = X_train.drop(columns=['ORRESRAW'])
        X_test = X_test.drop(columns=['ORRESRAW'])
        
    if 'AGEYR' in X_train.columns:
        
        # zscore age and education using StandardScaler
        scaler = StandardScaler()
        scaler.fit(X_train['AGEYR'].values.reshape(-1, 1))
        X_train['AGEYR'] = scaler.transform(X_train['AGEYR'].values.reshape(-1, 1))
        X_test['AGEYR'] = scaler.transform(X_test['AGEYR'].values.reshape(-1, 1))
        
        # add quadratic and cubic age, and interactions with apoe for age, quadratic age, and cubic age
        X_train['AGEYR2'] = X_train['AGEYR'] ** 2
        X_train['AGEYR3'] = X_train['AGEYR'] ** 3
        
        X_test['AGEYR2'] = X_test['AGEYR'] ** 2
        X_test['AGEYR3'] = X_test['AGEYR'] ** 3

        if len(df_utils.pull_columns_by_prefix(X_train, ['APOEGN']).columns.to_list()) > 0:
            # Create interaction variables
            for col in df_utils.pull_columns_by_prefix(X_train, ['APOEGN']).columns.to_list():
                for age_col in ['AGEYR', 'AGEYR2', 'AGEYR3']:
                    X_train[f'interaction_{col}_{age_col}'] = X_train[col] * X_train[age_col]
                    X_test[f'interaction_{col}_{age_col}'] = X_test[col] * X_test[age_col]
        
        # add interaction of age and education
        if 'EDCCNTU' in X_train.columns:
            scaler = StandardScaler()
            scaler.fit(X_train['EDCCNTU'].values.reshape(-1, 1))
            X_train['EDCCNTU'] = scaler.transform(X_train['EDCCNTU'].values.reshape(-1, 1))
            X_test['EDCCNTU'] = scaler.transform(X_test['EDCCNTU'].values.reshape(-1, 1))
        
            X_train['interaction_AGEYR_EDCCNTU'] = X_train['AGEYR'] * X_train['EDCCNTU']
            X_train['interaction_AGEYR2_EDCCNTU'] = X_train['AGEYR2'] * X_train['EDCCNTU']
            X_train['interaction_AGEYR3_EDCCNTU'] = X_train['AGEYR3'] * X_train['EDCCNTU']
            
            X_test['interaction_AGEYR_EDCCNTU'] = X_test['AGEYR'] * X_test['EDCCNTU']
            X_test['interaction_AGEYR2_EDCCNTU'] = X_test['AGEYR2'] * X_test['EDCCNTU']
            X_test['interaction_AGEYR3_EDCCNTU'] = X_test['AGEYR3'] * X_test['EDCCNTU']
            
    return X_train, y_train, X_test, y_test

# def get_prediction_times(X, y):
#     skf = StratifiedKFold(n_splits=10)
#     starts = []
#     ends = []
#     all_times = []
#     for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
#         X_train = X.loc[train_index].reset_index(drop=True)
#         X_test = X.loc[test_index].reset_index(drop=True)

#         X_train, y_train, X_test, y_test = preprocess_xtrain_xtest(X_train, X_test)

#         times = [x[1] for x in y_test]
#         starts.append(min(times))
#         ends.append(max(times))
#         all_times.extend(times)
#     time_range = sorted(list(set(all_times)))
#     time_range = [t for t in time_range if t >= max(starts) and t < min(ends)]
#     return time_range

def save_metrics(directory_path, model, times, X_train, y_train, X_test, y_test, fold):
    # times = [x[1] for x in y_test]
        
    # # lower, upper = min(times), max(times)
    # lower, upper = np.percentile(times, [5, 95])
    # times = np.arange(lower, upper)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_prob = np.vstack([fn(times) for fn in model.predict_survival_function(X_train)])
    test_prob = np.vstack([fn(times) for fn in model.predict_survival_function(X_test)])

    train_brier = brier_score(y_train, y_train, train_prob, times)           
    test_brier = brier_score(y_train, y_test, test_prob, times)

    train_int_brier = integrated_brier_score(y_train, y_train, train_prob, times)           
    test_int_brier = integrated_brier_score(y_train, y_test, test_prob, times)

    train_auc = cumulative_dynamic_auc(y_train, y_train, train_pred, times)
    test_auc = cumulative_dynamic_auc(y_train, y_test, test_pred, times)

    train_ci_ipcw = concordance_index_ipcw(y_train, y_train, train_pred, tau=times[-1])
    test_ci_ipcw = concordance_index_ipcw(y_train, y_test, test_pred, tau=times[-1])

    train_ci_harrell = concordance_index_censored([x[0] for x in y_train], [x[1] for x in y_train], train_pred)
    test_ci_harrell = concordance_index_censored([x[0] for x in y_test], [x[1] for x in y_test], test_pred)
    
    print('Saving results...')
    save_labels_probas(directory_path, y_train, train_pred, y_test, test_pred,
                        other_file_info=f'_fold_{fold}', survival=True,
                        surv_model=model, train_surv_fn=train_prob, test_surv_fn=test_prob)
    save_pickle(f'{directory_path}/times_{fold}.pkl', times)
    save_pickle(f'{directory_path}/train_brier_{fold}.pkl', train_brier)
    save_pickle(f'{directory_path}/test_brier_{fold}.pkl', test_brier)
    save_pickle(f'{directory_path}/train_int_brier_{fold}.pkl', train_int_brier)
    save_pickle(f'{directory_path}/test_int_brier_{fold}.pkl', test_int_brier)
    save_pickle(f'{directory_path}/train_ci_ipcw_{fold}.pkl', train_ci_ipcw)
    save_pickle(f'{directory_path}/test_ci_ipcw_{fold}.pkl', test_ci_ipcw)
    save_pickle(f'{directory_path}/train_ci_harrell_{fold}.pkl', train_ci_harrell)
    save_pickle(f'{directory_path}/test_ci_harrell_{fold}.pkl', test_ci_harrell)
    save_pickle(f'{directory_path}/train_auc_{fold}.pkl', train_auc)
    save_pickle(f'{directory_path}/test_auc_{fold}.pkl', test_auc)

def calculate_survival_probability(ctv, X, time_point):
    """
    Calculate survival probability at a specific time point
    
    Parameters:
    -----------
    ctv : CoxTimeVaryingFitter
        Fitted Cox model
    X : DataFrame
        Design matrix
    time_point : float
        Time point to calculate survival probability
    
    Returns:
    --------
    risk_scores : array
        Risk scores (1 - survival probability) for each subject
    """
    # Calculate baseline cumulative hazard
    baseline_cumulative_hazard = ctv.baseline_cumulative_hazard_
    
    # Get the partial hazards for the given design matrix
    partial_hazards = ctv.predict_partial_hazard(X)
    
    # Find the cumulative hazard at the specified time point
    cumulative_hazard_at_time = baseline_cumulative_hazard.loc[
        baseline_cumulative_hazard.index <= time_point
    ].max()
    
    # Calculate survival probabilities
    survival_probs = np.exp(-cumulative_hazard_at_time.values * partial_hazards.values)
    
    # Return risk scores (1 - survival probability)
    risk_scores = 1 - pd.Series(np.array(survival_probs).flatten(), index=X.index)
    return risk_scores

def find_two_thresholds_with_constraints(y_true, y_scores, sensitivity_threshold, specificity_threshold):
    """
    Find two thresholds that achieve desired sensitivity and specificity while minimizing intermediates.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_scores : array-like
        Predicted probabilities or scores
    sensitivity_threshold : float
        Minimum required sensitivity (e.g., 0.90 for 90%)
    specificity_threshold : float
        Minimum required specificity (e.g., 0.90 for 90%)
    
    Returns:
    --------
    tuple : (lower_threshold, upper_threshold) or None if no valid thresholds found
    """
    thresholds = np.linspace(0, 1, 200)  # More granular threshold search
    best_thresholds = None
    min_intermediate_prop = float('inf')
    
    for lower_idx, lower_t in enumerate(thresholds):
        for upper_t in thresholds[lower_idx + 1:]:
            # Classify predictions
            pos_pred = y_scores > upper_t
            neg_pred = y_scores < lower_t
            intermediate = (y_scores >= lower_t) & (y_scores <= upper_t)
            
            # Skip if too many intermediates
            intermediate_prop = np.mean(intermediate)
            if intermediate_prop > 0.50:  # 30% threshold for intermediates
                continue
            
            # Calculate sensitivity and specificity
            pos_true = y_true == 1
            neg_true = y_true == 0
            
            sensitivity = np.sum(pos_pred & pos_true) / np.sum(pos_true)
            specificity = np.sum(neg_pred & neg_true) / np.sum(neg_true)
            
            # Check if meets sensitivity/specificity constraints
            if sensitivity >= sensitivity_threshold and specificity >= specificity_threshold:
                if intermediate_prop < min_intermediate_prop:
                    min_intermediate_prop = intermediate_prop
                    best_thresholds = (lower_t, upper_t)
    
    if best_thresholds is None:
        raise ValueError(f"No thresholds found meeting constraints: sens>={sensitivity_threshold}, spec>={specificity_threshold}")
    
    return best_thresholds

def calculate_time_dependent_metrics(X_train, X_test, y_train, y_test, ctv):
    """Calculate time-dependent metrics for Cox model."""
    # Get unique time points in the baseline_cumulative_hazard
    baseline_time_points = ctv.baseline_cumulative_hazard_.index

    # # Filter time points to only include those where events occurred
    # all_time_points = np.unique(np.concatenate([
    #     X_train[X_train['label'] == 1]['stop'].unique(), 
    #     X_test[X_test['label'] == 1]['stop'].unique()
    # ]))
    # all_time_points = all_time_points[
    #     (all_time_points >= baseline_time_points.min()) & 
    #     (all_time_points <= baseline_time_points.max())
    # ]

    # For time points, round baseline_time_points.min() up to nearest integer, and round baseline_time_points.max() down to nearest integer
    all_time_points = np.arange(int(np.ceil(baseline_time_points.min())), int(np.floor(baseline_time_points.max())) + 1)

    # Initialize results
    results_list = []
    
    for i, time_point in enumerate(all_time_points):
        print(f"Processing time point {i} of {len(all_time_points)}. Time: {datetime.now()}")

        # Create time-specific labels
        train_labels = (y_train['stop'] <= time_point) & (y_train['label'] == 1)
        test_labels = (y_test['stop'] <= time_point) & (y_test['label'] == 1)

        # Calculate risk scores
        risk_scores_at_time = calculate_survival_probability(ctv, X_test, time_point)
        train_risk_scores_at_time = calculate_survival_probability(ctv, X_train, time_point)
        
        # Threshold finding methods
        def find_optimal_threshold_youdens_j(y_true, y_scores):
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            j_statistic = tpr - fpr
            optimal_idx = np.argmax(j_statistic)
            return thresholds[optimal_idx]
        
        def find_optimal_threshold_f1(y_true, y_scores):
            thresholds = np.linspace(0, 1, 100)
            f1_scores = [f1_score(y_true, y_scores > threshold) for threshold in thresholds]
            return thresholds[np.argmax(f1_scores)]
        
        def find_optimal_threshold_mcc(y_true, y_scores):
            thresholds = np.linspace(0, 1, 100)
            mcc_scores = [matthews_corrcoef(y_true, y_scores > threshold) for threshold in thresholds]
            return thresholds[np.argmax(mcc_scores)]

        threshold_methods = {
            'Youden_J': find_optimal_threshold_youdens_j,
            'Max_F1': find_optimal_threshold_f1,
            'Max_MCC': find_optimal_threshold_mcc
        }

        for method_name, threshold_func in threshold_methods.items():
            try:
                optimal_threshold = threshold_func(train_labels, train_risk_scores_at_time)
                binary_predictions = (risk_scores_at_time > optimal_threshold).astype(int)
                
                tn, fp, fn, tp = confusion_matrix(test_labels, binary_predictions).ravel()
                
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                f1 = f1_score(test_labels, binary_predictions)
                mcc = matthews_corrcoef(test_labels, binary_predictions)
                balanced_accuracy = (sensitivity + specificity) / 2
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                
                auc = roc_auc_score(test_labels, risk_scores_at_time)
                c_index = concordance_index(test_labels, risk_scores_at_time)
                brier_score = np.mean((risk_scores_at_time - test_labels)**2)
                
                result_dict = {
                    'time_point': time_point,
                    'method': method_name,
                    'threshold': optimal_threshold,
                    'lower_threshold': None,  # Not applicable for single threshold
                    'upper_threshold': None,  # Not applicable for single threshold
                    'intermediate_proportion': None,  # Not applicable for single threshold
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'f1_score': f1,
                    'mcc': mcc,
                    'auc': auc,
                    'c_index': c_index,
                    'brier_score': brier_score,
                    'balanced_accuracy': balanced_accuracy,
                    'ppv': ppv,
                    'npv': npv
                }

                results_list.append(result_dict)
                
            except Exception as e:
                print(f"Error at time point {time_point}, method {method_name}: {str(e)}")
                continue
    
        # Two-threshold approach
        # Define constraint levels
        constraint_levels = [0.5, 0.6, 0.7, 0.8]

        constraint_failed = False
        for constraint in constraint_levels:
            if constraint_failed:
                print(f"Skipping constraint {constraint*100}% as lower constraint already failed")
                continue
            try:
                method_name = f'Two_Threshold_{int(constraint*100)}'
                lower_t, upper_t = find_two_thresholds_with_constraints(
                    train_labels, 
                    train_risk_scores_at_time,
                    sensitivity_threshold=constraint,
                    specificity_threshold=constraint
                )
                
                # Calculate metrics for test set using two thresholds
                pos_pred = risk_scores_at_time > upper_t
                neg_pred = risk_scores_at_time < lower_t
                intermediate = (risk_scores_at_time >= lower_t) & (risk_scores_at_time <= upper_t)
                
                pos_true = test_labels == 1
                neg_true = test_labels == 0
                
                # Calculate metrics
                sensitivity = np.sum(pos_pred & pos_true) / np.sum(pos_true)
                specificity = np.sum(neg_pred & neg_true) / np.sum(neg_true)
                intermediate_prop = np.mean(intermediate)
                
                # Calculate PPV and NPV
                ppv = np.sum(pos_pred & pos_true) / np.sum(pos_pred) if np.sum(pos_pred) > 0 else None
                npv = np.sum(neg_pred & neg_true) / np.sum(neg_pred) if np.sum(neg_pred) > 0 else None
                
                result_dict = {
                    'time_point': time_point,
                    'method': method_name,
                    'threshold': None,
                    'lower_threshold': lower_t,
                    'upper_threshold': upper_t,
                    'intermediate_proportion': intermediate_prop,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'f1_score': None,
                    'mcc': None,
                    'auc': None,
                    'c_index': None,
                    'brier_score': None,
                    'balanced_accuracy': None,
                    'ppv': ppv,
                    'npv': npv
                }
                
                results_list.append(result_dict)
                
            except ValueError as e:
                print(f"Error at time point {time_point} for {method_name}: {str(e)}")
                constraint_failed = True  # Set flag to skip higher constraints
                continue
            except Exception as e:
                print(f"Unexpected error at time point {time_point} for {method_name}: {str(e)}")
                continue

    return pd.DataFrame(results_list)

def get_model_coefficients(ctv_model):
    """
    Extract coefficients and their statistics from a fitted CoxTimeVaryingFitter model.
    
    Parameters:
    -----------
    ctv_model : CoxTimeVaryingFitter
        A fitted Cox time-varying model
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing coefficients and their statistics
    """
    # Get the summary dataframe
    coef_df = ctv_model.summary
    
    # Add absolute coefficient values for importance ranking
    coef_df['abs_coef'] = abs(coef_df['coef'])
    
    # Sort by absolute coefficient value to see most influential features
    coef_df_sorted = coef_df.sort_values('abs_coef', ascending=False)
    
    # Calculate hazard ratio and confidence intervals
    coef_df_sorted['hazard_ratio'] = np.exp(coef_df_sorted['coef'])
    coef_df_sorted['hr_lower_ci'] = np.exp(coef_df_sorted['coef'] - 1.96 * coef_df_sorted['se(coef)'])
    coef_df_sorted['hr_upper_ci'] = np.exp(coef_df_sorted['coef'] + 1.96 * coef_df_sorted['se(coef)'])
    
    return coef_df_sorted

def inner_cross_validation(directory_path, fold_num, X_train_outer, y_train_outer, param_grid, n_inner_splits=5):
    """Perform inner cross-validation for hyperparameter optimization."""
    unique_train_bids = X_train_outer['BID'].unique()
    y_train_outer_grouped = X_train_outer.groupby('BID')['label'].max().reset_index()
    
    results_list = []
    
    for penalizer, l1_ratio in product(param_grid['penalizer'], param_grid['l1_ratio']):
        print(f"Testing penalizer={penalizer}, l1_ratio={l1_ratio}")
        
        inner_cv = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(inner_cv.split(unique_train_bids, y_train_outer_grouped['label'])):
            print(f"Processing inner fold {fold + 1} of {n_inner_splits}. Time: {datetime.now()}")
            
            train_bids = unique_train_bids[train_idx]
            val_bids = unique_train_bids[val_idx]
            
            X_train_inner = X_train_outer[X_train_outer['BID'].isin(train_bids)].reset_index(drop=True)
            X_val_inner = X_train_outer[X_train_outer['BID'].isin(val_bids)].reset_index(drop=True)
            
            X_train_inner, y_train_inner, X_val_inner, y_val_inner = preprocess_xtrain_xtest(
                X_train_inner, X_val_inner, time_vary=True
            )
            
            try:
                ctv = CoxTimeVaryingFitter(penalizer=penalizer, l1_ratio=l1_ratio)
                ctv.fit(X_train_inner, id_col="BID", start_col="start", 
                       stop_col="stop", event_col="label", show_progress=True,
                       fit_options={"max_steps": 1000})
                
                metrics_df = calculate_time_dependent_metrics(
                    X_train_inner, X_val_inner, y_train_inner, y_val_inner, ctv
                )
                
                # Add parameters and fold information
                metrics_df['fold'] = fold
                metrics_df['penalizer'] = penalizer
                metrics_df['l1_ratio'] = l1_ratio
                
                results_list.append(metrics_df)
                
            except Exception as e:
                print(f"Error in fold {fold}: {str(e)}")
                continue
    
    # Combine all results
    if not results_list:
        raise ValueError("No valid results obtained from inner cross-validation")
    
    all_results = pd.concat(results_list, ignore_index=True)
    
    # save all_results
    all_results.to_parquet(f'{directory_path}/training_all_results_fold_{fold_num}.parquet')

    # Calculate average performance for each parameter combination
    avg_results = (all_results.groupby(['penalizer', 'l1_ratio', 'method'])
                             .agg({
                                 'c_index': 'mean',
                                 'auc': 'mean',
                                 'brier_score': 'mean'
                             })
                             .reset_index())
    
    # save avg_results
    avg_results.to_parquet(f'{directory_path}/training_avg_results_fold_{fold_num}.parquet')

    # Find best parameters (using c-index as primary metric)
    best_params_idx = avg_results.groupby(['penalizer', 'l1_ratio'])['c_index'].mean().idxmax()
    # best_params = pd.Series(best_params_idx, name='value').to_dict()
    best_params = {
        'penalizer': best_params_idx[0],
        'l1_ratio': best_params_idx[1]
    }
    
    return {
        'best_params': best_params,
        'detailed_results': all_results,
        'averaged_results': avg_results
    }

def run_analysis(data, predictors, fold_num, param_grid):
    """Run complete analysis with nested cross-validation."""
    # full_results = []
    
    # for predictor in predictors:
    print(f"\nAnalyzing predictor: {predictors}")
    
    X, directory_path = get_X(df=data, predictor=predictors, model='coxtimevary', time_vary=True)
    X = preprocess_cats(X)
    if 'APOEGN_None' in X.columns:
        X = X.drop(columns=['APOEGN_None'])
    # if 'APOEGN_E2/E2' in X.columns:
    #     X = X.drop(columns=['APOEGN_E2/E2'])
    # if 'RACE_84' in X.columns:
    #     X = X.drop(columns=['RACE_84'])
    # if 'RACE_100' in X.columns:
    #     X = X.drop(columns=['RACE_100'])
    
    predictor_results = []
    unique_bids = X['BID'].unique()
    y = X.groupby('BID')['label'].max().reset_index()
    
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=2345678)
    
    for fold, (train_bids_index, test_bids_index) in enumerate(outer_cv.split(unique_bids, y['label'])):
        if fold != fold_num:
            continue
        print(f"Processing outer fold {fold + 1}/{10}")
        
        train_bids = unique_bids[train_bids_index]
        test_bids = unique_bids[test_bids_index]
        
        X_train_outer = X[X['BID'].isin(train_bids)].reset_index(drop=True)
        X_test = X[X['BID'].isin(test_bids)].reset_index(drop=True)
        
        # Inner cross-validation
        cv_results = inner_cross_validation(directory_path, fold, X_train_outer, y, param_grid)
        print(cv_results)
        best_params = cv_results['best_params']
        print(best_params)

        # Train final model with best parameters
        X_train_outer, y_train, X_test, y_test = preprocess_xtrain_xtest(
            X_train_outer, X_test, time_vary=True
        )
        
        ctv = CoxTimeVaryingFitter(
            penalizer=best_params['penalizer'],
            l1_ratio=best_params['l1_ratio']
        )
        ctv.fit(X_train_outer, id_col="BID", start_col="start", 
                stop_col="stop", event_col="label", show_progress=True)
        
        # Get feature importances for this fold
        coef_df = get_model_coefficients(ctv)
        coef_df['fold'] = fold
        coef_df['predictor'] = predictors
        coef_df.to_parquet(f'{directory_path}/training_coefficients_fold_{fold}.parquet')
    
        # Calculate test metrics
        test_metrics = calculate_time_dependent_metrics(
            X_train_outer, X_test, y_train, y_test, ctv
        )
        
        # Add fold information
        test_metrics['outer_fold'] = fold
        test_metrics['predictor'] = predictors
        test_metrics['best_penalizer'] = best_params['penalizer']
        test_metrics['best_l1_ratio'] = best_params['l1_ratio']
        
        predictor_results.append(test_metrics)
    
    # Combine results for this predictor
    predictor_df = pd.concat(predictor_results, ignore_index=True)
    
    # Calculate summary statistics
    summary_stats = (predictor_df.groupby(['predictor', 'method'])
                                .agg({
                                    'c_index': ['mean', 'std'],
                                    'auc': ['mean', 'std'],
                                    'brier_score': ['mean', 'std']
                                })
                                .reset_index())
    
    full_results = {
        'predictor': predictors,
        'detailed_results': predictor_df,
        'summary_stats': summary_stats
    }
    
    # save full_results
    predictor_df.to_parquet(f'{directory_path}/test_detailed_results_fold_{fold_num}.parquet')
    summary_stats.to_parquet(f'{directory_path}/test_summary_stats_fold_{fold_num}.parquet')

   

def main():
    param_grid = {
        'penalizer': [0.1, 1, 10],
        'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0]
    }

    # predictors, fold = parse_args()
    predictors, fold = 'demographics', 0
    data = pd.read_parquet('../../tidy_data/A4/ptau217_allvisits.parquet')
    run_analysis(data, predictors, fold, param_grid)

if __name__ == '__main__':
    main()
