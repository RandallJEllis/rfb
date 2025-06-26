import pandas as pd
import argparse
import os
import numpy as np
import logging
import joblib
from flaml import AutoML
import sys
sys.path.append('../ukb_func')
from ml_utils import encode_categorical_vars
from df_utils import pull_columns_by_prefix

parser = argparse.ArgumentParser()
parser.add_argument('--pacc_col', type=str, default='PACC.raw')
parser.add_argument('--visit', type=int)
args = parser.parse_args()
pacc_col = args.pacc_col
visit = args.visit

ptau = '../../tidy_data/A4/ptau_base.parquet'
ptau = pd.read_parquet(ptau)

# keep first for for each id
ptau = ptau.groupby('id').first().reset_index()

pacc = pd.read_csv(f'../../raw_data/A4_oct302024/clinical/Derived Data/PACC.csv')
pacc = pacc[pacc.BID.isin(ptau.id)]
visits = np.sort(pacc.VISCODE.unique())[1:]

lancet_load_path = '../../tidy_data/A4'
habits = pd.read_parquet(f'{lancet_load_path}/habits.parquet')
psychwell = pd.read_parquet(f'{lancet_load_path}/psychwell.parquet')
vitals = pd.read_parquet(f'{lancet_load_path}/vitals.parquet')
centiloids = pd.read_parquet(f'{lancet_load_path}/centiloids.parquet')

results_main_path = f'../../results/A4/PACC/'

# encode sex, ethnicity, APOEe4 alleles, education qualifications
catcols = [
        'SEX',
        'APOEGN',
        "SUBUSE",
        ]

# for v in visits:
pacc_v = pacc[pacc.VISCODE == visit]

# drop rows where pacc_col is missing
pacc_v = pacc_v.dropna(subset=[pacc_col])

for fold in range(5):
    train_df = pd.read_parquet(f'../../tidy_data/A4/train_{fold}.parquet')
    test_df = pd.read_parquet(f'../../tidy_data/A4/val_{fold}.parquet')

    train_df = train_df[train_df.id.isin(pacc_v.BID)]
    train_df = train_df.groupby('id').first().reset_index()
    train_df = train_df.drop(columns=['age_z', 'age_centered', 'age_z_squared', 'age_z_cubed',
                                        'age_centered_squared', 'age_centered_cubed', 'educ_z', 'ptau_z',
                                        'ptau_boxcox'])
    test_df = test_df[test_df.id.isin(pacc_v.BID)]
    test_df = test_df.groupby('id').first().reset_index()
    test_df = test_df.drop(columns=['age_z', 'age_centered', 'age_z_squared', 'age_z_cubed',
                                        'age_centered_squared', 'age_centered_cubed', 'educ_z', 'ptau_z',
                                        'ptau_boxcox'])

    train_df = train_df.merge(pacc_v.loc[:, ['BID', pacc_col]], left_on='id', right_on='BID', how='inner').drop(columns=['BID'])
    test_df = test_df.merge(pacc_v.loc[:, ['BID', pacc_col]], left_on='id', right_on='BID', how='inner').drop(columns=['BID'])

    train_df['ds'] = 'train'
    test_df['ds'] = 'test'
    df = pd.concat([train_df, test_df])

    habits_df = habits[habits.BID.isin(df.id)]
    habits_df = habits_df.groupby('BID').first().reset_index()
    psychwell_df = psychwell[psychwell.BID.isin(df.id)]
    psychwell_df = psychwell_df.groupby('BID').first().reset_index()
    vitals_df = vitals[vitals.BID.isin(df.id)]
    vitals_df = vitals_df.groupby('BID').first().reset_index()
    centiloids_df = centiloids[centiloids.BID.isin(df.id)]
    centiloids_df = centiloids_df.groupby('BID').first().reset_index()

    df = df.merge(habits_df.loc[:, ['BID', "SMOKE", "ALCOHOL", "SUBUSE", "AEROBIC", "WALKING"]], left_on='id', right_on='BID', how='left').drop(columns=['BID'])
    df = df.merge(psychwell_df.loc[:, ['BID', "GDTOTAL", "STAITOTAL"]], left_on='id', right_on='BID', how='left').drop(columns=['BID'])
    df = df.merge(vitals_df.loc[:, ['BID', "VSBPSYS", "VSBPDIA", "BMI"]], left_on='id', right_on='BID', how='left').drop(columns=['BID'])
    df = df.merge(centiloids_df.loc[:, ['BID', "AMYLCENT"]], left_on='id', right_on='BID', how='left').drop(columns=['BID'])



    categ_enc = encode_categorical_vars(df, catcols)
    df = df.drop(columns=catcols).join(categ_enc)

    # create age**2
    df['age_squared'] = df.age ** 2
    
    # create age*APOE interactions
    apoe_cols = [c for c in df.columns if 'APOE' in c]
    for apoe_col in apoe_cols:
        df[f'interaction_age_{apoe_col}'] = df.age * df[apoe_col]
        df[f'interaction_age_squared_{apoe_col}'] = df.age_squared * df[apoe_col]

    if fold == 0:
        feature_sets = {
            'demographics': pull_columns_by_prefix(df, [f'APOE', 'interaction', 'age', 'educ', 'SEX']).columns.tolist(),
            'demographics_no_apoe': pull_columns_by_prefix(df, [f'age', 'educ', 'SEX']).columns.tolist(),
            "lancet": pull_columns_by_prefix(df, [f'SMOKE', 'ALCOHOL', 'SUBUSE', 'AEROBIC', 'WALKING', 'GDTOTAL', 'STAITOTAL', 'VSBPSYS', 'VSBPDIA', 'BMI']).columns.tolist(),
            'demographics_lancet': pull_columns_by_prefix(df, [f'APOE', 'interaction', 'age', 'educ', 'SEX',
                                                                'SMOKE', 'ALCOHOL', 'SUBUSE', 'AEROBIC', 'WALKING',
                                                                'GDTOTAL', 'STAITOTAL', 'VSBPSYS', 'VSBPDIA', 'BMI']).columns.tolist(),
            'demographics_lancet_no_apoe': pull_columns_by_prefix(df, [f'age', 'educ', 'SEX',
                                                                        'SMOKE', 'ALCOHOL', 'SUBUSE', 'AEROBIC', 'WALKING',
                                                                        'GDTOTAL', 'STAITOTAL', 'VSBPSYS', 'VSBPDIA', 'BMI']).columns.tolist(),

            "ptau": pull_columns_by_prefix(df, [f'ptau']).columns.tolist(),
            "ptau_demographics_no_apoe": pull_columns_by_prefix(df, [f'ptau', 'age', 'educ', 'SEX']).columns.tolist(),
            "ptau_demographics": pull_columns_by_prefix(df, [f'ptau', 'interaction', 'age', 'educ', 'SEX', 'APOE']).columns.tolist(),
            'ptau_demographics_lancet': pull_columns_by_prefix(df, [f'ptau', 'interaction', 'age', 'educ', 'SEX', 'APOE',
                                                                        'SMOKE', 'ALCOHOL', 'SUBUSE', 'AEROBIC', 'WALKING',
                                                                        'GDTOTAL', 'STAITOTAL', 'VSBPSYS', 'VSBPDIA', 'BMI']).columns.tolist(),
            'ptau_demographics_lancet_no_apoe': pull_columns_by_prefix(df, [f'ptau', 'age', 'educ', 'SEX',
                                                                            'SMOKE', 'ALCOHOL', 'SUBUSE', 'AEROBIC', 'WALKING',
                                                                            'GDTOTAL', 'STAITOTAL', 'VSBPSYS', 'VSBPDIA', 'BMI']).columns.tolist(),

            "centiloids": pull_columns_by_prefix(df, [f'AMYLCENT']).columns.tolist(),
            "centiloids_demographics_no_apoe": pull_columns_by_prefix(df, [f'AMYLCENT', 'age', 'educ', 'SEX']).columns.tolist(),
            "centiloids_demographics": pull_columns_by_prefix(df, [f'AMYLCENT', 'interaction', 'age', 'educ', 'SEX', 'APOE']).columns.tolist(),
            'centiloids_demographics_lancet': pull_columns_by_prefix(df, [f'AMYLCENT', 'interaction', 'age', 'educ', 'SEX', 'APOE',
                                                                            'SMOKE', 'ALCOHOL', 'SUBUSE', 'AEROBIC', 'WALKING',
                                                                            'GDTOTAL', 'STAITOTAL', 'VSBPSYS', 'VSBPDIA', 'BMI']).columns.tolist(),
            'centiloids_demographics_lancet_no_apoe': pull_columns_by_prefix(df, [f'AMYLCENT', 'age', 'educ', 'SEX',
                                                                                    'SMOKE', 'ALCOHOL', 'SUBUSE', 'AEROBIC', 'WALKING',
                                                                                    'GDTOTAL', 'STAITOTAL', 'VSBPSYS', 'VSBPDIA', 'BMI']).columns.tolist(),
            "ptau_centiloids": pull_columns_by_prefix(df, [f'ptau', 'AMYLCENT']).columns.tolist(),
            "ptau_centiloids_demographics_no_apoe": pull_columns_by_prefix(df, [f'ptau', 'AMYLCENT', 'age', 'educ', 'SEX']).columns.tolist(),
            "ptau_centiloids_demographics": pull_columns_by_prefix(df, [f'ptau', 'AMYLCENT', 'interaction', 'age', 'educ', 'SEX', 'APOE']).columns.tolist(),
            "ptau_centiloids_demographics_lancet": pull_columns_by_prefix(df, [f'ptau', 'AMYLCENT', 'interaction', 'age', 'educ', 'SEX', 'APOE',
                                                                                'SMOKE', 'ALCOHOL', 'SUBUSE', 'AEROBIC', 'WALKING',
                                                                                'GDTOTAL', 'STAITOTAL', 'VSBPSYS', 'VSBPDIA', 'BMI']).columns.tolist(),
            "ptau_centiloids_demographics_lancet_no_apoe": pull_columns_by_prefix(df, [f'ptau', 'AMYLCENT', 'age', 'educ', 'SEX',
                                                                                        'SMOKE', 'ALCOHOL', 'SUBUSE', 'AEROBIC', 'WALKING',
                                                                                        'GDTOTAL', 'STAITOTAL', 'VSBPSYS', 'VSBPDIA', 'BMI']).columns.tolist(),
        }

    for model_type, features in feature_sets.items():
        print('Visit: ', visit)
        print('Fold: ', fold)
        print('Model type: ', model_type)
        print('Features: ', features)

        results_dir = f'{results_main_path}/outcome_{pacc_col}/visit_{visit}/{model_type}/fold_{fold}'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # set up logging
        logging.basicConfig(filename=f'{results_dir}/logging.txt', level=logging.INFO)
        logging.info(f"Running experiment with visit {visit} and fold {fold}")
        logging.info(f"Model type: {model_type}")
        logging.info(f"Features: {features}")

        try:
            train_df = df[df.ds == 'train'].drop(columns=['ds'])
            test_df = df[df.ds == 'test'].drop(columns=['ds'])

            y_train = train_df.pop(pacc_col).values
            y_test = test_df.pop(pacc_col).values
            X_train = train_df.loc[:, features].values
            X_test = test_df.loc[:, features].values

            logging.info(f"Training the model")
            automl_settings = {
                "task": "regression",
                "time_budget": 10,
                "metric": "mse",
                "n_jobs": -1,
                "eval_method": 'cv',
                "n_splits": 5,
                "early_stop": True,
                "log_training_metric": True,
                "model_history": True,
                "seed": 1234321,
                "estimator_list": ['lgbm'],
                "log_file_name": f'{results_dir}/results_log.json'
                }
            
            logging.info(f"Fitting the model")
            automl = AutoML()
            automl.fit(X_train, y_train, **automl_settings)

            logging.info(f"Saving the model")
            # save the model
            best_model = automl.model.estimator

            # Save just the best model
            joblib.dump(best_model, f'{results_dir}/flaml_best_model.joblib')

            logging.info(f"Saving the predictions")
            # save the test set predictions
            y_pred = automl.predict(X_test)
            results = pd.DataFrame({
                'y_test': y_test,
                'y_pred': y_pred
            })
            results.to_parquet(f'{results_dir}/test_labels_predictions.parquet', index=False)

            # save the train set predictions
            y_pred = automl.predict(X_train)
            results = pd.DataFrame({
                'y_train': y_train,
                'y_pred': y_pred
            })
            results.to_parquet(f'{results_dir}/train_labels_predictions.parquet', index=False)
            
            logging.info(f"Finished fold {fold}")

        except Exception as e:
            logging.error(f"Error occurred: {str(e)}", exc_info=True)  # Includes traceback

        finally:
            logging.info(f"Finished the experiment")