import sys
sys.path.append('../../../rfb/code/')

from bootstrap import run_bootstrap
# from f3 import f3_metric
import argparse
import os
import pandas as pd
import numpy as np
from flaml import AutoML

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder



def load_datasets(output_path):
    X = pd.read_parquet(f'{output_path}/X.parquet')
    y = np.load(f'{output_path}/y.npy')
    return X, y


def load_proteomics(data_path):
    df = pd.read_parquet(data_path + 'proteomics/proteomics.parquet')

    # only keep proteins from Instance 0
    cols_remove = []
    for c in df.columns:
        if '-1' in c:
            cols_remove.append(c)
        elif '-2' in c:
            cols_remove.append(c)
        elif '-3' in c:
            cols_remove.append(c)
    df = df.drop(columns=cols_remove)
    df = df.dropna(subset=df.columns[1:], how='all')

    return df


def load_demographics(data_path):
    # import age, sex, education, site, assessment date
    df = pd.read_parquet(data_path +
                                   'demographics/demographics_df.parquet'
                                   )
    return df

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description='Random feature baselines pipeline')

    # Add arguments
    # parser.add_argument('njobs', type=str, help='Number of cores')
    parser.add_argument('--task_id', type=int,
                        help='Task ID (i.e., njobs / number of outcomes per job)')
    parser.add_argument('--outcome', type=str,
                        help='Feature being predicted')
    parser.add_argument('--output_path', type=str,
                        help='Path to save outputs')
    parser.add_argument('--data_path', type=str,
                        help='Path to pull data')

    # Parse the arguments
    args = parser.parse_args()
    task_id = args.task_id
    outcome = args.outcome
    data_path = args.data_path
    output_path = args.output_path

    print(f'Running job number, {task_id}')
    print(f'Outcome, {outcome}')
    print(f'Data path, {data_path}')
    print(f'Output path, {output_path}')

    # load or create datasets
    if os.path.isfile(f'{output_path}/X.parquet') and \
       os.path.isfile(f'{output_path}/y.npy'):
       print('Loading datasets')
       X, y = load_datasets(output_path)

    else:
        print('Building datasets')
        prot = load_proteomics(data_path)
        demo = load_demographics(data_path)
        df = prot.merge(demo)

        if outcome in ['dementia', 'acd', 'ACD', "alz"]:
            # import dx dates across dementia diagnosis Field IDs
            acd = pd.read_parquet(data_path + 'acd/allcausedementia.parquet')

            # Strings to search for
            search_strings = ['42018', '42020', '42022', '42024', '131036',
                              '130836', '130838', '130840', '130842']

            # Identify column names that start with any of the search strings
            matching_columns = [col for col in acd.columns
                                if any(col.startswith(s)
                                       for s in search_strings)]

            # Filter rows where 1+ value in the subset of columns is not NA
            acd = acd[acd[matching_columns].notna().any(axis=1)]
            acd = acd.loc[:, ['eid'] + matching_columns]

            # remove patients diagnosed with dementia before proteomics
            for col in acd.columns[1:].tolist():
                acd[col] = pd.to_datetime(acd.loc[:, col])
            acd['first_dx'] = acd.iloc[:, 1:].min(axis=1)

            df['53-0.0'] = pd.to_datetime(df['53-0.0'])
            cases = df.merge(acd)

            cases['date_diff'] = cases.first_dx - cases['53-0.0']
            cases = cases[cases.date_diff >= pd.Timedelta(0)]

            controls = df[~df.eid.isin(acd.eid)]

            df = pd.concat([controls, cases])
            df['label'] = df['eid'].isin(acd.eid).astype(int)

            '''
            Apolipoprotein E (APOE) genotype has three major alleles
            (epsilon 2, epsilon 3, epsilon 4):
            epsilon2 - rs429358-T, rs7412-T
            epsilon3 - rs429358-T, rs7412-C
            epsilon4 - rs429358-C, rs7412-C

            These two single nucleotide polymorphisms (SNPs; rs429358, rs7412)
            were extracted from the UK Biobank genomics data using Plink2.

            The relevant columns of this output are:
            IID - The equivalent of eid in our other files
            rs429358_C - 1 means they have a C at the genomic locus;
            0 means they have a T
            rs7412_T - 1 means they have a T at the genomic locus;
            0 means they have a C

            We will encode these polymorphisms based on the two allele columns.
            '''

            alleles = pd.read_csv(
                f'{data_path}/apoe4_snps/plink_outputs/apoee4_snps.raw',
                sep='\t'
                )
            alleles['apoe_polymorphism'] = np.nan
            # alleles.loc[(alleles.rs429358_C == 0) &
            #             (alleles.rs7412_T == 0), 'apoe_polymorphism'] = 'e3/e3'
            # alleles.loc[(alleles.rs429358_C == 1) &
            #             (alleles.rs7412_T == 0), 'apoe_polymorphism'] = 'e3/e4'
            # alleles.loc[(alleles.rs429358_C == 0) &
            #             (alleles.rs7412_T == 1), 'apoe_polymorphism'] = 'e2/e3'
            # alleles.loc[(alleles.rs429358_C == 1) &
            #             (alleles.rs7412_T == 1), 'apoe_polymorphism'] = 'e2/e4'
            # alleles.loc[(alleles.rs429358_C == 2) &
            #             (alleles.rs7412_T == 0), 'apoe_polymorphism'] = 'e4/e4'
            # alleles.loc[(alleles.rs429358_C == 0) &
            #             (alleles.rs7412_T == 2), 'apoe_polymorphism'] = 'e2/e2'


            alleles.loc[(alleles.rs429358_C == 0), 'apoe_polymorphism'] = 0
            alleles.loc[(alleles.rs429358_C == 1), 'apoe_polymorphism'] = 1
            alleles.loc[(alleles.rs429358_C == 2), 'apoe_polymorphism'] = 2

            # Example merge
            df = df.merge(alleles[['IID', 'apoe_polymorphism']], left_on='eid',
                          right_on='IID', how='left')
            print(df.shape)

            df = get_last_completed_education(df, instance=0)

        elif outcome in ['hip fracture', 'hip_fracture']:
            eid_set = pd.read_table('../tidy_data/hip_fracture_eid.txt',
                                    header=None).iloc[:, 0].tolist()
            df['label'] = df['eid'].isin(eid_set).astype(int)

        # process date column
        date_cols = ['53-0.0']
        for s in date_cols:
            df[s] = pd.to_datetime(df[s])
            df[s] = df[s].fillna(value=np.nan)
            df[s] = (df[s] - min(df[s]))
            df[s] = df[s].apply(lambda x: x.days)

        # encode categorical columns (sex and site)
        catcols = [
                   '31-0.0',
                   '54-0.0',
                   '21000-0.0'
                   # 'apoe_polymorphism',
                   # 'max_educ_complete'
                   ]
        enc = OneHotEncoder()
        enc.fit(df.loc[:, catcols])
        categ_enc = pd.DataFrame(enc.transform(df.loc[:, catcols]).toarray(),
                                 columns=enc.get_feature_names_out(catcols))

        y = df.label.values

        # drop all unneeded columns, including but not limited to columns from
        # other instances, all protein columns, and original unencoded
        # categorical columns
        # X = df.drop(columns=['eid', 'label', '53-1.0', '53-2.0', '53-3.0',
        #                      '54-1.0', '54-2.0', '54-3.0', '21003-1.0',
        #                      '21003-2.0', '21003-3.0', '21003-1.0_squared',
        #                      '21003-2.0_squared', '21003-3.0_squared'] +
        #             catcols +
        #             prot.columns.tolist()
        #             )

        # age and encoded sex, APOE alleles, education as in
        # Guo et al. 2024 (Nat. Aging)

        if 'wo_demo' in output_path:
            X = df.loc[:, prot.columns[1:].tolist()]
        elif 'with_demo' in output_path:
            if outcome in ['dementia', 'acd', 'ACD', "alz"]:
                X = df.loc[:,
                           ['21003-0.0'] + prot.columns[1:].tolist()].join(
                            categ_enc)
            elif outcome in ['hip fracture', 'hip_fracture']:
                X = df.loc[:,
                           prot.columns[1:].tolist()].join(
                            categ_enc)
        elif 'demographics' in output_path:
            X = categ_enc

        X.to_parquet(f'{output_path}/X.parquet')
        np.save(f'{output_path}/y.npy', y)

    print(X.shape)
    if 'demographics' not in output_path:
        proteins = pd.read_csv('../tidy_data/proteomics/protein_colnames.txt',
                               header=None)
        proteins = proteins[0].tolist()

        if outcome in ['dementia', 'acd', 'ACD', "alz"]:
            protein_bank = [p for p in proteins if p not in [
                                        '242-0', '711-0', '883-0',
                                        '902-0', '1137-0', '1141-0',
                                        '1288-0', '1631-0', '1840-0',
                                        '1896-0', '1912-0'
                                        ]]
            n_proteins = 11
        else:
            protein_bank = proteins
            n_proteins = 18

        print(X.shape, y.shape)
        # train-test split
        train_idx, test_idx, y_train, y_test = train_test_split(
                list(range(X.shape[0])), y, test_size=0.2, stratify=y,
                random_state=99)

        np.random.seed(seed=task_id)
        p = np.random.choice(protein_bank, size=n_proteins, replace=False)
        print(f'Proteins: {p}')
        p_drop = set(proteins).difference(set(p))
        X = X.drop(columns=p_drop)
        print(X.shape)

    else:
        # train-test split
        train_idx, test_idx, y_train, y_test = train_test_split(
                list(range(X.shape[0])), y, test_size=0.2, stratify=y,
                random_state=99)
        p = None  # not using proteins

    automl = AutoML()
    automl.fit(
               X.iloc[train_idx], y_train, task="classification",
               time_budget=600,
               metric='roc_auc'
               )

    print('AutoML finished.')
    print(f'Best model: {automl.best_estimator}')

    y_train_pred = automl.predict_proba(X.iloc[train_idx])[:, 1]
    y_test_pred = automl.predict_proba(X.iloc[test_idx])[:, 1]

    # beta = 3
    n_bootstrap = 10000
    print('Bootstrapping training set')
    res = run_bootstrap(f'{output_path}', f'training_{task_id}',
                        y_train,
                        y_train_pred, n_bootstrap, youden=True,
                        return_results=True, proteins=p)
    res = res.sort_values(by='auroc', ascending=False)
    top_percent_thresh = res.best_thresh.values[:int(0.1*res.shape[0])]
    thresh_to_use = np.median(top_percent_thresh)
    print(f'Decision threshold: {thresh_to_use}\n')

    print('Bootstrapping test set')
    run_bootstrap(f'{output_path}', f'test_{task_id}', y_test,
                  y_test_pred, n_bootstrap, youden=True,
                  threshold=thresh_to_use, proteins=p)


if __name__ == '__main__':
    main()
