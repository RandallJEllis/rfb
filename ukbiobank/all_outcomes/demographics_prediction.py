from bootstrap import run_bootstrap
from f3 import f3_metric

import os
import pandas as pd
import numpy as np
from flaml import AutoML

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def main():

    # load or create datasets
    if os.path.isfile('../tidy_data/demographics/X_noAge.npy') and \
       os.path.isfile('../tidy_data/demographics/y_noAge.npy'):
        print('Loading datasets')
        X = np.load('../tidy_data/demographics/X_noAge.npy')
        y = np.load('../tidy_data/demographics/y_noAge.npy')
    else:
        print('Building datasets')
        path = '../../proj_idp/tidy_data/'
        prot = pd.read_parquet(path + 'proteomics/proteomics.parquet')

        # only keep proteins from Instance 0
        cols_remove = []
        for c in prot.columns:
            if '-1' in c:
                cols_remove.append(c)
            elif '-2' in c:
                cols_remove.append(c)
            elif '-3' in c:
                cols_remove.append(c)
        prot = prot.drop(columns=cols_remove)

        # import age, sex, education, site, assessment date
        demographics = pd.read_parquet(path +
                                       'demographics/demographics_df.parquet'
                                       )
        df = prot.merge(demographics)

        # import dx dates across dementia diagnosis Field IDs
        acd = pd.read_parquet(path + 'acd/allcausedementia.parquet')

        # Strings to search for
        search_strings = ['42018', '42020', '42022', '42024', '131036',
                          '130836', '130838', '130840', '130842']

        # Identify column names that start with any of the search strings
        matching_columns = [col for col in acd.columns if any(col.startswith(s)
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
        cases = cases[cases.date_diff > pd.Timedelta(0)]

        controls = df[~df.eid.isin(acd.eid)]

        df = pd.concat([controls, cases])
        df['label'] = df['eid'].isin(acd.eid).astype(int)

        '''
        Apolipoprotein E (APOE) genotype has three major alleles (epsilon 2,
         epsilon 3, epsilon 4):
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

        alleles = pd.read_csv('/n/groups/patel/randy/proj_idp/tidy_data/apoe4_snps/plink_outputs/apoee4_snps.raw', sep='\t')
        alleles['apoe_polymorphism'] = np.nan
        alleles.loc[(alleles.rs429358_C == 0) &
                    (alleles.rs7412_T == 0), 'apoe_polymorphism'] = 'e3/e3'
        alleles.loc[(alleles.rs429358_C == 1) &
                    (alleles.rs7412_T == 0), 'apoe_polymorphism'] = 'e3/e4'
        alleles.loc[(alleles.rs429358_C == 0) &
                    (alleles.rs7412_T == 1), 'apoe_polymorphism'] = 'e2/e3'
        alleles.loc[(alleles.rs429358_C == 1) &
                    (alleles.rs7412_T == 1), 'apoe_polymorphism'] = 'e2/e4'
        alleles.loc[(alleles.rs429358_C == 2) &
                    (alleles.rs7412_T == 0), 'apoe_polymorphism'] = 'e4/e4'
        alleles.loc[(alleles.rs429358_C == 0) &
                    (alleles.rs7412_T == 2), 'apoe_polymorphism'] = 'e2/e2'

        # Example merge
        df = df.merge(alleles[['IID', 'apoe_polymorphism']], left_on='eid',
                      right_on='IID', how='left')
        print(df.shape)

        # education columns have array indices 0-5 for each instance
        # for each patient, take the max value of the Instance 0 columns
        # to represent max education completed
        educ_ins0_cols = df.columns[df.columns.str.startswith('6138-0')]
        max_educ_ins0 = df.loc[:, educ_ins0_cols].max(axis=1)
        df['max_educ_complete'] = max_educ_ins0

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
                   # '54-0.0',
                   'apoe_polymorphism',
                   'max_educ_complete'
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
        # X = df.loc[:, ['21003-0.0']].join(categ_enc)
        X = categ_enc
        np.save('../tidy_data/demographics/X_noAge.npy', X)
        np.save('../tidy_data/demographics/y_noAge.npy', y)

    print(X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=2024)

    automl = AutoML()
    automl.fit(X_train, y_train, task="classification", time_budget=600,
               metric='roc_auc')

    print('AutoML finished.')
    print(f'Best model: {automl.best_estimator}')

    y_train_pred = automl.predict_proba(X_train)[:, 1]
    y_test_pred = automl.predict_proba(X_test)[:, 1]

    # beta = 3
    n_bootstrap = 10000
    print('Bootstrapping training set')
    res = run_bootstrap('../tidy_data/demographics/', 'training_noAge', y_train,
                        y_train_pred, n_bootstrap, youden=True,
                        return_results=True)
    res = res.sort_values(by='auroc', ascending=False)
    top_percent_thresh = res.best_thresh.values[:int(0.1*res.shape[0])]
    thresh_to_use = np.median(top_percent_thresh)
    print(f'Decision threshold: {thresh_to_use}\n')

    print('Bootstrapping test set')
    run_bootstrap('../tidy_data/demographics/', 'test_noAge', y_test,
                  y_test_pred, n_bootstrap, youden=True,
                  threshold=thresh_to_use)


if __name__ == '__main__':
    main()
