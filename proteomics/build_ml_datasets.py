import sys
sys.path.append('../ukb_func')
import rfb
import icd
import ml_utils
import df_utils
import ukb_utils
import utils
import dementia_utils

import argparse
import os
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
from sklearn.preprocessing import OneHotEncoder

# Create the argument parser
parser = argparse.ArgumentParser(
    description='Build datasets for autoML experiments')

# Add arguments
# parser.add_argument('njobs', type=str, help='Number of cores')
parser.add_argument('--data_path', type=str)
parser.add_argument('--output_path', type=str)

# Parse the arguments
args = parser.parse_args()
data_path = args.data_path
output_path = args.output_path

utils.check_folder_existence(output_path)

data_instance = 0

df = pd.read_parquet(data_path + 'proteomics/proteomics.parquet')

prot_keep = []
for c in df.columns:
    if '-0' in c:
        prot_keep.append(f'{c[:-2]}-0')
df = df.loc[:, ['eid'] + prot_keep]
df = df.dropna(subset=df.columns[1:], how='all').reset_index(drop=True)

demo = ukb_utils.load_demographics(data_path)
df = df.merge(demo, on='eid')

# import dx dates across dementia diagnosis Field IDs
acd = pd.read_parquet(data_path + 'acd/allcausedementia.parquet')

df = dementia_utils.remove_pre_instance_dementia(df, 2, acd)

# APOEe4 alleles
alleles = pd.read_csv(
    f'{data_path}/apoe4_snps/plink_outputs/apoee4_snps.raw',
    sep='\t'
    )
df = dementia_utils.apoe_alleles(df, alleles)

# latest education qualification
df = rfb.get_last_completed_education(df, instance=data_instance)

# encode sex, ethnicity, APOEe4 alleles, education qualifications
catcols = [
        '31-0.0',
        '21000-0.0',
        'apoe_polymorphism',
        'max_educ_complete'
        ]
categ_enc = ml_utils.encode_categorical_vars(df, catcols)

y = df.label.values
X = df.loc[:,
            [f'21003-{data_instance}.0', f'54-{data_instance}.0', '845-0.0'] + prot_keep].join(categ_enc)


# cv indices based on region
region_lookup = pd.read_csv('../../metadata/coding10.tsv', sep='\t')
region_indices = ukb_utils.group_assessment_center(X, data_instance, region_lookup)

X.to_parquet(f'{output_path}/X.parquet')
np.save(f'{output_path}/y.npy', y)
ml_utils.save_pickle(f'{output_path}/region_cv_indices.pickle', region_indices)

