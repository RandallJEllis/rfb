import sys

sys.path.append("../ukb_func")
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
parser = argparse.ArgumentParser(description="Build datasets for autoML experiments")

# Add arguments
# parser.add_argument('njobs', type=str, help='Number of cores')
parser.add_argument("--data_path", type=str, default="../../../proj_idp/tidy_data/")
parser.add_argument(
    "--output_path", type=str, default="../../tidy_data/dementia/proteomics/"
)

# Parse the arguments
args = parser.parse_args()
data_path = args.data_path
output_path = args.output_path

utils.check_folder_existence(output_path)

data_instance = 0

df = pd.read_parquet(data_path + "proteomics/proteomics.parquet")

prot_keep = []
for c in df.columns:
    if "-0" in c:
        prot_keep.append(f"{c[:-2]}-0")
df = df.loc[:, ["eid"] + prot_keep]
df = df.dropna(subset=df.columns[1:], how="all").reset_index(drop=True)

demo = ukb_utils.load_demographics(data_path)
df = df.merge(demo, on="eid")

# import dx dates across dementia diagnosis Field IDs
# 42018, 42020, 42022, 42024 for "Date of [disease] report"
# 130836-130843, 131036-37 for "Date [ICD prefix] first reported"
acd = pd.read_parquet(data_path + "acd/allcausedementia.parquet")

df = dementia_utils.remove_pre_instance_dementia(df, data_instance, acd)

# APOEe4 alleles
snps = pd.read_csv(
    f"{data_path}/snps/plink_outputs/snps_aging_dementia_AD.raw", sep="\t"
)
# chr11 = pd.read_csv(
#     f'{data_path}/snps/plink_outputs/chr11.raw',
#     sep='\t'
#     )
df = dementia_utils.merge_alleles(df, snps)

# latest education qualification
df = ukb_utils.get_last_completed_education(df, instance=data_instance)

lancet_vars = pd.read_parquet(
    "../../tidy_data/dementia/lancet2024/lancet2024_preprocessed.parquet"
)
keep_lancet_vars = [
    "eid",
    "4700-0.0",
    "5901-0.0",
    "30780-0.0",
    "head_injury",
    "22038-0.0",
    "20161-0.0",
    "alcohol_consumption",
    "hypertension",
    "obesity",
    "diabetes",
    "hearing_loss",
    "depression",
    "freq_friends_family_visit",
    "2020-0.0",
    "24012-0.0",
    "24018-0.0",
    "24019-0.0",
    "24006-0.0",
    "24015-0.0",
    "24011-0.0",
]

df = df.merge(lancet_vars.loc[:, keep_lancet_vars], on="eid")

# encode sex, ethnicity, APOEe4 alleles, education qualifications
catcols = ["31-0.0", "2020-0.0", "21000-0.0", "apoe_polymorphism", "max_educ_complete"]
alleles = df_utils.pull_columns_by_prefix(df, "rs").columns.tolist()
catcols.extend(alleles)

categ_enc = ml_utils.encode_categorical_vars(df, catcols)

keep_lancet_vars_no2020_or_eid = [
    "4700-0.0",
    "5901-0.0",
    "30780-0.0",
    "head_injury",
    "22038-0.0",
    "20161-0.0",
    "alcohol_consumption",
    "hypertension",
    "obesity",
    "diabetes",
    "hearing_loss",
    "depression",
    "freq_friends_family_visit",
    "24012-0.0",
    "24018-0.0",
    "24019-0.0",
    "24006-0.0",
    "24015-0.0",
    "24011-0.0",
]

y = df.label.values
X = df.loc[
    :,
    ["eid", f"21003-{data_instance}.0", f"54-{data_instance}.0", "845-0.0"]
    + keep_lancet_vars_no2020_or_eid
    + prot_keep,
].join(categ_enc)


# cv indices based on region
region_lookup = pd.read_csv("../../metadata/coding10.tsv", sep="\t")
region_indices = ukb_utils.group_assessment_center(X, data_instance, region_lookup)

X.to_parquet(f"{output_path}/X.parquet")
np.save(f"{output_path}/y.npy", y)
ml_utils.save_pickle(f"{output_path}/region_cv_indices.pickle", region_indices)
