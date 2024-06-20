import pandas as pd
import os

'''
This script returns two dataframes of ICD codes and corresponding dates.

Input arguments:
fieldid_icd - Field ID for ICD code type
fieldid_date - Field ID for ICD code date
code_icd - Codes you want to search for

'41270' is the Field ID for ICD 10 codes
'41271' is the Field ID for ICD 9 codes

'41280' is the Field ID for dates of ICD 10 codes
'41281' is the Field ID for dates of ICD 9 codes

Example:
icd10_df, icd10_date_df = pull_icds(['41270'],
                                    ['41280'],
                                    ['S720', 'S721', 'S722'])
icd9_df, icd9_date_df = pull_icds(['41271'], ['41281'], ['820'])
'''


# Function to check if a value starts with any of the given strings
def starts_with_any(value, start_strings):
    if value is None:
        return False
    return any(str(value).startswith(s) for s in start_strings)


def pull_icds(fieldid_icd, fieldid_date, code_icd):

    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir,
                             '../../../proj_idp/tidy_data/icd9_icd10.parquet')

    # df = pd.read_parquet('../../proj_idp/tidy_data/icd9_icd10.parquet')
    df = pd.read_parquet(file_path)
    # Identify column names that start with any of the search strings
    icd_matching_columns = [col
                            for col in df.columns
                            if any(col.startswith(s)
                                   for s in fieldid_icd)]

    icd_date_columns = [col
                        for col in df.columns
                        if any(col.startswith(s)
                               for s in fieldid_date)]

    # Apply the filtering
    match_df = df[df[icd_matching_columns].map(
                lambda x: starts_with_any(x, code_icd)).any(axis=1)]

    icd_df = match_df.loc[:, ['eid'] + icd_matching_columns].reset_index(
                                                            drop=True)
    icd_date_df = match_df.loc[:, ['eid'] + icd_date_columns].reset_index(
                                                            drop=True)

    return icd_df, icd_date_df
