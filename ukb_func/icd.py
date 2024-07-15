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
def _starts_with_any(value, start_strings):
    if value is None:
        return False
    return any(str(value).startswith(s) for s in start_strings)


def pull_icds(fieldid_icd, fieldid_date, code_icd):

    '''
    fieldid_icd: list - either 41270 (ICD10) or 41271 (ICD9) (diagnoses)
    fieldid_date: list - either 41280 (ICD10) or 41281 (ICD9) (Date of first in-patient diagnosis)
    code_icd: list - ICD codes you want to pull
    '''

    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir,
                             '../../../proj_idp/tidy_data/icd9_icd10.parquet')

    df = pd.read_parquet(file_path)
    # Subset columns for either ICD9 or ICD10
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
                lambda x: _starts_with_any(x, code_icd)).any(axis=1)]

    icd_df = match_df.loc[:, ['eid'] + icd_matching_columns].reset_index(
                                                            drop=True)
    icd_date_df = match_df.loc[:, ['eid'] + icd_date_columns].reset_index(
                                                            drop=True)

    return icd_df, icd_date_df


def _find_earliest_dates(icd_df, icd_date_df, code_icd):
    """
    Finds the earliest date for each row where the ICD code is present.
    
    icd_df: DataFrame containing ICD codes
    icd_date_df: DataFrame containing dates corresponding to the ICD codes
    code_icd: list of ICD codes to search for
    
    Returns:
    DataFrame with earliest dates for each row where the ICD code is present.
    """
    earliest_dates = []

    for i, row in icd_df.iterrows():
        indices = [j for j, col in enumerate(icd_df.columns[1:]) if any(str(row[col]).startswith(code) for code in code_icd)]
        if indices:
            dates = [pd.to_datetime(icd_date_df.iloc[i, j + 1], errors='coerce') for j in indices]
            earliest_date = min(dates)
        else:
            earliest_date = pd.NaT
        earliest_dates.append(earliest_date)

    icd_df['earliest_date'] = earliest_dates
    return icd_df[['eid', 'earliest_date']]
    

def pull_earliest_dates(fieldid_icd, fieldid_date, code_icd):
    icd_df, icd_date_df = pull_icds(fieldid_icd, fieldid_date, code_icd)
    earliest_dates = _find_earliest_dates(icd_df, icd_date_df, code_icd)
    return earliest_dates


def remove_patients_with_icd_before_instance(df_data, datecol_df_data, fieldid_icd, fieldid_date, code_icd):
    icd_df, icd_date_df = pull_icds(fieldid_icd, fieldid_date, code_icd)
    earliest_dates = _find_earliest_dates(icd_df, icd_date_df, code_icd)
    earliest_dates.loc[:, 'earliest_date'] = pd.to_datetime(earliest_dates.loc[:, 'earliest_date'])

    cases = df_data[df_data.eid.isin(earliest_dates.eid)]
    cases.loc[:, datecol_df_data] = pd.to_datetime(cases.loc[:, datecol_df_data])
    cases = cases.merge(earliest_dates, on='eid')


    cases['date_diff'] = cases.earliest_date - cases[datecol_df_data]
    cases = cases[cases.date_diff > pd.Timedelta(0)]

    controls = df_data[~df_data.eid.isin(earliest_dates.eid)]
    df = pd.concat([controls, cases])
    df['label'] = df['eid'].isin(cases.eid).astype(int)

    return df
    # cases = cases[cases.eid.isin(both_eid)]

    # controls = df[~df.eid.isin(exclude_df.eid)]
    # cases = df[df.eid.isin(cases.eid)]

    # df = pd.concat([controls, cases])
    # df['label'] = df['eid'].isin(cases.eid).astype(int)