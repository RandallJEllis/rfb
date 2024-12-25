import pandas as pd
import numpy as np
import sys
sys.path.append('../../ukb_func')
import os
from t2e import *

def vectorized_age_calculation(data):
    """
    Calculate age updates for each patient based on collection date differences.
    Resets cumulative sum for each new patient ID.
    
    Parameters:
    data (pandas.DataFrame): DataFrame containing columns 'BID', 'AGEYR', 'COLLECTION_DATE_DAYS_CONSENT'
    
    Returns:
    pandas.DataFrame: DataFrame with updated AGEYR column
    """
    # Make a copy to avoid modifying the original
    result = data.copy()
    
    # Create a mask for rows with the same BID as the previous row
    same_bid_mask = result['BID'] == result['BID'].shift(1)
    
    # Calculate age increments for rows with the same BID
    age_increments = np.where(same_bid_mask, 
                             (result['COLLECTION_DATE_DAYS_CONSENT'] - result['COLLECTION_DATE_DAYS_CONSENT'].shift(1)) / 365.25, 
                             0)
    
    # Create groups based on BID
    bid_groups = (result['BID'] != result['BID'].shift(1)).cumsum()
    
    # Convert numpy array to pandas Series for groupby operation
    age_increments_series = pd.Series(age_increments, index=result.index)
    
    # Calculate cumulative sum within each BID group
    cumulative_age_increments = age_increments_series.groupby(bid_groups).cumsum()
    
    # Add cumulative increments to original AGEYR
    result['AGEYR'] = result['AGEYR'] + cumulative_age_increments
    
    return result

# find all patients with a CDR of 0.5 or higher for two consecutive visits
def time_to_consecutive_CDR(sub_df):
    
    # Create a rolling window of size 2 to check for consecutive values >= 0.5
    consecutive_highs = ((sub_df['CDGLOBAL'] >= 0.5).rolling(2).sum() == 2)
    
    # Find the index of the second value in the consecutive pair
    if consecutive_highs.any():
        # Get the index of the first occurrence
        index_of_second = consecutive_highs.idxmax()  # idxmax gives the first True index
        # Return the corresponding date value
        return sub_df.loc[index_of_second, 'CDADTC_DAYS_T0']
    else:
        return None  # Return None if no consecutive values >= 0.5 are found

def merge_demo(df):
    demo = pd.read_csv(f'../../raw_data/A4_oct302024/clinical/Derived Data/SUBJINFO.csv')
    demo = demo[demo.BID.isin(df.BID.unique())]
    df = df.merge(demo.loc[:, ['BID', 'AGEYR', 'SEX', 'RACE', 'EDCCNTU', 'ETHNIC', 'APOEGN', 'TX']], on='BID', how='left')
    return df

def merge_cdr(df, cdr, sv):

    cdr = cdr.sort_values(by=['BID', 'VISCODE'])
    sv = sv.sort_values(by=['BID', 'VISCODE'])

    # identify patients with CDR >= 0.5 for two consecutive visits
    result = cdr.groupby('BID').apply(time_to_consecutive_CDR)

    # merge to collate the time to event
    result = result.reset_index(name='time_to_event') # FI = functional impairment
    df = df.merge(result, on='BID', how='left')

    # for patients who do not reach criteria for functional impairment (CDR of 0.5+ for two consecutive visits), we will get the number of days after baseline that the patient was last seen
    cases = df[df['time_to_event'].notna()]
    cases['label'] = 1
    controls = df[df['time_to_event'].isna()]
    controls['label'] = 0

    sv_controls = sv[(sv['BID'].isin(controls.BID)) & (sv.SVUSEDTC_DAYS_CONSENT.notna())]
    control_t2e = sv_controls.groupby('BID').SVUSEDTC_DAYS_CONSENT.max().reset_index(name='time_to_event').sort_values(by='time_to_event')
    controls = controls.drop(columns=['time_to_event']).merge(control_t2e, on='BID', how='left')
    controls = controls[controls.time_to_event.notna()]

    df = pd.concat([cases, controls], axis=0).reset_index(drop=True)

    final_visit = sv[(sv['BID'].isin(df.BID)) & (sv.SVUSEDTC_DAYS_CONSENT.notna())]
    final_visit = final_visit.groupby('BID').SVUSEDTC_DAYS_CONSENT.max().reset_index(name='final_visit').sort_values(by='final_visit')
    df = df.merge(final_visit[['BID', 'final_visit']], on='BID', how='left')

    # print out descriptive stats
    cases = result[result.time_to_event.notna()].BID.unique()
    print(f'{len(cases)} patients have a CDR of 0.5 or higher for two consecutive visits')

    cases_with_ptau217 = np.intersect1d(result[result.time_to_event.notna()].BID.unique(), df.BID.unique())
    print(f'{len(cases_with_ptau217)}/{len(cases)} patients have both CDR data and pTau217 data')

    # for c in df.columns:
    #     if df[c].nunique() < 10:
    #         print(c, df[c].unique())

    return df

def fix_time_dependent_labels(df, id_col='BID', time_col='COLLECTION_DATE_DAYS_CONSENT', 
                            time_to_event_col='time_to_event', label_col='label'):
    """
    Fix labels in a time-to-event dataset with customizable column names.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing time-to-event data
    id_col : str
        Name of the ID column
    time_col : str
        Name of the time column
    time_to_event_col : str
        Name of the column containing time to event
    label_col : str
        Name of the label column
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with corrected labels
    """
    df_fixed = df.copy()
    
    for subject_id in df_fixed[id_col].unique():
        mask = df_fixed[id_col] == subject_id
        time_to_event = df_fixed.loc[mask, time_to_event_col].iloc[0]
        df_fixed.loc[mask, label_col] = (
            df_fixed.loc[mask, time_col] >= time_to_event
        ).astype(int)
    
    return df_fixed

def get_ptau():
    """
    Processes pTau217 biomarker data and merges it with Clinical Dementia Rating (CDR) data.
    Steps:
    1. Reads and cleans pTau217 biomarker data.
    2. Removes invalid values and handles '<LLOQ' and '>ULOQ' cases.
    3. Converts 'ORRES' column to float.
    4. Reads and processes CDR data.
    5. Merges pTau217 and CDR data.
    6. Corrects specific data values based on external source.
    7. Calculates start and stop times for each patient.
    8. Drops rows where start is greater than stop.
    9. Adjusts labels for rows where start equals stop.
    10. Calculates age-related vectors.
    Returns:
        pd.DataFrame: Processed dataset with merged pTau217 and CDR data.
    """

    ptau217 = pd.read_csv(f'../../raw_data/A4_oct302024/clinical/External Data/biomarker_pTau217.csv')
    ptau217.drop(columns=['TESTCD', 'TEST', 'STAT', 'REASND', 'NAM', 'SPEC', 'METHOD', 'COMMENT', 'COMMENT2'], inplace=True)
    
    ptau217 = ptau217.sort_values(by=['BID', 'VISCODE'])
    print(ptau217.shape)

    # remove invalud values
    ptau217 = ptau217[ptau217['ORRES'].notna()]
    print(ptau217.shape)

    min_ptau = ptau217[ ~ ptau217['ORRES'].isin(['<LLOQ', '>ULOQ'])]['ORRESRAW'].min()
    if min_ptau < 1:
        ptau217.loc[ptau217['ORRES'] == '<LLOQ', 'ORRESRAW'] = min_ptau ** 2
        ptau217.loc[ptau217['ORRES'] == '<LLOQ', 'ORRES'] = min_ptau ** 2
    else:
        ptau217.loc[ptau217['ORRES'] == '<LLOQ', 'ORRESRAW'] = np.sqrt(min_ptau)
        ptau217.loc[ptau217['ORRES'] == '<LLOQ', 'ORRES'] = np.sqrt(min_ptau)
    ptau217 = ptau217[ ~ ptau217['ORRES'].isin(['<LLOQ', '>ULOQ'])].reset_index(drop=True)
    print(ptau217.shape)
    ptau217['ORRES'] = ptau217['ORRES'].astype(float)

    # print the number of patients with ptau217 data
    ptau217_pts = ptau217.BID.unique()
    print(f'{len(ptau217_pts)} patients have pTau217 data')

    # cases are defined by having a Clinical Dementia Rating (CDR) of 0.5 or higher for two consecutive visits
    # this definition is used to define "functional impairment" in this paper: https://link.springer.com/content/pdf/10.14283/jpad.2024.122.pdf
    cdr = pd.read_csv(f'../../raw_data/A4_oct302024/clinical/Raw Data/cdr.csv')
    cdr = cdr.sort_values(by=['BID', 'VISCODE'])

    sv = pd.read_csv(f'../../raw_data/A4_oct302024/clinical/Derived Data/SV.csv')
    sv.rename(columns={'VISITCD': 'VISCODE'}, inplace=True)

    cdr = cdr.merge(sv[['BID', 'VISCODE', 'SVUSEDTC_DAYS_CONSENT']], on=['BID', 'VISCODE'])

    cdr_pts = cdr.BID.unique()
    print(f'{len(cdr_pts)} patients have CDR data')

    # print the number of patinets with both ptau217 and CDR data
    cdr_ptau217 = np.intersect1d(ptau217_pts, cdr_pts)
    print(f'{len(cdr_ptau217)} patients have both pTau217 and CDR data')

    # instead of ptau217, could use last_ptau217 or visit6
    data = merge_cdr(ptau217, cdr, sv)
    data = merge_demo(data)

    # incorrect value; correct value pulled from /raw_data/A4_oct302024/clinical/Derived Data/SV.csv
    data.loc[(data['BID'] == 'B69890108') & (data.COLLECTION_DATE_DAYS_T0 == 84), 'COLLECTION_DATE_DAYS_CONSENT'] = 2591
    data.loc[(data['BID'] == 'B69890108') & (data.COLLECTION_DATE_DAYS_T0 == 84), 'COLLECTION_DATE_DAYS_T0'] = 2450

    data = data.sort_values(by=['BID', 'VISCODE']).reset_index(drop=True)
    
    # calculate start and stop times for each patient
    stop = data.COLLECTION_DATE_DAYS_CONSENT.shift(-1)
    stop.iloc[-1] = data.final_visit.iloc[-1]
    stop = np.where(data.BID != data.BID.shift(-1), data.final_visit, stop)

    data['start'] = data.COLLECTION_DATE_DAYS_CONSENT
    data['stop'] = stop
    print(data.shape)

    print('Dropping rows where start > stop')

    # In rows where start is greater than stop, overwrite stop
    data['stop'] = np.where(data['start'] > data['stop'], data['start'], data['stop'])
    # data = data[data['start'] <= data['stop']].reset_index(drop=True)
    # data['label'] = (data.stop > data.time_to_event).astype(int)
    print(data.shape)

    # print('Dropping rows where start == stop')
    # # drop rows where start == stop. for patients with no events, this is all that's needed. for patients with events, if the previous time step does not have an event, change the label to 1
    # rows_drop = data[(data.start == data.stop) & (data.label == 0)].index
    # data.drop(rows_drop, inplace=True)
    # data = data.reset_index(drop=True)
    # print(data.shape)

    # adjust labels for rows where start == stop
    rows = data[(data.start == data.stop)].index
    for r in rows:
        if data.label[r-1] != data.label[r]:
            # overwrite value at previous index to 1
            data.loc[r-1, 'label'] = 1
    data = data[data.start <= data.stop].reset_index(drop=True)
    print(data.shape)

    data = vectorized_age_calculation(data)

    # Assuming your DataFrame is called 'data'
    data = fix_time_dependent_labels(data)

    return data

def main():
    
    data = get_ptau()
    data.to_parquet('../../tidy_data/A4/ptau217_allvisits.parquet')



if __name__ == '__main__':
    main()
