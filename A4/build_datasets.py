"""
A4 Study Data Processing Script

This script processes data from the A4 (Anti-Amyloid Treatment in Asymptomatic Alzheimer's Disease) study,
focusing on pTau217 biomarker data and Clinical Dementia Rating (CDR) measurements. The script:

1. Processes pTau217 biomarker measurements
2. Identifies cases based on CDR progression (CDR ≥ 0.5 for two consecutive visits)
3. Calculates time-to-event data for survival analysis
4. Merges demographic and genetic information
5. Performs data cleaning and validation

Key Features:
- Case Definition: Subjects with CDR ≥ 0.5 for two consecutive visits
- Controls: Subjects who never meet case criteria
- Time-to-event: Days from baseline to diagnosis (cases) or last visit (controls)
- Demographics: Age, sex, education, race, ethnicity
- Genetics: APOE genotype information

Output:
- Processed dataset saved as parquet file containing:
  * BID: Subject identifier
  * pTau217 measurements
  * Time-to-event data
  * Demographic information
  * Case/control labels
"""

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
    
    Args:
        data (pd.DataFrame): DataFrame containing:
            - BID: Subject identifier
            - AGEYR: Baseline age
            - COLLECTION_DATE_DAYS_CONSENT: Days from consent
    
    Returns:
        pd.DataFrame: DataFrame with updated AGEYR column reflecting age at each timepoint
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
    """
    Find time to first occurrence of CDR ≥ 0.5 for two consecutive visits.
    
    Args:
        sub_df (pd.DataFrame): DataFrame for a single subject containing:
            - CDGLOBAL: CDR global score
            - CDADTC_DAYS_CONSENT: Days from consent
    
    Returns:
        float or None: Days to consecutive CDR ≥ 0.5, or None if criteria not met
    """
    # Create a rolling window of size 2 to check for consecutive values >= 0.5
    consecutive_highs = ((sub_df['CDGLOBAL'] >= 0.5).rolling(2).sum() == 2)
    
    # Find the index of the second value in the consecutive pair
    if consecutive_highs.any():
        # Get the index of the first occurrence
        index_of_second = consecutive_highs.idxmax()  # idxmax gives the first True index
        # Return the corresponding date value
        return sub_df.loc[index_of_second, 'CDADTC_DAYS_CONSENT']
    else:
        return None  # Return None if no consecutive values >= 0.5 are found

def merge_demo(df):
    """
    Merge demographic information with main dataset.
    
    Args:
        df (pd.DataFrame): Main dataset with BID column
        
    Returns:
        pd.DataFrame: Dataset with added demographic information
    """
    demo = pd.read_csv(f'../../raw_data/A4_oct302024/clinical/Derived Data/SUBJINFO.csv')
    demo = demo[demo.BID.isin(df.BID.unique())]
    df = df.merge(
        demo.loc[:, ['BID', 'AGEYR', 'SEX', 'RACE', 'EDCCNTU', 'ETHNIC', 'APOEGN', 'TX']],
        on='BID',
        how='left')
    return df

def merge_cdr(df, cdr, sv):
    """
    Process CDR data and merge with main dataset.
    
    Args:
        df (pd.DataFrame): Main dataset
        cdr (pd.DataFrame): CDR measurements
        sv (pd.DataFrame): Study visit data
        
    Returns:
        tuple: (processed DataFrame, array of case BIDs)
    """
    
    # Sort data chronologically
    cdr = cdr.sort_values(by=['BID', 'VISCODE'])
    sv = sv.sort_values(by=['BID', 'VISCODE'])

    # Find time to consecutive CDR ≥ 0.5 for each subject
    result = cdr.groupby('BID').apply(time_to_consecutive_CDR)
    result = result.reset_index(name='time_to_event') # FI = functional impairment
    
    # merge to collate the time to event
    df = df.merge(result, on='BID', how='left')

    # Process cases and controls separately
    cases = df[df['time_to_event'].notna()]
    cases_bid = cases.BID.unique()
    cases['label'] = 1
    
    controls = df[df['time_to_event'].isna()]
    controls['label'] = 0

    # For controls, get time to last visit
    sv_controls = sv[(sv['BID'].isin(controls.BID)) & (sv.SVUSEDTC_DAYS_CONSENT.notna())]
    control_t2e = sv_controls.groupby('BID').SVUSEDTC_DAYS_CONSENT.max().reset_index(name='time_to_event').sort_values(by='time_to_event')
    
    # Update control time to event
    controls = controls.drop(columns=['time_to_event']).merge(control_t2e, on='BID', how='left')
    controls = controls[controls.time_to_event.notna()]

    # Combine cases and controls
    df = pd.concat([cases, controls], axis=0).reset_index(drop=True)
    
    # Add final visit information
    final_visit = sv[(sv['BID'].isin(df.BID)) & (sv.SVUSEDTC_DAYS_CONSENT.notna())]
    final_visit = final_visit.groupby('BID').SVUSEDTC_DAYS_CONSENT.max().reset_index(name='final_visit').sort_values(by='final_visit')
    df = df.merge(final_visit[['BID', 'final_visit']], on='BID', how='left')

    # Print summary statistics
    cases = result[result.time_to_event.notna()].BID.unique()
    print(f'{len(cases)} patients have a CDR of 0.5 or higher for two consecutive visits')

    cases_with_ptau217 = np.intersect1d(result[result.time_to_event.notna()].BID.unique(), df.BID.unique())
    print(f'{len(cases_with_ptau217)}/{len(cases)} patients have both CDR data and pTau217 data')

    return df, cases_bid

def fix_time_dependent_labels(df, cases_bid, id_col='BID', time_col='stop', 
                        time_to_event_col='time_to_event', label_col='label'):
    """
    Prepare time-to-event data for survival analysis by:
    1. Correctly setting labels based on event times
    2. Keeping only observations up to and including the first occurrence of case status
    
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
        DataFrame with corrected labels and trimmed to first case occurrence
    """
    # First, correctly set all labels based on event times
    df_fixed = df.copy()
    
    processed_data = []

    # add controls
    processed_data.append(df_fixed[~df_fixed[id_col].isin(cases_bid)])
    
    for subject_id in cases_bid:
        # Get all rows for this subject
        subject_data = df_fixed[df_fixed[id_col] == subject_id].copy()
        
        # Sort by time to ensure we process observations chronologically
        subject_data = subject_data.sort_values('start')
        
        # Get the time to event for this subject
        time_to_event = subject_data[time_to_event_col].iloc[0]
        
        # Set correct labels based on time comparison
        subject_data[label_col] = (subject_data[time_col] >= time_to_event).astype(int)
        
        # Find the first occurrence of label = 1 
        first_case_idx = subject_data[subject_data[label_col] == 1].index.min()
        
        if pd.isna(first_case_idx):
            # If no case status found, keep all observations
            processed_data.append(subject_data)
        else:
            # Keep all observations up to and including the first case
            processed_data.append(
                subject_data.loc[:first_case_idx]
            )
            # if sum(subject_data[label_col]) > 1:
            #     print(subject_id)
    
    # Combine all processed subject data
    final_data = pd.concat(processed_data, axis=0).reset_index(drop=True)
    
    return final_data

def merge_e2_carriers(df):
    """
    Consolidate APOE E2 carrier status.
    
    Args:
        df (pd.DataFrame): Dataset containing APOEGN column
        
    Returns:
        pd.DataFrame: Dataset with consolidated E2 carrier status
    """

    df.APOEGN = df.APOEGN.replace({
        "E2/E3": "E2_carrier",
        "E2/E2": "E2_carrier"
    })
    return df


def get_ptau():
    """
    Process pTau217 biomarker data and merge with clinical data.
    
    Returns:
        tuple: (processed DataFrame, array of case BIDs)
    """
    # Load and clean pTau217 data
    ptau217 = pd.read_csv(f'../../raw_data/A4_oct302024/clinical/External Data/biomarker_pTau217.csv')
    ptau217.drop(columns=['TESTCD', 'TEST', 'STAT', 'REASND', 'NAM', 'SPEC', 'METHOD', 'COMMENT', 'COMMENT2'], inplace=True)
    
    ptau217 = ptau217.sort_values(by=['BID', 'VISCODE'])
    print(ptau217.shape)

    # Remove invalid values and convert to float
    ptau217 = ptau217[ptau217['ORRES'].notna()]
    ptau217 = ptau217[ ~ ptau217['ORRES'].isin(['<LLOQ', '>ULOQ'])].reset_index(drop=True)
    ptau217['ORRES'] = ptau217['ORRES'].astype(float)

    # print the number of patients with ptau217 data
    ptau217_pts = ptau217.BID.unique()
    print(f'{len(ptau217_pts)} patients have pTau217 data')

    # cases are defined by having a Clinical Dementia Rating (CDR) of 0.5 or higher for two consecutive visits
    # this definition is used to define "functional impairment" in this paper: https://link.springer.com/content/pdf/10.14283/jpad.2024.122.pdf
    # Load CDR data
    cdr = pd.read_csv(f'../../raw_data/A4_oct302024/clinical/Raw Data/cdr.csv')
    cdr_pts = cdr.BID.unique()
    print(f'{len(cdr_pts)} patients have CDR data')

    # print the number of patients with both ptau217 and CDR data
    cdr_ptau217 = np.intersect1d(ptau217_pts, cdr_pts)
    print(f'{len(cdr_ptau217)} patients have both pTau217 and CDR data')

    cdr = cdr.sort_values(by=['BID', 'VISCODE'])

    sv = pd.read_csv(f'../../raw_data/A4_oct302024/clinical/Derived Data/SV.csv')
    sv.rename(columns={'VISITCD': 'VISCODE'}, inplace=True)
    
    # Merge CDR and visit data
    cdr = cdr.merge(sv[['BID', 'VISCODE', 'SVUSEDTC_DAYS_CONSENT']], on=['BID', 'VISCODE'])

    # Process and merge all data
    data, cases_bid = merge_cdr(ptau217, cdr, sv)
    data = merge_demo(data)

    # Fix specific data point based on external validation
    data.loc[(data['BID'] == 'B69890108') & (data.COLLECTION_DATE_DAYS_T0 == 84),
              'COLLECTION_DATE_DAYS_CONSENT'] = 2591
    data.loc[(data['BID'] == 'B69890108') & (data.COLLECTION_DATE_DAYS_T0 == 84),
              'COLLECTION_DATE_DAYS_T0'] = 2450
    
    # Sort chronologically
    data = data.sort_values(by=['BID', 'VISCODE']).reset_index(drop=True)

    # For BIDs not in cases_bid list, if COLLECTION_DATE_DAYS_CONSENT is greater than time_to_event, set time_to_event to COLLECTION_DATE_DAYS_CONSENT
    # this is a weird time lag issue that we need to address where the last pTau time point is coded as later than the patient's last visit
    data['time_to_event'] = np.where(
        (~data['BID'].isin(cases_bid)) & (data['COLLECTION_DATE_DAYS_CONSENT'] > data['time_to_event']),
        data['COLLECTION_DATE_DAYS_CONSENT'],
        data['time_to_event'])

    # Consolidate E2 carrier status
    data = merge_e2_carriers(data)
    return data, cases_bid

def main():
    """
    Main execution function. Processes data and saves to parquet file.
    """
    data, cases_bid = get_ptau()
    data.to_parquet('../../tidy_data/A4/ptau217_allvisits.parquet')

if __name__ == '__main__':
    main()
