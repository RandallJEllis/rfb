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
# from t2e import *
from sklearn.model_selection import StratifiedKFold
from scipy import stats

N_SPLITS = 5
N_BINS = 4

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
        
    Notes:
        Merges the following demographic variables:
        - AGEYR: Age at baseline
        - SEX: Biological sex
        - RACE: Self-reported race
        - EDCCNTU: Years of education
        - ETHNIC: Ethnicity
        - APOEGN: APOE genotype
        - TX: Treatment assignment
    """
    demo = pd.read_csv(f'../../raw_data/A4_oct302024/clinical/Derived Data/SUBJINFO.csv')
    demo = demo[demo.BID.isin(df.BID.unique())]
    df = df.merge(
        demo.loc[:, ['BID', 'AGEYR', 'SEX', 'RACE', 'EDCCNTU', 'ETHNIC', 'APOEGN', 'TX']],
        on='BID',
        how='left')
    return df

def phenotype_cdr(df, cdr, sv):
    """
    Process CDR data and merge with main dataset.
    
    Args:
        df (pd.DataFrame): Main dataset
        cdr (pd.DataFrame): CDR measurements
        sv (pd.DataFrame): Study visit data
        
    Returns:
        tuple: (processed DataFrame, array of case BIDs)
        
    Notes:
        - Cases are defined as subjects with CDR ≥ 0.5 for two consecutive visits
        - Controls are subjects who never meet case criteria
        - Time-to-event is calculated from baseline to diagnosis or last visit
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
    Prepare time-to-event data for survival analysis by correctly handling time-dependent labels.
    
    Args:
        df (pd.DataFrame): DataFrame containing time-to-event data
        cases_bid (array-like): Array of BIDs identified as cases
        id_col (str): Name of the ID column
        time_col (str): Name of the time column
        time_to_event_col (str): Name of the column containing time to event
        label_col (str): Name of the label column
    
    Returns:
        pd.DataFrame: DataFrame with corrected labels and trimmed to first case occurrence
        
    Notes:
        - For cases, keeps only observations up to and including first occurrence of case status
        - For controls, keeps all observations
        - Ensures correct temporal alignment of labels with events
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

def load_ptau_data(file_path):
    ptau217 = pd.read_csv(file_path)
    ptau217.drop(columns=['TESTCD', 'TEST', 'STAT', 'REASND', 'NAM', 'SPEC', 'METHOD', 'COMMENT', 'COMMENT2'], inplace=True)
    return ptau217

def clean_ptau_data(ptau217):
    ptau217 = ptau217[ptau217['ORRES'].notna()]
    ptau217 = ptau217[~ptau217['ORRES'].isin(['<LLOQ', '>ULOQ'])].reset_index(drop=True)
    ptau217['ORRES'] = ptau217['ORRES'].astype(float)
    return ptau217

def get_ptau():
    """
    Process pTau217 biomarker data and merge with clinical data.
    
    Returns:
        tuple: (processed DataFrame, array of case BIDs)
    """
    # Load and clean pTau217 data
    ptau217 = load_ptau_data(f'../../raw_data/A4_oct302024/clinical/External Data/biomarker_pTau217.csv')
    ptau217 = ptau217.sort_values(by=['BID', 'VISCODE'])
    ptau217 = clean_ptau_data(ptau217)

    # cases are defined by having a Clinical Dementia Rating (CDR) of 0.5 or higher for two consecutive visits
    # this definition is used to define "functional impairment" in this paper: https://link.springer.com/content/pdf/10.14283/jpad.2024.122.pdf
    # Load CDR data
    cdr = pd.read_csv(f'../../raw_data/A4_oct302024/clinical/Raw Data/cdr.csv')
    cdr = cdr.sort_values(by=['BID', 'VISCODE'])

    sv = pd.read_csv(f'../../raw_data/A4_oct302024/clinical/Derived Data/SV.csv')
    sv.rename(columns={'VISITCD': 'VISCODE'}, inplace=True)
    
    # Merge CDR and visit data
    cdr = cdr.merge(sv[['BID', 'VISCODE', 'SVUSEDTC_DAYS_CONSENT']], on=['BID', 'VISCODE'])

    # Process and merge all data
    data, cases_bid = phenotype_cdr(ptau217, cdr, sv)
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

# Create five datasets for five-fold cross-validation, stratified by label and time-to-event
def create_stratified_folds(data, n_splits=N_SPLITS, n_bins=N_BINS, random_state=42):
    """
    Create stratified cross-validation folds based on label and time-to-event.
    
    Args:
        data (pd.DataFrame): Input dataset
        n_splits (int): Number of folds for cross-validation
        n_bins (int): Number of bins for stratifying time-to-event
        random_state (int): Random seed for reproducibility
    
    Returns:
        pd.DataFrame: DataFrame with BID and fold assignments
        
    Notes:
        - Stratifies by both case/control status and time-to-event quartiles
        - Ensures balanced distribution of cases across folds
        - Maintains temporal distribution within case groups
    """
    # Aggregate data by BID, keeping label and time_to_event
    bids = (data[['id', 'label', 'time_to_event']]
            .sort_values(['label', 'time_to_event'], ascending=[False, True])
            .drop_duplicates('id')
            .reset_index(drop=True))
    
    # Verify label consistency
    # assert bids['label'].nunique() == data.groupby('BID')['label'].nunique().max(), \
    #     "Not all BIDs have a consistent label. Please resolve label inconsistencies."
    
    # Create time-to-event bins only for cases (label=1)
    cases = bids[bids['label'] == 1].copy()
    controls = bids[bids['label'] == 0].copy()
    
    # Create bins for cases based on time-to-event
    cases['time_bin'] = pd.qcut(cases['time_to_event'], q=n_bins, labels=False)
    # Assign -1 as time_bin for controls
    controls['time_bin'] = -1
    
    # Combine cases and controls
    bids = pd.concat([cases, controls])
    
    # Create a composite stratification variable
    bids['strata'] = bids['label'].astype(str) + '_' + bids['time_bin'].astype(str)
    
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Prepare the data for splitting
    X = bids['id']
    y = bids['strata']
    
    # Initialize the 'fold' column
    bids['fold'] = -1
    
    # Assign fold numbers
    for fold_number, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        bids.loc[val_idx, 'fold'] = fold_number
    
    # Verify fold assignment
    assert bids['fold'].min() >= 0 and bids['fold'].max() < n_splits, "Fold assignment error."
    
    return bids[['id', 'fold']]

# Process each fold
def process_fold(data, id_col, fold_assignments, fold):
    """
    Process individual cross-validation fold with appropriate data transformations.
    
    Args:
        data (pd.DataFrame): Complete dataset
        fold_assignments (pd.DataFrame): Fold assignments for each BID
        fold (int): Current fold number
    
    Returns:
        tuple: (train_set, val_set) processed training and validation datasets
        
    Notes:
        Performs the following transformations:
        - Age centering and standardization
        - Education level standardization
        - pTau217 (ORRES) standardization and Box-Cox transformation
        - Time variable scaling
        - Prints detailed fold statistics for quality control
    """
    # Merge fold assignments with original data
    data2 = data.merge(fold_assignments, on=id_col)
    
    # Define validation and training sets
    val_set = data2[data2['fold'] == fold].copy()
    train_set = data2[data2['fold'] != fold].copy()
    
    # Print fold information
    print(f"Fold {fold + 1}:")
    print(f"  Training BIDs: {train_set[id_col].nunique()}")
    print(f"  Validation BIDs: {val_set[id_col].nunique()}")
    print(f"  Positive class in Training: {train_set['label'].mean() * 100:.2f}%")
    print(f"  Positive class in Validation: {val_set['label'].mean() * 100:.2f}%")
    
    # Print time-to-event distribution for cases
    train_cases = train_set[train_set['label'] == 1]
    val_cases = val_set[val_set['label'] == 1]
    print("\nTime-to-event statistics for cases (years):")
    print("  Training:")
    print(f"    Mean: {train_cases['time_to_event'].mean():.2f}")
    print(f"    Median: {train_cases['time_to_event'].median():.2f}")
    print("  Validation:")
    print(f"    Mean: {val_cases['time_to_event'].mean():.2f}")
    print(f"    Median: {val_cases['time_to_event'].median():.2f}\n")
    
    # Preprocess data
    # zscore AGEYR
    training_age_mean = train_set.age.mean()
    training_age_std = train_set.age.std()
    train_set['age_z'] = (train_set.age - training_age_mean) / training_age_std
    val_set['age_z'] = (val_set.age - training_age_mean) / training_age_std
    
    train_set['age_centered'] = train_set.age - training_age_mean
    val_set['age_centered'] = val_set.age - training_age_mean
    
    # square and cube age
    train_set['age_z_squared'] = train_set.age_z ** 2
    train_set['age_z_cubed'] = train_set.age_z ** 3
    val_set['age_z_squared'] = val_set.age_z ** 2
    val_set['age_z_cubed'] = val_set.age_z ** 3
    
    train_set['age_centered_squared'] = train_set.age_centered ** 2
    train_set['age_centered_cubed'] = train_set.age_centered ** 3
    val_set['age_centered_squared'] = val_set.age_centered ** 2
    val_set['age_centered_cubed'] = val_set.age_centered ** 3
    
    # zscore EDCCNTU
    training_edc_mean = train_set.educ.mean()
    training_edc_std = train_set.educ.std()
    train_set['educ_z'] = (train_set.educ - training_edc_mean) / training_edc_std
    val_set['educ_z'] = (val_set.educ - training_edc_mean) / training_edc_std
    
    # zscore ORRES
    training_ptau_mean = train_set.ptau.mean()
    training_ptau_std = train_set.ptau.std()
    train_set['ptau_z'] = (train_set.ptau - training_ptau_mean) / training_ptau_std
    val_set['ptau_z'] = (val_set.ptau - training_ptau_mean) / training_ptau_std
    
    # boxcox transform ORRES
    train_set['ptau_boxcox'], lambda_val = stats.boxcox(train_set.ptau)
    val_set['ptau_boxcox'] = stats.boxcox(val_set.ptau, lmbda=lambda_val)
    
    # scale time variables
    train_set['visit_to_days_yr'] = train_set.visit_to_days
    train_set['time_to_event_yr'] = train_set.time_to_event 
    val_set['visit_to_days_yr'] = val_set.visit_to_days
    val_set['time_to_event_yr'] = val_set.time_to_event
    
    return train_set, val_set

def get_centiloids(data):
    """
    Process centiloid data and merge with clinical data.
    
    Args:
        data (pd.DataFrame): Input dataset
    """

    # import SUBJINFO which has centiloid values
    subjinfo = pd.read_csv('../../raw_data/A4_oct302024/clinical/Derived Data/SUBJINFO.csv')
    subjinfo = subjinfo[subjinfo.BID.isin(data.id)]
    
    '''
    imaging_pet_va has a column called scan_date_DAYS_CONSENT which has completely incorrect values.
    The only unique value for VISCODE is 2, and the range of scan_date_DAYS_CONSENT is 1027 to 2290.
    Why would the second visit of a study be 1000 days after the first visit? The answer is that the column is wrong.
    To remedy this, we will use the SV.csv file to get the correct number of days since consent for VISCODE 2.
    '''

    imaging_pet_va = pd.read_csv('../../raw_data/A4_oct302024/clinical/External Data/imaging_PET_VA.csv')
    imaging_pet_va = imaging_pet_va[imaging_pet_va.BID.isin(data.id)]
    
    sv = pd.read_csv(f'../../raw_data/A4_oct302024/clinical/Derived Data/SV.csv')
    sv.rename(columns={'VISITCD': 'VISCODE'}, inplace=True)
        
    # Merge CDR and visit data
    imaging_pet_va = imaging_pet_va.merge(sv[['BID', 'VISCODE', 'SVUSEDTC_DAYS_CONSENT', 'SVUSEDTC_DAYS_T0']], on=['BID', 'VISCODE'])
    subjinfo = subjinfo.merge(imaging_pet_va, on='BID', how='left')
    
    subjinfo['SVUSEDTC_YEARS_CONSENT'] = subjinfo['SVUSEDTC_DAYS_CONSENT'] / 365.25
    subjinfo['SVUSEDTC_YEARS_T0'] = subjinfo['SVUSEDTC_DAYS_T0'] / 365.25

    return subjinfo


def save_cv_folds(data, output_dir):
    """
    Save cross-validation folds as parquet files.
    
    Args:
        data (pd.DataFrame): Input dataset
    """

    fold_assignments = create_stratified_folds(data)

    for fold in range(5):
        # Process the fold
        train_set, val_set = process_fold(data, 'id', fold_assignments, fold)

        # print overlapping BIDs between training and validation sets
        train_bids = set(train_set['id'])
        val_bids = set(val_set['id'])
        overlap = train_bids.intersection(val_bids)
        print(f"  Overlapping BIDs: {len(overlap)}\n")

        # Save datasets
        train_set.to_parquet(f'{output_dir}/train_{fold}.parquet')
        val_set.to_parquet(f'{output_dir}/val_{fold}.parquet')
    
def main():
    """
    Main execution function for data processing pipeline.
    
    Process Flow:
    1. Load and process pTau217 and CDR data
    2. Create stratified cross-validation folds
    3. Process each fold with appropriate transformations
    4. Save processed datasets as parquet files
    
    Output Files:
    - ptau217_allvisits.parquet: Complete processed dataset
    - train_{fold}_new.parquet: Training data for each fold
    - val_{fold}_new.parquet: Validation data for each fold
    """
    data, _ = get_ptau()
    data.rename(columns={'BID': 'id', 'COLLECTION_DATE_DAYS_CONSENT': 'visit_to_days',
                         'AGEYR': 'age', 'EDCCNTU': 'educ', 'ORRES': 'ptau'}, inplace=True)
    data.to_parquet('../../tidy_data/A4/ptau_base.parquet')

    centiloids = get_centiloids(data)
    centiloids.to_parquet('../../tidy_data/A4/centiloids_subjinfo.parquet')

    save_cv_folds(data, '../../tidy_data/A4')

    # save datasets for sensitivity analysis with only amyloid-positive patients
    # subset positive-PET for sensitivity analysis
    centiloids_positive = centiloids[centiloids.overall_score == 'positive']
    data_positive = data[data.id.isin(centiloids_positive.BID)]
    data_positive = data_positive.merge(centiloids_positive[['BID', 'AMYLCENT', 'overall_score']],
                                                left_on='id', right_on='BID', how='left')

    # save ptau_pet_positive and subjinfo_positive
    data_positive.to_parquet('../../tidy_data/A4/amyloid_positive/ptau_pet_positive.parquet')
    centiloids_positive.to_parquet('../../tidy_data/A4/amyloid_positive/centiloids_pet_positive.parquet')

    save_cv_folds(data_positive, '../../tidy_data/A4/amyloid_positive')
    
if __name__ == "__main__":
    main()
