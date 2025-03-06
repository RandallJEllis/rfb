"""
Dataset Processing Script for Multiple PET Cohorts

This script processes PET (Positron Emission Tomography) data from multiple cohorts:
- OASIS (Open Access Series of Imaging Studies)
- NACC (National Alzheimer's Coordinating Center)
- HABS (Harvard Aging Brain Study)
- ADNI (Alzheimer's Disease Neuroimaging Initiative)
- AIBL (Australian Imaging, Biomarkers and Lifestyle)

For each cohort, the script:
1. Loads and cleans raw data files (PET scans, clinical diagnoses, demographics)
2. Processes case-control data:
   - Cases: Subjects who developed dementia
   - Controls: Subjects who remained dementia-free
3. Computes temporal features:
   - Time from baseline visit
   - Time to event (diagnosis for cases, last visit for controls)
4. Adds demographic and genetic information
5. Saves processed datasets as parquet files

The output files contain standardized columns across cohorts:
- RID: Subject identifier
- EXAMDATE: Examination date
- visit_to_days: Days from baseline visit
- time_to_event: Days to diagnosis (cases) or last visit (controls)
- label: Case (1) or control (0)
- age: Age at examination
- demographic features (gender, education, etc.)
- genetic features (APOE genotype)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

def parse_dates(date):
    """
    Parse date strings handling both 2-digit and 4-digit year formats.
    
    Args:
        date (str): Date string in either MM/DD/YY or MM/DD/YYYY format
        
    Returns:
        pandas.Timestamp: Parsed datetime object
    """
    if '/' in date:
        if len(date.split('/')[-1]) == 2:  # Check if the year is 2 digits
            return pd.to_datetime(date, format='%m/%d/%y', errors='coerce')
        else:  # Year is 4 digits
            return pd.to_datetime(date, format='%m/%d/%Y', errors='coerce')
    else:
        return pd.to_datetime(date, errors='coerce')
    
def find_first_or_last_visit(df_list, id_col='RID', examdate_col='EXAMDATE', first_or_last='first'):
    """
    Find either the first or last visit date for each subject across multiple dataframes.
    
    Args:
        df_list (list): List of pandas DataFrames containing visit data
        id_col (str): Column name for subject identifier
        examdate_col (str): Column name for examination date
        first_or_last (str): Either 'first' or 'last' to determine which visit to find
        
    Returns:
        pandas.DataFrame: DataFrame with subject IDs and their first/last visit dates
    """
    # find the earliest/latest visit for each ID across all dataframes
    df_list = [df.sort_values(by=[id_col, examdate_col]) for df in df_list]

    if first_or_last == 'first':
        df_list = [df.groupby(id_col)[examdate_col].first().reset_index() for df in df_list]
    elif first_or_last == 'last':
        df_list = [df.groupby(id_col)[examdate_col].last().reset_index() for df in df_list]
    else:
        raise ValueError("first_or_last must be 'first' or 'last'")

    df_list = [df.rename({examdate_col: f'{first_or_last}_{examdate_col}'}, axis=1) for df in df_list]

    # merge the dataframes on ID
    merged = df_list[0].merge(df_list[1], on=id_col, how='outer')
    for df in df_list[2:]:
        merged = merged.merge(df, on=id_col, how='outer')
    
    if first_or_last == 'first':
        merged[examdate_col] = merged.apply(lambda row: min(row[f'{first_or_last}_{examdate_col}_x'], 
                                                        row[f'{first_or_last}_{examdate_col}_y']) 
                                        if pd.notna(row[f'{first_or_last}_{examdate_col}_x']) and pd.notna(row[f'{first_or_last}_{examdate_col}_y']) 
                                        else row[f'{first_or_last}_{examdate_col}_x'] 
                                        if pd.notna(row[f'{first_or_last}_{examdate_col}_x']) 
                                        else row[f'{first_or_last}_{examdate_col}_y'], axis=1)
    elif first_or_last == 'last':
        merged[examdate_col] = merged.apply(lambda row: max(row[f'{first_or_last}_{examdate_col}_x'], 
                                                        row[f'{first_or_last}_{examdate_col}_y']) 
                                        if pd.notna(row[f'{first_or_last}_{examdate_col}_x']) and pd.notna(row[f'{first_or_last}_{examdate_col}_y']) 
                                        else row[f'{first_or_last}_{examdate_col}_x'] 
                                        if pd.notna(row[f'{first_or_last}_{examdate_col}_x']) 
                                        else row[f'{first_or_last}_{examdate_col}_y'], axis=1)
    
    return merged
    
def map_time_from_baseline(df, earliest_baseline, id_col='RID', examdate_col='EXAMDATE'):
    """
    Calculate days from baseline visit for each examination date.
    
    Args:
        df (pandas.DataFrame): DataFrame containing visit data
        earliest_baseline (pandas.DataFrame): DataFrame with baseline dates for each subject
        id_col (str): Column name for subject identifier
        examdate_col (str): Column name for examination date
        
    Returns:
        pandas.DataFrame: Input DataFrame with added visit_to_days column
    """
    df['visit_to_days'] = df.apply(lambda row: (row[examdate_col] - earliest_baseline[earliest_baseline[id_col] == row[id_col]][examdate_col].values[0]).days, axis=1)
    return df


def process_oasis(oasis_file_path, oasis_diagnoses_file_path, oasis_demo_file_path):
    
    # Load PET centiloid data
    df = pd.read_csv(oasis_file_path)
    
    # Process session IDs and sort by visit day
    df[['ID', 'tracer', 'day']] = df['oasis_session_id'].str.split('_', expand=True)
    cols = ['ID', 'tracer', 'day'] + [col for col in df if col not in ['ID', 'tracer', 'day']]
    df = df[cols]
    df['day'] = df['day'].str.replace('d', '').astype(int)
    df = df.sort_values(by=['ID', 'day']).reset_index(drop=True)

    diagnoses = pd.read_csv(oasis_diagnoses_file_path)

    # Split the 'PUP_PUPTIMECOURSEDATA ID' column into new columns
    diagnoses[['ID', 'phase', 'day']] = diagnoses['OASIS_session_label'].str.split('_', expand=True)

    # Move the new columns to be the first columns
    cols = ['ID', 'phase', 'day'] + [col for col in diagnoses if col not in ['ID', 'phase', 'day']]
    diagnoses = diagnoses[cols]
    diagnoses['day'] = diagnoses['day'].str.replace('d', '').astype(int)
    diagnoses = diagnoses.sort_values(by=['ID', 'day'])


    diagnoses['dx_sum'] = diagnoses.iloc[:, 8:].sum(axis=1)
    cases_dx = diagnoses[diagnoses['dx_sum'] > 0]
    case_dx_first = cases_dx.groupby('ID').first().reset_index()
    cases_dx_ids = cases_dx.OASISID.unique()
    pet_pts = df.ID.unique()
    cases_pet = np.intersect1d(cases_dx_ids, pet_pts)

    latest_date = find_first_or_last_visit([df, diagnoses], id_col='ID', examdate_col='day', first_or_last='last')

    # check how many cases have a PET scan before the first diagnosis
    pet_case_df = []

    exclusions = []
    for c in case_dx_first.ID.unique():
        c_pet = df[df['ID'] == c]
        c_first_dx = case_dx_first[case_dx_first['ID'] == c]
        # Only include PET scans before diagnosis
        c_pet_before_dx = c_pet[c_pet['day'] < c_first_dx['day'].values[0]]
        if c_pet_before_dx.shape[0] == 0:
            continue
        else:
            c_pet = c_pet.rename({'day': 'visit_to_days'}, axis=1)
            c_pet.loc[:, 'time_to_event'] = c_first_dx[c_first_dx['ID'] == c]['day'].values[0]
            pet_case_df.append(c_pet)

    pet_case_df = pd.concat(pet_case_df, axis=0).reset_index(drop=True)
    pet_control_df = df[~df['ID'].isin(case_dx_first['ID'].unique())]
    pet_control_df = pet_control_df.rename({'day': 'visit_to_days'}, axis=1)

    control_t2e = []
    for c in pet_control_df.ID.unique():
        latest_visit = latest_date[latest_date['ID'] == c]
        control_t2e.append(latest_visit['day'].values[0])
    pet_control_df = pet_control_df.merge(pd.DataFrame({'ID': pet_control_df.ID.unique(), 'time_to_event': control_t2e}), on='ID', how='left')

    pet_df = pd.concat([pet_case_df, pet_control_df]).reset_index(drop=True)
    pet_df['label'] = np.array([1] * len(pet_case_df) + [0] * len(pet_control_df))

    demo = pd.read_csv(oasis_demo_file_path)
    demo.loc[demo.race == 'AIAN', 'race'] = 'ASIAN'

    pet_df = pet_df.merge(demo.loc[:,['OASISID', 'AgeatEntry', 'GENDER', 'EDUC', 'SES', 'race', 'APOE']], left_on='ID', right_on='OASISID', how='left')
    pet_df = pet_df.rename({'AgeatEntry': 'age'}, axis=1)
    pet_df['age'] = (pet_df['age'] + (pet_df['visit_to_days'] / 365.25))
    return pet_df


def process_nacc(nacc_file_path, nacc_centiloids_file_path):
    # Load raw NACC data
    # check if nacc_file_path ends in csv or parquet
    if nacc_file_path.endswith('.csv'):
        df = pd.read_csv(nacc_file_path)
    elif nacc_file_path.endswith('.parquet'):
        df = pd.read_parquet(nacc_file_path)
    else:
        raise ValueError(f"Invalid file extension: {nacc_file_path}")

    # Select relevant columns for analysis
    df = df.loc[:, ['NACCID', 'NACCADC', 'VISITDAY', 'VISITMO', 'VISITYR', 'NACCACTV',
                    'DEMENTED', 'NACCADMD', 'NACCALZD', 'NACCALZP', 'PROBAD', 'PROBADIF', 
                    'POSSAD', 'POSSADIF', 'NACCETPR',
                    'BIRTHMO', 'BIRTHYR', 'SEX', 'RACE', 'EDUC', 'NACCAGE', 'NACCAGEB', 'NACCAPOE', 'NACCNE4S', 'NACCUDSD',
                    'NACCTBI', 'TBI', 'TBIBRIEF', 'TBIWOLOS', 
                    'TOBAC30', 'TOBAC100', 'SMOKYRS', 'PACKSPER', 'QUITSMOK',
                    'ALCFREQ', 'ALCOHOL', 'ALCABUSE',
                    'HYPERT', 'HYPERTEN', 'HXHYPER', 'NACCAHTN', 'NACCHTNC',
                    'NACCBMI',
                    'NACCDBMD', 'DIABET', 'DIABETES',
                    'HEARING', 'HEARAID', 'HEARWAID',
                    'DEPD', 'DEPDSEV', 'NACCGDS', 'NACCADEP', 'DEPTREAT', 'DEP2YRS', 'DEPOTHR']]

    # Create datetime columns for visits and sort chronologically
    df = df.sort_values(by=['NACCID', 'VISITYR', 'VISITMO', 'VISITDAY'])
    df['VISITDATE'] = pd.to_datetime(pd.DataFrame({'year': df['VISITYR'], 'month': df['VISITMO'], 'day': df['VISITDAY']}))
    df['BIRTHDAY'] = pd.to_datetime(pd.DataFrame({'year': df['BIRTHYR'], 'month': df['BIRTHMO'], 'day': 15}))

    # Clean TBI (Traumatic Brain Injury) related columns
    # Replace unknown (9) and not available (-4) values with NaN
    tbi_cols = ['NACCTBI', 'TBI', 'TBIBRIEF', 'TBIWOLOS', 'TOBAC30', 'TOBAC100', 'ALCOHOL',
                'ALCABUSE', 'HYPERTEN', 'DIABET', 'DIABETES', 'HEARING', 'HEARAID', 
                'DEPD', 'DEP2YRS', 'DEPOTHR']
    for col in tbi_cols:
        df[col] = df[col].replace({9: np.nan, -4: np.nan})

    # Overwrite 8, 9, and -4 as NaN in the PACKSPER column
    for col in ['PACKSPER', 'ALCFREQ', 'HEARWAID', 'DEPDSEV']:
        df[col] = df[col].replace({8: np.nan, 9: np.nan, -4: np.nan})

    # Overwrite -4 as NaN in the HXHYPER column
    for col in ['HXHYPER', 'NACCAHTN', 'NACCHTNC', 'NACCDBMD', 'NACCADEP']:
        df[col] = df[col].replace({-4: np.nan})

    # Overwrite 8 and -4 as NaN in the HYPERT column
    df['HYPERT'] = df['HYPERT'].replace({8: np.nan, -4: np.nan})
    df['DEPTREAT'] = df['DEPTREAT'].replace({8: np.nan, -4: np.nan})

    # Overwrite 88, 99, and -4 as NaN in the SMOKYRS column
    df['SMOKYRS'] = df['SMOKYRS'].replace({88: np.nan, 99: np.nan, -4: np.nan})

    # Clean quit smoking data
    df['QUITSMOK'] = df['QUITSMOK'].replace({888: np.nan, 999: np.nan, -4: np.nan})

    # Clean BMI data
    df['NACCBMI'] = df['NACCBMI'].replace({888.8: np.nan, -4: np.nan})
    df['NACCBMI'] = np.where(df['NACCBMI'] > 800, np.nan, df['NACCBMI'])

    # Clean depression scale data
    df['NACCGDS'] = df['NACCGDS'].replace({88: np.nan, -4: np.nan})

    # Clean education data
    df['EDUC'] = df['EDUC'].replace({99: np.nan})

    # Load and process PET centiloid data
    centiloids = pd.read_csv(nacc_centiloids_file_path)
    centiloids.rename(columns={'SCANDATE': 'VISITDATE'}, inplace=True)
    centiloids['VISITDATE'] = pd.to_datetime(centiloids['VISITDATE'])
    centiloids = centiloids.sort_values(by=['NACCID', 'VISITDATE'])

    # Find common subjects between clinical and PET data
    common_ids = set(centiloids.NACCID).intersection(set(df.NACCID))
    print(f"Number of subjects with both clinical and PET data: {len(common_ids)}")
    
    # Filter data to include only subjects with both clinical and PET data
    df = df[df.NACCID.isin(common_ids)]
    centiloids = centiloids[centiloids.NACCID.isin(common_ids)]

    # Calculate baseline visits and time from baseline
    earliest_baseline = find_first_or_last_visit([df, centiloids], id_col='NACCID', examdate_col='VISITDATE', first_or_last='first')
    latest_date = find_first_or_last_visit([centiloids, df], id_col='NACCID', examdate_col='VISITDATE', first_or_last='last')
    latest_date = map_time_from_baseline(latest_date, earliest_baseline, id_col='NACCID', examdate_col='VISITDATE')

    # Map time from baseline for both datasets
    centiloids = map_time_from_baseline(centiloids, earliest_baseline, id_col='NACCID', examdate_col='VISITDATE') 
    df = map_time_from_baseline(df, earliest_baseline, id_col='NACCID', examdate_col='VISITDATE')

    # Exclude anyone with dementia at their first visit
 
    # ### outcomes
    # DEMENTED - dementia or not
    # NACCADMD - Reported current use of a FDAapproved medication for Alzheimer’s disease symptoms
    # NACCALZD - Presumptive etiologic diagnosis of the cognitive disorder — Alzheimer’s disease
    # NACCALZP - Primary, contributing, or noncontributing cause of observed cognitive impairment — Alzheimer’s disease (AD)
    # PROBAD - Presumptive etiologic diagnosis of the cognitive disorder — Probable Alzheimer’s disease
    # PROBADIF - Primary, contributing, or noncontributing cause of cognitive impairment — Probable Alzheimer’s disease
    # POSSAD - Presumptive etiologic diagnosis of the cognitive disorder — Possible Alzheimer’s disease
    # POSSADIF - Primary, contributing, or noncontributing cause of cognitive impairment — Possible Alzheimer’s disease
    # NACCETPR - Primary etiologic diagnosis (MCI); impaired, not MCI; or dementia
        
    case_dx = df[(df.DEMENTED == 1) | 
                 (df.NACCADMD == 1) | 
                 (df.NACCALZD == 1) | 
                 (df.PROBAD == 1) | 
                 (df.PROBADIF == 1) | 
                 (df.POSSAD == 1) | 
                 (df.POSSADIF == 1) | 
                 (df.NACCETPR == 1)]
    
    # Get first diagnosis date for each case
    case_dx_first = case_dx.sort_values(by=['NACCID', 'VISITDATE']).drop_duplicates(subset=['NACCID'], keep='first')

    # Process PET scans for cases
    pet_case_df = []

    # For each case subject
    for c in case_dx_first.NACCID.unique():
        c_pet = centiloids[centiloids.NACCID == c]
        c_first_dx = case_dx_first[case_dx_first.NACCID == c]
        # Only include PET scans before diagnosis
        c_pet_before_dx = c_pet[c_pet['VISITDATE'] < c_first_dx['VISITDATE'].values[0]]
        if c_pet_before_dx.shape[0] == 0:
            continue
        else:
            # Add time-to-event information
            c_pet = c_pet.merge(c_first_dx[['NACCID', 'visit_to_days']], on='NACCID', how='left')   
            c_pet.rename({'visit_to_days_x': 'visit_to_days'}, axis=1, inplace=True)
            c_pet.rename({'visit_to_days_y': 'time_to_event'}, axis=1, inplace=True)
            pet_case_df.append(c_pet)

    # Combine all case PET data
    pet_case_df = pd.concat(pet_case_df, axis=0).reset_index(drop=True)

    # Process control subjects (those who never developed dementia)
    pet_control_df = centiloids[~centiloids.NACCID.isin(case_dx_first.NACCID.unique())]

    # Add time-to-event information for controls (time to last visit)
    pet_control_df = pet_control_df.merge(latest_date[['NACCID', 'visit_to_days']], on='NACCID', how='left')
    pet_control_df.rename({'visit_to_days_x': 'visit_to_days'}, axis=1, inplace=True)
    pet_control_df.rename({'visit_to_days_y': 'time_to_event'}, axis=1, inplace=True)

    # Combine case and control data
    pet_df = pd.concat([pet_case_df, pet_control_df], axis=0).reset_index(drop=True)
    pet_df['label'] = [1] * pet_case_df.shape[0] + [0] * pet_control_df.shape[0]

    # Add demographic information
    print(f"Initial dataset size: {pet_df.shape}")
    pet_df = pet_df.merge(df.loc[:, ['NACCID', 'BIRTHDAY', 'SEX', 'NACCAPOE', 'NACCNE4S']].drop_duplicates(), 
                         on='NACCID', how='left')
    print(f"Dataset size after adding demographics: {pet_df.shape}")

    # Calculate age at examination
    pet_df['age'] = (pet_df['VISITDATE'] - pet_df['BIRTHDAY']).dt.days / 365.25
    return pet_df


def process_habs(habs_file_path, habs_centiloids_file_path, habs_demo_file_path, habs_pib_file_path):
    # Load clinical diagnosis data
    dx = pd.read_csv(habs_file_path)
    dx['NP_SessionDate'] = pd.to_datetime(dx['NP_SessionDate'])
    dx.rename({'NP_SessionDate': 'EXAMDATE', 'SubjIDshort': 'RID'}, axis=1, inplace=True)
    dx = dx.sort_values(by=['RID', 'EXAMDATE']).reset_index(drop=True)

    # Clean subject IDs by removing underscores
    dx['RID'] = dx['RID'].str.replace('_', '')

    # Load and process PET centiloid data
    centiloids = pd.read_csv(habs_centiloids_file_path)
    centiloids['SCANDATE.AMY'] = pd.to_datetime(centiloids['SCANDATE.AMY'])
    centiloids.rename({'SCANDATE.AMY': 'EXAMDATE', 'ID': 'RID'}, axis=1, inplace=True)
    centiloids = centiloids.sort_values(by=['RID', 'EXAMDATE']).reset_index(drop=True)

    # Calculate baseline visits and time from baseline
    earliest_baseline = find_first_or_last_visit([centiloids, dx], first_or_last='first')
    latest_date = find_first_or_last_visit([centiloids, dx], first_or_last='last')
    latest_date = map_time_from_baseline(latest_date, earliest_baseline)

    # Map time from baseline for both datasets
    centiloids = map_time_from_baseline(centiloids, earliest_baseline) 
    dx = map_time_from_baseline(dx, earliest_baseline)

    # Identify cases (subjects who developed dementia)
    case_dx = dx[dx.HABS_DX.isin(['Dementia'])]
    case_dx_first = case_dx.drop_duplicates(subset=['RID'], keep='first')

    df = centiloids

    # Report number of unique subjects
    print(f"Number of unique subjects with PET data: {df.RID.nunique()}")

    # Process PET scans for cases
    pet_case_df = []

    # For each case subject
    for c in case_dx_first.RID.unique():
        c_pet = df[df.RID == c]
        c_first_dx = case_dx_first[case_dx_first.RID == c]
        # Only include PET scans before diagnosis
        c_pet_before_dx = c_pet[c_pet['EXAMDATE'] < c_first_dx['EXAMDATE'].values[0]]
        if c_pet_before_dx.shape[0] == 0:
            continue
        else:
            # Add time-to-event information
            c_pet = c_pet.merge(c_first_dx[['RID', 'visit_to_days']], on='RID', how='left')   
            c_pet.rename({'visit_to_days_x': 'visit_to_days'}, axis=1, inplace=True)
            c_pet.rename({'visit_to_days_y': 'time_to_event'}, axis=1, inplace=True)
            pet_case_df.append(c_pet)

    # Combine all case PET data
    pet_case_df = pd.concat(pet_case_df, axis=0).reset_index(drop=True)

    # Process control subjects (those who never developed dementia)
    pet_control_df = df[~df.RID.isin(case_dx_first.RID.unique())]

    # Add time-to-event information for controls (time to last visit)
    pet_control_df = pet_control_df.merge(latest_date[['RID', 'visit_to_days']], on='RID', how='left')
    pet_control_df.rename({'visit_to_days_x': 'visit_to_days'}, axis=1, inplace=True)
    pet_control_df.rename({'visit_to_days_y': 'time_to_event'}, axis=1, inplace=True)

    # Combine case and control data
    pet_df = pd.concat([pet_case_df, pet_control_df], axis=0).reset_index(drop=True)
    pet_df['label'] = [1] * pet_case_df.shape[0] + [0] * pet_control_df.shape[0]

    # Load and process demographic information
    demo = pd.read_csv(habs_demo_file_path)
    demo = demo.drop_duplicates(subset=['SubjID', 'BiologicalSex', 'YrsOfEd', 'Race', 'Ethnicity', 
                                      'Holingshead', 'APOE_haplotype', 'E4_Status']).reset_index(drop=True)
    demo = demo.rename({'SubjID': 'RID'}, axis=1)

    # Clean subject IDs by removing underscores
    demo['RID'] = demo['RID'].str.replace('_', '')

    # Add demographic information to PET data
    print(f"Initial dataset size: {pet_df.shape}")
    pet_df = pet_df.merge(demo.loc[:, ['RID', 'BiologicalSex', 'YrsOfEd', 'Race', 'Ethnicity', 
                                      'Holingshead', 'APOE_haplotype', 'E4_Status']], 
                         on='RID', how='left')
    print(f"Dataset size after adding demographics: {pet_df.shape}")

    # Load and process PIB PET data
    pib = pd.read_csv(habs_pib_file_path)
    pib.rename({'SubjIDshort': 'RID', 'PIB_SessionDate': 'EXAMDATE'}, axis=1, inplace=True)
    pib['RID'] = pib['RID'].str.replace('_', '')
    pib['EXAMDATE'] = pd.to_datetime(pib['EXAMDATE'])

    # Merge PIB PET data with main dataset
    pet_df = pet_df.merge(pib[['RID', 'EXAMDATE', 'PIB_Age']], on=['RID', 'EXAMDATE'], how='left')
    return pet_df


def process_adni(adni_pet_file_path, adni_dx_file_path, adni_apoe_file_path, adni_demo_file_path):
    # Load PET amyloid data from UC Berkeley analysis
    df = pd.read_csv(adni_pet_file_path)
    df.rename({'SCANDATE': 'EXAMDATE'}, axis=1, inplace=True)
    df['EXAMDATE'] = df['EXAMDATE'].apply(parse_dates)

    # Load clinical diagnosis data
    dx = pd.read_csv(adni_dx_file_path)
    dx = dx[dx.EXAMDATE.notna()]
    dx['EXAMDATE'] = dx['EXAMDATE'].apply(parse_dates)
    dx.sort_values(by=['RID', 'EXAMDATE'], inplace=True)

    # Load APOE genotype data
    apoe = pd.read_csv(adni_apoe_file_path)
    apoe['genotype'] = apoe['APGEN1'].astype(str) + apoe['APGEN2'].astype(str)

    # Load and clean demographic data
    demo = pd.read_csv(adni_demo_file_path)
    
    # Filter to valid gender codes (1=male, 2=female)
    demo = demo[demo.PTGENDER.isin([1,2])]

    # Select core demographic variables and remove duplicates
    demo = demo.loc[:, ['RID', 'PTGENDER', 'PTDOB', 'PTEDUCAT']].drop_duplicates()

    # Handle duplicate RIDs with missing education
    demo = demo[demo.PTEDUCAT.notnull()]

    # For remaining duplicates, take mean of education values
    demo = demo.groupby(['RID', 'PTGENDER', 'PTDOB'], as_index=False)['PTEDUCAT'].mean()

    # Calculate baseline visits and time from baseline
    earliest_baseline = find_first_or_last_visit([df, dx], first_or_last='first')
    latest_date = find_first_or_last_visit([df, dx], first_or_last='last')
    latest_date = map_time_from_baseline(latest_date, earliest_baseline)

    # Map time from baseline for both datasets
    df = map_time_from_baseline(df, earliest_baseline) 
    dx = map_time_from_baseline(dx, earliest_baseline)

    # Identify cases (subjects who developed dementia)
    # Case definition: DIAGNOSIS=3 (Dementia) or other dementia diagnosis (DXOTHDEM=1) or AD diagnosis (DXAD=1)
    case_dx = dx[(dx.DIAGNOSIS == 3) | (dx.DXOTHDEM == 1) | (dx.DXAD == 1)]
    case_dx_first = case_dx.drop_duplicates(subset=['RID'], keep='first')

    # Report number of unique subjects
    print(f"Number of unique subjects with PET data: {df.RID.nunique()}")

    # Process PET scans for cases
    pet_case_df = []

    # For each case subject
    for c in case_dx_first.RID.unique():
        c_pet = df[df.RID == c]
        c_first_dx = case_dx_first[case_dx_first.RID == c]
        # Only include PET scans before diagnosis
        c_pet_before_dx = c_pet[c_pet['EXAMDATE'] < c_first_dx['EXAMDATE'].values[0]]
        if c_pet_before_dx.shape[0] == 0:
            continue
        else:
            # Add time-to-event information
            c_pet = c_pet.merge(c_first_dx[['RID', 'visit_to_days']], on='RID', how='left')   
            c_pet.rename({'visit_to_days_x': 'visit_to_days'}, axis=1, inplace=True)
            c_pet.rename({'visit_to_days_y': 'time_to_event'}, axis=1, inplace=True)
            pet_case_df.append(c_pet)

    # Combine all case PET data
    pet_case_df = pd.concat(pet_case_df, axis=0).reset_index(drop=True)

    # Process control subjects (those who never developed dementia)
    pet_control_df = df[~df.RID.isin(case_dx_first.RID.unique())]

    # Add time-to-event information for controls (time to last visit)
    pet_control_df = pet_control_df.merge(latest_date[['RID', 'visit_to_days']], on='RID', how='left')
    pet_control_df.rename({'visit_to_days_x': 'visit_to_days'}, axis=1, inplace=True)
    pet_control_df.rename({'visit_to_days_y': 'time_to_event'}, axis=1, inplace=True)

    # Combine case and control data
    pet_df = pd.concat([pet_case_df, pet_control_df], axis=0).reset_index(drop=True)
    pet_df['label'] = [1] * pet_case_df.shape[0] + [0] * pet_control_df.shape[0]

    # Add demographic information
    pet_df = pet_df.merge(demo[['RID', 'PTGENDER', 'PTDOB', 'PTEDUCAT']].drop_duplicates(), on='RID', how='left')

    # Calculate age at examination
    pet_df['PTDOB'] = pd.to_datetime(pet_df['PTDOB'], format='%m/%Y')
    pet_df['EXAMDATE'] = pd.to_datetime(pet_df['EXAMDATE'])
    pet_df['age'] = pet_df['EXAMDATE'].dt.year - pet_df['PTDOB'].dt.year

    # Add APOE genotype information
    pet_df = pet_df.merge(apoe[['RID', 'genotype']], on='RID', how='left')
    return pet_df


def process_aibl(aibl_cdr_file_path, aibl_centiloids_file_path, aibl_dx_file_path, aibl_demo_file_path, aibl_apoe_file_path):
    # Load and preprocess CDR (Clinical Dementia Rating) data
    cdr = pd.read_csv(aibl_cdr_file_path)
    cdr['EXAMDATE'] = cdr['EXAMDATE'].apply(parse_dates)
    cdr = cdr[cdr.EXAMDATE.notna()]

    # Load and preprocess PET centiloid measurements
    centiloids = pd.read_csv(aibl_centiloids_file_path)
    centiloids.rename({'EXAMDATE.AMY': 'EXAMDATE'}, axis=1, inplace=True)
    centiloids['EXAMDATE'] = centiloids['EXAMDATE'].apply(parse_dates)
    centiloids = centiloids[centiloids.EXAMDATE.notna()]

    # Calculate baseline visits and time from baseline
    earliest_baseline = find_first_or_last_visit([cdr, centiloids], first_or_last='first')
    latest_date = find_first_or_last_visit([cdr, centiloids], first_or_last='last')
    latest_date = map_time_from_baseline(latest_date, earliest_baseline)

    # Load clinical diagnosis data
    dx = pd.read_csv(aibl_dx_file_path)
    
    # Filter to subjects with either CDR or PET data
    dx = dx[dx.RID.isin(set(cdr.RID).union(set(centiloids.RID)))]

    # Merge diagnosis data with CDR data to get examination dates
    dx = dx.merge(cdr[['RID', 'VISCODE', 'EXAMDATE']], on=['RID', 'VISCODE'], how='left')
    dx.sort_values(by=['RID', 'EXAMDATE'], inplace=True)

    # Map time from baseline for all datasets
    cdr = map_time_from_baseline(cdr, earliest_baseline) 
    dx = map_time_from_baseline(dx, earliest_baseline)
    centiloids = map_time_from_baseline(centiloids, earliest_baseline)

    # Identify cases (subjects who developed dementia)
    # Case definition: DXCURREN > 2 (same as ADNI)
    case_dx = dx[dx.DXCURREN > 2]
    case_dx_first = case_dx.drop_duplicates(subset=['RID'], keep='first')

    # Report number of unique subjects
    print(f"Number of unique subjects with PET data: {centiloids.RID.nunique()}")

    # Process PET scans for cases
    pet_case_df = []

    # For each case subject
    for c in case_dx_first.RID.unique():
        c_pet = centiloids[centiloids.RID == c]
        c_first_dx = case_dx_first[case_dx_first.RID == c]
        # Only include PET scans before diagnosis
        c_pet_before_dx = c_pet[c_pet['EXAMDATE'] < c_first_dx['EXAMDATE'].values[0]]
        if c_pet_before_dx.shape[0] == 0:
            continue
        else:
            # Add time-to-event information
            c_pet = c_pet.merge(c_first_dx[['RID', 'visit_to_days']], on='RID', how='left')   
            c_pet.rename({'visit_to_days_x': 'visit_to_days'}, axis=1, inplace=True)
            c_pet.rename({'visit_to_days_y': 'time_to_event'}, axis=1, inplace=True)
            pet_case_df.append(c_pet)

    # Combine all case PET data
    pet_case_df = pd.concat(pet_case_df, axis=0).reset_index(drop=True)

    # Process control subjects (those who never developed dementia)
    pet_control_df = centiloids[~centiloids.RID.isin(case_dx_first.RID.unique())]

    # Add time-to-event information for controls (time to last visit)
    pet_control_df = pet_control_df.merge(latest_date[['RID', 'visit_to_days']], on='RID', how='left')
    pet_control_df.rename({'visit_to_days_x': 'visit_to_days'}, axis=1, inplace=True)
    pet_control_df.rename({'visit_to_days_y': 'time_to_event'}, axis=1, inplace=True)

    # Combine case and control data
    pet_df = pd.concat([pet_case_df, pet_control_df], axis=0).reset_index(drop=True)
    pet_df['label'] = [1] * pet_case_df.shape[0] + [0] * pet_control_df.shape[0]

    # Load and process demographic information
    demo = pd.read_csv(aibl_demo_file_path)
    pet_df = pet_df.merge(demo[['RID', 'PTGENDER', 'PTDOB']].drop_duplicates(), on='RID', how='left')

    # Calculate age at examination
    pet_df['PTDOB'] = pd.to_datetime(pet_df['PTDOB'], format='%Y')
    pet_df['EXAMDATE'] = pd.to_datetime(pet_df['EXAMDATE'])
    pet_df['age'] = pet_df['EXAMDATE'].dt.year - pet_df['PTDOB'].dt.year

    # Add APOE genotype information
    apoe = pd.read_csv(aibl_apoe_file_path)
    apoe['genotype'] = apoe['APGEN1'].astype(str) + apoe['APGEN2'].astype(str)
    pet_df = pet_df.merge(apoe[['RID', 'genotype']], on='RID', how='left')
    return pet_df


# Create five datasets for five-fold cross-validation, stratified by label and time-to-event
def create_stratified_folds(data, n_splits=5, n_bins=4, random_state=42):
    # Aggregate data by id, keeping label and time_to_event
    bids = (data[['id', 'event', 'time_to_event']]
            .sort_values(['event', 'time_to_event'], ascending=[False, True])
            .drop_duplicates('id')
            .reset_index(drop=True))
    
    # Verify event consistency
    # assert bids['event'].nunique() == data.groupby('BID')['label'].nunique().max(), \
    #     "Not all BIDs have a consistent event. Please resolve label inconsistencies."
    
    # Create time-to-event bins only for cases (event=1)
    cases = bids[bids['event'] == 1].copy()
    controls = bids[bids['event'] == 0].copy()
    
    # Create bins for cases based on time-to-event
    cases['time_bin'] = pd.qcut(cases['time_to_event'], q=n_bins, labels=False)
    # Assign -1 as time_bin for controls
    controls['time_bin'] = -1
    
    # Combine cases and controls
    bids = pd.concat([cases, controls])
    
    # Create a composite stratification variable
    bids['strata'] = bids['event'].astype(str) + '_' + bids['time_bin'].astype(str)
    
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
def process_fold(data, fold_assignments, fold):
    # Merge fold assignments with original data
    data2 = data.merge(fold_assignments, on='id')
    
    # Define validation and training sets
    val_set = data2[data2['fold'] == fold].copy()
    train_set = data2[data2['fold'] != fold].copy()
    
    # Print fold information
    print(f"Fold {fold + 1}:")
    print(f"  Training IDs: {train_set['id'].nunique()}")
    print(f"  Validation IDs: {val_set['id'].nunique()}")
    print(f"  Positive class in Training: {train_set['event'].mean() * 100:.2f}%")
    print(f"  Positive class in Validation: {val_set['event'].mean() * 100:.2f}%")
    
    # Print time-to-event distribution for cases
    train_cases = train_set[train_set['event'] == 1]
    val_cases = val_set[val_set['event'] == 1]
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
    
    train_set['age_centered'] = train_set.age - training_age_mean
    val_set['age_centered'] = val_set.age - training_age_mean
    
    train_set['age_centered_squared'] = train_set.age_centered ** 2
    val_set['age_centered_squared'] = val_set.age_centered ** 2

    #zscore centiloids
    training_centiloids_mean = train_set.centiloids.mean()
    training_centiloids_std = train_set.centiloids.std()
    
    train_set['centiloids_z'] = (train_set.centiloids - training_centiloids_mean) #/ training_centiloids_std
    val_set['centiloids_z'] = (val_set.centiloids - training_centiloids_mean) #/ training_centiloids_std

    train_set['centiloids_z_squared'] = train_set.centiloids_z ** 2
    val_set['centiloids_z_squared'] = val_set.centiloids_z ** 2

    # set id column to string
    train_set['id'] = train_set['id'].astype(str)
    val_set['id'] = val_set['id'].astype(str)

    
    return train_set, val_set


def replace_apoe(df):
        df.loc[df.apoe.isin([23, 23.0, 32, 32.0, '32']), 'apoe'] = '23'
        df.loc[df.apoe.isin([24, 24.0, 42, 42.0, '42']), 'apoe'] = '24'
        df.loc[df.apoe.isin([34, 34.0, 43, 43.0, '43']), 'apoe'] = '34'
        df.loc[df.apoe.isin([22, 22.0, '22']), 'apoe'] = 'e2_carriers'
        df.loc[df.apoe.isin([23, 23.0, '23']), 'apoe'] = 'e2_carriers'
        df.loc[df.apoe.isin([33, 33.0]), 'apoe'] = '33'
        df.loc[df.apoe.isin([44, 44.0]), 'apoe'] = '44'
        df.loc[df.apoe == "-4-4", 'apoe'] = np.nan # AIBL
        df.loc[df.apoe == 9, 'apoe'] = np.nan # NACC


        # convert apoe column to string
        df['apoe'] = df['apoe'].astype('category')
        return df

def add_one_day_to_zero_time(df):
    df.loc[df.time == 0, 'time'] = 1
    df.loc[df.time_to_event == 0, 'time_to_event'] = 1
    return df

def make_cv_folds(output_path):
    # Load processed datasets
    adni = pd.read_parquet(output_path + 'adni.parquet')
    adni = adni.loc[:, ["RID", "CENTILOIDS", "visit_to_days",
                        "time_to_event", "label", "PTGENDER",
                        "age", "genotype"]]
    adni.columns = ["id", "centiloids", "time", "time_to_event",
                    "event", "sex", "age", "apoe"]
    adni = replace_apoe(adni)

    # divide time and time_to_event by 365.25
    adni = add_one_day_to_zero_time(adni)
    adni['time'] = adni['time'] / 365.25
    adni['time_to_event'] = adni['time_to_event'] / 365.25


    aibl = pd.read_parquet(output_path + 'aibl.parquet')
    aibl = aibl.loc[:, ["RID", "CENTILOIDS", "visit_to_days",
                        "time_to_event", "label", "PTGENDER",
                        "age", "genotype"]]
    aibl.columns = ["id", "centiloids", "time", "time_to_event",
                    "event", "sex", "age", "apoe"]
    aibl = replace_apoe(aibl)

    # divide time and time_to_event by 365.25
    aibl = add_one_day_to_zero_time(aibl)
    aibl['time'] = aibl['time'] / 365.25
    aibl['time_to_event'] = aibl['time_to_event'] / 365.25


    habs = pd.read_parquet(output_path + 'habs.parquet')
    habs = habs.loc[:, ["RID", "CENTILOIDS", "visit_to_days",
                        "time_to_event", "label", "BiologicalSex",
                        "PIB_Age", "APOE_haplotype"]]
    habs.columns = ["id", "centiloids", "time", "time_to_event",
                    "event", "sex", "age", "apoe"]
    habs.loc[habs.sex == "F", 'sex'] = 2
    habs.loc[habs.sex == "M", 'sex'] = 1
    habs = replace_apoe(habs)

    # divide time and time_to_event by 365.25
    habs = add_one_day_to_zero_time(habs)
    habs['time'] = habs['time'] / 365.25
    habs['time_to_event'] = habs['time_to_event'] / 365.25

    nacc = pd.read_parquet(output_path + 'nacc.parquet')
    nacc = nacc.loc[:, ["NACCID", "CENTILOIDS", "visit_to_days",
                        "time_to_event", "label", "SEX",
                        "age", "NACCAPOE"]]
    nacc.columns = ["id", "centiloids", "time", "time_to_event",
                    "event", "sex", "age", "apoe"]
    # https://files.alz.washington.edu/documentation/dervarprev.pdf
    # 1 = 33
    # 2 = 34
    # 3 = 23
    # 4 = 44
    # 5 = 24
    # 6 = 22
    # 9 = NA
    nacc.loc[nacc.apoe == 1, 'apoe'] = '33'
    nacc.loc[nacc.apoe == 2, 'apoe'] = '34'
    nacc.loc[nacc.apoe == 3, 'apoe'] = '23'
    nacc.loc[nacc.apoe == 4, 'apoe'] = '44'
    nacc.loc[nacc.apoe == 5, 'apoe'] = '24'
    nacc.loc[nacc.apoe == 6, 'apoe'] = '22'
    nacc = replace_apoe(nacc)

    # divide time and time_to_event by 365.25
    nacc = add_one_day_to_zero_time(nacc)
    nacc['time'] = nacc['time'] / 365.25
    nacc['time_to_event'] = nacc['time_to_event'] / 365.25

    oasis = pd.read_parquet(output_path + 'oasis.parquet')
    oasis = oasis.loc[:, ["ID", "Centiloid_fSUVR_TOT_CORTMEAN", "visit_to_days",
                            "time_to_event", "label", "GENDER",
                            "age", "APOE"]]
    oasis.columns = ["id", "centiloids", "time", "time_to_event",
                    "event", "sex", "age", "apoe"]
    oasis = replace_apoe(oasis)

    # divide time and time_to_event by 365.25
    oasis = add_one_day_to_zero_time(oasis)
    oasis['time'] = oasis['time'] / 365.25
    oasis['time_to_event'] = oasis['time_to_event'] / 365.25

    # concatenate all datasets and create stratified folds
    data = pd.concat([adni, aibl, habs, nacc, oasis])

    #### Create folds for each cohort
    for i, val_set in enumerate([adni, aibl, habs, nacc, oasis]):
        train_set = [ds for j, ds in enumerate([adni, aibl, habs, nacc, oasis]) if j != i]
        train_set = pd.concat(train_set)

        # Preprocess data
        # zscore AGEYR
        training_age_mean = train_set.age.mean()
        training_age_std = train_set.age.std()
        
        train_set['age_centered'] = train_set.age - training_age_mean
        val_set['age_centered'] = val_set.age - training_age_mean
        
        train_set['age_centered_squared'] = train_set.age_centered ** 2
        val_set['age_centered_squared'] = val_set.age_centered ** 2

        #zscore centiloids
        training_centiloids_mean = train_set.centiloids.mean()
        training_centiloids_std = train_set.centiloids.std()
        
        train_set['centiloids_z'] = (train_set.centiloids - training_centiloids_mean) #/ training_centiloids_std
        val_set['centiloids_z'] = (val_set.centiloids - training_centiloids_mean) #/ training_centiloids_std

        train_set['centiloids_z_squared'] = train_set.centiloids_z ** 2
        val_set['centiloids_z_squared'] = val_set.centiloids_z ** 2

        # set id column to string
        train_set['id'] = train_set['id'].astype(str)
        val_set['id'] = val_set['id'].astype(str)

        train_set.to_parquet(output_path + f'train_{i}_test_by_cohort.parquet')
        val_set.to_parquet(output_path + f'val_{i}_test_by_cohort.parquet')


    #### Create stratified folds combining all cohorts
    # fold_assignments = create_stratified_folds(data)

    # for fold in range(5):
    #     # Process the fold
    #     train_set, val_set = process_fold(data, fold_assignments, fold)

    #     # print overlapping BIDs between training and validation sets
    #     train_bids = set(train_set['id'])
    #     val_bids = set(val_set['id'])
    #     overlap = train_bids.intersection(val_bids)
    #     print(f"  Overlapping IDs: {len(overlap)}\n")

    #     # Save datasets
    #     train_set.to_parquet(output_path + f'train_{fold}_new.parquet')
    #     val_set.to_parquet(output_path + f'val_{fold}_new.parquet')

    
if __name__ == '__main__':
    # set file paths for cohorts
    main_path = '../../../datasets/'

    oasis_file_path = main_path + 'OASIS/csv/raw/OASIS3_amyloid_centiloid.csv'
    oasis_diagnoses_file_path = main_path + 'OASIS/csv/raw/OASISdiagTable.csv'
    oasis_demo_file_path = main_path + 'OASIS/csv/raw/OASIS3_demographics.csv'

    nacc_file_path = '../../pet_all_cohorts/tidy_data/investigator_ftldlbd_nacc65.parquet'
    nacc_centiloids_file_path = main_path + 'NACC/csv/raw/investigator_scan_pet_nacc65/investigator_scan_amyloidpetgaain_nacc65.csv'

    habs_file_path = main_path + 'HABS/csv/raw/ClinicalMeasures_HABS_DataRelease_2.0.csv'
    habs_centiloids_file_path = main_path + 'HABS/csv/processed/HABS_CENTILOIDS.csv'
    habs_demo_file_path = main_path + 'HABS/csv/raw/Demographics_HABS_DataRelease_2.0.csv'
    habs_pib_file_path = main_path + 'HABS/csv/raw/PIB_FS6_DVR_HABS_DataRelease_2.0.csv'

    adni_pet_file_path = main_path + 'ALL_ADNI_LONIdownload/csv/raw/PET_UCBerkeley/UCBERKELEY_AMY_6MM_17Oct2023.csv'
    adni_dx_file_path = main_path + 'ALL_ADNI_LONIdownload/csv/raw/DXSUM_PDXCONV_ADNIALLcsv.csv'
    adni_apoe_file_path = main_path + 'ALL_ADNI_LONIdownload/csv/raw/APOERES_01Jun2023.csv'
    adni_demo_file_path = main_path + 'ALL_ADNI_LONIdownload/csv/raw/PTDEMOG_29Nov2023.csv'

    aibl_cdr_file_path = main_path + 'AIBL/csv/raw/aibl_cdr_01-Jun-2018.csv'
    aibl_centiloids_file_path = main_path + 'AIBL/csv/processed/AIBL_CENTILOIDS_varuna.csv'
    aibl_dx_file_path = main_path + 'AIBL/csv/raw/aibl_pdxconv_01-Jun-2018.csv'
    aibl_demo_file_path = main_path + 'AIBL/csv/raw/aibl_ptdemog_01-Jun-2018.csv'
    aibl_apoe_file_path = main_path + 'AIBL/csv/raw/aibl_apoeres_01-Jun-2018.csv'

    output_path = '../../pet_all_cohorts/tidy_data/'

    
    #######################
    # Process OASIS Data #
    #######################
    oasis_df = process_oasis(oasis_file_path, oasis_diagnoses_file_path, oasis_demo_file_path)
    oasis_df.to_parquet(output_path + 'oasis.parquet')


    #######################
    # Process NACC Data #
    #######################
    nacc_df = process_nacc(nacc_file_path, nacc_centiloids_file_path)
    nacc_df.to_parquet(output_path + 'nacc.parquet')


    #######################
    # Process HABS Data #
    #######################
    habs_df = process_habs(habs_file_path, habs_centiloids_file_path, habs_demo_file_path, habs_pib_file_path)
    habs_df.to_parquet(output_path + 'habs.parquet')


    #######################
    # Process ADNI Data #
    #######################
    adni_df = process_adni(adni_pet_file_path, adni_dx_file_path, adni_apoe_file_path, adni_demo_file_path)
    adni_df.to_parquet(output_path + 'adni.parquet')


    #######################
    # Process AIBL Data #
    #######################
    aibl_df = process_aibl(aibl_cdr_file_path, aibl_centiloids_file_path, aibl_dx_file_path, aibl_demo_file_path, aibl_apoe_file_path)
    aibl_df.to_parquet(output_path + 'aibl.parquet')


    #################################
    # Create cross-validation folds #
    #################################
    make_cv_folds(output_path)