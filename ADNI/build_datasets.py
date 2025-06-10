# Import required libraries
import pandas as pd
import sys
import os

# Add pet module directory to path and import functions
sys.path.append("../pet")
from build_pet_datasets import find_first_or_last_visit, map_time_from_baseline

# Add A4 module directory to path and import functions
sys.path.append("../A4")
from build_a4_datasets import create_stratified_folds, process_fold


def format_data(path, earliest_baseline):
    """Format data by standardizing date column and filtering by RID"""
    df = pd.read_csv(path)
    df.rename({"VISDATE": "EXAMDATE"}, axis=1, inplace=True)
    df = df[df.RID.isin(earliest_baseline.RID)]
    df.EXAMDATE = pd.to_datetime(df.EXAMDATE, format="%Y-%m-%d", errors="coerce")
    return df


# Define paths to various input data files
main_path = "../../raw_data/ADNI/"
pet_path = main_path + "PET_Image_Acquisition/UCBERKELEY_AMY_6MM_05Mar2025.csv"
ptau_path = (
    main_path + "Biospecimen_Results/UPENN_PLASMA_FUJIREBIO_QUANTERIX_05Mar2025.csv"
)
dx_path = main_path + "Diagnosis/DXSUM_06Mar2025.csv"
apoe_path = main_path + "Demographics/APOERES_06Mar2025.csv"
demo_path = main_path + "Demographics/PTDEMOG_06Mar2025.csv"
depression_path = main_path + "GDSCALE_06Mar2025.csv"
medhist_path = main_path + "_Medical_History/MEDHIST_11Mar2025.csv"
neuroexm_path = main_path + "_Medical_History/NEUROEXM_11Mar2025.csv"

# Path for Nightingale data
adni_nightingale_path = (
    main_path + "Biospecimen_Results 2/ADNINIGHTINGALELONG_05_24_21_11Mar2025.csv"
)

# Path for MODHACH data
modhach_path = main_path + "MODHACH_11Mar2025.csv"

# Path for vitals data
vitals_path = main_path + "Physical_Neurological_Exams/VITALS_06Mar2025.csv"

# Load and initially process main dataframes
pet_df = pd.read_csv(pet_path, low_memory=False)
pet = pet_df[pet_df.qc_flag > 0]  # Filter for quality control
dx_df = pd.read_csv(dx_path)
apoe_df = pd.read_csv(apoe_path)
demo_df = pd.read_csv(demo_path)
ptau_df = pd.read_csv(ptau_path, low_memory=False)

# Flag for subsetting amyloid positive cases
for subset_amyloid_positive in [True, False]:

    print(f"subsetting amyloid positive: {subset_amyloid_positive}")

    pet = pet_df[pet_df.qc_flag > 0]

    # Set save path based on amyloid positive subsetting
    if subset_amyloid_positive:
        save_path = "../../tidy_data/ADNI/amyloid_positive/"
        pet = pet[pet.AMYLOID_STATUS_COMPOSITE_REF == 1]
    else:
        save_path = "../../tidy_data/ADNI/"
    os.makedirs(save_path, exist_ok=True)

    # Process ptau data
    ptau = ptau_df.copy()
    print(f"starting shape of ptau:, {ptau.shape}")
    print(f"number of unique subjects in ptau:, {ptau.RID.nunique()}")
    ptau.EXAMDATE = pd.to_datetime(ptau.EXAMDATE, format="%Y-%m-%d", errors="coerce")

    # Process PET data
    pet.rename({"SCANDATE": "EXAMDATE"}, axis=1, inplace=True)
    pet.EXAMDATE = pd.to_datetime(pet.EXAMDATE, format="%Y-%m-%d", errors="coerce")
    pet = pet.iloc[:, :20]  # Keep only first 20 columns
    print(f"starting shape of pet:, {pet.shape}")
    print(f"number of unique subjects in pet:, {pet.RID.nunique()}")

    # Process diagnosis data
    dx = dx_df[dx_df.EXAMDATE.notna()]  # Remove rows with missing exam dates
    dx.EXAMDATE = pd.to_datetime(dx.EXAMDATE, format="%Y-%m-%d", errors="coerce")
    dx.sort_values(by=["RID", "EXAMDATE"], inplace=True)

    # Process demographic data
    demo = demo_df.rename({"VISDATE": "EXAMDATE"}, axis=1)
    demo.EXAMDATE = pd.to_datetime(demo.EXAMDATE, format="%Y-%m-%d", errors="coerce")

    # Find common IDs between PET and ptau data
    ptau_pet_common_ids = set(ptau.RID)  # & set(pet.RID)
    print(f"Number of ptau subjects with PET data: {len(ptau_pet_common_ids)}")

    # Filter all datasets to include only subjects with both PET and ptau data
    ptau = ptau[ptau.RID.isin(ptau_pet_common_ids)].reset_index(drop=True)
    pet = pet[pet.RID.isin(ptau_pet_common_ids)].reset_index(drop=True)
    demo = demo[demo.RID.isin(ptau_pet_common_ids)].reset_index(drop=True)
    dx = dx[dx.RID.isin(ptau_pet_common_ids)].reset_index(drop=True)

    # Find earliest baseline visit and latest visit dates
    earliest_baseline = find_first_or_last_visit(
        [ptau, pet, demo, dx], first_or_last="first"
    )
    latest_date = find_first_or_last_visit([ptau, pet, demo, dx], first_or_last="last")
    latest_date = map_time_from_baseline(latest_date, earliest_baseline)

    # Calculate time from baseline for all datasets
    ptau = map_time_from_baseline(ptau, earliest_baseline)
    pet = map_time_from_baseline(pet, earliest_baseline)
    demo = map_time_from_baseline(demo, earliest_baseline)
    dx = map_time_from_baseline(dx, earliest_baseline)

    # Filter to valid gender codes (1=male, 2=female)
    print(f"starting shape of demographics:, {demo.shape}")
    demo = demo[demo.PTGENDER.isin([1, 2])]
    print(f"filtering by sex, {demo.shape}")

    # Select core demographic variables and remove duplicates
    demo = demo.loc[:, ["RID", "PTGENDER", "PTDOB", "PTEDUCAT"]].drop_duplicates()
    print(f"dropping demographics duplicates {demo.shape}")
    # Handle duplicate RIDs with missing education
    demo = demo[demo.PTEDUCAT.notnull()]
    print(f"removing rows with no education {demo.shape}")
    # For remaining duplicates, take mean of education values
    demo = demo.groupby(["RID", "PTGENDER", "PTDOB"], as_index=False)["PTEDUCAT"].mean()

    # Identify cases (subjects who developed dementia)
    # Case definition: DIAGNOSIS=3 (Dementia) or other dementia diagnosis (DXOTHDEM=1) or AD diagnosis (DXAD=1)
    case_dx = dx[(dx.DIAGNOSIS == 3) | (dx.DXOTHDEM == 1) | (dx.DXAD == 1)]
    case_dx_first = case_dx.drop_duplicates(subset=["RID"], keep="first")
    print(f"number of unique subjects in case_dx_first:, {case_dx_first.RID.nunique()}")

    # Process ptau measurements for cases
    ptau_case_df = []
    # pet_case_df = []

    n_remove = 0
    # For each case subject
    for c in case_dx_first.RID.unique():
        c_ptau = ptau[ptau.RID == c]
        c_pet = pet[pet.RID == c]
        c_first_dx = case_dx_first[case_dx_first.RID == c]

        # Only include ptau measurements and PET scans before diagnosis
        c_ptau_before_dx = c_ptau[c_ptau["EXAMDATE"] < c_first_dx["EXAMDATE"].values[0]]
        # c_pet_before_dx = c_pet[c_pet["EXAMDATE"] < c_first_dx["EXAMDATE"].values[0]]
        if c_ptau_before_dx.shape[0] == 0:  # or c_pet_before_dx.shape[0] == 0:
            n_remove += 1
            continue
        else:
            # Add time-to-event information
            c_ptau = c_ptau.merge(
                c_first_dx[["RID", "visit_to_years"]], on="RID", how="left"
            )
            c_ptau.rename({"visit_to_years_x": "visit_to_years"}, axis=1, inplace=True)
            c_ptau.rename({"visit_to_years_y": "time_to_event"}, axis=1, inplace=True)
            ptau_case_df.append(c_ptau)
            # pet_case_df.append(c_pet)
    print(
        f"Number of cases removed due to no pTau measurements or PET scans before diagnosis: {n_remove}"
    )

    # Combine all case PET data
    ptau_case_df = pd.concat(ptau_case_df, axis=0).reset_index(drop=True)
    # pet_case_df = pd.concat(pet_case_df, axis=0).reset_index(drop=True)
    # Process control subjects (those who never developed dementia)
    ptau_control_df = ptau[~ptau.RID.isin(case_dx_first.RID.unique())]
    # pet_control_df = pet[~pet.RID.isin(case_dx_first.RID.unique())]

    # exclusions for controls
    exclusions = dx[
        (dx.DIAGNOSIS > 1)
        | (dx.DXMCI == 1)
        | (dx.DXMPTR2 == 1)
        | (dx.DXMPTR5 == 1)
        | (dx.DXMPTR6 == 0)
        | (dx.DXPARK == 1)
        | (dx.DXPATYP == 1)
    ]
    exclusions = exclusions[exclusions.RID.isin(ptau_control_df.RID)]
    # Remove excluded subjects from control group
    print(f"Number of controls (before exclusions): {ptau_control_df.RID.nunique()}")
    print(f"Number of excluded subjects: {exclusions.RID.nunique()}")
    ptau_control_df = ptau_control_df[~ptau_control_df.RID.isin(exclusions.RID)]
    # pet_control_df = pet_control_df[~pet_control_df.RID.isin(exclusions.RID)]
    # Add time-to-event information for controls (time to last visit)
    ptau_control_df = ptau_control_df.merge(
        latest_date[["RID", "visit_to_years"]], on="RID", how="left"
    )
    ptau_control_df.rename({"visit_to_years_x": "visit_to_years"}, axis=1, inplace=True)
    ptau_control_df.rename({"visit_to_years_y": "time_to_event"}, axis=1, inplace=True)

    # Combine case and control data
    print(f"case data shape: {ptau_case_df.shape}")
    print(f"control data shape: {ptau_control_df.shape}")
    print(f"number of unique subjects in case data: {ptau_case_df.RID.nunique()}")
    print(f"number of unique subjects in control data: {ptau_control_df.RID.nunique()}")
    ptau = pd.concat([ptau_case_df, ptau_control_df], axis=0).reset_index(drop=True)
    ptau["label"] = [1] * ptau_case_df.shape[0] + [0] * ptau_control_df.shape[0]
    print(f"ptau.RID.nunique(): {ptau.RID.nunique()}")

    # Add demographic information
    ptau = ptau.merge(
        demo[["RID", "PTGENDER", "PTDOB", "PTEDUCAT"]].drop_duplicates(),
        on="RID",
        how="left",
    )

    # Calculate age at examination
    ptau["PTDOB"] = pd.to_datetime(ptau["PTDOB"], format="%m/%Y")
    ptau["EXAMDATE"] = pd.to_datetime(ptau["EXAMDATE"])
    ptau["age"] = ptau["EXAMDATE"].dt.year - ptau["PTDOB"].dt.year

    # Add APOE genotype information
    ptau = ptau.merge(apoe_df[["RID", "GENOTYPE"]], on="RID", how="left")

    ptau.rename(
        columns={"RID": "id", "PTEDUCAT": "educ", "pT217_F": "ptau"}, inplace=True
    )

    ptau.GENOTYPE.replace({"2/3": "E2_carrier", "2/2": "E2_carrier"}, inplace=True)
    print(ptau.GENOTYPE.value_counts())

    data = ptau
    fold_assignments = create_stratified_folds(data)
    val_sets = []

    for fold in range(5):
        # Process the fold
        train_set, val_set = process_fold(data, "id", fold_assignments, fold)

        # print overlapping BIDs between training and validation sets
        train_bids = set(train_set["id"])
        val_bids = set(val_set["id"])
        overlap = train_bids.intersection(val_bids)
        print(f"  Overlapping BIDs: {len(overlap)}\n")

        # Save datasets
        train_set.to_parquet(f"{save_path}/train_{fold}.parquet")
        val_set.to_parquet(f"{save_path}/val_{fold}.parquet")

    depression = format_data(depression_path, earliest_baseline)
    depression = depression[depression.RID.isin(earliest_baseline.RID)]
    depression = map_time_from_baseline(depression, earliest_baseline)
    depression.to_parquet(f"{save_path}/depression.parquet")

    medhist = format_data(medhist_path, earliest_baseline)
    medhist = medhist[medhist.RID.isin(earliest_baseline.RID)]
    medhist = map_time_from_baseline(medhist, earliest_baseline)
    medhist.to_parquet(f"{save_path}/medhist.parquet")

    neuroexm = format_data(neuroexm_path, earliest_baseline)
    neuroexm = neuroexm[neuroexm.RID.isin(earliest_baseline.RID)]
    neuroexm = map_time_from_baseline(neuroexm, earliest_baseline)
    neuroexm.to_parquet(f"{save_path}/neuroexm.parquet")

    adni_nightingale = format_data(adni_nightingale_path, earliest_baseline)
    adni_nightingale = adni_nightingale[
        adni_nightingale.RID.isin(earliest_baseline.RID)
    ]
    adni_nightingale = map_time_from_baseline(adni_nightingale, earliest_baseline)
    adni_nightingale = adni_nightingale.loc[:, ["RID", "visit_to_years", "LDL_C"]]
    # adni_nightingale.GLN = adni_nightingale.GLN.astype(float)
    # adni_nightingale.PYRUVATE = adni_nightingale.PYRUVATE.astype(float)
    # adni_nightingale.CREATININE = adni_nightingale.CREATININE.astype(float)
    adni_nightingale.to_parquet(f"{save_path}/adni_nightingale.parquet")

    modhach = format_data(modhach_path, earliest_baseline)
    modhach = modhach[modhach.RID.isin(earliest_baseline.RID)]
    modhach = map_time_from_baseline(modhach, earliest_baseline)
    modhach.to_parquet(f"{save_path}/modhach.parquet")

    vitals = format_data(vitals_path, earliest_baseline)
    vitals = vitals[vitals.RID.isin(earliest_baseline.RID)]
    # VSWEIGHT	1a. Weight
    # VSWTUNIT	1b. Weight Units - 1=pounds, 2=kilograms
    # VSHEIGHT	2a. Height
    # VSHTUNIT	2b. Height Units - 1=inches, 2=centimeters

    # convert all heights to meters, whether in inches or centimeters
    vitals["height_m"] = vitals["VSHEIGHT"]
    vitals.loc[vitals["VSHTUNIT"] == 1, "height_m"] = vitals["VSHEIGHT"] * 0.0254
    vitals.loc[vitals["VSHTUNIT"] == 2, "height_m"] = vitals["VSHEIGHT"] / 100

    # convert all weights to kilograms, whether in pounds or kilograms
    vitals["weight_kg"] = vitals["VSWEIGHT"]
    vitals.loc[vitals["VSWTUNIT"] == 1, "weight_kg"] = vitals["VSWEIGHT"] * 0.453592
    vitals.loc[vitals["VSWTUNIT"] == 2, "weight_kg"] = vitals["VSWEIGHT"]

    vitals["BMI"] = vitals["weight_kg"] / (vitals["height_m"] ** 2)
    vitals = map_time_from_baseline(vitals, earliest_baseline)
    vitals.to_parquet(f"{save_path}/vitals.parquet")
