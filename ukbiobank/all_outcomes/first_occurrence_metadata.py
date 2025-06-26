import pandas as pd

# Save metadata on outcomes

df = pd.read_parquet('../tidy_data/proteomics_first_occurrences.parquet')
dd = pd.read_csv('../../proj_idp/Data_Dictionary_Showcase.csv',
                 usecols=(range(10)))
outcomes = pd.read_csv('../tidy_data/outcome_colnames.txt', header=None)
fid_outcomes = []
for fid in outcomes.iloc[:, 0]:
    f = fid[:fid.index('-')]
    fid_outcomes.append(int(f))

fo_dd = dd[dd.FieldID.isin(fid_outcomes)]
fo_df = df.loc[:, list(set(df.columns).intersection(set(outcomes.iloc[:, 0])))]

fid_outcomes_data = []
for fid in fo_df.columns:
    f = fid[:fid.index('-')]
    fid_outcomes_data.append(int(f))

cases = fo_df.shape[0] - fo_df.isna().sum()
outcome_cases_df = pd.DataFrame({'FieldID': fid_outcomes_data,
                                 'Proteomics_cases': cases})

cases = fo_df.shape[0] - fo_df.isna().sum()

outcome_cases_df = pd.DataFrame({'FieldID': fid_outcomes_data,
                                 'Proteomics_cases': cases})
outcome_cases_df = pd.merge(outcome_cases_df, dd)

# move Participants column next to Cases column to make it easy to compare the
# proteomics N for the disease with the total UKB N
columns = outcome_cases_df.columns.tolist()
column_to_move = columns.pop(columns.index('Participants'))
columns.insert(2, column_to_move)

# Reorder the columns in the DataFrame
outcome_cases_df = outcome_cases_df[columns]

outcome_cases_df.rename(columns={"Participants": "UKB_cases"}, inplace=True)
outcome_cases_df.to_csv('first_occurrence_metadata.csv')
