import pandas as pd
import os
from datetime import datetime

path = '../tidy_data/bootstrap'
files = os.listdir(f'{path}/individual_results')

protein_df_l = []
df_l = []

for i, f in enumerate(files):
    if i % 50 == 0:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f'File {i+1} of {len(files)}, Current Time = {current_time}')
    df = pd.read_parquet(f'{path}/{f}')
    # df['outcome'] = df['outcome'].str.split('-', n=1).str[0].astype(int)

    # the proteins and outcome columns take up an incredible amount of memory
    # since the proteins repeat across bootstraps, we will create a separate dataframe
    # with nfeats, outcomes, proteins, and iterations which will have a small number of unique rows
    protein_df = df.groupby(['n_features', 'outcome', 'iteration']).first().reset_index()
    protein_df['outcome'] = protein_df['outcome'].str.split('-', n=1).str[0].astype(int)
    protein_df_l.append(protein_df)

    # df = df.drop(columns=['proteins'])
    # df_l.append(df)

protein_df_l = pd.concat(protein_df_l)
protein_df_l.to_parquet(f'{path}/protein_full_bs_results.parquet')

# df_l = pd.concat(df_l)
# df_l.to_parquet(f'{path}/full_bs_results.parquet')
