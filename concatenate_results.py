import pandas as pd
import os

path = '../tidy_data/bootstrap'
files = os.listdir(path)

df_l = []

for i, f in enumerate(files):
    if i % 50 == 0:
        print(i)
    df = pd.read_parquet(f'{path}/{f}')
    df_l.append(df)

df_l = pd.concat(df_l)
df_l.to_parquet('../tidy_data/full_bs_results.parquet')
