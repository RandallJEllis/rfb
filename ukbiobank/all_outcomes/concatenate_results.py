import pandas as pd
import os
from datetime import datetime

path = '../tidy_data/bootstrap'
files = os.listdir(path)

df_l = []

for i, f in enumerate(files):
    if i % 50 == 0:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(f'File {i+1} of {len(files)}, Current Time = {current_time}')
    df = pd.read_parquet(f'{path}/{f}')
    df = df.drop(columns=['proteins'])
    df['outcome'] = df['outcome'].str.split('-', n=1).str[0].astype(int)
    df_l.append(df)

df_l = pd.concat(df_l)
df_l.to_parquet('../tidy_data/full_bs_results.parquet')
