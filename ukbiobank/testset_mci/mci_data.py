import sys
sys.path.append('../ukb_func')
import icd
import os
import pandas as pd
import numpy as np
from datetime import datetime

mci_df = icd.pull_earliest_dates(['41270'], ['41280'], ['F067'])
mci_df.to_parquet('../../tidy_data/dementia/mci/mci_data.parquet')