import pandas as pd
from .ukb_func import icd


icd10_df, icd10_date_df = icd.pull_icds(['41270'],
                                    ['41280'],
                                    ['S720', 'S721', 'S722'])
icd9_df, icd9_date_df = icd.pull_icds(['41271'], ['41281'], ['820'])

eid_set = list(set(icd10_df.eid).union(set(icd9_df.eid)))
with open("../tidy_data/hip_fracture_eid.txt", "w") as output:
    for item in eid_set:
        output.write(str(item) + "\n")
