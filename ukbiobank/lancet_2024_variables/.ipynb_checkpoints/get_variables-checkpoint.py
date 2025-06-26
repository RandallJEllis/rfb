import sys
sys.path.append('../ukb_func')
import ukb_utils
import icd
import utils
import df_utils
import utils
import pandas as pd

'''
Age cataract diagnosed: 4700
Age when diabetes-related eye disease diagnosed - 5901

LDL - 30780

Head Injury - ICD codes S00-S09

Physical inactivity - MET minutes per week for moderate activity - 22038

Smoking - Pack years of smoking - 20161

Excessive alcohol consumption - 1558

Hypertension
Source of report of primary/secondary hypertension: 131287/131295
Date ^ reported: 131286/131294
Source for gestational with/without proteinuria: 132189/132187
Date ^ reported: 132188/132186
Source for pre-existing hypertension complicating pregnancy, childbirth and the puerperium: 132181
Date ^ reported: 132180

Obesity - Date E66 first reported: 130792; Source: 130793

Diabetes (Date first reported; Source)
Insulin-dependent: 130706; 130707
Non-insulin-dependent: 130708; 130709
Malnutrition-related: 130710; 130711
Other specified: 130712; 130713
Unspecified: 130714; 130715
During pregnancy: 132202; 132203

Hearing loss
Currently suffering: 28627
Length of time suffering from hearing loss: 28628
Extent affected by hearing loss: 28629
Date first reported; Source:
Conductive and sensorineural: 131258; 131259
Other hearing loss: 131260; 131261

Depression (Date first reported; Source)
Depressive episode: 130894; 130895
Recurrent: 130896; 130897

Infrequent social contact:
Frequency of friends/family visit: 1031
Loneliness, isolation: 2020

Air pollution
Inverse distance to nearest major road: 24012
Nitrogen dioxide air pollution, 2007: 24018
Particulate matter air pollution (pm10), 2007: 24019
Particulate matter air pollution (pm2.5), 2010: 24006
Sum of read length of major roads within 100m: 24015
Traffic intensity on nearest major road: 24011

NOTE: we don't have 28627, 28628, 28629
'''

ids_of_interest = [4700, 5901, 30780, 22038, 20161, 1558, 131287, 131295, 131286, 131294, 132189, 132187, 132188, 132186, 132181, 132180, 130792, 130793,
             130706, 130707, 130708, 130709, 130710, 130711, 130712, 130713, 130714, 130715, 132202, 132203, 28627, 28628, 28629, 131258, 131259, 131260, 131261,
             130894, 130895, 130896, 130897, 1031, 2020, 24012, 24018, 24019, 24006, 24015, 24011]
ids_of_interest = [str(x) for x in ids_of_interest]

df1 = pd.read_csv('../../../../uk_biobank/project_52887_676883/ukb676883.csv', nrows=1)
filtered_columns = ['eid'] + [col for col in df1.columns if col.split('-')[0] in ids_of_interest]
df1 = pd.read_csv('../../../../uk_biobank/project_52887_676883/ukb676883.csv', usecols=filtered_columns)

df2 = pd.read_csv('../../../../uk_biobank/project_52887_669338/ukb669338.csv', nrows=1)
filtered_columns = ['eid'] + [col for col in df2.columns if col.split('-')[0] in ids_of_interest]
df2 = pd.read_csv('../../../../uk_biobank/project_52887_669338/ukb669338.csv', usecols=filtered_columns)
ins0_col = ['eid'] + [x for x in df2.columns if '-0' in x]
df2 = df2.loc[:, ins0_col]

df1 = ukb_utils.remove_participants_full_missing(df1)
df2 = ukb_utils.remove_participants_full_missing(df2)
df3 = df1.merge(df2, how='outer', on='eid')

# Head injury ICD codes don't have pre-made source/date columns
icd_df, icd_date_df = icd.pull_icds(['41270'], ['41280'], ['S00', 'S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09'])
icd = icd_df.merge(icd_date_df, on='eid', how='outer')

lancet_vars = df3
head_injury_icd = icd

# Age cataract diagnosed and Age when diabetes-related eye disease diagnosed are NA or Age. Transform so that NA=0 and non-NA=1
lancet_vars['4700-0.0'] = lancet_vars['4700-0.0'].notna().astype(int)
lancet_vars['5901-0.0'] = lancet_vars['5901-0.0'].notna().astype(int)

# head injury column where the value is 0 if the eid is not in the eid column of head_injury_icd, and 1 if it is
lancet_vars['head_injury'] = lancet_vars.eid.isin(head_injury_icd.eid).astype(int)
 
# ordinal encoding for alcohol intake frequency 
# Define the correct order for the ordinal variable
correct_order = [-3, 6, 5, 4, 3, 2, 1]
lancet_vars['alcohol_consumption'] = pd.Categorical(lancet_vars['1558-0.0'], categories=correct_order, ordered=True)
# Convert the ordered categorical data to integer codes
lancet_vars['alcohol_consumption'] = lancet_vars['alcohol_consumption'].cat.codes

# hypertension
lancet_vars = ukb_utils.binary_encode_column_membership_datacoding2171(lancet_vars, ['131287-0.0', '131295-0.0', '132189-0.0', '132187-0.0', '132181-0.0'], 'hypertension')

# obesity 
lancet_vars = ukb_utils.binary_encode_column_membership_datacoding2171(lancet_vars, ['130793-0.0'], 'obesity')

# diabetes
lancet_vars = ukb_utils.binary_encode_column_membership_datacoding2171(lancet_vars, ['130707-0.0', '130709-0.0', '130711-0.0', '130713-0.0', '130715-0.0', '132203-0.0'], 'diabetes')

# hearing loss
lancet_vars = ukb_utils.binary_encode_column_membership_datacoding2171(lancet_vars, ['131259-0.0', '131261-0.0'], 'hearing_loss')

# depression
lancet_vars = ukb_utils.binary_encode_column_membership_datacoding2171(lancet_vars, ['130895-0.0', '130897-0.0'], 'depression')

# ordinal encoding for frequency of friends/family visit
correct_order = [-3, -1, 7, 6, 5, 4, 3, 2, 1]
lancet_vars['freq_friends_family_visit'] = pd.Categorical(lancet_vars['1031-0.0'], categories=correct_order, ordered=True)
# Convert the ordered categorical data to integer codes
lancet_vars['freq_friends_family_visit'] = lancet_vars['freq_friends_family_visit'].cat.codes

utils.check_folder_existence('../../tidy_data/UKBiobank/dementia/lancet2024')
lancet_vars.to_parquet('../../tidy_data/UKBiobank/dementia/lancet2024/lancet2024_preprocessed.parquet')