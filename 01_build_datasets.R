library(arrow)
library(data.table)
library(tidyverse)

source('./code/id_loc.R')
source('./code/helper_functions.R')

### PROTEINS
prot <- read_parquet('./tidy_data/proteomics/proteomics.parquet')

# extract Instance 0 because Instance 2 N is ~2000 (no Instance 1 for proteomics)
ins0 <- grep('-0', names(prot))
prot <- prot |> select(all_of(c('eid', colnames(prot)[ins0])))

# 44 patients removed (no proteomics measured at Instance 0)
prot <- remove_participants_full_missing(prot, colnames(prot)[2:ncol(prot)])

### FIRST OCCURRENCE VARIABLES
data_dictionary <- read.csv('./Data_Dictionary_Showcase.csv')
fo <- data_dictionary |> 
                      slice( grep('First occurrences', Path) ) |>
                      slice( grep('Date', Field) )

# find dataset with first occurrence variables
id_loc(fo$FieldID[1]) # project_52887_676883
data_folder <- '../../uk_biobank/project_52887_676883/'
file_path <- paste0(data_folder,"/ukb676883.csv")

# import one row and subset the columns with Field IDs for brain imaging phenotypes
ukb.fo = fread(file_path, nrows=1)

# pull Field IDs without instance values
ukb_field_ids <- get_field_ids(colnames(ukb.fo))
# Finding indices of ukb_field_ids values present in Field IDs
indices_members <- match(ukb_field_ids[,1], fo$FieldID, nomatch = 0)
# Filter the indices that have a match (non-zero values)
indices_matches <- which(indices_members > 0)

# length(indices_matches) = 1130; all First occurrences variables are found
ukb.fo = fread(file_path, select = c('eid', colnames(ukb.fo)[indices_matches]))

# full overlap
print(length(intersect(prot$eid, ukb.fo$eid)) / nrow(prot)) 

# remove columns where nobody in the proteomics dataset has an entry
ukb.fo <- ukb.fo[ukb.fo$eid %in% prot$eid, ]
na_col <- colSums(is.na(ukb.fo))

length(which(na_col == nrow(ukb.fo))) # 75 columns with full NA (i.e., nobody has this ICD code)
length(which(na_col > (nrow(ukb.fo) - 50))) # 523 columns less than 50 non-NA entries

col_remove <- which(na_col > (nrow(ukb.fo) - 50))

ukb.fo <- ukb.fo %>% select(!all_of(col_remove))


### AGE, SEX, SITE, DATE
assd <- read_parquet('./tidy_data/age_sex_site_date/age_sex_site_date_df.parquet')
# proteomics are from Instance 0 so pull relevant variables
colnames(assd)
assd <- assd |> select(all_of(c('eid', '31-0.0', '53-0.0', '54-0.0', '21003-0.0', '21003-0.0_squared')))

### MERGE DFS TOGETHER
df_final <- merge(prot, ukb.fo, by='eid', all.x=T)
df_final <- merge(df_final, assd, by='eid', all.x=T)

# Replace empty strings with NA
df_final[df_final == ""] <- NA
na_col <- colSums(is.na(df_final))
col_all_na <- which(na_col == nrow(df_final)) # should be 0

write_parquet(df_final, '../rfb/tidy_data/proteomics_first_occurrences.parquet')

# save protein columns to sample randomly from
prot_names <- colnames(prot)[2:ncol(prot)]
fwrite(list(prot_names), file = '../rfb/tidy_data/protein_colnames.txt')

# save outcome columns to iterate over all outcomes
outcome_names <- colnames(ukb.fo)[2:ncol(ukb.fo)]
fwrite(list(outcome_names), file = '../rfb/tidy_data/outcome_colnames.txt')

