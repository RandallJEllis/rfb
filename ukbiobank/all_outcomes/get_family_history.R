library(data.table)

source('./code/id_loc.R')
source('./code/helper_functions.R')

data_folder <- '../../uk_biobank/project_52887_669338/'
file_path <- paste0(data_folder,"/ukb669338.csv")
ukb.fhx = fread(file_path, nrows=1)

# pull Field IDs without instance values
ukb_field_ids <- get_field_ids(colnames(ukb.fhx))
# Finding indices of ukb_field_ids values present in Field IDs
indices_members <- match(ukb_field_ids[,1], '20107', nomatch = 0)
# Filter the indices that have a match (non-zero values)
indices_matches <- which(indices_members > 0)

ukb.fhx = fread(file_path, select = c('eid', colnames(ukb.fhx)[indices_matches]))
