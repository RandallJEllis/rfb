library(arrow)
library(mice)
library(JMbayes2)
library(tidyverse)

# NOTES
# GO TO THE END AFTER JOING MODEL FITS
# APOEGN consistently has terrible Rhats. Maybe remove the least common ones? 

# Five-fold CV based on BID, stratified by label, can use df_surv variable for this
# For each fold:
# mice imputation - either do this before CV (not great), or just run it separately on train and test with same settings, method, predictor matrix
# Fit to train
# Use jointFit object to predict on test
# Find best threshold, calculate metrics, store

# Open question: do we include COLLECTION_DATE_DAYS_CONSENT, time_to_event, and label when imputing? I think so, because we are only doing it for 
# set the working directory to where this script is located
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Read the parquet file
df <- read_parquet('../../tidy_data/A4/ptau_habits_phyneuro.parquet')
df <- df %>% select(c('BID', 'ORRES', 'COLLECTION_DATE_DAYS_CONSENT',
                      'time_to_event', 'label', 'AGEYR', 'SEX', 'EDCCNTU', 'APOEGN', 'RACE',
                       'PXCARD', # 'PXPULM',
                       'SMOKE', 'ALCOHOL', 'SUBUSE', 'AEROBIC', 'WALKING'
                      ))

# remove one patient who has a missing APOE genotype, which we do not want to impute
df[is.na(df$APOEGN),]
df <- df[df$BID != unique(df[is.na(df$APOEGN),]$BID),]

df$BID = as.integer(factor(df$BID))
df$APOEGN = factor(df$APOEGN)
df$PXCARD = factor(df$PXCARD)
# df$PXPULM = factor(df$PXPULM)
df$SUBUSE = factor(df$SUBUSE)
df$SEX = factor(df$SEX)
df$RACE = factor(df$RACE)
# df$label = factor(df$label)

# Check missingness pattern
md.pattern(df)

# MICE requires specifying methods for each variable
# You can use two-level methods for continuous variables measured at Level 1 (within-person)
# For example, use "2l.pan" for Y and cov_cont if you consider them hierarchical.
# For categorical variables, you might still rely on standard methods like "polyreg"

init <- mice(df, maxit=0) # initialize mice to see current default methods
meth <- init$method
pred <- init$predictorMatrix

# Set the cluster variable (BID) to indicate hierarchy
# We don't impute the ID itself, but we must set its role:
pred <- make.predictorMatrix(df)
pred[, "BID"] <- -2  # No variable should use 'id' to predict itself
meth <- make.method(df)
meth["BID"] <- ""     # Do not impute 'id'

# For two-level imputation, we need to specify the top-level ID in the method
# Assume Y and cov_cont are continuous variables that vary within individuals
# Use two-level methods (2l.pan) for these continuous variables
# cov_cat is categorical - we'll use "polyreg" (polytomous regression)
meth["ORRES"] <- "2l.pan"
meth["SMOKE"] <- "2l.pan"
meth["ALCOHOL"] <- "2l.pan"
meth["AEROBIC"] <- "2l.pan"
meth["WALKING"] <- "2l.pan"
meth["APOEGN"] <- "polyreg"
meth["PXCARD"] <- "polyreg"
# meth["PXPULM"] <- "polyreg"
meth["SUBUSE"] <- "polyreg"

# Now we must specify which variables are level-1 and which is level-2.
# The cluster variable 'id' is level-2. 
# According to the mice documentation for 2l.* methods, you must specify a variable called .cluster in the predictor matrix.
# We assign the cluster variable (id) to .cluster attribute, not to be predicted, but used for grouping.
# Alternatively, we can specify this using a block structure or via arguments in mice.

# Indicate the cluster variable
attr(pred, "2l.groups") <- list(level2 = "BID")

# Now run mice with these methods and predictor matrix
imp <- mice(df, 
            method = meth, 
            predictorMatrix = pred, 
            m = 5,      # number of imputations
            maxit = 10, # iterations
            seed = 123)

# Check the imputation output
summary(imp)

# After imputation, you can complete the data and fit models:
completed_data_list <- complete(imp, "all") 
# 'completed_data_list' is a list of imputed datasets.

# Now, you can fit your longitudinal or survival models on each imputed dataset and pool the results

df <- completed_data_list[[1]]

# Made Rhat values worse for APOE and ORRES in Survival model and sigma in Longitudinal model
# df$ORRES <- scale(df$ORRES)
# df$AGEYR <- scale(df$AGEYR)
# df$EDCCNTU <- scale(df$EDCCNTU)

df_surv <- df %>%
  group_by(BID) %>%
  slice_max(order_by = label, n = 1, with_ties = FALSE) %>%
  ungroup()

df_lm <- df %>% select(c('BID', 'COLLECTION_DATE_DAYS_CONSENT', 'ORRES', 'AGEYR', 'SEX', 'EDCCNTU', 'APOEGN', 'RACE', 'PXCARD', 'SMOKE', 'ALCOHOL',
                         'SUBUSE', 'AEROBIC', 'WALKING'))
# Fit a linear mixed-effects model for A over time
lmeFit <- lme(ORRES ~ AGEYR + SEX + EDCCNTU + APOEGN,
              random = ~ COLLECTION_DATE_DAYS_CONSENT | BID, data = df_lm, na.action=na.exclude)

# Create a Surv object for survival data
# survObj <- Surv(df_surv$time_to_event, df_surv$label)

# Fit a Cox model for the survival part
coxFit <- coxph(Surv(time_to_event, label) ~ AGEYR + SEX + EDCCNTU + APOEGN + RACE + PXCARD + SMOKE + ALCOHOL + SUBUSE + AEROBIC + WALKING, data = df_surv, x = TRUE)
# coxFit <- coxph(survObj ~ SEX + EDCCNTU + APOEGN + SMOKE + ALCOHOL + SUBUSE + AEROBIC + WALKING, data = df_surv, x = TRUE)

# # Fit the joint model
# jointFit <- jm(coxFit, lmeFit, time_var = "COLLECTION_DATE_DAYS_CONSENT", 
#                # You can specify which longitudinal variable links to survival:
#                # joint_model = "weibull-aft", 
#                n_chains = 3, n_iter = 12000, n_burnin = 1000)

jointFit <- jm(
  Surv_object = coxFit,
  Mixed_objects = list(lmeFit),
  time_var = "COLLECTION_DATE_DAYS_CONSENT",
  functional_forms = list("ORRES" = ~ value(ORRES)),
  data_Surv = df_surv,
  id_var = "BID"
)

summary(jointFit)


new_subject_long <- df

new_subject_surv <- df_surv

new_subject <- merge(new_subject_long, 
                     new_subject_surv, 
                     by = "BID", 
                     all = TRUE)

# Try predictions starting from time 0
predictions <- predict(jointFit, 
                       newdata = new_subject_long,   # longitudinal data
                       newdata2 = new_subject_surv,  # survival data
                       process = "event",
                       times = seq(0, 5, by = 0.5))

### THIS SEEMS TO WORK!!!
# First, let's identify which columns we need from each dataset
# Assuming BID is your ID variable
new_subject_long <- df_lm  # Your longitudinal data
new_subject_surv <- df_surv  # Your survival data

# Check column names
print("Longitudinal data columns:")
print(names(new_subject_long))
print("Survival data columns:")
print(names(new_subject_surv))

# When merging, we need to handle any duplicate columns
# Let's be explicit about which columns we want from each
new_subject <- merge(
  # From longitudinal data, keep ID and all longitudinal measurements
  new_subject_long,
  # From survival data, keep only ID, time, and status
  new_subject_surv[, c("BID", "time_to_event", "label")],
  by = "BID", 
  all = TRUE
)

# Check the structure of merged data
print("Merged data columns:")
print(names(new_subject))

# Create prediction version of data with time set to 0
new_subject_pred <- new_subject
new_subject_pred$time_to_event <- 0

# Use a coarser time grid - maybe every 30 days or so
time_points <- seq(0, max_time, by = 30)  # Or adjust based on your needs
print(paste("Number of time points:", length(time_points)))

predictions <- predict(jointFit, 
                       newdata = new_subject_pred,
                       process = "event",
                       times = time_points)

# Now try predictions
predictions <- predict(jointFit, 
                       newdata = new_subject,
                       process = "event",
                       times = seq(0, 5, by = 0.5))
