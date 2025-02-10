library(mice)
library(arrow)
library(JMbayes2)
library(tidyverse)
library(pec)
library(timeROC)
library(pROC)
library(yardstick)
library(dplyr)
library(ggplot2)

# Five-fold CV based on BID, stratified by label, can use df_surv variable for this
# For each fold:
# Fit to train
# Use jointFit object to predict on test
# Fit baseline cox to train, predict on test

# set the working directory to where this script is located
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# get BIDs to remove
df <- read_parquet('../../tidy_data/A4/ptau217_allvisits.parquet')

jm_lme <- list()
jm_cox <- list()
jm_l <- list()
baseline_cox <- list()
res_roc <- list()
res_auc <- list()
res_baseline_roc <- list()
times_used <- list()
for (fold in seq(0,4)){
  
  # Read the parquet file
  train_df <- read_parquet(paste0('../../tidy_data/A4/train_',fold,'.parquet'))
  val_df <- read_parquet(paste0('../../tidy_data/A4/val_',fold,'.parquet'))
  
  # df <- df %>% select(c('BID', 'ORRES', 'COLLECTION_DATE_DAYS_CONSENT',
  #                       'time_to_event', 'label', 'AGEYR', 'SEX', 'EDCCNTU', 'APOEGN', 'RACE',
  #                        'PXCARD', # 'PXPULM',
  #                        'SMOKE', 'ALCOHOL', 'SUBUSE', 'AEROBIC', 'WALKING'
  #                       ))
  
  print('Imported datasets')
  
  # Create a list containing both data frames
  df_list <- list(train_df = train_df, 
                  val_df = val_df
  )
  
  # Define a function to process each data frame
  process_df <- function(d) {
    
    # set E3/E3 (most common genotype) as reference level
    # Convert relevant columns to factors
    d$APOEGN <- factor(d$APOEGN)
    d <- within(d, APOEGN <- relevel(APOEGN, ref = "E3/E3"))
    
    # Uncomment and modify the following lines if needed
    # d$PXCARD <- factor(d$PXCARD)
    # d$PXPULM <- factor(d$PXPULM)
    # d$SUBUSE <- factor(d$SUBUSE)
    d$SEX <- factor(d$SEX)
    d$RACE <- factor(d$RACE)
    # d$label <- factor(d$label)
    
    return(d)
  }
  
  # Apply the processing function to each data frame in the list
  df_list <- lapply(df_list, process_df)
  
  # Assign the processed data frames back to their original variables
  train_df <- df_list$train_df
  val_df <- df_list$val_df
  
  train_df <- train_df %>%
    group_by(BID) %>%
    mutate(
      AGEYR = first(AGEYR),
      AGEYR_z = first(AGEYR_z),
      AGEYR_z_squared = first(AGEYR_z_squared),
      AGEYR_z_cubed = first(AGEYR_z_cubed),
    ) %>%
    ungroup()
  
  val_df <- val_df %>%
    group_by(BID) %>%
    mutate(
      AGEYR = first(AGEYR),
      AGEYR_z = first(AGEYR_z),
      AGEYR_z_squared = first(AGEYR_z_squared),
      AGEYR_z_cubed = first(AGEYR_z_cubed),
    ) %>%
    ungroup()
  

  coxFit <- coxph(
    Surv(time_to_event_yr, label) ~ ORRES_boxcox + AGEYR_z + AGEYR_z_squared + 
      AGEYR_z_cubed + SEX + EDCCNTU_z + APOEGN + AGEYR_z*APOEGN +
      AGEYR_z_squared*APOEGN + AGEYR_z_cubed*APOEGN, # + PXCARD + SMOKE + ALCOHOL + SUBUSE + AEROBIC + WALKING,
    data = train_df,
    id = train_df$BID,
    x = TRUE
  )
  
  cox_l[[paste0('Fold_',fold+1)]] <- coxFit
  
  baseline_coxFit <- coxph(
    Surv(time_to_event_yr, label) ~ AGEYR_z + AGEYR_z_squared + 
      AGEYR_z_cubed + SEX + EDCCNTU_z + APOEGN + AGEYR_z*APOEGN +
      AGEYR_z_squared*APOEGN + AGEYR_z_cubed*APOEGN, # + PXCARD + SMOKE + ALCOHOL + SUBUSE + AEROBIC + WALKING,
    data = train_df,
    id = train_df$BID,
    x = TRUE
  )
  
  baseline_cox_l[[paste0('Fold_',fold+1)]] <- baseline_coxFit
  
  ### Use the cox model alone as the baseline
  # predict risk on training data
  train_risk_scores <- predict(coxFit,
                         newdata = train_df, 
                         process = 'event')
  # Compute time-dependent ROC and AUROC
  print('Starting Cox model ROC analysis')
  train_roc <- timeROC(
    T = train_df$time_to_event_yr,
    delta = train_df$label,
    marker = train_risk_scores,
    cause = 1,  # Specify the event of interest
    times = seq(3, 8),
    iid = TRUE  # Optional: Compute influence functions for confidence intervals
  )
  
  train_roc_l[[paste0('Fold_',fold+1)]] <- train_roc
  
  baseline_train_risk_scores <- predict(baseline_coxFit,
                                        newdata = train_df, 
                                        process = 'event')
  baseline_train_roc <- timeROC(
    T = train_df$time_to_event_yr,
    delta = train_df$label,
    marker = baseline_train_risk_scores,
    cause = 1,  # Specify the event of interest
    times = seq(3, 8),
    iid = TRUE  # Optional: Compute influence functions for confidence intervals
  )
  
  baseline_train_roc_l[[paste0('Fold_',fold+1)]] <- baseline_train_roc
  
  # predict risk on val data
  val_risk_scores <- predict(coxFit,
                               newdata = val_df, 
                               process = 'event')
  # Compute time-dependent ROC and AUROC
  print('Starting Cox model ROC analysis')
  val_roc <- timeROC(
    T = val_df$time_to_event_yr,
    delta = val_df$label,
    marker = val_risk_scores,
    cause = 1,  # Specify the event of interest
    times = seq(3, 8),
    iid = TRUE  # Optional: Compute influence functions for confidence intervals
  )
  
  # Choose a threshold (e.g., median risk score)
  threshold <- median(val_risk_scores)
  
  time_points <- seq(3,8)
  
  # Initialize PPV and NPV storage
  ppv <- numeric(length(time_points))
  npv <- numeric(length(time_points))
  
  # Loop through time points
  for (i in seq_along(time_points)) {
    # Sensitivity and specificity at this time point
    sens <- val_roc$TP[i]  # True positive rate
    spec <- 1 - val_roc$FP[i]  # True negative rate
    
    # Prevalence (proportion of cases at this time point)
    prevalence <- mean(val_df$time_to_event_yr <= time_points[i] & val_df$label == 1)
    
    # Calculate PPV and NPV
    ppv[i] <- (sens * prevalence) / (sens * prevalence + (1 - spec) * (1 - prevalence))
    npv[i] <- (spec * (1 - prevalence)) / ((1 - sens) * prevalence + spec * (1 - prevalence))
  }
  
  # Combine results
  results <- data.frame(
    Time = time_points,
    PPV = ppv,
    NPV = npv
  )
  
  print(results)
  
  
  val_sespppvnpv <- SeSpPPVNPV(
    cutpoint = threshold,
    T = val_df$time_to_event_yr,
    delta = val_df$label,
    marker = val_risk_scores,
    cause = 1,  # Specify the event of interest
    times = seq(1, 8),
    iid = TRUE  # Optional: Compute influence functions for confidence intervals
  )
  
  val_roc_l[[paste0('Fold_',fold+1)]] <- val_roc
  
  baseline_val_risk_scores <- predict(baseline_coxFit,
                                        newdata = val_df, 
                                        process = 'event')
  baseline_val_roc <- timeROC(
    T = val_df$time_to_event_yr,
    delta = val_df$label,
    marker = baseline_val_risk_scores,
    cause = 1,  # Specify the event of interest
    times = seq(3, 8),
    iid = TRUE  # Optional: Compute influence functions for confidence intervals
  )
  
  baseline_val_roc_l[[paste0('Fold_',fold+1)]] <- baseline_val_roc
  
}


auc = c(as.numeric(val_roc_l$Fold_1$AUC),
        as.numeric(val_roc_l$Fold_2$AUC),
        as.numeric(val_roc_l$Fold_3$AUC),
        as.numeric(val_roc_l$Fold_4$AUC),
        as.numeric(val_roc_l$Fold_5$AUC)
)
baseline_auc = c(as.numeric(baseline_val_roc_l$Fold_1$AUC),
                 as.numeric(baseline_val_roc_l$Fold_2$AUC),
                 as.numeric(baseline_val_roc_l$Fold_3$AUC),
                 as.numeric(baseline_val_roc_l$Fold_4$AUC),
                 as.numeric(baseline_val_roc_l$Fold_5$AUC)
)
times = c(seq(3,8),
          seq(3,8),
          seq(3,8),
          seq(3,8),
          seq(3,8)
)
folds = rep(c(rep(1, 6),
              rep(2, 6),
              rep(3, 6),
              rep(4, 6),
              rep(5, 6)), 2)

results_df <- data.frame(
  model = c(rep("pTau217", length(auc)), rep("Baseline", length(baseline_auc))),
  auc = c(auc, baseline_auc),
  time = c(times, times),
  fold = folds
)
# write_csv(results_df, '../../tidy_data/A4/twoCox_results_AUC.csv')
results_df <- read.csv('../../tidy_data/A4/twoCox_results_AUC.csv')

# Save objects
saveRDS(jm_lme, "../../tidy_data/A4/fitted_jm_lme.rds")
saveRDS(jm_cox, "../../tidy_data/A4/fitted_jm_cox.rds")
saveRDS(jm_l, "../../tidy_data/A4/fitted_jm.rds")
# saveRDS(base_cox, "../../tidy_data/A4/fitted_baseline_cox.rds")
saveRDS(res_roc, "../../tidy_data/A4/results_joint_model_ROC.rds")
saveRDS(res_auc, "../../tidy_data/A4/results_joint_model_AUROC.rds")
saveRDS(res_baseline_roc, "../../tidy_data/A4/results_baseline_cox_model_AUROC.rds")
saveRDS(times_used, "../../tidy_data/A4/results_followuptimes_AUROC.rds")

# Load
res_roc <- readRDS("../../tidy_data/A4/results_joint_model_ROC.rds")
jm_lme <- readRDS("../../tidy_data/A4/fitted_jm_lme.rds")
jm__cox <- readRDS("../../tidy_data/A4/fitted_jm_cox.rds")
jm_l <- readRDS("../../tidy_data/A4/fitted_jm.rds")

# To load it back:
# your_list <- readRDS("filename.rds")


# Example summarization (if not already done)
summary_df <- results_df %>%
  group_by(model, time) %>%
  summarise(
    mean_AUC = mean(auc, na.rm = TRUE),
    sd_AUC = sd(auc, na.rm = TRUE),
    ymin = pmax(mean_AUC - sd_AUC, 0),
    ymax = pmin(mean_AUC + sd_AUC, 1),
    .groups = 'drop'
  )


# Create the plot
publication_plot_viridis <- ggplot(summary_df, aes(x = time, y = mean_AUC, color = model, fill = model)) +
  
  geom_ribbon(aes(ymin = ymin, ymax = ymax), 
              alpha = 0.3, 
              color = NA, 
              show.legend = FALSE) +
  
  geom_line(linewidth = 1.2) +
  
  geom_point(size = 3, shape = 21, fill = "white") +
  
  # Apply viridis color palette
  scale_color_viridis_d(option = "D", begin = 0.2, end = 0.8) +
  scale_fill_viridis_d(option = "D", begin = 0.2, end = 0.8) +
  
  labs(
    title = "Time-varying AUROC in the A4 Study",
    subtitle = "pTau217 vs. Baseline",
    x = "Follow-up Time (years)",
    y = "AUROC",
    color = "Model",
    fill = "Model"
  ) +
  
  theme_bw(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
    plot.subtitle = element_text(size = 14, hjust = 0.5),
    axis.title = element_text(face = "bold", size = 14),
    axis.text = element_text(size = 12),
    legend.title = element_text(face = "bold", size = 14),
    legend.text = element_text(size = 12),
    panel.grid.major = element_line(color = "grey80"),
    panel.grid.minor = element_blank(),
    legend.position = "right"
  ) + 
  scale_y_continuous(limits = c(0.6, 0.9))

# Display the plot
print(publication_plot_viridis)

# Save the plot
ggsave("../../tidy_data/A4/AUC_Over_Time_Publication_Viridis.pdf", 
       plot = publication_plot_viridis, 
       width = 8, 
       height = 6, 
       dpi = 300)


plot(val_roc_l$Fold_3, time = 6)
plot(baseline_val_roc_l$Fold_3, time = 6, add = T)

# Define common FPR points
fpr_common <- seq(0, 1, by = 0.001)  # 0%, 1%, ..., 100%
# 
# # Function to interpolate TPR at common FPR points
# interpolate_tpr <- function(roc_obj, fpr_points, model = c('joint_model', 'cox')) {
#   # Extract sensitivities (TPR) and specificities (1 - FPR)
#   tpr <- roc_obj$TP
#   fpr <- roc_obj$FP
#   
#   # Ensure ROC curve starts at (0,0) and ends at (1,1)
#   if (fpr[1] > 0) {
#     fpr <- c(0, fpr)
#     tpr <- c(0, tpr)
#   }
#   if (tail(fpr, n=1) < 1) {
#     fpr <- c(fpr, 1)
#     tpr <- c(tpr, 1)
#   }
#   
#   # Interpolate TPR at common FPR points
#   interp_tpr <- approx(x = fpr, y = tpr, xout = fpr_points, ties = mean)$y
#   
#   return(interp_tpr)
# }

# Can't get the abaove function to work with the Cox results, running it separately
interp_tpr <- function(model_list){
  interp_tpr_l <- list()
  for (fold in 1:5){
    fold_interp_trp <- list()
    for (col in 1:dim(model_list[[paste0('Fold_',fold)]]$TP)[2]){
      # Extract sensitivities (TPR) and specificities (1 - FPR)
      tpr <- model_list[[paste0('Fold_',fold)]]$TP[,col]
      fpr <- model_list[[paste0('Fold_',fold)]]$FP[,col]
      
      # Ensure ROC curve starts at (0,0) and ends at (1,1)
      if (fpr[1] > 0) {
        fpr <- c(0, fpr)
        tpr <- c(0, tpr)
      }
      if (tail(fpr, n=1) < 1) {
        fpr <- c(fpr, 1)
        tpr <- c(tpr, 1)
      }
      
      # Interpolate TPR at common FPR points
      interp_tpr <- approx(x = fpr, y = tpr, xout = fpr_common, ties = mean)$y
      fold_interp_trp[[col]] <- interp_tpr
    }
    interp_tpr_l[[paste0('fold_',fold)]] <- fold_interp_trp
  }
  return(interp_tpr_l)
}

bl_interp_tpr_l <- interp_tpr(baseline_val_roc_l)
ptau_interp_tpr_l <- interp_tpr(val_roc_l)

# Convert to data frame for easier manipulation
### BASELINE
year3_bl_tpr_df <- as.data.frame(cbind(
  unlist(bl_interp_tpr_l[['fold_1']][1]), 
  unlist(bl_interp_tpr_l[['fold_2']][1]), 
  unlist(bl_interp_tpr_l[['fold_3']][1]), 
  unlist(bl_interp_tpr_l[['fold_4']][1]), 
  unlist(bl_interp_tpr_l[['fold_5']][1]) 
)
)
colnames(year3_bl_tpr_df) <- c('Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5')

year4_bl_tpr_df <- as.data.frame(cbind(
  unlist(bl_interp_tpr_l[['fold_1']][2]), 
  unlist(bl_interp_tpr_l[['fold_2']][2]), 
  unlist(bl_interp_tpr_l[['fold_3']][2]), 
  unlist(bl_interp_tpr_l[['fold_4']][2]), 
  unlist(bl_interp_tpr_l[['fold_5']][2]) 
)
)
colnames(year4_bl_tpr_df) <- c('Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5')

year5_bl_tpr_df <- as.data.frame(cbind(
  unlist(bl_interp_tpr_l[['fold_1']][3]), 
  unlist(bl_interp_tpr_l[['fold_2']][3]), 
  unlist(bl_interp_tpr_l[['fold_3']][3]), 
  unlist(bl_interp_tpr_l[['fold_4']][3]), 
  unlist(bl_interp_tpr_l[['fold_5']][3]) 
)
)
colnames(year5_bl_tpr_df) <- c('Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5')

year6_bl_tpr_df <- as.data.frame(cbind(
  unlist(bl_interp_tpr_l[['fold_1']][4]), 
  unlist(bl_interp_tpr_l[['fold_2']][4]), 
  unlist(bl_interp_tpr_l[['fold_3']][4]), 
  unlist(bl_interp_tpr_l[['fold_4']][4]), 
  unlist(bl_interp_tpr_l[['fold_5']][4]) 
)
)
colnames(year6_bl_tpr_df) <- c('Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5')

year7_bl_tpr_df <- as.data.frame(cbind(
  unlist(bl_interp_tpr_l[['fold_1']][5]), 
  unlist(bl_interp_tpr_l[['fold_2']][5]), 
  unlist(bl_interp_tpr_l[['fold_3']][5]), 
  unlist(bl_interp_tpr_l[['fold_4']][5]), 
  unlist(bl_interp_tpr_l[['fold_5']][5]) 
)
)
colnames(year7_bl_tpr_df) <- c('Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5')

year8_bl_tpr_df <- as.data.frame(cbind(
  unlist(bl_interp_tpr_l[['fold_1']][6]), 
  unlist(bl_interp_tpr_l[['fold_2']][6]), 
  unlist(bl_interp_tpr_l[['fold_3']][6]), 
  unlist(bl_interp_tpr_l[['fold_4']][6]), 
  unlist(bl_interp_tpr_l[['fold_5']][6]) 
)
)
colnames(year8_bl_tpr_df) <- c('Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5')

year3_bl_tpr_df$fpr <- fpr_common
year4_bl_tpr_df$fpr <- fpr_common
year5_bl_tpr_df$fpr <- fpr_common
year6_bl_tpr_df$fpr <- fpr_common
year7_bl_tpr_df$fpr <- fpr_common
year8_bl_tpr_df$fpr <- fpr_common

# Reshape data for summarization
year3_bl_tpr_long <- year3_bl_tpr_df %>%
  pivot_longer(cols = starts_with("Fold_"), names_to = "Fold", values_to = "TPR")
year4_bl_tpr_long <- year4_bl_tpr_df %>%
  pivot_longer(cols = starts_with("Fold_"), names_to = "Fold", values_to = "TPR")
year5_bl_tpr_long <- year5_bl_tpr_df %>%
  pivot_longer(cols = starts_with("Fold_"), names_to = "Fold", values_to = "TPR")
year6_bl_tpr_long <- year6_bl_tpr_df %>%
  pivot_longer(cols = starts_with("Fold_"), names_to = "Fold", values_to = "TPR")
year7_bl_tpr_long <- year7_bl_tpr_df %>%
  pivot_longer(cols = starts_with("Fold_"), names_to = "Fold", values_to = "TPR")
year8_bl_tpr_long <- year8_bl_tpr_df %>%
  pivot_longer(cols = starts_with("Fold_"), names_to = "Fold", values_to = "TPR")



#### pTau217
year3_ptau_tpr_df <- as.data.frame(cbind(
  unlist(ptau_interp_tpr_l[['fold_1']][1]), 
  unlist(ptau_interp_tpr_l[['fold_2']][1]), 
  unlist(ptau_interp_tpr_l[['fold_3']][1]), 
  unlist(ptau_interp_tpr_l[['fold_4']][1]), 
  unlist(ptau_interp_tpr_l[['fold_5']][1]) 
)
)
colnames(year3_ptau_tpr_df) <- c('Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5')

year4_ptau_tpr_df <- as.data.frame(cbind(
  unlist(ptau_interp_tpr_l[['fold_1']][2]), 
  unlist(ptau_interp_tpr_l[['fold_2']][2]), 
  unlist(ptau_interp_tpr_l[['fold_3']][2]), 
  unlist(ptau_interp_tpr_l[['fold_4']][2]), 
  unlist(ptau_interp_tpr_l[['fold_5']][2]) 
)
)
colnames(year4_ptau_tpr_df) <- c('Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5')

year5_ptau_tpr_df <- as.data.frame(cbind(
  unlist(ptau_interp_tpr_l[['fold_1']][3]), 
  unlist(ptau_interp_tpr_l[['fold_2']][3]), 
  unlist(ptau_interp_tpr_l[['fold_3']][3]), 
  unlist(ptau_interp_tpr_l[['fold_4']][3]), 
  unlist(ptau_interp_tpr_l[['fold_5']][3]) 
)
)
colnames(year5_ptau_tpr_df) <- c('Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5')

year6_ptau_tpr_df <- as.data.frame(cbind(
  unlist(ptau_interp_tpr_l[['fold_1']][4]), 
  unlist(ptau_interp_tpr_l[['fold_2']][4]), 
  unlist(ptau_interp_tpr_l[['fold_3']][4]), 
  unlist(ptau_interp_tpr_l[['fold_4']][4]), 
  unlist(ptau_interp_tpr_l[['fold_5']][4]) 
)
)
colnames(year6_ptau_tpr_df) <- c('Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5')

year7_ptau_tpr_df <- as.data.frame(cbind(
  unlist(ptau_interp_tpr_l[['fold_1']][5]), 
  unlist(ptau_interp_tpr_l[['fold_2']][5]), 
  unlist(ptau_interp_tpr_l[['fold_3']][5]), 
  unlist(ptau_interp_tpr_l[['fold_4']][5]), 
  unlist(ptau_interp_tpr_l[['fold_5']][5]) 
)
)
colnames(year7_ptau_tpr_df) <- c('Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5')

year8_ptau_tpr_df <- as.data.frame(cbind(
  unlist(ptau_interp_tpr_l[['fold_1']][6]), 
  unlist(ptau_interp_tpr_l[['fold_2']][6]), 
  unlist(ptau_interp_tpr_l[['fold_3']][6]), 
  unlist(ptau_interp_tpr_l[['fold_4']][6]), 
  unlist(ptau_interp_tpr_l[['fold_5']][6]) 
)
)
colnames(year8_ptau_tpr_df) <- c('Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5')

year3_ptau_tpr_df$fpr <- fpr_common
year4_ptau_tpr_df$fpr <- fpr_common
year5_ptau_tpr_df$fpr <- fpr_common
year6_ptau_tpr_df$fpr <- fpr_common
year7_ptau_tpr_df$fpr <- fpr_common
year8_ptau_tpr_df$fpr <- fpr_common

# Reshape data for summarization
year3_ptau_tpr_long <- year3_ptau_tpr_df %>%
  pivot_longer(cols = starts_with("Fold_"), names_to = "Fold", values_to = "TPR")
year4_ptau_tpr_long <- year4_ptau_tpr_df %>%
  pivot_longer(cols = starts_with("Fold_"), names_to = "Fold", values_to = "TPR")
year5_ptau_tpr_long <- year5_ptau_tpr_df %>%
  pivot_longer(cols = starts_with("Fold_"), names_to = "Fold", values_to = "TPR")
year6_ptau_tpr_long <- year6_ptau_tpr_df %>%
  pivot_longer(cols = starts_with("Fold_"), names_to = "Fold", values_to = "TPR")
year7_ptau_tpr_long <- year7_ptau_tpr_df %>%
  pivot_longer(cols = starts_with("Fold_"), names_to = "Fold", values_to = "TPR")
year8_ptau_tpr_long <- year8_ptau_tpr_df %>%
  pivot_longer(cols = starts_with("Fold_"), names_to = "Fold", values_to = "TPR")


summarize_tpr <- function(tpr_long, folds){
  K <- 5 # number of folds
  # Calculate summary statistics
  summary_tpr <- tpr_long %>%
    group_by(fpr) %>%
    summarize(
      Mean_TPR = mean(TPR, na.rm = TRUE),
      SD_TPR = sd(TPR, na.rm = TRUE),
      SE_TPR = SD_TPR / sqrt(n()),
      CI_Lower = Mean_TPR - qt(0.975, df = K - 1) * SE_TPR,
      CI_Upper = Mean_TPR + qt(0.975, df = K - 1) * SE_TPR
    )
  
  # Ensure confidence intervals are within [0,1]
  summary_tpr <- summary_tpr %>%
    mutate(
      CI_Lower = pmax(CI_Lower, 0),
      CI_Upper = pmin(CI_Upper, 1)
    )
  return(summary_tpr)
}

year <- 6
bl_summ <- summarize_tpr(get(paste0('year',year,'_bl_tpr_long')), 5)
bl_summ <- bl_summ %>%
  mutate(Model = "Baseline")

ptau_summ <- summarize_tpr(get(paste0('year',year,'_ptau_tpr_long')), 5)
ptau_summ <- ptau_summ %>%
  mutate(Model = "pTau217")

# Combine the two data frames into one
combined <- bind_rows(bl_summ, ptau_summ)

# Plot the combined ROC curves
ggplot(combined, aes(x = fpr, y = Mean_TPR, color = Model, fill = Model)) +
  geom_line(linewidth = 1) +  # Plot ROC lines
  geom_ribbon(aes(ymin = CI_Lower, ymax = CI_Upper), alpha = 0.2) +  # Add confidence intervals
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +  # Diagonal line
  labs(
    title = paste0("Year ",year+1, " - Comparison of Mean ROC Curves"),
    x = "False Positive Rate (FPR)",
    y = "True Positive Rate (TPR)",
    color = "Model",
    fill = "Model"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),  # Center the plot title
    legend.position = "bottom"  # Position the legend at the bottom
  )



library(survival)
lung <- survival::lung
set.seed(1953) # a good year
nvisit <- floor(pmin(lung$time/30.5, 12))
response <- rbinom(nrow(lung), nvisit, .05) > 0
badfit <- survfit(Surv(time/365.25, status) ~ response, data=lung)
plot(badfit, mark.time=FALSE, lty=1:2, xlab="Years post diagnosis", ylab="Survival")
legend(1.5, .85, c("Responders", "Non-responders"),
         lty=2:1, bty='n')
summary(badfit)
