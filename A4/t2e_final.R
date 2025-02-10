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
  
  # lmeFit <- lme(
  #   ORRES_boxcox ~ AGEYR_z + AGEYR_z_squared + AGEYR_z_cubed + SEX + 
  #     EDCCNTU_z + APOEGN + AGEYR_z*APOEGN + AGEYR_z_squared*APOEGN + 
  #     AGEYR_z_cubed*APOEGN,
  #   random = ~ COLLECTION_DATE_DAYS_CONSENT_yr | BID,
  #   data = train_df,
  #   na.action=na.exclude,
  #   control = lmeControl(opt = 'optim')
  # )
  # 
  # null_lmeFit <- lme(
  #   ORRES_boxcox ~ COLLECTION_DATE_DAYS_CONSENT_yr,
  #   random = ~ COLLECTION_DATE_DAYS_CONSENT_yr | BID,
  #   data = train_df,
  #   na.action=na.exclude,
  #   control = lmeControl(opt = 'optim')
  # )

  # jm_lme[[paste0("fold_",fold+1)]] <- lmeFit
  # print('LME fitted')
  
  # train_df_case_label_allT <- train_df %>%
  #   group_by(BID) %>%
  #   mutate(label = ifelse(any(label == 1),
  #                         1,
  #                         label))
  # train_df_unique <- train_df_case_label_allT %>% distinct(BID, .keep_all = TRUE)
  
  coxFit <- coxph(
    Surv(time_to_event_yr, label) ~ AGEYR_z + AGEYR_z_squared + 
      AGEYR_z_cubed + SEX + EDCCNTU_z + APOEGN + AGEYR_z*APOEGN +
      AGEYR_z_squared*APOEGN + AGEYR_z_cubed*APOEGN, # + PXCARD + SMOKE + ALCOHOL + SUBUSE + AEROBIC + WALKING,
    data = train_df,
    id = train_df$BID,
    x = TRUE
  )

  # jm_cox[[paste0("fold_",fold+1)]] <- coxFit
  # print('Cox fitted')

  # jointFit <- jm(
  #   Surv_object = coxFit,
  #   Mixed_objects = list(lmeFit),
  #   time_var = "COLLECTION_DATE_DAYS_CONSENT_yr",
  #   functional_forms = list("ORRES_boxcox" = ~ value(ORRES_boxcox) + slope(ORRES_boxcox)),
  #   data_Surv = train_df,
  #   id_var = "BID"
  # )

  # jm_l[[paste0("fold_",fold+1)]] <- jointFit
  # print('Joint model fitted')

  #### https://drizopoulos.github.io/JMbayes2/articles/Dynamic_Predictions.html
  # Calculate roc curve. Uses thresholds 0 to 1 in steps of 0.01. Gives TP, FP, best threshold by Youden and F1
  print('Starting joint model ROC analysis')
  dt = 1
  first_year <- as.integer(min(train_df$COLLECTION_DATE_DAYS_CONSENT_yr))
  last_year <- as.integer(max(train_df$COLLECTION_DATE_DAYS_CONSENT_yr)) +2
  print(first_year)
  print(last_year)

  res_roc_iter <- list()
  res_auc_iter <- c()
  followup_time <- c()
  for (tstart in seq(first_year,last_year)){
    
    thoriz <- tstart + dt
    
    # This is a strange quirk of tvROC (function to calculate ROC curves). The 
    # function subsets newdata before the start of the interval (Tstart) by
    # the longitudinal time variable (COLLECTION_DATE_DAYS_CONSENT_yr), but
    # then subsets the subset for rows where the time to event 
    # (time_to_event_yr) is less than the time horizon (Tstart + Dt) but where
    # the label is 1. With the way the data is set up for the Cox model, where
    # time points before the time to event are labeled 0, and time points at or
    # after the time to event are labeled 1, tvROC will never find events. 
    # The solution is, for all patients who ever experience an event, fix all
    # values in the label column at 1. 
    val_df_case_label_allT <- val_df %>%
      group_by(BID) %>%
      mutate(label = ifelse(any(label == 1 & time_to_event_yr <= thoriz),
                                     1,
                                     label))
    
    print(
      paste0(
        'Interval: Year ',tstart, ' to ',thoriz
      )
    )
    
    ids_pre_tstart <- unique(
      val_df_case_label_allT[
        val_df_case_label_allT$COLLECTION_DATE_DAYS_CONSENT_yr <= tstart,
        ]$BID
      )
    interval_events <- val_df_case_label_allT[val_df_case_label_allT$BID %in% ids_pre_tstart 
                                              & val_df_case_label_allT$time_to_event_yr >= tstart
                                              & val_df_case_label_allT$time_to_event_yr <= thoriz
                                              & val_df_case_label_allT$label == 1
                                              ,]
    # val_df_unique <- val_df_case_label_allT %>% distinct(BID, .keep_all = TRUE)
    if (nrow(interval_events) > 0){
      roc <- tvROC(jointFit, newdata = val_df_case_label_allT, 
                   Tstart = tstart, 
                   Dt = dt,
                   type_weights = c("IPCW")
      )
      res_roc_iter[[paste0("year_",tstart)]] <- roc
      # plot(roc)
      auc <- tvAUC(roc)
      res_auc_iter <- c(res_auc_iter, auc$auc)
      followup_time <- c(followup_time, tstart+dt)

    }
    else {
      print("No events in interval. Continuing to next interval.")
      next
    }
    # calibration_plot(jointFit, newdata = df_surv, Tstart = tstart, Dt = dt)
    # calibration_metrics(jointFit, df_surv, Tstart = tstart, Dt = dt)
    # tvBrier(jointFit, newdata = df_surv, Tstart = tstart, Dt = dt)
    # tvBrier(jointFit, newdata = df_surv, Tstart = tstart, Dt = dt, integrated = TRUE)
    # tvBrier(jointFit, newdata = df_surv, Tstart = tstart, Dt = dt, integrated = TRUE,
    #         type_weights = "IPCW")
    # tvEPCE(jointFit, newdata = df_surv, Tstart = tstart, Dt = 1)
  }
  res_roc[[paste0("fold_",fold+1)]] <- res_roc_iter
  res_auc[[paste0("fold_",fold+1)]] <- res_auc_iter
  times_used[[paste0("fold_",fold+1)]] <- followup_time
    
  ### Use the cox model alone as the baseline
  # predict risk on validation data
  risk_scores <- predict(coxFit,
                         newdata = train_df, 
                         process = 'event')

  # Compute time-dependent ROC and AUROC
  print('Starting Cox model ROC analysis')
  baseline_roc <- timeROC(
    T = train_df$time_to_event_yr,
    delta = train_df$label,
    marker = risk_scores,
    cause = 1,  # Specify the event of interest
    times = seq(1, 10),
    iid = TRUE  # Optional: Compute influence functions for confidence intervals
  )
  
  res_baseline_roc[[paste0("fold_",fold+1)]] <- baseline_roc
  
}

####
library(survAUC)

# Function to calculate time-dependent AUC for multiple time windows
calculate_cox_tvAUC <- function(coxFit, newdata, start_times, horizon) {
  # Prepare results storage
  aucs <- numeric(length(start_times))
  
  # Iterate through start times
  for (i in seq_along(start_times)) {
    tstart <- start_times[i]
    tend <- tstart + horizon
    
    # Subset data to observations at risk at tstart
    subset_data <- newdata[newdata$time_to_event_yr > tstart, ]
    
    # Predict risks at tstart
    pred_risks <- predict(coxFit, newdata = subset_data, type = "risk")
    
    # Calculate time-dependent AUC
    aucs[i] <- survAUC::AUC.sh(
      Surv.time = subset_data$time_to_event_yr, 
      Surv.event = subset_data$label, 
      risk = pred_risks, 
      times = tend
    )
  }
  
  return(aucs)
}

# Example usage
cox_aucs <- calculate_cox_tvAUC(
  coxFit, 
  newdata = val_df_case_label_allT, 
  start_times = tstart, 
  horizon = dt
)

lp <- predict(coxFit)
lpnew <- predict(coxFit, newdata = val_df)
Surv.rsp <- Surv(train_df$time_to_event_yr, train_df$label)
Surv.rsp.new <- Surv(val_df$time_to_event_yr, val_df$label)
times <- seq(2,7,1)
AUC_sh <- survAUC::AUC.sh(Surv.rsp, Surv.rsp.new, lp, lpnew, times)

#####
# Fit joint model

# Predict survival probabilities at Thoriz
preds <- GLMMadaptive::predict(jointFit,
                 newdata = val_df,
                 process = 'event',
                 times = thoriz
                   )
pred_probs <- preds$pred[ which(preds$times <= thoriz)
                          & which(preds$times >= tstart)
                          ]

# Use timeROC with joint model predictions
jm_roc_result <- timeROC(
  T = jm_preds$data$time_to_event_yr, 
  delta = jm_preds$data$label, 
  marker = jm_preds$preds$pred, 
  cause = 1, 
  times = seq(1,10)
)

####

for (nn in 1:5){
  print(nn)
  print(res_auc[[nn]])
  print(res_baseline_roc[[nn]]$AUC)
  print(times_used[[nn]])
}

joint_model_auc = c(as.numeric(res_auc[[1]]),
                    as.numeric(res_auc[[2]]),
                    as.numeric(res_auc[[3]]),
                    as.numeric(res_auc[[4]]),
                    as.numeric(res_auc[[5]]))
baseline_model_auc = c(as.numeric(res_baseline_roc[[1]]$AUC),
                       as.numeric(res_baseline_roc[[2]]$AUC),
                       as.numeric(res_baseline_roc[[3]]$AUC),
                       as.numeric(res_baseline_roc[[4]]$AUC), 
                       as.numeric(res_baseline_roc[[5]]$AUC)
                       )
times = c(times_used[[1]],
          times_used[[2]],
          times_used[[3]],
          times_used[[4]],
          times_used[[5]]
          )
folds = rep(c(rep(1, 6),
          rep(2, 6),
          rep(3, 6),
          rep(4, 6),
          rep(5, 6)), 2)

results_df <- data.frame(
  model = c(rep("pTau217-JM", length(joint_model_auc)), rep("Baseline", length(baseline_model_auc))),
  auc = c(joint_model_auc, baseline_model_auc),
  time = c(times, times),
  fold = folds
)
# write_csv(results_df, '../../tidy_data/A4/results_AUC.csv')
results_df <- read.csv('../../tidy_data/A4/results_AUC.csv')

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
    subtitle = "pTau217 joint model vs. Baseline Cox model",
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
  scale_y_continuous(limits = c(0.3, 0.9))

# Display the plot
print(publication_plot_viridis)

# Save the plot
ggsave("../../tidy_data/A4/AUC_Over_Time_Publication_Viridis.pdf", 
       plot = publication_plot_viridis, 
       width = 8, 
       height = 6, 
       dpi = 300)


time_points <- seq(first_year+1,last_year+1)
# overlay roc curves
plot(res_roc[[1]]$year_6)
plot(baseline_roc_results, time = time_points[5], col = "red", title = FALSE, add = T)


# Define common FPR points
fpr_common <- seq(0, 1, by = 0.01)  # 0%, 1%, ..., 100%

# Function to interpolate TPR at common FPR points
interpolate_tpr <- function(roc_obj, fpr_points, model = c('joint_model', 'cox')) {
    # Extract sensitivities (TPR) and specificities (1 - FPR)
    tpr <- roc_obj$TP
    fpr <- roc_obj$FP
    
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
    interp_tpr <- approx(x = fpr, y = tpr, xout = fpr_points, ties = mean)$y
  
  return(interp_tpr)
}

# Can't get the abaove function to work with the Cox results, running it separately
interp_tpr_l <- list()
for (fold in 1:5){
  fold_interp_trp <- list()
  for (col in 1:dim(res_baseline_roc[[paste0('fold_',fold)]]$TP)[2]){
    # Extract sensitivities (TPR) and specificities (1 - FPR)
    tpr <- res_baseline_roc[[paste0('fold_',fold)]]$TP[,col]
    fpr <- res_baseline_roc[[paste0('fold_',fold)]]$FP[,col]
    
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
    interp_tpr <- approx(x = fpr, y = tpr, xout = fpr_points, ties = mean)$y
    fold_interp_trp[[col]] <- interp_tpr
  }
  interp_tpr_l[[paste0('fold_',fold)]] <- fold_interp_trp
}

# Convert to data frame for easier manipulation
year2_bl_tpr_df <- as.data.frame(cbind(
  unlist(interp_tpr_l[['fold_1']][1]), 
  unlist(interp_tpr_l[['fold_2']][1]), 
  unlist(interp_tpr_l[['fold_3']][1]), 
  unlist(interp_tpr_l[['fold_4']][1]), 
  unlist(interp_tpr_l[['fold_5']][1])
), 
)
colnames(year2_bl_tpr_df) <- c('Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5')

year3_bl_tpr_df <- as.data.frame(cbind(
  unlist(interp_tpr_l[['fold_1']][2]), 
  unlist(interp_tpr_l[['fold_2']][2]), 
  unlist(interp_tpr_l[['fold_3']][2]), 
  unlist(interp_tpr_l[['fold_4']][2]), 
  unlist(interp_tpr_l[['fold_5']][2]) 
)
)
colnames(year3_bl_tpr_df) <- c('Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5')

year4_bl_tpr_df <- as.data.frame(cbind(
  unlist(interp_tpr_l[['fold_1']][3]), 
  unlist(interp_tpr_l[['fold_2']][3]), 
  unlist(interp_tpr_l[['fold_3']][3]), 
  unlist(interp_tpr_l[['fold_4']][3]), 
  unlist(interp_tpr_l[['fold_5']][3]) 
)
)
colnames(year4_bl_tpr_df) <- c('Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5')

year5_bl_tpr_df <- as.data.frame(cbind(
  unlist(interp_tpr_l[['fold_1']][4]), 
  unlist(interp_tpr_l[['fold_2']][4]), 
  unlist(interp_tpr_l[['fold_3']][4]), 
  unlist(interp_tpr_l[['fold_4']][4]), 
  unlist(interp_tpr_l[['fold_5']][4]) 
)
)
colnames(year5_bl_tpr_df) <- c('Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5')

year6_bl_tpr_df <- as.data.frame(cbind(
  unlist(interp_tpr_l[['fold_1']][5]), 
  unlist(interp_tpr_l[['fold_2']][5]), 
  unlist(interp_tpr_l[['fold_3']][5]), 
  unlist(interp_tpr_l[['fold_4']][5]), 
  unlist(interp_tpr_l[['fold_5']][5]) 
)
)
colnames(year6_bl_tpr_df) <- c('Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5')

year7_bl_tpr_df <- as.data.frame(cbind(
  unlist(interp_tpr_l[['fold_1']][6]), 
  unlist(interp_tpr_l[['fold_2']][6]), 
  unlist(interp_tpr_l[['fold_3']][6]), 
  unlist(interp_tpr_l[['fold_4']][6]), 
  unlist(interp_tpr_l[['fold_5']][6]) 
)
)
colnames(year7_bl_tpr_df) <- c('Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5')

year2_bl_tpr_df$fpr <- fpr_common
year3_bl_tpr_df$fpr <- fpr_common
year4_bl_tpr_df$fpr <- fpr_common
year5_bl_tpr_df$fpr <- fpr_common
year6_bl_tpr_df$fpr <- fpr_common
year7_bl_tpr_df$fpr <- fpr_common

# Reshape data for summarization
year2_bl_tpr_long <- year2_bl_tpr_df %>%
  pivot_longer(cols = starts_with("Fold_"), names_to = "Fold", values_to = "TPR")
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



#### Joint model ROC curves
# Apply interpolation to all ROC curves
f1_tpr_matrix <- sapply(res_roc[[1]], interpolate_tpr, fpr_points = fpr_common, model='joint_model')
f2_tpr_matrix <- sapply(res_roc[[2]], interpolate_tpr, fpr_points = fpr_common, model='joint_model')
f3_tpr_matrix <- sapply(res_roc[[3]], interpolate_tpr, fpr_points = fpr_common, model='joint_model')
f4_tpr_matrix <- sapply(res_roc[[4]], interpolate_tpr, fpr_points = fpr_common, model='joint_model')
f5_tpr_matrix <- sapply(res_roc[[5]], interpolate_tpr, fpr_points = fpr_common, model='joint_model')

# Convert to data frame for easier manipulation
year2_tpr_df <- as.data.frame(cbind(f1_tpr_matrix[,1], 
                                    f2_tpr_matrix[,1],
                                    f3_tpr_matrix[,1],
                                    f4_tpr_matrix[,1],
                                    f5_tpr_matrix[,1]
                                    ), 
                              )
colnames(year2_tpr_df) <- c('Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5')

year3_tpr_df <- as.data.frame(cbind(f1_tpr_matrix[,2], 
                                    f2_tpr_matrix[,2],
                                    f3_tpr_matrix[,2],
                                    f4_tpr_matrix[,2],
                                    f5_tpr_matrix[,2]
)
)
colnames(year3_tpr_df) <- c('Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5')

year4_tpr_df <- as.data.frame(cbind(f1_tpr_matrix[,3], 
                                    f2_tpr_matrix[,3],
                                    f3_tpr_matrix[,3],
                                    f4_tpr_matrix[,3],
                                    f5_tpr_matrix[,3]
)
)
colnames(year4_tpr_df) <- c('Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5')

year5_tpr_df <- as.data.frame(cbind(f1_tpr_matrix[,4], 
                                    f2_tpr_matrix[,4],
                                    f3_tpr_matrix[,4],
                                    f4_tpr_matrix[,4],
                                    f5_tpr_matrix[,4]
)
)
colnames(year5_tpr_df) <- c('Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5')

year6_tpr_df <- as.data.frame(cbind(f1_tpr_matrix[,5], 
                                    f2_tpr_matrix[,5],
                                    f3_tpr_matrix[,5],
                                    f4_tpr_matrix[,5],
                                    f5_tpr_matrix[,5]
)
)
colnames(year6_tpr_df) <- c('Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5')

year7_tpr_df <- as.data.frame(cbind(f1_tpr_matrix[,6], 
                                    f2_tpr_matrix[,6],
                                    f3_tpr_matrix[,6],
                                    f4_tpr_matrix[,6],
                                    f5_tpr_matrix[,6]
)
)
colnames(year7_tpr_df) <- c('Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5')

year2_tpr_df$fpr <- fpr_common
year3_tpr_df$fpr <- fpr_common
year4_tpr_df$fpr <- fpr_common
year5_tpr_df$fpr <- fpr_common
year6_tpr_df$fpr <- fpr_common
year7_tpr_df$fpr <- fpr_common

# Reshape data for summarization
year2_tpr_long <- year2_tpr_df %>%
  pivot_longer(cols = starts_with("Fold_"), names_to = "Fold", values_to = "TPR")
year3_tpr_long <- year3_tpr_df %>%
  pivot_longer(cols = starts_with("Fold_"), names_to = "Fold", values_to = "TPR")
year4_tpr_long <- year4_tpr_df %>%
  pivot_longer(cols = starts_with("Fold_"), names_to = "Fold", values_to = "TPR")
year5_tpr_long <- year5_tpr_df %>%
  pivot_longer(cols = starts_with("Fold_"), names_to = "Fold", values_to = "TPR")
year6_tpr_long <- year6_tpr_df %>%
  pivot_longer(cols = starts_with("Fold_"), names_to = "Fold", values_to = "TPR")
year7_tpr_long <- year7_tpr_df %>%
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

year <- 7
bl_summ <- summarize_tpr(get(paste0('year',year,'_bl_tpr_long')), 5)
bl_summ <- bl_summ %>%
  mutate(Model = "Baseline")

jm_summ <- summarize_tpr(get(paste0('year',year,'_tpr_long')), 5)
jm_summ <- jm_summ %>%
  mutate(Model = "pTau217-JM")

# Combine the two data frames into one
combined <- bind_rows(bl_summ, jm_summ)

# Plot the combined ROC curves
ggplot(combined, aes(x = fpr, y = Mean_TPR, color = Model, fill = Model)) +
  geom_line(size = 1) +  # Plot ROC lines
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



## Visualize Age vs. pTau217
library(ggplot2)

# Create a sequence of AGEYR values for plotting
age_seq <- seq(min(df$AGEYR_z, na.rm = TRUE), max(df$AGEYR_z, na.rm = TRUE), length.out = 100)

# Predict ORRES_boxcox based on the fixed effects
spline_pred <- predict(lmeFit, newdata = data.frame(
  AGEYR_z = age_seq,
  SEX = "1",            # Assuming 'SEX' is coded as 1 and 2
  EDCCNTU_z = 0,          # Mean of z-scored EDCCNTU
  APOEGN = "E2/E2"  # Replace with the reference category of APOEGN
), level = 0)

# Plot
ggplot(data = data.frame(AGEYR_z = age_seq, ORRES_boxcox = spline_pred), aes(x = AGEYR_z, y = ORRES_boxcox)) +
  geom_line(color = "blue") +
  labs(title = "Relationship Between Age and ORRES_boxcox",
       x = "Age (z-scored)",
       y = "ORRES_boxcox Predicted") +
  theme_minimal()

## Plot individual-specific trajectories
pts_morethanone_timepoints <- names(table(df$BID)[table(df$BID) > 1])
df_pred <- df %>%
  mutate(predictORRES = predict(lmeFit, level = 1))  # level = 1 for random effects

ggplot(df_pred %>% filter(BID %in% sample(pts_morethanone_timepoints, 30)),
       aes(x = COLLECTION_DATE_DAYS_CONSENT_yr, y = ORRES_boxcox, color = as.factor(BID))) +
  geom_point() +
  geom_line(aes(y = predictORRES)) +
  labs(title = "Individual-Specific Trajectories",
       x = "Time (years)",
       y = "ORRES_boxcox",
       color = "BID") +
  theme_minimal()


ROC.bili.cox<-timeROC(T=val_df$time_to_event_yr,
                      delta=val_df$label,marker=val_df$bili,
                      other_markers=as.matrix(pbc[,c("chol","albumin")]),
                      cause=1,weighting="cox",
                      times=quantile(pbc$time,probs=seq(0.2,0.8,0.1)))


modtvROC <- function(object, newdata, Tstart, Thoriz = NULL, Dt = NULL, 
          type_weights = c("model-based", "IPCW"), ...) 
  {
    if (!inherits(object, "jm")) 
      stop("Use only with 'jm' objects.\n")
    if (!is.data.frame(newdata) || nrow(newdata) == 0) 
      stop("'newdata' must be a data.frame with more than one rows.\n")
    if (is.null(Thoriz) && is.null(Dt)) 
      stop("either 'Thoriz' or 'Dt' must be non null.\n")
    if (!is.null(Thoriz) && Thoriz <= Tstart) 
      stop("'Thoriz' must be larger than 'Tstart'.")
    if (is.null(Thoriz)) 
      Thoriz <- Tstart + Dt
    type_censoring <- object$model_info$type_censoring
    if (object$model_info$CR_MS) 
      stop("'tvROC()' currently only works for right censored data.")
    type_weights <- match.arg(type_weights)
    Tstart <- Tstart + 1e-06
    Thoriz <- Thoriz + 1e-06
    id_var <- object$model_info$var_names$idVar
    time_var <- object$model_info$var_names$time_var
    Time_var <- object$model_info$var_names$Time_var
    event_var <- object$model_info$var_names$event_var
    if (is.null(newdata[[id_var]])) 
      stop("cannot find the '", id_var, "' variable in newdata.", 
           sep = "")
    if (is.null(newdata[[time_var]])) 
      stop("cannot find the '", time_var, "' variable in newdata.", 
           sep = "")
    if (any(sapply(Time_var, function(nmn) is.null(newdata[[nmn]])))) 
      stop("cannot find the '", paste(Time_var, collapse = ", "), 
           "' variable(s) in newdata.", sep = "")
    if (is.null(newdata[[event_var]])) 
      stop("cannot find the '", event_var, "' variable in newdata.", 
           sep = "")
    tt <- if (type_censoring == "right") 
      newdata[[Time_var]]
    else newdata[[Time_var[2L]]]
    newdata[[id_var]] <- newdata[[id_var]][, drop = TRUE]
    id <- newdata[[id_var]]
    id <- match(id, unique(id))
    tt <- ave(tt, id, FUN = function(t) rep(tail(t, 1L) > Tstart, 
                                            length(t)))
    newdata <- newdata[as.logical(tt), ]
    newdata <- newdata[newdata[[time_var]] <= Tstart, ]
    if (!nrow(newdata)) 
      stop("there are no data on subjects who had an observed event time after Tstart ", 
           "and longitudinal measurements before Tstart.")
    test1 <- newdata[[Time_var]] < Thoriz & newdata[[event_var]] == 
      1
    if (!any(test1)) 
      stop("it seems that there are no events in the interval [Tstart, Thoriz).")
    newdata2 <- newdata
    newdata2[[Time_var]] <- Tstart
    newdata2[[event_var]] <- 0
    preds <- predict(object, newdata = newdata2, process = "event", 
                     times = Thoriz, ...)
    return(list(preds = preds, data = newdata2))
}
