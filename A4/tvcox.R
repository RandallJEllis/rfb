library(arrow)
library(tidyverse)
library(pec)
library(timeROC)
library(pROC)
library(yardstick)
library(dplyr)
library(ggplot2)
library(survival)
library(survminer)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("pub_figures.R")
source("metrics.R")

# Load fonts
extrafont::loadfonts()

cut_time_data <- function(td_data, interval_years = 0.8) {
  # Create sequence of timepoints for each ID
  td_data %>%
    group_by(id) %>%
    mutate(
      # Round start and stop times to nearest interval
      tstart = floor(tstart / interval_years) * interval_years,
      tstop = ceiling(tstop / interval_years) * interval_years
    ) %>%
    # If this creates duplicate rows, keep last observation
    group_by(id, tstart, tstop) %>%
    slice_tail(n = 1) %>%
    ungroup()
}

format_df <- function(df) {
  df$SEX <- factor(df$SEX)
  df$APOEGN <- factor(df$APOEGN)
  df <- within(df, APOEGN <- relevel(APOEGN, ref = "E3/E3"))

  base <- df[!duplicated(df$BID), c("BID", "time_to_event", "label", 
                                    "AGEYR_centered", "AGEYR_centered_squared",
                                    "AGEYR_centered_cubed", "SEX",
                                    "EDCCNTU_z", "APOEGN")]
  tv_covar <- df[, c("BID", "COLLECTION_DATE_DAYS_CONSENT", "ORRES_boxcox")]
  habits <- habits[habits$BID %in% df$BID, c("BID",
                                             "COLLECTION_DATE_DAYS_CONSENT",
                                             "SMOKE", "ALCOHOL", "SUBUSE",
                                             "AEROBIC", "WALKING")]
  psychwell <- psychwell[psychwell$BID %in% df$BID, c("BID",
                                                "COLLECTION_DATE_DAYS_CONSENT",
                                                "GDTOTAL", "STAITOTAL")]
  vitals <- vitals[vitals$BID %in% df$BID, c("BID",
                                            "COLLECTION_DATE_DAYS_CONSENT",
                                            "VSBPSYS", "VSBPDIA")]
  # print(dim(habits))
  # print(dim(psychwell))
  # print(dim(vitals))
  colnames(base) <- c("id", "time", "event", "age", "age2", "age3",
                      "sex", "educ", "apoe")
  colnames(tv_covar) <- c("id", "time", "ptau")
  colnames(habits) <- c("id", "time", "smoke", "alcohol", "subuse",
                        "aerobic", "walking")
  colnames(psychwell) <- c("id", "time", "gdtotal", "staital")
  colnames(vitals) <- c("id", "time", "vsbsys", "vsdia")


  base$time <- base$time / 365.25
  tv_covar$time <- tv_covar$time / 365.25
  habits$time <- habits$time / 365.25
  psychwell$time <- psychwell$time / 365.25
  vitals$time <- vitals$time / 365.25

  # Create initial time-dependent data
  td_data <- tmerge(
    data1 = base,
    data2 = base,
    id = id,
    tstart = 0,
    tstop = time
  )

  # Add the event column
  td_data <- tmerge(
    td_data,
    base,
    id = id,
    event = event(time, event)
  )

  td_data <- tmerge(
    td_data,
    tv_covar,
    id = id,
    ptau = tdc(time, ptau)
  )

  td_data <- tmerge(
    td_data,
    habits,
    id = id,
    smoke = tdc(time, smoke),
    alcohol = tdc(time, alcohol),
    subuse = tdc(time, subuse),
    aerobic = tdc(time, aerobic),
    walking = tdc(time, walking)
  )

  td_data <- tmerge(
    td_data,
    psychwell,
    id = id,
    gdtotal = tdc(time, gdtotal),
    staital = tdc(time, staital)
  )

  td_data <- tmerge(  
    td_data,
    vitals,
    id = id,
    vsbsys = tdc(time, vsbsys),
    vsdia = tdc(time, vsdia)
  )

  td_data <- td_data[order(td_data$id), ]
  td_data <- td_data[complete.cases(td_data), ]

  # First, let's store the baseline age for each person
  baseline_ages <- td_data %>%
    group_by(id) %>%
    slice_min(tstart) %>%
    select(id, baseline_age = age)

  # Now update the age column to reflect actual age at each timepoint
  td_data_updated <- td_data %>%
    left_join(baseline_ages, by = "id") %>%
    mutate(
      # Convert tstart from days to years and add to baseline age
      age = baseline_age + (tstart / 365.25)
    ) %>%
    select(-baseline_age)  # Remove the temporary baseline_age column

  td_data_updated <- cut_time_data(td_data_updated)
  return(td_data_updated)
}

# Helper function to process model predictions
process_model_predictions <- function(preds, model_name, val_df, t,
                                      fixed_breaks, fold) {
  risk_groups <- cut(preds, breaks = fixed_breaks, include.lowest = TRUE)
  cal_data <- data.frame()

  for (group in levels(risk_groups)) {
    group_data <- val_df[risk_groups == group, ]
    if (nrow(group_data) > 0) {
      surv_fit <- survfit(Surv(tstop, event) ~ 1, data = group_data)
      surv_summary <- summary(surv_fit, times = t)

      if (length(surv_summary$surv) > 0) {
        cal_data <- rbind(cal_data, data.frame(
          fold = fold,
          time = t,
          model = model_name,
          risk_group = group,
          pred = mean(preds[risk_groups == group]),
          actual = 1 - surv_summary$surv[1]
        ))
      }
    }
  }
  return(cal_data)
}

test_assumptions <- function(cox_model) {
  # Test proportional hazards assumption
  test <- cox.zph(cox_model)
  print(test)

  # Test linearity of the predictor
  ggcoxzph(test)
}

# read in data
habits <- read_parquet("../../tidy_data/A4/habits.parquet")
habits$SUBUSE <- as.factor(habits$SUBUSE)
psychwell <- read_parquet("../../tidy_data/A4/psychwell.parquet")
vitals <- read_parquet("../../tidy_data/A4/vitals.parquet")


val_df_l <- list()
model_l <- list()
baseline_model_l <- list()
metrics_l <- list()
baseline_metrics_l <- list()
plots_l <- list()
baseline_plots_l <- list()

for (fold in seq(0, 4)) {
  print(paste0("Fold ", fold + 1))

  # Read the parquet file
  df <- read_parquet(paste0("../../tidy_data/A4/train_",
                            fold,
                            "_new.parquet"))
  val_df <- read_parquet(paste0("../../tidy_data/A4/val_",
                                fold,
                                "_new.parquet"))


  df <- format_df(df)
  val_df <- format_df(val_df)

  # zscore smoke, alcohol, aerobic, walking, gdtotal, staital, vsbsys, vsdia
  # calculate means and SDs for training and transform training and test sets

  means <- apply(df[, c("smoke", "alcohol", "aerobic", "walking",
                        "gdtotal", "staital", "vsbsys", "vsdia"
                        )],
                2, mean, na.rm = TRUE)
  sds <- apply(df[, c("smoke", "alcohol", "aerobic", "walking",
                      "gdtotal", "staital", "vsbsys", "vsdia"
                      )],
              2, sd, na.rm = TRUE)

  df[, c("smoke", "alcohol", "aerobic", "walking",
         "gdtotal", "staital", "vsbsys", "vsdia"
         )] <- scale(
          df[, c("smoke", "alcohol", "aerobic", "walking",
         "gdtotal", "staital", "vsbsys", "vsdia"
         )],
         center = means,
         scale = sds)

  val_df[, c("smoke", "alcohol", "aerobic", "walking",
             "gdtotal", "staital", "vsbsys", "vsdia"
             )] <- scale(
              val_df[, c("smoke", "alcohol", "aerobic", "walking",
             "gdtotal", "staital", "vsbsys", "vsdia"
             )],
             center = means,
             scale = sds)
  val_df_l[[paste0("fold_", fold + 1)]] <- val_df

  # Fit Cox model with time-varying covariates
  # Use a time-transform function in the Cox model
  model <- coxph(
    Surv(tstart, tstop, event) ~ ptau + age + age2 + sex + educ +
      smoke + alcohol + subuse + 
      aerobic + walking + 
      gdtotal + staital +
      vsbsys + vsdia + 
      apoe + age * apoe + age2 * apoe,
    data = df,
    x = TRUE
  )

  model_l[[paste0("fold_", fold + 1)]] <- model

  baseline_model <- coxph(
    Surv(tstart, tstop, event) ~ age + age2 + sex + educ +
      smoke + alcohol + subuse + 
      aerobic + walking + 
      gdtotal + staital +
      vsbsys + vsdia + 
      apoe + age * apoe + age2 * apoe,
    data = df,
    x = TRUE
  )

  baseline_model_l[[paste0("fold_", fold + 1)]] <- baseline_model

  print("Models fitted")

  # # View model summary
  # summary(model)

  # # Extract deviance residuals
  # residuals <- residuals(baseline_model, type = "deviance")

  # # Generate Q-Q plot
  # qqnorm(residuals, main = "Q-Q Plot of Deviance Residuals")
  # qqline(residuals, col = "red")
  # ggcoxdiagnostics(model)


  # Define time points for evaluation
  # Take first event, and round it up to the nearest multiple of 365
  # Take last event, and round it down to the nearest multiple of 365
  # first_event <- min(val_df[val_df$event == 1, ]$time)
  # last_event <- max(val_df[val_df$event == 1, ]$time)
  eval_times <- seq(3, 8)

  # Calculate all metrics
  metrics_results <- calculate_survival_metrics(
    model = model,
    data = val_df,
    times = eval_times
  )

  metrics_l[[paste0("fold_", fold + 1)]] <- metrics_results

  baseline_metrics_results <- calculate_survival_metrics(
    model = baseline_model,
    data = val_df,
    times = eval_times
  )

  baseline_metrics_l[[paste0("fold_", fold + 1)]] <- baseline_metrics_results
  
  print("Metrics calculated")

  # Print numerical results
  print("Time-dependent AUC:")
  print(metrics_results$auc)
  print(baseline_metrics_results$auc)

  print("\nHarrell's C-index:")
  print(metrics_results$concordance)
  print(baseline_metrics_results$concordance)

  print("\nIntegrated Brier Score:")
  print(mean(metrics_results$brier$brier, na.rm = TRUE))
  print(mean(baseline_metrics_results$brier$brier, na.rm = TRUE))

}

# Calculate p-values comparing AUCs between models at each time point
# First combine timeROC objects from each fold for each model
biomarker_trocs <- list(
  metrics_l$fold_1$troc,
  metrics_l$fold_2$troc, 
  metrics_l$fold_3$troc,
  metrics_l$fold_4$troc,
  metrics_l$fold_5$troc
)

baseline_trocs <- list(
  baseline_metrics_l$fold_1$troc,
  baseline_metrics_l$fold_2$troc,
  baseline_metrics_l$fold_3$troc, 
  baseline_metrics_l$fold_4$troc,
  baseline_metrics_l$fold_5$troc
)

# Initialize list to store p-values
all_pvalues <- list()

# Calculate p-values for each time point using timeROC::compare
for (fold in seq_along(biomarker_trocs)) {
  # Compare ROC curves for this fold and time point
  comparison <- timeROC::compare(
    biomarker_trocs[[fold]],
    baseline_trocs[[fold]],
    adjusted = TRUE  # Use adjusted p-values
  )
  # Store p-values for this fold
  all_pvalues[[fold]] <- comparison$p_values_AUC[2,]
}

# Convert to data frame
all_pvalues_df <- do.call(rbind, all_pvalues)
hist(all_pvalues_df)
mean(all_pvalues_df)
median(all_pvalues_df)
range(all_pvalues_df)

collate_metric <- function(metrics_l, baseline_metrics_l, metric = "auc") {

  if (metric != "concordance") {
    ptau <- c(as.numeric(metrics_l$fold_1[[metric]][[metric]]),
      as.numeric(metrics_l$fold_2[[metric]][[metric]]),
      as.numeric(metrics_l$fold_3[[metric]][[metric]]),
      as.numeric(metrics_l$fold_4[[metric]][[metric]]),
      as.numeric(metrics_l$fold_5[[metric]][[metric]])
    )
    baseline <- c(as.numeric(baseline_metrics_l$fold_1[[metric]][[metric]]),
      as.numeric(baseline_metrics_l$fold_2[[metric]][[metric]]),
      as.numeric(baseline_metrics_l$fold_3[[metric]][[metric]]),
      as.numeric(baseline_metrics_l$fold_4[[metric]][[metric]]),
      as.numeric(baseline_metrics_l$fold_5[[metric]][[metric]])
    )
  } else {
    ptau <- c(as.numeric(metrics_l$fold_1[[metric]][["AppCindex"]][["coxph"]]),
      as.numeric(metrics_l$fold_2[[metric]][["AppCindex"]][["coxph"]]),
      as.numeric(metrics_l$fold_3[[metric]][["AppCindex"]][["coxph"]]),
      as.numeric(metrics_l$fold_4[[metric]][["AppCindex"]][["coxph"]]),
      as.numeric(metrics_l$fold_5[[metric]][["AppCindex"]][["coxph"]])
    )

    baseline <- c(as.numeric(
      baseline_metrics_l$fold_1[[metric]][["AppCindex"]][["coxph"]]),
      as.numeric(baseline_metrics_l$fold_2[[metric]][["AppCindex"]][["coxph"]]),
      as.numeric(baseline_metrics_l$fold_3[[metric]][["AppCindex"]][["coxph"]]),
      as.numeric(baseline_metrics_l$fold_4[[metric]][["AppCindex"]][["coxph"]]),
      as.numeric(baseline_metrics_l$fold_5[[metric]][["AppCindex"]][["coxph"]])
    )
  }
  times <- c(as.numeric(metrics_l$fold_1[[metric]]$time),
    as.numeric(metrics_l$fold_2[[metric]]$time),
    as.numeric(metrics_l$fold_3[[metric]]$time),
    as.numeric(metrics_l$fold_4[[metric]]$time),
    as.numeric(metrics_l$fold_5[[metric]]$time)
  )
  folds <- rep(c(rep(1, 6),
                rep(2, 6),
                rep(3, 6),
                rep(4, 6),
                rep(5, 6)), 2)

  df <- data.frame(
    model = c(rep("Biomarker", length(ptau)),
              rep("Baseline", length(baseline))),
    metric = c(ptau, baseline),
    time = c(times, times),
    fold = folds
  )
  names(df)[names(df) == "metric"] <- metric

  return(df)
}

########################################################
# AUC, Brier Score, and Concordance Over Time
auc_results <- collate_metric(metrics_l, baseline_metrics_l, metric = "auc")
brier_results <- collate_metric(metrics_l, baseline_metrics_l, metric = "brier")
concordance_results <- collate_metric(metrics_l, baseline_metrics_l,
                                      metric = "concordance")

write_csv(auc_results, "../../tidy_data/A4/results_AUC.csv")
write_csv(brier_results, "../../tidy_data/A4/results_brier.csv")
write_csv(concordance_results, "../../tidy_data/A4/results_concordance.csv")

# Save objects
saveRDS(model_l, "../../tidy_data/A4/fitted_ptau_cox.rds")
saveRDS(baseline_model_l, "../../tidy_data/A4/fitted_baseline_cox.rds")
saveRDS(metrics_l, "../../tidy_data/A4/ptau_metrics.rds")
saveRDS(baseline_metrics_l, "../../tidy_data/A4/baseline_metrics.rds")
saveRDS(plots_l, "../../tidy_data/A4/ptau_plots.rds")
saveRDS(baseline_plots_l, "../../tidy_data/A4/baseline_plots.rds")

# plot auc over time
auc_summary <- auc_results %>%
  group_by(model, time) %>%
  summarise(
    mean_AUC = mean(auc, na.rm = TRUE),
    sd_AUC = sd(auc, na.rm = TRUE),
    ymin = pmax(mean_AUC - sd_AUC, 0),
    ymax = pmin(mean_AUC + sd_AUC, 1),
    .groups = "drop"
  )

auc_plot <- td_plot(auc_summary, metric = "auc")

# Display the plot
print(auc_plot)

# Save the plot
ggsave("../../tidy_data/A4/final_auc_Over_Time.pdf",
       plot = auc_plot,
       width = 8,
       height = 6,
       dpi = 300)

# plot brier score over time
brier_summary <- brier_results %>%
  group_by(model, time) %>%
  summarise(
    mean_brier = mean(brier, na.rm = TRUE),
    sd_brier = sd(brier, na.rm = TRUE),
    ymin = pmax(mean_brier - sd_brier, 0),
    ymax = pmin(mean_brier + sd_brier, 1),
    .groups = "drop"
  )

# print differences between biomarker and baseline at each time point
brier_summary %>%
  group_by(time) %>%
  summarise(
    mean_diff = mean_brier[model == "Biomarker"] - mean_brier[model == "Baseline"],
    .groups = "drop"
  ) %>%
  print()

ibs_results <- pec::pec(list("Biomarker" = model_l[[1]], "Baseline" = baseline_model_l[[1]]),
        data = val_df_l[[1]],
        formula = Surv(tstop, event) ~ ptau,
        # times = eval_times,
        # metrics = "ibs",
        testIBS = TRUE,
        testTimes = eval_times)

ibs_results

brier_plot <- td_plot(brier_summary, metric = "brier")

# Display the plot
print(brier_plot)

# Save the plot
ggsave("../../tidy_data/A4/final_brier_Over_Time.pdf",
       plot = brier_plot,
       width = 8,
       height = 6,
       dpi = 300)

# plot concordance over time
concordance_summary <- concordance_results %>%
  group_by(model, time) %>%
  summarise(
    mean_concordance = mean(concordance, na.rm = TRUE),
    sd_concordance = sd(concordance, na.rm = TRUE),
    ymin = pmax(mean_concordance - sd_concordance, 0),
    ymax = pmin(mean_concordance + sd_concordance, 1),
    .groups = "drop"
  )

# print differences between biomarker and baseline at each time point
concordance_summary %>%
  group_by(time) %>%
  summarise(
    mean_diff = mean_concordance[model == "Biomarker"] - mean_concordance[model == "Baseline"],
    .groups = "drop"
  ) %>%
  print()


concordance_plot <- td_plot(concordance_summary, metric = "concordance")

# Display the plot
print(concordance_plot)

# Save the plot
ggsave("../../tidy_data/A4/final_concordance_Over_Time.pdf",
       plot = concordance_plot,
       width = 8,
       height = 6,
       dpi = 300)



########################################################
# Calibration plots
# Initialize lists to store predictions for each time point
# this helps us create consistent bins
all_preds_by_time <- list()
cal_data_all <- list()  # This will store calibration data for all time points

# First pass: collect all predictions for each time point to create fixed bins
for (t in eval_times) {
  all_preds <- c()

  for (fold in seq(0, 4)) {

    baseline_model <- overwrite_na_coef_to_zero(
      baseline_model_l[[paste0("fold_", fold + 1)]]
    )
    model <- overwrite_na_coef_to_zero(
      model_l[[paste0("fold_", fold + 1)]]
    )

    # Get predictions from both models for this fold and time point
    pred_probs_base <- 1 - pec::predictSurvProb(
      baseline_model,
      newdata = val_df_l[[paste0("fold_", fold + 1)]],
      times = t
    )
    pred_probs_bio <- 1 - pec::predictSurvProb(
      model,
      newdata = val_df_l[[paste0("fold_", fold + 1)]],
      times = t
    )
    all_preds <- c(all_preds, pred_probs_base, pred_probs_bio)
  }

  # Store all predictions for this time point
  all_preds_by_time[[as.character(t)]] <- all_preds
}



# Second pass: calculate calibration using fixed bins for each time point
for (t in eval_times) {
  fixed_breaks <- quantile(
    all_preds_by_time[[as.character(t)]],
    probs = seq(0, 1, length.out = 11)
  )
  cal_data_folds <- list()

  for (fold in seq(0, 4)) {
    baseline_model <- overwrite_na_coef_to_zero(
      baseline_model_l[[paste0("fold_", fold + 1)]]
    )
    model <- overwrite_na_coef_to_zero(
      model_l[[paste0("fold_", fold + 1)]]
    )

    pred_probs_base <- 1 - pec::predictSurvProb(
      baseline_model,
      newdata = val_df_l[[paste0("fold_", fold + 1)]],
      times = t
    )
    pred_probs_bio <- 1 - pec::predictSurvProb(
      model,
      newdata = val_df_l[[paste0("fold_", fold + 1)]],
      times = t
    )

    base_data <- process_model_predictions(
      pred_probs_base,
      "Baseline",
      val_df_l[[paste0("fold_", fold + 1)]],
      t,
      fixed_breaks,
      fold
    )
    bio_data <- process_model_predictions(
      pred_probs_bio,
      "Biomarker",
      val_df_l[[paste0("fold_", fold + 1)]],
      t,
      fixed_breaks,
      fold
    )

    cal_data_folds[[paste0("fold_", fold + 1)]] <- rbind(base_data, bio_data)
  }

  cal_data_all[[as.character(t)]] <- cal_data_folds
}

# Combine all data and calculate confidence intervals
all_cal_data <- do.call(rbind, lapply(cal_data_all, function(time_data) {
  do.call(rbind, time_data)
}))

# Sort data within each time-model combination and use rolling windows
cal_data_avg <- all_cal_data %>%
  group_by(time, model) %>%
  arrange(pred) %>%
  mutate(
    # Calculate rolling means and SDs using 3 adjacent risk groups
    rolling_mean = zoo::rollmean(
      actual,
      k = 3,
      fill = NA,
      align = "center"
    ),
    rolling_sd = zoo::rollapply(
      actual,
      width = 3,
      FUN = sd,
      fill = NA,
      align = "center"
    )
  ) %>%
  group_by(time, model, risk_group) %>%
  summarize(
    pred = mean(pred),
    actual = mean(actual),
    sd = mean(rolling_sd, na.rm = TRUE),
    lower = actual - sd,  # mean - 1 SD
    upper = actual + sd,  # mean + 1 SD
    .groups = "drop"
  )



plots <- calibration_plots(cal_data_avg, eval_times, model_colors)
print(plots)

# save plot
ggsave("../../tidy_data/A4/final_calibration_plots.pdf",
       plot = plots,
       width = 8,
       height = 6,
       dpi = 300)


########################################################
# Decision curve analysis
# Initialize lists to store predictions for each time point
all_preds_by_time <- list()
dca_data_all <- list()

# First pass: collect all predictions for each time point
for (t in eval_times) {
  all_preds <- c()

  for (fold in seq(0, 4)) {
    baseline_model <- overwrite_na_coef_to_zero(
      baseline_model_l[[paste0("fold_", fold + 1)]]
    )
    model <- overwrite_na_coef_to_zero(
      model_l[[paste0("fold_", fold + 1)]]
    )

    # Get predictions from both models for this fold and time point
    pred_probs_base <- 1 - pec::predictSurvProb(
      baseline_model,
      newdata = val_df_l[[paste0("fold_", fold + 1)]],
      times = t
    )
    pred_probs_bio <- 1 - pec::predictSurvProb(
      model,
      newdata = val_df_l[[paste0("fold_", fold + 1)]],
      times = t
    )

    # Create data frame for this fold's predictions
    val_data <- val_df_l[[paste0("fold_", fold + 1)]]
    dca_fold_data <- data.frame(
      fold = fold,
      time = t,
      tstop = val_data$tstop,
      event = val_data$event,
      baseline_pred = pred_probs_base,
      biomarker_pred = pred_probs_bio
    )

    dca_data_all[[paste0("t", t, "_fold", fold)]] <- dca_fold_data
  }
}

# Combine all DCA data
all_dca_data <- do.call(rbind, dca_data_all)

# Create decision curve plots
plots <- list()
model_colors <- c("Baseline" = "#287271", "Biomarker" = "#B63679")

for (t in eval_times) {
  # Store net benefit values for each fold
  fold_results <- list()
  
  for (fold in seq(0, 4)) {
    t_data <- all_dca_data[all_dca_data$time == t & all_dca_data$fold == fold,]
    
    # Calculate DCA for this fold
    dca_baseline <- stdca(data = t_data, outcome = "event", ttoutcome = "tstop",
                         timepoint = t, predictors = "baseline_pred",
                         xstart = 0, xstop = 1,
                         probability = FALSE, harm = NULL, graph = FALSE)
    
    dca_biomarker <- stdca(data = t_data, outcome = "event", ttoutcome = "tstop",
                          timepoint = t, predictors = "biomarker_pred",
                          xstart = 0, xstop = 1,
                          probability = FALSE, harm = NULL, graph = FALSE)
    
    fold_results[[fold + 1]] <- list(
      threshold = dca_baseline$net.benefit$threshold,
      none = dca_baseline$net.benefit$none,
      all = dca_baseline$net.benefit$all,
      baseline = dca_baseline$net.benefit$baseline_pred,
      biomarker = dca_biomarker$net.benefit$biomarker_pred
    )
  }
  
  # Calculate mean and SD across folds
  thresholds <- fold_results[[1]]$threshold
  n_thresholds <- length(thresholds)
  
  # Initialize matrices to store values
  none_vals <- matrix(NA, nrow = 5, ncol = n_thresholds)
  all_vals <- matrix(NA, nrow = 5, ncol = n_thresholds)
  baseline_vals <- matrix(NA, nrow = 5, ncol = n_thresholds)
  biomarker_vals <- matrix(NA, nrow = 5, ncol = n_thresholds)
  
  # Fill matrices with values from each fold
  for (fold in 1:5) {
    none_vals[fold,] <- fold_results[[fold]]$none
    all_vals[fold,] <- fold_results[[fold]]$all
    baseline_vals[fold,] <- fold_results[[fold]]$baseline
    biomarker_vals[fold,] <- fold_results[[fold]]$biomarker
  }
  
  # Calculate means and SDs
  none_mean <- colMeans(none_vals, na.rm = TRUE)
  all_mean <- colMeans(all_vals, na.rm = TRUE)
  baseline_mean <- colMeans(baseline_vals, na.rm = TRUE)
  biomarker_mean <- colMeans(biomarker_vals, na.rm = TRUE)
  
  # Calculate SD 
  baseline_sd <- apply(baseline_vals, 2, sd, na.rm = TRUE)
  biomarker_sd <- apply(biomarker_vals, 2, sd, na.rm = TRUE)
  
  is_leftmost <- as.numeric(t) %in% c(3, 6)
  is_bottom <- as.numeric(t) >= 6
  is_middle_bottom <- t == 7
  
  current_plot <- ggplot() +
    # Add reference lines
    geom_line(data = data.frame(x = thresholds, y = none_mean),
              aes(x = x, y = y, linetype = "Treat None"), color = "gray50") +
    geom_line(data = data.frame(x = thresholds, y = all_mean),
              aes(x = x, y = y, linetype = "Treat All"), color = "gray50") +
    # Add model lines with confidence bands using SD instead of SE
    geom_ribbon(data = data.frame(
      x = thresholds,
      y = baseline_mean,
      ymin = baseline_mean - baseline_sd,  # Changed from 1.96 * SE to SD
      ymax = baseline_mean + baseline_sd
    ),
    aes(x = x, y = y, ymin = ymin, ymax = ymax, fill = "Baseline"), alpha = 0.2) +
    geom_ribbon(data = data.frame(
      x = thresholds,
      y = biomarker_mean,
      ymin = biomarker_mean - biomarker_sd,  # Changed from 1.96 * SE to SD
      ymax = biomarker_mean + biomarker_sd
    ),
    aes(x = x, y = y, ymin = ymin, ymax = ymax, fill = "Biomarker"), alpha = 0.2) +
    geom_line(data = data.frame(x = thresholds, y = baseline_mean),
              aes(x = x, y = y, color = "Baseline"), linewidth = 1) +
    geom_line(data = data.frame(x = thresholds, y = biomarker_mean),
              aes(x = x, y = y, color = "Biomarker"), linewidth = 1) +
    scale_color_manual(values = model_colors, name = "Model") +
    scale_fill_manual(values = model_colors, name = "Model") +
    scale_linetype_manual(
      values = c("Treat None" = "dashed", "Treat All" = "dotted"),
      name = "Strategy"
    ) +
    scale_y_continuous(limits = c(-0.05, NA)) +
    labs(
      x = if (is_bottom) "Threshold Probability" else "",
      y = if (is_leftmost) "Net Benefit" else "",
      title = paste(t, "years")
    ) +
    get_publication_theme() +
    theme(legend.position = if (is_middle_bottom) "bottom" else "none") +
    theme(aspect.ratio = 0.7)

  plots[[as.character(t)]] <- current_plot
}

# Create final decision curve plot
dca_plots <- wrap_plots(plots, ncol = 3)
print(dca_plots)

# save plot
ggsave("../../tidy_data/A4/final_DCA_Over_Time.pdf",
       plot = dca_plots,
       width = 8,
       height = 6,
       dpi = 300)


########################################################
# ROC and Prediction Error Curves
# Initialize lists to store predictions for each time point
all_preds_by_time <- list()
roc_data_all <- list()
pe_data_all <- list()

# Create consistent time points
eval_times_pe <- seq(3, 8)

for (fold in seq(0, 4)) {
  baseline_model <- overwrite_na_coef_to_zero(
    baseline_model_l[[paste0("fold_", fold + 1)]]
  )
  model <- overwrite_na_coef_to_zero(
    model_l[[paste0("fold_", fold + 1)]]
  )
  
  val_data <- val_df_l[[paste0("fold_", fold + 1)]]
  
  # Run create_additional_figures for this fold with consistent time points
  additional_figs <- create_additional_figures(
    baseline_model = baseline_model,
    biomarker_model = model,
    data = val_data,
    times = eval_times  # Original eval_times for ROC curves
  )

  # Extract ROC data
  roc_data <- additional_figs$dynamic_roc$data
  roc_data$fold <- fold
  roc_data_all[[fold + 1]] <- roc_data
  
  # Extract prediction error data
  pe_data <- additional_figs$prediction_error$data
  pe_data$fold <- fold
  pe_data_all[[fold + 1]] <- pe_data
}

# Combine all data
all_roc_data <- do.call(rbind, roc_data_all)
all_pe_data <- do.call(rbind, pe_data_all)

# Calculate AUC summary statistics
auc_stats <- all_roc_data %>%
  group_by(Model, Time, fold) %>%
  # Calculate AUC for each fold using trapezoidal rule
  summarise(
    AUC = pracma::trapz(FPR, TPR),
    .groups = 'drop'
  ) %>%
  group_by(Model, Time) %>%
  summarise(
    mean_auc = mean(AUC, na.rm = TRUE),
    sd_auc = sd(AUC, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  mutate(
    auc_label = sprintf("AUC = %.2f ± %.2f", mean_auc, sd_auc)
  ) %>%
  pivot_wider(
    id_cols = Time,
    names_from = Model,
    values_from = auc_label
  ) %>%
  mutate(
    panel_title = sprintf("%d years\n%s\n%s", 
                         Time, 
                         Baseline,
                         Biomarker)
  )

# Calculate means and SDs for ROC curves with smoothing
roc_summary <- all_roc_data %>%
  group_by(Model, Time, fold) %>%
  arrange(FPR) %>%
  distinct(FPR, .keep_all = TRUE) %>%
  do({
    new_fpr <- seq(min(.$FPR), max(.$FPR), length.out = 100)
    new_tpr <- approx(.$FPR, .$TPR, xout = new_fpr, method = "linear")$y
    data.frame(
      FPR = new_fpr,
      TPR = new_tpr
    )
  }) %>%
  ungroup() %>%
  group_by(Model, Time, FPR) %>%
  summarise(
    mean_TPR = mean(TPR, na.rm = TRUE),
    sd_TPR = sd(TPR, na.rm = TRUE),
    .groups = 'drop'
  )

# Plot ROC curves with smoothing and uncertainty
model_colors <- c("Baseline" = "#287271", "Biomarker" = "#B63679")

p5 <- ggplot(roc_summary, aes(x = FPR, y = mean_TPR, color = Model)) +
  geom_ribbon(aes(ymin = pmax(mean_TPR - sd_TPR, 0),
                  ymax = pmin(mean_TPR + sd_TPR, 1),
                  fill = Model), 
              alpha = 0.2, 
              color = NA) +
  geom_smooth(se = FALSE, method = "loess", span = 0.2, linewidth = 1) +
  geom_abline(slope = 1, intercept = 0,
              linetype = "dashed", color = "gray50") +
  scale_color_manual(values = model_colors) +
  scale_fill_manual(values = model_colors) +
  facet_wrap(~Time, 
             labeller = labeller(Time = function(x) {
               auc_stats$panel_title[auc_stats$Time == x]
             })) +
  labs(
    x = "False Positive Rate",
    y = "True Positive Rate", 
    title = "Dynamic ROC Curves",
    subtitle = "At Different Follow-up Times"
  ) +
  coord_equal() +
  get_publication_theme() +
  theme(
    panel.spacing = unit(1, "cm"),  # Increase spacing between panels
    axis.text.x = element_text(angle = 0, hjust = 0.5, size = 8),  # Adjust x-axis text
    plot.margin = margin(0.5, 0.5, 0.5, 0.5, "cm")  # Add margin around entire plot
  )

print(p5)

# save plot with adjusted width-to-height ratio
ggsave("../../tidy_data/A4/final_ROCcurves_Over_Time.pdf",
       plot = p5,
       width = 14,  # Increased width
       height = 6,
       dpi = 300)

# Year 7
# Create individual panel for time = 7
roc_t7 <- roc_summary %>% 
  filter(Time == 7)

p5_t7 <- ggplot(roc_t7, aes(x = FPR, y = mean_TPR, color = Model)) +
  geom_ribbon(aes(ymin = pmax(mean_TPR - sd_TPR, 0),
                  ymax = pmin(mean_TPR + sd_TPR, 1),
                  fill = Model), 
              alpha = 0.2, 
              color = NA) +
  geom_smooth(se = FALSE, method = "loess", span = 0.2, linewidth = 1) +
  geom_abline(slope = 1, intercept = 0,
              linetype = "dashed", color = "gray50") +
  scale_color_manual(values = model_colors) +
  scale_fill_manual(values = model_colors) +
  labs(
    x = "False Positive Rate",
    y = "True Positive Rate",
    title = paste("7 years\n", 
                 auc_stats$Baseline[auc_stats$Time == 7], "\n",
                 auc_stats$Biomarker[auc_stats$Time == 7])
  ) +
  coord_equal() +
  get_publication_theme() +
  theme(
    legend.position = "bottom",
    plot.margin = margin(0.5, 0.5, 0.5, 0.5, "cm"),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1)  # Add solid border
  )

print(p5_t7)

# Save the individual panel
ggsave("../../tidy_data/A4/final_ROCcurve_7years.pdf",
       plot = p5_t7,
       width = 6,
       height = 6,
       dpi = 300)

# Calculate means and SEs for prediction error, averaging across folds first
pe_summary <- all_pe_data %>%
  group_by(Model, time, fold) %>%
  summarise(
    fold_error = mean(error, na.rm = TRUE),
    .groups = 'drop'
  ) %>%
  group_by(Model, time) %>%
  summarise(
    mean_error = mean(fold_error, na.rm = TRUE),
    sd_error = sd(fold_error, na.rm = TRUE),
    .groups = 'drop'
  )

# Plot prediction error with uncertainty
p6 <- ggplot(pe_summary, aes(x = time, y = mean_error, 
                            color = Model)) +
  geom_ribbon(aes(ymin = mean_error - sd_error,
                  ymax = mean_error + sd_error,
                  fill = Model), alpha = 0.2, color = NA) +
  geom_line(linewidth = 1) +
  # Add points at each year
  geom_point(data = pe_summary[pe_summary$time %in% 3:8,], 
             size = 3) +
  scale_color_manual(
    values = model_colors,
    labels = c("Baseline", "Biomarker")
  ) +
  scale_fill_manual(
    values = model_colors,
    labels = c("Baseline", "Biomarker")
  ) +
  labs(
    x = "Time (Years)",
    y = "Prediction Error",
    title = "Integrated Prediction Error",
    subtitle = "Lower Values Indicate Better Performance"
  ) +
  get_publication_theme()
print(p6)

# save plot
ggsave("../../tidy_data/A4/final_prediction_error.pdf",
       plot = p6,
       width = 8,
       height = 6,
       dpi = 300)
########################################################






