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

format_df <- function(df) {
  df$SEX <- factor(df$SEX)
  df$APOEGN <- factor(df$APOEGN)
  df <- within(df, APOEGN <- relevel(APOEGN, ref = "E3/E3"))

  base <- df[!duplicated(df$BID), c("BID", "time_to_event", "label", "AGEYR_z",
                                    "AGEYR_z_squared", "AGEYR_z_cubed", "SEX",
                                    "EDCCNTU_z", "APOEGN")]
  tv_covar <- df[, c("BID", "COLLECTION_DATE_DAYS_CONSENT", "ORRES_boxcox")]
  colnames(base) <- c("id", "time", "event", "age", "age2", "age3",
                      "sex", "educ", "apoe")
  colnames(tv_covar) <- c("id", "time", "ptau")

  base$time <- base$time / 365.25
  tv_covar$time <- tv_covar$time / 365.25

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

  return(td_data_updated)
}

overwrite_na_coef_to_zero <- function(model) {
  if (length(names(which(is.na(coef(model))))) > 0) {
    # set coefficients to zero
    for (n in names(which(is.na(coef(model))))) {
      model$coefficients[[n]] <- 0
    }
  }
  return(model)
}

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
  val_df_l[[paste0("fold_", fold + 1)]] <- val_df

  # Fit Cox model with time-varying covariates
  # Use a time-transform function in the Cox model
  model <- coxph(
    Surv(tstart, tstop, event) ~ ptau + age + age2 + age3 + sex + educ +
      apoe + age * apoe + age2 * apoe + age3 * apoe,
    data = df,
    x = TRUE
  )

  model_l[[paste0("fold_", fold + 1)]] <- model

  baseline_model <- coxph(
    Surv(tstart, tstop, event) ~ age + age2 + age3 + sex + educ +
      apoe + age * apoe + age2 * apoe + age3 * apoe,
    data = df,
    x = TRUE
  )

  baseline_model_l[[paste0("fold_", fold + 1)]] <- baseline_model

  print("Models fitted")

  # View model summary
  # summary(model)

  # Extract deviance residuals
  # residuals <- residuals(model, type = "deviance")

  # Generate Q-Q plot
  # qqnorm(residuals, main = "Q-Q Plot of Deviance Residuals")
  # qqline(residuals, col = "red")
  # ggcoxdiagnostics(model)


  # Define time points for evaluation
  # Take first event, and round it up to the nearest multiple of 365
  # Take last event, and round it down to the nearest multiple of 365
  first_event <- min(val_df[val_df$event == 1, ]$time)
  last_event <- max(val_df[val_df$event == 1, ]$time)
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
    se = mean(rolling_sd, na.rm = TRUE) / sqrt(n()),
    ci_lower = actual - 1.96 * se,
    ci_upper = actual + 1.96 * se,
    .groups = "drop"
  )

# Modified plotting function to include confidence intervals
calibration_plots <- function(cal_data, times, model_colors) {

  publication_theme <- get_publication_theme()
  model_colors <- c("Baseline" = "#287271", "Biomarker" = "#B63679")

  plots <- list()

  for (t in times) {
    t_data <- cal_data[cal_data$time == t, ]

    is_leftmost <- as.numeric(t) %in% c(3, 6)
    is_bottom <- as.numeric(t) >= 6
    is_middle_bottom <- t == 7

    max_limit <- max(max(t_data$pred), max(t_data$actual)) * 1.05

    current_plot <- ggplot(t_data,
                           aes(x = pred, y = actual, color = model,
                               fill = model)) +
      geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), alpha = 0.2) +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                  color = "gray50") +
      geom_line(linewidth = 1) +
      geom_point(size = 2) +
      scale_color_manual(values = model_colors, name = "Model") +
      scale_fill_manual(values = model_colors, name = "Model") +
      labs(x = if (is_bottom) "Predicted Probability" else "",
           y = if (is_leftmost) "Observed Probability" else "",
           title = paste0(t, " years")) +
      coord_equal(xlim = c(0, max_limit), ylim = c(0, max_limit)) +
      publication_theme +
      theme(legend.position = if (is_middle_bottom) "bottom" else "none")

    plots[[as.character(t)]] <- current_plot
  }

  return(wrap_plots(plots, ncol = 3))
}

# Create final plots
plots <- calibration_plots(cal_data_avg, eval_times, model_colors)
print(plots)




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
  
  # Calculate mean and SE across folds
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
  
  # Calculate means and SEs
  none_mean <- colMeans(none_vals, na.rm = TRUE)
  all_mean <- colMeans(all_vals, na.rm = TRUE)
  baseline_mean <- colMeans(baseline_vals, na.rm = TRUE)
  biomarker_mean <- colMeans(biomarker_vals, na.rm = TRUE)
  
  baseline_se <- apply(baseline_vals, 2, sd, na.rm = TRUE) / sqrt(5)
  biomarker_se <- apply(biomarker_vals, 2, sd, na.rm = TRUE) / sqrt(5)
  
  is_leftmost <- as.numeric(t) %in% c(3, 6)
  is_bottom <- as.numeric(t) >= 6
  is_middle_bottom <- t == 7
  
  current_plot <- ggplot() +
    # Add reference lines
    geom_line(data = data.frame(x = thresholds, y = none_mean),
              aes(x = x, y = y, linetype = "Treat None"), color = "gray50") +
    geom_line(data = data.frame(x = thresholds, y = all_mean),
              aes(x = x, y = y, linetype = "Treat All"), color = "gray50") +
    # Add model lines with confidence bands
    geom_ribbon(data = data.frame(
      x = thresholds,
      y = baseline_mean,
      ymin = baseline_mean - 1.96 * baseline_se,
      ymax = baseline_mean + 1.96 * baseline_se
    ),
    aes(x = x, y = y, ymin = ymin, ymax = ymax, fill = "Baseline"), alpha = 0.2) +
    geom_ribbon(data = data.frame(
      x = thresholds,
      y = biomarker_mean,
      ymin = biomarker_mean - 1.96 * biomarker_se,
      ymax = biomarker_mean + 1.96 * biomarker_se
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
    theme(legend.position = if (is_middle_bottom) "bottom" else "none")
  
  plots[[as.character(t)]] <- current_plot
}

# Create final decision curve plot
dca_plots <- wrap_plots(plots, ncol = 3)
print(dca_plots)





collate_metric <- function(metrics_l, baseline_metrics_l, metric = "auc") {

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

auc_results <- collate_metric(metrics_l, baseline_metrics_l, metric = "auc")
brier_results <- collate_metric(metrics_l, baseline_metrics_l, metric = "brier")

write_csv(auc_results, "../../tidy_data/A4/results_AUC.csv")
write_csv(brier_results, "../../tidy_data/A4/results_brier.csv")

# Save objects
saveRDS(model_l, "../../tidy_data/A4/fitted_ptau_cox.rds")
saveRDS(baseline_model_l, "../../tidy_data/A4/fitted_baseline_cox.rds")
saveRDS(metrics_l, "../../tidy_data/A4/ptau_metrics.rds")
saveRDS(baseline_metrics_l, "../../tidy_data/A4/baseline_metrics.rds")
saveRDS(plots_l, "../../tidy_data/A4/ptau_plots.rds")
saveRDS(baseline_plots_l, "../../tidy_data/A4/baseline_plots.rds")

auc_summary <- auc_results %>%
  group_by(model, time) %>%
  summarise(
    mean_AUC = mean(auc, na.rm = TRUE),
    sd_AUC = sd(auc, na.rm = TRUE),
    ymin = pmax(mean_AUC - sd_AUC, 0),
    ymax = pmin(mean_AUC + sd_AUC, 1),
    .groups = "drop"
  )

brier_summary <- brier_results %>%
  group_by(model, time) %>%
  summarise(
    mean_brier = mean(brier, na.rm = TRUE),
    sd_brier = sd(brier, na.rm = TRUE),
    ymin = pmax(mean_brier - sd_brier, 0),
    ymax = pmin(mean_brier + sd_brier, 1),
    .groups = "drop"
  )

publication_plot_viridis <- td_plot(auc_summary, metric = "auc")

# Display the plot
print(publication_plot_viridis)

# Save the plot
ggsave("../../tidy_data/A4/final_auc_Over_Time_Publication_Viridis.pdf",
       plot = publication_plot_viridis,
       width = 8,
       height = 6,
       dpi = 300)



# Pub figures
first_event <- min(val_df[val_df$event == 1, ]$time)
last_event <- max(val_df[val_df$event == 1, ]$time)
eval_times <- seq(ceiling(first_event),
                  floor(last_event),
                  by = 1)

t_horizon <- 8 # years
# Add predictions to your dataframe
val_df$baseline_pred <- predict(baseline_model,
                                type = "lp",
                                times = t_horizon,
                                newdata = val_df)
val_df$biomarker_pred <- predict(model,
                                 type = "lp",
                                 times = t_horizon,
                                 newdata = val_df)

figures <- create_publication_figures(
  baseline_model = baseline_model,
  biomarker_model = model,
  auc_summary = auc_summary[auc_summary$time > 2, ],
  brier_summary = brier_summary[brier_summary$time > 2, ],
  cal_data = cal_data_avg,
  data = val_df,
  times = eval_times
)

# Save plots
ggsave("combined_performance.pdf", figures$combined_plot,
       width = 12, height = 10, dpi = 300)
ggsave("time_dependent_auc.pdf", figures$time_dependent_auc,
       width = 6, height = 5, dpi = 300)
ggsave("decision_curve.pdf", figures$decision_curve,
       width = 6, height = 5, dpi = 300)



# additional
# Use the function
additional_figures <- create_additional_figures(
  baseline_model = baseline_model,
  biomarker_model = model,
  data = td_data,
  times = eval_times
)

# Save the new plots
ggsave("dynamic_roc.pdf", additional_figures$dynamic_roc,
       width = 8, height = 6, dpi = 300)
ggsave("prediction_error.pdf", additional_figures$prediction_error,
       width = 8, height = 6, dpi = 300)
ggsave("additional_performance.pdf", additional_figures$combined_additional,
       width = 12, height = 6, dpi = 300)