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
library(extrafont)
extrafont::loadfonts()
font_import()
loadfonts(device = "postscript")

# cut_time_data <- function(td_data, interval_years = 1.7) {
#   # Create sequence of timepoints for each ID
#   td_data %>%
#     group_by(id) %>%
#     mutate(
#       # Round start and stop times to nearest interval
#       tstart = floor(tstart / interval_years) * interval_years,
#       tstop = ceiling(tstop / interval_years) * interval_years
#     ) %>%
#     # If this creates duplicate rows, keep last observation
#     group_by(id, tstart, tstop) %>%
#     slice_tail(n = 1) %>%
#     ungroup()
# }

format_df <- function(df, lancet = FALSE, habits, psychwell, vitals) {
  df$SEX <- factor(df$SEX)
  df$APOEGN <- factor(df$APOEGN)
  df <- within(df, APOEGN <- relevel(APOEGN, ref = "E3/E3"))

  base <- df[!duplicated(df$BID), c(
    "BID", "time_to_event", "label",
    "AGEYR_centered", "AGEYR_centered_squared",
    "AGEYR_centered_cubed", "SEX",
    "EDCCNTU_z", "APOEGN"
  )]
  tv_covar <- df[, c("BID", "COLLECTION_DATE_DAYS_CONSENT", "ORRES_boxcox")]
  colnames(base) <- c(
    "id", "time", "event", "age", "age2", "age3",
    "sex", "educ", "apoe"
  )
  colnames(tv_covar) <- c("id", "time", "ptau")

  if (lancet) {
    habits <- habits[habits$BID %in% df$BID, c(
      "BID",
      "COLLECTION_DATE_DAYS_CONSENT",
      "SMOKE", "ALCOHOL", "SUBUSE",
      "AEROBIC", "WALKING"
    )]
    psychwell <- psychwell[psychwell$BID %in% df$BID, c(
      "BID",
      "COLLECTION_DATE_DAYS_CONSENT",
      "GDTOTAL", "STAITOTAL"
    )]
    vitals <- vitals[vitals$BID %in% df$BID, c(
      "BID",
      "COLLECTION_DATE_DAYS_CONSENT",
      "VSBPSYS", "VSBPDIA"
    )]
    colnames(habits) <- c(
      "id", "time", "smoke", "alcohol", "subuse",
      "aerobic", "walking"
    )
    colnames(psychwell) <- c("id", "time", "gdtotal", "staital")
    colnames(vitals) <- c("id", "time", "vsbsys", "vsdia")

    habits$time <- habits$time / 365.25
    psychwell$time <- psychwell$time / 365.25
    vitals$time <- vitals$time / 365.25
  }

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

  if (lancet) {
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
  }

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
      age = baseline_age + (tstart)
    ) %>%
    select(-baseline_age) # Remove the temporary baseline_age column

  # if (lancet) {
  #   td_data_updated <- cut_time_data(td_data_updated)
  # }

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

# Define model formulas
get_model_formula <- function(model_type, lancet = FALSE) {
  base_formulas <- list(
    "ptau_demographics" = Surv(tstart, tstop, event) ~ ptau + age + age2 +
      sex + educ + apoe + age * apoe + age2 * apoe,
    "demographics" = Surv(tstart, tstop, event) ~ age + age2 +
      sex + educ + apoe + age * apoe + age2 * apoe,
    "ptau" = Surv(tstart, tstop, event) ~ ptau,
    "demographics_no_apoe" = Surv(tstart, tstop, event) ~ age + age2 +
      sex + educ,
    "ptau_demographics_no_apoe" = Surv(tstart, tstop, event) ~ ptau +
      age + age2 +
      sex + educ
  )

  formula <- base_formulas[[model_type]]

  if (lancet) {
    formula <- update(formula, . ~ . +
                        smoke + alcohol + subuse +
                        aerobic + walking +
                        gdtotal + staital +
                        vsbsys + vsdia)
  }

  return(formula)
}

# Initialize lists to store results for all models
models_list <- list(
  "ptau_demographics" = list(),
  "ptau_demographics_lancet" = list(),
  "demographics" = list(),
  "demographics_lancet" = list(),
  "ptau" = list(),
  "demographics_no_apoe" = list(),
  "demographics_lancet_no_apoe" = list(),
  "ptau_demographics_no_apoe" = list(),
  "ptau_demographics_lancet_no_apoe" = list()
)

metrics_list <- list()
val_df_l <- list()

for (fold in seq(0, 4)) {
  print(paste0("Fold ", fold + 1))

  # Read and format data
  train_df_raw <- read_parquet(paste0(
    "../../tidy_data/A4/train_", fold, "_new.parquet"
  ))
  val_df_raw <- read_parquet(paste0(
    "../../tidy_data/A4/val_", fold, "_new.parquet"
  ))

  # Fit all models
  for (model_name in names(models_list)) {
    print(paste("Fitting model:", model_name))

    # Determine if this is a Lancet model
    is_lancet <- grepl("lancet", model_name)

    df <- format_df(train_df_raw, lancet = is_lancet,
                    habits, psychwell, vitals)
    val_df <- format_df(val_df_raw, lancet = is_lancet,
                        habits, psychwell, vitals)
    val_df_l[[paste0("fold_", fold + 1, "_", model_name)]] <- val_df

    # Z-score variables if using Lancet variables
    if (is_lancet) {
      lancet_vars <- c(
        "smoke", "alcohol", "aerobic", "walking",
        "gdtotal", "staital", "vsbsys", "vsdia"
      )
      means <- apply(df[, lancet_vars], 2, mean, na.rm = TRUE)
      sds <- apply(df[, lancet_vars], 2, sd, na.rm = TRUE)

      df[, lancet_vars] <- scale(df[, lancet_vars],
        center = means, scale = sds
      )
      val_df[, lancet_vars] <- scale(val_df[, lancet_vars],
        center = means, scale = sds
      )
    }

    # Get base model type
    base_type <- gsub("_lancet", "", model_name)

    # Get formula
    formula <- get_model_formula(base_type, is_lancet)

    # Fit model
    model <- coxph(formula, data = df, x = TRUE)
    models_list[[model_name]][[paste0("fold_", fold + 1)]] <- model

    # Calculate metrics
    eval_times <- seq(3, 8.5, by = 0.5)
    metrics_results <- calculate_survival_metrics(
      model = model,
      data = val_df,
      times = eval_times
    )
    if (!model_name %in% names(metrics_list)) {
      metrics_list[[model_name]] <- list()
    }
    metrics_list[[model_name]][[paste0("fold_", fold + 1)]] <- metrics_results
  }
}

# Save results
saveRDS(models_list, "../../tidy_data/A4/fitted_models_all.rds")
saveRDS(metrics_list, "../../tidy_data/A4/metrics_all.rds")
saveRDS(val_df_l, "../../tidy_data/A4/val_df_all.rds")

# load results
metrics_list <- readRDS("../../tidy_data/A4/metrics_all.rds")
val_df_l <- readRDS("../../tidy_data/A4/val_df_all.rds")
models_list <- readRDS("../../tidy_data/A4/fitted_models_all.rds")

# Collate metrics
collate_metric <- function(metrics_list, metric = "auc") {
  all_results <- data.frame()

  for (model_name in names(metrics_list)) {
    if (metric != "concordance") {
      # Extract metric values for each fold
      fold_values <- sapply(1:5, function(fold) {
        as.numeric(
          metrics_list[[model_name]][[paste0(
            "fold_",
            fold
          )]][[metric]][[metric]]
        )
      })
    } else {
      # Extract concordance values for each fold
      fold_values <- sapply(1:5, function(fold) {
        as.numeric(
          metrics_list[[model_name]][[paste0(
            "fold_",
            fold
          )]][[metric]][["AppCindex"]][["coxph"]]
        )
      })
    }

    # Get times for this model (assuming same across folds)
    times <- as.numeric(metrics_list[[model_name]]$fold_1[[metric]]$time)

    # Create data frame for this model
    model_df <- data.frame(
      model = model_name,
      metric = rep(fold_values),
      time = rep(times, times = 5),
      fold = rep(1:5, each = length(times))
    )

    all_results <- rbind(all_results, model_df)
  }

  return(all_results)
}

collate_metrics_all_models <- function(metrics_list, metric = "auc") {
  # Since collate_metric now handles multiple models directly,
  # we can just call it once and return the results
  results <- collate_metric(metrics_list, metric = metric)
  return(results)
}

# Generate and save results for each metric
metrics_to_collect <- c("auc", "brier", "concordance")

for (metric in metrics_to_collect) {
  results <- collate_metrics_all_models(metrics_list, metric)
  write_csv(
    results,
    paste0("../../tidy_data/A4/results_", metric, "_all_models.csv")
  )
}

# Calculate p-values comparing AUCs between models at each time point
# First combine timeROC objects from each fold for each model
biomarker_trocs <- list(
  metrics_list$ptau_demographics$fold_1$troc,
  metrics_list$ptau_demographics_lancet$fold_1$troc,
  metrics_list$demographics$fold_1$troc,
  metrics_list$demographics_lancet$fold_1$troc,
  metrics_list$ptau$fold_1$troc,
  metrics_list$demographics_no_apoe$fold_1$troc,
  metrics_list$demographics_lancet_no_apoe$fold_1$troc,
  metrics_list$ptau_demographics_no_apoe$fold_1$troc,
  metrics_list$ptau_demographics_lancet_no_apoe$fold_1$troc
)

baseline_trocs <- list(
  metrics_list$ptau_demographics$fold_1$troc,
  metrics_list$ptau_demographics_lancet$fold_1$troc,
  metrics_list$demographics$fold_1$troc,
  metrics_list$demographics_lancet$fold_1$troc,
  metrics_list$ptau$fold_1$troc,
  metrics_list$demographics_no_apoe$fold_1$troc,
  metrics_list$demographics_lancet_no_apoe$fold_1$troc,
  metrics_list$ptau_demographics_no_apoe$fold_1$troc,
  metrics_list$ptau_demographics_lancet_no_apoe$fold_1$troc
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
  all_pvalues[[fold]] <- comparison$p_values_AUC[2, ]
}

# Convert to data frame
all_pvalues_df <- do.call(rbind, all_pvalues)
hist(all_pvalues_df)
mean(all_pvalues_df)
median(all_pvalues_df)
range(all_pvalues_df)

########################################################
# AUC, Brier Score, and Concordance Over Time
auc_results <- collate_metric(metrics_list, metric = "auc")
brier_results <- collate_metric(metrics_list, metric = "brier")
concordance_results <- collate_metric(metrics_list, metric = "concordance")

# plot auc over time
auc_summary <- auc_results %>%
  group_by(model, time) %>%
  summarise(
    mean_AUC = mean(metric, na.rm = TRUE),
    sd_AUC = sd(metric, na.rm = TRUE),
    ymin = pmax(mean_AUC - sd_AUC, 0),
    ymax = pmin(mean_AUC + sd_AUC, 1),
    .groups = "drop"
  )

auc_plot <- td_plot(auc_summary %>% filter(time < 8), metric = "auc")

# Display the plot
print(auc_plot)

# Save plots
ggsave("../../tidy_data/A4/final_auc_Over_Time.pdf",
       plot = auc_plot,
       width = 8,
       height = 6,
       dpi = 300)

# Initialize lists to store ROC data
roc_data_all <- list()

# Create consistent time points
eval_times <- eval_times

# Define models to analyze
models_to_analyze <- c(
  # "demographics",
  # "demographics_no_apoe",
  "demographics_lancet",
  "ptau",
  "ptau_demographics_lancet"
)

# Initialize dataframe to store ROC curves
all_roc_curves <- data.frame()

for (fold in 0:4) {
  for (model_name in models_to_analyze) {
    # Get timeROC object from metrics list
    troc <- metrics_list[[model_name]][[paste0("fold_", fold + 1)]]$troc

    # Extract ROC curves for each time point
    for (t in eval_times) {
      # Get ROC curve data for this time
      idx <- which(troc$times == t)
      if (length(idx) > 0) {
        roc_data <- unique(data.frame(
          FPR = troc$FP[, idx],
          TPR = troc$TP[, idx],
          Time = t,
          Model = model_name,
          fold = fold
        ))
        all_roc_curves <- rbind(all_roc_curves, roc_data)
      }
    }
  }
}

# Calculate summary statistics
roc_summary <- all_roc_curves %>%
  # First, bin FPR values to create discrete groups
  mutate(FPR_bin = round(FPR, digits = 2)) %>%
  group_by(Model, Time, FPR_bin) %>%
  summarise(
    mean_TPR = mean(TPR, na.rm = TRUE),
    sd_TPR = sd(TPR, na.rm = TRUE),
    FPR = mean(FPR_bin),  # Use the mean FPR within each bin
    .groups = "drop"
  ) %>%
  # Remove any remaining NAs and ensure we have enough data points
  filter(!is.na(sd_TPR)) %>%
  group_by(Model, Time) %>%
  filter(n() >= 10) %>%  # Only keep time points with at least 10 data points
  ungroup()

model_colors <- c(
  "demographics_lancet" = "#E69F00",    # orange
  "ptau" = "#CC79A7",                   # pink
  "ptau_demographics_lancet" = "#009292" # turquoise
)

# Create faceted plot of ROC curves
roc_plot <- ggplot(roc_summary, aes(x = FPR, y = mean_TPR, color = Model)) +
  geom_ribbon(aes(
    ymin = pmax(mean_TPR - sd_TPR, 0),
    ymax = pmin(mean_TPR + sd_TPR, 1),
    fill = Model
  ), alpha = 0.2, color = NA) +
  geom_line(linewidth = 1) +
  geom_abline(
    slope = 1, intercept = 0,
    linetype = "dashed", color = "gray50"
  ) +
  scale_color_manual(values = model_colors, labels = c(
    "Demo + Lancet",
    "pTau217",
    "Demo + pTau217\n+ Lancet"
  )) +
  scale_fill_manual(values = model_colors, labels = c(
    "Demo + Lancet",
    "pTau217",
    "Demo + pTau217\n+ Lancet"
  ), guide = "none") +
  facet_wrap(~Time,
    labeller = labeller(Time = function(x) sprintf("%s years", x))
  ) +
  labs(
    x = "False Positive Rate",
    y = "True Positive Rate",
    title = "ROC Curves at Different Time Points"
  ) +
  coord_equal() +
  theme_minimal() +
  theme(
    panel.spacing = unit(1, "cm"),
    axis.text = element_text(size = 8),
    plot.margin = margin(0.5, 0.5, 0.5, 0.5, "cm"),
    plot.title = element_text(hjust = 0.5)  # Center the title
  )

# Display the plot
print(roc_plot)

# Save the plot
ggsave("../../tidy_data/A4/ROC_curves_by_timepoint.pdf",
       plot = roc_plot,
       width = 12,
       height = 8,
       dpi = 300)

### Individual year plots

# Find the year with the largest difference in AUC between the two models
dd <- auc_summary %>%
  filter(model %in% c("demographics_lancet", "ptau_demographics_lancet")) %>%
  pivot_wider(
    id_cols = time,
    names_from = model,
    values_from = mean_AUC
  ) %>%
  mutate(auc_difference = ptau_demographics_lancet - demographics_lancet) %>%
  select(time, auc_difference)

# Print the differences
print(dd)

# Find the year with the largest difference in AUC between the two models
year <- dd$time[which.max(dd$auc_difference)]

# Create individual panel for time = 7
roc_year <- roc_summary %>%
  filter(Time == year)

model_labels <- c(
  "Demo + Lancet",
  "pTau217",
  "Demo + pTau217\n+ Lancet"
)

# Create named vector for mapping model names to labels
names(model_labels) <- c(
  "demographics_lancet",
  "ptau",
  "ptau_demographics_lancet"
)

p_year <- ggplot(roc_year, aes(x = FPR, y = mean_TPR, color = Model)) +
  geom_ribbon(
    aes(
      ymin = pmax(mean_TPR - sd_TPR, 0),
      ymax = pmin(mean_TPR + sd_TPR, 1),
      fill = Model
    ),
    alpha = 0.2,
    color = NA
  ) +
  geom_line(linewidth = 2) +
  geom_abline(
    slope = 1, intercept = 0,
    linetype = "dashed", color = "gray50"
  ) +
  scale_color_manual(values = model_colors, labels = model_labels) +
  scale_fill_manual(values = model_colors, labels = model_labels) +
  labs(
    x = "False Positive Rate",
    y = "True Positive Rate",
    title = paste(
      year, "years\n",
      "Demo + Lancet: ", 
      sprintf("%.3f ± %.3f", 
        auc_summary$mean_AUC[auc_summary$time == year &
                               auc_summary$model == "demographics_lancet"],
        auc_summary$sd_AUC[auc_summary$time == year &
                               auc_summary$model == "demographics_lancet"]
      ), "\n",
      "pTau217: ",
      sprintf("%.3f ± %.3f",
        auc_summary$mean_AUC[auc_summary$time == year &
                               auc_summary$model == "ptau"],
        auc_summary$sd_AUC[auc_summary$time == year &
                               auc_summary$model == "ptau"]
      ), "\n",
      "Demo + pTau217 + Lancet: ",
      sprintf("%.3f ± %.3f",
        auc_summary$mean_AUC[auc_summary$time == year &
                               auc_summary$model == "ptau_demographics_lancet"],
        auc_summary$sd_AUC[auc_summary$time == year &
                               auc_summary$model == "ptau_demographics_lancet"]
      )
    )
  ) +
  coord_equal() +
  get_publication_theme() +
  theme(
    legend.position = "bottom",
    plot.margin = margin(0.5, 0.5, 0.5, 0.5, "cm"),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1)
  )

print(p_year)

# Save plots
ggsave(paste0("../../tidy_data/A4/final_ROCcurve_", year, "years.pdf"),
  plot = p_year,
  width = 6,
  height = 6,
  dpi = 300
)


# plot brier score over time
brier_summary <- brier_results %>%
  group_by(model, time) %>%
  summarise(
    mean_brier = mean(metric, na.rm = TRUE),
    sd_brier = sd(metric, na.rm = TRUE),
    ymin = pmax(mean_brier - sd_brier, 0),
    ymax = pmin(mean_brier + sd_brier, 1),
    .groups = "drop"
  )

# print differences between biomarker and baseline at each time point
# brier_summary %>%
#   group_by(time) %>%
#   summarise(
#     mean_diff = mean_brier[model == "Biomarker"] -
#       mean_brier[model == "Baseline"],
#     .groups = "drop"
#   ) %>%
#   print()

# ibs_results <- pec::pec(list("Biomarker" = models_list$ptau_demographics$fold_1,
#                              "Baseline" = models_list$ptau_demographics$fold_1),
#   data = val_df_l$fold_1,
#   formula = Surv(tstop, event) ~ ptau,
#   testIBS = TRUE,
#   testTimes = seq(3, 8)
# )

# ibs_results

brier_plot <- td_plot(brier_summary %>% filter(time < 8), metric = "brier", all_models = F)

# Display the plot
print(brier_plot)

# Save plots
ggsave("../../tidy_data/A4/final_brier_Over_Time.pdf",
  plot = brier_plot,
  width = 8,
  height = 6,
  dpi = 300
)

# plot concordance over time
concordance_summary <- concordance_results %>%
  group_by(model, time) %>%
  summarise(
    mean_concordance = mean(metric, na.rm = TRUE),
    sd_concordance = sd(metric, na.rm = TRUE),
    ymin = pmax(mean_concordance - sd_concordance, 0),
    ymax = pmin(mean_concordance + sd_concordance, 1),
    .groups = "drop"
  )

# print differences between biomarker and baseline at each time point
# concordance_summary %>%
#   group_by(time) %>%
#   summarise(
#     mean_diff = mean_concordance[model == "Biomarker"] -
#       mean_concordance[model == "Baseline"],
#     .groups = "drop"
#   ) %>%
#   print()

concordance_plot <- td_plot(concordance_summary %>% filter(time < 8),
                            metric = "concordance")

# Display the plot
print(concordance_plot)

# Save plots
ggsave("../../tidy_data/A4/final_concordance_Over_Time.pdf",
  plot = concordance_plot,
  width = 8,
  height = 6,
  dpi = 300
)

########################################################
# Calculate calibration data
cal_data_avg <- calculate_calibration_data(models_list, val_df_l)

# Create calibration plots
plots <- calibration_plots(cal_data_avg, seq(3, 8), model_colors)
print(plots)

# Save plots
ggsave("../../tidy_data/A4/final_calibration_plots.pdf",
  plot = plots,
  width = 8,
  height = 6,
  dpi = 300
)

########################################################
# Decision curve analysis
# Collect predictions and create DCA data
dca_data_all <- list()
models_to_analyze <- c(
  "demographics",
  "demographics_no_apoe",
  "demographics_lancet",
  "ptau",
  "ptau_demographics_lancet"
)

# Collect predictions for each time point and model
for (t in seq(3, 8)) {
  for (fold in 0:4) {
    for (model_name in models_to_analyze) {
      model <- overwrite_na_coef_to_zero(
        models_list[[model_name]][[paste0("fold_", fold + 1)]]
      )

      pred_probs <- 1 - pec::predictSurvProb(
        model,
        newdata = val_df_l[[paste0("fold_", fold + 1, "_", model_name)]],
        times = t
      )

      val_data <- val_df_l[[paste0("fold_", fold + 1, "_", model_name)]]
      dca_data_all[[paste0("t", t, "_fold",
                           fold, "_", model_name)]] <- data.frame(
        fold = fold,
        time = t,
        model = model_name,
        tstop = val_data$tstop,
        event = val_data$event,
        pred_prob = pred_probs
      )
    }
  }
}

# Combine all DCA data
all_dca_data <- do.call(rbind, dca_data_all)

# Create and save DCA plots
dca_plots <- dca_plots(all_dca_data)
print(dca_plots)

ggsave("../../tidy_data/A4/final_DCA_Over_Time.pdf",
       plot = dca_plots,
       width = 8,
       height = 6,
       dpi = 300)

# ########################################################
# # ROC and Prediction Error Curves
# # Initialize lists to store ROC data
# roc_data_all <- list()

# # Create consistent time points
# eval_times <- seq(3, 8)

# # Define models to analyze
# models_to_analyze <- c(
#   "demographics_lancet",
#   "ptau",
#   "ptau_demographics_lancet"
# )

# # Update model colors
# model_colors <- c(
#   "demographics_lancet" = "#E69F00",    # orange
#   "ptau" = "#CC79A7",                   # pink
#   "ptau_demographics_lancet" = "#009292" # turquoise
# )

# # Initialize dataframe to store ROC curves
# all_roc_curves <- data.frame()

# for (fold in 0:4) {
#   for (model_name in models_to_analyze) {
#     # Get timeROC object from metrics list
#     troc <- metrics_list[[model_name]][[paste0("fold_", fold + 1)]]$troc
    
#     # Extract ROC curves for each time point
#     for (t in eval_times) {
#       # Get ROC curve data for this time
#       idx <- which(troc$times == t)
#       if (length(idx) > 0) {
#         roc_data <- data.frame(
#           FPR = troc$FP[, idx],
#           TPR = troc$TP[, idx],
#           Time = t,
#           Model = model_name,
#           fold = fold
#         )
#         all_roc_curves <- rbind(all_roc_curves, roc_data)
#       }
#     }
#   }
# }

# # Calculate summary statistics
# roc_summary <- all_roc_curves %>%
#   # First, bin FPR values to create discrete groups
#   mutate(FPR_bin = round(FPR, digits = 2)) %>%
#   group_by(Model, Time, FPR_bin) %>%
#   summarise(
#     mean_TPR = mean(TPR, na.rm = TRUE),
#     sd_TPR = sd(TPR, na.rm = TRUE),
#     FPR = mean(FPR_bin),  # Use the mean FPR within each bin
#     .groups = "drop"
#   ) %>%
#   # Remove any remaining NAs
#   filter(!is.na(sd_TPR))

# # Plot ROC curves with smoothing and uncertainty
# p5 <- ggplot(roc_summary, aes(x = FPR, y = mean_TPR, color = Model)) +
#   geom_ribbon(aes(
#     ymin = pmax(mean_TPR - sd_TPR, 0),
#     ymax = pmin(mean_TPR + sd_TPR, 1),
#     fill = Model
#   ), alpha = 0.2, color = NA) +
#   geom_line(linewidth = 1) +
#   geom_abline(
#     slope = 1, intercept = 0,
#     linetype = "dashed", color = "gray50"
#   ) +
#   scale_color_manual(
#     values = model_colors,
#     labels = c(
#       "Demo + Lancet",
#       "pTau217",
#       "Demo + pTau217\n+ Lancet"
#     )
#   ) +
#   scale_fill_manual(
#     values = model_colors,
#     labels = c(
#       "Demo + Lancet",
#       "pTau217",
#       "Demo + pTau217\n+ Lancet"
#     )
#   ) +
#   facet_wrap(~Time,
#     labeller = labeller(Time = function(x) {
#       sprintf("%s years", x)
#     })
#   ) +
#   labs(
#     x = "False Positive Rate",
#     y = "True Positive Rate",
#     title = "Dynamic ROC Curves",
#     subtitle = "At Different Follow-up Times"
#   ) +
#   coord_equal() +
#   get_publication_theme() +
#   theme(
#     panel.spacing = unit(1, "cm"),
#     axis.text.x = element_text(angle = 0, hjust = 0.5, size = 8),
#     plot.margin = margin(0.5, 0.5, 0.5, 0.5, "cm")
#   )

# print(p5)

# # Save plots
# ggsave("../../tidy_data/A4/final_ROCcurves_Over_Time.pdf",
#        plot = p5,
#        width = 14,
#        height = 6,
#        dpi = 300)

# if (lancet) {
#   year <- 3
# } else {
#   year <- 7
# }

# # Create individual panel for time = 7
# roc_year <- roc_summary %>%
#   filter(Time == year)

# p5_year <- ggplot(roc_year, aes(x = FPR, y = mean_TPR, color = Model)) +
#   geom_ribbon(
#     aes(
#       ymin = pmax(mean_TPR - sd_TPR, 0),
#       ymax = pmin(mean_TPR + sd_TPR, 1),
#       fill = Model
#     ),
#     alpha = 0.2,
#     color = NA
#   ) +
#   geom_smooth(se = FALSE, method = "loess", span = 0.2, linewidth = 1) +
#   geom_abline(
#     slope = 1, intercept = 0,
#     linetype = "dashed", color = "gray50"
#   ) +
#   scale_color_manual(values = model_colors) +
#   scale_fill_manual(values = model_colors) +
#   labs(
#     x = "False Positive Rate",
#     y = "True Positive Rate",
#     title = paste(
#       year, "years\n",
#       auc_stats$Baseline[auc_stats$Time == year], "\n",
#       auc_stats$Biomarker[auc_stats$Time == year]
#     )
#   ) +
#   coord_equal() +
#   get_publication_theme() +
#   theme(
#     legend.position = "bottom",
#     plot.margin = margin(0.5, 0.5, 0.5, 0.5, "cm"),
#     panel.border = element_rect(color = "black", fill = NA, linewidth = 1)
#   )

# print(p5_year)

# # Save plots
# ggsave("../../tidy_data/A4/final_ROCcurve_7years.pdf",
#   plot = p5_year,
#   width = 6,
#   height = 6,
#   dpi = 300
# )

# ########################################################