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
library(riskRegression)
library(this.path)

setwd(dirname(this.path()))

source("plot_figures.R")
source("metrics.R")

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

for (amyloid_positive_only in c(#TRUE, 
                                FALSE)) {
  load_path = "../../tidy_data/A4/"
  lancet_load_path = "../../tidy_data/A4/"
  if (amyloid_positive_only) {
    load_path = paste0(load_path, "amyloid_positive/")
  } 
  print(paste('Amyloid positive only: ', amyloid_positive_only))
  # tmerge data for all models
  format_df <- function(df, #ptau = FALSE, lancet = FALSE, pet = FALSE,
                        habits, psychwell, vitals, centiloids) {
    df$SEX <- factor(df$SEX)
    df$APOEGN <- factor(df$APOEGN)
    df <- within(df, APOEGN <- relevel(APOEGN, ref = "E3/E3"))

    base <- df[!duplicated(df$id), c(
      "id", "time_to_event", "label",
      "age_centered", "age_centered_squared",
      "age_centered_cubed", "SEX", "educ_z",
      "APOEGN"
    )]

    colnames(base) <- c(
      "id", "time", "event",
      "age", "age2",
      "age3", "sex", "educ",
      "apoe"
    )

    centiloids <- centiloids[centiloids$BID %in% df$id, c(
      "BID", "COLLECTION_YEARS_CONSENT", "AMYLCENT"
    )]
    colnames(centiloids) <- c(
      "id", "time", "centiloids"
    )

    tv_covar <- df[, c("id", "visit_to_days", "ptau_boxcox")]
    colnames(tv_covar) <- c("id", "time", "ptau")

    # if (lancet) {
    habits <- habits[habits$BID %in% df$id, c(
      "BID",
      "COLLECTION_DATE_DAYS_CONSENT",
      "SMOKE", "ALCOHOL", "SUBUSE",
      "AEROBIC", "WALKING"
    )]
    psychwell <- psychwell[psychwell$BID %in% df$id, c(
      "BID",
      "COLLECTION_DATE_DAYS_CONSENT",
      "GDTOTAL", "STAITOTAL"
    )]
    vitals <- vitals[vitals$BID %in% df$id, c(
      "BID",
      "COLLECTION_DATE_DAYS_CONSENT",
      "VSBPSYS", "VSBPDIA", "BMI"
    )]
    colnames(habits) <- c(
      "id", "time", "smoke", "alcohol", "subuse",
      "aerobic", "walking"
    )
    colnames(psychwell) <- c("id", "time", "gdtotal", "staital")
    colnames(vitals) <- c("id", "time", "vsbsys", "vsdia", "bmi")

    habits$time <- habits$time / 365.25
    psychwell$time <- psychwell$time / 365.25
    vitals$time <- vitals$time / 365.25
    # }

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

    # if (ptau) {
      # Add the ptau column
    td_data <- tmerge(
      td_data,
      tv_covar,
      id = id,
      ptau = tdc(time, ptau)
    )
    # }

    # if (pet) {
      # Add the centiloids column
    td_data <- tmerge(
      td_data,
      centiloids,
      id = id,
      centiloids = tdc(time, centiloids)
    )
    # }

    # if (lancet) {
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
      vsdia = tdc(time, vsdia),
      bmi = tdc(time, bmi)
    )
    # }

    # td_data <- td_data[complete.cases(td_data), ]

    # First, let's store the baseline age for each person
    baseline_ages <- td_data %>%
      group_by(id) %>%
      slice_min(tstart) %>%
      select(id, baseline_age = age)

    # Now update the age column to reflect actual age at each timepoint
    td_data <- td_data %>%
      left_join(baseline_ages, by = "id") %>%
      mutate(
        # Convert tstart from days to years and add to baseline age
        age = baseline_age + (tstart)
      ) %>%
      select(-baseline_age) # Remove the temporary baseline_age column


    td_data <- td_data[order(td_data$id, td_data$tstart), ]
    # Perform last observation carried forward (LOCF) within each subject
    td_data <- td_data %>%
      group_by(id) %>%
      fill(everything(), .direction = "down") %>%
      # Also carry first value backward for any remaining NAs
      fill(everything(), .direction = "up") %>%
      ungroup()

    # print(dim(td_data))
    td_data <- td_data[complete.cases(td_data), ]
    # print(dim(td_data))
    # update age2
    td_data$age2 <- td_data$age^2

    # update age3
    td_data$age3 <- td_data$age^3

    # if (lancet) {
    #   td_data_updated <- cut_time_data(td_data_updated)
    # }

    return(td_data)
  }

  # read in Lancet data
  habits <- read_parquet(paste0(lancet_load_path, "habits.parquet"))
  habits$SUBUSE <- as.factor(habits$SUBUSE)
  psychwell <- read_parquet(paste0(lancet_load_path, "psychwell.parquet"))
  vitals <- read_parquet(paste0(lancet_load_path, "vitals.parquet"))
  centiloids <- read_parquet(paste0(lancet_load_path, "centiloids.parquet"))

  # Define model formulas
  get_model_formula <- function(model_type, lancet = FALSE) {
    base_formulas <- list(
      "demographics_no_apoe" = Surv(tstart, tstop, event) ~ age + age2 +
        sex + educ,
      "demographics" = Surv(tstart, tstop, event) ~ age + age2 +
        sex + educ +
        apoe + age * apoe + age2 * apoe,
      "lancet" = Surv(tstart, tstop, event) ~ 1,
      "ptau" = Surv(tstart, tstop, event) ~ ptau,
      "ptau_demographics_no_apoe" = Surv(tstart, tstop, event) ~ ptau +
        age + age2 +
        sex + educ,
      "ptau_demographics" = Surv(tstart, tstop, event) ~ ptau + age + age2 +
        sex + educ + 
        apoe + age * apoe + age2 * apoe,
      "centiloids" = Surv(tstart, tstop, event) ~ centiloids,
      "centiloids_demographics_no_apoe" = Surv(tstart, tstop, event) ~ centiloids +
        age + age2 +
        sex + educ,
      "centiloids_demographics" = Surv(tstart, tstop, event) ~ centiloids +
        age + age2 +
        sex + educ + apoe + age * apoe + age2 * apoe,
      "ptau_centiloids" = Surv(tstart, tstop, event) ~ ptau + centiloids,
      "ptau_centiloids_demographics_no_apoe" = Surv(tstart, tstop, event) ~ ptau + centiloids +
        age + age2 +
        sex + educ,
      "ptau_centiloids_demographics" = Surv(tstart, tstop, event) ~ ptau + centiloids +
        age + age2 +
        sex + educ + apoe + age * apoe + age2 * apoe
    )

    formula <- base_formulas[[model_type]]

    if (lancet) {
      formula <- update(formula, . ~ . +
                          smoke + alcohol + subuse +
                          aerobic + walking +
                          gdtotal + staital +
                          vsbsys + vsdia + bmi)
    }

    return(formula)
  }

  eval_times <- seq(3, 7)

  lancet_vars <- c(
          "smoke", "alcohol", "aerobic", "walking",
          "gdtotal", "staital", "vsbsys", "vsdia",
          "bmi"
        )

  # Initialize lists to store results for all models
  models_list <- list(
    "demographics_no_apoe" = list(),
    "demographics" = list(),
    "demographics_lancet_no_apoe" = list(),
    "demographics_lancet" = list(),
    "lancet" = list(),
    "ptau" = list(),
    "ptau_demographics_no_apoe" = list(),
    "ptau_demographics" = list(),
    "ptau_demographics_lancet_no_apoe" = list(),
    "ptau_demographics_lancet" = list(),
    "centiloids" = list(),
    "centiloids_demographics_no_apoe" = list(),
    "centiloids_demographics" = list(),
    "centiloids_demographics_lancet_no_apoe" = list(),
    "centiloids_demographics_lancet" = list(),
    "ptau_centiloids" = list(),
    "ptau_centiloids_demographics_no_apoe" = list(),
    "ptau_centiloids_demographics" = list(),
    "ptau_centiloids_demographics_lancet_no_apoe" = list(),
    "ptau_centiloids_demographics_lancet" = list()
  )

  val_df_l <- list()
  train_df_l <- list()

  # Initialize lists to store results for all models
  metrics_list <- list()

  # iterate over folds and run experiments
  for (fold in seq(0, 4)) {
    print(paste0("Fold ", fold + 1))

    # Read and format data
    train_df_raw <- read_parquet(paste0(
      load_path, "train_", fold, ".parquet"
    ))
    val_df_raw <- read_parquet(paste0(
      load_path, "val_", fold, ".parquet"
    ))

    df <- format_df(train_df_raw, #ptau = is_ptau, lancet = is_lancet, pet = is_pet,
                      habits, psychwell, vitals, centiloids)
    val_df <- format_df(val_df_raw, #ptau = is_ptau, lancet = is_lancet,
                          habits, psychwell, vitals, centiloids)

    # scale lancet variables
    means <- apply(df[, lancet_vars], 2, mean, na.rm = TRUE)
    sds <- apply(df[, lancet_vars], 2, sd, na.rm = TRUE)

    df[, lancet_vars] <- scale(df[, lancet_vars],
      center = means, scale = sds
    )
    val_df[, lancet_vars] <- scale(val_df[, lancet_vars],
      center = means, scale = sds
    )
    
    train_df_l[[paste0("fold_", fold + 1)]] <- df
    val_df_l[[paste0("fold_", fold + 1)]] <- val_df
    
    # Fit all models
    for (model_name in names(models_list)) {
      print(paste("Fitting model:", model_name))

      # Determine if this is a Lancet model
      is_lancet <- grepl("lancet", model_name)
      is_ptau <- grepl("ptau", model_name)
      is_pet <- grepl("centiloids", model_name)

      # print number of unique ids in df and val_df
      # print(paste0("Number of unique ids in df: ", length(unique(df$id))))
      # print(paste0("Number of unique ids in val_df: ", length(unique(val_df$id))))
      # print(fold)
      # print(model_name)
      # print(dim(df))
      # print(dim(val_df))
      # val_df_l[[paste0("fold_", fold + 1, "_", model_name)]] <- val_df
      # train_df_l[[paste0("fold_", fold + 1, "_", model_name)]] <- df
      # Z-score variables if using Lancet variables
      # if (is_lancet) {
      # }

      # Get base model type
      base_type <- gsub("_lancet", "", model_name)

      # Get formula
      formula <- get_model_formula(base_type, is_lancet)

      # Fit model
      model <- coxph(formula, data = df, id=id, x = TRUE)
      gc()
      models_list[[model_name]][[paste0("fold_", fold + 1)]] <- model

      # Calculate metrics
      metrics_results <- calculate_survival_metrics(
        model = model,
        model_name = model_name,
        data = val_df,
        times = eval_times
      )
      if (!model_name %in% names(metrics_list)) {
        metrics_list[[model_name]] <- list()
      }
      gc()
      metrics_list[[model_name]][[paste0("fold_", fold + 1)]] <- metrics_results
      gc()
    }
  }

  # Save results
  qs::qsave(models_list, paste0(load_path, "fitted_models_id.qs"))
  qs::qsave(val_df_l, paste0(load_path, "val_df_l_id.qs"))
  qs::qsave(train_df_l, paste0(load_path, "train_df_l_id.qs"))
  qs::qsave(metrics_list, paste0(load_path, "metrics_id.qs"))

  get_auc_ci_all_folds <- function(metrics_list, summarize = FALSE) {
    # Initialize empty dataframe for results
    all_results <- data.frame()

    # Loop through each model
    for (model_name in names(metrics_list)) {
      # Loop through each fold
      fold_results <- lapply(1:5, function(fold) {
        troc <- metrics_list[[model_name]][[paste0("fold_", fold)]]$troc
        ci <- timeROC:::confint.ipcwsurvivalROC(troc)

        data.frame(
          model = model_name,
          time = troc$times,
          auc = troc$AUC,
          ci_lower = ci$CI_AUC[, 1] / 100,
          ci_upper = ci$CI_AUC[, 2] / 100,
          fold = fold
        )
      })

      # Combine results from all folds
      model_results <- do.call(rbind, fold_results)

      if (summarize) {
        # Calculate mean values across folds for each time point
        summary_stats <- aggregate(
          cbind(auc, ci_lower, ci_upper) ~ model + time,
          data = model_results,
          FUN = mean
        )
      } else {
        summary_stats <- model_results
      }

      all_results <- rbind(all_results, summary_stats)
    }

    # Sort results by model and time
    all_results <- all_results[order(all_results$model, all_results$time), ]

    return(all_results)
  }

  auc_summary <- get_auc_ci_all_folds(metrics_list)
  write_parquet(auc_summary, paste0(load_path, "auc_summary_id.parquet"))
}
