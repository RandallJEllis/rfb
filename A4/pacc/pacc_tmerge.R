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

rsq <- function (x, y) cor(x, y) ^ 2
calculate_adj_r2 <- function(r_squared, n, p) {
  # n = number of observations
  # p = number of predictors (excluding intercept)
  return(1 - (1 - r_squared) * ((n - 1) / (n - p - 1)))
}

for (amyloid_positive_only in c(TRUE,
                                FALSE)) {
  load_path = "../../tidy_data/A4/"
  lancet_load_path = "../../tidy_data/A4/"
  if (amyloid_positive_only) {
    load_path = paste0(load_path, "amyloid_positive/")
  } 
  print(paste('Amyloid positive only: ', amyloid_positive_only))
  # tmerge data for all models
  format_df <- function(df, #ptau = FALSE, lancet = FALSE, pet = FALSE,
                        habits, psychwell, vitals, centiloids, pacc) {
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

    tv_pacc <- pacc[pacc$BID %in% df$id, c(
      "BID", "COLLECTION_DATE_DAYS_CONSENT", "PACC.raw", "PACC"
    )]
    colnames(tv_pacc) <- c("id", "time", "PACC.raw", "PACC")

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
    tv_pacc$time <- tv_pacc$time / 365.25
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

    td_data <- tmerge(
      td_data,
      tv_pacc,
      id = id,
      PACC.raw = tdc(time, PACC.raw),
      PACC = tdc(time, PACC)
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
  pacc <- read_parquet(paste0(load_path, "pacc.parquet"))
  habits <- read_parquet(paste0(lancet_load_path, "habits.parquet"))
  habits$SUBUSE <- as.factor(habits$SUBUSE)
  psychwell <- read_parquet(paste0(lancet_load_path, "psychwell.parquet"))
  vitals <- read_parquet(paste0(lancet_load_path, "vitals.parquet"))
  centiloids <- read_parquet(paste0(lancet_load_path, "centiloids.parquet"))

  # Define model formulas
  get_model_formula <- function(model_type, pacc_col, lancet = FALSE) {
    base_formulas <- list(
        "demographics_no_apoe" = as.formula(paste0(pacc_col, ' ~ tstop + age + age2 +
        sex + educ')),
        "demographics" = as.formula(paste0(pacc_col, ' ~ tstop + age + age2 +
        sex + educ +
        apoe + age * apoe + age2 * apoe')),
        "lancet" = as.formula(paste0(pacc_col, ' ~ tstop')),
        "ptau" = as.formula(paste0(pacc_col, ' ~ tstop + ptau')),
        "ptau_demographics_no_apoe" = as.formula(paste0(pacc_col, ' ~ tstop + ptau +
        age + age2 +
        sex + educ')),
        "ptau_demographics" = as.formula(paste0(pacc_col, ' ~ tstop + ptau + age + age2 +
        sex + educ + 
        apoe + age * apoe + age2 * apoe')),
        "centiloids" = as.formula(paste0(pacc_col, ' ~ tstop + centiloids')),
        "centiloids_demographics_no_apoe" = as.formula(paste0(pacc_col, ' ~ tstop + centiloids +
        age + age2 +
        sex + educ')),
        "centiloids_demographics" = as.formula(paste0(pacc_col, ' ~ tstop + centiloids +
        age + age2 +
        sex + educ + apoe + age * apoe + age2 * apoe')),
        "ptau_centiloids" = as.formula(paste0(pacc_col, ' ~ tstop + ptau + centiloids')),
        "ptau_centiloids_demographics_no_apoe" = as.formula(paste0(pacc_col, ' ~ tstop + ptau + centiloids +
        age + age2 +
        sex + educ')),
        "ptau_centiloids_demographics" = as.formula(paste0(pacc_col, ' ~ tstop + ptau + centiloids +
        age + age2 +
        sex + educ + apoe + age * apoe + age2 * apoe'))
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

#   eval_times <- seq(3, 7)

  lancet_vars <- c(
          "smoke", "alcohol", "aerobic", "walking",
          "gdtotal", "staital", "vsbsys", "vsdia",
          "bmi"
        )

  get_metrics <- function(train_df, val_df, model){
    train_predictions <- predict(model, train_df)
    val_predictions <- predict(model, val_df)

    # Calculate R-squared 
    train_rsquared <- rsq(train_predictions, train_df$PACC.raw)
    train_adj_r2 <- calculate_adj_r2(train_rsquared, nrow(train_df), length(model$coefficients) - 1)
    val_rsquared <- rsq(val_predictions, val_df$PACC.raw)
    val_adj_r2 <- calculate_adj_r2(val_rsquared, nrow(val_df), length(model$coefficients) - 1)

    # ptau_demo_train_rsquared_l <- c(ptau_demo_train_rsquared_l, train_adj_r2)
    # ptau_demo_val_rsquared_l <- c(ptau_demo_val_rsquared_l, val_adj_r2)

    # calculate RMSE
    train_rmse <- sqrt(mean((train_predictions - train_df$PACC.raw)^2))
    val_rmse <- sqrt(mean((val_predictions - val_df$PACC.raw)^2))

    # ptau_demo_train_rmse_l <- c(ptau_demo_train_rmse_l, train_rmse)
    # ptau_demo_val_rmse_l <- c(ptau_demo_val_rmse_l, val_rmse)

    return(list(train_rsquared = train_rsquared, val_rsquared = val_rsquared, train_adj_r2 = train_adj_r2, val_adj_r2 = val_adj_r2, train_rmse = train_rmse, val_rmse = val_rmse))
  }
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

  for (pacc_col in c('PACC.raw', 'PACC')) {
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
                        habits, psychwell, vitals, centiloids, pacc)
        val_df <- format_df(val_df_raw, #ptau = is_ptau, lancet = is_lancet,
                            habits, psychwell, vitals, centiloids, pacc)

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

            # Get base model type
            base_type <- gsub("_lancet", "", model_name)

            # Get formula
            formula <- get_model_formula(base_type, pacc_col, is_lancet)

            # model <- lm(formula, data = df)
            # df$id <- factor(df$id, levels = levels(df$id))
            # val_df$id <- factor(val_df$id)

            model <- bam(formula, data = df)

            df$predicted <- predict(model, newdata = df, na.action = na.pass)
            val_df$predicted <- predict(model, newdata = val_df, na.action = na.pass)

            train_rsquared <- yardstick::rsq_vec(df$PACC.raw, df$predicted)
            val_rsquared <- yardstick::rsq_vec(val_df$PACC.raw, val_df$predicted)

            train_adj_rsquared <- calculate_adj_r2(train_rsquared, length(df$PACC.raw), sum(summary(model)$edf))
            val_adj_rsquared <- calculate_adj_r2(val_rsquared, length(val_df$PACC.raw), sum(summary(model)$edf))

            train_rmse <- yardstick::rmse_vec(df$PACC.raw, df$predicted)
            val_rmse <- yardstick::rmse_vec(val_df$PACC.raw, val_df$predicted)

            # metrics <- get_metrics(df, val_df, model)

            # Save model
            models_list[[model_name]][[paste0("fold_", fold + 1)]] <- model

            # Save metrics
            metrics_list[[model_name]][[paste0("fold_", fold + 1)]] <- metrics
        }
    }

    }

    # # t-test
    # t_test <- t.test(ptau_demo_val_rsquared_l, demo_val_rsquared_l, paired = TRUE)
    # p_value <- t_test$p.value
    # print(paste0('p-value: ', p_value))

    # combine results into a dataframe
    results <- data.frame()
    for (model_name in names(models_list)) {
        # model_results <- data.frame()
        for (fold in seq(0,4)) {
            results <- rbind(results, data.frame(
                model_name = model_name,
                pacc_col = pacc_col,
                fold = fold,
                train_rsquared = metrics_list[[model_name]][[paste0("fold_", fold + 1)]]$train_rsquared,
                val_rsquared = metrics_list[[model_name]][[paste0("fold_", fold + 1)]]$val_rsquared,
                train_rmse = metrics_list[[model_name]][[paste0("fold_", fold + 1)]]$train_rmse,
                val_rmse = metrics_list[[model_name]][[paste0("fold_", fold + 1)]]$val_rmse
            ))
        }
    
    
        agg_results <- results %>%
            group_by(model_name) %>%
            summarise(train_mean_rsquared = mean(train_rsquared),
                    train_sd_rsquared = sd(train_rsquared),
                    train_ci_lower_rsquared = mean(train_rsquared) - qt(0.975, 4) * sd(train_rsquared) / sqrt(5),
                    train_ci_upper_rsquared = mean(train_rsquared) + qt(0.975, 4) * sd(train_rsquared) / sqrt(5),
                    val_mean_rsquared = mean(val_rsquared),
                    val_sd_rsquared = sd(val_rsquared),
                    val_ci_lower_rsquared = mean(val_rsquared) - qt(0.975, 4) * sd(val_rsquared) / sqrt(5),
                    val_ci_upper_rsquared = mean(val_rsquared) + qt(0.975, 4) * sd(val_rsquared) / sqrt(5),
                    train_mean_rmse = mean(train_rmse),
                    train_sd_rmse = sd(train_rmse),
                    train_ci_lower_rmse = mean(train_rmse) - qt(0.975, 4) * sd(train_rmse) / sqrt(5),
                    train_ci_upper_rmse = mean(train_rmse) + qt(0.975, 4) * sd(train_rmse) / sqrt(5),
                    val_mean_rmse = mean(val_rmse),
                    val_sd_rmse = sd(val_rmse),
                    val_ci_lower_rmse = mean(val_rmse) - qt(0.975, 4) * sd(val_rmse) / sqrt(5),
                    val_ci_upper_rmse = mean(val_rmse) + qt(0.975, 4) * sd(val_rmse) / sqrt(5)
                )
            
        }
    
    

    # save results
    write_parquet(results, paste0('../../results/A4/PACC/tmerge_model/tmerge_results_', pacc_col, '.parquet'))
    print(paste0('Saved results for ', pacc_col))

    write_parquet(agg_results, paste0('../../results/A4/PACC/tmerge_model/tmerge_agg_results_', pacc_col, '.parquet'))
    print(paste0('Saved aggregated results for ', pacc_col))

    # save metrics
    qs::qsave(metrics_list, paste0('../../results/A4/PACC/tmerge_model/tmerge_metrics_', pacc_col, '.qs'))
    print(paste0('Saved metrics for ', pacc_col))

    # save models
    qs::qsave(models_list, paste0('../../results/A4/PACC/tmerge_model/tmerge_models_', pacc_col, '.qs'))
    print(paste0('Saved models for ', pacc_col))
}


library(ggplot2)

ggplot(df, aes(x = tstop, y = predicted, group = id)) +
  geom_line(alpha = 0.3) +  # All subjects
  theme_minimal() +
  labs(title = "Subject-Level Predicted Trajectories",
       x = "Time", y = "Predicted PACC Score")




