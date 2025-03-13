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

source("../A4/plot_figures.R")
source("../A4/metrics.R")

for (amyloid_positive_only in c(TRUE, FALSE)) {
  load_path = "../../tidy_data/ADNI/"

  if (amyloid_positive_only) {
    load_path = paste0(load_path, "amyloid_positive/")
  } 

  # # Load fonts
  # library(extrafont)
  # extrafont::loadfonts()
  # font_import()
  # loadfonts(device = "postscript")

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

  # tmerge data for all models
  format_df <- function(df, #ptau = FALSE, lancet = FALSE, pet = FALSE,
                        medhist, neuroexam, adni_nightingale, modhach, 
                        vitals, depression, centiloids,
                        mean_lancet_vars=NULL, sd_lancet_vars=NULL) {
    df$PTGENDER <- factor(df$PTGENDER)
    df$GENOTYPE <- factor(df$GENOTYPE)
    df <- within(df, GENOTYPE <- relevel(GENOTYPE, ref = "3/3"))

    base <- df[!duplicated(df$id), c(
      "id", "time_to_event_yr", "label",
      "age_centered", "age_centered_squared",
      "age_centered_cubed", "PTGENDER", "educ_z",
      "GENOTYPE"
    )]

    colnames(base) <- c(
      "id", "time", "event",
      "age", "age2",
      "age3", "sex", "educ",
      "apoe"
    )

    # get centiloids
    centiloids <- centiloids[centiloids$RID %in% df$id, c(
      "RID", "visit_to_years", "CENTILOIDS"
    )]
    colnames(centiloids) <- c(
      "id", "time", "centiloids"
    )

    tv_covar <- df[, c("id", "visit_to_years", "ptau_boxcox")]
    colnames(tv_covar) <- c("id", "time", "ptau")

    # get medhist
    medhist <- medhist[medhist$RID %in% df$id, c(
      "RID", "visit_to_years",
      "MH3HEAD", "MH4CARD", "MH14ALCH", "MH14AALCH", 
      "MH14BALCH", "MH14CALCH", "MH16SMOK", "MH16ASMOK",
      "MH16BSMOK", "MH16CSMOK" 
    )]
    colnames(medhist) <- c(
      "id", "time", 
      "heent", "cardio", "alc_abuse", "alc_avg_drinks_day",
      "alc_abuse_years", "time_since_abuse", "smoking",
      "smoke_avg_packs_day",
      "smoking_years", "time_since_smoking" 
    )

    # get neuroexam
    neuroexm <- neuroexam[neuroexam$RID %in% df$id, c(
      "RID", "visit_to_years",
      "NXVISUAL", "NXAUDITO"
    )]
    colnames(neuroexm) <- c(
      "id", "time",
      "visual_impairment", "audit_impairment"
    )

    # adni nightingale
    adni_nightingale <- adni_nightingale[
      adni_nightingale$RID %in% df$id, c(
      "RID", "visit_to_years",
      "LDL_C"
    )]
    colnames(adni_nightingale) <- c(
      "id", "time", "ldl"
    )

    # modhach
    modhach <- modhach[modhach$RID %in% df$id, c(
      "RID", "visit_to_years",
      "HMHYPERT"
    )]
    colnames(modhach) <- c(
      "id", "time", "hypertension"
    )

    # get vitals
    vitals <- vitals[vitals$RID %in% df$id, c(
      "RID", "visit_to_years", "BMI"
    )]
    colnames(vitals) <- c(
      "id", "time", "bmi"
    )

    # get depression
    depression <- depression[depression$RID %in% df$id, c(
      "RID", "visit_to_years", "GDTOTAL"
    )]
    colnames(depression) <- c(
      "id", "time", "gdtotal"
    )
    

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

    # add medhist
    td_data <- tmerge(
      td_data,
      medhist,
      id = id,
      heent = tdc(time, heent, 0),
      cardio = tdc(time, cardio, 0),
      alc_abuse = tdc(time, alc_abuse, 0),
      alc_avg_drinks_day = tdc(time, alc_avg_drinks_day, 0),
      alc_abuse_years = tdc(time, alc_abuse_years, 0),
      time_since_abuse = tdc(time, time_since_abuse, -99),
      smoking = tdc(time, smoking, 0),
      smoke_avg_packs_day = tdc(time, smoke_avg_packs_day, 0),
      smoking_years = tdc(time, smoking_years, 0),
      time_since_smoking = tdc(time, time_since_smoking, -99)
    )

    # add neuroexam
    td_data <- tmerge(
      td_data,
      neuroexm,
      id = id,
      visual_impairment = tdc(time, visual_impairment, 0),
      audit_impairment = tdc(time, audit_impairment, 0)
    )

    # add adni nightingale
    td_data <- tmerge(
      td_data,
      adni_nightingale,
      id = id,
      ldl = tdc(time, ldl, median(adni_nightingale$ldl, na.rm = TRUE))
    )

    # add modhach
    td_data <- tmerge(
      td_data,
      modhach,
      id = id,
      hypertension = tdc(time, hypertension, 0)
    )

    # add depression
    td_data <- tmerge(
      td_data,
      depression,
      id = id,
      gdtotal = tdc(time, gdtotal, median(depression$gdtotal, na.rm = TRUE))
    )

    # add vitals
    td_data <- tmerge(
      td_data,
      vitals,
      id = id,
      bmi = tdc(time, bmi, median(vitals$bmi, na.rm = TRUE))
    )
    
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
  

    # zscore continuous lancet factors 
    continuous_lancet_vars <- c(
      'alc_avg_drinks_day', 
      'alc_abuse_years', 
      'time_since_abuse',
      'smoke_avg_packs_day',
      'smoking_years',
      'time_since_smoking',
      'ldl',
      'bmi',
      'gdtotal'
      )

    if (!is.null(mean_lancet_vars) && !is.null(sd_lancet_vars)) {
      print("Using provided mean and sd")
      } else {
      print("Calculating mean and sd")
      mean_lancet_vars <- apply(td_data[, continuous_lancet_vars],
                   2, mean, na.rm = TRUE)
      sd_lancet_vars <- apply(td_data[, continuous_lancet_vars],
                  2, sd, na.rm = TRUE)
    }

    print(mean_lancet_vars)
    print(sd_lancet_vars)
    td_data[, continuous_lancet_vars] <- scale(
      td_data[, continuous_lancet_vars],
      center = mean_lancet_vars, scale = sd_lancet_vars
    )
    
    # convert categorical variables to factors
    categorical_lancet_vars <- c( "heent", "cardio",
    "alc_abuse",  "smoking", "visual_impairment",
    "audit_impairment", "hypertension")
    td_data[, categorical_lancet_vars] <- lapply(
      td_data[, categorical_lancet_vars],
      factor
    )
      
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

    return(list(td_data, mean_lancet_vars, sd_lancet_vars))
  }

  # Define model formulas
  get_model_formula <- function(model_type, lancet = FALSE) {
    base_formulas <- list(
      "demographics_no_apoe" = Surv(tstart, tstop, event) ~ age + age2 +
        sex + educ,
      "demographics" = Surv(tstart, tstop, event) ~ age + age2 +
        sex + educ +
        apoe #+ age * apoe + age2 * apoe
        ,
      "lancet" = Surv(tstart, tstop, event) ~ 1,
      "ptau" = Surv(tstart, tstop, event) ~ ptau,
      "ptau_demographics_no_apoe" = Surv(tstart, tstop, event) ~ ptau +
        age + age2 +
        sex + educ,
      "ptau_demographics" = Surv(tstart, tstop, event) ~ ptau + age + age2 +
        sex + educ + 
        apoe #+ age * apoe + age2 * apoe
        ,
      "centiloids" = Surv(tstart, tstop, event) ~ centiloids,
      "centiloids_demographics_no_apoe" = Surv(tstart, tstop, event) ~ centiloids +
        age + age2 +
        sex + educ,
      "centiloids_demographics" = Surv(tstart, tstop, event) ~ centiloids +
        age + age2 +
        sex + educ + apoe #+ age * apoe + age2 * apoe
        ,
      "ptau_centiloids" = Surv(tstart, tstop, event) ~ ptau + centiloids,
      "ptau_centiloids_demographics_no_apoe" = Surv(tstart, tstop, event) ~ ptau + centiloids +
        age + age2 +
        sex + educ,
      "ptau_centiloids_demographics" = Surv(tstart, tstop, event) ~ ptau + centiloids +
        age + age2 +
        sex + educ + apoe #+ age * apoe + age2 * apoe
    )

    formula <- base_formulas[[model_type]]

    if (lancet) {
      formula <- update(formula, . ~ . +
                          #smoke + alcohol + subuse +
                          #aerobic + walking +
                          gdtotal #+ staital +
                          #vsbsys + vsdia
                          )
    }

    return(formula)
  }

  eval_times <- seq(3, 7)

  lancet_vars <- c(
          #"smoke", "alcohol", "aerobic", "walking",
          "gdtotal"
          #, "staital", "vsbsys", "vsdia"
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

  # read in Lancet data
  medhist <- read_parquet(paste0(load_path, "medhist.parquet"))
  neuroexam <- read_parquet(paste0(load_path, "neuroexm.parquet"))
  adni_nightingale <- read_parquet(paste0(load_path, "adni_nightingale.parquet"))
  modhach <- read_parquet(paste0(load_path, "modhach.parquet"))
  vitals <- read_parquet(paste0(load_path, "vitals.parquet"))
  centiloids <- read_parquet(paste0(load_path, "pet.parquet"))
  depression <- read_parquet(paste0(load_path, "depression.parquet"))

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

    format_train <- format_df(train_df_raw, #ptau = is_ptau, lancet = is_lancet, pet = is_pet,
                      medhist, neuroexam, adni_nightingale, modhach, 
                      vitals, depression, centiloids)
    df <- format_train[[1]]
    mean_lancet_vars <- format_train[[2]]
    sd_lancet_vars <- format_train[[3]]
    format_val <- format_df(val_df_raw, #ptau = is_ptau, lancet = is_lancet, pet = is_pet,
                      medhist, neuroexam, adni_nightingale, modhach, 
                      vitals, depression, centiloids,
                      mean_lancet_vars, sd_lancet_vars)
    val_df <- format_val[[1]]
    

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
      formula <- get_model_formula(base_type, is_lancet)

      # Fit model
      # if model gives overflow error, remove age2*apoe
      tryCatch({
        model <- coxph(formula, data = df, x = TRUE)
      }, error = function(e) {
        formula <- update(formula, . ~ . - age2 * apoe)
        model <- coxph(formula, data = df, x = TRUE)
      })

      # if model still gives overflow error, remove age*apoe
      tryCatch({
        model <- coxph(formula, data = df, x = TRUE)
      }, error = function(e) {
        formula <- update(formula, . ~ . - age * apoe)
        model <- coxph(formula, data = df, x = TRUE)
      })
      
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
  # saveRDS(models_list, paste0("../../tidy_data/A4/fitted_models.rds"))
  qs::qsave(models_list, paste0(load_path, "fitted_models.qs"))
  qs::qsave(val_df_l, paste0(load_path, "val_df_l.qs"))
  qs::qsave(train_df_l, paste0(load_path, "train_df_l.qs"))
  qs::qsave(metrics_list, paste0(load_path, "metrics.qs"))

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
  write_parquet(auc_summary, paste0(load_path, "auc_summary.parquet"))
  head(auc_summary)

  # calculate mean and SD of auc for each model and time and sort by mean auc
  print(auc_summary %>%
    group_by(model, time) %>%
    summarise(mean_auc = mean(auc), sd_auc = sd(auc)) %>%
    arrange(desc(mean_auc)),
    n=1000
  )
}

# # drop duplicates based on certain columns
# train_df_l <- qs::qread(paste0("../../tidy_data/ADNI/train_df_l.qs"))

# # keep last row for each id
# train_df_l_no_duplicates <- lapply(train_df_l, function(df) {
#   df %>%
#     group_by(id) %>%
#     slice_tail(n = 1) %>%
#     ungroup()
# })



# # count apoe genotypes in cases and controls, normalize by total number of cases or controls
# train_df_l_no_duplicates$fold_1 %>% 
#   group_by(event) %>%
#   count(apoe) %>%
#   mutate(apoe_freq = n / sum(n))


# length(unique(train_df_raw$id))
# length(unique(val_df_raw$id))
# length(unique(centiloids$id))
