library(riskRegression)

overwrite_na_coef_to_zero <- function(model) {
  if (length(names(which(is.na(coef(model))))) > 0) {
    # set coefficients to zero
    for (n in names(which(is.na(coef(model))))) {
      model$coefficients[[n]] <- 0
    }
  }
  return(model)
}

# Function to calculate Brier score at a specific time point
calculate_brier_at_time <- function(model, data, t) {
  # Create prediction dataset up to time t
  pred_data <- data %>%
    filter(tstart <= t) %>%
    group_by(id) %>%
    filter(row_number() == n()) %>%  # Get last record for each subject
    ungroup()
  # Print debugging info
  # cat("\nTime point:", t)
  # cat("\nNumber of subjects:", nrow(pred_data))

  # Calculate inverse probability of censoring weights
  cens_model <- survfit(Surv(tstart, tstop, 1 - event) ~ 1, data = data)
  cens_probs <- summary(cens_model, times = t)$surv[1] # Take first (and should be only) probability

  if (is.na(cens_probs) || cens_probs <= 0) {
    warning(sprintf("Invalid censoring probability at time %f", t))
    return(NA)
  }

  # Get weights - same weight for all subjects at time t
  weights <- rep(1 / cens_probs, nrow(pred_data))
  # cat("\nNumber of weights:", length(weights))

  gc()
  # Get survival predictions at time t
  surv_probs <- try({
    # Use predictSurvProb from pec package which handles time-varying covariates
    1 - pec::predictSurvProb(model, newdata = pred_data, times = t)
  })
  gc()

  if (inherits(surv_probs, "try-error")) {
    warning(sprintf("Error getting survival predictions at time %f", t))
    return(NA)
  }

  # cat("\nLength of survival probabilities:", length(surv_probs))

  if (length(surv_probs) != nrow(pred_data)) {
    warning(sprintf(
      "Mismatch in prediction lengths at time %f: surv_probs=%d, pred_data=%d",
      t, length(surv_probs), nrow(pred_data)
    ))
    return(NA)
  }

  # Calculate observed status at time t
  observed_status <- ifelse(pred_data$tstop <= t & pred_data$event == 1, 1, 0)
  # cat("\nLength of observed status:", length(observed_status))

  # Verify all vectors have same length
  if (length(observed_status) != length(surv_probs) ||
        length(observed_status) != length(weights)) {
    warning(sprintf(
      "Vector length mismatch at time %f: obs=%d, surv=%d, weights=%d",
      t, length(observed_status), length(surv_probs), length(weights)
    ))
    return(NA)
  }
  # Calculate weighted Brier score
  brier_score <- weighted.mean(
    (observed_status - surv_probs)^2,
    w = weights,
    na.rm = TRUE
  )

  # cat("\nCalculated Brier score:", brier_score, "\n")
  return(brier_score)
}

# Function to calculate Brier scores across multiple time points
calculate_brier_scores <- function(model, data, times) {
  # Calculate Brier score for each time point
  brier_scores <- sapply(times, function(t) {
    tryCatch({
      calculate_brier_at_time(model, data, t)
    }, error = function(e) {
      warning(sprintf("Error calculating Brier score at time %f: %s", t,
                      e$message))
      return(NA)
    })
  })

  # Calculate integrated Brier score using trapezoidal rule
  # Only use non-NA values
  valid_scores <- !is.na(brier_scores)
  if (sum(valid_scores) >= 2) {
    ibs <- pracma::trapz(times[valid_scores], brier_scores[valid_scores]) /
      (max(times[valid_scores]) - min(times[valid_scores]))
  } else {
    ibs <- NA
    warning("Not enough valid Brier scores to calculate IBS")
  }

  # Create results dataframe
  results <- list(
    scores = data.frame(
      time = times,
      brier = brier_scores
    ),
    ibs = ibs
  )
  
  return(results)
}

# Function to calculate multiple metrics at specific time points
calculate_survival_metrics <- function(model, model_name, data, times) {
  # Initialize results list
  results <- list()

  # Print diagnostic information about IDs
  # print(paste("Model:", model_name))
  # print(paste("Number of unique IDs:", length(unique(data$id))))
  # print("First few IDs:")
  # print(head(data$id))

  # Print information about missing values
  # print("Number of rows with missing values:")
  # print(colSums(is.na(data)))

  # Print information about time-varying covariates
  # print("Number of measurements per ID:")
  # id_counts <- table(data$id)
  # print(summary(id_counts))

  # Sort data consistently by id and time
  data <- data[order(data$id, data$tstop), ]

  # Get linear predictor (which doesn't vary with time)
  risk_scores <- predict(model, newdata = data, type="lp")

  # print("Number of risk scores:")
  # print(length(risk_scores))
  # print("Number of NA risk scores:")
  # print(sum(is.na(risk_scores)))

  # 1. Time-dependent AUC using timeROC
  troc <- timeROC(
    T = data$tstop,
    delta = data$event,
    marker = risk_scores,  # using linear predictor instead of risk
    cause = 1,
    times = times,
    weighting = "marginal",
    iid = TRUE
  )
  gc()
  auc_results <- data.frame(
    time = times,
    auc = troc$AUC
  )

  results$auc <- auc_results
  results$troc <- troc

  # 2. Brier Scores and IBS - using our existing function
  brier_results <- calculate_brier_scores(model, data, times)
  results$brier <- brier_results$scores
  results$ibs <- brier_results$ibs

  # 3. Concordance Index (Harrell's C)
  model <- overwrite_na_coef_to_zero(model)
  c_index <- pec::cindex(model,
                         formula = Surv(time = tstop, event = event) ~ 1,
                         data = data,
                         eval.times = times,
                         pred.times = times,
                         confInt = TRUE,
                         confLevel = 0.95)
  results$concordance <- c_index

  return(results)
}

compare_tvaurocs <- function(trocs_x, trocs_y) {
  # Initialize list to store results
  all_results <- list()
  
  # Loop through each fold
  for (fold in seq_along(trocs_x)) {
    
    # Compare timeROC objects using timeROC::compare
    comparison <- timeROC::compare(trocs_x[[fold]],
                                   trocs_y[[fold]],
                                   adjusted = TRUE
    )
    
    # Store results for this fold
    all_results[[fold]] <- data.frame(
      fold = fold,
      time = trocs_x[[fold]]$times,
      auc_x = trocs_x[[fold]]$AUC,
      auc_y = trocs_y[[fold]]$AUC,
      auc_diff = trocs_y[[fold]]$AUC - trocs_x[[fold]]$AUC,
      p_value = comparison$p_values_AUC[2, ]
    )
  }
  
  # Combine results from all folds
  all_results_df <- do.call(rbind, all_results)
  
  # Calculate summary statistics
  summary_stats <- aggregate(
    cbind(auc_diff, p_value, auc_x, auc_y) ~ time,
    data = all_results_df,
    FUN = function(x) c(mean = mean(x), sd = sd(x))
  )
  
  # Calculate confidence intervals for each fold
  ci_data <- data.frame()
  for (fold in seq_along(trocs_x)) {
    ci_x <- timeROC:::confint.ipcwsurvivalROC(trocs_x[[fold]])
    ci_y <- timeROC:::confint.ipcwsurvivalROC(trocs_y[[fold]])
    
    ci_data <- rbind(ci_data, data.frame(
      fold = fold,
      time = trocs_x[[fold]]$times,
      ci_lower_x = ci_x$CI_AUC[, 1] / 100,
      ci_upper_x = ci_x$CI_AUC[, 2] / 100,
      ci_lower_y = ci_y$CI_AUC[, 1] / 100,
      ci_upper_y = ci_y$CI_AUC[, 2] / 100
    ))
  }
  
  # Calculate mean CIs across folds
  ci_summary <- aggregate(
    cbind(ci_lower_x, ci_upper_x, ci_lower_y, ci_upper_y) ~ time,
    data = ci_data,
    FUN = mean
  )
  
  # Merge with existing summary stats
  final_summary <- merge(summary_stats, ci_summary, by = "time")

  # print out all p-values
  print("Summary of AUC differences and p-values by time point:")
  print(range(pvals_compare_trocs$all_results$p_value))
  print(mean(pvals_compare_trocs$all_results$p_value))
  print(sd(pvals_compare_trocs$all_results$p_value))
  print(median(pvals_compare_trocs$all_results$p_value))

  # how many p-values are less than 0.05? out of how many total p-values?
  sum(pvals_compare_trocs$all_results$p_value < 0.05) /
    length(pvals_compare_trocs$all_results$p_value)
  
  return(list(
    all_results = all_results_df,
    summary = final_summary
  ))
}

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

calculate_SeSpPPVNPV <- function(model, val_data, times) {
  risk_scores <- predict(model, val_data)
  
  # Find optimal cutpoint using Youden's index
  se_sp_ppv_npv <- list()
  for (cutpoint in seq(min(risk_scores), max(risk_scores), length.out = 100)) {
    se_sp_ppv_npv_results <- SeSpPPVNPV(
      cutpoint = cutpoint,
      T = val_data$time,
      delta = val_data$event,
      marker = risk_scores,
      cause = 1,
      weighting = "marginal",
      times = times
    )
    se_sp_ppv_npv[[paste0("cutpoint_", cutpoint)]] <- se_sp_ppv_npv_results
  }
  
  youden_index_list <- list()
  for (cutpoint in names(se_sp_ppv_npv)) {
    youden_index <- se_sp_ppv_npv[[cutpoint]]$TP + (1 - se_sp_ppv_npv[[cutpoint]]$FP) - 1
    youden_index_list[[cutpoint]] <- mean(youden_index)
  }
  
  best_cutpoint <- names(youden_index_list)[which.max(youden_index_list)]
  return(se_sp_ppv_npv[[best_cutpoint]])
}


SpSpPPVNPV_summary <- function(models_list, model_names, val_df_l) {
  metrics_over_time <- list()

  # Calculate metrics for each model and fold
  for (model_name in model_names) {
    metrics_over_time[[model_name]] <- list()
    for (fold in 1:5) {
      model <- models_list[[model_name]][[paste0("fold_", fold)]]
      val_data <- val_df_l[[paste0("fold_", fold)]]
      
      metrics <- calculate_SeSpPPVNPV(model, val_data, times = seq(3, 7))
      
      # Store results in a data frame
      metrics_df <- data.frame(
        time = metrics$times,
        sensitivity = metrics$TP,
        specificity = 1 - metrics$FP,
        ppv = metrics$PPV,
        npv = metrics$NPV,
        model = model_name,
        fold = fold
      )
      
      metrics_over_time[[model_name]][[fold]] <- metrics_df
    }
  }

  # Combine all results into a single data frame
  all_metrics <- do.call(rbind, lapply(metrics_over_time, function(x) do.call(rbind, x)))

  # Calculate mean and standard error for each metric
  metrics_summary <- all_metrics %>%
    group_by(model, time) %>%
    summarise(
      mean_sensitivity = mean(sensitivity, na.rm = TRUE),
      sd_sensitivity = sd(sensitivity, na.rm = TRUE) / sqrt(n()),
      mean_specificity = mean(specificity, na.rm = TRUE),
      sd_specificity = sd(specificity, na.rm = TRUE) / sqrt(n()),
      mean_ppv = mean(ppv, na.rm = TRUE),
      sd_ppv = sd(ppv, na.rm = TRUE) / sqrt(n()),
      mean_npv = mean(npv, na.rm = TRUE),
      sd_npv = sd(npv, na.rm = TRUE) / sqrt(n()),
      .groups = "drop"
    )

  return(metrics_summary)
}


print_model_stats <- function(models_list, var_string) {

  coefs <- list()
  pvals <- list()

  # iterate over models that have ptau in them
  for (model_name in names(models_list)) {
    if (grepl(var_string, model_name)) {
      for (fold in 1:5) {
        model <- models_list[[model_name]][[paste0("fold_", fold)]]
        coefs[[model_name]][[paste0("fold_", fold)]] <-
          exp(model$coefficients[var_string])
        pvals[[model_name]][[paste0("fold_", fold)]] <-
          summary(model)$coefficients[var_string, "Pr(>|z|)"]
      }
    }
  }

  print(paste0(var_string, " p-values:"))
  print(paste0("Range: ", range(unlist(pvals))))
  print(paste0("Mean: ", mean(unlist(pvals))))
  print(paste0("SD: ", sd(unlist(pvals))))
  print(paste0(var_string, " coefficients:"))
  print(paste0("Range: ", range(unlist(coefs))))
  print(paste0("Mean: ", mean(unlist(coefs))))
  print(paste0("SD: ", sd(unlist(coefs))))
}



pull_trocs <- function(model_name) {
  list(
    metrics_list[[model_name]]$fold_1$troc,
    metrics_list[[model_name]]$fold_2$troc,
    metrics_list[[model_name]]$fold_3$troc,
    metrics_list[[model_name]]$fold_4$troc,
    metrics_list[[model_name]]$fold_5$troc
  )
}


print_pvalue_latex_table <- function(results_table) {
  results_table_wide <- results_table %>%
    pivot_wider(id_cols = fold, names_from = time, values_from = p_value) %>%
    mutate(across(-fold, ~round(., digits = 4))) %>%
  mutate(across(-fold, ~ifelse(. < 0.05, paste0("\\textbf{", ., "}"), as.character(.))))

# latex table of results_table_wide with 4 decimal places
  xtable_obj <- xtable(results_table_wide)
  digits(xtable_obj) <- c(0, rep(4, ncol(results_table_wide)))  # Set digits for each column
  print(xtable_obj, type = "latex", sanitize.text.function = function(x) x)  # Don't escape LaTeX commands
}

pull_roc_summary <- function(model_names, eval_times) {
  # Initialize lists to store ROC data
  roc_data_all <- list()

  # Initialize dataframe to store ROC curves
  all_roc_curves <- data.frame()

  for (fold in 0:4) {
    for (model_name in model_names) {
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
      # Calculate standard error for each bin
      pooled_se = sqrt(var(TPR, na.rm = TRUE) / n()),
      # Calculate confidence intervals
      ci_lower = mean_TPR - 1.96 * pooled_se,
      ci_upper = mean_TPR + 1.96 * pooled_se,
      FPR = mean(FPR_bin),
      .groups = "drop"
    ) %>%
    # Remove any remaining NAs and ensure we have enough data points
    filter(!is.na(pooled_se)) %>%
    group_by(Model, Time) %>%
    filter(n() >= 10) %>%  # Only keep time points with at least 10 data points
    ungroup()
  roc_summary$Model <- factor(roc_summary$Model, levels=model_names)

  return(roc_summary)
}

calc_pAUC <- function(df, model_names, threshold = 0.25) {
    library(pracma)  # For numerical integration

    # Input validation
    if (!all(model_names %in% unique(df$Model))) {
        stop("Some model names not found in data")
    }
    if (threshold <= 0 || threshold > 1) {
        stop("Threshold must be between 0 and 1")
    }

    # Initialize results dataframe with proper structure
    pAUC_normalized <- data.frame(
        Model = character(),
        Time = numeric(),
        pAUC = numeric(),
        stringsAsFactors = FALSE
    )

    # Filter for FPR ≤ threshold and ensure data is ordered
    df_filtered <- df[df$FPR <= threshold, ] %>%
        arrange(Model, Time, FPR)

    # Calculate pAUC for each model and time point
    for (model_name in model_names) {
        model_df <- df_filtered[df_filtered$Model == model_name, ]
        
        for (time in unique(model_df$Time)) {
            model_df_time <- model_df[model_df$Time == time, ]
            
            # Skip if insufficient data points
            if (nrow(model_df_time) < 2) {
                warning(sprintf("Insufficient data points for model %s at time %s", 
                              model_name, time))
                next
            }

            # Compute partial AUC using trapezoidal integration
            pAUC <- tryCatch({
                trapz(model_df_time$FPR, model_df_time$mean_TPR)
            }, error = function(e) {
                warning(sprintf("Integration failed for model %s at time %s: %s", 
                              model_name, time, e$message))
                return(NA)
            })

            # Add results to dataframe
            pAUC_normalized <- rbind(pAUC_normalized, data.frame(
                Model = model_name,
                Time = time,
                pAUC = pAUC / threshold
            ))
        }
    }

    # for each model, calculate the average and sd of pAUC across time points
    pAUC_normalized <- pAUC_normalized %>%
      group_by(Model) %>%
      summarise(mean_pAUC = mean(pAUC),
                sd_pAUC = sd(pAUC))

    # sort by mean_pAUC in ascending order
    pAUC_normalized <- pAUC_normalized %>%
      arrange(mean_pAUC)

    return(pAUC_normalized)
}



print_auc_latex_table <- function(auc_summary) {
  agg_auc_summary <- aggregate(
    cbind(auc, ci_lower, ci_upper) ~ model + time,
    data = auc_summary,
    FUN = mean
  )

  wide_auc_summary <- agg_auc_summary %>%
    group_by(model, time) %>%
    mutate(
      formatted_value = sprintf("%.2f (%.2f-%.2f)", auc, ci_lower, ci_upper)
    ) %>%
    select(model, time, formatted_value) %>%
    pivot_wider(
      id_cols = model,
      names_from = time,
      values_from = formatted_value,
      names_glue = "{ifelse(is.na(.name), 'model', paste0(.name, 'y'))}"
    )

  # map model names to labels
  model_labels <- c(
    "demographics_lancet" = "Demo+Lancet",
    "ptau_demographics_lancet" = "pTau217+Demo+Lancet",
    "demographics" = "Demo",
    "demographics_no_apoe" = "Demo (-APOE)",
    "lancet" = "Lancet",
    "ptau" = "pTau217",
    "ptau_demographics" = "pTau217+Demo",
    "ptau_demographics_no_apoe" = "pTau217+Demo (-APOE)",
    "ptau_demographics_lancet_no_apoe" = "pTau217+Demo+Lancet (-APOE)",
    "demographics_lancet_no_apoe" = "Demo+Lancet (-APOE)",
    "centiloids_demographics_lancet" = "PET+Demo+Lancet",
    "ptau_centiloids_demographics_lancet" = "pTau217+PET+Demo+Lancet",
    "centiloids" = "PET",
    "centiloids_demographics" = "PET+Demo",
    "centiloids_demographics_no_apoe" = "PET+Demo (-APOE)",
    "centiloids_demographics_lancet" = "PET+Demo+Lancet",
    "centiloids_demographics_lancet_no_apoe" = "PET+Demo+Lancet (-APOE)",
    "ptau_centiloids" = "pTau217+PET",
    "ptau_centiloids_demographics" = "pTau217+PET+Demo",
    "ptau_centiloids_demographics_no_apoe" = "pTau217+PET+Demo (-APOE)",
    "ptau_centiloids_demographics_lancet" = "pTau217+PET+Demo+Lancet",
    "ptau_centiloids_demographics_lancet_no_apoe" = "pTau217+PET+Demo+Lancet (-APOE)"
  )

  # clean up model names
  wide_auc_summary$model <- model_labels[wide_auc_summary$model]

  # reorder rows of wide_auc_summary in ascending order of 3y auc, then 4y auc, then 5y auc, then 6y auc, then 7y auc
  wide_auc_summary <- wide_auc_summary %>%
    arrange(wide_auc_summary$`3y`, wide_auc_summary$`4y`, wide_auc_summary$`5y`,
            wide_auc_summary$`6y`, wide_auc_summary$`7y`)
  # print with 4 decimal places
  print(wide_auc_summary)

  # latex table of wide_auc_summary
  library(xtable)
  print(xtable(wide_auc_summary), type = "latex")
}