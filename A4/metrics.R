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