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
  cat("\nTime point:", t)
  cat("\nNumber of subjects:", nrow(pred_data))

  # Calculate inverse probability of censoring weights
  cens_model <- survfit(Surv(tstart, tstop, 1 - event) ~ 1, data = data)
  cens_probs <- summary(cens_model, times = t)$surv[1] # Take first (and should be only) probability

  if (is.na(cens_probs) || cens_probs <= 0) {
    warning(sprintf("Invalid censoring probability at time %f", t))
    return(NA)
  }

  # Get weights - same weight for all subjects at time t
  weights <- rep(1 / cens_probs, nrow(pred_data))
  cat("\nNumber of weights:", length(weights))

  # Get survival predictions at time t
  surv_probs <- try({
    # Use predictSurvProb from pec package which handles time-varying covariates
    1 - pec::predictSurvProb(model, newdata = pred_data, times = t)
  })

  if (inherits(surv_probs, "try-error")) {
    warning(sprintf("Error getting survival predictions at time %f", t))
    return(NA)
  }

  cat("\nLength of survival probabilities:", length(surv_probs))

  if (length(surv_probs) != nrow(pred_data)) {
    warning(sprintf(
      "Mismatch in prediction lengths at time %f: surv_probs=%d, pred_data=%d",
      t, length(surv_probs), nrow(pred_data)
    ))
    return(NA)
  }

  # Calculate observed status at time t
  observed_status <- ifelse(pred_data$tstop <= t & pred_data$event == 1, 1, 0)
  cat("\nLength of observed status:", length(observed_status))

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

  cat("\nCalculated Brier score:", brier_score, "\n")
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

  # Get linear predictor (which doesn't vary with time)
  risk_scores <- predict(model, newdata = data, type="lp")

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