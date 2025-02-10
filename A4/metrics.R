


# Function to calculate Brier score at a specific time point
calculate_brier_at_time <- function(model, data, t) {
  # Create prediction dataset up to time t
  pred_data <- data %>%
    filter(tstart <= t) %>%
    group_by(id) %>%  # Assuming there's an id column for subjects
    filter(row_number() == n()) %>%  # Get last record for each subject
    ungroup()
  
  # Update age at time t for prediction
  pred_data <- pred_data %>%
    mutate(
      age_t = age + t  # Update age based on your tt function
    )
  
  # Fit survival curves
  surv_fit <- survfit(model, newdata = pred_data)
  
  # Find the survival probabilities at time t
  # Get the index of the time point closest to t
  time_index <- which.min(abs(surv_fit$time - t))
  
  # Extract survival probabilities at that time
  surv_probs <- surv_fit$surv[time_index, ]
  
  # Calculate observed status at time t
  observed_status <- ifelse(pred_data$tstop <= t & pred_data$event == 1, 1, 0)
  
  # Calculate Brier score
  brier_score <- mean((observed_status - (1 - surv_probs))^2, na.rm = TRUE)
  
  return(brier_score)
}

# Function to calculate Brier scores across multiple time points
calculate_brier_scores <- function(model, data, times) {
  # Calculate Brier score for each time point
  brier_scores <- sapply(times, function(t) {
    tryCatch({
      calculate_brier_at_time(model, data, t)
    }, error = function(e) {
      warning(sprintf("Error calculating Brier score at time %f: %s", t, e$message))
      return(NA)
    })
  })
  
  # Create results dataframe
  results <- data.frame(
    time = times,
    brier = brier_scores
  )
  
  return(results)
}

# Function to calculate multiple metrics at specific time points
calculate_survival_metrics <- function(model, data, times) {
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
    iid = TRUE
  )
  
  auc_results <- data.frame(
    time = times,
    auc = troc$AUC#,
    # auc_lower = troc$AUC - 1.96 * sqrt(var(troc$AUC)),
    # auc_upper = troc$AUC + 1.96 * sqrt(var(troc$AUC))
  )
  
  results$auc <- auc_results
  
  # 2. Brier Scores
  brier_results <- calculate_brier_scores(model, data, times)
  results$brier <- brier_results
  
  # 3. Concordance Index (Harrell's C)
  c_index <- concordance(model)
  results$concordance <- c_index
  
  return(results)
}