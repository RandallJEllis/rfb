#CLAUDE 

library(timeROC)
library(survival)
library(pROC)
library(MLmetrics)

# Load required packages
library(JMbayes2)
library(survival)
library(nlme)
library(splines)

# Generate example dataset
# Longitudinal data for variable A (measured every 1-3 years)
set.seed(123)
n_subjects <- 100
max_years <- 10

# Create longitudinal data
long_data <- data.frame(
  id = rep(1:n_subjects, each = 4),
  year = unlist(lapply(1:n_subjects, function(x) round(sort(runif(4, 0, max_years)), 1))),
  A = rnorm(n_subjects * 4, mean = 5, sd = 1)
)

# Create survival data
surv_data <- data.frame(
  id = 1:n_subjects,
  time = rexp(n_subjects, 0.1),
  status = rbinom(n_subjects, 1, 0.7)
)
surv_data$time <- pmin(surv_data$time, max_years)

# More frequent measurements for other variables
other_vars <- data.frame(
  id = rep(1:n_subjects, each = 20),
  month = rep(seq(0, max_years, length.out = 20), n_subjects),
  var_b = rnorm(n_subjects * 20),
  var_c = rnorm(n_subjects * 20)
)

# Fit the longitudinal mixed model for variable A
lme_fit <- lme(A ~ ns(year, 2),
               random = ~ ns(year, 2) | id,
               data = long_data,
               control = lmeControl(opt = "optim"))

# Fit the survival model
cox_fit <- coxph(Surv(time, status) ~ 1,
                 data = surv_data)

# Fit the joint model
joint_model <- jm(
  Surv_object = cox_fit,
  Mixed_objects = list(lme_fit),
  time_var = "year",
  functional_forms = list("A" = ~ value(A)),
  data_Surv = surv_data,
  id_var = "id"
)

# # Function to extract predictions
# get_predictions <- function(joint_model, newdata, times) {
#   # Survival predictions
#   surv_pred <- predict(joint_model, 
#                        newdata = newdata,
#                        process = "event",
#                        times = times)
#   
#   # Longitudinal predictions for A
#   long_pred <- predict(joint_model,
#                        newdata = newdata,
#                        process = "longitudinal",
#                        times = times)
#   
#   return(list(survival = surv_pred, longitudinal = long_pred))
# }

new_subject_long <- data.frame(
  id = rep(max(long_data$id) + 1, 4),
  year = c(0, 2, 4, 6),
  A = c(5.2, 5.5, 5.8, 6.0)
)

new_subject_surv <- data.frame(
  id = max(long_data$id) + 1,
  time = 1,  # Start at time 0
  status = 0
)

new_subject <- merge(new_subject_long, 
                     new_subject_surv, 
                     by = "id", 
                     all = TRUE)

# Try predictions starting from time 0
predictions <- predict(joint_model, 
                       newdata = new_subject,
                       process = "event",
                       times = seq(0, 5, by = 0.5))


#' Find optimal threshold using various criteria
#' @param true_status Vector of true outcomes
#' @param pred_prob Vector of predicted probabilities
#' @param method Character string indicating optimization method
#' @param min_class_size Minimum size for any class (as proportion)
#' @return List containing optimal threshold(s) and performance metrics
find_optimal_threshold <- function(true_status, pred_prob, 
                                   method = c("youden", "f1", "mcc", "two_cutoff"),
                                   min_class_size = 0.1) {
  
  method <- match.arg(method)
  
  # Function to calculate performance metrics at a given threshold
  calc_metrics <- function(threshold) {
    pred_binary <- ifelse(pred_prob > threshold, 1, 0)
    TP <- sum(pred_binary == 1 & true_status == 1)
    TN <- sum(pred_binary == 0 & true_status == 0)
    FP <- sum(pred_binary == 1 & true_status == 0)
    FN <- sum(pred_binary == 0 & true_status == 1)
    
    sensitivity <- TP / (TP + FN)
    specificity <- TN / (TN + FP)
    precision <- TP / (TP + FP)
    
    # Handle edge cases
    if(is.na(precision)) precision <- 0
    
    f1 <- 2 * (precision * sensitivity) / (precision + sensitivity)
    if(is.na(f1)) f1 <- 0
    
    # Calculate MCC
    numerator <- (TP * TN) - (FP * FN)
    denominator <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc <- ifelse(denominator == 0, 0, numerator / denominator)
    
    list(
      threshold = threshold,
      sensitivity = sensitivity,
      specificity = specificity,
      precision = precision,
      f1 = f1,
      mcc = mcc,
      youden = sensitivity + specificity - 1
    )
  }
  
  # Generate sequence of thresholds to evaluate
  thresholds <- seq(0.1, 0.9, by = 0.01)
  
  if(method %in% c("youden", "f1", "mcc")) {
    # Calculate metrics for all thresholds
    results <- lapply(thresholds, calc_metrics)
    
    # Extract optimization criterion
    criterion_values <- sapply(results, function(x) {
      switch(method,
             youden = x$youden,
             f1 = x$f1,
             mcc = x$mcc)
    })
    
    # Find optimal threshold
    opt_idx <- which.max(criterion_values)
    optimal_result <- results[[opt_idx]]
    
    return(list(
      method = method,
      threshold = optimal_result$threshold,
      performance = optimal_result
    ))
    
  } else if(method == "two_cutoff") {
    # Generate all pairs of thresholds
    threshold_pairs <- expand.grid(
      lower = thresholds,
      upper = thresholds
    )
    threshold_pairs <- threshold_pairs[threshold_pairs$lower < threshold_pairs$upper, ]
    
    # Function to evaluate two-cutoff performance
    evaluate_pair <- function(lower, upper) {
      pred_classes <- cut(pred_prob,
                          breaks = c(-Inf, lower, upper, Inf),
                          labels = c(0, "intermediate", 1))
      
      # Calculate metrics for non-intermediate samples
      decisive_idx <- pred_classes != "intermediate"
      if(sum(decisive_idx) < min_class_size * length(true_status)) {
        return(list(score = -Inf))
      }
      
      pred_binary <- as.numeric(as.character(pred_classes[decisive_idx]))
      true_binary <- true_status[decisive_idx]
      
      TP <- sum(pred_binary == 1 & true_binary == 1)
      TN <- sum(pred_binary == 0 & true_binary == 0)
      FP <- sum(pred_binary == 1 & true_binary == 0)
      FN <- sum(pred_binary == 0 & true_binary == 1)
      
      sensitivity <- TP / (TP + FN)
      specificity <- TN / (TN + FP)
      
      # Proportion of intermediate samples
      prop_intermediate <- mean(pred_classes == "intermediate")
      
      # Combined score: balance between performance and decisiveness
      score <- (sensitivity + specificity) / 2 * (1 - prop_intermediate)
      
      list(
        score = score,
        sensitivity = sensitivity,
        specificity = specificity,
        prop_intermediate = prop_intermediate
      )
    }
    
    # Evaluate all threshold pairs
    pair_results <- apply(threshold_pairs, 1, function(pair) {
      evaluate_pair(pair[1], pair[2])
    })
    
    # Find optimal pair
    scores <- sapply(pair_results, function(x) x$score)
    opt_idx <- which.max(scores)
    opt_pair <- threshold_pairs[opt_idx, ]
    
    return(list(
      method = "two_cutoff",
      lower_threshold = opt_pair$lower,
      upper_threshold = opt_pair$upper,
      performance = pair_results[[opt_idx]]
    ))
  }
}

#' Enhanced calculate_survival_metrics function with threshold optimization
calculate_survival_metrics <- function(joint_model, validation_data, time_points, 
                                       id_var = "id", time_var = "time", 
                                       status_var = "status") {
  
  # First ensure all subjects start at time 0 for prediction
  validation_data_pred <- validation_data
  validation_data_pred$time <- 0  # Set all starting times to 0
  
  # Get dynamic predictions for all subjects
  pred_surv <- predict(joint_model,
                       newdata = validation_data,
                       process = "event",
                       times = time_points,
                       return_data = TRUE)
  
  # Initialize results lists
  results <- list(
    dynamic_auc = list(),
    brier = list(),
    precision = list(),
    recall = list(),
    specificity = list(),
    mcc = list(),
    integrated_brier = NULL
  )
  
  # Add threshold optimization results
  results$threshold_optimization <- list()
  
  # Calculate optimal thresholds for each time point
  for (t in time_points) {
    current_data <- data.frame(
      time = validation_data[[time_var]],
      status = validation_data[[status_var]],
      pred = 1 - pred_surv$pred[, which(pred_surv$times == t)]
    )
    
    valid_indices <- current_data$time >= t
    current_data <- current_data[valid_indices, ]
    true_status <- ifelse(current_data$time <= t & current_data$status == 1, 1, 0)
    pred_prob <- current_data$pred
    
    # Calculate optimal thresholds using different methods
    results$threshold_optimization[[as.character(t)]] <- list(
      youden = find_optimal_threshold(true_status, pred_prob, "youden"),
      f1 = find_optimal_threshold(true_status, pred_prob, "f1"),
      mcc = find_optimal_threshold(true_status, pred_prob, "mcc"),
      two_cutoff = find_optimal_threshold(true_status, pred_prob, "two_cutoff")
    )
  }
  
  # Add threshold optimization summary
  results$plot_thresholds <- function() {
    # Extract thresholds over time
    times <- as.numeric(names(results$threshold_optimization))
    
    # Single threshold methods
    thresholds <- data.frame(
      time = rep(times, 3),
      threshold = c(
        sapply(results$threshold_optimization, function(x) x$youden$threshold),
        sapply(results$threshold_optimization, function(x) x$f1$threshold),
        sapply(results$threshold_optimization, function(x) x$mcc$threshold)
      ),
      method = rep(c("Youden's J", "F1 Score", "MCC"), each = length(times))
    )
    
    # Two-cutoff thresholds
    two_cutoff <- data.frame(
      time = times,
      lower = sapply(results$threshold_optimization, 
                     function(x) x$two_cutoff$lower_threshold),
      upper = sapply(results$threshold_optimization, 
                     function(x) x$two_cutoff$upper_threshold)
    )
    
    # Create plots
    par(mfrow = c(2, 1))
    
    # Plot single thresholds
    plot(thresholds$time, thresholds$threshold, 
         type = "n", ylim = c(0, 1),
         xlab = "Time", ylab = "Threshold",
         main = "Optimal Thresholds Over Time")
    
    colors <- c("blue", "red", "green")
    for(i in 1:3) {
      method_data <- thresholds[thresholds$method == unique(thresholds$method)[i], ]
      lines(method_data$time, method_data$threshold, col = colors[i])
    }
    legend("topright", unique(thresholds$method), col = colors, lty = 1)
    
    # Plot two-cutoff thresholds
    plot(two_cutoff$time, two_cutoff$lower, 
         type = "l", col = "purple", ylim = c(0, 1),
         xlab = "Time", ylab = "Threshold",
         main = "Two-Cutoff Thresholds Over Time")
    lines(two_cutoff$time, two_cutoff$upper, col = "orange")
    legend("topright", c("Lower Threshold", "Upper Threshold"), 
           col = c("purple", "orange"), lty = 1)
    
    par(mfrow = c(1, 1))
  }
  
  return(results)
}

# First combine validation data with longitudinal data
validation_long <- long_data[long_data$id %in% surv_data$id, ]
validation_combined <- merge(validation_long, 
                             surv_data, 
                             by = "id", 
                             all = TRUE)

# Check the maximum time in our validation data
max_valid_time <- max(validation_combined$time)
print(paste("Maximum time in validation data:", max_valid_time))

# Check the minimum time
min_valid_time <- min(validation_combined$time)
print(paste("Minimum time in validation data:", min_valid_time))

# Use a more conservative time range
time_points <- seq(0, max_valid_time - 0.5, by = 0.5)

# Now try metrics calculation
metrics <- calculate_survival_metrics(
  joint_model = joint_model,
  validation_data = validation_combined,
  time_points = seq(0, 5, by = 0.5)  # Using more conservative time points
)

# View threshold optimization results
metrics$plot_thresholds()

# Access specific threshold results
print(metrics$threshold_optimization[["1"]]$youden)
print(metrics$threshold_optimization[["1"]]$two_cutoff)



library(jmbayes2)
library(rBayesianOptimization)
library(splines)
library(parallel)
library(doParallel)
library(foreach)

#' Optimize hyperparameters for joint models using Bayesian Optimization
#' @param train_data Training dataset
#' @param valid_data Validation dataset
#' @param n_iter Number of optimization iterations
#' @param init_points Number of random initialization points
#' @param n_cores Number of cores for parallel processing
optimize_joint_model <- function(train_data, valid_data, n_iter = 50, 
                                 init_points = 10, n_cores = 4) {
  
  # Setup parallel processing
  registerDoParallel(cores = n_cores)
  
  #' Objective function for optimization
  #' Returns negative integrated Brier score (to maximize)
  evaluate_joint_model <- function(n_knots_long, n_knots_base, assoc_type, 
                                   random_effects_depth) {
    
    tryCatch({
      # Round integer parameters
      n_knots_long <- round(n_knots_long)
      n_knots_base <- round(n_knots_base)
      random_effects_depth <- round(random_effects_depth)
      
      # Create spline terms for longitudinal model
      spline_terms <- ns(train_data$year, df = n_knots_long)
      
      # Build random effects formula based on depth
      random_formula <- switch(random_effects_depth,
                               1 ~ 1,
                               2 ~ 1 + year,
                               3 ~ ns(year, df = min(n_knots_long, 3)),
                               4 ~ ns(year, df = min(n_knots_long, 4))
      )
      
      # Fit longitudinal model
      lme_fit <- try(
        lme(A ~ spline_terms,
            random = random_formula,
            data = train_data,
            control = lmeControl(opt = "optim", maxIter = 100)),
        silent = TRUE
      )
      
      if(inherits(lme_fit, "try-error")) {
        return(list(Score = -999))  # Penalty for failed fits
      }
      
      # Fit survival model
      cox_fit <- coxph(Surv(time, status) ~ 1, data = train_data)
      
      # Define association structure
      assoc_struct <- switch(round(assoc_type),
                             1 ~ value(A),
                             2 ~ slope(A),
                             3 ~ area(A),
                             4 ~ value(A) + slope(A)
      )
      
      # Fit joint model
      joint_model <- try(
        jm(Surv_object = cox_fit,
           Mixed_objects = list(lme_fit),
           time_var = "year",
           functional_forms = list("A" = assoc_struct),
           data_Surv = train_data,
           id_var = "id",
           n_knots = n_knots_base),
        silent = TRUE
      )
      
      if(inherits(joint_model, "try-error")) {
        return(list(Score = -999))
      }
      
      # Calculate performance metrics on validation set
      time_points <- seq(0.5, max(valid_data$time), length.out = 10)
      pred_surv <- predict(joint_model,
                           newdata = valid_data,
                           process = "event",
                           times = time_points)
      
      # Calculate Brier scores
      brier_scores <- sapply(seq_along(time_points), function(i) {
        t <- time_points[i]
        pred <- 1 - pred_surv[, i]
        true_status <- ifelse(valid_data$time <= t & valid_data$status == 1, 1, 0)
        mean((true_status - pred)^2, na.rm = TRUE)
      })
      
      # Return negative mean Brier score (since we want to maximize)
      return(list(Score = -mean(brier_scores, na.rm = TRUE)))
      
    }, error = function(e) {
      return(list(Score = -999))  # Penalty for errors
    })
  }
  
  # Define parameter bounds
  bounds <- list(
    n_knots_long = c(2, 6),        # Number of knots for longitudinal splines
    n_knots_base = c(5, 15),       # Number of knots for baseline hazard
    assoc_type = c(1, 4),          # Association structure type
    random_effects_depth = c(1, 4)  # Complexity of random effects
  )
  
  # Run Bayesian Optimization
  opt_results <- BayesianOptimization(
    FUN = evaluate_joint_model,
    bounds = bounds,
    init_points = init_points,
    n_iter = n_iter,
    acq = "ucb",   # Upper confidence bound acquisition function
    kappa = 2.576  # Trade-off between exploration and exploitation
  )
  
  # Get best parameters
  best_params <- opt_results$Best_Par
  
  # Fit final model with best parameters
  final_model <- fit_best_model(train_data, best_params)
  
  return(list(
    best_params = best_params,
    optimization_history = opt_results$History,
    final_model = final_model
  ))
}

#' Helper function to fit model with best parameters
fit_best_model <- function(data, params) {
  # Similar to evaluate_joint_model but returns the fitted model
  # instead of performance metrics
  spline_terms <- ns(data$year, df = round(params$n_knots_long))
  
  random_formula <- switch(round(params$random_effects_depth),
                           1 ~ 1,
                           2 ~ 1 + year,
                           3 ~ ns(year, df = min(round(params$n_knots_long), 3)),
                           4 ~ ns(year, df = min(round(params$n_knots_long), 4))
  )
  
  lme_fit <- lme(A ~ spline_terms,
                 random = random_formula,
                 data = data,
                 control = lmeControl(opt = "optim"))
  
  cox_fit <- coxph(Surv(time, status) ~ 1, data = data)
  
  assoc_struct <- switch(round(params$assoc_type),
                         1 ~ value(A),
                         2 ~ slope(A),
                         3 ~ area(A),
                         4 ~ value(A) + slope(A)
  )
  
  joint_model <- jm(Surv_object = cox_fit,
                    Mixed_objects = list(lme_fit),
                    time_var = "year",
                    functional_forms = list("A" = assoc_struct),
                    data_Surv = data,
                    id_var = "id",
                    n_knots = round(params$n_knots_base))
  
  return(joint_model)
}


