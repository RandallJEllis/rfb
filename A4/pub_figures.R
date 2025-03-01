library(survival)
library(timeROC)
library(riskRegression)
library(ggplot2)
library(patchwork)  # for combining plots
library(scales)     # for nice formatting
library(viridis)    # for colorblind-friendly colors
library(gridExtra)
library(cowplot)
library(ggscidca)

# Theme for consistent styling
get_publication_theme <- function() {
  publication_theme <- theme_minimal() +
    theme(
      text = element_text(family = "Arial", size = 12),
      axis.title = element_text(size = 12, face = "bold"),
      axis.text = element_text(size = 10),
      legend.title = element_text(size = 12, face = "bold"),
      legend.text = element_text(size = 10),
      legend.position = "bottom",
      panel.grid.minor = element_blank(),
      panel.border = element_rect(fill = NA, color = "grey80"),
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 10, hjust = 0.5)
    )
  return(publication_theme)
}

td_plot <- function(summary_df, metric = "auc", all_models = FALSE, 
                    model_colors = NULL, model_labels = NULL) {
  # Filter models if all_models is FALSE and model_labels is provided
  if (!all_models && !is.null(model_labels)) {
    summary_df <- summary_df %>%
      filter(model %in% names(model_labels))
  }

  # If model_colors not provided, generate colors using viridis
  if (is.null(model_colors)) {
    unique_models <- unique(summary_df$model)
    model_colors <- setNames(
      viridis(length(unique_models)),
      unique_models
    )
  }

  # If model_labels not provided, use model names directly
  if (is.null(model_labels)) {
    model_labels <- setNames(
      unique(summary_df$model),
      unique(summary_df$model)
    )
  }

  # Set y-axis label based on metric
  y_labels <- list(
    auc = "AUROC",
    brier = "Brier Score",
    concordance = "Concordance"
  )

  titles <- list(
    auc = "Time-varying AUROC",
    brier = "Time-varying Brier score",
    concordance = "Time-varying Concordance"
  )

  # Create base plot using appropriate y-value column
  if (metric == "auc") {
    y_col <- "auc"
    lower_col <- "ci_lower"
    upper_col <- "ci_upper"
  } else {
    y_col <- "mean_metric"
    lower_col <- "ymin"
    upper_col <- "ymax"
  }

  # Create base plot
  base_plot <- ggplot(summary_df,
    aes(x = time,
        y = .data[[y_col]],  # Use .data to refer to column dynamically
        color = model,
        fill = model)) +
    labs(
      title = titles[[metric]],
      x = "Follow-up Time (years)",
      y = y_labels[[metric]]
    )

  # Add confidence intervals
  base_plot <- base_plot +
    geom_ribbon(aes(ymin = .data[[lower_col]], 
                    ymax = .data[[upper_col]]),
                alpha = 0.1,
                show.legend = FALSE)

  # Add common elements
  base_plot +
    geom_line(linewidth = 1.2) +
    geom_point(size = 3, shape = 21, fill = "white") +
    scale_color_manual(values = model_colors,
                       labels = model_labels,
                       name = "Model") +
    scale_fill_manual(values = model_colors,
                      labels = model_labels,
                      name = "Model") +
    theme_bw(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
      plot.subtitle = element_text(size = 14, hjust = 0.5),
      axis.title = element_text(face = "bold", size = 14),
      axis.text = element_text(size = 12),
      legend.title = element_text(face = "bold", size = 14),
      legend.text = element_text(size = 12),
      panel.grid.minor = element_blank(),
      legend.position = "right"
    )
}

# Helper function to process predictions and calibration data for one model
process_calibration_data <- function(model_name, model, val_df, time,
                                     fixed_breaks, fold) {
  # Get predictions for current model
  pred_probs <- 1 - pec::predictSurvProb(
    model,
    newdata = val_df,
    times = time
  )

  # Create risk groups using fixed breaks
  risk_groups <- cut(pred_probs, breaks = fixed_breaks, include.lowest = TRUE)

  # Calculate calibration metrics for each risk group
  cal_data <- data.frame()
  for (group in levels(risk_groups)) {
    group_data <- val_df[risk_groups == group, ]
    if (nrow(group_data) > 0) {
      surv_fit <- survfit(Surv(tstop, event) ~ 1, data = group_data)
      surv_summary <- summary(surv_fit, times = time)

      if (length(surv_summary$surv) > 0) {
        cal_data <- rbind(cal_data, data.frame(
          fold = fold,
          time = time,
          model = model_name,
          risk_group = group,
          pred = mean(pred_probs[risk_groups == group]),
          actual = 1 - surv_summary$surv[1]
        ))
      }
    }
  }

  return(cal_data)
}

# Function to calculate calibration data across all models and folds
calculate_calibration_data <- function(models_list, val_df_l,
                                     times = seq(3, 8),
                                     selected_models = NULL) {
  # If no models specified, use all available models
  if (is.null(selected_models)) {
    selected_models <- names(models_list)
  }

  cal_data_all <- list()

  # First pass: collect all predictions to create fixed breaks
  for (t in times) {
    all_preds_by_model <- list()

    # Collect predictions across all folds and models
    for (fold in 0:4) {
      for (model_name in selected_models) {
        model <- overwrite_na_coef_to_zero(
          models_list[[model_name]][[paste0("fold_", fold + 1)]]
        )

        pred_probs <- 1 - pec::predictSurvProb(
          model,
          newdata = val_df_l[[paste0("fold_", fold + 1, "_", model_name)]],
          times = t
        )

        # Store predictions by model
        all_preds_by_model[[model_name]] <- c(
          all_preds_by_model[[model_name]],
          pred_probs
        )
      }
    }

    # Calculate fixed breaks using all predictions
    all_preds <- unlist(all_preds_by_model)
    raw_breaks <- quantile(all_preds, probs = seq(0, 1, length.out = 11))
    fixed_breaks <- numeric(length(raw_breaks))

    # Handle duplicate break points
    for (i in seq_along(raw_breaks)) {
      duplicates <- sum(raw_breaks[1:i] == raw_breaks[i])
      fixed_breaks[i] <- if (duplicates > 1) {
        raw_breaks[i] + (duplicates - 1) * .Machine$double.eps
      } else {
        raw_breaks[i]
      }
    }

    # Second pass: calculate calibration using fixed breaks
    cal_data_time <- list()

    for (fold in 0:4) {
      fold_data <- list()

      for (model_name in selected_models) {
        model <- overwrite_na_coef_to_zero(
          models_list[[model_name]][[paste0("fold_", fold + 1)]]
        )

        model_data <- process_calibration_data(
          model_name,
          model,
          val_df_l[[paste0("fold_", fold + 1, "_", model_name)]],
          t,
          fixed_breaks,
          fold
        )

        fold_data[[model_name]] <- model_data
      }

      cal_data_time[[paste0("fold_", fold + 1)]] <- do.call(rbind, fold_data)
    }

    cal_data_all[[as.character(t)]] <- cal_data_time
  }

  # Combine and summarize calibration data
  all_cal_data <- do.call(rbind, lapply(names(cal_data_all), function(t) {
    do.call(rbind, cal_data_all[[t]])
  }))

  # First, calculate the mean predictions and observed outcomes for each fold
  fold_level <- all_cal_data %>%
    group_by(time, model, risk_group, fold) %>%
    summarize(
      pred_mean = mean(pred),
      actual_prop = mean(actual),
      n = n(),
      .groups = "keep"
    )

  # Then aggregate across folds to get the final calibration points
  calibration_points <- fold_level %>%
    group_by(time, model, risk_group) %>%
    summarize(
      pred = mean(pred_mean),  # Average prediction across folds
      actual = mean(actual_prop),  # Average actual proportion across folds
      sd = sd(actual_prop),  # SD
      lower = actual - sd,
      upper = actual + sd,
      n_folds = n(),  # Number of folds contributing to this point
      .groups = "drop"
    )

  return(calibration_points)
}

# Modified plotting function to include confidence intervals
calibration_plots <- function(cal_data, times, model_colors = NULL, model_labels = NULL) {
  publication_theme <- get_publication_theme()
  
  # If model_colors not provided, generate colors using viridis
  if (is.null(model_colors)) {
    unique_models <- unique(cal_data$model)
    model_colors <- setNames(
      viridis(length(unique_models)),
      unique_models
    )
  }

  # If model_labels not provided, use model names directly
  if (is.null(model_labels)) {
    model_labels <- setNames(
      unique(cal_data$model),
      unique(cal_data$model)
    )
  }

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
      geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.2) +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                  color = "gray50") +
      geom_line(linewidth = 1) +
      geom_point(size = 2) +
      scale_color_manual(values = model_colors, name = "Model", labels = model_labels) +
      scale_fill_manual(values = model_colors, name = "Model", labels = model_labels) +
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

dca_plots <- function(all_dca_data, times = seq(3, 8), 
                     model_colors = NULL, model_labels = NULL) {
  # If model_colors not provided, generate colors using viridis
  if (is.null(model_colors)) {
    unique_models <- unique(all_dca_data$model)
    model_colors <- setNames(
      viridis(length(unique_models)),
      unique_models
    )
  }

  # If model_labels not provided, use model names directly
  if (is.null(model_labels)) {
    model_labels <- setNames(
      unique(all_dca_data$model),
      unique(all_dca_data$model)
    )
  }

  plots <- list()

  for (t in times) {
    # Pre-filter data for this timepoint
    t_data_all <- all_dca_data[all_dca_data$time == t, ]
    
    # Initialize matrices for all models
    models_to_analyze <- unique(t_data_all$model)
    thresholds <- NULL
    model_vals <- list()
    
    # Process each fold
    for (fold in unique(t_data_all$fold)) {
      fold_data <- t_data_all[t_data_all$fold == fold, ]
      
      # Calculate DCA once for reference strategies
      dca_ref <- stdca(
        data = fold_data[fold_data$model == models_to_analyze[1], ],
        outcome = "event",
        ttoutcome = "tstop", 
        timepoint = t,
        predictors = "pred_prob",
        xstart = 0,
        xstop = 1,
        probability = FALSE,
        harm = NULL,
        graph = FALSE
      )
      
      # Store thresholds on first iteration
      if (is.null(thresholds)) {
        thresholds <- dca_ref$net.benefit$threshold
        n_thresholds <- length(thresholds)
        none_vals <- matrix(NA, nrow = length(unique(t_data_all$fold)), ncol = n_thresholds)
        all_vals <- matrix(NA, nrow = length(unique(t_data_all$fold)), ncol = n_thresholds)
        for (model_name in models_to_analyze) {
          model_vals[[model_name]] <- matrix(NA, nrow = length(unique(t_data_all$fold)), ncol = n_thresholds)
        }
      }
      
      # Store reference values
      none_vals[fold + 1, ] <- dca_ref$net.benefit$none
      all_vals[fold + 1, ] <- dca_ref$net.benefit$all
      
      # Calculate DCA for each model
      for (model_name in models_to_analyze) {
        model_data <- fold_data[fold_data$model == model_name, ]
        if (nrow(model_data) > 0) {
          dca_model <- stdca(
            data = model_data,
            outcome = "event",
            ttoutcome = "tstop",
            timepoint = t,
            predictors = "pred_prob",
            xstart = 0,
            xstop = 1,
            probability = FALSE,
            harm = NULL,
            graph = FALSE
          )
          model_vals[[model_name]][fold + 1, ] <- dca_model$net.benefit$pred_prob
        }
      }
    }

    # Calculate means and SDs
    none_mean <- colMeans(none_vals, na.rm = TRUE)
    all_mean <- colMeans(all_vals, na.rm = TRUE)
    model_means <- lapply(model_vals, colMeans, na.rm = TRUE)
    model_sds <- lapply(model_vals, function(x) apply(x, 2, sd, na.rm = TRUE))

    is_leftmost <- as.numeric(t) %in% c(3, 6)
    is_bottom <- as.numeric(t) >= 6
    is_middle_bottom <- t == 7

    # Create plot
    current_plot <- create_dca_plot(
      thresholds, none_mean, all_mean, 
      model_means, model_sds, model_colors,
      models_to_analyze, t, 
      is_leftmost, is_bottom, is_middle_bottom
    )

    plots[[as.character(t)]] <- current_plot
  }

  # Create final combined plot
  wrap_plots(plots, ncol = 3) +
    plot_layout(guides = "collect") & 
    theme(legend.position = "bottom")
}

# Helper function to create individual DCA plot
create_dca_plot <- function(thresholds, none_mean, all_mean,
                           model_means, model_sds, model_colors,
                           models_to_analyze, t,
                           is_leftmost, is_bottom, is_middle_bottom) {

  current_plot <- ggplot() +
    # Add reference lines
    geom_line(
      data = data.frame(x = thresholds, y = none_mean),
      aes(x = x, y = y, linetype = "Treat None"), 
      color = "gray50"
    ) +
    geom_line(
      data = data.frame(x = thresholds, y = all_mean),
      aes(x = x, y = y, linetype = "Treat All"), 
      color = "gray50"
    )

  # Add model lines and ribbons
  for (model_name in models_to_analyze) {
    plot_data <- data.frame(
      x = thresholds,
      y = model_means[[model_name]],
      ymin = model_means[[model_name]] - model_sds[[model_name]],
      ymax = model_means[[model_name]] + model_sds[[model_name]],
      model = model_name
    )
    
    current_plot <- current_plot +
      geom_ribbon(
        data = plot_data,
        aes(x = x, y = y, ymin = ymin, ymax = ymax, fill = model),
        alpha = 0.2
      ) +
      geom_line(
        data = plot_data,
        aes(x = x, y = y, color = model), 
        linewidth = 1
      )
  }

  current_plot +
    scale_color_manual(
      values = model_colors,
      name = "Model",
      labels = c(
        "Demographics",
        "Demographics (no APOE)",
        "Demographics + Lifestyle",
        "Plasma p-tau217",
        "Full Model"
      )
    ) +
    scale_fill_manual(
      values = model_colors,
      name = "Model",
      labels = c(
        "Demographics",
        "Demographics (no APOE)",
        "Demographics + Lifestyle",
        "Plasma p-tau217",
        "Full Model"
      )
    ) +
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
    theme(
      legend.position = if (is_middle_bottom) "bottom" else "none",
      legend.box = "vertical",
      aspect.ratio = 0.7
    )
}

# Function to create publication-quality figures
create_publication_figures <- function(baseline_model, biomarker_model,
                                       data, auc_summary, brier_summary,
                                       cal_data, times) {
  publication_theme <- get_publication_theme()

  # Colors
  model_colors <- c("Baseline" = "#287271", "Biomarker" = "#B63679")

  # 1. Time-Dependent AUC Plot
  td_auc <- td_plot(auc_summary, metric = "auc")
  td_brier <- td_plot(brier_summary, metric = "brier")

  # 2. Calibration Plots
  calibration <- calibration_plots(cal_data, times, model_colors)

  # 3. Decision Curve Analysis
  dca_plots_l <- dca_plots(baseline_model, biomarker_model,
                           data, times, model_colors)

  # Combine plots
  combined_plot <- (
    (td_auc | dca_plots_l$final_plot) /
      (calibration$final_plot | plot_spacer())
  ) +
    plot_layout(widths = c(1, 1)) +
    plot_annotation(
      title = "Model Performance Comparison",
      subtitle = "Baseline vs. Biomarker Model",
      theme = theme(
        plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 12, hjust = 0.5)
      )
    )

  return(list(
    time_dependent_auc = td_auc,
    time_dependent_brier = td_brier,
    calibration = calibration$all_plots,
    decision_curve = dca_plots_l$all_plots,
    combined_plot = combined_plot
  ))
}

library(survival)
library(timeROC)
library(riskRegression)
library(ggplot2)
library(patchwork)
library(survcomp)
library(pec)

create_additional_figures <- function(models, val_data_dict, times) {
  publication_theme <- get_publication_theme()

  # Initialize lists to store ROC curves for each model
  roc_curves <- list()
  roc_data_list <- list()

  # Calculate ROC curves for each model using its corresponding validation data
  for (model_name in names(models)) {
    # Use the appropriate validation data for this model
    val_data <- val_data_dict[[model_name]]

    roc_curves[[model_name]] <- timeROC(
      T = val_data$tstop,
      delta = val_data$event,
      marker = predict(models[[model_name]], newdata = val_data, type = "lp"),
      cause = 1,
      times = times,
      iid = TRUE,
      ROC = TRUE
    )

    # Create data frame for this model
    roc_data_list[[model_name]] <- data.frame(
      FPR = as.vector(roc_curves[[model_name]]$FP),
      TPR = as.vector(roc_curves[[model_name]]$TP),
      Model = model_name,
      Time = rep(times, each = length(roc_curves[[model_name]]$FP) / length(times))
    )
  }

  # Combine all ROC data
  roc_data <- do.call(rbind, roc_data_list)

  # Create ROC plot
  p5 <- ggplot(roc_data, aes(x = FPR, y = TPR, color = Model)) +
    geom_line(linewidth = 1) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
    scale_color_manual(
      values = c(
        "demographics" = "#287271",
        "demographics_no_apoe" = "#B63679",
        "demographics_lancet" = "#4B0082",
        "ptau" = "#FF4500",
        "ptau_demographics_lancet" = "#008000"
      ),
      labels = c(
        "Demographics",
        "Demographics (no APOE)",
        "Demographics + Lifestyle",
        "Plasma p-tau217",
        "Full Model"
      )
    ) +
    facet_wrap(~Time, labeller = label_both) +
    labs(
      x = "False Positive Rate",
      y = "True Positive Rate",
      title = "Dynamic ROC Curves",
      subtitle = "At Different Follow-up Times"
    ) +
    coord_equal() +
    publication_theme

  # Calculate prediction error curves using appropriate validation data for each model
  pe <- tryCatch(
    {
      # Create a list to store predictions for each model
      predictions <- list()
      for (model_name in names(models)) {
        val_data <- val_data_dict[[model_name]]
        predictions[[model_name]] <- pec::predictSurvProb(
          models[[model_name]],
          newdata = val_data,
          times = times
        )
      }

      # Use the first validation dataset's structure for the reference
      reference_data <- val_data_dict[[names(models)[1]]]

      pec(
        object = predictions,
        data = reference_data,
        times = times,
        exact = FALSE,
        reference = TRUE,
        splitMethod = "none",
        formula = Surv(tstop, event) ~ 1,
        start = 3,
        verbose = FALSE
      )
    },
    error = function(e) {
      warning("Prediction error calculation failed, returning NULL")
      NULL
    }
  )

  # Create prediction error plot if calculation succeeded
  if (!is.null(pe)) {
    # Create data frame for all models
    pe_data <- data.frame(
      time = rep(pe$time, length(models)),
      error = unlist(lapply(names(models), 
                            function(model_name) pe$AppErr[[model_name]])),
      Model = factor(rep(names(models), each = length(pe$time)))
    )

    p6 <- ggplot(pe_data, aes(x = time, y = error, color = Model)) +
      geom_line(linewidth = 1) +
      scale_color_manual(
        values = c(
          "demographics" = "#287271",
          "demographics_no_apoe" = "#B63679",
          "demographics_lancet" = "#4B0082",
          "ptau" = "#FF4500",
          "ptau_demographics_lancet" = "#008000"
        ),
        labels = c(
          "Demographics",
          "Demographics (no APOE)",
          "Demographics + Lifestyle",
          "Plasma p-tau217",
          "Full Model"
        )
      ) +
      labs(
        x = "Time (Years)",
        y = "Prediction Error",
        title = "Integrated Prediction Error",
        subtitle = "Lower Values Indicate Better Performance"
      ) +
      publication_theme
  } else {
    p6 <- NULL
  }

  # Create combined plot
  combined_additional <- (p5 + p6) +
    plot_annotation(
      title = "Additional Model Performance Metrics",
      theme = theme(
        plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
      )
    )

  return(list(
    dynamic_roc = list(
      plot = p5,
      data = roc_data
    ),
    prediction_error = list(
      plot = p6,
      data = if (!is.null(pe)) pe_data else NULL
    ),
    combined_additional = combined_additional,
    troc = roc_curves[[1]] # Return first ROC curve for backwards compatibility
  ))
}
