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
library(tidyverse)

# Theme for consistent styling
get_publication_theme <- function() {
  publication_theme <- theme_minimal() +
    theme(
      # text = element_text(family = "Arial", size = 12),
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

get_colors_labels <- function() {
# Define colors for all models using a colorblind-friendly palette
  lookup_model_colors <- c(
    # Core models
    "demographics_lancet" = "#E69F00",         # orange-yellow
    "ptau" = "#882255",   # wine
    "ptau_demographics_lancet" = "#CC79A7",    # pink
    "centiloids" = "#225588",                  # indigo/deep blue
    "centiloids_demographics_lancet" = "#79A2CC",     # lighter indigo blue
    "ptau_centiloids_demographics_lancet" = "#009E73",  # green
    
    "demographics" = "#0072B2",                # blue
    "demographics_no_apoe" = "#D55E00",        # vermillion
    
    # Lancet variations
    "lancet" = "#56B4E9",                      # sky blue
    "demographics_lancet_no_apoe" = "#F0E442", # yellow
    
    # pTau combinations
    "ptau_demographics" = "#44AA99",           # teal
    "ptau_demographics_no_apoe" = "#882255",   # wine red
    "ptau_demographics_lancet_no_apoe" = "#117733", # green
    
    # PET (centiloids) base models
    "centiloids_demographics" = "#AA4499", # purple
    "centiloids_demographics_no_apoe" = "#AA4499", # purple
    
    # PET with Lancet
    "centiloids_demographics_lancet_no_apoe" = "#CC6677", # olive
    
    # PET with pTau combinations
    "ptau_centiloids" = "#999933",   # rose
    "ptau_centiloids_demographics" = "#AA4400",  # brown
    "ptau_centiloids_demographics_no_apoe" = "#888888", # grey
    
    "ptau_centiloids_demographics_lancet_no_apoe" = "#44AA99" # blue green
  )

  # map model names to labels
  lookup_model_labels <- c(
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

  return(list(
    colors = lookup_model_colors,
    labels = lookup_model_labels
  ))
}

td_plot <- function(summary_df, all_results_df=NULL, model_names, metric = "auc", all_models = FALSE) {
  # Filter models if all_models is FALSE
  if (!all_models) {
    summary_df <- summary_df %>%
      filter(model %in% 
               model_names)
  }

  lookup_model_colors <- get_colors_labels()$colors
  lookup_model_labels <- get_colors_labels()$labels

  # Create model_labels by subsetting lookup_model_labels
  model_labels <- lookup_model_labels[model_names]

  # Create model_colors by subsetting lookup_model_colors
  model_colors <- lookup_model_colors[model_names]

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

  base_plot <- ggplot() +
  # Add confidence interval ribbons
  geom_ribbon(data = summary_df,
              aes(x = time, 
                  ymin = .data[[lower_col]], 
                  ymax = .data[[upper_col]], 
                  fill = model),
              alpha = 0.2) +
  # Plot mean lines
  geom_line(data = summary_df, 
            aes(x = time, 
                y = .data[[y_col]], 
                color = model),
            linewidth = 1) +
  # Add white circles at each time point
  geom_point(data = summary_df,
             aes(x = time, 
                 y = .data[[y_col]], 
                 color = model),
             size = 3, shape = 21, fill = "white") +
  labs(
    title = titles[[metric]],
    x = "Time (years)",
    y = y_labels[[metric]]
  ) +
  scale_color_manual(values = model_colors,
                    labels = model_labels,
                    name = "Model") +
  scale_fill_manual(values = model_colors,
                    labels = model_labels,
                    name = "Model") +
  {if (metric == "auc") {
    list(
      scale_y_continuous(breaks = seq(0.3, 1, by = 0.1)),
      scale_x_continuous(breaks = seq(1, 10, 1)),
      geom_hline(yintercept = 0.5, linetype = "dotted", color = "gray50")
    )
  }} +
  theme_bw(base_size = 14) +
  theme(
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
      plot.subtitle = element_text(size = 14, hjust = 0.5),
      axis.title = element_text(face = "bold", size = 14),
      axis.text = element_text(size = 12),
      legend.title = element_text(face = "bold", size = 14),
      legend.text = element_text(size = 12),
      panel.grid.minor = element_blank(),
      panel.grid = element_blank(),
      panel.background = element_rect(fill = "white", color = NA),
      legend.position = "right"
    )

  # if (!is.null(all_results_df)) {
  #   base_plot <- base_plot +
  #   # Plot individual fold lines
  #   geom_line(data = all_results_df, 
  #             aes(x = time, y = metric, color = model, 
  #               group = interaction(model, fold)),
  #           linewidth = 0.5, alpha = 0.3) 
  #   # Add white circles at each time point
  #   # geom_point(data = all_results_df,
  #   #           aes(x = time, y = metric, color = model),
  #   #           size = 1, shape = 21, fill = "white")
  # }

  return(base_plot)




  # Create base plot
  # base_plot <- ggplot(summary_df,
  #   aes(x = time,
  #       y = .data[[y_col]],  # Use .data to refer to column dynamically
  #       color = model,
  #       fill = model)) +
  #   # Add white circles at each time point
  #   geom_point(data = summary_df,
  #            aes(x = time, y = auc, color = model),
  #            size = 3, shape = 21, fill = "white") +
  #   labs(
  #     title = titles[[metric]],
  #     x = "Follow-up Time (years)",
  #     y = y_labels[[metric]]
  #   ) + 

  #   # plot individual fold lines
  #   geom_line(data = all_results_df, 
  #           aes(x = time, y = auc, color = model, 
  #               group = interaction(model, fold)),
  #           linewidth = 0.5, alpha = 0.3) +
  #   # Add white circles at each time point
  #   geom_point(data = all_results_df,
  #             aes(x = time, y = auc, color = model),
  #             size = 1, shape = 21, fill = "white")
 
  # # Add confidence intervals
  # base_plot <- base_plot +
  #   geom_ribbon(aes(ymin = .data[[lower_col]], 
  #                   ymax = .data[[upper_col]]),
  #               alpha = 0.1,
  #               show.legend = FALSE)

  # # Add common elements
  # base_plot +
  #   # geom_line(linewidth = 1.2) +
  #   # geom_point(size = 3, shape = 21, fill = "white") +
  #   scale_color_manual(values = model_colors,
  #                      labels = model_labels,
  #                      name = "Model") +
  #   scale_fill_manual(values = model_colors,
  #                     labels = model_labels,
  #                     name = "Model") +
  #   theme_bw(base_size = 14) +
  #   theme(
  #     plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
  #     plot.subtitle = element_text(size = 14, hjust = 0.5),
  #     axis.title = element_text(face = "bold", size = 14),
  #     axis.text = element_text(size = 12),
  #     legend.title = element_text(face = "bold", size = 14),
  #     legend.text = element_text(size = 12),
  #     panel.grid.minor = element_blank(),
  #     panel.grid = element_blank(),
  #     panel.background = element_rect(fill = "white", color = NA),
  #     legend.position = "right"
  #   )
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
                                       times = seq(3, 8)) {
  selected_models <- c(
    # "demographics", "demographics_no_apoe",
    "demographics_lancet", "ptau",
    "ptau_demographics_lancet"
  )

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
calibration_plots <- function(cal_data, times, model_colors) {

  publication_theme <- get_publication_theme()
  model_colors <- c(
      "demographics_lancet" = "#E69F00",    # orange
      "ptau" = "#CC79A7",                   # pink
      "ptau_demographics_lancet" = "#009292"  # turquoise
      # "demographics" = "#440154",           # deep purple
      # "demographics_no_apoe" = "#009E73"   # teal
    )

  model_labels <- c(
      # "Demo", "Demo (-APOE)",
      "Demo + Lancet", 
      #"Demo+ Lancet\n(-APOE)",
      "pTau217", 
      # "Demo + pTau217",
      # "Demo + pTau217\n(-APOE)", 
      # "Demo + pTau217\n+ Lancet (-APOE)",
      "Demo + pTau217\n+ Lancet"
    )

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

dca_plots <- function(all_dca_data, times = seq(3, 8), model_colors = NULL) {
  if (is.null(model_colors)) {
    model_colors <- c(
      "demographics" = "#287271",
      "demographics_no_apoe" = "#B63679", 
      "demographics_lancet" = "#E69F00",    # orange
      "ptau" = "#CC79A7",                  # pink
      "ptau_demographics_lancet" = "#009292"  # turquoise
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


plot_auc_over_time <- function(auc_summary, model_names) {
  sub_auc_summary <- auc_summary %>% 
    filter(model %in% model_names) %>%
    mutate(fold = as.factor(fold), # Make fold a factor for better grouping
          metric = auc)  
  sub_auc_summary$model <- factor(sub_auc_summary$model, levels=model_names)

  agg_sub_auc_summary <- aggregate(
    cbind(auc, ci_lower, ci_upper) ~ model + time,
    data = sub_auc_summary,
    FUN = mean
  )
  agg_sub_auc_summary$model <- factor(agg_sub_auc_summary$model, levels=model_names)

  # plot auc over time
  auc_plot <- td_plot(agg_sub_auc_summary,
                      sub_auc_summary,
                      model_names,
                      metric = "auc")

  return(auc_plot)
}

plot_all_roc_curves <- function(model_names, eval_times) {
  width <- 8
  height <- 6
  roc_summary <- pull_roc_summary(model_names, eval_times)

  lookup_model_colors <- get_colors_labels()$colors
  lookup_model_labels <- get_colors_labels()$labels

  model_colors <- lookup_model_colors[model_names]
  model_labels <- lookup_model_labels[model_names]

  # Create faceted plot of ROC curves
  roc_plot <- ggplot(roc_summary, aes(x = FPR, y = mean_TPR, color = Model)) +
    geom_ribbon(aes(
      ymin = ci_lower,
      ymax = ci_upper,
      fill = Model
    ), alpha = 0.3, color = NA) +
    geom_line(linewidth = 0.5) +
    geom_abline(
      slope = 1, intercept = 0,
      linetype = "dashed", color = "gray50"
    ) +
    scale_color_manual(values = model_colors, labels = model_labels) +
    scale_fill_manual(values = model_colors, labels = model_labels, guide = "none") +
    facet_wrap(~Time,
      labeller = labeller(Time = function(x) sprintf("%s years", x))
    ) +
    labs(
      x = "False Positive Rate",
      y = "True Positive Rate",
      title = "ROC Curves at Different Time Points"
    ) +
    coord_fixed() +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
      plot.subtitle = element_text(size = 14, hjust = 0.5),
      axis.title = element_text(face = "bold", size = 14),
      # axis.text = element_text(size = 12),
      legend.title = element_text(face = "bold", size = 14),
      legend.text = element_text(size = 12),
      panel.grid.minor = element_blank(),
      # panel.grid = element_blank(),
      # panel.background = element_rect(fill = "white", color = NA),
      legend.position = "right",
      panel.spacing = unit(1, "cm"),
      axis.text = element_text(size = 8),
      plot.margin = margin(0.5, 0.5, 0.5, 0.5, "cm")
    )

  return(roc_plot)
}

plot_roc_biggest_year_difference <- function(auc_summary, agg_auc_summary, model_names, eval_times) {
  lookup_model_colors <- get_colors_labels()$colors
  lookup_model_labels <- get_colors_labels()$labels

  model_colors <- lookup_model_colors[model_names]
  model_labels <- lookup_model_labels[model_names]

  mean_diffs <- agg_auc_summary %>%
    filter(model %in% model_names) %>%
    pivot_wider(
      id_cols = time,
      names_from = model,
      values_from = auc
    ) %>%
    mutate(auc_difference = .data[[model_names[2]]] - .data[[model_names[1]]]) %>%
    select(time, auc_difference)

  # Find the year with the largest difference in AUC between the two models
  year <- mean_diffs$time[which.max(abs(mean_diffs$auc_difference))]

  ##### Figure 1B - Create individual panel for time = 7
  roc_summary <- pull_roc_summary(model_names, eval_times)
  roc_year <- roc_summary %>%
    filter(Time == year)

  model_labels <- lookup_model_labels[model_names]
  model_colors <- lookup_model_colors[model_names]

  p_year <- ggplot(roc_year, aes(x = FPR, y = mean_TPR, color = Model)) +
    geom_ribbon(
      aes(
        ymin = ci_lower,
        ymax = ci_upper,
        fill = Model
      ),
      alpha = 0.3,
      color = NA
    ) +
    geom_line(linewidth = 1) +
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
        # Create the text output using map2 from purrr
        paste(
          map2(model_names, model_labels, function(model_name, model_label) {
            sprintf(
              "%s: %.3f (%.3f-%.3f)",
              model_label,
              mean(auc_summary$auc[auc_summary$time == year & 
                              auc_summary$model == model_name]),
              mean(auc_summary$ci_lower[auc_summary$time == year & 
                                  auc_summary$model == model_name]),
              mean(auc_summary$ci_upper[auc_summary$time == year & 
                                  auc_summary$model == model_name])
            )
          }) %>% 
            paste(collapse = "\n")
        )

      )
    ) +
    coord_fixed() +
    get_publication_theme() +
    theme(
      legend.position = "bottom",
      plot.margin = margin(0.5, 0.5, 0.5, 0.5, "cm"),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5)
    )

  # return plot and year
  return(list(plot = p_year, year = year))
}

plot_brier_over_time <- function(metrics_list, model_names) {
  brier_results <- collate_metric(metrics_list, metric = "brier")
  brier_summary <- brier_results %>%
    group_by(model, time) %>%
  summarise(
    mean_metric = mean(metric, na.rm = TRUE),
    sd_metric = sd(metric, na.rm = TRUE),
    ymin = pmax(mean_metric - sd_metric, 0),
    ymax = pmin(mean_metric + sd_metric, 1),
    .groups = "drop"
  )
  brier_summary$model <- factor(brier_summary$model, levels=model_names)

  # Figure 1D - plot brier score over time
  brier_plot <- td_plot(brier_summary,
                        model_names=model_names,
                        metric = "brier",
                        all_models = F)

  return(brier_plot)
}

plot_concordance_over_time <- function(metrics_list, model_names) {
  concordance_results <- collate_metric(metrics_list, metric = "concordance")
  cc_sub <- concordance_results %>%
    filter(model %in% model_names) %>%
    mutate(fold = as.factor(fold),
          metric = metric)
  cc_sub$model <- factor(cc_sub$model, levels=model_names)

  concordance_summary <- concordance_results %>%
    group_by(model, time) %>%
    summarise(
      mean_metric = mean(metric, na.rm = TRUE),
      sd_metric = sd(metric, na.rm = TRUE),
      ymin = pmax(mean_metric - sd_metric, 0),
      ymax = pmin(mean_metric + sd_metric, 1),
      .groups = "drop"
    )
  concordance_summary$model <- factor(concordance_summary$model, levels=model_names)


  # Figure 1C - plot concordance over time
  concordance_plot <- td_plot(concordance_summary,
                              concordance_results,
                              model_names=model_names,
                              metric = "concordance")

  return(concordance_plot)
}


histogram_pvals <- function(results_table) {
  # Fig S1 - Histogram of p-values, bin size 0.05
  hist_pvalues <- ggplot(pvals_compare_trocs$all_results,
                        aes(x = p_value)) +
  geom_histogram(breaks = seq(0, 1, by = 0.05), 
                 fill = "#009292", 
                 alpha = 0.8,
                 color = "white") +  # Add white lines between bars
  geom_vline(xintercept = 0.05, linetype = "dashed", color = "red") +
  labs(
    x = "p-value",
    y = "Count"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 12),
    axis.text = element_text(size = 12),  # Increased from 10
    axis.title = element_text(size = 14),  # Increased from 11
    panel.grid.major = element_line(linewidth = 0.3),  # Thicker grid lines
    panel.grid.minor = element_line(linewidth = 0.15)  # Thicker minor grid lines
  )
  return(hist_pvalues)
}

plot_SeSpPPVNPV <- function(data, metric) {

  color_label_info <- get_colors_labels()
  colors <- color_label_info$colors
  labels <- color_label_info$labels

  # set y-axis label to capitalize first letter unless it's PPV or NPV
  if (!metric %in% c("ppv", "npv")) {
    y_label <- tools::toTitleCase(metric)
  } else {
    y_label <- toupper(tools::toTitleCase(metric))
  }

  ggplot(data, aes(x = time, y = get(paste0("mean_", metric)), color = model)) +
    geom_ribbon(
      aes(
        ymin = get(paste0("mean_", metric)) - get(paste0("sd_", metric)),
        ymax = get(paste0("mean_", metric)) + get(paste0("sd_", metric)),
        fill = model
      ),
      alpha = 0.2,
      color = NA
    ) +
    geom_line(linewidth = 1) +
    # Add white circles at each time point
    geom_point(aes(color = model), fill = "white", size = 3, shape = 21) +
    scale_color_manual(values = colors, labels = labels) +
    scale_fill_manual(values = colors, labels = labels) +
    labs(
      x = "Time (years)",
      y = y_label,
      color = "Model",
      fill = "Model"  # Add fill legend
    ) +
    theme_bw(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
      plot.subtitle = element_text(size = 14, hjust = 0.5),
      axis.title = element_text(face = "bold", size = 14),
      axis.text = element_text(size = 12),
      legend.title = element_text(face = "bold", size = 14),
      legend.text = element_text(size = 12),
      panel.grid.minor = element_blank(),
      panel.grid = element_blank(),
      panel.background = element_rect(fill = "white", color = NA),
      legend.position = "right"
    )
}

save_all_figures <- function(model_names, models_list, metrics_list, train_df_l, val_df_l, width, height, dpi, main_path) {
  ##### FIGURE 1A: AUC over time
  # # AUROC and CIs 
  # print("Plotting AUC over time")
  # auc_summary <- read_parquet(paste0(main_path, "auc_summary.parquet"))
  # agg_auc_summary <- aggregate(
  #   cbind(auc, ci_lower, ci_upper) ~ model + time,
  #   data = auc_summary,
  #   FUN = mean
  # )
  # auc_plot <- plot_auc_over_time(auc_summary, model_names)

  # # Save plots
  # ggsave(paste0(main_path, "final_auc_Over_Time.pdf"),
  #       plot = auc_plot,
  #       width = width,
  #       height = height,
  #       dpi = 300)

  # print("Plotting individual year ROC curves")
  # roc_plot <- plot_all_roc_curves(model_names, eval_times=seq(3, 7))
  # # Save the plot
  # ggsave(paste0(main_path, "ROC_curves_by_timepoint.pdf"),
  #       plot = roc_plot,
  #       width = width * 1.5,
  #       height = height,
  #       dpi = 300)


  # # Find the year with the largest difference in AUC between demographics_lancet and ptau_demographics_lancet
  # print("Plotting ROC curve for the year with the largest difference in AUC")
  # p_year <- plot_roc_biggest_year_difference(auc_summary,
  #                                           agg_auc_summary, 
  #                                           model_names,
  #                                           eval_times=seq(3, 7))
  # # Save plots
  # ggsave(paste0(main_path, "final_ROCcurve_", p_year$year, "years.pdf"),
  #   plot = p_year$plot,
  #   width = width,
  #   height = height,
  #   dpi = 300
  # )


  # ###### Figure 1D: BRIER SCORE - plot brier score over time
  # print("Plotting Brier score over time")
  # brier_plot <- plot_brier_over_time(metrics_list, model_names)

  # # Save plots
  # ggsave(paste0(main_path, "final_brier_Over_Time.pdf"),
  #   plot = brier_plot,
  #   width = width,
  #   height = height,
  #   dpi = 300
  # )

  # ##### Figure 1C: plot concordance over time
  # print("Plotting concordance over time")
  # concordance_plot <- plot_concordance_over_time(metrics_list, model_names)

  # # Save plots
  # ggsave(paste0(main_path, "final_concordance_Over_Time.pdf"),
  #   plot = concordance_plot,
  #   width = width,
  #   height = height,
  #   dpi = 300
  # )

  ########################################################
  # Sensitivity, Specificity, PPV, NPV
  # Function to calculate SeSpPPVNPV for a model and fold
  # Initialize list to store results

  ##### Figure 1E: Sensitivity, Specificity, PPV, NPV
  print("Plotting sensitivity, specificity, PPV, NPV")
  # Set up parallel processing
  df_sespppvnpv <- SeSpPPVNPV_summary(models_list, model_names, train_df_l, val_df_l)
  write_parquet(df_sespppvnpv, paste0(main_path, "sespppvnpv_summary.parquet"))

  # Create individual plots
  sensitivity_plot <- plot_SeSpPPVNPV(df_sespppvnpv, "sensitivity")
  specificity_plot <- plot_SeSpPPVNPV(df_sespppvnpv, "specificity")
  ppv_plot <- plot_SeSpPPVNPV(df_sespppvnpv, "ppv")
  npv_plot <- plot_SeSpPPVNPV(df_sespppvnpv, "npv")

  ggsave(
    paste0(main_path, "sensitivity_plot.pdf"),
    plot = sensitivity_plot,
    width = width * 1.2,
    height = 6,
    dpi = dpi
  )

  ggsave(
    paste0(main_path, "specificity_plot.pdf"),
    plot = specificity_plot,
    width = width * 1.2,
    height = 6,
    dpi = dpi
  )

  ggsave(
    paste0(main_path, "ppv_plot.pdf"),
    plot = ppv_plot,
    width = width * 1.2,
    height = 6,
    dpi = dpi
  )

  ggsave(
    paste0(main_path, "npv_plot.pdf"),
    plot = npv_plot,
    width = width * 1.2,
    height = 6,
    dpi = dpi
  )
}