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

td_plot <- function(summary_df, model_colors, metric = "auc") {
  model_colors <- c("Baseline" = "#287271", "Biomarker" = "#B63679")

  if (metric == "auc") {
    ggplot(summary_df,
           aes(x = time,
               y = mean_AUC,
               color = model,
               fill = model)) +
      geom_ribbon(aes(ymin = ymin, ymax = ymax),
                  alpha = 0.3,
                  color = NA,
                  show.legend = FALSE) +
      geom_line(linewidth = 1.2) +
      geom_point(size = 3, shape = 21, fill = "white") +
      scale_color_manual(values = model_colors) +
      scale_fill_manual(values = model_colors) +
      labs(
        title = "Time-varying AUROC",
        x = "Follow-up Time (years)",
        y = "AUROC",
        color = "Model",
        fill = "Model"
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
        legend.position = "right"
      ) +
      scale_y_continuous(limits = c(0.5, 0.8))
  } else if (metric == "brier") {
    ggplot(summary_df,
           aes(x = time,
               y = mean_brier,
               color = model,
               fill = model)) +
      geom_ribbon(aes(ymin = ymin, ymax = ymax),
                  alpha = 0.3,
                  color = NA,
                  show.legend = FALSE) +
      geom_line(linewidth = 1.2) +
      geom_point(size = 3, shape = 21, fill = "white") +
      scale_color_manual(values = model_colors) +
      scale_fill_manual(values = model_colors) +
      labs(
        title = "Time-varying Brier score",
        x = "Follow-up Time (years)",
        y = "Brier Score",
        color = "Model",
        fill = "Model"
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
        legend.position = "right"
      )
  } else if (metric == "concordance") {
    ggplot(summary_df,
           aes(x = time,
               y = mean_concordance,
               color = model,
               fill = model)) +
      geom_ribbon(aes(ymin = ymin, ymax = ymax),
                  alpha = 0.3,
                  color = NA,
                  show.legend = FALSE) +
      geom_line(linewidth = 1.2) +
      geom_point(size = 3, shape = 21, fill = "white") +
      scale_color_manual(values = model_colors) +
      scale_fill_manual(values = model_colors) +
      labs(
        title = "Time-varying Concordance",
        x = "Follow-up Time (years)",
        y = "Concordance",
        color = "Model",
        fill = "Model"
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
        legend.position = "right"
      ) +
      scale_y_continuous(limits = c(0.5, 0.85))
  }
}

# Modified plotting function to include confidence intervals
calibration_plots <- function(cal_data, times, model_colors) {

  publication_theme <- get_publication_theme()
  model_colors <- c("Baseline" = "#287271", "Biomarker" = "#B63679")

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
      scale_color_manual(values = model_colors, name = "Model") +
      scale_fill_manual(values = model_colors, name = "Model") +
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

dca_plots <- function(baseline_model, biomarker_model,
                      data, times, model_colors) {
  plots <- list()
  final_plot <- NULL

  for (t in times) {
    is_leftmost <- as.numeric(t) %in% c(3, 5, 7)
    is_bottom <- as.numeric(t) >= 7
    is_last <- t == 8

    # First run with wide range to find valid threshold range
    temp_dca_base <- stdca(data = data, outcome = "event", ttoutcome = "tstop",
                           timepoint = t, predictors = "baseline_pred",
                           xstart = 0, xstop = 1,
                           probability = FALSE, harm = NULL, graph = FALSE)
    temp_dca_bio <- stdca(data = data, outcome = "event", ttoutcome = "tstop",
                          timepoint = t, predictors = "biomarker_pred",
                          xstart = 0, xstop = 1,
                          probability = FALSE, harm = NULL, graph = FALSE)

    # Find last valid threshold
    max_thresh <- min(
      max(
        temp_dca_base$net.benefit$threshold[
          !is.na(temp_dca_base$net.benefit$baseline_pred)
        ]
      ),
      max(
        temp_dca_bio$net.benefit$threshold[
          !is.na(temp_dca_bio$net.benefit$biomarker_pred)
        ]
      )
    )

    # Rerun with adaptive xstop
    dca_baseline <- stdca(data = data, outcome = "event", ttoutcome = "tstop",
                          timepoint = t, predictors = "baseline_pred",
                          xstart = 0, xstop = max_thresh,
                          probability = FALSE, harm = NULL, graph = FALSE)

    dca_biomarker <- stdca(data = data, outcome = "event", ttoutcome = "tstop",
                           timepoint = t, predictors = "biomarker_pred",
                           xstart = 0, xstop = max_thresh,
                           probability = FALSE, harm = NULL, graph = FALSE)

    current_plot <- ggplot() +
      geom_line(data = data.frame(
        x = dca_baseline$net.benefit$threshold,
        y = dca_baseline$net.benefit$none
      ),
      aes(x = x, y = y, linetype = "Treat None"), color = "gray50") +
      geom_line(data = data.frame(
        x = dca_baseline$net.benefit$threshold,
        y = dca_baseline$net.benefit$all
      ),
      aes(x = x, y = y, linetype = "Treat All"), color = "gray50") +
      geom_line(data = data.frame(
        x = dca_baseline$net.benefit$threshold,
        y = dca_baseline$net.benefit$baseline_pred
      ),
      aes(x = x, y = y, color = "Baseline"), linewidth = 1) +
      geom_line(data = data.frame(
        x = dca_biomarker$net.benefit$threshold,
        y = dca_biomarker$net.benefit$biomarker_pred
      ),
      aes(x = x, y = y, color = "Biomarker"), linewidth = 1) +
      scale_color_manual(values = model_colors, name = "Model") +
      scale_linetype_manual(
        values = c("Treat None" = "dashed", "Treat All" = "dotted"),
        name = "Linetype"
      ) +
      scale_y_continuous(limits = c(-0.05, NA)) +
      labs(
        x = if (is_bottom) "Threshold Probability" else "",
        y = if (is_leftmost) "Net Benefit" else "",
        title = paste(t, "years")
      )

    plots[[as.character(t)]] <- current_plot

    if (t == tail(times, 1)) {
      final_plot <- current_plot
    }
  }

  results <- wrap_plots(plots, ncol = 2) +
    plot_layout(guides = "collect") +
    plot_annotation(
      theme = theme(
        legend.position = "bottom",
        legend.box = "horizontal"
      )
    )

  return(list(
    all_plots = results,
    final_plot = final_plot
  ))
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

create_additional_figures <- function(baseline_model, biomarker_model,
                                      data, times) {
  publication_theme <- get_publication_theme()
  model_colors <- c("Baseline" = "#287271", "Biomarker" = "#B63679")

  # 1. Cumulative/Dynamic ROC Curves at different time points  
  # ROC curve for biomarker model
  roc_biomarker <- timeROC(
    T = data$tstop,
    delta = data$event,
    marker = predict(biomarker_model, newdata = data, type = "lp"),
    cause = 1,
    times = times,
    iid = TRUE,
    ROC = TRUE
  )

  # ROC curve for baseline model
  roc_baseline <- timeROC(
    T = data$tstop,
    delta = data$event,
    marker = predict(baseline_model, newdata = data, type = "lp"),
    cause = 1,
    times = times,
    iid = TRUE,
    ROC = TRUE
  )

  # Create data frames with consistent column names
  biomarker_df <- data.frame(
    FPR = as.vector(roc_biomarker$FP),
    TPR = as.vector(roc_biomarker$TP),
    Model = "Biomarker",
    Time = rep(times, each = length(roc_biomarker$FP) / length(times))
  )
  
  baseline_df <- data.frame(
    FPR = as.vector(roc_baseline$FP),
    TPR = as.vector(roc_baseline$TP),
    Model = "Baseline",
    Time = rep(times, each = length(roc_baseline$FP) / length(times))
  )

  # Combine the data frames
  roc_data <- rbind(biomarker_df, baseline_df)

  p5 <- ggplot(roc_data, aes(x = FP, y = TP,
                             color = Model,
                             linetype = factor(Time))) +
    geom_line(linewidth = 1) +
    geom_abline(slope = 1, intercept = 0,
                linetype = "dashed", color = "gray50") +
    scale_color_manual(values = model_colors) +
    scale_linetype_discrete(name = "Time (Years)") +
    labs(
      x = "False Positive Rate",
      y = "True Positive Rate",
      title = "Dynamic ROC Curves", 
      subtitle = "At Different Follow-up Times"
    ) +
    coord_equal() +
    publication_theme

  # 2. Integrated Prediction Error
  # Create consistent time points from 3-8 years
  eval_times <- seq(3, 8)

  # Calculate prediction error curves with consistent time points
  pe <- tryCatch({
    pec(
      list("Baseline" = baseline_model, "Biomarker" = biomarker_model),
      data = data,
      times = eval_times,  # Use consistent eval_times
      exact = FALSE,
      reference = TRUE,
      splitMethod = "none",
      formula = Surv(tstop, event) ~ 1,
      start = 3,
      verbose = FALSE
    )
  }, error = function(e) {
    warning("Prediction error calculation failed, returning NULL")
    NULL
  })
  
  # Only create the plot if pe calculation succeeded
  if (!is.null(pe)) {
    pe_data <- data.frame(
      time = rep(pe$time, 2),
      error = c(pe$AppErr$Baseline, pe$AppErr$Biomarker),
      Model = factor(rep(c("Baseline", "Biomarker"),
                        each = length(pe$time)))
    )
    
    p6 <- ggplot(pe_data, aes(x = time, y = error, color = Model)) +
      geom_line(linewidth = 1) +
      scale_color_manual(
        values = model_colors,
        labels = c("Baseline", "Biomarker")
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

  # Create combined plot with new figures
  combined_additional <- (p5 + p6) +
    plot_annotation(
      title = "Additional Model Performance Metrics",
      theme = theme(
        plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
      )
    )

  return(list(
    dynamic_roc = p5,
    prediction_error = p6,
    combined_additional = combined_additional
  ))
}