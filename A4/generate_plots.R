library(ggplot2)
library(xtable)

setwd('/n/groups/patel/randy/rfb/code/A4/')

source("plot_figures.R")
source("metrics.R")

# Load fonts
library(extrafont)
extrafont::loadfonts()
font_import()
loadfonts(device = "postscript")

models_list <- qs::qread("../../tidy_data/A4/fitted_models.qs")
metrics_list <- qs::qread("../../tidy_data/A4/metrics.qs")

########################################################
# Bayes Information Criterion
for (model_name in names(models_list)) {
  for (fold in 1:5) {
    model <- models_list[[model_name]][[paste0("fold_", fold)]]
    print(paste0("Model: ", model_name, " Fold: ", fold, " BIC: ", BIC(model)))
  }
}

########################################################
# sensitivity, specificity, PPV, NPV
# timeROC:::confint.ipcwsurvivalROC(metrics_list$demographics_lancet$fold_1$troc)

# timeROC::plotAUCcurve(metrics_list$demographics_lancet$fold_1$troc)

# iterate over all cutpoints and store SeSpPPVNPV for each time point
risk_scores <- predict(models_list$demographics_lancet$fold_1,
                       val_df_l$fold_1_demographics_lancet)

se_sp_ppv_npv <- list()
for (cutpoint in seq(min(risk_scores), max(risk_scores), length.out = 200)) {
  se_sp_ppv_npv_results <- SeSpPPVNPV(cutpoint = cutpoint,
    T = val_df_l$fold_1_demographics_lancet$time,
    delta = val_df_l$fold_1_demographics_lancet$event,
    marker = risk_scores,
    cause = 1,
    weighting = "marginal",
    times = seq(3, 7)
  )
  se_sp_ppv_npv[[paste0("cutpoint_", cutpoint)]] <- se_sp_ppv_npv_results
}

youden_index_list <- list()
for (cutpoint in names(se_sp_ppv_npv)) {
  youden_index <- se_sp_ppv_npv[[cutpoint]]$TP + (1 - se_sp_ppv_npv[[cutpoint]]$FP) - 1
  youden_index_list[[cutpoint]] <- mean(youden_index)
}

# find the cutpoint that maximizes Youden's J index
best_cutpoint <- names(youden_index_list)[which.max(youden_index_list)]
print(se_sp_ppv_npv[[best_cutpoint]])


########################################################
# pull all ptau coefficients and p-values
# iterate over all models and folds in models_list to do this
ptau_coefs <- list()
ptau_pvals <- list()

# iterate over models that have ptau in them
for (model_name in names(models_list)) {
  if (grepl("ptau", model_name)) {
    for (fold in 1:5) {
      model <- models_list[[model_name]][[paste0("fold_", fold)]]
      ptau_coefs[[model_name]][[paste0("fold_", fold)]] <-
        exp(model$coefficients["ptau"])
      ptau_pvals[[model_name]][[paste0("fold_", fold)]] <-
        summary(model)$coefficients["ptau", "Pr(>|z|)"]
    }
  }
}

range(unlist(ptau_pvals))
mean(unlist(ptau_coefs))
sd(unlist(ptau_coefs))


########################################################  
# pull all centiloids coefficients and p-values
# iterate over all models and folds in models_list to do this
centiloids_coefs <- list()
centiloids_pvals <- list()

for (model_name in names(models_list)) {
  if (grepl("centiloids", model_name)) {
    for (fold in 1:5) {
      model <- models_list[[model_name]][[paste0("fold_", fold)]]
      centiloids_coefs[[model_name]][[paste0("fold_", fold)]] <-
        exp(model$coefficients["centiloids"])
      centiloids_pvals[[model_name]][[paste0("fold_", fold)]] <-
        summary(model)$coefficients["centiloids", "Pr(>|z|)"]
    }
  }
}

range(unlist(centiloids_pvals))
mean(unlist(centiloids_coefs))
sd(unlist(centiloids_coefs))


########################################################
# Fig S1 and Table S1
# Calculate p-values comparing AUCs between two models at each time point
# First combine timeROC objects from each fold for each model
demo_lancet_trocs <- list(
  metrics_list$demographics_lancet$fold_1$troc,
  metrics_list$demographics_lancet$fold_2$troc,
  metrics_list$demographics_lancet$fold_3$troc,
  metrics_list$demographics_lancet$fold_4$troc,
  metrics_list$demographics_lancet$fold_5$troc
)

ptau_demo_lancet_trocs <- list(
  metrics_list$ptau_demographics_lancet$fold_1$troc,
  metrics_list$ptau_demographics_lancet$fold_2$troc,
  metrics_list$ptau_demographics_lancet$fold_3$troc,
  metrics_list$ptau_demographics_lancet$fold_4$troc,
  metrics_list$ptau_demographics_lancet$fold_5$troc
)

centiloids_demo_lancet_trocs <- list(
  metrics_list$centiloids_demographics_lancet$fold_1$troc,
  metrics_list$centiloids_demographics_lancet$fold_2$troc,
  metrics_list$centiloids_demographics_lancet$fold_3$troc,
  metrics_list$centiloids_demographics_lancet$fold_4$troc,
  metrics_list$centiloids_demographics_lancet$fold_5$troc
)

ptau_centiloids_demo_lancet_trocs <- list(
  metrics_list$ptau_centiloids_demographics_lancet$fold_1$troc,
  metrics_list$ptau_centiloids_demographics_lancet$fold_2$troc,
  metrics_list$ptau_centiloids_demographics_lancet$fold_3$troc,
  metrics_list$ptau_centiloids_demographics_lancet$fold_4$troc,
  metrics_list$ptau_centiloids_demographics_lancet$fold_5$troc
)

# demo+lancet vs ptau+demo+lancet
pvals_compare_trocs <- compare_tvaurocs(demo_lancet_trocs,
                                        ptau_demo_lancet_trocs)

# print out all p-values
print("Summary of AUC differences and p-values by time point:")
print(range(pvals_compare_trocs$all_results$p_value))
print(mean(pvals_compare_trocs$all_results$p_value))
print(sd(pvals_compare_trocs$all_results$p_value))
print(median(pvals_compare_trocs$all_results$p_value))

# how many p-values are less than 0.05? out of how many total p-values?
sum(pvals_compare_trocs$all_results$p_value < 0.05) /
  length(pvals_compare_trocs$all_results$p_value)

# Create detailed results table
results_table <- pvals_compare_trocs$all_results
write.csv(results_table, "../../tidy_data/A4/auc_comparison_results_demo_lancet_vs_ptau_demo_lancet.csv",
          row.names = FALSE)

# Table S1 - pivot table of p-values where each row is a fold and each column is a time point
results_table_wide <- results_table %>%
  pivot_wider(id_cols = fold, names_from = time, values_from = p_value) %>%
  mutate(across(-fold, ~round(., digits = 4))) %>%
  mutate(across(-fold, ~ifelse(. < 0.05, paste0("\\textbf{", ., "}"), as.character(.))))

# latex table of results_table_wide with 4 decimal places
xtable_obj <- xtable(results_table_wide)
digits(xtable_obj) <- c(0, rep(4, ncol(results_table_wide)))  # Set digits for each column
print(xtable_obj, type = "latex", sanitize.text.function = function(x) x)  # Don't escape LaTeX commands

# Fig S1 - Histogram of p-values, bin size 0.05
hist_pvalues <- ggplot(pvals_compare_trocs$all_results,
                      aes(x = p_value)) +
  geom_histogram(breaks = seq(0, 1, by = 0.05), 
                 fill = "#009292", 
                 alpha = 0.8,
                 color = "white") +  # Add white lines between bars
  geom_vline(xintercept = 0.05, linetype = "dashed", color = "red") +
  labs(
    title = "Histogram of p-values comparing\nDemographics+Lancet vs. pTau-217+Demographics+Lancet\nacross folds and time points",
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

print(hist_pvalues)
ggsave("../../tidy_data/A4/pvalue_histogram_pTau217_Demo_Lancet_vs_Demo_Lancet.pdf",
       plot = hist_pvalues,
       width = 8,
       height = 6,
       dpi = 300)

# # Boxplots at each time point of AUC differences for pTau217+Demographics+Lancet vs Demographics+Lancet
# library(ggplot2)
# auc_plot <- ggplot(pvals_compare_trocs$all_results,
#        aes(x = factor(time), y = auc_diff)) +
#   geom_boxplot(fill = "#009292", alpha = 0.8) +
#   geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
#   labs(
#     title = "AUC Differences Between Models\n(pTau-217+Demographics+Lancet vs Demographics+Lancet)",
#     x = "Time (years)",
#     y = "AUC Difference"
#   ) +
#   theme_minimal() +
#   theme(
#     plot.title = element_text(hjust = 0.5, size = 12),
#     axis.text = element_text(size = 10),
#     axis.title = element_text(size = 11)
#   )

# print(auc_plot)
# ggsave(
#   "../../tidy_data/A4/auc_differences_boxplot.pdf",
#   plot = auc_plot,
#   width = 8,
#   height = 6,
#   dpi = 300
# )

# demo+lancet vs centiloids+demo+lancet
pvals_compare_trocs <- compare_tvaurocs(demo_lancet_trocs,
                                        centiloids_demo_lancet_trocs)

# print out all p-values
print("Summary of AUC differences and p-values by time point:")
print(range(pvals_compare_trocs$all_results$p_value))
print(mean(pvals_compare_trocs$all_results$p_value))
print(sd(pvals_compare_trocs$all_results$p_value))
print(median(pvals_compare_trocs$all_results$p_value))

# how many p-values are less than 0.05? out of how many total p-values?
sum(pvals_compare_trocs$all_results$p_value < 0.05) /
  length(pvals_compare_trocs$all_results$p_value)

# Create detailed results table
results_table <- pvals_compare_trocs$all_results
write.csv(results_table, "../../tidy_data/A4/auc_comparison_results_demo_lancet_vs_centiloids_demo_lancet.csv",
          row.names = FALSE)

# Table S2 - pivot table of p-values where each row is a fold and each column is a time point
results_table_wide <- results_table %>%
  pivot_wider(id_cols = fold, names_from = time, values_from = p_value) %>%
  mutate(across(-fold, ~round(., digits = 4))) %>%
  mutate(across(-fold, ~ifelse(. < 0.05, paste0("\\textbf{", ., "}"), as.character(.))))

# latex table of results_table_wide with 4 decimal places
library(xtable)
xtable_obj <- xtable(results_table_wide)
digits(xtable_obj) <- c(0, rep(4, ncol(results_table_wide)))  # Set digits for each column
print(xtable_obj, type = "latex", sanitize.text.function = function(x) x)  # Don't escape LaTeX commands

# Fig S2 - Histogram of p-values, bin size 0.05
hist_pvalues <- ggplot(pvals_compare_trocs$all_results,
                      aes(x = p_value)) +
  geom_histogram(breaks = seq(0, 1, by = 0.05), 
                 fill = "#009292", 
                 alpha = 0.8,
                 color = "white") +  # Add white lines between bars
  geom_vline(xintercept = 0.05, linetype = "dashed", color = "red") +
  labs(
    title = "Histogram of p-values comparing\nDemographics+Lancet vs. Centiloids+Demographics+Lancet\nacross folds and time points",
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

print(hist_pvalues)
ggsave("../../tidy_data/A4/pvalue_histogram_centiloids_Demo_Lancet_vs_Demo_Lancet.pdf",
       plot = hist_pvalues,
       width = 8,
       height = 6,
       dpi = 300)

########################################################
# Generate and save results for each metric
metrics_to_collect <- c("auc", "brier", "concordance")

for (metric in metrics_to_collect) {
  results <- collate_metric(metrics_list, metric)
  write_csv(
    results,
    paste0("../../tidy_data/A4/results_", metric, "_all_models.csv")
  )
}

# Thinking we only use this for Brier and Concordance because we're using the
# timeROC package to calculate the confidence intervals for AUROC
# AUC, Brier Score, and Concordance Over Time
# auc_results <- collate_metric(metrics_list, metric = "auc")

# plot auc over time
# auc_summary <- auc_results %>%
#   group_by(model, time) %>%
#   summarise(
#     mean_AUC = mean(metric, na.rm = TRUE),
#     sd_AUC = sd(metric, na.rm = TRUE),
#     ymin = pmax(mean_AUC - sd_AUC, 0),
#     ymax = pmin(mean_AUC + sd_AUC, 1),
#     .groups = "drop"
#   )

########################################################
# Function to extract AUROC and CIs for all folds

auc_summary <- read_parquet("../../tidy_data/A4/auc_summary.parquet")
agg_auc_summary <- aggregate(
  cbind(auc, ci_lower, ci_upper) ~ model + time,
  data = auc_summary,
  FUN = mean
)

# Table S3 - reshape auc_summary to wide format
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



##### FIGURE 1A: AUC over time

width <- 8
height <- 6
model_names <- c("demographics_lancet",
                  "ptau",
                  "ptau_demographics_lancet",
                  "centiloids",
                  "centiloids_demographics_lancet",
                  "ptau_centiloids_demographics_lancet")

auc_plot <- plot_auc_over_time(auc_summary, model_names)

# Save plots
ggsave("../../tidy_data/A4/final_auc_Over_Time.pdf",
       plot = auc_plot,
       width = width,
       height = height,
       dpi = 300)



roc_plot <- plot_all_roc_curves(model_names, eval_times=seq(3, 7))
# Save the plot
ggsave("../../tidy_data/A4/ROC_curves_by_timepoint.pdf",
       plot = roc_plot,
       width = width * 1.5,
       height = height,
       dpi = 300)


rr <- pull_roc_summary(model_names, seq(3, 7))
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


pauc_res <- calc_pAUC(rr, model_names, threshold = 0.25)
pauc_res$Model <- model_labels[pauc_res$Model]
pauc_res

# latex table of results_table_wide with 4 decimal places
xtable_obj <- xtable(pauc_res)
digits(xtable_obj) <- c(0, rep(4, ncol(pauc_res)))  # Set digits for each column
print(xtable_obj, type = "latex", sanitize.text.function = function(x) x)  # Don't escape LaTeX commands



### Individual year plots
# Find the year with the largest difference in AUC between demographics_lancet and ptau_demographics_lancet
p_year <- plot_roc_biggest_year_difference(auc_summary,
                                           agg_auc_summary, 
                                           model_names,
                                           eval_times=seq(3, 7))
# Save plots
ggsave(paste0("../../tidy_data/A4/final_ROCcurve_", p_year$year, "years.pdf"),
  plot = p_year$plot,
  width = width,
  height = height,
  dpi = 300
)


###### Figure 1D: BRIER SCORE - plot brier score over time
plot_brier_over_time <- function() {
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

brier_plot <- plot_brier_over_time()

# Display the plot
print(brier_plot)

# Save plots
ggsave("../../tidy_data/A4/final_brier_Over_Time.pdf",
  plot = brier_plot,
  width = width,
  height = height,
  dpi = 300
)

# find year with biggest difference in brier between the two models
# mean_diffs <- brier_summary %>%
#   filter(model %in% c("demographics_lancet", "ptau_demographics_lancet")) %>%
#   pivot_wider(
#     id_cols = time,
#     names_from = model,
#     values_from = mean_metric
#   ) %>%
#   mutate(brier_difference = ptau_demographics_lancet -
#         demographics_lancet) %>%
#   select(time, brier_difference) %>%
#   as.data.frame() # Convert to data.frame to avoid tibble's default rounding

# print(mean_diffs)

##### Figure 1C: plot concordance over time
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

# Display the plot
print(concordance_plot)

# Save plots
ggsave("../../tidy_data/A4/final_concordance_Over_Time.pdf",
  plot = concordance_plot,
  width = width,
  height = height,
  dpi = 300
)

# find year with biggest difference in concordance between the two models
mean_diffs <- concordance_summary %>%
  filter(model %in% c("demographics_lancet", "ptau_demographics_lancet")) %>%
  pivot_wider(
    id_cols = time,
    names_from = model,
    values_from = mean_metric
  ) %>%
  mutate(concordance_difference = ptau_demographics_lancet -
        demographics_lancet) %>%
  select(time, concordance_difference) %>%
  as.data.frame() # Convert to data.frame to avoid tibble's default rounding

print(mean_diffs)

########################################################
# Calculate calibration data
cal_data_avg <- calculate_calibration_data(models_list, val_df_l)

# Create calibration plots
plots <- calibration_plots(cal_data_avg, seq(3, 8), model_colors)
print(plots)

# Save plots
ggsave("../../tidy_data/A4/final_calibration_plots.pdf",
  plot = plots,
  width = 8,
  height = 6,
  dpi = 300
)

########################################################
# Decision curve analysis
# Collect predictions and create DCA data
dca_data_all <- list()
models_to_analyze <- c(
  "demographics",
  "demographics_no_apoe",
  "demographics_lancet",
  "ptau",
  "ptau_demographics_lancet"
)

# Collect predictions for each time point and model
for (t in seq(3, 8)) {
  for (fold in 0:4) {
    for (model_name in models_to_analyze) {
      model <- overwrite_na_coef_to_zero(
        models_list[[model_name]][[paste0("fold_", fold + 1)]]
      )

      pred_probs <- 1 - pec::predictSurvProb(
        model,
        newdata = val_df_l[[paste0("fold_", fold + 1, "_", model_name)]],
        times = t
      )

      val_data <- val_df_l[[paste0("fold_", fold + 1, "_", model_name)]]
      dca_data_all[[paste0("t", t, "_fold",
                           fold, "_", model_name)]] <- data.frame(
        fold = fold,
        time = t,
        model = model_name,
        tstop = val_data$tstop,
        event = val_data$event,
        pred_prob = pred_probs
      )
    }
  }
}

# Combine all DCA data
all_dca_data <- do.call(rbind, dca_data_all)

# Create and save DCA plots
dca_plots <- dca_plots(all_dca_data)
print(dca_plots)

ggsave("../../tidy_data/A4/final_DCA_Over_Time.pdf",
       plot = dca_plots,
       width = 8,
       height = 6,
       dpi = 300
)


########################################################
# Clinical risk reclassification
library(nricens) # For NRI calculations with survival data

find_events_within_horizon <- function(data, horizon, newdata) {
  # Create a mapping from ID to event status within horizon
  event_summary <- data %>%
    group_by(id) %>%
    summarize(
      event_occurred = any(event == 1),
      event_time = ifelse(event_occurred, min(tstop[event == 1]), Inf),
      within_horizon = event_occurred & event_time <= horizon
    )
  
  # Match the event status to the IDs in newdata
  event_status <- numeric(nrow(newdata))
  for (i in 1:nrow(newdata)) {
    id_match <- which(event_summary$id == newdata$id[i])
    if (length(id_match) > 0) {
      event_status[i] <- as.numeric(event_summary$within_horizon[id_match])
    }
  }

  return(event_status)
}

model1 <- models_list$demographics_lancet$fold_1
model2 <- models_list$ptau_demographics_lancet$fold_1
df <- train_df_l$fold_1_demographics_lancet
all.equal(train_df_l$fold_1_demographics_lancet,
          train_df_l$fold_1_ptau_demographics_lancet) # must be TRUE

# Define prediction time horizon
horizon <- 5 # years

# Create a baseline dataset for prediction (use data at a specific time point)
newdata <- df[df$tstart < 5, ] # or another relevant baseline

# Get predicted survival probabilities
pred_surv1 <- 1 - summary(survfit(model1, newdata = newdata),
                          times = horizon)$surv
pred_surv2 <- 1 - summary(survfit(model2, newdata = newdata),
                          times = horizon)$surv     

# Define risk categories (modify based on your clinical context)
risk_cats <- c(0, 0.05, 0.10, 0.20, 1)
risk_labels <- c("0-5%", "5-10%", "10-20%", ">20%")

# Categorize predicted risks
risk_cat1 <- cut(pred_surv1, breaks=risk_cats, labels=risk_labels, include.lowest=TRUE)
risk_cat2 <- cut(pred_surv2, breaks=risk_cats, labels=risk_labels, include.lowest=TRUE)

# Create reclassification table
reclass_table <- table(risk_cat1, risk_cat2)
print(reclass_table)

# Calculate percentage reclassified in each risk category
percent_reclass <- numeric(length(risk_labels))
names(percent_reclass) <- risk_labels

for (i in 1:length(risk_labels)) {
  cat <- risk_labels[i]
  n_total <- sum(reclass_table[i,])
  n_reclass <- n_total - reclass_table[i,i]
  percent_reclass[i] <- 100 * n_reclass / n_total
}
print(percent_reclass)

# Create a dataframe with predicted risks and actual outcomes
reclass_df <- data.frame(
  id = newdata$id,  # Use actual IDs from newdata
  risk_cat1 = risk_cat1,
  risk_cat2 = risk_cat2,
  pred_risk1 = pred_surv1,
  pred_risk2 = pred_surv2
)

# Add outcome information - pass newdata to ensure proper ID matching
event_within_horizon <- find_events_within_horizon(df, horizon, newdata)
reclass_df$event <- event_within_horizon

# Extract relevant time information from your original dataset
# We need to find the actual observed time (either event time or censoring time)
time_info <- df %>%
  group_by(id) %>%
  summarize(
    max_time = max(tstop),
    event_time = ifelse(any(event == 1), min(tstop[event == 1]), max_time),
    time = pmin(event_time, horizon)  # Censor at horizon for NRI analysis
  )

# Join this time information to your reclass_df
reclass_df <- left_join(reclass_df, select(time_info, id, time), by = "id")

# Now check that we have the time column
head(reclass_df)

# Check dimensions for nricens
print(paste("Length of pred_surv1:", length(pred_surv1)))
print(paste("Length of pred_surv2:", length(pred_surv2)))
print(paste("Length of reclass_df$time:", length(reclass_df$time)))
print(paste("Length of reclass_df$event:", length(reclass_df$event)))

# Now call nricens with the corrected data
nri_result <- nricens(
  time = reclass_df$time,      # Time to event or censoring
  event = reclass_df$event,    # Event indicator (1=event, 0=censored)
  p.std = pred_surv1,          # Predicted risks from standard model
  p.new = pred_surv2,          # Predicted risks from new model
  cut = risk_cats[-1],         # Cut points for risk categories
  t0 = horizon                 # Time horizon for prediction
)

# Calculate observed event rates within each cell of the reclassification table
observed_rates <- aggregate(event ~ risk_cat1 + risk_cat2,
                            data = reclass_df, FUN = mean)

# Format as a matrix similar to Table 3 in the paper
observed_matrix <- matrix(NA, nrow=length(risk_labels),
                          ncol=length(risk_labels))
rownames(observed_matrix) <- colnames(observed_matrix) <- risk_labels

for (i in 1:nrow(observed_rates)) {
  r <- which(risk_labels == observed_rates$risk_cat1[i])
  c <- which(risk_labels == observed_rates$risk_cat2[i])
  observed_matrix[r, c] <- round(100 * observed_rates$event[i], 1)
}
print(observed_matrix)


# Make sure all vectors have the same length before calling nricens
# Check dimensions of inputs
print(paste("Length of pred_surv1:", length(pred_surv1)))
print(paste("Length of pred_surv2:", length(pred_surv2)))
print(paste("Length of reclass_df$time:", length(reclass_df$time)))
print(paste("Length of reclass_df$event:", length(reclass_df$event)))

# Make sure we're using the same individuals for all vectors
# Create a complete data frame with all required variables
nri_data <- data.frame(
  id = newdata$id,
  pred_surv1 = pred_surv1[1,],
  pred_surv2 = pred_surv2[1,],
  time = reclass_df$time,
  event = reclass_df$event
)

# Remove any rows with NA values
nri_data <- nri_data[complete.cases(nri_data), ]

# Check the structure of our data
str(nri_data)

# Make sure time and event are properly formatted
# time should be numeric and event should be 0/1
nri_data$time <- as.numeric(nri_data$time)
nri_data$event <- as.numeric(nri_data$event)

# Try using the function with minimal parameters first
nri_result <- nricens(
  time = reclass_df$time,      # Time to event or censoring
  event = reclass_df$event,    # Event indicator (1=event, 0=censored)
  p.std = pred_surv1,          # Predicted risks from standard model
  p.new = pred_surv2,          # Predicted risks from new model
  cut = risk_cats[-1],         # Cut points for risk categories
  t0 = horizon                 # Time horizon for prediction
)

print(summary(nri_result))
print(nri_result$nri)
# Overall reclassification table
print(nri_result$rtab)

# Reclassification table for cases (events)
print(nri_result$rtab.case)

# Reclassification table for controls (non-events)
print(nri_result$rtab.ctrl)

# Plots
### Sankey Plot
library(networkD3)

# Prepare data for Sankey diagram
links <- data.frame(
  source = character(),
  target = character(),
  value = numeric(),
  group = character()
)

# Get risk categories
risk_cats_labels <- c("< 5%", "5-10%", "10-20%", "20-100%", "≥ 100%")

# Create links from the reclassification tables
for (i in 1:nrow(nri_result$rtab)) {
  for (j in 1:ncol(nri_result$rtab)) {
    if (nri_result$rtab[i,j] > 0) {
      # For all individuals
      links <- rbind(links, data.frame(
        source = paste("Old:", risk_cats_labels[i]),
        target = paste("New:", risk_cats_labels[j]),
        value = nri_result$rtab[i,j],
        group = "All"
      ))
    }
  }
}

# Create Sankey diagram
nodes <- data.frame(
  name = unique(c(links$source, links$target))
)

links$source <- match(links$source, nodes$name) - 1
links$target <- match(links$target, nodes$name) - 1

sankeyNetwork(Links = links, Nodes = nodes,
              Source = "source", Target = "target",
              Value = "value", NodeID = "name",
              LinkGroup = "group", fontSize = 12)

### Risk Shift Plot

# Create a proper data frame
plot_df <- data.frame(
  Model1 = nri_result$p.std[1, ],
  Model2 = nri_result$p.new[1, ],
  Event = ifelse(rep(1:2, length.out = length(nri_result$p.std)) == 1, "Case", "Control")
)

# Check the structure
head(plot_df)

# Create the plot and explicitly print it
rsp <- ggplot(plot_df, aes(x = Model1, y = Model2, color = Event)) +
  geom_point(alpha = 0.3) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  scale_color_manual(values = c("Case" = "red", "Control" = "blue")) +
  labs(x = "Risk from Model 1", y = "Risk from Model 2",
       title = "Change in Predicted Risk Between Models") +
  theme_minimal() +
  coord_fixed(ratio = 1)

# Explicitly print the plot
print(rsp)

# Also save the plot to ensure it's being generated
ggsave("../../tidy_data/A4/risk_shift_plot.pdf",
       plot = rsp,
       width = 8,
       height = 6,
       dpi = 300)

### NRI components
nri_components <- data.frame(
  Component = c("Overall NRI", "Events (NRI+)", "Non-events (NRI-)"),
  Estimate = c(nri_result$nri[1,1], nri_result$nri[2,1], nri_result$nri[3,1]),
  Lower = c(nri_result$nri[1,2], nri_result$nri[2,2], nri_result$nri[3,2]),
  Upper = c(nri_result$nri[1,3], nri_result$nri[2,3], nri_result$nri[3,3])
)

# Create bar plot with error bars
nri_components_plot <- ggplot(nri_components, aes(x = Component, y = Estimate, fill = Component)) +
  geom_bar(stat = "identity", width = 0.6) +
  geom_errorbar(aes(ymin = Lower, ymax = Upper), width = 0.2) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(title = "Net Reclassification Improvement Components",
       y = "NRI Value", x = "") +
  theme_minimal() +
  theme(legend.position = "none")

# Explicitly print the plot
print(nri_components_plot)

# Also save the plot to ensure it's being generated
ggsave("../../tidy_data/A4/nri_components_plot.pdf",
       plot = nri_components_plot,
       width = 8,
       height = 6,
       dpi = 300)


### Heatmap of reclassification table
# Convert reclassification table to data frame
reclass_df <- as.data.frame(as.table(nri_result$rtab))
names(reclass_df) <- c("Old", "New", "Count")

# Calculate percentages for text labels
total_count <- sum(reclass_df$Count)
reclass_df$Percent <- round(100 * reclass_df$Count / total_count, 1)
reclass_df$Label <- paste0(reclass_df$Count, "\n(", reclass_df$Percent, "%)")

# Create publication-quality heat map
heatmap_reclass_df <- ggplot(reclass_df, aes(x = New, y = Old, fill = Count)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = Label), 
            # Adjust text color based on background brightness for better contrast
            color = ifelse(reclass_df$Count > mean(reclass_df$Count) * 1.5, "white", "black"),
            fontface = "bold", size = 4) +  # Increased text size
  scale_fill_viridis_c(option = "mako",    # Changed to "mako" for better contrast
                       trans = "log", 
                       name = "Number of\nPatients",
                       guide = guide_colorbar(title.position = "top",
                                            barwidth = 10, 
                                            barheight = 0.5)) +
  labs(title = "Risk Reclassification Matrix",
       subtitle = "Demographics + Lancet vs. pTau-217 + Demographics + Lancet",
       x = "Risk Category with pTau-217 Model", 
       y = "Risk Category with Demographics Model",
       caption = "Numbers show count and percentage of total patients") +
  scale_x_discrete(position = "top") +
  theme_minimal(base_family = "Helvetica") +
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle = element_text(size = 11, hjust = 0.5, margin = margin(b = 15)),
    plot.caption = element_text(size = 9, hjust = 1, margin = margin(t = 10)),
    axis.title = element_text(face = "bold", size = 12),
    axis.text = element_text(size = 10, face = "bold"),
    legend.position = "right",
    legend.title = element_text(size = 10, face = "bold"),
    legend.text = element_text(size = 9),
    panel.grid = element_blank(),
    panel.border = element_rect(fill = NA, color = "gray30", linewidth = 0.5),
    plot.margin = margin(20, 20, 20, 20)
  ) +
  # Add diagonal line highlighting to show unchanged classifications
  geom_tile(data = subset(reclass_df, Old == New), 
            aes(x = New, y = Old), 
            fill = NA, color = "black", linewidth = 1.2)

# Explicitly print the plot
print(heatmap_reclass_df)

# Also save the plot to ensure it's being generated
ggsave("../../tidy_data/A4/risk_reclassification_heatmap.pdf",
       plot = heatmap_reclass_df,
       width = 8, height = 7, dpi = 300)

### Observed Event Rate Plot
# Create data frame with observed event rates
event_rates <- data.frame(
  Old = character(),
  New = character(),
  Count = numeric(),
  EventRate = numeric()
)

risk_levels <- colnames(nri_result$rtab)

for (i in 1:length(risk_levels)) {
  for (j in 1:length(risk_levels)) {
    total = nri_result$rtab[i,j]
    if (total > 0) {
      events = nri_result$rtab.case[i,j]
      event_rates <- rbind(event_rates, data.frame(
        Old = risk_levels[i],
        New = risk_levels[j],
        Count = total,
        EventRate = events/total*100
      ))
    }
  }
}

# Plot event rates
event_rate_plot <- ggplot(event_rates, aes(x = New, y = Old, fill = EventRate)) +
  geom_tile() +
  geom_text(aes(label = sprintf("%.1f%%", EventRate)), 
            color = ifelse(event_rates$EventRate > 50, "white", "black")) +
  scale_fill_gradient(low = "white", high = "red") +
  labs(title = "Observed Event Rate by Risk Reclassification",
       x = "New Model Risk Category", 
       y = "Original Model Risk Category",
       fill = "Event Rate (%)") +
  theme_minimal()

# Explicitly print the plot
print(event_rate_plot)

# Also save the plot to ensure it's being generated
ggsave("../../tidy_data/A4/event_rate_plot.pdf",
       plot = event_rate_plot,
       width = 8,
       height = 6,
       dpi = 300)

### Decision Curve Analysis
library(rmda)

# Create decision curve analysis dataframe
dca_df <- data.frame(
  # event = reclass_df$event,
  event = ifelse(rep(1:2, length.out = length(nri_result$p.std)) == 1, 0, 1),
  p.std = nri_result$p.std[1,],  # Extract first row of matrix
  p.new = nri_result$p.new[1,]   # Extract first row of matrix
)
# dca_df$event <- as.factor(dca_df$event)
# Verify the data is properly formatted
print("Data structure:")
str(dca_df)
print(paste("Number of events:", sum(dca_df$event)))
print(paste("Range of p.std:", min(dca_df$p.std), "to", max(dca_df$p.std)))
print(paste("Range of p.new:", min(dca_df$p.new), "to", max(dca_df$p.new)))


# Create decision curve
decision_curve <- decision_curve(
  formula = event ~ p.std + p.new,
  data = dca_df,
  thresholds = seq(0, 0.5, by = 0.01)
)


# Plot decision curve
decision_curve_plot <- plot_decision_curve(decision_curve, 
                    curve.names = c("Standard Model", "New Model"),
                    xlab = "Threshold Probability (%)",
                    ylab = "Net Benefit",
                    cost.benefit.axis = TRUE,
                    col = c("blue", "red"),
                    confidence.intervals = FALSE)

# Explicitly print the plot
print(decision_curve_plot)

# Also save the plot to ensure it's being generated
ggsave("../../tidy_data/A4/decision_curve_plot.pdf",
       plot = decision_curve_plot,
       width = 8,
       height = 6,
       dpi = 300)


# ########################################################
# # ROC and Prediction Error Curves
# # Initialize lists to store ROC data
# roc_data_all <- list()

# # Create consistent time points
# eval_times <- seq(3, 8)

# # Define models to analyze
# models_to_analyze <- c(
#   "demographics_lancet",
#   "ptau",
#   "ptau_demographics_lancet"
# )

# # Update model colors
# model_colors <- c(
#   "demographics_lancet" = "#E69F00",    # orange
#   "ptau" = "#CC79A7",                   # pink
#   "ptau_demographics_lancet" = "#009292" # turquoise
# )

# # Initialize dataframe to store ROC curves
# all_roc_curves <- data.frame()

# for (fold in 0:4) {
#   for (model_name in models_to_analyze) {
#     # Get timeROC object from metrics list
#     troc <- metrics_list[[model_name]][[paste0("fold_", fold + 1)]]$troc
    
#     # Extract ROC curves for each time point
#     for (t in eval_times) {
#       # Get ROC curve data for this time
#       idx <- which(troc$times == t)
#       if (length(idx) > 0) {
#         roc_data <- data.frame(
#           FPR = troc$FP[, idx],
#           TPR = troc$TP[, idx],
#           Time = t,
#           Model = model_name,
#           fold = fold
#         )
#         all_roc_curves <- rbind(all_roc_curves, roc_data)
#       }
#     }
#   }
# }

# # Calculate summary statistics
# roc_summary <- all_roc_curves %>%
#   # First, bin FPR values to create discrete groups
#   mutate(FPR_bin = round(FPR, digits = 2)) %>%
#   group_by(Model, Time, FPR_bin) %>%
#   summarise(
#     mean_TPR = mean(TPR, na.rm = TRUE),
#     sd_TPR = sd(TPR, na.rm = TRUE),
#     FPR = mean(FPR_bin),  # Use the mean FPR within each bin
#     .groups = "drop"
#   ) %>%
#   # Remove any remaining NAs
#   filter(!is.na(sd_TPR))

# # Plot ROC curves with smoothing and uncertainty
# p5 <- ggplot(roc_summary, aes(x = FPR, y = mean_TPR, color = Model)) +
#   geom_ribbon(aes(
#     ymin = pmax(mean_TPR - sd_TPR, 0),
#     ymax = pmin(mean_TPR + sd_TPR, 1),
#     fill = Model
#   ), alpha = 0.2, color = NA) +
#   geom_line(linewidth = 1) +
#   geom_abline(
#     slope = 1, intercept = 0,
#     linetype = "dashed", color = "gray50"
#   ) +
#   scale_color_manual(
#     values = model_colors,
#     labels = c(
#       "Demo + Lancet",
#       "pTau217",
#       "Demo + pTau217\n+ Lancet"
#     )
#   ) +
#   scale_fill_manual(
#     values = model_colors,
#     labels = c(
#       "Demo + Lancet",
#       "pTau217",
#       "Demo + pTau217\n+ Lancet"
#     )
#   ) +
#   facet_wrap(~Time,
#     labeller = labeller(Time = function(x) {
#       sprintf("%s years", x)
#     })
#   ) +
#   labs(
#     x = "False Positive Rate",
#     y = "True Positive Rate",
#     title = "Dynamic ROC Curves",
#     subtitle = "At Different Follow-up Times"
#   ) +
#   coord_equal() +
#   get_publication_theme() +
#   theme(
#     panel.spacing = unit(1, "cm"),
#     axis.text.x = element_text(angle = 0, hjust = 0.5, size = 8),
#     plot.margin = margin(0.5, 0.5, 0.5, 0.5, "cm")
#   )

# print(p5)

# # Save plots
# ggsave("../../tidy_data/A4/final_ROCcurves_Over_Time.pdf",
#        plot = p5,
#        width = 14,
#        height = 6,
#        dpi = 300)

# if (lancet) {
#   year <- 3
# } else {
#   year <- 7
# }

# # Create individual panel for time = 7
# roc_year <- roc_summary %>%
#   filter(Time == year)

# p5_year <- ggplot(roc_year, aes(x = FPR, y = mean_TPR, color = Model)) +
#   geom_ribbon(
#     aes(
#       ymin = pmax(mean_TPR - sd_TPR, 0),
#       ymax = pmin(mean_TPR + sd_TPR, 1),
#       fill = Model
#     ),
#     alpha = 0.2,
#     color = NA
#   ) +
#   geom_smooth(se = FALSE, method = "loess", span = 0.2, linewidth = 1) +
#   geom_abline(
#     slope = 1, intercept = 0,
#     linetype = "dashed", color = "gray50"
#   ) +
#   scale_color_manual(values = model_colors) +
#   scale_fill_manual(values = model_colors) +
#   labs(
#     x = "False Positive Rate",
#     y = "True Positive Rate",
#     title = paste(
#       year, "years\n",
#       auc_stats$Baseline[auc_stats$Time == year], "\n",
#       auc_stats$Biomarker[auc_stats$Time == year]
#     )
#   ) +
#   coord_equal() +
#   get_publication_theme() +
#   theme(
#     legend.position = "bottom",
#     plot.margin = margin(0.5, 0.5, 0.5, 0.5, "cm"),
#     panel.border = element_rect(color = "black", fill = NA, linewidth = 1)
#   )

# print(p5_year)

# # Save plots
# ggsave("../../tidy_data/A4/final_ROCcurve_7years.pdf",
#   plot = p5_year,
#   width = 6,
#   height = 6,
#   dpi = 300
# )

# ########################################################

# Function to calculate SeSpPPVNPV for a model and fold
calculate_SeSpPPVNPV <- function(model, val_data, times) {
  risk_scores <- predict(model, val_data)
  
  # Find optimal cutpoint using Youden's index
  se_sp_ppv_npv <- list()
  for (cutpoint in seq(min(risk_scores), max(risk_scores), length.out = 200)) {
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

# Initialize list to store results
metrics_over_time <- list()

# Calculate metrics for each model and fold
for (model_name in c("demographics",
                     "lancet",
                     "demographics_lancet",
                     "ptau_demographics_lancet",
                     "centiloids_demographics_lancet",
                     "ptau_centiloids_demographics_lancet")) {#names(models_list)) {
  metrics_over_time[[model_name]] <- list()
  for (fold in 1:5) {
    model <- models_list[[model_name]][[paste0("fold_", fold)]]
    val_data <- val_df_l[[paste0("fold_", fold, "_", model_name)]]
    
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
    se_sensitivity = sd(sensitivity, na.rm = TRUE) / sqrt(n()),
    mean_specificity = mean(specificity, na.rm = TRUE),
    se_specificity = sd(specificity, na.rm = TRUE) / sqrt(n()),
    mean_ppv = mean(ppv, na.rm = TRUE),
    se_ppv = sd(ppv, na.rm = TRUE) / sqrt(n()),
    mean_npv = mean(npv, na.rm = TRUE),
    se_npv = sd(npv, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

# Create plots for each metric
plot_metric <- function(data, metric, title) {
  ggplot(data, aes(x = time, y = get(paste0("mean_", metric)), color = model)) +
    geom_line(linewidth = 1) +
    geom_ribbon(
      aes(
        ymin = get(paste0("mean_", metric)) - get(paste0("se_", metric)),
        ymax = get(paste0("mean_", metric)) + get(paste0("se_", metric)),
        fill = model
      ),
      alpha = 0.2,
      color = NA
    ) +
    # Add white circles at each time point
    geom_point(aes(fill = model), color = "white", size = 3, shape = 21) +
    labs(
      title = title,
      x = "Time (years)",
      y = metric,
      color = "Model",
      fill = "Model"  # Add fill legend
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5),
      legend.position = "bottom"
    )
}

# Create individual plots
sensitivity_plot <- plot_metric(metrics_summary, "sensitivity", "Sensitivity Over Time")
specificity_plot <- plot_metric(metrics_summary, "specificity", "Specificity Over Time")
ppv_plot <- plot_metric(metrics_summary, "ppv", "Positive Predictive Value Over Time")
npv_plot <- plot_metric(metrics_summary, "npv", "Negative Predictive Value Over Time")

# Combine plots into a single figure
combined_plot <- gridExtra::grid.arrange(
  sensitivity_plot,
  specificity_plot,
  ppv_plot,
  npv_plot,
  ncol = 2,
  nrow = 2
)

# Save the combined plot
ggsave(
  "../../tidy_data/A4/diagnostic_metrics_over_time.pdf",
  plot = combined_plot,
  width = 12,
  height = 10,
  dpi = 300
)

