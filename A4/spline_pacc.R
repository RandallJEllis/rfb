# spline model for pacc
library(mgcv)
library(arrow)
library(this.path)
library(tidyverse)
library(splines)
library(this.path)
setwd(dirname(this.path()))
source('plot_figures.R')

rsq <- function (x, y) cor(x, y) ^ 2
calculate_adj_r2 <- function(r_squared, n, p) {
  # n = number of observations
  # p = number of predictors (excluding intercept)
  return(1 - (1 - r_squared) * ((n - 1) / (n - p - 1)))
}

preprocess_df <- function(df, pacc_col, pacc, habits, psychwell, vitals, centiloids){
    df$age2 <- df$age_centered^2

    df <- df %>%
        group_by(id) %>%
        slice(1) %>%
        ungroup()

    # merge with pacc and other datasets
    pacc <- pacc[pacc$VISCODE >= 6, ]
    df <- merge(df, pacc, by.x = 'id', by.y = 'BID', all.x = TRUE)
    # convert VISCODE to days
    # VISCODE 6 = 0
    # for each VISCODE, add 30 days for each integer above 6
    df <- df %>%
        mutate(pacc_time = case_when(
            VISCODE.y == 6 ~ 0,
            VISCODE.y > 6 ~ (VISCODE.y - 6) * (30/365.25)
        ))

    df <- merge(df, habits[habits$VISCODE == 1, c('BID', "SMOKE", "ALCOHOL", "SUBUSE",
                                                            "AEROBIC", "WALKING")], by.x = 'id', by.y = 'BID', all.x = TRUE)
    df <- merge(df, psychwell[psychwell$VISCODE == 6, c('BID', "GDTOTAL", "STAITOTAL")], by.x = 'id', by.y = 'BID', all.x = TRUE)
    df <- merge(df, vitals[vitals$VISCODE == 6, c('BID', "VSBPSYS", "VSBPDIA", "BMI")], by.x = 'id', by.y = 'BID', all.x = TRUE)
    df <- merge(df, centiloids[, c('BID', "AMYLCENT")], by.x = 'id', by.y = 'BID', all.x = TRUE)

    df <- df[!is.na(df[[pacc_col]]), ]

    df <- df[complete.cases(df[, c('ptau_boxcox', 'age_centered', 'SEX', 'educ_z', 'APOEGN', "SMOKE", "ALCOHOL", "SUBUSE",
                                    "AEROBIC", "WALKING", "GDTOTAL", "STAITOTAL", "VSBPSYS", "VSBPDIA", "BMI", "AMYLCENT")]), ]

    # drop ptau and educ columns because we're going to use ptau_boxcox
    df <- df %>% select(-c(ptau, educ))
    df <- df %>% rename(ptau = ptau_boxcox,
                        sex = SEX,
                        educ = educ_z,
                        apoe = APOEGN,
                        smoke = SMOKE,
                        alcohol = ALCOHOL,
                        subuse = SUBUSE,
                        aerobic = AEROBIC,
                        walking = WALKING,
                        gdtotal = GDTOTAL,
                        staital = STAITOTAL,
                        vsbsys = VSBPSYS,
                        vsdia = VSBPDIA,
                        bmi = BMI,
                        centiloids = AMYLCENT)

    # set E3/E3 as the reference level for apoe
    df$apoe <- as.factor(df$apoe)
    df$apoe <- relevel(df$apoe, ref = "E3/E3")

    return(df)
}

# load data
pacc <- read.csv('../../raw_data/A4_oct302024/clinical/Derived Data/PACC.csv')
lancet_load_path = "../../tidy_data/A4/"
habits <- read_parquet(paste0(lancet_load_path, "habits.parquet"))
habits$SUBUSE <- as.factor(habits$SUBUSE)
psychwell <- read_parquet(paste0(lancet_load_path, "psychwell.parquet"))
vitals <- read_parquet(paste0(lancet_load_path, "vitals.parquet"))
centiloids <- read_parquet(paste0(lancet_load_path, "centiloids.parquet"))

 # For habits, psychwell, vitals, perform last observation carried forward (LOCF) within each subject
fill_na_within_group <- function(df) {
  df %>%
    group_by(BID) %>%
    fill(everything(), .direction = "down") %>%
    fill(everything(), .direction = "up") %>%
    ungroup()
}

# Apply the function to each dataset
habits <- fill_na_within_group(habits)
psychwell <- fill_na_within_group(psychwell)
vitals <- fill_na_within_group(vitals)


# Define model formulas
get_model_formula <- function(model_type, pacc_col, lancet = FALSE) {
    base_formulas <- list(
        "demographics_no_apoe" = as.formula(paste0(pacc_col, ' ~ age + age2 +
        sex + educ')),
        "demographics" = as.formula(paste0(pacc_col, ' ~ age + age2 +
        sex + educ +
        apoe + age * apoe + age2 * apoe')),
        "lancet" = as.formula(paste0(pacc_col, ' ~ 1')),
        "ptau" = as.formula(paste0(pacc_col, ' ~ ptau')),
        "ptau_demographics_no_apoe" = as.formula(paste0(pacc_col, ' ~ ptau +
        age + age2 +
        sex + educ')),
        "ptau_demographics" = as.formula(paste0(pacc_col, ' ~ ptau + age + age2 +
        sex + educ + 
        apoe + age * apoe + age2 * apoe')),
        "centiloids" = as.formula(paste0(pacc_col, ' ~ centiloids')),
        "centiloids_demographics_no_apoe" = as.formula(paste0(pacc_col, ' ~ centiloids +
        age + age2 +
        sex + educ')),
        "centiloids_demographics" = as.formula(paste0(pacc_col, ' ~ centiloids +
        age + age2 +
        sex + educ + apoe + age * apoe + age2 * apoe')),
        "ptau_centiloids" = as.formula(paste0(pacc_col, ' ~ ptau + centiloids')),
        "ptau_centiloids_demographics_no_apoe" = as.formula(paste0(pacc_col, ' ~ ptau + centiloids +
        age + age2 +
        sex + educ')),
        "ptau_centiloids_demographics" = as.formula(paste0(pacc_col, ' ~ ptau + centiloids +
        age + age2 +
        sex + educ + apoe + age * apoe + age2 * apoe'))
    )

    formula <- base_formulas[[model_type]]

    if (lancet) {
        formula <- update(formula, . ~ . +
                            smoke + alcohol + subuse +
                            aerobic + walking +
                            gdtotal + staital +
                            vsbsys + vsdia + bmi)
    }

    return(formula)
}

get_metrics <- function(train_pacc_df, val_pacc_df, model){
    train_predictions <- predict(model, train_pacc_df)
    val_predictions <- predict(model, val_pacc_df)

    # Calculate R-squared 
    train_rsquared <- rsq(train_predictions, train_pacc_df$PACC.raw)
    train_adj_r2 <- calculate_adj_r2(train_rsquared, nrow(train_pacc_df), length(model$coefficients) - 1)
    val_rsquared <- rsq(val_predictions, val_pacc_df$PACC.raw)
    val_adj_r2 <- calculate_adj_r2(val_rsquared, nrow(val_pacc_df), length(model$coefficients) - 1)

    # ptau_demo_train_rsquared_l <- c(ptau_demo_train_rsquared_l, train_adj_r2)
    # ptau_demo_val_rsquared_l <- c(ptau_demo_val_rsquared_l, val_adj_r2)

    # calculate RMSE
    train_rmse <- sqrt(mean((train_predictions - train_pacc_df$PACC.raw)^2))
    val_rmse <- sqrt(mean((val_predictions - val_pacc_df$PACC.raw)^2))

    # ptau_demo_train_rmse_l <- c(ptau_demo_train_rmse_l, train_rmse)
    # ptau_demo_val_rmse_l <- c(ptau_demo_val_rmse_l, val_rmse)

    return(list(train_rsquared = train_rsquared, val_rsquared = val_rsquared, train_adj_r2 = train_adj_r2, val_adj_r2 = val_adj_r2, train_rmse = train_rmse, val_rmse = val_rmse))
}


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
metrics_list <- list()

lancet_vars <- c(
          "smoke", "alcohol", "aerobic", "walking",
          "gdtotal", "staital", "vsbsys", "vsdia",
          "bmi"
        )

for (pacc_col in c('PACC.raw', 'PACC')) {
    # ptau_demo_train_rsquared_l <- c()
    # ptau_demo_val_rsquared_l <- c()
    # demo_train_rsquared_l <- c()
    # demo_val_rsquared_l <- c()
    # ptau_demo_train_rmse_l <- c()
    # ptau_demo_val_rmse_l <- c()
    # demo_train_rmse_l <- c()
    # demo_val_rmse_l <- c()

    for (fold in seq(0,4)) {
        train_df <- read_parquet(paste0('../../tidy_data/A4/train_', fold, '.parquet'))
        val_df <- read_parquet(paste0('../../tidy_data/A4/val_', fold, '.parquet'))

        train_pacc_df <- preprocess_df(train_df, pacc_col, pacc, habits, psychwell, vitals, centiloids)
        val_pacc_df <- preprocess_df(val_df, pacc_col, pacc, habits, psychwell, vitals, centiloids)

        # scale lancet variables
        means <- apply(train_pacc_df[, c(lancet_vars)], 2, mean, na.rm = TRUE)
        sds <- apply(train_pacc_df[, c(lancet_vars)], 2, sd, na.rm = TRUE)

        train_pacc_df[, c(lancet_vars)] <- scale(train_pacc_df[, c(lancet_vars)],
        center = means, scale = sds
        )
        val_pacc_df[, c(lancet_vars)] <- scale(val_pacc_df[, c(lancet_vars)],
        center = means, scale = sds
        )

        for (model_name in names(models_list)) {
            print(paste("Fitting model:", model_name))

            # Determine if this is a Lancet model
            is_lancet <- grepl("lancet", model_name)
            is_ptau <- grepl("ptau", model_name)
            is_pet <- grepl("centiloids", model_name)

            # Get base model type
            base_type <- gsub("_lancet", "", model_name)

            # Get formula
            formula <- get_model_formula(base_type, pacc_col, is_lancet)

            model <- lm(formula, data = train_pacc_df)
            metrics <- get_metrics(train_pacc_df, val_pacc_df, model)

            # Save model
            models_list[[model_name]][[paste0("fold_", fold + 1)]] <- model

            # Save metrics
            metrics_list[[model_name]][[paste0("fold_", fold + 1)]] <- metrics
        }


       
        # formula <- as.formula(paste0(pacc_col, ' ~ ptau_boxcox + age_centered + age2 + c(SEX) +
        #                                         educ_z + c(APOEGN) + ns(pacc_time, df = 4) +
        #                                         age_centered * c(APOEGN) + age2 * c(APOEGN) +
        #                                         SMOKE + ALCOHOL + c(SUBUSE) + 
        #                                         AEROBIC + WALKING + GDTOTAL + 
        #                                         STAITOTAL + VSBPSYS + VSBPDIA + BMI'))
        #                                         # + AMYLCENT'))
        # model <- lm(formula, data = train_pacc_df)
        # metrics <- get_metrics(train_pacc_df, val_pacc_df, model)
        # ptau_demo_train_rsquared_l <- c(ptau_demo_train_rsquared_l, metrics$train_rsquared)
        # ptau_demo_val_rsquared_l <- c(ptau_demo_val_rsquared_l, metrics$val_rsquared)
        # ptau_demo_train_rmse_l <- c(ptau_demo_train_rmse_l, metrics$train_rmse)
        # ptau_demo_val_rmse_l <- c(ptau_demo_val_rmse_l, metrics$val_rmse)

        # Demo + Lancet
        # formula <- as.formula(paste0(pacc_col, ' ~ age_centered + age2 + c(SEX) +
        #                                         educ_z + c(APOEGN) + ns(pacc_time, df = 4) +
        #                                         age_centered * c(APOEGN) + age2 * c(APOEGN) +
        #                                         SMOKE + ALCOHOL + c(SUBUSE) + 
        #                                         AEROBIC + WALKING + GDTOTAL + 
        #                                         STAITOTAL + VSBPSYS + VSBPDIA + BMI'))
        #                                         # + AMYLCENT'))
        # model <- lm(formula, data = train_pacc_df)
        # metrics <- get_metrics(train_pacc_df, val_pacc_df, model)
        # demo_train_rsquared_l <- c(demo_train_rsquared_l, metrics$train_rsquared)
        # demo_val_rsquared_l <- c(demo_val_rsquared_l, metrics$val_rsquared)
        # demo_train_rmse_l <- c(demo_train_rmse_l, metrics$train_rmse)
        # demo_val_rmse_l <- c(demo_val_rmse_l, metrics$val_rmse)

    }

    # # t-test
    # t_test <- t.test(ptau_demo_val_rsquared_l, demo_val_rsquared_l, paired = TRUE)
    # p_value <- t_test$p.value
    # print(paste0('p-value: ', p_value))

    # combine results into a dataframe
    results <- data.frame()
    for (model_name in names(models_list)) {
        # model_results <- data.frame()
        for (fold in seq(0,4)) {
            results <- rbind(results, data.frame(
                model_name = model_name,
                pacc_col = pacc_col,
                fold = fold,
                train_rsquared = metrics_list[[model_name]][[paste0("fold_", fold + 1)]]$train_rsquared,
                val_rsquared = metrics_list[[model_name]][[paste0("fold_", fold + 1)]]$val_rsquared,
                train_rmse = metrics_list[[model_name]][[paste0("fold_", fold + 1)]]$train_rmse,
                val_rmse = metrics_list[[model_name]][[paste0("fold_", fold + 1)]]$val_rmse
            ))
        }
    
    
        agg_results <- results %>%
            group_by(model_name) %>%
            summarise(train_mean_rsquared = mean(train_rsquared),
                    train_sd_rsquared = sd(train_rsquared),
                    train_ci_lower_rsquared = mean(train_rsquared) - qt(0.975, 4) * sd(train_rsquared) / sqrt(5),
                    train_ci_upper_rsquared = mean(train_rsquared) + qt(0.975, 4) * sd(train_rsquared) / sqrt(5),
                    val_mean_rsquared = mean(val_rsquared),
                    val_sd_rsquared = sd(val_rsquared),
                    val_ci_lower_rsquared = mean(val_rsquared) - qt(0.975, 4) * sd(val_rsquared) / sqrt(5),
                    val_ci_upper_rsquared = mean(val_rsquared) + qt(0.975, 4) * sd(val_rsquared) / sqrt(5),
                    train_mean_rmse = mean(train_rmse),
                    train_sd_rmse = sd(train_rmse),
                    train_ci_lower_rmse = mean(train_rmse) - qt(0.975, 4) * sd(train_rmse) / sqrt(5),
                    train_ci_upper_rmse = mean(train_rmse) + qt(0.975, 4) * sd(train_rmse) / sqrt(5),
                    val_mean_rmse = mean(val_rmse),
                    val_sd_rmse = sd(val_rmse),
                    val_ci_lower_rmse = mean(val_rmse) - qt(0.975, 4) * sd(val_rmse) / sqrt(5),
                    val_ci_upper_rmse = mean(val_rmse) + qt(0.975, 4) * sd(val_rmse) / sqrt(5)
                )
            
        }
    
    

    # save results
    write_parquet(results, paste0('../../results/A4/PACC/spline_model/cubic_spline_results_', pacc_col, '.parquet'))
    print(paste0('Saved results for ', pacc_col))

    write_parquet(agg_results, paste0('../../results/A4/PACC/spline_model/cubic_spline_agg_results_', pacc_col, '.parquet'))
    print(paste0('Saved aggregated results for ', pacc_col))

    # save metrics
    qs::qsave(metrics_list, paste0('../../results/A4/PACC/spline_model/cubic_spline_metrics_', pacc_col, '.qs'))
    print(paste0('Saved metrics for ', pacc_col))

    # save models
    qs::qsave(models_list, paste0('../../results/A4/PACC/spline_model/cubic_spline_models_', pacc_col, '.qs'))
    print(paste0('Saved models for ', pacc_col))
}


models_list <- qs::qread('../../results/A4/PACC/spline_model/cubic_spline_models_PACC.raw.qs')

summary(models_list$demographics_lancet$`fold_3`)


# make boxplots
pacc_col <- 'PACC.raw'
results <- read_parquet(paste0('../../results/A4/PACC/spline_model/cubic_spline_results_', pacc_col, '.parquet'))

# plot results
results <- results[results$model_name %in% c("demographics_lancet",
                                            "ptau",
                                            "ptau_demographics_lancet",
                                            "centiloids",
                                            "centiloids_demographics_lancet",
                                            "ptau_centiloids_demographics_lancet"), ]

# rename models
# results <- results %>%
#     mutate(model_name = case_when(
#         model_name == "demographics_lancet" ~ "Demo+Lancet",
#         model_name == "ptau" ~ "pTau217",
#         model_name == "ptau_demographics_lancet" ~ "Demo+Lancet+pTau217",
#         model_name == "centiloids" ~ "Centiloids",
#         model_name == "centiloids_demographics_lancet" ~ "Demo+Lancet+Centiloids",
#         model_name == "ptau_centiloids_demographics_lancet" ~ "Demo+Lancet+pTau217+Centiloids"
#     ))

# pull colors from plot_figures.R
labels_colors <- get_colors_labels()
colors <- labels_colors$colors[results$model_name]
labels <- labels_colors$labels[results$model_name]

# use ggplot to plot results
ggplot(results, aes(x = factor(model_name, levels = c("demographics_lancet", "ptau", "ptau_demographics_lancet", 
                                                     "centiloids", "centiloids_demographics_lancet", 
                                                     "ptau_centiloids_demographics_lancet")), 
                    y = val_rsquared, color = model_name)) +
  geom_boxplot(fill = "white") +
  geom_point(
    size = 2,
    alpha = .3,
    position = position_jitter(
      seed = 1, width = .2
    )
  ) +
  scale_color_manual(values = unique(colors), labels = unique(labels), name = "Model",
                    breaks = c("demographics_lancet", "ptau", "ptau_demographics_lancet", 
                              "centiloids", "centiloids_demographics_lancet", 
                              "ptau_centiloids_demographics_lancet")) +
  scale_fill_manual(values = unique(colors), labels = unique(labels), name = "Model",
                    breaks = c("demographics_lancet", "ptau", "ptau_demographics_lancet", 
                              "centiloids", "centiloids_demographics_lancet", 
                              "ptau_centiloids_demographics_lancet")) +
  scale_shape_manual(values = c(16, 17, 15, 18, 14, 3)) +
  scale_x_discrete(labels = labels) +
  labs(y = "R²") +
  labs(x = "Model") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1, size = 14, face = "bold"),
    axis.text.y = element_text(size = 14, face = "bold"),
    axis.title.x = element_text(size = 16, face = "bold"),
    axis.title.y = element_text(size = 16, face = "bold"),
    legend.text = element_text(size = 14, face = "bold"),
    legend.title = element_text(size = 16, face = "bold"),
    plot.title = element_text(size = 18, face = "bold")
  )

ggsave(paste0('../../results/A4/PACC/spline_model/cubic_spline_boxplot_', pacc_col, '.pdf'), width = 10, height = 8)
