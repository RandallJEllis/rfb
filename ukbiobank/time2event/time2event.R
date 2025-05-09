# Load necessary libraries
library(flexsurv)
library(arrow)
library(janitor)
library(ggplot2)

# Simulate some example survival data
set.seed(123)  # For reproducibility
n <- 100  # Number of observations
time <- rweibull(n, shape = 1.5, scale = 5)  # Simulated survival times
status <- rbinom(n, 1, 0.7)  # Censoring indicator (1 = event, 0 = censored)
age <- rnorm(n, mean = 50, sd = 10)  # Simulated age covariate
treatment <- sample(c(0, 1), n, replace = TRUE)  # Simulated treatment group (0 or 1)

# Combine into a data frame
data <- data.frame(time = time, status = status, age = age, treatment = treatment)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
df <- read_parquet('../tidy_data/dementia/proteomics/X_time2event.parquet')
df <- clean_names(df)

num_controls <- dim(df)[1] - 1453

hist(df$time2event[1:num_controls])
hist(df$time2event[(num_controls+1):dim(df)[1]])

# control_t2e = sample(df$time2event[(num_controls+1):dim(df)[1]], num_controls,
#                      replace=T)
# df$time2event[1:num_controls] = control_t2e
df$time2event[1:num_controls] = 7300

# Fit an AFT model using the Weibull distribution
aft_model <- flexsurvreg(Surv(time2event, label) ~ x21003_0_0 + apoe_polymorphism_0_0 + 
     apoe_polymorphism_1_0 + apoe_polymorphism_2_0, data = df, dist = "weibull")

# Print the model summary
summ_df = summary(aft_model)[[1]]


ss.weibull.sep.boot <- standsurv(aft_model,
     type = "survival",
            at = list(list(x21003_0_0 = 65, apoe_polymorphism_0_0=1, apoe_polymorphism_1_0=0, apoe_polymorphism_2_0=0),
                      list(x21003_0_0 = 65, apoe_polymorphism_0_0=0, apoe_polymorphism_1_0=1, apoe_polymorphism_2_0=0),
                      list(x21003_0_0 = 65, apoe_polymorphism_0_0=0, apoe_polymorphism_1_0=0, apoe_polymorphism_2_0=1),
                      list(x21003_0_0 = 75, apoe_polymorphism_0_0=1, apoe_polymorphism_1_0=0, apoe_polymorphism_2_0=0),
                      list(x21003_0_0 = 75, apoe_polymorphism_0_0=0, apoe_polymorphism_1_0=1, apoe_polymorphism_2_0=0),
                      list(x21003_0_0 = 75, apoe_polymorphism_0_0=0, apoe_polymorphism_1_0=0, apoe_polymorphism_2_0=1)
                    ),
     # t = seq(0,7300),
     ci = TRUE,
     boot = TRUE,
     B = 100,
     seed = 2367)
#> Calculating bootstrap standard errors / confidence intervals

# Reorder the 'at' factor for legend order
attr(ss.weibull.sep.boot, "standpred_at")$at <- factor(attr(ss.weibull.sep.boot, "standpred_at")$at,
     levels = c("x21003_0_0=65, apoe_polymorphism_0_0=1, apoe_polymorphism_1_0=0, apoe_polymorphism_2_0=0",
                "x21003_0_0=65, apoe_polymorphism_0_0=0, apoe_polymorphism_1_0=1, apoe_polymorphism_2_0=0",
                "x21003_0_0=65, apoe_polymorphism_0_0=0, apoe_polymorphism_1_0=0, apoe_polymorphism_2_0=1",
                "x21003_0_0=75, apoe_polymorphism_0_0=1, apoe_polymorphism_1_0=0, apoe_polymorphism_2_0=0",
                "x21003_0_0=75, apoe_polymorphism_0_0=0, apoe_polymorphism_1_0=1, apoe_polymorphism_2_0=0",
                "x21003_0_0=75, apoe_polymorphism_0_0=0, apoe_polymorphism_1_0=0, apoe_polymorphism_2_0=1"))

ggplot(ss.weibull.sep.boot, ci = TRUE) +  
     geom_ribbon(aes(x = time, ymin = survival_lci, ymax = survival_uci, fill = at), 
     alpha = 0.2, 
     data = attr(ss.weibull.sep.boot, "standpred_at")) +
     geom_line(aes(x = time, y = survival, color = at), 
     data = attr(ss.weibull.sep.boot, "standpred_at")) +

     # Manually specify the colors
     scale_color_manual(name = 'Group', values = c("red", "orange", "yellow", "green", "blue", "purple"), 
     labels = c("Age 65, 0 alleles", "Age 65, 1 alleles", "Age 65, 2 alleles",
                "Age 75, 0 alleles", "Age 75, 1 alleles", "Age 75, 2 alleles")) +
          
     scale_fill_manual(name='',values = c("red", "orange", "yellow", "green", "blue", "purple"), 
     labels = c("Age 65, 0 alleles", "Age 65, 1 alleles", "Age 65, 2 alleles",
                "Age 75, 0 alleles", "Age 75, 1 alleles", "Age 75, 2 alleles"), guide='none') + 
     
     # Convert x-axis tick values from days to years
     scale_x_continuous(name = "Time (years)", 
          breaks = c(0,5,10,15,20) * 365.25,
          labels = function(x) round(x / 365.25, 2)) +
          
     # axis labels
     labs(x = "Time (years)", y = "Survival Probability")

     
ggsave('ukb_apoe_alleles.pdf', width = 6, height = 3.33)
     
     # Use guide_legend to control the legend order independently
     # guides(color = guide_legend(order = 1),
     # fill = guide_legend(order = 1))








# # Plot Kaplan-Meier curves and fitted AFT model survival curves
# library(survival)

# # Fit Kaplan-Meier for comparison
# km_fit <- survfit(Surv(time2event, label) ~ x21003_0_0, data = df)

# # Plot Kaplan-Meier curves
# plot(km_fit, col = c("red", "blue"), lty = 1:2, main = "Kaplan-Meier vs AFT Model", xlab = "Time", ylab = "Survival Probability")

# # Add fitted survival curves from the AFT model
# lines(aft_model, col = c("red", "blue"), lty = 1:2, ci = FALSE)
# legend("topright", legend = c("KM Group 0", "KM Group 1", "AFT Group 0", "AFT Group 1"), 
#        col = c("red", "blue", "red", "blue"), lty = c(1, 1, 2, 2))

# # Plot estimated hazard functions
# plot(aft_model, type = "hazard", main = "Estimated Hazard Function", xlab = "Time", ylab = "Hazard Rate")


# # Plot survival curves for different levels of a covariate (e.g., age)
# plot(aft_model, newdata = data.frame(x21003_0_0 = c(40, 50, 60, 70, 80)), 
#      xlab = "Time", ylab = "Survival Probability", main = "Survival Curves by Age")


# # QQ Plot to check distributional assumptions
# flexsurv::qqsurvreg(aft_model, type = "cox-snell", main = "QQ Plot of Cox-Snell Residuals")
