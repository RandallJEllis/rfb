# Load necessary libraries
library(flexsurv)
library(arrow)
library(janitor)

# Simulate some example survival data
set.seed(123)  # For reproducibility
n <- 100  # Number of observations
time <- rweibull(n, shape = 1.5, scale = 5)  # Simulated survival times
status <- rbinom(n, 1, 0.7)  # Censoring indicator (1 = event, 0 = censored)
age <- rnorm(n, mean = 50, sd = 10)  # Simulated age covariate
treatment <- sample(c(0, 1), n, replace = TRUE)  # Simulated treatment group (0 or 1)

# Combine into a data frame
data <- data.frame(time = time, status = status, age = age, treatment = treatment)

df <- read_parquet('../rfb/tidy_data/dementia/proteomics/X_time2event.parquet')
df <- clean_names(df)

num_controls <- dim(df)[1] - 1453

hist(df$time2event[1:num_controls])
hist(df$time2event[(num_controls+1):dim(df)[1]])

control_t2e = sample(df$time2event[(num_controls+1):dim(df)[1]], num_controls,
                     replace=T)
df$time2event[1:num_controls] = control_t2e

# Fit an AFT model using the Weibull distribution
aft_model <- flexsurvreg(Surv(time2event, label) ~ x21003_0_0, data = df, dist = "weibull")

# Print the model summary
summ_df = summary(aft_model)[[1]]


# Plot Kaplan-Meier curves and fitted AFT model survival curves
library(survival)

# Fit Kaplan-Meier for comparison
km_fit <- survfit(Surv(time2event, label) ~ x21003_0_0, data = df)

# Plot Kaplan-Meier curves
plot(km_fit, col = c("red", "blue"), lty = 1:2, main = "Kaplan-Meier vs AFT Model", xlab = "Time", ylab = "Survival Probability")

# Add fitted survival curves from the AFT model
lines(aft_model, col = c("red", "blue"), lty = 1:2, ci = FALSE)
legend("topright", legend = c("KM Group 0", "KM Group 1", "AFT Group 0", "AFT Group 1"), 
       col = c("red", "blue", "red", "blue"), lty = c(1, 1, 2, 2))

# Plot estimated hazard functions
plot(aft_model, type = "hazard", main = "Estimated Hazard Function", xlab = "Time", ylab = "Hazard Rate")


# Plot survival curves for different levels of a covariate (e.g., age)
plot(aft_model, newdata = data.frame(x21003_0_0 = c(40, 50, 60, 70, 80)), 
     xlab = "Time", ylab = "Survival Probability", main = "Survival Curves by Age")


# QQ Plot to check distributional assumptions
flexsurv::qqsurvreg(aft_model, type = "cox-snell", main = "QQ Plot of Cox-Snell Residuals")
