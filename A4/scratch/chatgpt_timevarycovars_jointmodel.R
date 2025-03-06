# Install and load required packages
install.packages("nlme")
install.packages("survival")
install.packages("JMbayes")

library(nlme)
library(survival)
library(JMbayes)
library(JMbayes2)

# Set seed for reproducibility
set.seed(123)

# Number of individuals
N <- 100

# We will simulate survival times from an exponential distribution 
# and introduce some censoring.
# True survival times
true_surv_times <- rexp(N, rate = 0.1)  # mean survival ~ 10 units
# Administrative censoring at time = 12 (just as an example)
censoring_time <- 12
obs_surv_times <- pmin(true_surv_times, censoring_time)
event <- as.numeric(true_surv_times <= censoring_time)

# Create data_surv with id, time (survival time), and event indicator
data_surv <- data.frame(
  id = 1:N,
  time = obs_surv_times,
  event = event
)

# Now simulate the longitudinal measurements of A.
# Assume A changes linearly over time with random intercept and slope per individual.
# For simplicity, let's say each individual has between 3 and 7 measurements of A.
library(dplyr)

data_long <- lapply(1:N, function(i) {
  # Random number of measurements for each individual
  n_meas <- sample(3:7, 1)
  
  # Measurement times for A: 0 up to about 10, 
  # spaced every 1 to 3 years randomly (irregular)
  time_points <- cumsum(runif(n_meas, min = 1, max = 3))
  
  # True underlying intercept and slope for A for this individual
  # Intercept around 10, slope around -0.5, plus some variation
  intercept_i <- 10 + rnorm(1, sd = 2)
  slope_i <- -0.5 + rnorm(1, sd = 0.2)
  
  # Generate A measurements
  A_values <- intercept_i + slope_i * time_points + rnorm(n_meas, sd = 1)
  
  data.frame(
    id = i,
    time_A = time_points,
    A = A_values
  )
}) %>%
  bind_rows()

# Sort by id and time_A just for neatness
data_long <- data_long %>%
  arrange(id, time_A)

# Have a look at the first few rows
head(data_long)
head(data_surv)

# Now you have data_long and data_surv ready.
# data_long: (id, time_A, A)
# data_surv: (id, time, event)



# Example: 
# data_long has columns: id, time_A, A (the repeated measurements)
# data_surv has columns: id, time (follow-up time), event (0/1)

# Fit a linear mixed-effects model for A over time
lmeFit <- lme(A ~ time_A, random = ~ time_A | id, data = data_long)

# Create a Surv object for survival data
survObj <- Surv(data_surv$time, data_surv$event)

# Fit a Cox model for the survival part
coxFit <- coxph(survObj ~ 1, data = data_surv, x = TRUE)

# Fit the joint model
jointFit <- jm(coxFit, lmeFit, time_var = "time_A", 
               # You can specify which longitudinal variable links to survival:
               joint_model = "shared_betas")

summary(jointFit)

# Once fitted, you can make dynamic predictions, plot fitted trajectories, etc.
# For example:
newData <- data_long[data_long$id == 67, ]  # a single subject
newData$survObj <- Surv(rep(0,3))

dynPred <- predict(jointFit, newdata = newData, times = c(2, 3, 4)) 
# predicts survival probabilities at times 2, 3, and 4 given the subject's data
