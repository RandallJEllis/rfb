#CHATGPT O1 pro

# ------------------------------------------------------------
# Simulate Data
# ------------------------------------------------------------
set.seed(123)

n <- 200  # number of patients
max_follow <- 5  # maximum follow-up in years
visit_times <- sort(unique(round(runif(n * 4, 0, max_follow), 1))) # irregular visit times

# Each patient has their own "true" trajectory for a biomarker.
# Let's assume a random intercept and slope model.
# Biomarker trajectory: Y_ij = beta0 + b0_i + (beta1 + b1_i)*t_ij + error
# Survival: hazard depends on current true biomarker level.

beta0 <- 2
beta1 <- -0.3
sigma_y <- 0.5  # residual SD for longitudinal measurements

# Random effects
b0 <- rnorm(n, 0, 0.5)
b1 <- rnorm(n, 0, 0.2)

# Generate true event times using a Weibull model dependent on true biomarker level
# Hazard at time t: h(t) = h0(t)*exp(gamma * true_biomarker(t))
# where h0(t) is baseline hazard and gamma is association parameter.

gamma <- 0.5
shape <- 1.2  # shape of Weibull
scale <- 2.0  # scale of Weibull

# True underlying biomarker trajectory for each patient at any time t
true_biomarker <- function(t, i) {
  (beta0 + b0[i]) + (beta1 + b1[i]) * t
}

# Event times: simulate survival times by inversion
# Survival function of Weibull: S(t) = exp(-(t/scale)^shape)
# Let’s incorporate the longitudinal process:
# Hazard(t) = h0(t)*exp(gamma * true_biomarker(t)), 
# For simplicity, we assume a piecewise approximation by evaluating at a fine grid.
time_grid <- seq(0, max_follow, by = 0.01)

T_true <- rep(NA, n)
for (i in 1:n) {
  h0_t <- (shape / scale) * (time_grid / scale)^(shape - 1)
  # Hazard with covariate
  h_t <- h0_t * exp(gamma * true_biomarker(time_grid, i))
  # Cumulative hazard
  H_t <- cumsum(h_t * 0.01)
  S_t <- exp(-H_t)
  # draw a uniform random to get event time
  u <- runif(1)
  # Find first time where S_t < u
  if(any(S_t < u)) {
    T_true[i] <- time_grid[which(S_t < u)[1]]
  } else {
    # No event within follow-up
    T_true[i] <- max_follow
  }
}

# Censoring
C <- runif(n, 2, max_follow)  # random censoring
time_obs <- pmin(T_true, C)
event <- as.numeric(T_true <= C)

# Irregular measurement times per patient:
long_data <- data.frame()
for (i in 1:n) {
  # Simulate irregular visit times for this patient
  patient_visits <- sort(sample(visit_times, size = sample(2:6, 1), replace = FALSE))
  
  # True and observed biomarker
  Y_true <- true_biomarker(patient_visits, i)
  Y_obs <- Y_true + rnorm(length(Y_true), 0, sigma_y)
  
  # Keep only visits before the event/censor time
  keep <- patient_visits <= time_obs[i]
  patient_visits <- patient_visits[keep]
  Y_obs <- Y_obs[keep]
  
  # Only append data if there is at least one measurement
  if (length(patient_visits) > 0) {
    long_data <- rbind(long_data,
                       data.frame(id = i,
                                  time = patient_visits,
                                  Y = Y_obs))
  } 
  # If no visits remain (length(patient_visits) == 0), this patient ends up with no longitudinal data.
  # This could still happen in reality, and you may want to ensure that the simulation 
  # or censoring mechanism does not produce too many patients with no measurements.
}


surv_data <- data.frame(id = 1:n, event_time = time_obs, event = event)

# ------------------------------------------------------------
# Approach 1: Cox model with LOCF
# ------------------------------------------------------------
# We will create a time-dependent dataframe suitable for coxph using LOCF.
library(survival)

# To apply LOCF, we need to expand the survival dataset into intervals
td_data <- surv_data[rep(1:n, each = length(visit_times)),]
td_data$obs_time <- rep(visit_times, times = n)
td_data <- td_data[td_data$obs_time <= td_data$event_time,]

# Merge the observed measurements
td_data <- merge(td_data, long_data, by = "id", all.x = TRUE)

# Sort by id and obs_time
td_data <- td_data[order(td_data$id, td_data$obs_time),]

# LOCF: for each patient, carry the last observed Y forward
td_data$Y_locf <- ave(td_data$Y, td_data$id, FUN = function(x) {
  # carry forward last observation
  zoo::na.locf(x, na.rm = FALSE)
})

# Remove rows without any observed Y (if occurs at start)
td_data <- td_data[!is.na(td_data$Y_locf),]

# Create a counting process style dataset for survival:
# For each id, create intervals [previous_obs_time, obs_time)
td_data$start <- ave(td_data$obs_time, td_data$id, FUN = function(x) c(0, head(x, -1)))
td_data$stop <- td_data$obs_time

# Final step: the last row per id matches the event time or censor
last_rows <- !duplicated(td_data$id, fromLast = TRUE)
if (any(last_rows)) {
  td_data$stop[last_rows] <- td_data$event_time[last_rows]
}

# Since each subject's event status is already known from surv_data:
td_data$event <- 0
td_data$event[last_rows] <- surv_data$event[match(td_data$id[last_rows], surv_data$id)]

# Clean
td_data$event <- ifelse(last_rows, td_data$event, 0)
td_data <- td_data[td_data$stop > td_data$start,] # ensure positive intervals

# Fit Cox model with LOCF
cox_locf <- coxph(Surv(start, stop, event) ~ Y_locf, data = td_data, timefix = FALSE)
summary(cox_locf)

# ------------------------------------------------------------
# Approach 2: Joint Modeling using JM (or JMbayes2)
# ------------------------------------------------------------
# We will fit a linear mixed model for the biomarker and then a Cox model for the event,
# followed by a joint model to link them.

# Load packages
library(nlme)
library(survival)
library(JMbayes2) # Instead of JMbayes

# Longitudinal model (lme)
fitLME <- lme(Y ~ time, random = ~ time | id, data = long_data)

# Survival model (coxph)
fitSURV <- coxph(Surv(time, event) ~ 1, data = surv_data, x = TRUE)

# Ensure IDs match between datasets
ids_long <- unique(long_data$id)
ids_surv <- unique(surv_data$id)

missing_in_surv <- setdiff(ids_long, ids_surv)
missing_in_long <- setdiff(ids_surv, ids_long)

if (length(missing_in_surv) > 0) {
  long_data <- long_data[!long_data$id %in% missing_in_surv, ] 
}

if (length(missing_in_long) > 0) {
  surv_data <- surv_data[!surv_data$id %in% missing_in_long, ] 
}

# Re-fit models after alignment if needed
fitLME <- lme(Y ~ time, random = ~ time | id, data = long_data)
fitSURV <- coxph(Surv(time, event) ~ 1, data = surv_data, x = TRUE)

# Fit the joint model with JMbayes2
fitJM2 <- jm(
  Mixed_objects = fitLME,    # longitudinal model first
  Surv_object = fitSURV,     # survival model second
  time_var = "time",
  iter = 30000, 
  warmup = 5000, 
  chains = 1, 
  thin = 10
)

summary(fitJM2)





# ------------------------------------------------------------
# Comparison of Results
# ------------------------------------------------------------
# The cox_locf model provides an estimate of the hazard ratio per unit change in Y_locf.
# The joint model provides a more principled estimate of the longitudinal-survival association.

# In simulations, often the LOCF approach biases the effect toward null or inflates it 
# because it does not properly handle measurement error and missingness pattern.
# The joint model uses the underlying latent trajectory for each subject, reducing bias.

# Check and compare the results:
cat("Cox with LOCF estimate (log hazard ratio):", coef(cox_locf), "\n")
cat("Joint model association parameter (log hazard ratio):", fixef(fitJM2), "\n")


# In real data or more extensive simulations, one would replicate this simulation many times 
# to empirically show that the LOCF approach is biased, while the joint modeling approach 
# consistently recovers the true parameter closer to the simulation truth.
