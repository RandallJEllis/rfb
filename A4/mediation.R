# Install and load required packages
required_packages <- c("mediation", "lavaan", "dagitty", "tipr", "MatchIt", "tmle", "SuperLearner")
for(pkg in required_packages) {
  if(!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# Mediation Analysis
# Now try the mediation analysis
run_mediation_analysis <- function(data) {
  # First stage: mediator model
  fit.mediator <- lm(ORRES ~ AGEYR + SEX + APOEGN + EDCCNTU, data = data)
  
  # Second stage: outcome model
  fit.outcome <- glm(label ~ AGEYR + ORRES + SEX + APOEGN + EDCCNTU, 
                     family = binomial(link = "logit"), 
                     data = data)
  
  # Estimate mediation effects
  med.fit <- mediate(fit.mediator, fit.outcome, 
                     treat = "AGEYR", 
                     mediator = "ORRES",
                     boot = TRUE, 
                     sims = 1000,
                     boot.ci.type = "bca")
  
  return(med.fit)
}

library(arrow)
df = read_parquet('./A4/ptau217_baseline.parquet')
print(dim(df))
df = df[df$APOEGN != 'E2/E2',]
print(dim(df))
df$AGEYR = scale(df$AGEYR)
df$EDCCNTU = scale(df$EDCCNTU)
df$ORRES = scale(df$ORRES)
df$AGEYR <- as.numeric(df$AGEYR[,1])  # If it's a single-column matrix
df$ORRES <- as.numeric(df$ORRES[,1])  # If it's a single-column matrix

df$SEX = as.factor(df$SEX)
df$APOEGN = as.factor(df$APOEGN)

fit.outcome <- glm(label ~ AGEYR + SEX + APOEGN + EDCCNTU, 
                   family = binomial(link = "logit"), 
                   data = df)

med.fit = run_mediation_analysis(df)
# Assuming your mediation results are stored in 'med.fit'

# 1. Basic summary
summary(med.fit)

# 2. Detailed visualization of effects
plot(med.fit)

# 3. Create a more detailed visualization of direct and indirect effects
ggplot2_mediation_plot <- function(med_fit) {
  # Extract effect estimates
  acme <- med_fit$d0
  ade <- med_fit$z0
  total <- med_fit$tau.coef
  
  # Create data frame for plotting
  effects_df <- data.frame(
    Effect = c("Indirect (ACME)", "Direct (ADE)", "Total"),
    Estimate = c(acme, ade, total),
    Lower = c(med_fit$d0.ci[1], med_fit$z0.ci[1], med_fit$tau.ci[1]),
    Upper = c(med_fit$d0.ci[2], med_fit$z0.ci[2], med_fit$tau.ci[2])
  )
  
  # Create plot
  ggplot(effects_df, aes(x = Effect, y = Estimate)) +
    geom_point(size = 3) +
    geom_errorbar(aes(ymin = Lower, ymax = Upper), width = 0.2) +
    coord_flip() +
    theme_minimal() +
    labs(title = "Mediation Analysis Results",
         subtitle = "Effect estimates with 95% confidence intervals",
         y = "Effect Size",
         x = "")
}

# 4. Calculate proportion mediated
prop_mediated <- med.fit$n0 / med.fit$tau.coef * 100

# Create summary table
mediation_summary <- data.frame(
  Effect = c("Indirect Effect (ACME)", "Direct Effect (ADE)", "Total Effect"),
  Estimate = c(med.fit$d0, med.fit$z0, med.fit$tau.coef),
  CI_Lower = c(med.fit$d0.ci[1], med.fit$z0.ci[1], med.fit$tau.ci[1]),
  CI_Upper = c(med.fit$d0.ci[2], med.fit$z0.ci[2], med.fit$tau.ci[2]),
  P_Value = c(med.fit$d0.p, med.fit$z0.p, med.fit$tau.p)
)

# Structural Equation Modeling
run_sem_analysis <- function(data) {
  # Define model
  model <- '
    # Measurement model
    cognitive =~ memory + executive + processing_speed
    
    # Structural model
    cognitive ~ age + education + apoe
    pTau ~ age + cognitive
    amyloid ~ pTau + age
    
    # Allow for correlations
    memory ~~ executive
    executive ~~ processing_speed
  '
  
  # Fit model
  fit <- sem(model, data = data)
  return(fit)
}

# Sensitivity Analysis for Unmeasured Confounding
run_sensitivity <- function(data, outcome_model) {
  # Tipr analysis for robustness to unmeasured confounding
  sens <- tipr(outcome_model,
               data = data,
               exposure = "biomarker_status",
               outcome = "cognitive_decline",
               covariates = c("age", "education", "apoe"))
  
  return(sens)
}