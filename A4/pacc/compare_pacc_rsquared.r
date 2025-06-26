library(lme4)
library(emmeans)
library(arrow)
library(this.path)

setwd(dirname(this.path()))
# Assume a data.frame like:


# data <- data.frame(
#   r_squared = c(...),
#   model = factor(rep(c("ModelA", "ModelB", "ModelC"), each = 5)),
#   fold = factor(rep(1:5, times = 3))
# )

spline_res = read_parquet('../../results/A4/PACC/spline_model/cubic_spline_results_PACC.parquet')

model_fit <- lmer(val_rsquared ~ model_name + (1 | fold), data = spline_res)

# ANOVA to test overall differences
anova(model_fit)

# Pairwise comparisons with Tukey correction
comparisons <- emmeans(model_fit, pairwise ~ model_name, adjust = "tukey")
contrasts <- data.frame(comparisons$contrasts)

contrasts[contrasts$contrast == "demographics_lancet - ptau",]
