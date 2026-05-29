#!/usr/bin/env Rscript
# Confirmatory linear mixed model for the loss comparison.
#   metric ~ loss + model + scheme + (1|dataset) + (1|dataset:trait)
# MSE is the reference level for `loss`. Usage:
#   Rscript lmm.R <long.csv> <metric> <calibration> <out.csv>
suppressMessages(library(lme4))
have_lt <- requireNamespace("lmerTest", quietly = TRUE)
if (have_lt) suppressMessages(library(lmerTest))
a <- commandArgs(trailingOnly = TRUE)
csv <- a[1]; metric <- a[2]; calibration <- a[3]; out <- a[4]

d <- read.csv(csv, check.names = FALSE)
d <- d[d$calibration == calibration & d$loss %in% c("mse", "pearson", "hybrid", "ccc"), ]
d$loss <- relevel(factor(d$loss), ref = "mse")
d$model <- factor(d$model)
d$y <- d[[metric]]
d <- d[is.finite(d$y), ]
has_scheme <- ("scheme" %in% names(d)) && (length(unique(d$scheme)) > 1)
if (has_scheme) d$scheme <- factor(d$scheme)

# Drop random-effect / fixed terms with a single level (avoids singular fits).
re <- c("(1|dataset)")
if (length(unique(d$trait)) > 1) re <- c(re, "(1|dataset:trait)")
fixed <- "loss"
if (length(unique(d$model)) > 1) fixed <- paste(fixed, "+ model")
if (has_scheme) fixed <- paste(fixed, "+ scheme")
form <- as.formula(paste("y ~", fixed, "+", paste(re, collapse = " + ")))

m <- lmer(form, data = d, REML = TRUE,
          control = lmerControl(check.conv.singular = .makeCC("ignore", tol = 1e-4)))
co <- as.data.frame(coef(summary(m)))          # lme4: Est/SE/t ; lmerTest adds df + Pr(>|t|)
co$term <- rownames(co)
ci <- tryCatch(as.data.frame(confint(m, method = "Wald", parm = "beta_"))[co$term, ],
               error = function(e) NULL)
if (!is.null(ci)) { co$ci_lo <- ci[, 1]; co$ci_hi <- ci[, 2] }
write.csv(co, out, row.names = FALSE)
cat("metric:", metric, " calibration:", calibration, " n:", nrow(d),
    " lmerTest:", have_lt, "\n")
print(co)
