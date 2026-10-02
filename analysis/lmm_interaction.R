#!/usr/bin/env Rscript
# Palier-1 confirmatory interaction model: does architecture MODERATE the loss
# benefit? Fits  metric ~ loss * model + (1|dataset) + (1|dataset:trait)  with MSE
# and CNN as reference levels, on the cell-aggregate table (one row per
# dataset x trait x model x loss), so the loss:model interaction terms are the
# formal moderation test the additive model could not provide.
#
# Usage: Rscript lmm_interaction.R <long_p1.csv> <metric> <calibration> <out.csv>
suppressMessages(library(lme4))
have_lt <- requireNamespace("lmerTest", quietly = TRUE)
if (have_lt) suppressMessages(library(lmerTest))
a <- commandArgs(trailingOnly = TRUE)
csv <- a[1]; metric <- a[2]; calibration <- a[3]; out <- a[4]

d <- read.csv(csv, check.names = FALSE)
d <- d[d$calibration == calibration & d$loss %in% c("mse", "pearson", "hybrid", "ccc"), ]
d$loss  <- relevel(factor(d$loss),  ref = "mse")
d$model <- relevel(factor(d$model), ref = "cnn")
d$y <- d[[metric]]
d <- d[is.finite(d$y), ]

re <- c("(1|dataset)")
if (length(unique(d$trait)) > 1) re <- c(re, "(1|dataset:trait)")
form <- as.formula(paste("y ~ loss * model +", paste(re, collapse = " + ")))

m <- lmer(form, data = d, REML = TRUE,
          control = lmerControl(check.conv.singular = .makeCC("ignore", tol = 1e-4)))
co <- as.data.frame(coef(summary(m)))
co$term <- rownames(co)
ci <- tryCatch(as.data.frame(confint(m, method = "Wald", parm = "beta_"))[co$term, ],
               error = function(e) NULL)
if (!is.null(ci)) { co$ci_lo <- ci[, 1]; co$ci_hi <- ci[, 2] }
write.csv(co, out, row.names = FALSE)
cat("metric:", metric, " calibration:", calibration, " n:", nrow(d),
    " lmerTest:", have_lt, "\n")
print(co)
