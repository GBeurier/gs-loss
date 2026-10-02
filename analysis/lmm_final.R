#!/usr/bin/env Rscript
# Corroborative hierarchical model for paired loss-minus-MSE contrasts.
# The response has already been averaged over CV folds, so folds are not treated
# as independent observations. Repeated architectures within a trait-dataset
# share a task random intercept, nested within the biological panel.
suppressMessages(library(lme4))
have_lt <- requireNamespace("lmerTest", quietly = TRUE)
if (have_lt) suppressMessages(library(lmerTest))

a <- commandArgs(trailingOnly = TRUE)
if (length(a) != 3) stop("usage: lmm_final.R paired_deltas.csv metric out.csv")
d <- read.csv(a[1], check.names = FALSE)
metric <- a[2]
d <- d[d$metric == metric & is.finite(d$delta), ]
d$panel <- factor(d$panel)
d$task <- interaction(d$dataset, d$trait, drop = TRUE)
d$model <- factor(d$model)
d$loss <- factor(d$loss, levels = c("pearson", "hybrid", "ccc"))

m <- lmer(delta ~ 0 + loss:model + (1 | panel) + (1 | task), data = d,
          REML = TRUE,
          control = lmerControl(check.conv.singular = .makeCC("ignore", tol = 1e-4)))
co <- as.data.frame(coef(summary(m)))
co$term <- rownames(co)
ci <- suppressMessages(confint(m, method = "Wald", parm = "beta_"))
co$ci_lo <- ci[, 1]
co$ci_hi <- ci[, 2]
co$metric <- metric
co$n_task_model_loss <- nrow(d)
co$n_tasks <- length(unique(d$task))
co$n_panels <- length(unique(d$panel))
write.csv(co, a[3], row.names = FALSE)
cat("metric:", metric, "rows:", nrow(d), "tasks:", length(unique(d$task)),
    "panels:", length(unique(d$panel)), "lmerTest:", have_lt, "\n")
print(co)
