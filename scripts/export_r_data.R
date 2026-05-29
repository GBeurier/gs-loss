#!/usr/bin/env Rscript
# Export the two R-hosted benchmark datasets to plain CSV so the Python pipeline
# is self-contained and reproducible.
#   Usage: Rscript export_r_data.R <outdir> [wheat|soynam|all]
suppressMessages({
  ok <- requireNamespace("BGLR", quietly = TRUE) && requireNamespace("SoyNAM", quietly = TRUE)
})
args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args) >= 1) args[1] else "data_cache"
what <- if (length(args) >= 2) args[2] else "all"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

if (what %in% c("all", "wheat")) {
  suppressMessages(library(BGLR)); data(wheat)
  wd <- file.path(outdir, "wheat"); dir.create(wd, showWarnings = FALSE)
  write.table(wheat.X, file.path(wd, "X.csv"), sep = ",", row.names = FALSE, col.names = FALSE)
  write.table(wheat.Y, file.path(wd, "Y.csv"), sep = ",", row.names = FALSE,
              col.names = c("env1", "env2", "env3", "env4"))
  cat("wheat exported:", nrow(wheat.X), "x", ncol(wheat.X), "\n")
}

if (what %in% c("all", "soynam")) {
  suppressMessages(library(SoyNAM))
  sd <- file.path(outdir, "soynam"); dir.create(sd, showWarnings = FALSE)
  traits <- c("yield", "height", "protein", "oil")
  for (tr in traits) {
    b <- BLUP(trait = tr, family = "all", env = "all", MAF = 0.05, impute = "FM")
    gz <- gzfile(file.path(sd, paste0(tr, "_geno.csv.gz")), "w")
    write.table(b$Gen, gz, sep = ",", row.names = FALSE, col.names = FALSE)
    close(gz)
    write.csv(data.frame(y = b$Phen, family = b$Fam),
              file.path(sd, paste0(tr, "_pheno.csv")), row.names = FALSE)
    cat("soynam", tr, "exported:", nrow(b$Gen), "x", ncol(b$Gen), "\n")
  }
}
