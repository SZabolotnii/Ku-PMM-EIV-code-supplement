#!/usr/bin/env Rscript
# Gate G0 (SPEC 7): with U == 0 the TC-PMM2 closed form must coincide
# numerically with the portfolio oracle EstemPMM::lm_pmm2() on the same data.
#
# The PMM kernel is NEVER re-implemented (CLAUDE.md, frozen decision 4): the R
# package stays the reference, TC-PMM2 lives in Python, and this script is the
# bridge.  Python emits the fixture (data + its nu = 0 fit + the OLS-residual
# moments it used, which are exactly EstemPMM::compute_moments); R refits with
# lm_pmm2() and compares.
#
# Usage:  Rscript scripts/oracle_check.R
# Interpreter override: KU_PMM_EIV_PYTHON

suppressWarnings(suppressMessages({
  ok <- requireNamespace("EstemPMM", quietly = TRUE)
}))

root <- normalizePath(file.path(dirname(sub("--file=", "",
        grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE)[1])), ".."))
fixdir <- file.path(root, "evidence", "g0_oracle")

if (!ok) {
  cat("[SKIPPED-ENVIRONMENT] EstemPMM is not installed; G0 cannot run.\n")
  cat("  Nothing is installed system-wide by this script.\n")
  cat("  Fallback: scripts/verify.py runs a pure-Python reduction check (G0b).\n")
  quit(status = 0)
}

py <- Sys.getenv("KU_PMM_EIV_PYTHON", unset = file.path(root, ".venv", "bin", "python"))
if (!file.exists(py)) py <- "python3"

cat("EstemPMM version :", as.character(packageVersion("EstemPMM")), "\n")
cat("python           :", py, "\n")
cat("fixture dir      :", fixdir, "\n\n")

st <- system2(py, c(file.path(root, "scripts", "estimators.py"), "--g0-fixture", fixdir),
              stdout = TRUE, stderr = TRUE)
if (!is.null(attr(st, "status")) && attr(st, "status") != 0) {
  cat(st, sep = "\n"); stop("failed to build the G0 fixture")
}

library(EstemPMM)
exp_df <- read.csv(file.path(fixdir, "g0_expected.csv"))
tol <- 1e-6
res <- data.frame()

for (i in seq_len(nrow(exp_df))) {
  row <- exp_df[i, ]
  dat <- read.csv(file.path(fixdir, sprintf("g0_data_%d.csv", row$dataset)))
  fit <- EstemPMM::lm_pmm2(y ~ x, data = dat, tol = 1e-12, max_iter = 200)
  cf <- as.numeric(coef(fit))
  d0 <- abs(cf[1] - row$b0_py); d1 <- abs(cf[2] - row$b1_py)
  # the moments must agree too, otherwise the comparison is not like-for-like
  dm <- max(abs(fit@m2 - row$m2), abs(fit@m3 - row$m3), abs(fit@m4 - row$m4))
  res <- rbind(res, data.frame(dataset = row$dataset, eps = row$eps_key, n = row$n,
                               b0_R = cf[1], b0_py = row$b0_py,
                               b1_R = cf[2], b1_py = row$b1_py,
                               max_dbeta = max(d0, d1), max_dmoment = dm,
                               pass = (max(d0, d1) < tol)))
}

cat(sprintf("%-8s %-10s %6s %14s %14s %12s %12s %6s\n",
            "dataset", "eps", "n", "b1_R", "b1_py", "max|dbeta|", "max|dmoment|", "gate"))
for (i in seq_len(nrow(res))) {
  r <- res[i, ]
  cat(sprintf("%-8d %-10s %6d %14.10f %14.10f %12.3e %12.3e %6s\n",
              r$dataset, r$eps, r$n, r$b1_R, r$b1_py, r$max_dbeta, r$max_dmoment,
              ifelse(r$pass, "PASS", "FAIL")))
}

write.csv(res, file.path(fixdir, "g0_result.csv"), row.names = FALSE)
overall <- all(res$pass)
cat(sprintf("\n[%s] G0  TC-PMM2(U=0) == EstemPMM::lm_pmm2, max|dbeta| = %.3e < %.0e\n",
            ifelse(overall, "PASS", "FAIL"), max(res$max_dbeta), tol))
cat(sprintf("%d/%d datasets passed\n", sum(res$pass), nrow(res)))
if (!overall) quit(status = 1)
