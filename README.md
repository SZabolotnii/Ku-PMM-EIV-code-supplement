# Ku-PMM-EIV Code Supplement

Public verification supplement for the manuscript:

**Optimal weighting of corrected degree-two scores in errors-in-variables models**
*Serhii Zabolotnii and Vladyslav Khotunov*

Repository: <https://github.com/SZabolotnii/Ku-PMM-EIV-code-supplement>

Archived on Zenodo: release v1.0.0 is
[doi:10.5281/zenodo.23482300](https://doi.org/10.5281/zenodo.23482300); the concept DOI
[10.5281/zenodo.23482299](https://doi.org/10.5281/zenodo.23482299) always resolves to the
latest release.

Release v1.0.0 is the state cited by the shortened version of the manuscript (21 pages in
the journal's review format, with a 31-page Supplementary Material). Earlier commits served longer
versions of the same manuscript; section and figure numbers below refer to v1.0.0.

## What this repository is for

The manuscript states a negative result and a constructive one, and every number
it prints comes from a stored evidence file rather than from a hand transcription.
This repository ships that evidence together with the code that produced it, so
that a reader can check any printed number in **seconds**, and re-derive it from
scratch in **minutes to an hour** if they want to.

Start here:

```sh
python scripts/verify_reported_values.py
```

It re-computes 76 reported numbers from `evidence/` and compares each against the
value printed in the manuscript, at the precision the manuscript prints it. It
needs no simulation, no network and no R.

## The one thing to know before running the gates

```sh
python scripts/verify.py --run confirmatory      # exits 1, by design
```

**Three of the ten pre-registered gates fail, and that is the manuscript's own
reported result**, not a broken supplement. G2 (measured versus closed-form
efficiency) passes 133/144 sub-checks, G3 (cost of adaptive weights) 94/96, and
G6 (heavy-tailed lognormal measurement error) 52/72; the paper reports all three
failures in Table 2 and Section 4.4 ("What failed"), with the detail in Sections
S2.10–S2.11 of its Supplementary Material, rather than hiding them. `verify.py` therefore exits
non-zero, and `verify_reported_values.py` asserts that the failures are *exactly*
those counts — a supplement in which the gates silently passed would be the
suspicious one.

## Layout

| Path | What it holds |
|---|---|
| `scripts/verify_reported_values.py` | headline check: every printed number against stored evidence |
| `scripts/verify.py` | the pre-registered gates G0–G6 over a stored run |
| `scripts/run_study.py` | the Monte Carlo driver (regenerates `evidence/<run>/`) |
| `scripts/dgp.py`, `estimators.py`, `mc_engine.py` | data-generating processes, the five estimators, the replication loop with common random numbers |
| `scripts/make_tables.py`, `make_figures.py` | regenerate every manuscript table and figure into `output/` (article Figure 1 = `fig1_are_theory_vs_mc.pdf`, Figure 2 = `fig8_validation.pdf`; the others belong to the Supplementary Material) |
| `scripts/realdata_*.py` | the three real-data studies (data are downloaded, not redistributed — see `docs/DATA_POLICY.md`) |
| `scripts/oracle_check.R` | gate G0 only: cross-check against the R package `EstemPMM` |
| `theory/` | the symbolic derivation, an independent numerical cross-check, the generated closed forms, and the replicate-identification argument |
| `evidence/` | stored results of every run cited by the manuscript |
| `output/` | tables and figures regenerated from `evidence/` |

## Estimators

Five appear in the manuscript and in the evidence files under these names:

- `naive_ls`, `naive_pmm2` — uncorrected, shown as lower baselines;
- `tc_ls` — corrected least squares (Fuller), the main competitor;
- `tc_pmm2_oracle` / `tc_pmm2_adaptive` — the corrected degree-two score carrying
  the weight transplanted from the error-free problem; this is the estimator the
  negative half of the paper argues against;
- `tc_pmm2star_oracle` / `tc_pmm2star_adaptive` — the same score with the weight
  re-optimised in context; this is the paper's proposal;
- `ref_ls`, `ref_pmm2` — fitted on the unobserved true covariate, as unattainable
  references.

`oracle` means the moments of the regression error are supplied exactly;
`adaptive` means they are recovered from the data by cumulant deconvolution.

## What the evidence does and does not contain

Shipped: the per-cell summaries, the paired comparisons, the run manifests (seed
hierarchy, wall time, software versions, any design reduction) and the
machine-written gate reports — everything the manuscript cites.

Not shipped: the raw per-replication draws (`estimates_raw.csv.gz`, ~230 MB).
They are fully regenerable from the seeds recorded in each `manifest.json`; see
RUNBOOK.

## Requirements

Python 3.11+ with `numpy`, `scipy`, `sympy`, `matplotlib`, `pandas`, `pyarrow`
(`pip install -r requirements.txt`). The runs reported in the manuscript used
Python 3.14.5, numpy 2.5.2, scipy 1.18.0, sympy 1.14.0.

R with [`EstemPMM`](https://cran.r-project.org/package=EstemPMM) is needed **only**
for gate G0, which checks that the corrected score collapses onto the published
degree-two estimator when the measurement error is switched off. Everything else
is pure Python.

## Licence

Code: MIT (see `LICENSE`). The manuscript itself is not part of this repository.
Third-party data are neither included nor redistributed — see
`docs/DATA_POLICY.md`.
