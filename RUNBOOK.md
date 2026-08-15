# RUNBOOK

Execution notes for the Ku-PMM-EIV verification supplement.
Last updated: 2026-08-15.

Three levels of checking, in increasing cost. Level 1 is enough to verify the
manuscript; levels 2 and 3 exist so that "verify" does not have to mean "trust
the stored files".

---

## Level 1 — the printed numbers (seconds, no network, no R)

```sh
pip install -r requirements.txt
python scripts/verify_reported_values.py
```

Expected: `all 43 reported values reproduced from stored evidence`, exit 0.

Each check names the manuscript section it comes from. The expected values are
transcribed from the manuscript at the precision it prints them; the actual
values are recomputed from `evidence/`. A mismatch means the manuscript and the
evidence disagree — it is never a reason to edit the expected value.

## Level 1b — regenerate the tables and figures

```sh
python scripts/make_tables.py     # -> output/tables/*.tex
python scripts/make_figures.py    # -> output/figures/*.pdf
```

These read `evidence/` and write the exact LaTeX tables and PDF figures used in
the manuscript. Nothing in either script contains a typed-in number.
`output/figures/figure_data_provenance.json` records which evidence file every
panel and every table came from.

## Level 2 — the pre-registered gates (seconds)

```sh
python scripts/verify.py --run confirmatory
```

**This exits 1, and that is the reported result.** Seven of ten gates pass:

| Gate | Outcome | |
|---|---|---|
| G0, G0b | PASS | reduction to the error-free estimator at `U ≡ 0` |
| C1 | PASS | independently typed algebra vs the symbolic derivation |
| G1 | PASS | unbiasedness, 576/576 |
| **G2** | **FAIL 133/144** | measured vs closed-form efficiency within 2 MC SE |
| G2b | PASS | `ARE* ≤ 1`, 144/144 |
| **G3** | **FAIL 94/96** | cost of adaptive weights ≤ 10% |
| G4 | PASS | sandwich coverage in [93,97]%, 480/480 |
| G5 | PASS | Gaussian-error control |
| **G6** | **FAIL 52/72** | heavy-tailed lognormal measurement error |

The manuscript analyses each failure (Section 8.9) instead of relaxing the gate;
thresholds were fixed before the confirmatory run. `verify_reported_values.py`
asserts these four counts, so a supplement in which the gates quietly passed
would be the one to distrust.

Other stored runs accept the same flag: `--run lambda_star`, `--run sensitivity`,
`--run negative_nu3`.

## Level 3 — regenerate the evidence from seeds

```sh
python scripts/run_study.py --mode confirmatory     # 180 cells, M=2000, ~22 min
python scripts/run_study.py --mode lambda_star      # 108 cells,        ~22 min
python scripts/run_study.py --mode sensitivity      #  28 cells,        ~2 min
python scripts/run_study.py --mode asymptotics      # closed forms only, seconds
python scripts/run_study.py --mode negative_nu3     # 156 cells,        ~5 min
python scripts/run_study.py --mode robustness       #  12 cells (non-Gaussian X + SIMEX), ~7 min
```

Wall times are from the reported runs (9 workers). Each mode overwrites
`evidence/<mode>/` and writes a fresh `manifest.json`.

Randomness is a `numpy` `SeedSequence` spawned as
`[base, key(stream), key(cfg), n, rep]` with base entropy **20260814**, recorded
in every manifest. Within a replication all estimators see the same data (common
random numbers), and the latent covariate and regression error are shared across
reliability levels — which is why the confirmatory and `lambda_star` runs agree
digit for digit on the 432 summary cells they have in common.

A quick smoke test of the whole pipeline, if you only want to know that it runs:

```sh
python scripts/run_study.py --mode quick --reps 20 --run-name smoke \
    --reduction-reason "smoke test"
```

Any reduction of `M` below its default is recorded in the manifest, so a reduced
run can never be mistaken for a reported one.

### Gate G0 needs R

```sh
Rscript scripts/oracle_check.R      # requires the CRAN package EstemPMM
```

G0 checks that the corrected score collapses onto the published degree-two
estimator when the measurement error is switched off. `verify.py` reads the
stored result of this check; re-run it only if you want the cross-check against
an independent implementation rather than against a stored file. G0b does the
same comparison against a second, pure-Python solver and needs no R.

## The theory side

```sh
python theory/are_symbolic.py           # derives the closed forms with sympy
python theory/are_numeric_check.py      # independent numerical cross-check
python theory/replicate_identification.py
```

`are_symbolic.py` derives the three second-moment blocks and the efficiency
formula; `are_formulas.py` is its generated output and is what the estimators
import. `are_numeric_check.py` re-derives the same quantities by a second,
independent route and compares — the manuscript quotes their agreement to
2.963e-16 over 96 configurations. `replicate_identification.py` demonstrates the
result behind Remark 2.1: with two replicate measurements per subject the odd
moments of the measurement error are not identified at any sample size, and with
three or more they are.

Both `theory/*_out.json` files shipped here are the stored outputs the manuscript
cites; re-running the scripts overwrites them with fresh draws, so `git diff`
after a run is expected and not a failure.

## The real-data studies

These download their inputs; nothing third-party is redistributed here. See
`docs/DATA_POLICY.md` before running them.

```sh
python scripts/realdata_nhanes.py        # NHANES 2013-2014, ~50 MB download
python scripts/realdata_census_nmf.py    # 2020 Census NMF, ~41 MB parquet
python scripts/realdata_nmf_svi.py       # NMF covariate x SVI response, 462 cells
```

Each verifies the SHA-256 of every file it fetches against the digest stored in
the corresponding `evidence/*/manifest.json` and refuses to proceed on a
mismatch, so an upstream reissue is detected rather than silently absorbed.
Their stored outputs are already in `evidence/`, so level 1 verifies the
real-data numbers without any download.
