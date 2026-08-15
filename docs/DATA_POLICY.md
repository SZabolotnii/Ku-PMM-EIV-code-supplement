# Data and redistribution boundary

This repository contains **code and derived results only**. No third-party data
file is redistributed here, and no manuscript source is included.

## What is in the repository

- `evidence/` — results *derived* from the sources below: per-cell summaries,
  paired comparisons, run manifests and gate reports. These are our own output.
- `output/` — tables and figures regenerated from `evidence/`.
- `scripts/`, `theory/` — our code.

## What is not, and how to obtain it

Each real-data script downloads its own inputs into `data/` (git-ignored) and
verifies the SHA-256 of every file against the digest recorded in the
corresponding `evidence/*/manifest.json`. A mismatch is reported, never
silently absorbed.

| Source | Used by | Provenance |
|---|---|---|
| NHANES 2013–2014 (`BPX_H`, `BIOPRO_H`, `DEMO_H`) | `scripts/realdata_nhanes.py` | US Centers for Disease Control and Prevention. A work of the United States government, in the public domain. Retrieved by unauthenticated request from `wwwn.cdc.gov`. |
| 2020 Census Noisy Measurement File | `scripts/realdata_census_nmf.py` | US Census Bureau, released 2023 under a fixed S3 prefix. A dated, immutable release; public domain. |
| CDC/ATSDR Social Vulnerability Index 2020 | `scripts/realdata_nmf_svi.py` | CDC/ATSDR, published state CSVs; public domain. |

All three are United States government works and could in principle be
redistributed. We do not, for two reasons that are worth stating rather than
assuming: the files are large relative to this repository, and NHANES
occasionally reissues a file under the same URL — a pinned digest plus a fresh
download makes such a reissue *visible*, whereas a vendored copy would quietly
hide the divergence.

Because the derived results are shipped, the manuscript's real-data numbers can
be verified with `scripts/verify_reported_values.py` **without downloading
anything**. The downloads are needed only to reproduce those derived results
from the raw sources.

## One disclosed edit to the shipped evidence

The run manifests and gate reports record the absolute path and interpreter of
the machine that produced them. Those strings were rewritten to `<repo>` and
`<home>` before publication — 18 lines across 14 files, all of them path
metadata. No number, seed, digest or result was touched. The parent repository
retains the originals.

## Deliberately excluded

- **Raw Monte Carlo replications** (`evidence/*/estimates_raw.csv.gz`, ~230 MB) —
  regenerable from the seed hierarchy recorded in each `manifest.json`.
- **The manuscript and its LaTeX sources** — this is a code supplement, not a
  preprint mirror.
- **Copyrighted literature.** The parent research directory keeps PDFs of the
  works this paper builds on, including a doctoral thesis obtained from a
  national-library deposit copy. Those are licensed to their publishers and
  depositing libraries, not to us, and none is included here. Every one is cited
  in the manuscript with a resolvable identifier.
- **Internal research notes** — prior-art assessments, specifications and
  progress reports live in the private parent repository. What they license is a
  claim in the manuscript; what verifies a claim is the code and evidence here.
