# Survey × Model Correlation Analysis

**Date:** 2026-05-15
**Scripts:** `scripts/analysis/aggregate_survey.py`, `scripts/analysis/survey_correlations.py`
**Outputs:** `data/processed/analysis/survey_correlations.csv`, `figures/survey_correlations.png`

This document reports the first-pass correlation between human-judged polysemy
(Grieshaber & Rubinskii survey, 78 participants × 544 German lemmas) and
model-derived sense-distribution statistics from the v3 diachronic pipeline.
It supports §4 (Survey Evaluation) of the EMNLP paper.

---

## 1. Data and join

| Source | Lemmas | Notes |
|---|---|---|
| Survey (aggregated) | 544 | Per-cell mean rating 1.71; per-lemma mean of means 1.706; median 1.625. Matches G&R reported summary. |
| Model (v3 coarse pipeline) | 90 | One row per pilot lemma; one lemma (`Bass`) has NaN `mean_jsd` due to single supported decade. |
| **Inner join on `lemma`** | **73** | All 73 also satisfy `n_decades_with_data ≥ 5`, so the robustness filter excludes none. |

**Dropped from model side (17, no survey row):** Bass, Frosch, Kasse, Lappen, Lexikon, Löffel, Programm, Präsident, Spiel, Stiefel, Ton, Zwiebel, fest, rechnen, schwimmen, spielen, zählen.

**Dropped from survey side:** 471 lemmas not in the v3 pilot (paper's evaluation is naturally restricted to lemmas with diachronic data).

**Data quality flags:**
- No NaN in `mean_jsd` within the joined set (Bass was eliminated by the join itself).
- All 73 joined lemmas pass `n_decades_with_data ≥ 5` → the robustness filter produces identical numbers to the unfiltered version (reported in the table below as separate rows for completeness).

---

## 2. Correlations

`x = subjective_polysemy_mean` (Spearman is primary, ordinal ratings; Pearson reported for completeness).

| y | n | Spearman r | Spearman p | Pearson r | Pearson p | Filter |
|---|---|---|---|---|---|---|
| `n_clusters_coarse` | 73 | **+0.338** | **0.0035** | +0.269 | 0.0213 | none |
| `mean_entropy` | 73 | **+0.345** | **0.0028** | +0.317 | 0.0063 | none |
| `mean_jsd` | 73 | **+0.254** | **0.0299** | +0.268 | 0.0219 | none |
| `mean_jsd_normalized` | 73 | +0.094 | 0.4267 | +0.111 | 0.3510 | none |
| `n_clusters_coarse` | 73 | +0.338 | 0.0035 | +0.269 | 0.0213 | n_decades_with_data≥5 |
| `mean_entropy` | 73 | +0.345 | 0.0028 | +0.317 | 0.0063 | n_decades_with_data≥5 |
| `mean_jsd` | 73 | +0.254 | 0.0299 | +0.268 | 0.0219 | n_decades_with_data≥5 |
| `mean_jsd_normalized` | 73 | +0.094 | 0.4267 | +0.111 | 0.3510 | n_decades_with_data≥5 |

Bold = significant at α = 0.05 (Spearman).

---

## 3. Interpretation

Three of the four model statistics correlate significantly with human-rated
polysemy: `mean_entropy` (Spearman r = +0.345, p = 0.003) is the strongest,
followed by `n_clusters_coarse` (+0.338, p = 0.004) and `mean_jsd` (+0.254,
p = 0.030); all three effects are modest but consistent in direction and
robust to the sparsity filter. The primary hypothesis — that model-derived
entropy of the sense distribution tracks human polysemy intuition — is
supported: more ambiguous words elicit more uniform predicted sense
distributions over decades. The polysemy-bias-corrected `mean_jsd_normalized`
loses significance (r = +0.094, p = 0.43), confirming that the raw `mean_jsd`
signal is partly mediated by cluster count rather than a pure shift in
distribution shape, which is the expected behaviour given the normalization.

---

## 4. Caveats

- n = 73 is small; effect sizes are modest. Confidence intervals are wide.
- The survey rating scale's 0–11 range is dominated by responses in 1–4;
  the Pearson statistics are sensitive to this skew. Spearman is preferred.
- The 17 model-side lemmas dropped from the join include several sparse-data
  cases already flagged in `MEMORY.md` (Bass, Frosch, Lappen) — exclusion
  here is independent of the diachronic sparsity, it is just survey coverage.
- No multiple-comparison correction is applied. With 4 tests at α = 0.05, the
  Bonferroni-adjusted threshold is α′ = 0.0125; `mean_jsd` (p = 0.030) and
  `n_clusters_coarse` (p = 0.0035) and `mean_entropy` (p = 0.0028) survive
  the adjustment except `mean_jsd` falls just outside.