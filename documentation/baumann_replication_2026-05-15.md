# §5 Baumann Replication — German Diachronic Pilot

**Date:** 2026-05-15
**Scripts:** `scripts/analysis/baumann_replication.py` (consumes
`scripts/analysis/baumann_predictors.py` output)
**Outputs:**
- `data/processed/analysis/baumann_bivariate.csv`
- `data/processed/analysis/baumann_regression.csv`
- `data/processed/analysis/hamilton_innovation.csv`
- `figures/baumann_replication.png`

Tests Baumann et al. (2023)'s diversification predictions on the German
v3 pilot, with `entropy_slope` (slope of decade-level Shannon entropy) as
the diversification proxy. All four numeric predictors are
z-standardised before regression so coefficients are directly comparable
to Baumann's reported English `b` values.

---

## 1. Data

Restricted to lemmas with complete data across `entropy_slope`,
`log_frequency`, `discriminability_desc`, and `concreteness`.

**n = 87** (of 90 v3 pilot lemmas; 3 excluded due to NaN on at least one
predictor — Bass with single supported decade, and two single-sense
lemmas with NaN `discriminability_desc`).

---

## 2. Bivariate correlations

`entropy_slope` vs. each predictor, on z-standardised data (Spearman and
Pearson are scale-invariant, so standardisation does not affect r or p
here — reported jointly for downstream use). Baumann's directional
prediction is in the final column.

| Predictor | Spearman r | Spearman p | Pearson r | Pearson p | n | Baumann predicts |
|---|---|---|---|---|---|---|
| log_frequency | -0.044 | 0.6857 | +0.035 | 0.7503 | 87 | − |
| discriminability_desc | **+0.220** | **0.0406** | +0.130 | 0.2303 | 87 | + |
| concreteness | +0.201 | 0.0623 | +0.182 | 0.0915 | 87 | + |

Bold = significant at α = 0.05 (Spearman, primary).

---

## 3. Multiple regression (standardised coefficients)

### Model A — description-based discriminability

`entropy_slope ~ log_frequency + discriminability_desc`

| Predictor | b | SE | t | p | 95 % CI |
|---|---|---|---|---|---|
| const | +0.000 | 0.107 | +0.00 | 1.0000 | [−0.214, +0.214] |
| log_frequency | +0.046 | 0.108 | +0.42 | 0.6744 | [−0.170, +0.261] |
| discriminability_desc | +0.134 | 0.108 | +1.23 | 0.2209 | [−0.082, +0.349] |

**Model fit:** n = 87, R² = 0.019, adj-R² = -0.004, F p = 0.4476 → not significant overall.

### Model B — concreteness as discriminability proxy

`entropy_slope ~ log_frequency + concreteness`

| Predictor | b | SE | t | p | 95 % CI |
|---|---|---|---|---|---|
| const | +0.000 | 0.105 | +0.00 | 1.0000 | [−0.208, +0.208] |
| log_frequency | +0.234 | 0.133 | +1.76 | 0.0826 | [−0.031, +0.500] |
| concreteness | **+0.326** | 0.133 | +2.44 | **0.0167** | [+0.060, +0.591] |

**Model fit:** n = 87, R² = 0.067, adj-R² = 0.045, F p = 0.0533 → marginally significant overall.

---

## 4. Hamilton bonus — Law of Innovation at the sense level

Does total polysemy (`n_clusters_coarse`) predict diachronic redistribution
(`mean_jsd`)?

| Pair | n | Spearman r | Spearman p | Pearson r | Pearson p |
|---|---|---|---|---|---|
| n_clusters_coarse ↔ mean_jsd | 89 | **+0.714** | **< 0.0001** | +0.732 | < 0.0001 |

Strong and highly significant — Hamilton's claim that more polysemous
words show more sense-distribution change holds clearly at the
coarse-cluster level for German.

---

## 5. Interpretation (3 sentences)

Of Baumann's three diversification predictors, only **`discriminability_desc`
replicates as a bivariate correlate of `entropy_slope`** at p < 0.05
(Spearman r = +0.220, p = 0.041) — and only **concreteness emerges as a
significant predictor in the joint regression** (Model B b = +0.326,
p = 0.017), with frequency showing essentially zero effect (Spearman
r = −0.044, p = 0.69) and the wrong sign in both regression models.
Compared to Baumann's reported English coefficients (frequency
b ≈ −0.20 to −0.48; discriminability b ≈ +0.10 to +0.23), our
discriminability and concreteness magnitudes are in-range or slightly
larger, but **the strong negative frequency effect that anchors Baumann's
analysis fails to replicate in German**. The Hamilton-style polysemy ↔
change-rate correlation (r = +0.71) is by contrast a complete and very
strong replication, suggesting the diachronic-change signal in our pipeline
is robust at the lemma-aggregate level even when finer-grained Baumann
predictors are noisy.

---

## 6. Caveats

- n = 87 is small; bivariate correlations have wide 95 % CIs (roughly
  ±0.20 around point estimates of r).
- `entropy_slope` is centred near 0 with a narrow spread (mean −0.0002,
  std 0.0024); few lemmas have an |slope| that's individually significant,
  so the dependent variable is itself noisy. Trends may sharpen with the
  full 503-lemma annotation once Amandine + Nadja merge in.
- `discriminability_desc` is computed from a single embedding model
  (E5-large `weighted`); other representations (LLM definitions only,
  or alternative encoders) may yield different magnitudes.
- The `log_frequency` sign flip is striking but not necessarily anomalous:
  the German pilot is stratified by polysemy rather than frequency, so the
  frequency range is narrower than Baumann's English vocabulary.
- All p-values reported uncorrected for multiple comparisons. With three
  bivariate tests Bonferroni-adjusted threshold is α′ = 0.0167 → only
  Model B's concreteness coefficient remains significant under that
  stricter threshold. The Hamilton bonus passes by orders of magnitude.