# English Survey × Hu et al. (2019) Correlation Analysis

**Date:** 2026-05-15
**Scripts:** `scripts/analysis/aggregate_survey_english.py`, `scripts/analysis/survey_correlations_english.py`
**Outputs:** `data/processed/analysis/survey_correlations_english.csv`, `figures/survey_correlations_english.png`

Cross-linguistic counterpart to the German Survey × Model correlation
(`documentation/survey_correlations_2026-05-15.md`). Tests whether the same
C2 (human ↔ model) alignment we found in German also holds in English, using
Amandine Grieshaber's English polysemy survey and Hu et al. (2019)'s
per-word sense statistics.

---

## 1. Data and join

| Source | Lemmas | Notes |
|---|---|---|
| English survey (aggregated) | 558 | 67 participants × 558 lemmas. Ratings 0–20 with mode 2; per-cell mean 2.05; per-lemma mean of means 2.037 (matches G&R reported ≈ 2.04 ✓). |
| Hu et al. (2019) per-word metrics | 3,220 | Per-word entropy & JSD computed from raw `y` in `prob_fitting_10.data`. 138 empty payloads in the pickle were skipped; 26 lemmas have NaN `mean_jsd` (single-decade data). |
| **Inner join on `lemma`** | **557** | Nearly the entire English survey is covered by Hu et al. |

**Dropped from survey side (1):** `liable`.

**Dropped from model side (2,663 of 3,220):** Hu et al. covers a much larger vocabulary than the survey targeted; the dropped lemmas are simply lemmas the survey didn't ask about (e.g. `abandon`, `ability`, `about`, `above`, …).

**Data-quality flags:**
- 0 NaN `mean_jsd` in the joined set (the 26 single-decade Hu lemmas all fell outside the survey).
- No support filter applied (the only available column was `n_decades`, not used in this pass).

---

## 2. Correlations

`x = subjective_polysemy_mean` (Spearman primary, Pearson reported for completeness).

| y | n | Spearman r | Spearman p | Pearson r | Pearson p |
|---|---|---|---|---|---|
| `n_senses` | 557 | **+0.680** | **< 0.0001** | +0.628 | < 0.0001 |
| `mean_entropy` | 557 | **+0.586** | **< 0.0001** | +0.593 | < 0.0001 |
| `mean_jsd` | 557 | **+0.537** | **< 0.0001** | +0.499 | < 0.0001 |

All three correlations are highly significant and survive any reasonable
multiple-comparison correction.

---

## 3. Side-by-side: German vs. English

Direct comparison with the German results
(`documentation/survey_correlations_2026-05-15.md`):

| Pair | German Spearman r | German p | German n | English Spearman r | English p | English n |
|---|---|---|---|---|---|---|
| n_clusters_coarse / n_senses | +0.338 | 0.0035 | 73 | **+0.680** | < 0.0001 | 557 |
| mean_entropy | +0.345 | 0.0028 | 73 | **+0.586** | < 0.0001 | 557 |
| mean_jsd | +0.254 | 0.0299 | 73 | **+0.537** | < 0.0001 | 557 |

English correlations are roughly **2× stronger** than the German ones across
all three pairs. n is also 7.6× larger on the English side, so the contrast
is partly a power effect on detectable effect-size estimates, but the
point-estimate gap is substantial enough to be substantive.

---

## 4. Interpretation

The C2 hypothesis (model-derived sense statistics track human polysemy
judgments) is supported cross-linguistically and more strongly in English
than German: every Spearman correlation in English is highly significant
(p < 0.0001) and noticeably larger than the German counterpart. The
strongest English predictor is `n_senses` (r = +0.680), suggesting English
participants are largely counting senses when rating polysemy — Hu et al.'s
WordNet-derived sense inventory aligns with that intuition almost
directly. The gap between German and English (e.g. mean_entropy r = 0.345
vs. 0.586) is consistent with two non-exclusive explanations: (a) Hu et al.'s
COHA-fitted WordNet sense distributions are cleaner / more lexicographically
calibrated than our GermaNet → LLM-enriched → coarse-clustered German
pipeline, and (b) the German pilot covers only 90 lemmas → 73 in the join,
yielding wider confidence intervals on point estimates.

---

## 5. Caveats

- The 67 English-survey rows include 67 IDs (CASE 46 onwards, non-contiguous);
  one row may be a screening response. Aggregation treats them all uniformly.
- The English rating scale (0–20) is wider than German (0–11) but follows the
  same heavy-tail-toward-1 shape — Spearman is robust to this.
- Hu et al.'s `n_senses` correlation (+0.680) is the highest because subjective
  polysemy is essentially measuring "how many senses do I notice?"; this is a
  ceiling argument for the upper bound of any model-based correlation.
- No frequency-weighted or POS-stratified analysis here — pure per-lemma join.
- No multiple-comparison correction needed: all p-values < 0.0001 ≪ α′ = 0.0167
  (Bonferroni for 3 tests at α = 0.05).