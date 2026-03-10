# Coarse Sense Clustering — Implementation & Research Plan

**Status:** Planning (as of 2026-03-10)
**Motivation:** Diachronic trajectories from the 90-word pilot are noisy. 
Primary hypothesis: GermaNet is too fine-grained for stable diachronic sense tracking. 
Coarse clustering of sense inventories is the proposed remedy before any scale-up.

---

## 1. Current Repo Understanding

### Pipeline as implemented

| Stage | Key files |
|---|---|
| GermaNet parsing | `scripts/data/process_germanet.py` → `data/processed/germanet/senses_base.csv` (2,022 rows) |
| LLM enrichment | `scripts/data/enrich_senses_with_llm.py` → `senses_full.jsonl` (752 words) / `senses_pilot.jsonl` (90 words, 595 senses) |
| Sense embeddings | `src/wsd/embedding_strategies.py::embed_with_representation("weighted")` — `0.5 * E5-large(llm_definition) + 0.5 * mean(E5-large(llm_example_1..5))`, 1024-dim, L2-normalized |
| WSD prediction | `scripts/diachronic/predict_dwds_senses.py` → `data/processed/diachronic/predictions/pilot_predictions.csv` (24,325 rows) |
| Aggregation | `scripts/diachronic/aggregate_temporal.py` → `aggregated/pilot_counts.csv`, `pilot_stats.csv` |
| Change metrics | `scripts/diachronic/compute_change_metrics.py` → `metrics/pilot_entropy.csv`, `pilot_divergence.csv`, `pilot_emergence.csv`, `pilot_decline.csv` |

**Predictions CSV columns:** `lemma, sentence, year, decade, genre, predicted_sense_id, confidence, sense_class, sense_paraphrase, date, bibl, corpus`

**Sense IDs throughout:** `synset_id` (e.g. `s56941`). One sense = one GermaNet synset.

---

## 2. Diagnosis of the Current Problem

Three distinct noise sources with different remedies:

### Source 1: Fine-grained senses with near-identical embeddings (→ clustering fixes this)
GermaNet verbs often have 6–9 senses within the same semantic class (e.g. `schlagen` has 6 `Kontakt` senses). In a 1024-dim space these cluster very tightly. The model is essentially choosing between near-tied candidates on every prediction. Oscillations between these senses decade-to-decade drive fake JSD.

### Source 2: Data sparsity per word-decade (→ clustering helps, does not fully fix)
Words like `groß` have only ~47 total contexts across 10 decades (~5/decade). With 5–9 fine senses, a single misclassification swings a proportion by 17–50%. Coarse merging reduces the number of competing categories, which helps, but words with fewer than ~100 total contexts remain unreliable regardless of clustering level.

### Source 3: A pre-existing bug in `load_senses()` — fixed

**Found and fixed (2026-03-10).** `src/diachronic/io.py::load_senses()` called `pd.read_csv()` on `senses_pilot.jsonl`. Pandas cannot meaningfully parse JSONL as CSV — it would produce a mangled DataFrame with no `llm_definition` column, causing `embed_with_representation("weighted")` to raise a `ValueError` immediately. The prediction script would crash rather than silently degrade.

**What this means for the existing predictions:** `pilot_predictions.csv` was generated before the `Rework data-dir structure` commit (which moved and converted the senses file to JSONL). At that point the senses file was still a CSV with proper columns, so the predictions are **likely valid**. The bug only affects any future re-run of `predict_dwds_senses.py`.

**Fix applied:** `load_senses()` now detects `.jsonl` vs `.csv` by extension. For JSONL it uses `pd.read_json(lines=True)` and expands `llm_examples` → `llm_example_1..5` (1-indexed, matching `embedding_strategies.py`). The CSV path is unchanged.

### Inconsistencies between docs and code

| Inconsistency | Impact |
|---|---|
| `pipeline.md` references `pilot_predictions_weighted_v2.csv`, `pilot_weighted_v2_decade_sense_counts.csv` — neither exists | Docs are out of date with current data dir structure |
| `pipeline.md` references `src.data.sense_representations` — this module doesn't exist | Actual module is `src/wsd/embedding_strategies.py` |
| `compare_embedding_methods.py` uses 0-indexed `llm_example_0..4`; `embedding_strategies.py` expects `llm_example_1..5` | These paths never intersect, but a trap when writing new code — always follow `embedding_strategies.py` |

---

## 3. Recommended Clustering Strategy

One concrete approach — not a menu of options.

**Unit:** Cluster per lemma. Sense similarity only matters within a word's own sense inventory.

**Representation:** The same weighted embedding already used for WSD — `embed_with_representation("weighted")` from `src/wsd/embedding_strategies.py`. This ensures clusters correspond exactly to the senses the WSD model itself confuses. A prediction that was "wrong" at fine level (assigned to a sibling sense in the cluster) becomes "correct" at coarse level.

**Algorithm:** `sklearn.cluster.AgglomerativeClustering`

**Linkage:** Average (UPGMA). GermaNet sense neighborhoods are not tight spheres — average linkage reflects true inter-cluster mean distance rather than being dominated by outliers (complete linkage) or chaining artifacts (single linkage). Most defensible choice for semantic sense groupings.

**Distance metric:** Cosine distance (`1 - cosine_similarity`). E5-large embeddings are already L2-normalized, so this equals `1 - dot_product`.

**Stopping criterion:** Distance threshold starting at **0.25** (cosine similarity > 0.75). Rationale: with 96.8% top-3 accuracy, the true sense is almost always within cosine distance ~0.1–0.2 of the prediction. Senses within distance 0.25 are senses the model routinely confuses in practice. This is the threshold to validate first; 0.20 and 0.30 are the immediate alternatives if 0.25 looks wrong on inspection.

**Fallback constraints:**
- Minimum 2 clusters per lemma (never collapse to a single category)
- Maximum clusters = `max(2, n_senses // 2)` (guarantee meaningful reduction)
- Words with 1 sense: pass through unchanged, map to `"{lemma}_c0"`

---

## 4. Integration into This Repo

### Design principle: add new files only, do not modify existing pipeline scripts

```
src/wsd/
  sense_clustering.py              ← new module

scripts/
  compute_sense_clusters.py        ← generate mapping + visual artifacts
  review_sense_clusters.py         ← generate Markdown review sheets per lemma

scripts/diachronic/
  apply_coarse_mapping.py          ← transform fine predictions → coarse

data/processed/germanet/
  sense_clusters_0.25.json         ← primary output: fine→coarse mapping
  sense_clusters_0.25_reviewed.json ← post-manual-review approved version

data/processed/sense_clustering/
  dendrograms/                     ← {lemma}_dendrogram.png (90 files)
  similarity_matrices/             ← {lemma}_sim_matrix.png (90 files)
  review/
    {lemma}_review.md              ← one per lemma (90 files)
    _summary.md                    ← all-lemmas overview

data/processed/diachronic/
  predictions/
    pilot_predictions.csv          (unchanged)
    pilot_predictions_coarse_0.25.csv  ← new
  aggregated/
    pilot_counts_coarse_0.25.csv   ← new
    pilot_stats_coarse_0.25.csv    ← new
  metrics/
    pilot_entropy_coarse_0.25.csv  ← new
    pilot_divergence_coarse_0.25.csv  ← new
    pilot_emergence_coarse_0.25.csv   ← new
    pilot_decline_coarse_0.25.csv     ← new
```

### Mapping file format (`sense_clusters_0.25.json`)

```json
{
  "schlagen": {
    "s56941": "schlagen_c0",
    "s56959": "schlagen_c0",
    "s56972": "schlagen_c0",
    "s56930": "schlagen_c0",
    "s56713": "schlagen_c1",
    "s56641": "schlagen_c2"
  },
  "gehen": { ... }
}
```

Cluster IDs use `"{lemma}_c{index}"` — globally unique and human-readable.

### Key functions in `sense_clustering.py`

- `cluster_senses_for_lemma(embeddings, sense_ids, threshold, min_clusters, max_clusters)` → `Dict[fine_id, coarse_id]`
- `cluster_all_lemmas(senses_df, embedder, threshold)` → full mapping dict
- `generate_coarse_cluster_id(lemma, cluster_index)` → `str`
- `get_cluster_centroids(embeddings, mapping, sense_ids)` → for optional Phase B direct prediction later

### Reuse of existing aggregation and metrics scripts

`aggregate_temporal.py` and `compute_change_metrics.py` can be run **unchanged** on `pilot_predictions_coarse_0.25.csv` — they only operate on `(lemma, decade, predicted_sense_id)`, and the coarse cluster ID is a drop-in replacement for the fine sense ID.

---

## 5. Design Choice: Fine-then-Coarse vs. Direct Coarse Prediction

**Recommendation: Option A (fine predictions → coarse mapping) as the first version.**

| | Option A: fine→coarse mapping | Option B: direct coarse prediction |
|---|---|---|
| Reuses existing predictions | Yes | No — requires GPU re-run |
| Iteration speed on threshold | Minutes (re-apply JSON) | Hours per threshold tested |
| Scientific validity | Valid — merging confusion is the explicit intent | Also valid |
| Provenance | Fine predictions fully preserved | Fine predictions replaced |
| Thesis defensibility | Strong — methodology is transparent | Also strong |
| Implementation complexity | Low | Moderate |

**Why Option A first:** The clustering strategy is explicitly designed to merge senses the model confuses. A fine-level "wrong" prediction assigned to a same-cluster neighbor is correct at coarse level — this is exactly the intended behavior, and it requires no additional compute.

**When Option B is worth doing:** If after Phase 3 per-decade confidence scores are systematically low for certain high-polysemy words (context embeddings sit equidistant between senses in different clusters), Option B (WSD against cluster centroids) is worth running as a verification step. Treat it as Phase 4, not Phase 1.

---

## 6. Top-k / Soft Assignment

Stay with top-1 at coarse level for now. The coarse merging already captures most of what soft assignment would provide: ~45% of fine-level "wrong" predictions assigned to same-cluster neighbors become correct at coarse level, effectively raising accuracy without any model changes.

Boltzmann-weighted assignment over top-3 coarse candidates is a valid Phase 4 extension for specific high-polysemy words that remain noisy after clustering. It adds a temperature hyperparameter that needs calibration — defer until the simpler version is evaluated.

---

## 7. Phased Implementation Plan

### Phase 1: Exploratory Clustering (1–2 days)

**Goal:** Generate cluster artifacts for all 90 words and calibrate the threshold.

**Tasks:**
1. Implement `src/wsd/sense_clustering.py`. Start in `notebooks/analysis/sense_clustering_exploration.ipynb` for interactive development on 3–4 words before writing the script.
2. Write and run `scripts/compute_sense_clusters.py` at threshold 0.25.
3. Write and run `scripts/review_sense_clusters.py`.

**Outputs:**
- `data/processed/germanet/sense_clusters_0.25.json`
- 90 dendrograms, 90 similarity matrices, 90 review Markdown files, `_summary.md`

**Decision gate:** Inspect dendrograms for `schlagen`, `gehen`, `spielen`, `fassen`, `Geist`, `Bank`. For `schlagen`, the 6 `Kontakt` physical-striking senses should merge at distance < 0.25 while semantically distinct senses (defeat in competition, affect someone) stay separate. If everything merges → lower threshold to 0.20. If nothing merges → raise to 0.30.

**Risks:** Very polysemous verbs (14–19 senses) may need per-word threshold overrides. The dendrogram supports this: tree can be cut at any level and the override recorded in `sense_clusters_0.25_overrides.json`.

---

### Phase 2: Manual Review and Threshold Selection (2–3 days)

**Goal:** Produce a thesis-defensible, approved cluster mapping.

**Tasks:**
1. Review all 90 `review/{lemma}_review.md` files. Depth focus on: words with n_senses ≥ 9 (~15 words), and the top-20 words by mean JSD from `pilot_divergence.csv`.
2. Record any overrides in `sense_clusters_0.25_overrides.json` (same structure as main mapping; entries here take precedence at merge time).
3. Document reasoning in `documentation/research_log.md` for any word where automatic clustering was changed.
4. Produce `sense_clusters_0.25_reviewed.json`.

**Outputs:** Reviewed mapping JSON, override file, research log entries.

**Decision gate:** You can articulate a coherent semantic story for each cluster in the top-20 most-changed words. No cluster merges two categorically distinct meanings (e.g. `spielen_instrument` must never merge with `spielen_sport`).

---

### Phase 3: Coarse Diachronic Rerun (1–2 days)

**Goal:** Re-aggregate the fine predictions using the approved mapping and recompute all metrics at coarse level.

**Tasks:**
1. Write and run `scripts/diachronic/apply_coarse_mapping.py` → `pilot_predictions_coarse_0.25.csv`.
2. Run existing `aggregate_temporal.py` on the coarse predictions → coarse counts/stats CSVs.
3. Run existing `compute_change_metrics.py` on coarse counts → all four coarse metrics CSVs.
4. Update `notebooks/analysis/diachronic_pilot_results.ipynb` to load fine and coarse files side-by-side.

**Outputs:** All coarse data files listed in Section 4.

**Decision gate:** For the top-10 high-JSD words, does the coarse analysis show lower period-to-period JSD variance (fewer zigzags) and lower mean entropy (more decisive assignments)? If yes, proceed to Phase 4. If specific words remain just as noisy, flag them as data-sparse and exclude from final analysis.

---

### Phase 4: Validation and Interpretation (3–5 days)

**Goal:** Extract 3–5 thesis-quality case studies.

**Tasks:**
1. Re-rank top-10 most-changed words using coarse metrics. Note: some formerly high-JSD words will calm down (they were noisy, not genuinely changed); others will maintain high JSD (genuine change). Words that calm down are informative — they show the fine-grained method was misleading.
2. For each top candidate, read 15–30 raw sentences from `pilot_predictions.csv` for the key transition decade. Do the sentences confirm the detected shift?
3. Cross-reference with DWDS dictionary change notes and available secondary literature.
4. Write up findings in `documentation/research_log.md`.

**Outputs:** 3–5 documented case studies, updated top-10 ranking, candidate thesis section.

---

## 8. Success Criteria for the 90-word Pilot

| Criterion | How to measure |
|---|---|
| **Lower entropy at coarse level** | Mean entropy across all lemma-decade pairs: `pilot_entropy_coarse` vs. `pilot_entropy`. Target: ≥0.1 bits decrease on average. |
| **Fewer zigzag trajectories** | Per-lemma variance of consecutive JSD values. Target: ≥60% of words show reduced variance at coarse level. |
| **Top-10 are interpretable** | Each word has a one-paragraph semantic story with historical grounding. |
| **Emergence/decline events are plausible** | Spot-check 20 events: does the sentence evidence in the key decade support the detected shift? |
| **≥3 good case studies** | Each must have: ≥10 contexts per decade in the transition period; a monotonic or step-change coarse trajectory across ≥3 consecutive decades; a plausible cultural/historical explanation. |

**Note:** Words with fewer than ~100 total contexts should be flagged as unreliable regardless of clustering level. Coarse clustering reduces sense-confusion noise, not data-sparsity noise.

---

## 9. Immediate Next Actions

### Action 1 — Implement `src/wsd/sense_clustering.py` in a notebook first (day 1)
Use `notebooks/analysis/sense_clustering_exploration.ipynb`. Test on `schlagen`, `Bank`, `Geist`, `spielen`. Generate the dendrogram for `schlagen` and check which senses merge at distance 0.25. This single visual check calibrates the threshold before running all 90 words.

### Action 2 — Write and run the two scripts on all 90 pilot words (day 2)
`scripts/compute_sense_clusters.py` and `scripts/review_sense_clusters.py`. Open `_summary.md` to see sense reduction ratios across the full vocabulary (e.g. "spielen: 12 senses → 5 clusters"). This is the first concrete result to reason about.

---

## Traps to Avoid

1. **Do not use `pd.read_csv` on JSONL files.** Use `load_senses()` from `src/diachronic/io.py`, which now handles both formats correctly.
2. **Do not copy the 0-indexed convention from `compare_embedding_methods.py`** (`llm_example_0..4`). The authoritative convention is `embedding_strategies.py` (1-indexed). Mixing these is a silent bug.
3. **Include 1-sense words in the mapping JSON** (map their single sense → `"{lemma}_c0"`). `apply_coarse_mapping.py` will encounter them and needs a valid entry.
4. **Do not apply coarse mapping to the atemporal eval set.** The 54.8% accuracy was measured at fine granularity. A coarse accuracy comparison is a separate, later question.
5. **Flag data-sparse words explicitly.** Words like `groß` (~47 total contexts) will remain noisy after clustering. Do not let them dominate the "top changed" ranking and mislead case study selection.