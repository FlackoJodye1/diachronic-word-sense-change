# Project Roadmap & Status

A quick-reference document tracking what's done, what's in progress, and what's next.

**Last updated:** 2026-10-27. Paper-level status and claims live in `documentation/paper_plan.md`; robustness numbers live in `analysis/robustness/RESULTS.md`.

---

## Project Overview

**Goal:** Build a German WSD benchmark using GermaNet + DWDS, then use it to analyze how word senses change over time (1900–1999).

**Three pillars:**
1. **Data preparation** — sense inventories, corpus contexts, LLM-enriched representations
2. **Atemporal WSD** — find the best disambiguation method (model + text representation)
3. **Diachronic analysis** — apply best method to historical data, track sense change over decades

---

## Phase 1: Data Preparation — COMPLETE

| Step | Status | Key Output |
|------|--------|------------|
| Process GermaNet v19.0 (synsets, relations, enriched texts) | Done | `data/processed/germanet/senses_base.csv` (2,022 senses, 752 words) |
| Select 750 target words (Charbonnier & Wartena 2019) | Done | `data/raw/target_words/target_words_clean.xlsx` |
| Fetch DWDS corpus contexts | Done | `data/processed/dwds/` (4,809 sentences for 25-word pilot) |
| Generate LLM-enriched sense representations (def, 1ex, 5ex) | Done | `data/processed/germanet/senses_full.jsonl` |
| Convert sense data to JSONL format | Done | `scripts/data/convert_to_jsonl.py` |

---

## Phase 2: Atemporal WSD Method Optimization — COMPLETE

**Purpose:** Find the best embedding model + text representation for German WSD.

### 2a. Quick Eval Set (method comparison) — COMPLETE

| Step | Status | Details |
|------|--------|---------|
| Select 10 eval words (stratified by polysemy + POS) | Done | Doktor, Antwort, wagen, Glas, Schatz, springen, bitter, Bahn, sitzen, hören |
| Fetch & annotate 93 gold instances | Done | `data/evaluation/atemporal/input/gold_annotations.csv` |
| Compare 5 models x 8 text variants | Done | `data/evaluation/atemporal/results/embedding_comparison.csv` |
| Weight tuning (alpha for def+examples) | Done | alpha=0.5 is optimal (`data/evaluation/atemporal/results/weight_tuning.csv`) |
| Error analysis | Done | 64% of errors are near-misses (correct sense ranked 2nd) |

**Best result:** `intfloat/multilingual-e5-large` + weighted(0.5 x definition + 0.5 x avg(5 examples)) = **43.0% fine / 84.9% coarse accuracy**, 95.7% top-3, 100% top-5. Reported against two baselines (2026-04-20): random 24.5% / 50.5%, and text-only ablation (E5 + GermaNet hypernyms) 36.6% / 75.3%. *Note: weighted was originally reported as 54.8% fine / 81.7% coarse — recalibrated 2026-04-07 after `sense_text_representations.jsonl` was re-generated with core_gloss; see research log.*

**Key scripts:** `scripts/atemporal/`

### 2c. Re-enrichment with Short Core Glosses — COMPLETE

**Purpose:** Add short `core_gloss` field to sense JSONL to reduce boilerplate-driven distance compression for clustering. WSD `weighted` representation unchanged (llm_definition still used).

**Outcome:** core_gloss tuning gave best 51.6% at α=0.3; llm_definition+examples at α=0.5 gives 43.0% (recalibrated 2026-04-07; was originally 54.8% before sense file re-generation). `weighted` uses `llm_definition`. core_gloss used separately for clustering only.

| Step | Status | Details |
|------|--------|---------|
| Update enrichment prompt in `enrich_senses_with_llm.py` | Done | `config/prompts/core_gloss.txt` — max 10 words, no boilerplate |
| Re-run enrichment on 596 pilot senses | Done | `senses_pilot.jsonl` now has `core_gloss` field alongside `llm_definition` |
| Re-enrich 10 eval words' sense file | Done | `data/evaluation/atemporal/input/sense_text_representations.jsonl` updated with `core_gloss` |
| Re-tune α on 93-instance eval set | Done | core_gloss: 51.6% best (α=0.3) — worse than llm_definition. `weighted` reverted to llm_definition at α=0.5. |
| Re-run `predict_dwds_senses.py` (v3 predictions) | **Skipped** | core_gloss worse for WSD — no reason to re-run; v2 predictions remain current best |
| Compare 3 representations in clustering notebook | Done | A (core_gloss only) wins: distance ratio ~5× vs ~1.7× Original. B unexpectedly worst — examples compress distances. |
| Recalibrate thresholds with A, choose final threshold | Done | t=0.10 chosen: first organic split for schlagen, Geist 4-way stable, spielen coherent under max_clusters=6 |
| Improve core_gloss prompt + full re-generation | Done | Added sibling context, POS norms, no-synonym rule. Re-generated all 596 glosses via `regen_core_glosses.py --all`. All splits now organic; Geist 3-way, spielen 3-way at t=0.10. t=0.10 confirmed. |
| Gloss quality review + prompt refinement | Done | Identified over-abstraction failure mode (2026-03-14). Iteratively refined prompt to lean version (12 rules → 6, "keine Liste" constraint). 3 manual patches applied (s52049, s58716, s58405). See research log 2026-03-14. |
| Extend test set to 10 words | Done | Added `gehen` (19 senses) and `fassen` (12 senses) to notebook. t=0.10 confirmed on both. |
| Remove max_clusters cap | Done | No upper bound on clusters. Threshold determines granularity; sparse cells filtered at analysis time. schlagen=8, spielen=8, gehen=14, fassen=6 organic clusters at t=0.10. |
| Run `compute_sense_clusters.py` on all 90 words at t=0.10 | Done | `sense_clusters_0.10.json` + `sense_clustering/t010/cluster_summary.csv`. Mean 6.6→3.7 clusters (39.5% reduction). 11 words constrained by min=2. |
| Manual review of all 90 cluster assignments | Done | All 305 clusters reviewed and labelled. Overrides applied where needed. |
| Produce `sense_clusters_reviewed.json` | Done | 90 lemmas, 305 clusters, all labels filled in. Canonical mapping for downstream pipeline. |

See research log entries 2026-03-11 and 2026-03-12 for full rationale.

### 2b. Second Annotation Set (benchmark expansion) — NOT STARTED

| Step | Status | Details |
|------|--------|---------|
| Select 10 more words | Done | Dose, Stoff, Keller, Witz, Gut, sprechen, sitzen, verfahren, neu, leer |
| Fetch 98 contexts | Done | `data/evaluation/atemporal/annotation/eval_contexts.csv` |
| Generate annotation tasks | Done | `data/evaluation/atemporal/annotation/annotation_tasks.json` |
| **Annotate instances** | **NOT DONE** | 0/98 annotated |

**Purpose:** Expand the gold standard beyond 10 words for a more credible benchmark. Not blocking anything else.

---

## Phase 3: Diachronic Pilot Study — IN PROGRESS

**Purpose:** Apply best WSD method to historical DWDS data and analyze how word senses change across decades (1900–1999).

### 3a. Data & Pipeline — v3 IN PROGRESS

| Step | Status | Details |
|------|--------|---------|
| Select 116 pilot words (stratified by polysemy) | Done | `data/raw/target_words/pilot_words_diachronic.txt` |
| Fetch historical DWDS contexts | Done (v2) | 30,337 sentences across 10 decades (`pilot_contexts.jsonl`). Re-fetched with `--max-tokens 60`, `--limit 1000`, no sentence cap. 0.3% truncation. Genre drift documented in research log. |
| Run baseline WSD predictions | Done | `data/processed/diachronic/dwds_predictions.csv` (5,155 predictions, 90 words with sufficient data) |
| Run weighted WSD predictions (v1) | Done (superseded) | `data/processed/diachronic/pilot_predictions_weighted.csv` — based on 6,474 sentences with --max-tokens 35 cap |
| Aggregate by decade (v1) | Done (superseded) | `data/processed/diachronic/pilot_weighted_decade_*` |
| Compute change metrics (v1) | Done (superseded) | `data/processed/diachronic/pilot_weighted_filtered_decade_*.csv` |
| Initial analysis & visualization | Done | `data/processed/diachronic/PILOT_STUDY_SUMMARY.md`, 14 plots in `data/processed/diachronic/plots/` |
| Corpus quality assessment | Done | Sentences usable (avg 21 words, <1% truncated). Fixed expressions (~20%) documented as known limitation. Genre/decade imbalance documented. |
| Re-fetch with relaxed cap (--max-tokens 60) | Done | 30,337 sentences (5x increase). 0.3% truncation. Output: `pilot_contexts.jsonl` |
| Run weighted WSD predictions (v2) | Done (superseded) | `data/processed/diachronic/predictions/pilot_predictions.csv` — 24,325 predictions, 90 words |
| Aggregate by decade (v2) | Done (superseded) | `data/processed/diachronic/aggregated/pilot_counts.csv` |
| Compute change metrics (v2) | Done (superseded) | `data/processed/diachronic/metrics/pilot_*.csv` — Max JSD: 0.835. |
| Re-bin to 25-year periods | Decided against | Only 22/894 word-decades sparse (2%). Keeping 10-year decades. |
| Re-fetch with limit=5000 for full corpus coverage | Done (2026-03-23) | 30,337 → 90,130 contexts (+197%). `pilot_contexts_v3.jsonl`. `--limit 5000 --min-tokens 6 --max-tokens 60`. |
| Run v3 predictions | Done (2026-03-24) | 90/90 lemmas, 72,591 rows (+198% vs v2). Output: `pilot_predictions_v3.csv`. |
| Apply coarse mapping (v3) | Done (2026-03-24) | `apply_coarse_mapping.py` → `pilot_predictions_v3_coarse.csv`. 0 rows dropped. |
| Re-aggregate + recompute metrics (v3) | Done (2026-03-24) | `pilot_v3_coarse_sense_counts.csv`, `pilot_v3_coarse_stats.csv`. Mean JSD=0.112, 806 transitions, 126 emergence / 118 decline events. |
| Add pagination to `DWDSClient` | Done (2026-03-23) | `corpus_csv(paginate=True)` loops `&p=1,2,...` until page < limit. Off by default — use with deterministic sort only, not `random`. |
| Add session-cookie auth to `DWDSClient` | Done (2026-03-23) | `DWDSClient(session_cookie=...)` + `--session-cookie` arg in fetch script. Cookie extracted from browser DevTools after DWDS login. |
| **Re-fetch including ZEIT corpus** | Pending | Waiting on auth implementation + DWDS login. ZEIT covers 1946–present. |

**Key finding (v1):** Weighted method has +4.9% higher confidence than baseline. 82 emergence events, 84 decline events detected. Verbs dominate the most-changed words.

**Key finding (v2):** 518 emergence events, 521 decline events. Max JSD 0.835 (no more JSD=1.0 artifacts). Only 22 sparse word-decades remaining. Results substantially more reliable than v1.

**Top 10 words by temporal change (v3, normalized JSD):** Film, Lappen, Saft, schneiden, springen, Stiefel, Nest, Kasse, Hütte, Meister. Verbs dominate raw JSD but normalization favours simpler words with cleaner shifts.

### 3b. Analysis & Validation — IN PROGRESS

| Step | Status | Details |
|------|--------|---------|
| Coarse atemporal sanity check | Done | 43.0% fine → 84.9% coarse on 93-instance eval set. Baselines: random 24.5% / 50.5%, text-only 36.6% / 75.3%. Antwort merged to 1 cluster (model cannot split). All 96 lemmas in `sense_clusters_reviewed.json`. |
| Apply coarse mapping → `pilot_predictions_coarse.csv` | Done | `scripts/diachronic/apply_coarse_mapping.py`. 0 rows dropped. `sense_class`/`sense_paraphrase` dropped (fine-grained, not meaningful at coarse level). |
| Re-aggregate with coarse sense IDs | Done | `pilot_coarse_sense_counts.csv`, `pilot_coarse_stats.csv`. 2,450 lemma-period-sense rows (vs 3,852 fine). |
| Recompute change metrics at coarse level | Done | Mean JSD=0.165, Max JSD=0.532, 194 emergence / 197 decline events (vs 518/521 fine). Top words: spielen, fassen, Kasse, passen. |
| Restructure analysis notebooks | Done | `diachronic_pilot_results.ipynb` stripped to coarse-only (50 cells, 7 sections). `sense_granularity_analysis.ipynb` created for fine vs. coarse methodological comparison. Section 7 (Case Study Candidates) added with Spearman ρ monotonicity scoring. |
| Refactor results notebook (2026-03-19) | Done | Exclusions now filtered at load time (all downstream cells clean). Composite score removed; rankings use avg_jsd (primary) and entropy_range (secondary). Word totals table enriched with avg_jsd and max_jsd. Per-lemma stacked area chart collapses to top-6 + Other; contexts-per-decade support table added above chart. |
| Extend rankings to normalized JSD (2026-03-20) | Done | Added `normalized_jsd = avg_jsd / n_coarse_senses` as third ranking metric. Three-panel bar chart shows all three rankings. All section 3 and 4 plots (small-multiples, JSD-over-time, emergence/decline) now use top-5 by normalized JSD consistently. |
| Re-run analysis notebook on v3 data | Done (2026-03-27) | All sections re-run on v3. Updated rankings, candidates, and trajectory plots in `diachronic_pilot_results.ipynb`. |
| Identify case study candidates (v3) | Done (2026-03-27) | 6 automated strong candidates (springen, Löffel, Welle, Hof, Stamm, Kugel). Manual reliability review (§7.1–7.2) narrows to **2 selected: `springen` and `Welle`**. Remaining 4 excluded: Stamm/Löffel (sparse later decades), Hof/Kugel (non-monotonic). |
| **Manual sentence-level validation of case studies** | **Next** | Validate springen and Welle predictions against raw DWDS sentences. Confirm trend is real and not a WSD artifact. |
| Validate against DWDS dictionary / lexicographic evidence | Not done | Cross-reference emergence/decline with historical records |
| Statistical significance testing | Not done | Chi-square on distribution shifts, confidence intervals |
| Investigate JSD=1.0 values | Done (v2) | Eliminated by re-fetching with --max-tokens 60. Only 22/894 word-decades remain sparse. Max JSD now 0.835. |
| WWII case study (schlagen, geben, handeln, werfen) | Not done | Pattern visible in Section 6 of results notebook — needs manual validation before reporting |
| Publication-ready figures | Not done | Trajectory plots in notebook are exploratory only |

### 3c. Key Concern

The JSD=1.0 artifact problem was resolved in v2 (re-fetch with --max-tokens 60, 30,337 sentences). The coarse accuracy (84.9%) is sufficient to detect large, sustained shifts but may not resolve subtle or short-lived changes. The 84.9% figure is measured on contemporary texts — accuracy on 1900s data is unknown and likely lower. *(2026-10-27: an exploratory split of the eval set finds no deficit on older text, with coarse accuracy at 88.7% before 1950 vs. 77.4% from 1950, n.s. and only 31 late instances. See `analysis/robustness/RESULTS.md` §6. Separately, decade-level JSD turned out to be ~90% sampling noise, so change detection moves to period bins (§5).)* **Case studies must show large, monotonic trajectory changes across multiple consecutive decades. Manual sentence-level validation is not optional.** See research log 2026-03-18.

---

## Phase 4: Scale Up & Publish — IN PROGRESS

| Step | Status |
|------|--------|
| Expand gold annotations (target: 100-200 words) | Not started |
| Add 2nd annotator, compute inter-annotator agreement (kappa > 0.7) | Not started |
| Generate `core_gloss` for expanded word set | Done (2026-04-02) | `generate_expanded_senses.py` run on 413 polysemous non-pilot lemmas → `senses_expanded.jsonl` (1229 senses). Workers reduced to 1 to avoid memory pressure on Mac. |
| Cluster expanded words + merge into reviewed mapping | Done (2026-04-02) | `compute_sense_clusters.py --merge-into sense_clusters_reviewed.json` → 407 new lemmas added (6 eval lemmas already present). 160/413 constrained. `sense_clusters_reviewed.json` now covers 503 lemmas total. |
| Annotate cluster labels for expanded words | Done (2026-05-16) | Three annotators (Paul, Amandine, Nadia), merged via `scripts/clustering/merge_reviews.py` → `sense_clusters_reviewed.json` (503 lemmas, fully reviewed). |
| Fetch DWDS contexts for expanded words | Done | `data/processed/dwds/expanded_contexts_v3.jsonl`, combined with pilot in `combined_contexts_v3.jsonl`. Same v3 settings. |
| Run predictions + aggregate for expanded words | Done (2026-05-18) | `combined_predictions_v3.csv` (318,611 rows, 502 lemmas) → `pilot_v3_coarse_*` metrics, `per_lemma_summary.csv`. |
| Survey correlations (G&R 2024, German + English) | Done (2026-05-14/15) | `documentation/survey_correlations_2026-05-15.md`, `…_english_2026-05-15.md`. |
| Robustness pass (ARR resubmission) | Done (2026-10-16, extended 2026-10-27) | JSD null baseline, 6 binnings, frequency partials, WSD by era. Recommendation: 3-period binning. `analysis/robustness/RESULTS.md`. |
| DWUG DE evaluation (reviewer request) | Inventory done (2026-10-27) | `analysis/dwug/INVENTORY.md`. Plan decisions pending before any computation. |
| Hamilton extension on period bins | Not started | Blocked on binning decision; see `paper_plan.md` §5.5. |
| Add ZEIT corpus (Die Zeit 1946–present) | **Pending institutional access** (applied 2026-03-23). Would significantly increase 1950s–1990s density with consistent Zeitung register. Decide on kern+ZEIT merge strategy once access confirmed. |
| Draft paper (methods, results, discussion) | Not started |
| Release benchmark dataset | Not started |

---

## Key Files & Directories

| What | Where |
|------|-------|
| Sense inventory | `data/processed/germanet/senses_base.csv` |
| LLM-enriched senses | `data/processed/germanet/senses_full.jsonl` |
| Atemporal gold annotations | `data/evaluation/atemporal/input/gold_annotations.csv` |
| Atemporal results | `data/evaluation/atemporal/results/` |
| Diachronic predictions (v3, current) | `data/processed/diachronic/predictions/pilot_predictions_v3.csv` |
| Coarse predictions (v3, current) | `data/processed/diachronic/predictions/pilot_predictions_v3_coarse.csv` |
| Coarse metrics (v3, current) | `data/processed/diachronic/metrics/pilot_v3_coarse_*.csv` |
| Diachronic predictions (v2, superseded) | `data/processed/diachronic/predictions/pilot_predictions.csv` |
| Coarse predictions (v2, superseded) | `data/processed/diachronic/predictions/pilot_predictions_coarse.csv` |
| Coarse metrics (v2, superseded) | `data/processed/diachronic/metrics/pilot_coarse_*.csv` |
| Diachronic plots | `data/processed/diachronic/plots/` |
| Sense granularity analysis | `notebooks/analysis/sense_granularity_analysis.ipynb` |
| **Robustness analyses (ARR resubmission)** | **`analysis/robustness/RESULTS.md`** — JSD null baseline, H2 partial correlation, survey reliability, temporal-binning sensitivity. Read this before quoting any JSD-based number. |
| Robustness scripts | `analysis/robustness/robustness_analyses.py` (§1–4), `analysis/robustness/binning_sensitivity.py` (§5–6) |
| DWUG DE evaluation | `analysis/dwug/INVENTORY.md`; data in `data/external/dwug_de/` (gitignored, CC BY-ND 4.0) |
| Atemporal scripts | `scripts/atemporal/` |
| Diachronic scripts | `scripts/diachronic/` |
| Data prep scripts | `scripts/data/` |
| Analysis scripts | `scripts/analysis/` |
| Full methodology docs | `documentation/documentation.md` |
| Pipeline guide | `documentation/pipeline.md` |
| Research log | `documentation/research_log.md` |

---

## Terminology

- **Atemporal WSD** = disambiguating word senses without considering time. Used for method optimization.
- **Diachronic analysis** = tracking how sense distributions change over decades (1900–1999).
- **Quick eval set** = 10 words, 93 gold instances used to compare embedding methods. Part of atemporal work.
- **Diachronic pilot** = 116 words, 6,474 historical contexts. The main temporal analysis.
- **Weighted method** = 0.5 x definition_embedding + 0.5 x avg(5_example_embeddings). Best approach found.
- **Baseline method** = GermaNet template text only (`text_for_embedding` column).
