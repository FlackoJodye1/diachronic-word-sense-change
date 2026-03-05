# Project Roadmap & Status

A quick-reference document tracking what's done, what's in progress, and what's next.

**Last updated:** 2026-02-19 (v2 pipeline complete)

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
| Process GermaNet v19.0 (synsets, relations, enriched texts) | Done | `data/processed/germanet/target_word_senses.csv` (2,022 senses, 752 words) |
| Select 750 target words (Charbonnier & Wartena 2019) | Done | `data/raw/target_words/target_words_clean.xlsx` |
| Fetch DWDS corpus contexts | Done | `data/dwds_contexts/` (4,809 sentences for 25-word pilot) |
| Generate LLM-enriched sense representations (def, 1ex, 5ex) | Done | `data/processed/germanet/target_word_senses_enhanced.jsonl` |
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

**Best result:** `intfloat/multilingual-e5-large` + weighted(0.5 x definition + 0.5 x avg(5 examples)) = **54.8% accuracy**, 96.8% top-3, 100% top-5

**Key scripts:** `scripts/atemporal/`

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

### 3a. Data & Pipeline — COMPLETE

| Step | Status | Details |
|------|--------|---------|
| Select 116 pilot words (stratified by polysemy) | Done | `data/raw/target_words/pilot_words_diachronic.txt` |
| Fetch historical DWDS contexts | Done (v2) | 30,337 sentences across 10 decades (`pilot_diachronic_contexts_v2.jsonl`). Re-fetched with `--max-tokens 60`, `--limit 1000`, no sentence cap. 0.3% truncation. Genre drift documented in research log. |
| Run baseline WSD predictions | Done | `data/diachronic/dwds_predictions.csv` (5,155 predictions, 90 words with sufficient data) |
| Run weighted WSD predictions (v1) | Done (superseded) | `data/diachronic/pilot_predictions_weighted.csv` — based on 6,474 sentences with --max-tokens 35 cap |
| Aggregate by decade (v1) | Done (superseded) | `data/diachronic/pilot_weighted_decade_*` |
| Compute change metrics (v1) | Done (superseded) | `data/diachronic/pilot_weighted_filtered_decade_*.csv` |
| Initial analysis & visualization | Done | `data/diachronic/PILOT_STUDY_SUMMARY.md`, 14 plots in `data/diachronic/plots/` |
| Corpus quality assessment | Done | Sentences usable (avg 21 words, <1% truncated). Fixed expressions (~20%) documented as known limitation. Genre/decade imbalance documented. |
| Re-fetch with relaxed cap (--max-tokens 60) | Done | 30,337 sentences (5x increase). 0.3% truncation. Output: `pilot_diachronic_contexts_v2.jsonl` |
| Run weighted WSD predictions (v2) | Done | `data/diachronic/pilot_predictions_weighted_v2.csv` — 24,325 predictions, 90 words |
| Aggregate by decade (v2) | Done | `data/diachronic/pilot_weighted_v2_decade_sense_counts.csv`, `pilot_weighted_v2_decade_stats.csv` |
| Compute change metrics (v2, filtered) | Done | `data/diachronic/pilot_weighted_v2_filtered_decade_*.csv` — only 22/894 word-decades filtered (vs 371 in v1). Max JSD: 0.835. |
| Re-bin to 25-year periods | **TODO** | With v2 data, decade-level density is now much better (only 22 sparse word-decades). Assess whether binning is still needed after reviewing v2 results. |

**Key finding (v1):** Weighted method has +4.9% higher confidence than baseline. 82 emergence events, 84 decline events detected. Verbs dominate the most-changed words.

**Key finding (v2):** 518 emergence events, 521 decline events. Max JSD 0.835 (no more JSD=1.0 artifacts). Only 22 sparse word-decades remaining. Results substantially more reliable than v1.

**Top 10 words by temporal change (v1):** spielen, werfen, zeigen, laufen, schlagen, tragen, ziehen, Stamm, Satz, nehmen — **to be re-ranked using v2 metrics**

### 3b. Analysis & Validation — NOT STARTED

| Step | Status | Details |
|------|--------|---------|
| Manual validation of top 10 words | Not done | Check if detected changes are real or WSD artifacts |
| Validate against DWDS dictionary / lexicographic evidence | Not done | Cross-reference emergence/decline with historical records |
| Statistical significance testing | Not done | Chi-square on distribution shifts, confidence intervals |
| Investigate JSD=1.0 values | Not done | Many could be sparse-data artifacts (e.g., 1940s with few contexts) |
| WWII case study (schlagen, geben, handeln, werfen) | Not done | Interesting pattern but needs validation |
| Publication-ready figures | Not done | Only 14 plots for 3 words so far |

### 3c. Key Concern

Many of the dramatic-looking results (JSD=1.0, complete sense shifts) may be **artifacts of sparse data** in certain decades rather than genuine semantic change. The 1940s especially have fewer corpus contexts. Manual validation is the critical next step.

---

## Phase 4: Scale Up & Publish — NOT STARTED

| Step | Status |
|------|--------|
| Expand gold annotations (target: 100-200 words) | Not started |
| Add 2nd annotator, compute inter-annotator agreement (kappa > 0.7) | Not started |
| Extend diachronic analysis to more words (target: 150-200) | Not started |
| Draft paper (methods, results, discussion) | Not started |
| Release benchmark dataset | Not started |

---

## Key Files & Directories

| What | Where |
|------|-------|
| Sense inventory | `data/processed/germanet/target_word_senses.csv` |
| LLM-enriched senses | `data/processed/germanet/target_word_senses_enhanced.jsonl` |
| Atemporal gold annotations | `data/evaluation/atemporal/input/gold_annotations.csv` |
| Atemporal results | `data/evaluation/atemporal/results/` |
| Diachronic predictions | `data/diachronic/dwds_predictions.csv`, `pilot_predictions_weighted.csv` |
| Diachronic metrics | `data/diachronic/metrics_decade_*.csv`, `pilot_weighted_decade_*.csv` |
| Diachronic summary | `data/diachronic/PILOT_STUDY_SUMMARY.md` |
| Diachronic plots | `data/diachronic/plots/` |
| Atemporal scripts | `scripts/atemporal/` |
| Diachronic scripts | `scripts/diachronic/` |
| Data prep scripts | `scripts/data/` |
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
