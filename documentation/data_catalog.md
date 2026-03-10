# Data Catalog

Reference for all data files in the project: what each file is, which script produces it, and what it is used for.

**Last updated:** 2026-03-09

---

## Overview: Data Flow

```
Raw Sources                  Processed Intermediates              Analysis Outputs
──────────────               ────────────────────────             ────────────────
GermaNet XML        ──►  germanet_base/          ──►  germanet/senses_base.csv
                              (parsed tables)              (senses + LLM enrichment)

Target word lists   ──►  processed/dwds/                ──►  diachronic/pilot_predictions*.csv
  (pilot, full)           (DWDS sentences)                    (sense predictions per sentence)

                                                         ──►  diachronic/pilot_weighted_v2_*.csv
                                                              (aggregated metrics per decade)

Eval word lists     ──►  evaluation/atemporal/input/    ──►  evaluation/atemporal/results/
  (10 words)              (contexts + gold labels)            (accuracy, weight tuning)
```

---

## `data/raw/` — Source Data (Not Script-Generated)

| File / Directory | Contents | Notes |
|---|---|---|
| `GN_V190/GN_V190_XML/*.xml` | GermaNet v19.0 XML files (synsets, lexical units, relations) | Downloaded from GermaNet release; do not modify |
| `target_words/target_words_clean.xlsx` | 750 source words from Charbonnier & Wartena (2019) with concreteness ratings | Master word list |
| `target_words/word_list_750.txt` | Plain-text version of the 750 words | Used as `--word-list` arg by fetch/enrich scripts |
| `target_words/pilot_words_diachronic.txt` | 116 pilot words for diachronic study (manually curated subset) | Input to diachronic pipeline |

---

## `data/processed/germanet_base/` — Parsed GermaNet Tables

**Produced by:** `scripts/data/process_germanet.py`
**Purpose:** Normalized relational tables extracted from GermaNet XML. Foundation for all downstream sense processing.

| File | Contents |
|---|---|
| `sysnets/synsets.csv` | One row per synset (concept): ID, class, category |
| `sysnets/sense_meta.csv` | Synset-level metadata (paraphrases, wiktionary links) |
| `sysnets/sense_texts.csv` | Template-based text descriptions per synset |
| `lexical_layer/lexical_units.csv` | One row per lemma-sense pair: lemma, pos, synset_id |
| `lexical_layer/lemma_senses.csv` | Which senses belong to each lemma |
| `lexical_layer/candidate_map.pkl` | Lemma → candidate sense IDs mapping (pickled dict) |
| `semantic_layer/relations.csv` | Semantic relations between synsets (hypernym, meronym, etc.) |

---

## `data/processed/germanet/` — Enhanced Sense Representations

### Core sense files

| File | Produced by | Contents |
|---|---|---|
| `senses_base.csv` | `scripts/data/setup/rebuild_complete_wsd_senses.py` | Baseline sense file: GermaNet template text (`text_for_embedding`) for all target words. Input to LLM enrichment. |
| `senses_full.jsonl` | `scripts/data/enrich_senses_with_llm.py` | LLM-enriched senses for all 750 words. Contains `baseline`, `llm_definition`, and `llm_examples` (array of 5). |
| `senses_pilot.jsonl` | `scripts/data/enrich_senses_with_llm.py --word-list pilot_words_diachronic.txt` | LLM-enriched senses for the 90 pilot words with GermaNet data. **Used by the diachronic pipeline.** |

**Column reference (JSONL):**
```
lemma, pos, synset_id, class, baseline (template text),
llm_definition, llm_examples: [5 example sentences]
```

---

## `data/processed/target_words/`

| File | Produced by | Contents |
|---|---|---|
| `target_words_sense_counts.tsv` | `scripts/data/setup/calculate_germanet_senses.py` | Number of GermaNet senses per target word. Used to stratify word selection and identify highly polysemous words. |

---

## `data/processed/dwds/` — DWDS Corpus Sentences

**Produced by:** `scripts/data/fetch_dwds_contexts.py`
**Purpose:** Historical German sentences from the DWDS Kernkorpus (1900–1999), fetched via DWDS API.

| File | Contents |
|---|---|
| `pilot_contexts.jsonl` | **30,337 sentences** for 90 pilot words. Fetched with `--max-tokens 60`. This is the v2 context file used by all current analyses. |

**Record format:**
```json
{"lemma": "Geist", "sentence": "...", "year": 1956, "decade": 1950, "genre": "Zeitung", "bibl": "..."}
```

**Run:**
```bash
python scripts/data/fetch_dwds_contexts.py \
  --word-list data/raw/target_words/pilot_words_diachronic.txt \
  --output data/processed/dwds/pilot_contexts.jsonl \
  --max-tokens 60
```

---

## `data/evaluation/atemporal/` — Atemporal WSD Evaluation

Used for method development only (Phase 2). These files are stable — re-running scripts only updates `results/`.

### `input/` — Ground truth (do not overwrite)

| File | Produced by | Contents |
|---|---|---|
| `words.csv` | `scripts/annotation/select_eval_words.py` | 10 eval words, stratified by polysemy and POS |
| `contexts.csv` | `scripts/data/extract_eval_contexts.py` | 93 DWDS contexts for the 10 words |
| `gold_annotations.csv` | Manual annotation (via `scripts/annotation/app.py`) | Human sense labels for each context |
| `sense_text_representations.jsonl` | `scripts/data/enrich_senses_with_llm.py` | LLM-enriched senses for the 10 eval words |

### `annotation/` — Annotation infrastructure

| File | Produced by | Contents |
|---|---|---|
| `annotation_tasks.json` | `scripts/annotation/export_annotation_tasks_json.py` | Annotation tasks in JSON format |
| `annotation_tasks.pkl` | `scripts/annotation/generate_annotation_tasks.py` | Annotation tasks as Python objects |
| `annotation_summary.json` | `scripts/annotation/app.py` | Progress summary |
| `eval_contexts.csv` | `scripts/annotation/app.py` | Contexts formatted for the annotation app |

### `results/` — Evaluation outputs

| File | Produced by | Contents |
|---|---|---|
| `embedding_comparison.csv` | `scripts/atemporal/compare_embedding_methods.py` | Accuracy of each representation method (baseline, llm_definition, llm_examples, weighted) across embedding models |
| `weight_tuning.csv` | `scripts/atemporal/tune_representation_weights.py` | Grid search over α for weighted = α×definition + (1-α)×examples. Best: α=0.5 → 54.8% |
| `predictions.csv` | `scripts/atemporal/generate_predictions.py` | Raw predictions for all eval contexts under the best method |

---

## `data/processed/diachronic/` — Diachronic Pilot Study Results

All current files are from the **v2 pipeline** (contexts re-fetched with `--max-tokens 60`, JSD=1.0 artifacts eliminated).

### Step 1 — Sense predictions

| File | Produced by | Contents |
|---|---|---|
| `pilot_predictions_weighted_v2.csv` | `scripts/diachronic/predict_dwds_senses.py` | **24,325 predictions** for 90 words × all decades. One row per context sentence. Columns: `lemma, sentence, year, decade, predicted_sense_id, confidence, sense_class` |

**Run:**
```bash
PYTHONPATH=src python scripts/diachronic/predict_dwds_senses.py \
  --dwds-file data/processed/dwds/pilot_contexts.jsonl \
  --senses-file data/processed/germanet/senses_pilot.jsonl \
  --representation weighted \
  --output data/processed/diachronic/predictions/pilot_predictions.csv
```

### Step 2 — Temporal aggregation

| File | Produced by | Contents |
|---|---|---|
| `pilot_weighted_v2_decade_sense_counts.csv` | `scripts/diachronic/aggregate_temporal.py` | Sense usage counts per (lemma × decade × sense). Columns: `lemma, decade, predicted_sense_id, count, total, proportion` |
| `pilot_weighted_v2_decade_stats.csv` | `scripts/diachronic/aggregate_temporal.py` | Summary per (lemma × decade): n_contexts, n_senses used, dominant sense, dominant proportion |

**Run:**
```bash
PYTHONPATH=src python scripts/diachronic/aggregate_temporal.py \
  --input data/processed/diachronic/predictions/pilot_predictions.csv \
  --output-prefix pilot_weighted_v2
```

### Step 3 — Change metrics (min 5 contexts/decade filter applied)

| File | Produced by | Contents |
|---|---|---|
| `pilot_weighted_v2_filtered_decade_entropy.csv` | `scripts/diachronic/compute_change_metrics.py` | Shannon entropy per (lemma × decade). High entropy = more ambiguous / evenly distributed senses |
| `pilot_weighted_v2_filtered_decade_divergence.csv` | `scripts/diachronic/compute_change_metrics.py` | Jensen-Shannon divergence between consecutive decades per word. Measures how much the sense distribution shifted |
| `pilot_weighted_v2_filtered_decade_emergence.csv` | `scripts/diachronic/compute_change_metrics.py` | Sense emergence events (proportion crosses 10% threshold upward). 518 events detected |
| `pilot_weighted_v2_filtered_decade_decline.csv` | `scripts/diachronic/compute_change_metrics.py` | Sense decline events (proportion crosses 10% threshold downward). 521 events detected |

**Run:**
```bash
PYTHONPATH=src python scripts/diachronic/compute_change_metrics.py \
  --counts data/processed/diachronic/aggregated/pilot_counts.csv \
  --output-prefix pilot_weighted_v2_filtered \
  --min-contexts 5
```

### Step 4 — Visualizations

| File/Directory | Produced by | Contents |
|---|---|---|
| `plots/trajectory_*.png` | `scripts/diachronic/visualize_trajectories.py` | Stacked area charts of sense proportions over time, per word |
| `plots/heatmap_*.png` | `scripts/diachronic/visualize_trajectories.py` | Sense × decade heatmaps, per word |
| `plots/entropy_*.png` | `scripts/diachronic/visualize_trajectories.py` | Entropy over time, per word |
| `plots/entropy_heatmap.png` | `visualize_pilot_results.py` | All-words entropy heatmap |
| `plots/top_words_analysis.png` | `visualize_pilot_results.py` | Ranking by divergence and entropy range |
| `plots/wwii_patterns.png` | `visualize_pilot_results.py` | JSD trajectories for top words around WWII era |
| `plots/confidence_by_decade.png` | `visualize_pilot_results.py` | Prediction confidence distribution and by-decade breakdown |

### Reports

| File | Produced by | Contents |
|---|---|---|
| `pilot_study_analysis_report.md` | `analyze_diachronic_pilot.py` | Auto-generated markdown report: top-10 words by composite score, historical patterns, per-decade confidence stats |
| `PILOT_STUDY_SUMMARY.md` | Manually written | High-level summary of the pilot study design and v2 results |

---

## Root-Level Analysis Scripts

These scripts are entry points for exploratory analysis — not part of the core pipeline scripts under `scripts/`.

| Script | Reads | Writes | Purpose |
|---|---|---|---|
| `analyze_diachronic_pilot.py` | `pilot_predictions_weighted_v2.csv`, `pilot_weighted_v2_filtered_decade_*.csv` | `pilot_study_analysis_report.md` | Generates ranked word report and summary statistics |
| `visualize_pilot_results.py` | `pilot_predictions_weighted_v2.csv`, `pilot_weighted_v2_filtered_decade_divergence/entropy.csv` | `plots/confidence_by_decade.png`, `plots/top_words_analysis.png`, `plots/wwii_patterns.png`, `plots/entropy_heatmap.png` | Generates overview visualizations for pilot results |