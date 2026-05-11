# Data Catalog

Reference for all data files in the project: what each file is, which script produces it, and what it is used for.

**Last updated:** 2026-03-23

---

## Overview: Data Flow

```
Raw Sources                    Processed                         Analysis Outputs
───────────────                ─────────────────────────         ────────────────
GermaNet XML        ──►  germanet_base/ (parsed tables)
                    ──►  germanet/senses_base.csv
                    ──►  germanet/senses_pilot.jsonl      ──►  diachronic/predictions/pilot_predictions_v3.csv
                                                          ──►  diachronic/predictions/pilot_predictions_v3_coarse.csv
                                                          ──►  diachronic/aggregated/pilot_coarse_v3_*.csv
                                                          ──►  diachronic/metrics/pilot_coarse_v3_*.csv

Target word lists   ──►  dwds/pilot_contexts_v3.jsonl  (current)
                    ──►  dwds/pilot_contexts.jsonl      (v2, superseded)

Eval word lists     ──►  evaluation/atemporal/input/      ──►  evaluation/atemporal/results/
  (10 words)              (contexts + gold labels)              (accuracy, weight tuning)
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

## `data/processed/germanet/` — Sense Representations & Cluster Mapping

### Sense files

| File | Produced by | Contents |
|---|---|---|
| `senses_base.csv` | `scripts/data/setup/rebuild_complete_wsd_senses.py` | Baseline sense file: GermaNet template text for all target words. Input to LLM enrichment. |
| `senses_pilot.jsonl` | `scripts/data/enrich_senses_with_llm.py` + `scripts/data/regen_core_glosses.py` | LLM-enriched senses for the 90 pilot words. **Used by the diachronic pipeline.** |

**senses_pilot.jsonl field reference:**
```
lemma, pos, synset_id, class,
baseline          — GermaNet template text
core_gloss        — 5–10 word denotative gloss (used for clustering embeddings)
llm_definition    — verbose contrastive definition (used for WSD weighted representation)
llm_examples      — array of 5 usage examples (used for WSD weighted representation)
```

### Cluster mapping

| File | Produced by | Contents |
|---|---|---|
| `sense_clusters_reviewed.json` | `scripts/clustering/compute_sense_clusters.py` + manual review | **Canonical coarse cluster mapping.** 96 lemmas (90 pilot + 6 eval), 317 clusters. All cluster labels filled. Antwort manually merged to 1 cluster. |

**Structure:**
```json
{
  "schlagen": {
    "mapping": {"s56941": "schlagen_c0", "s56942": "schlagen_c0", ...},
    "clusters": {
      "schlagen_c0": {"label": "physically strike", "members": ["s56941", "s56942"]},
      ...
    }
  }
}
```

---

## `data/processed/sense_clustering/` — Clustering Artifacts

**Produced by:** `scripts/clustering/compute_sense_clusters.py`

| File | Contents |
|---|---|
| `t010/cluster_summary.csv` | Per-lemma clustering stats at t=0.10: `lemma, threshold, n_fine_senses, n_organic_clusters, n_coarse_clusters, constrained, n_singleton_clusters, reduction_ratio, mean_intra_cluster_similarity, min_intra_cluster_similarity, max_cluster_size` |

---

## `data/processed/dwds/` — DWDS Corpus Sentences

**Produced by:** `scripts/data/fetch_dwds_contexts.py`

| File | Contents | Status |
|---|---|---|
| `pilot_contexts_v3.jsonl` | **90,130 sentences** for 90 pilot words (1900–1999, DWDS kern, `--limit 5000 --min-tokens 6`) | **Current — use this** |
| `pilot_contexts.jsonl` | 30,337 sentences for 90 pilot words (1900–1999, DWDS kern, `--limit 1000 --min-tokens 8`) | v2 — superseded |

**v3 fetch settings:** `--limit 5000 --min-tokens 6 --max-tokens 60 --sort random --corpus kern`
- `--limit 5000`: API max before silent fallback to 100. Raises density from ~337 → ~1,000 contexts/word.
- `--min-tokens 6` (was 8): Reduces bias against short Zeitung headlines, which are disproportionately from early decades.

**Record format:**
```json
{"lemma": "Geist", "sentence": "...", "year": 1956, "decade": 1950, "genre": "Zeitung", "bibl": "...", "corpus": "kern", "source": "dwds_corpus_csv"}
```

**DWDSClient capabilities (as of 2026-03-23):**
- Session-cookie auth: pass `DWDS_SESSION_COOKIE` in `.env` or `--session-cookie` arg → enables ZEIT, FAZ, Spiegel corpora
- Pagination: `--paginate` flag fetches all pages (use with deterministic sort, not `--sort random`)
- ZEIT corpus: requires institutional access approval from DWDS team (applied 2026-03-23)

---

## `data/evaluation/atemporal/` — Atemporal WSD Evaluation

Used for method development only. Ground truth files are stable — re-running scripts only updates `results/`.

### `input/` — Ground truth (do not overwrite)

| File | Produced by | Contents |
|---|---|---|
| `words.csv` | `scripts/annotation/select_eval_words.py` | 10 eval words, stratified by polysemy and POS |
| `contexts.csv` | `scripts/data/extract_eval_contexts.py` | 93 DWDS contexts for the 10 words |
| `gold_annotations.csv` | Manual annotation | Human sense labels for each context |
| `sense_text_representations.jsonl` | `scripts/data/enrich_senses_with_llm.py` | LLM-enriched senses for the 10 eval words |

### `results/` — Evaluation outputs

| File | Produced by | Contents |
|---|---|---|
| `embedding_comparison.csv` | `scripts/atemporal/compare_embedding_methods.py` | Accuracy per representation method and embedding model |
| `weight_tuning.csv` | `scripts/atemporal/tune_representation_weights.py` | Grid search over α (core_gloss run). Best: α=0.3 → 51.6% (core_gloss+examples). Original llm_definition+examples: α=0.5 → 43.0% fine, 84.9% coarse |
| `predictions_random.csv` | `scripts/atemporal/generate_predictions.py --variant random` | Per-instance analytical random-chance baseline (no embedder). Expected fine: 24.5%, coarse: 50.5% |
| `predictions_text_only.csv` | `scripts/atemporal/generate_predictions.py --variant text-only` | Per-instance predictions for text-only ablation (E5 + GermaNet hypernym text, no LLM enrichment). Fine: 36.6%, coarse: 75.3% |
| `predictions.csv` | `scripts/atemporal/generate_predictions.py` | Raw predictions under best method (weighted). Fine: 43.0%, coarse: 84.9% |

---

## `data/processed/diachronic/` — Diachronic Pilot Study Results

**Current pipeline version: v3** (90,130 contexts, `--limit 5000 --min-tokens 6`). v2 files retained for reference.

The prediction script (`predict_dwds_senses.py`) was refactored for v3 to process one lemma at a time — peak RAM ~7 MB/step (vs. ~370 MB for full-batch v2). Supports `--resume` to continue after interruption.

### `predictions/`

| File | Produced by | Contents | Status |
|---|---|---|---|
| `pilot_predictions_v3.csv` | `scripts/diachronic/predict_dwds_senses.py` | **~72,000 fine-grained predictions** for 90 words × all decades. Columns: `lemma, sentence, year, decade, predicted_sense_id, confidence` | In progress — use `--resume` |
| `pilot_predictions_v3_coarse.csv` | `scripts/diachronic/apply_coarse_mapping.py` | Coarse predictions: fine `predicted_sense_id` replaced with coarse cluster ID | Pending v3 predictions |
| `pilot_predictions.csv` | `scripts/diachronic/predict_dwds_senses.py` | 24,325 fine predictions (v2, 30,337 contexts) | v2 — superseded |
| `pilot_predictions_coarse.csv` | `scripts/diachronic/apply_coarse_mapping.py` | Coarse v2 predictions | v2 — superseded |

### `aggregated/`

| File | Produced by | Contents | Status |
|---|---|---|---|
| `pilot_coarse_v3_sense_counts.csv` | `scripts/diachronic/aggregate_temporal.py` | Coarse sense counts per (lemma × decade × cluster) for v3 | Pending v3 predictions |
| `pilot_coarse_v3_stats.csv` | `scripts/diachronic/aggregate_temporal.py` | Coarse summary per (lemma × decade) for v3 | Pending v3 predictions |
| `pilot_counts.csv` | `scripts/diachronic/aggregate_temporal.py` | Fine v2 sense counts | v2 — superseded |
| `pilot_stats.csv` | `scripts/diachronic/aggregate_temporal.py` | Fine v2 summary | v2 — superseded |
| `pilot_coarse_sense_counts.csv` | `scripts/diachronic/aggregate_temporal.py` | Coarse v2 sense counts. 2,450 rows (vs 3,852 fine) | v2 — superseded |
| `pilot_coarse_stats.csv` | `scripts/diachronic/aggregate_temporal.py` | Coarse v2 summary | v2 — superseded |

### `metrics/`

All metrics use `--min-contexts 10` filter.

| File | Produced by | Contents | Status |
|---|---|---|---|
| `pilot_coarse_v3_entropy.csv` | `scripts/diachronic/compute_change_metrics.py` | Coarse Shannon entropy per (lemma × decade) for v3 | Pending v3 pipeline |
| `pilot_coarse_v3_divergence.csv` | `scripts/diachronic/compute_change_metrics.py` | Coarse JSD for v3 | Pending v3 pipeline |
| `pilot_coarse_v3_emergence.csv` | `scripts/diachronic/compute_change_metrics.py` | Coarse emergence events for v3 | Pending v3 pipeline |
| `pilot_coarse_v3_decline.csv` | `scripts/diachronic/compute_change_metrics.py` | Coarse decline events for v3 | Pending v3 pipeline |
| `pilot_coarse_divergence.csv` | `scripts/diachronic/compute_change_metrics.py` | Coarse JSD v2. Max JSD: 0.532. Mean JSD: 0.165 | v2 — superseded |
| `pilot_coarse_emergence.csv` | `scripts/diachronic/compute_change_metrics.py` | Coarse emergence events v2. 194 events | v2 — superseded |
| `pilot_coarse_decline.csv` | `scripts/diachronic/compute_change_metrics.py` | Coarse decline events v2. 197 events | v2 — superseded |
| `pilot_entropy.csv` | `scripts/diachronic/compute_change_metrics.py` | Fine entropy v2 | v2 — superseded |
| `pilot_divergence.csv` | `scripts/diachronic/compute_change_metrics.py` | Fine JSD v2. Max JSD: 0.835 | v2 — superseded |
| `pilot_emergence.csv` | `scripts/diachronic/compute_change_metrics.py` | Fine emergence events v2. 518 events | v2 — superseded |
| `pilot_decline.csv` | `scripts/diachronic/compute_change_metrics.py` | Fine decline events v2. 521 events | v2 — superseded |