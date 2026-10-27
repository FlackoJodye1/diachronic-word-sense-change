# German WSD Pipeline Guide

**Last updated:** 2026-03-23

A practical guide to the pipeline from raw data through diachronic analysis.

---

## Overview

```
Data Preparation → WSD Optimization → Diachronic Analysis
     (once)           (iterative)          (publication)
```

**Research goal:** Detect how the sense distribution of polysemous German words shifts across decades (1900–1999) using GermaNet sense inventories, LLM-enriched sense representations, and DWDS corpus data.

---

## Data Sources

| Source | What | Location |
|---|---|---|
| GermaNet v19.0 | German semantic network — synsets, sense inventories, relations | `data/external/GN_V190/` |
| DWDS Kernkorpus | Historical German sentences 1900–1999, via public API | `data/processed/dwds/` |
| Target words | 116 pilot words from Charbonnier & Wartena (2019) | `data/raw/target_words/` |

---

## Phase 1: Data Preparation

Done once. Outputs are stable.

### Step 1.1 — Process GermaNet

```bash
python scripts/data/process_germanet.py
```

Parses GermaNet XML into normalized tables. Output: `data/processed/germanet_base/`.
Then builds `senses_base.csv` (template-based text, no LLM yet).

### Step 1.2 — Enrich senses with LLM

```bash
python scripts/data/enrich_senses_with_llm.py \
  --word-list data/raw/target_words/pilot_words_diachronic.txt \
  --provider openai
```

For each sense, generates via LLM:
- `core_gloss` — 5–10 word denotative gloss (used for clustering)
- `llm_definition` — verbose contrastive definition (used for WSD)
- `llm_examples` — 5 diverse usage examples (used for WSD)

Output: `data/processed/germanet/senses_pilot.jsonl`

Supports checkpointing (resumes if interrupted). `core_gloss` can be regenerated independently via `scripts/data/regen_core_glosses.py`.

### Step 1.3 — Fetch DWDS contexts

```bash
python scripts/data/fetch_dwds_contexts.py \
  --words-file data/raw/target_words/pilot_words_diachronic.txt \
  --corpus kern \
  --limit 5000 \
  --min-tokens 6 \
  --max-tokens 60 \
  --sort random \
  --output data/processed/dwds/pilot_contexts_v3.jsonl
```

Queries the DWDS API per word, filters by quality (length, duplicates), extracts decade metadata.
Output: `pilot_contexts_v3.jsonl` — **90,130 sentences** for 90 words.

**For licensed corpora (ZEIT, FAZ, Spiegel):** set `DWDS_SESSION_COOKIE` in `.env` or pass `--session-cookie`. Use `--paginate` with a deterministic sort (not `--sort random`) to fetch all pages.

**API limits:** max `--limit` is 5,000 (API silently reverts to 100 above ~7,500). `kern` corpus ceiling is ~3,500 unique documents per word. *(Correction 2026-10-27: this was measured for Krone only. By kern frequency, 245 of 502 lemmas have more than 5,000 hits, so a single page with limit 5,000 is a sample for them. See `analysis/fetch_check/RESULTS.md` §4.)*

**Filter note:** `target_matches_token` keeps only sentences where the exact base form occurs exactly once, so inflected forms are dropped. See `analysis/fetch_check/RESULTS.md`.

**Access status (2026-10-27):** DWDS has agreed in writing to export and reuse of kern results (`documentation/DATA_SOURCES.md`). But `/r/` blocks automated access (robots.txt `Disallow: /r`, HTTP 403 for an honest user agent). `DWDSClient` currently gets through only because it sends a browser user agent. **Don't re-run this step until the technical access is cleared with DWDS.** Proposed client changes: `analysis/fetch_check/RESULTS.md`.

---

## Phase 2: WSD Optimization (Atemporal)

Optimize embedding method on the small gold eval set (93 instances, 10 words) before large-scale analysis. Results carry forward to Phase 3.

**Best configuration (final):**
- Model: `intfloat/multilingual-e5-large`
- Representation: `weighted` — 0.5 × `llm_definition` + 0.5 × avg(`llm_examples`)
- Fine accuracy: **43.0%** | Coarse accuracy: **84.9%**
- Baselines (fine/coarse): random 24.5% / 50.5%; text-only ablation (E5 + GermaNet hypernyms) 36.6% / 75.3%

```bash
# Compare embedding methods
python scripts/atemporal/compare_embedding_methods.py

# Tune α weight
python scripts/atemporal/tune_representation_weights.py

# Generate detailed predictions (error analysis)
python scripts/atemporal/generate_predictions.py
```

See `scripts/atemporal/README.md` for full detail.

---

## Phase 3: Diachronic Analysis

### Step 3.1 — Predict senses (fine-grained)

```bash
python scripts/diachronic/predict_dwds_senses.py \
  --dwds-file data/processed/dwds/pilot_contexts_v3.jsonl \
  --output data/processed/diachronic/predictions/pilot_predictions_v3.csv
```

Processes one lemma at a time (memory-efficient). Supports `--resume` to continue after a crash.
Output: `pilot_predictions_v3.csv` — ~72,000 rows expected (90 words, 26 skipped — no GermaNet senses).

### Step 3.2 — Apply coarse cluster mapping

```bash
PYTHONPATH=src python scripts/diachronic/apply_coarse_mapping.py \
  --predictions data/processed/diachronic/predictions/pilot_predictions.csv \
  --mapping data/processed/germanet/sense_clusters_reviewed.json \
  --output data/processed/diachronic/predictions/pilot_predictions_coarse.csv
```

Maps fine `predicted_sense_id` → coarse cluster ID using `sense_clusters_reviewed.json`.
0 rows dropped. Output: `pilot_predictions_coarse.csv` (same row count).

**Why coarse?** Fine-grained WSD accuracy (43.0%) introduces noise that inflates JSD. Coarse clusters absorb intra-cluster errors → 84.9% effective accuracy, max JSD reduced from 0.835 → 0.532.

### Step 3.3 — Aggregate by decade

```bash
PYTHONPATH=src python scripts/diachronic/aggregate_temporal.py \
  --input data/processed/diachronic/predictions/pilot_predictions_coarse.csv \
  --output-dir data/processed/diachronic/aggregated/ \
  --prefix pilot_coarse
```

Groups predictions by (lemma × decade × sense), computes counts and proportions.
Output: `pilot_coarse_sense_counts.csv`, `pilot_coarse_stats.csv`.

### Step 3.4 — Compute change metrics

```bash
PYTHONPATH=src python scripts/diachronic/compute_change_metrics.py \
  --counts data/processed/diachronic/aggregated/pilot_coarse_sense_counts.csv \
  --output-dir data/processed/diachronic/metrics/ \
  --prefix pilot_coarse \
  --min-contexts 10
```

Computes four metrics per word:

| Metric | Formula | Interpretation |
|---|---|---|
| Entropy | `-Σ(p × log₂(p))` | How evenly distributed senses are per decade |
| JSD | Jensen-Shannon Divergence | Magnitude of distribution shift between consecutive decades |
| Emergence | proportion crosses 10% upward | Sense gains usage |
| Decline | proportion crosses 10% downward | Sense loses usage |

**Coarse results (v3):** Mean JSD=0.112, 806 transitions, 126 emergence / 118 decline events.
Top candidates by normalized JSD: **springen, Welle** (selected as case studies after manual reliability review — see `diachronic_pilot_results.ipynb` §7.1–7.2). v2 figures: Mean JSD=0.165, Max JSD=0.532, 194 emergence / 197 decline events.

### Step 3.5 — Visualize

```bash
PYTHONPATH=src python scripts/diachronic/visualize_trajectories.py --lemma Geist
PYTHONPATH=src python scripts/diachronic/visualize_trajectories.py --all-lemmas
```

Generates stacked area charts, sense × decade heatmaps, entropy plots per word.

### Full pipeline (coarse)

```bash
python scripts/diachronic/predict_dwds_senses.py \
  --dwds-file data/processed/dwds/pilot_contexts_v3.jsonl \
  --output data/processed/diachronic/predictions/pilot_predictions_v3.csv && \
python scripts/diachronic/apply_coarse_mapping.py \
  --predictions data/processed/diachronic/predictions/pilot_predictions_v3.csv \
  --output data/processed/diachronic/predictions/pilot_predictions_v3_coarse.csv && \
python scripts/diachronic/aggregate_temporal.py \
  --input data/processed/diachronic/predictions/pilot_predictions_v3_coarse.csv \
  --output-dir data/processed/diachronic/aggregated/ --prefix pilot_coarse_v3 && \
python scripts/diachronic/compute_change_metrics.py \
  --counts data/processed/diachronic/aggregated/pilot_coarse_v3_sense_counts.csv \
  --output-dir data/processed/diachronic/metrics/ --prefix pilot_coarse_v3 --min-contexts 10
```

---

## Sense Clustering

Clustering is a one-time step that produces `sense_clusters_reviewed.json`. It does not need to be re-run unless the sense inventory changes.

```bash
# Cluster all pilot words at t=0.10
PYTHONPATH=src python scripts/clustering/compute_sense_clusters.py \
  --threshold 0.10

# Interactive review app
python scripts/clustering/review_app.py
```

**Configuration:**
- Embedding: `core_gloss` with `passage:` prefix, `intfloat/multilingual-e5-large`
- Algorithm: agglomerative clustering, average linkage, cosine distance
- Threshold: t=0.10 (confirmed on 10 words including high-polysemy cases)
- Constraint: min 2 clusters, no upper bound

**Output:** `sense_clusters_reviewed.json` — 96 lemmas (90 pilot + 6 eval), 317 clusters, all labels manually reviewed. See `documentation/coarse_clustering_plan.md` for full rationale.

---

## Code Architecture

**`src/` = library code** — pure functions, importable, no CLI.
**`scripts/` = entry points** — argparse, logging, glue code that chains `src/` components.

```
src/
├── wsd/
│   ├── embedder.py               # Embedding wrapper (E5-large)
│   ├── embedding_strategies.py   # embed_with_representation(), get_text_for_embedding()
│   ├── sense_clustering.py       # cluster_all_lemmas(), compute_cluster_summary()
│   ├── similarity_wsd.py         # Nearest-neighbour WSD
│   └── wsd_evaluator.py          # Evaluator, metrics, error analysis
├── diachronic/
│   ├── io.py                     # load_senses(), load_dwds_contexts()
│   ├── aggregation.py            # aggregate_by_period()
│   ├── metrics.py                # entropy, JSD, emergence/decline detection
│   ├── predictor.py              # predict_senses()
│   └── visualization.py          # trajectory/heatmap/entropy plots
├── annotation/
│   └── task.py
└── data/
    ├── dwds/                     # DWDS API client and fetcher
    ├── germanet/                 # GermaNet XML parsing
    └── llm/                      # LLM client and text generator

scripts/
├── data/                         # process_germanet, enrich_senses_with_llm, fetch_dwds_contexts, regen_core_glosses
├── atemporal/                    # compare_embedding_methods, tune_representation_weights, generate_predictions
├── clustering/                   # compute_sense_clusters, review_sense_clusters, review_app
└── diachronic/                   # predict_dwds_senses, apply_coarse_mapping, aggregate_temporal,
                                  # compute_change_metrics, visualize_trajectories
```

### Key usage patterns

```python
# Load senses — always use load_all_senses(), NOT pd.read_csv (llm_examples is a list)
# load_all_senses() merges senses_pilot.jsonl (90 pilot words) and
# sense_text_representations.jsonl (10 eval words) into one DataFrame.
# Use load_senses(path) only if you explicitly need a single file.
from src.diachronic.io import load_all_senses
senses = load_all_senses()  # covers all 96 lemmas

# Embed with best representation
from src.wsd.embedder import Embedder
from src.wsd.embedding_strategies import embed_with_representation
embedder = Embedder("intfloat/multilingual-e5-large", normalize=True)
embeddings = embed_with_representation(senses, embedder, "weighted")
# → (n_senses, 1024) array; weighted = 0.5×llm_definition + 0.5×avg(llm_examples)
# Note: embed_with_representation("weighted") expects columns llm_example_1..5 (1-indexed)
```

---

## Known Limitations

- 26 pilot words have no GermaNet senses (fahren, sehen, halten, stehen, etc.) — deferred to Phase 4
- DWDS kern corpus has genre drift (Belletristik ↓, Wissenschaft ↑) — structural, documented in research log
- Coarse accuracy (84.9%) measured on contemporary texts; accuracy on 1900s data is unknown and likely lower
- Words with <100 total contexts (e.g. `groß` ~47) are unreliable — exclude from case studies
- 11 words were constrained to min=2 clusters (would have been 1 organically) — flag in analysis

---

## Further Reading

- `documentation/roadmap.md` — project status and next steps
- `documentation/research_log.md` — dated entries for all key decisions
- `documentation/data_catalog.md` — complete file reference
- `documentation/coarse_clustering_plan.md` — clustering design decisions
- `scripts/atemporal/README.md` — atemporal WSD experiments detail