# German Word Sense Disambiguation (WSD)

A research system for diachronic Word Sense Disambiguation of German words using GermaNet senses, DWDS corpus data, and multilingual sentence embeddings.

## Project Overview

The goal is to track how the meaning distribution of polysemous German words shifts over time (1900–1999) by:
1. Enriching GermaNet senses with LLM-generated definitions and examples
2. Disambiguating ~30k DWDS corpus sentences using embedding-based WSD
3. Clustering fine-grained senses into coarser groupings to reduce noise
4. Aggregating predictions per decade and computing change metrics (JSD, emergence, decline)

## Current Status

| Stage | Status | Key Result |
|-------|--------|------------|
| Data preparation | Complete | `senses_pilot.jsonl` — 90 words, 595 senses, `core_gloss` + `llm_definition` + `llm_examples` |
| Atemporal WSD | Complete | 54.8% fine / **81.7% coarse** accuracy (E5-large, weighted representation) |
| Diachronic pipeline (v3) | In progress | 90,130 contexts (+197% vs v2). Predictions running overnight (use `--resume`). |
| Coarse clustering | Complete | 305 clusters (90 pilot lemmas), t=0.10, manual review done |
| Coarse diachronic rerun (v2) | Complete | Max JSD 0.835→0.532, events 518→194 (~60% reduction) |
| Analysis & validation | Pending v3 | Top v2 candidates: spielen, fassen, Kasse, passen — rankings will update after v3 pipeline |

## Key Numbers

- **Pilot corpus (v3)**: 90,130 sentences, 90 words, 1900–1999 (DWDS kern, `--limit 5000`)
- **WSD accuracy**: 54.8% fine-grained → 81.7% coarse-grained (93-instance eval set)
- **Sense clustering**: 595 fine senses → 305 coarse clusters (39.5% reduction, t=0.10)
- **Top change signal (v2)**: spielen, fassen, Kasse, passen (by JSD) — to be re-ranked on v3

## Repository Structure

```
├── data/
│   ├── processed/
│   │   ├── dwds/                  # Corpus contexts (pilot_contexts.jsonl)
│   │   ├── germanet/              # Enriched senses + cluster mapping
│   │   ├── diachronic/            # Predictions, aggregations, metrics
│   │   └── sense_clustering/      # Cluster summary artifacts
│   └── evaluation/
│       └── atemporal/             # 93-instance gold eval set (10 words)
├── notebooks/
│   ├── analysis/                  # diachronic_pilot_results.ipynb, sense_clustering_analysis.ipynb
│   └── exploration/               # Target word and corpus exploration
├── scripts/
│   ├── atemporal/                 # WSD method comparison and tuning
│   ├── clustering/                # Sense clustering + review app
│   ├── diachronic/                # Full diachronic pipeline scripts
│   └── data/                      # Data fetching and enrichment
└── src/
    ├── wsd/                       # Embedder, embedding strategies, sense clustering
    ├── diachronic/                # IO, aggregation, metrics
    ├── annotation/                # Annotation utilities
    └── data/                      # GermaNet and DWDS data loaders
```

## Diachronic Pipeline

```
senses_pilot.jsonl
    → scripts/diachronic/predict_dwds_senses.py       → pilot_predictions.csv
    → scripts/diachronic/apply_coarse_mapping.py      → pilot_predictions_coarse.csv
    → scripts/diachronic/aggregate_temporal.py        → pilot_coarse_sense_counts.csv
    → scripts/diachronic/compute_change_metrics.py    → pilot_coarse_{entropy,divergence,emergence,decline}.csv
```

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file with API keys used by LLM enrichment scripts:
```
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here

# Optional: DWDS session cookie for licensed corpora (ZEIT, FAZ, Spiegel)
# Log in at dwds.de → DevTools → Application → Cookies → copy session cookie value
DWDS_SESSION_COOKIE=
```

## Best WSD Configuration

**Model**: `intfloat/multilingual-e5-large`
**Representation**: Weighted (0.5 × `llm_definition` + 0.5 × avg(`llm_examples`))
**Clustering**: `core_gloss` embeddings, cosine distance threshold t=0.10, min 2 clusters

## Documentation

- `documentation/roadmap.md` — project status and next steps
- `documentation/research_log.md` — dated entries for all key decisions
- `documentation/pipeline.md` — full pipeline walkthrough
- `documentation/coarse_clustering_plan.md` — clustering design decisions
- `scripts/atemporal/README.md` — atemporal WSD experiments detail

## Known Limitations

- 26 pilot words have no GermaNet senses (fahren, sehen, halten, etc.) — deferred
- DWDS kern corpus has genre drift (Belletristik ↓, Wissenschaft ↑) — structural, documented
- Coarse accuracy (81.7%) measured on contemporary texts; accuracy on 1900s data is unknown
- Words with <100 total contexts are unreliable regardless of clustering