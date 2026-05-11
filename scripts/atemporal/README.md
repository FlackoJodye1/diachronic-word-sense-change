# Atemporal WSD Experiments

This directory contains scripts for **optimizing WSD methods** using a small evaluation set.
These experiments are **separate from the diachronic analysis pipeline**.

## Purpose

Find the best:
- Embedding model (E5-large, BGE-M3, etc.)
- Text representation (definition, examples, weighted combination)
- Representation weights (α×definition + (1-α)×examples)

## Data

- **Input**: `data/evaluation/atemporal/input/`
  - 93 gold-annotated instances
  - 10 words (stratified by polysemy)
  - LLM-generated sense texts
  
- **Output**: `data/evaluation/atemporal/results/`

## Scripts

### compare_embedding_methods.py
Compare different embedding models and text representations.
```bash
python scripts/atemporal/compare_embedding_methods.py
```

### generate_predictions.py
Generate detailed predictions for error analysis. Three variants:
- `--variant weighted` (default): E5-large + 0.5·definition + 0.5·mean(examples) — best system
- `--variant text-only`: E5-large + GermaNet hypernym text only (no LLM enrichment)
- `--variant random`: analytical random chance (1/n_senses fine, 1/n_clusters coarse) — no embedder

Add `--cluster-mapping` to also report coarse accuracy.
```bash
python scripts/atemporal/generate_predictions.py
python scripts/atemporal/generate_predictions.py --variant text-only --cluster-mapping data/processed/germanet/sense_clusters_reviewed.json
python scripts/atemporal/generate_predictions.py --variant random    --cluster-mapping data/processed/germanet/sense_clusters_reviewed.json
```

### evaluate_embeddings.py
Exploratory analysis of sense text quality (no WSD evaluation).
```bash
python scripts/atemporal/evaluate_embeddings.py
```

### real_dwds_wsd_evaluation.py
Evaluate on DWDS corpus without gold labels (MFS baseline reference).
```bash
python scripts/atemporal/real_dwds_wsd_evaluation.py
```

### tune_representation_weights.py
Optimize the weight α for combining definition and example embeddings.
```bash
python scripts/atemporal/tune_representation_weights.py
python scripts/atemporal/tune_representation_weights.py --fine-grid  # Fine-grained search
```

## Current Best Configuration

**Model**: `intfloat/multilingual-e5-large`
**Representation**: Weighted (0.5×definition + 0.5×avg(5 examples))

| Variant | Fine | Coarse |
|---|---|---|
| Random baseline | 24.5% | 50.5% |
| Text-only ablation (E5 + GermaNet hypernyms) | 36.6% | 75.3% |
| **Weighted (best, E5 + LLM def+examples)** | **43.0%** | **84.9%** |

**Reading the deltas:**
- Random → text-only: value of the encoder + GermaNet structure.
- Text-only → weighted: value of LLM-generated sense enrichment on top.

Random is analytical (1/n_senses fine, 1/n_clusters coarse — no embedder, no frequency data).

**Note (2026-04-07):** The `sense_text_representations.jsonl` was re-generated when `core_gloss` was added. LLM outputs for `llm_definition` and `llm_examples` changed due to non-determinism, lowering the weighted fine accuracy from the originally reported 54.8% to 43.0%. The coarse accuracy improved from 81.7% → 84.9%. α=0.5 remains the chosen weight (originally validated; `weight_tuning.csv` contains the core_gloss tuning run, not the current llm_definition one).

**Script**: `tune_representation_weights.py`
**Results**: `data/evaluation/atemporal/results/weight_tuning.csv`

## Workflow

1. Run experiments in this directory to optimize method
2. Once satisfied, use best method in diachronic pipeline (`scripts/diachronic/`)
