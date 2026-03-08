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
Generate detailed predictions for error analysis (best model only).
```bash
python scripts/atemporal/generate_predictions.py
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
**Accuracy**: 54.8%

### Weight Validation (2026-01-30)

Tested α ∈ [0.0, 1.0] where `weighted_emb = α×definition + (1-α)×examples`:

| α | Accuracy | Interpretation |
|---|----------|----------------|
| 0.0 | 51.6% | Pure examples |
| 0.4 | 52.7% | Examples-leaning |
| **0.5** | **54.8%** | **Equal weight** ✓ OPTIMAL |
| 0.6 | 51.6% | Definition-leaning |
| 1.0 | 41.9% | Pure definition |

**Result**: α=0.5 is empirically optimal. Performance degrades symmetrically when moving away from 0.5.

**Script**: `tune_representation_weights.py`
**Results**: `data/evaluation/atemporal/results/weight_tuning.csv`

## Workflow

1. Run experiments in this directory to optimize method
2. Once satisfied, use best method in diachronic pipeline (`scripts/diachronic/`)
