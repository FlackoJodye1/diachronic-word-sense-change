# German WSD Pipeline: Complete Guide

A comprehensive guide to the German Word Sense Disambiguation (WSD) pipeline, from data preparation through diachronic analysis.

**Version:** 2.0 (with integrated LLM sense enrichment)
**Last Updated:** 2026-03-08

---

## Table of Contents

1. [Overview](#overview)
2. [Data Sources](#data-sources)
3. [Phase 1: Data Preparation](#phase-1-data-preparation)
4. [Phase 2: WSD Evaluation](#phase-2-wsd-evaluation)
5. [Phase 3: Diachronic Analysis](#phase-3-diachronic-analysis)
6. [Code Architecture](#code-architecture)
7. [File Formats & Locations](#file-formats--locations)
8. [Usage Examples](#usage-examples)

---

## Overview

### Research Goals

1. **Build a German WSD benchmark** using GermaNet + DWDS corpus
2. **Analyze diachronic sense variation** (how word senses change 1900-1999)
3. **Optimize embedding methods** for German sense disambiguation

### Pipeline Philosophy

```
Data Preparation → Evaluation → Production Analysis
     (once)         (iterative)    (publication)
```

**Key Innovation:** LLM-enhanced sense representations improve accuracy from 36.6% → 54.8%

---

## Data Sources

### GermaNet v19.0

**What:** German semantic network (like WordNet)

**Structure:**
- 174,579 synsets (concepts)
- 225,000+ lexical units (word-sense pairs)
- Semantic relations (hypernym, meronym, etc.)
- **Problem:** Only 4.8% have definitions

**Our solution:** Enrich with template-based text + LLM-generated text

**Location:** `data/external/germanet_v190/`

### DWDS Corpus

**What:** Digital Dictionary of the German Language (Digitales Wörterbuch der deutschen Sprache)

**Coverage:**
- Kernkorpus: 1900-1999 (100M tokens)
- Zeitungskorpus: 1945-present
- Rich temporal and genre metadata

**Access:** Public API (rate-limited)

**Location:** `data/dwds_contexts/`

### Target Words

**Source:** Charbonnier & Wartena (2019) - 750 German words with concreteness ratings

**Subsets:**
- **Full set:** 750 words
- **Pilot set:** 116 words (for diachronic study)
- **Quick eval:** 10 words (for method comparison)

**Location:** `data/raw/target_words/`

---

## Phase 1: Data Preparation

The foundation layer - done once, reused everywhere.

### Step 1: Process GermaNet

**Script:** `scripts/data/process_germanet.py`

**What it does:**
1. Parses GermaNet XML files
2. Extracts synsets, lexical units, semantic relations
3. Creates template-based sense descriptions
4. Outputs baseline sense file

**Template format:**
```
"{lemma} ist eine Art von {hypernym}. {lemma} ist ein Begriff aus {class}"

Example: "Bank ist eine Art von Sitzgelegenheit. Bank ist ein Begriff aus Artefakt"
```

**Output:** `data/processed/germanet/target_word_senses.csv`

**Columns:**
- `lemma` - target word
- `pos` - part of speech (NOUN, VERB, ADJ)
- `synset_id` - GermaNet ID (e.g., s12345)
- `class` - semantic class (e.g., Artefakt)
- `paraphrase` - original GermaNet gloss (mostly empty)
- `text_for_embedding` - **baseline text representation**

**Run:**
```bash
python scripts/data/process_germanet.py
```

### Step 2: Enrich with LLM Text (NEW!)

**Script:** `scripts/data/enrich_senses_with_llm.py`

**What it does:**
1. Loads baseline sense file
2. For each sense, generates via LLM:
   - Natural language definition
   - 5 diverse usage examples
3. Saves enhanced file with both baseline + LLM text

**Why:** LLM-generated text improves WSD accuracy from 36.6% → 54.8%

**Input:** `target_word_senses.csv` (from Step 1)

**Output:**
- `target_word_senses_enhanced.csv` (CSV format)
- `target_word_senses_enhanced.jsonl` (JSONL format - recommended)

**New columns added (CSV):**
- `llm_definition` - LLM-generated definition
- `llm_example_1` through `llm_example_5` - usage examples

**JSONL structure:**
- `llm_examples` - array of 5 examples (cleaner than separate columns)
- `baseline` - renamed from `text_for_embedding` for clarity

**Scope options:**

```bash
# Pilot words only (90 words, ~$2, 15 min)
python scripts/data/enrich_senses_with_llm.py \
  --word-list data/raw/target_words/pilot_words_diachronic.txt \
  --provider openai

# All 750 words (~$10, 60 min)
python scripts/data/enrich_senses_with_llm.py \
  --provider openai

# Quick eval only (10 words, ~$0.20, 2 min)
python scripts/data/enrich_senses_with_llm.py \
  --word-list data/evaluation/quick_eval/words.csv \
  --provider openai
```

**Features:**
- Automatic checkpointing every 50 senses
- Resume from checkpoint if interrupted
- Dry-run mode for testing
- Cost estimation before running

### Step 3: Fetch DWDS Contexts

**Script:** `scripts/data/fetch_dwds_contexts.py`

**What it does:**
1. Queries DWDS API for corpus sentences
2. Filters by quality (length, duplicates, etc.)
3. Extracts temporal metadata (year, decade)
4. Saves as JSONL (one context per line)

**Output:** `data/dwds_contexts/*.jsonl`

**Example output:**
```json
{
  "lemma": "Geist",
  "sentence": "Der Geist der Zeit verlangt nach Veränderung.",
  "year": 1956,
  "decade": 1950,
  "genre": "Zeitung",
  "bibl": "Die Zeit, 15.03.1956"
}
```

**Run:**
```bash
python scripts/data/fetch_dwds_contexts.py \
  --word-list data/raw/target_words/pilot_words_diachronic.txt \
  --output data/dwds_contexts/pilot_diachronic_contexts.jsonl
```

---

## Phase 2: WSD Evaluation

Optimize embedding methods on small eval set before large-scale analysis.

### Goal

Find the best way to represent senses as embeddings for WSD.

### Evaluation Set

**10 carefully selected words:**
- Stratified by polysemy (low: 2-3, medium: 4-6, high: 7+)
- Balanced by POS (nouns, verbs, adjectives)
- 93 gold-annotated contexts

**Files:**
- `data/evaluation/quick_eval/words.csv` - 10 words
- `data/evaluation/quick_eval/contexts.csv` - 95 DWDS contexts
- `data/evaluation/quick_eval/gold_annotations.csv` - human labels

### Methods Compared

| Representation | Text Source | E5-Large Accuracy |
|----------------|-------------|-------------------|
| baseline | GermaNet template | 36.6% |
| llm_definition | LLM-generated definition | 41.9% |
| llm_example_1 | Single LLM example | 34-47% (varies) |
| llm_examples_avg | Average of 5 examples | ~48% |
| **weighted** | **0.5×def + 0.5×avg(ex)** | **54.8%** ✓ BEST |

### Best Configuration

**Model:** `intfloat/multilingual-e5-large` (1024-dim)
**Text:** Weighted combination (0.5 definition + 0.5 averaged examples)
**Accuracy:** 54.8%

**Weight optimization:** α=0.5 validated via grid search (see below)

**How it works:**
```python
# Embed definition
def_emb = embed("Eine Bank ist ein längliches Sitzmöbel...")

# Embed 5 examples and average
ex_embs = [
    embed("Er setzte sich auf die Bank im Park."),
    embed("Die alte Bank unter dem Baum..."),
    ...  # 3 more
]
avg_ex_emb = mean(ex_embs)

# Weighted combination
sense_emb = 0.5 * def_emb + 0.5 * avg_ex_emb
```

### Weight Validation

Tested α ∈ [0.0, 1.0] where `weighted_emb = α×definition + (1-α)×examples`:

| α | Accuracy | Interpretation |
|---|----------|----------------|
| 0.0 | 51.6% | Pure examples |
| 0.4 | 52.7% | Examples-leaning |
| **0.5** | **54.8%** | **Equal weight** ✓ |
| 0.6 | 51.6% | Definition-leaning |
| 1.0 | 41.9% | Pure definition |

**Result:** α=0.5 is empirically optimal. Performance degrades symmetrically in both directions.

```bash
# Run weight tuning
python scripts/atemporal/tune_representation_weights.py
```

### Run Evaluation

```bash
# Compare all methods
python scripts/atemporal/compare_embedding_methods.py

# Output: data/evaluation/quick_eval/embedding_comparison.csv
```

### Key Findings

1. **LLM text beats templates:** +18.2 pp improvement
2. **Combination beats single:** Definition alone (41.9%) < Weighted (54.8%)
3. **High top-k accuracy:** Top-3: 96.8%, Top-5: 100%
4. **Model matters:** E5-large > mpnet > MiniLM

---

## Phase 3: Diachronic Analysis

Track how word senses change over time (1900-1999).

### Overview

**Goal:** Detect sense emergence, decline, and distribution shifts across decades

**Data:** 30,337 DWDS contexts for 116 pilot words (v2, --max-tokens 60)

**Method:** Unsupervised WSD (using best method from Phase 2) + temporal aggregation

**Key Insight:** Use model predictions to detect patterns at scale, then validate interesting cases manually

### The Pipeline (4 Steps)

```
Historical Sentences → Predict Senses → Aggregate by Decade → Compute Metrics → Visualize
    (30,337)             (24,325)         (word × decade)      (518+521 events)  (plots)
```

### Step 1: Predict Senses

**Script:** `scripts/diachronic/predict_dwds_senses.py`

**What it does:**
1. Loads enhanced senses (with LLM text)
2. Embeds all sense texts once (using weighted method)
3. For each historical sentence:
   - Embeds the sentence
   - Compares to candidate sense embeddings
   - Predicts most similar sense
   - Records confidence score

**Example:**
```
Input: "Der Geist der Zeit verlangt nach Veränderung" (1956)
       ↓
Embed sentence → [0.23, -0.15, 0.87, ...] (1024-dim)
       ↓
Compare to 6 "Geist" senses:
  s28850 (ghost):      0.65
  s112176 (intellect): 0.82 ← WINNER
  s73964 (spirit):     0.71
  ...
       ↓
Predicted: s112176, confidence: 0.82
```

**Output:** `data/diachronic/pilot_predictions_weighted_v2.csv`

**Columns:**
- Original context columns (lemma, sentence, year, decade)
- `predicted_sense_id` - model's prediction
- `confidence` - cosine similarity score
- `sense_class` - semantic class of predicted sense

**Run:**
```bash
PYTHONPATH=src python scripts/diachronic/predict_dwds_senses.py \
  --dwds-file data/dwds_contexts/pilot_diachronic_contexts_v2.jsonl \
  --senses-file data/processed/germanet/target_word_senses_enhanced.jsonl \
  --representation weighted \
  --model intfloat/multilingual-e5-large
```

**Stats from v2 run:**
- 24,325 predictions (out of 30,337 contexts)
- 90 words analyzed
- Only 22/894 word-decades sparse (min 5 contexts filter)

### Step 2: Aggregate by Decade

**Script:** `scripts/diachronic/aggregate_temporal.py`

**What it does:**
Groups predictions by (word × decade) and counts sense usage.

**Example for "Geist" in 1950s:**
```
Raw predictions:
  1952: s28850  (ghost)
  1954: s112176 (intellect)
  1956: s112176 (intellect)
  1957: s112176 (intellect)
  1958: s73964  (spiritual)
  1959: s73963  (zeitgeist)

Aggregated:
  s28850:  1/6 = 16.7%
  s112176: 3/6 = 50.0% ← dominant
  s73964:  1/6 = 16.7%
  s73963:  1/6 = 16.7%
```

**Output files:**

1. `temporal_decade_sense_counts.csv` - counts per word-decade-sense
   ```csv
   lemma,decade,predicted_sense_id,count,total,proportion
   Geist,1950,s112176,3,6,0.500
   ```

2. `temporal_decade_stats.csv` - summary per word-decade
   ```csv
   lemma,decade,n_contexts,n_senses,dominant_sense,dominant_proportion
   Geist,1950,6,4,s112176,0.500
   ```

**Run:**
```bash
PYTHONPATH=src python scripts/diachronic/aggregate_temporal.py
```

### Step 3: Compute Change Metrics

**Script:** `scripts/diachronic/compute_change_metrics.py`

**What it does:**
Calculates quantitative metrics to detect and measure semantic change.

#### Metric 1: Entropy (Ambiguity)

**Formula:** `-Σ(p × log₂(p))`

**Interpretation:**
- Low (~0): One sense dominates → clear meaning
- High (2+): Many senses equally common → ambiguous

**Example:**
```
Geist in 1900s: {s28850: 67%, s112176: 17%, s73964: 17%}
Entropy = 1.08 (moderate ambiguity)

Geist in 1950s: {s28850: 17%, s112176: 50%, s73964: 17%, s73963: 17%}
Entropy = 1.79 (higher ambiguity - 4 senses more evenly distributed)
```

**Output:** `metrics_decade_entropy.csv`

#### Metric 2: Divergence (Distribution Change)

**Metric:** Jensen-Shannon Divergence (JSD)

**Range:** 0 (identical) to 1 (completely different)

**Example:**
```
Geist: 1950s → 1960s
1950s: {s28850: 17%, s112176: 50%, s73964: 17%, s73963: 17%}
1960s: {s28850: 25%, s112176: 75%, s73964: 0%, s73963: 0%}
JSD = 0.35 (moderate change)

Berg: 1900s → 1910s
1900s: {sense_A: 100%}
1910s: {sense_B: 100%}
JSD = 1.00 (complete shift!)
```

**Output:** `metrics_decade_divergence.csv`

#### Metric 3: Emergence (New Senses Appearing)

**Rule:** Sense goes from <10% → ≥10%

**Example:**
```
Geist sense s73963 (zeitgeist):
1940s: 0.0%   ← absent
1950s: 16.7%  ← EMERGED!
```

**Output:** `metrics_decade_emergence.csv` (518 events detected in v2)

#### Metric 4: Decline (Senses Disappearing)

**Rule:** Sense goes from ≥10% → <10%

**Example:**
```
Geist sense s73964 (spiritual being):
1950s: 16.7%  ← present
1960s: 0.0%   ← DECLINED!
```

**Output:** `metrics_decade_decline.csv` (521 events detected in v2)

**Run:**
```bash
PYTHONPATH=src python scripts/diachronic/compute_change_metrics.py

# Adjust threshold for more/less sensitivity
python scripts/diachronic/compute_change_metrics.py --threshold 0.05
```

### Step 4: Visualize

**Script:** `scripts/diachronic/visualize_trajectories.py`

**What it does:**
Generates publication-quality plots showing temporal patterns.

#### Plot Types

**1. Trajectory Plot (Stacked Area Chart)**

Shows sense proportions over time as colored bands.

```
     100% ┤████████████████████ red sense (appears late)
          │░░░░░░░░░███████████ green sense (declines)
          │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ orange sense (stable)
          │███████████████████ blue sense (dominant early)
       0% └─────────────────────────────────────────
         1900  1920  1940  1960  1980
```

**2. Heatmap (Sense × Decade Grid)**

Which sense dominates when (darker = higher proportion).

**3. Entropy Plot (Line Graph)**

How ambiguous the word is over time.

**4. Summary Plots**

Compare all 90 words at once (entropy distribution, divergence ranking, etc.)

**Run:**
```bash
# Single word
PYTHONPATH=src python scripts/diachronic/visualize_trajectories.py --lemma Geist

# All words
PYTHONPATH=src python scripts/diachronic/visualize_trajectories.py --all-lemmas
```

**Output:** `data/diachronic/plots/*.png`

### Complete Diachronic Pipeline

Run all 4 steps:

```bash
# Step 1: Predict senses
PYTHONPATH=src python scripts/diachronic/predict_dwds_senses.py

# Step 2: Aggregate by decade
PYTHONPATH=src python scripts/diachronic/aggregate_temporal.py

# Step 3: Compute metrics
PYTHONPATH=src python scripts/diachronic/compute_change_metrics.py

# Step 4: Visualize
PYTHONPATH=src python scripts/diachronic/visualize_trajectories.py --all-lemmas
```

### Interpreting Results

**What's reliable:**
- Large-scale patterns across multiple decades
- Comparisons between words
- Identifying candidates for manual analysis

**What's less reliable:**
- Individual decade-to-decade shifts (small samples)
- Exact sense proportions (model errors accumulate)
- Rare senses (insufficient data)

**Validation:**
- Manually inspect contexts for words with dramatic changes
- Cross-reference with DWDS dictionary notes
- Compare to historical/etymological sources

**Key findings from pilot (v2):**
- 518 emergence events, 521 decline events
- Max JSD: 0.835 (no more JSD=1.0 artifacts)
- Only 22/894 word-decades sparse after min-5-contexts filter

---

## Code Architecture

### Design Philosophy

**`src/` = Library Code (Reusable)**
- Import these in scripts and notebooks
- Pure functions, well-defined interfaces
- No CLI code, no `if __name__ == "__main__"`

**`scripts/` = Executable Entry Points**
- Run from command line
- Contains `argparse`, logging setup
- Glue code that chains `src/` components

### Module Structure

```
src/
├── annotation/           # Annotation task management
│   └── task.py
├── data/                 # Data processing
│   ├── dwds/            # DWDS API client
│   │   ├── client.py
│   │   ├── fetcher.py
│   │   └── util/
│   │       └── context_utils.py  # Text cleaning, deduplication
│   ├── germanet/        # GermaNet processing
│   │   ├── make_dataset.py
│   │   ├── parsing/
│   │   └── transforms/
│   ├── llm/             # LLM integration
│   │   ├── client.py
│   │   └── text_generator.py
├── diachronic/          # Temporal analysis
│   ├── io.py            # Data loading
│   ├── aggregation.py   # Temporal grouping
│   ├── metrics.py       # Entropy, divergence, change detection
│   ├── predictor.py     # Sense prediction
│   └── visualization.py # Plotting
└── wsd/                     # WSD engine
    ├── embedder.py          # Embedding wrapper
    ├── embedding_strategies.py # Sense text representations
    ├── similarity_wsd.py    # WSD algorithm
    └── wsd_evaluator.py     # Evaluator, metrics, error analysis

scripts/
├── data/                # Data preparation
│   ├── process_germanet.py
│   ├── enrich_senses_with_llm.py
│   └── fetch_dwds_contexts.py
├── diachronic/          # Temporal analysis runners
│   ├── predict_dwds_senses.py
│   ├── aggregate_temporal.py
│   ├── compute_change_metrics.py
│   └── visualize_trajectories.py
└── atemporal/           # WSD evaluation runners
    ├── compare_embedding_methods.py
    ├── evaluate_embeddings.py
    ├── generate_predictions.py
    ├── tune_representation_weights.py
    └── real_dwds_wsd_evaluation.py
```

### Key Modules

#### `src.data.sense_representations` (NEW!)

Utilities for working with sense text representations:

```python
from src.data.sense_representations import (
    embed_with_representation,
    get_text_for_embedding,
    validate_sense_file
)

# Load enhanced senses
senses = pd.read_csv('target_word_senses_enhanced.csv')

# Get text for different representations
baseline_texts = get_text_for_embedding(senses, "baseline")
llm_texts = get_text_for_embedding(senses, "llm_definition")

# Embed using best method (weighted)
embeddings = embed_with_representation(
    senses,
    embedder,
    representation="weighted"
)
# Returns: (n_senses, 1024) array
# Each = 0.5 * embed(definition) + 0.5 * avg(embed(examples))
```

#### `src.diachronic`

Complete toolkit for temporal analysis:

```python
from diachronic import (
    load_dwds_contexts,
    predict_senses,
    aggregate_by_period,
    compute_entropy_per_period,
    plot_sense_trajectory
)

# Load and predict
contexts = load_dwds_contexts('contexts.jsonl')
predictions = predict_senses(contexts, senses, representation="weighted")

# Aggregate and analyze
agg = aggregate_by_period(predictions, period_column="decade")
entropy = compute_entropy_per_period(agg)

# Visualize
plot_sense_trajectory(agg, lemma="Geist", output_dir="plots/")
```

#### `src.wsd`

WSD evaluation framework:

```python
from evaluation import Embedder, SimilarityWSD, WSDEvaluator

# Embed senses
embedder = Embedder("intfloat/multilingual-e5-large")
sense_embeddings = embedder.encode(sense_texts)

# Create WSD predictor
wsd = SimilarityWSD(sense_embeddings, sense_ids, sense_lemmas)

# Evaluate
results = wsd.evaluate(context_embeddings, lemmas, gold_sense_ids)
print(f"Accuracy: {results.accuracy:.3f}")
```

---

## File Formats & Locations

### Input Files

```
data/
├── external/
│   └── germanet_v190/              # GermaNet XML files
├── raw/
│   └── target_words/
│       ├── target_words_clean.xlsx # 750 source words
│       └── pilot_words_diachronic.txt  # 116 pilot words
```

### Processed Files

```
data/processed/
├── germanet/
│   ├── target_word_senses.csv              # Baseline (Step 1 output)
│   ├── target_word_senses_enhanced.csv     # With LLM text (Step 2 output - CSV)
│   ├── target_word_senses_enhanced.jsonl   # With LLM text (Step 2 output - JSONL)
│   └── llm_checkpoints/                    # Checkpoints during enrichment
│       ├── checkpoint_50.csv
│       └── checkpoint_100.csv
├── germanet_base/
│   ├── synsets.parquet
│   ├── lexical_units.parquet
│   └── relations.parquet
```

### Corpus Files

```
data/dwds_contexts/
├── pilot_diachronic_contexts_v2.jsonl  # 30,337 historical contexts (v2, --max-tokens 60)
└── fetch_progress_*.jsonl              # Other corpus samples
```

### Evaluation Files

```
data/evaluation/
├── quick_eval/
│   ├── words.csv                    # 10 eval words
│   ├── contexts.csv                 # 95 contexts
│   ├── gold_annotations.csv         # Human labels
│   └── embedding_comparison.csv     # Method comparison results
└── preliminary/
    └── input/
        └── sense_text_representations.jsonl  # LLM text for eval words
```

### Diachronic Analysis Files

```
data/diachronic/
├── pilot_predictions_weighted_v2.csv             # Step 1 output (v2)
├── pilot_weighted_v2_decade_sense_counts.csv     # Step 2 output
├── pilot_weighted_v2_decade_stats.csv            # Step 2 output
├── pilot_weighted_v2_filtered_decade_*.csv       # Step 3 output (min 5 contexts filter)
└── plots/                                        # Step 4 output
    ├── trajectory_*.png
    ├── heatmap_*.png
    └── entropy_*.png
```

### Key File Schemas

#### target_word_senses_enhanced (CSV format)

```csv
lemma,pos,synset_id,class,paraphrase,text_for_embedding,llm_definition,llm_example_1,...,llm_example_5
Geist,NOUN,s112176,Tops,,Geist ist eine Art von...,Eine übernatürliche...,Der Geist erschien...,Ein kalter Wind...
```

**Columns:**
- Baseline: `lemma`, `pos`, `synset_id`, `class`, `paraphrase`, `text_for_embedding`
- LLM: `llm_definition`, `llm_example_1` through `llm_example_5`

#### target_word_senses_enhanced (JSONL format) ✨ Recommended

```json
{
  "lemma": "Geist",
  "pos": "NOUN",
  "synset_id": "s112176",
  "class": "Tops",
  "baseline": "Geist ist eine Art von...",
  "llm_definition": "Eine übernatürliche...",
  "llm_examples": [
    "Der Geist erschien plötzlich...",
    "Ein kalter Wind...",
    "Die alte Burg...",
    "Nachts hört man...",
    "Der Spuk endete..."
  ]
}
```

**Benefits of JSONL:**
- Examples grouped as array (not 5 separate columns)
- Cleaner field names (`baseline` vs `text_for_embedding`)
- Easier to parse and manipulate in code
- One JSON object per line (efficient streaming)

#### dwds_predictions.csv

```csv
lemma,sentence,year,decade,predicted_sense_id,confidence,sense_class
Geist,Der Geist der Zeit...,1956,1950,s112176,0.82,Kognition
```

#### temporal_decade_sense_counts.csv

```csv
lemma,decade,predicted_sense_id,count,total,proportion
Geist,1950,s112176,3,6,0.500
```

---

## Usage Examples

### Complete Workflow: New Project

**1. Define target words**
```bash
echo -e "Bank\nGeist\nNetz" > my_words.txt
```

**2. Generate LLM text (if not exists)**
```bash
python scripts/data/enrich_senses_with_llm.py \
  --word-list my_words.txt \
  --output my_enhanced_senses.csv \
  --provider openai
```

**3. Fetch contexts**
```bash
python scripts/data/fetch_dwds_contexts.py \
  --word-list my_words.txt \
  --output my_contexts.jsonl
```

**4. Run WSD with best method**
```bash
PYTHONPATH=src python scripts/diachronic/predict_dwds_senses.py \
  --dwds-file my_contexts.jsonl \
  --senses-file my_enhanced_senses.jsonl \
  --representation weighted
```

**5. Analyze temporal patterns**
```bash
PYTHONPATH=src python scripts/diachronic/aggregate_temporal.py
PYTHONPATH=src python scripts/diachronic/compute_change_metrics.py
PYTHONPATH=src python scripts/diachronic/visualize_trajectories.py --all-lemmas
```

### Quick Experiments in Python

```python
import pandas as pd
from src.wsd import Embedder
from src.data.sense_representations import embed_with_representation

# Load enhanced senses
senses = pd.read_csv('data/processed/germanet/target_word_senses_enhanced.csv')

# Filter to words of interest
my_senses = senses[senses['lemma'].isin(['Geist', 'Bank'])]

# Create embedder
embedder = Embedder("intfloat/multilingual-e5-large")

# Compare representations
baseline_emb = embed_with_representation(my_senses, embedder, "baseline")
weighted_emb = embed_with_representation(my_senses, embedder, "weighted")

# Use for WSD...
```

### Check Available Representations

```python
from src.data.sense_representations import get_available_representations

senses = pd.read_csv('target_word_senses.csv')
print(get_available_representations(senses))
# → ['baseline']  (only baseline available)

senses_enhanced = pd.read_csv('target_word_senses_enhanced.csv')
print(get_available_representations(senses_enhanced))
# → ['baseline', 'llm_definition', 'llm_example_1', 'llm_examples_concat',
#    'llm_examples_avg', 'weighted']
```

---

## Summary

### Pipeline Phases

| Phase | Purpose | Duration | Cost |
|-------|---------|----------|------|
| **Phase 1: Data Prep** | Create canonical sense & context files | One-time | $0-10 |
| **Phase 2: Evaluation** | Find best embedding method | Iterative | $0 |
| **Phase 3: Production** | Large-scale analysis with best method | Per study | $0 |

### Key Decisions

✅ **Use weighted LLM representation** (54.8% accuracy vs 36.6% baseline)
✅ **E5-large model** (best performance)
✅ **One canonical enhanced sense file** (reuse everywhere)
✅ **Checkpointed LLM generation** (resumable)

### Current Status

All pipeline steps through Phase 3a are complete (v2). The next step is **manual validation**:
- Inspect contexts for top-changed words to confirm changes are real (not WSD artifacts)
- Cross-reference emergence/decline events with DWDS dictionary notes
- Statistical significance testing on distribution shifts

See `documentation/roadmap.md` for detailed status tracking.

---

## Quick Reference

### Most Common Commands

```bash
# Generate LLM text
python scripts/data/enrich_senses_with_llm.py --word-list WORDS.txt --provider openai

# Predict senses (diachronic)
PYTHONPATH=src python scripts/diachronic/predict_dwds_senses.py

# Complete diachronic pipeline
PYTHONPATH=src python scripts/diachronic/predict_dwds_senses.py && \
PYTHONPATH=src python scripts/diachronic/aggregate_temporal.py && \
PYTHONPATH=src python scripts/diachronic/compute_change_metrics.py && \
PYTHONPATH=src python scripts/diachronic/visualize_trajectories.py --all-lemmas

# Evaluate embedding methods
python scripts/atemporal/compare_embedding_methods.py
```

### Important Files

```
data/processed/germanet/target_word_senses_enhanced.jsonl           # Canonical sense file
data/dwds_contexts/pilot_diachronic_contexts_v2.jsonl               # Historical contexts (v2)
data/diachronic/pilot_predictions_weighted_v2.csv                   # Sense predictions (v2)
data/diachronic/pilot_weighted_v2_filtered_decade_*.csv             # Change metrics (v2)
data/diachronic/plots/                                              # Visualizations
```

**Note:** Both CSV and JSONL formats contain the same data. Use JSONL for cleaner structure and easier programming.

### Getting Help

- Pipeline overview: This file
- Detailed diachronic walkthrough: `diachronic_pipeline_guide.md`
- Research log: `research_log.md`
- Full documentation: `documentation.md`

---

*Version 2.0 - January 2026*