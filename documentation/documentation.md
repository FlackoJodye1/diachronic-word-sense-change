# German Word Sense Disambiguation: Research Documentation

A comprehensive methodology document for building a German WSD benchmark dataset and analyzing diachronic sense variation.

---

## Table of Contents

1. [Research Overview](#1-research-overview)
2. [Background & Related Work](#2-background--related-work)
3. [Data Sources](#3-data-sources)
4. [Target Word Selection](#4-target-word-selection)
5. [Methodology: Benchmark Creation](#5-methodology-benchmark-creation)
6. [Methodology: Embedding-Based WSD](#6-methodology-embedding-based-wsd)
7. [Methodology: Diachronic Analysis](#7-methodology-diachronic-analysis)
8. [Experiments & Evaluation](#8-experiments--evaluation)
9. [Data Formats & Schema](#9-data-formats--schema)
10. [Limitations & Future Work](#10-limitations--future-work)
11. [References](#11-references)

---

## 1. Research Overview

### 1.1 Motivation

Word Sense Disambiguation (WSD) remains a fundamental challenge in natural language processing, particularly for languages with fewer resources than English. German, despite being a major European language, lacks comprehensive WSD benchmark datasets comparable to English resources like SemCor or the SensEval/SemEval evaluation datasets.

This research addresses two interconnected gaps:

1. **Lack of German WSD benchmarks**: Existing German WSD resources are limited in scope, coverage, and accessibility. A high-quality, manually annotated benchmark is essential for developing and evaluating German WSD systems.

2. **Unexplored diachronic sense variation**: While computational approaches to semantic change detection have gained traction, few studies examine how individual word senses emerge, decline, or shift in distribution over time using sense-annotated data.

### 1.2 Research Questions

**RQ1 (Benchmark)**: Can we construct a reliable, reproducible German WSD benchmark dataset by combining GermaNet sense inventories with DWDS corpus contexts?

**RQ2 (Diachronic)**: How do sense distributions for polysemous German words change across decades, and can we detect patterns of sense emergence, decline, or broadening/narrowing?

**RQ3 (Methods)**: Which embedding-based approaches best capture sense distinctions for German WSD, given the sparse nature of GermaNet glosses?

### 1.3 Dual Research Goals

This project pursues two complementary objectives:

#### Goal 1: Build a German WSD Benchmark Dataset

- Select a curated set of polysemous German target words
- Sample real corpus contexts from DWDS with temporal metadata
- Map contexts to GermaNet sense inventories
- Manually annotate each context with the correct sense
- Establish evaluation protocols and baseline results

#### Goal 2: Analyze Diachronic Sense Variation

- Stratify corpus samples by decade (e.g., 1900s-1990s)
- Use annotated data to track sense distributions over time
- Identify cases of sense emergence, decline, or semantic shift
- Validate findings against historical lexicographic evidence

### 1.4 Key Contributions

1. **Dataset**: A manually annotated German WSD benchmark with temporal metadata
2. **Methodology**: A reproducible pipeline for creating WSD datasets from GermaNet + DWDS
3. **Analysis**: Empirical findings on diachronic sense variation in German
4. **Baselines**: Embedding-based WSD baseline results for future comparison

---

## 2. Background & Related Work

### 2.1 Word Sense Disambiguation

Word Sense Disambiguation is the task of determining which sense of a polysemous word is intended in a given context. For example, the German word "Bank" can mean:

- A bench (seating furniture)
- A financial institution
- A sandbank (geographical feature)

WSD is typically framed as a classification problem: given a target word in context, select the correct sense from a predefined inventory.

#### Approaches to WSD

| Approach | Description | Examples |
|----------|-------------|----------|
| **Knowledge-based** | Use lexical resources (definitions, relations) | Lesk algorithm, graph-based methods |
| **Supervised** | Train classifiers on sense-annotated corpora | SVM, neural classifiers |
| **Embedding-based** | Compare contextual embeddings to sense representations | BERT-based, sentence transformers |
| **Hybrid** | Combine knowledge and learned representations | Our approach |

### 2.2 German WSD Resources

#### GermaNet

GermaNet is the German equivalent of Princeton WordNet, providing a hierarchical organization of German lexical units into synsets (sets of synonymous words representing a concept).

**GermaNet v19.0 Statistics:**
- 174,579 synsets
- 225,000+ lexical units
- 189,330 semantic relations
- Coverage: Nouns, Verbs, Adjectives

**Limitation**: Only 4.8% of synsets have paraphrases (definitions), making direct gloss-based disambiguation challenging.

#### Existing German WSD Datasets

| Dataset | Size | Source | Limitations |
|---------|------|--------|-------------|
| GermaNet examples | ~10K | GermaNet | Limited coverage, no temporal info |
| WebCAGe | ~15K | Web corpus | Crowdsourced, quality varies |

### 2.3 Diachronic Semantics

Diachronic semantics studies how word meanings change over time. Computational approaches typically use:

- **Distributional methods**: Compare word embeddings across time slices
- **Sense-level analysis**: Track individual sense frequencies (requires annotation)

Our approach enables sense-level diachronic analysis by annotating corpus data with temporal metadata.

#### Types of Semantic Change

| Change Type | Description | Example |
|-------------|-------------|---------|
| **Emergence** | New sense appears | "Maus" (computer mouse) |
| **Decline** | Sense falls out of use | Archaic meanings |
| **Broadening** | Sense expands in scope | Metaphorical extensions |
| **Narrowing** | Sense becomes more specific | Technical specialization |

### 2.4 Gap This Work Addresses

Existing approaches to diachronic semantics typically:
- Operate at the word level (not sense level)
- Lack gold-standard sense annotations across time
- Focus on English or require parallel corpora

Our work provides:
- **Sense-level temporal tracking** with manual annotations
- **German-specific resources** combining GermaNet + DWDS
- **Reproducible methodology** for creating similar datasets

---

## 3. Data Sources

### 3.1 GermaNet

#### Overview

GermaNet (German Network) is a lexical-semantic wordnet for German, developed at the University of Tübingen. It organizes German lexical units into synsets connected by semantic relations.

**Version used**: GermaNet v19.0 (germanet_v190)

#### Structure

```
GermaNet
├── Synsets (concepts)
│   ├── synset_id (e.g., s10603)
│   ├── semantic_class (e.g., Artefakt, Geschehen, Gefuehl)
│   └── paraphrase (definition, when available)
├── Lexical Units (word-sense pairs)
│   ├── lemma (e.g., "Bank")
│   ├── pos (NOUN, VERB, ADJ)
│   └── synset_id (links to synset)
└── Relations
    ├── has_hypernym (is-a)
    ├── has_hyponym (type-of)
    ├── has_meronym (part-of)
    └── is_related_to (association)
```

#### Key Statistics

| Metric | Value |
|--------|-------|
| Total synsets | 174,579 |
| Total lexical units | 225,000+ |
| Semantic relations | 189,330 |
| Synsets with paraphrases | ~4.8% |
| Relation distribution | 94.8% hypernym relations |

#### Semantic Classes

GermaNet organizes synsets into semantic classes:

- **Artefakt**: Man-made objects
- **Geschehen**: Events, processes
- **Gefuehl**: Emotions, feelings
- **Kognition**: Mental concepts
- **Kommunikation**: Communication acts
- **Mensch**: Human-related
- **Tier**: Animals
- **Pflanze**: Plants
- **Ort**: Locations
- **Zeit**: Time-related
- **Nahrung**: Food
- **natPhaenomen**: Natural phenomena
- **Form**: Shapes, forms
- **Attribut**: Properties, attributes
- **Besitz**: Possessions

#### Limitations and Mitigations

**Problem**: Sparse paraphrases (only 4.8% of synsets have definitions)

**Our solution**: Enrich sense representations using semantic relations (see Section 5.1)

### 3.2 DWDS Corpus

#### Overview

DWDS (Digitales Wörterbuch der deutschen Sprache / Digital Dictionary of the German Language) is a comprehensive German language resource maintained by the Berlin-Brandenburg Academy of Sciences.

**URL**: https://www.dwds.de

#### Corpus Components

| Corpus | Description | Time Range | Size |
|--------|-------------|------------|------|
| Kernkorpus | Core reference corpus | 1900-1999 | ~100M tokens |
| Zeitungskorpus | Newspaper corpus | 1945-present | Large |
| Webkorpus | Web-crawled texts | 2010s | Very large |
| Belletristik | Literary fiction | Various | Medium |

#### API Access

We access DWDS via their public API:

```python
# Corpus concordance (CSV export)
GET https://www.dwds.de/r/?q={term}&view=csv&corpus={corpus}&limit={limit}

# Frequency timeseries
GET https://www.dwds.de/r/?q={term}&view=timeseries&corpus={corpus}&format=json

# Dictionary entries
GET https://www.dwds.de/api/wb/snippet/?q={term}
```

#### Returned Data Fields

| Field | Description | Example |
|-------|-------------|---------|
| `Beleg` | Full sentence context | "Er sitzt auf der Bank." |
| `Datum` | Date of source | "1956" |
| `Bibl` | Bibliographic info | "Die Zeit, 15.03.1956" |
| `Textklasse` | Genre classification | "Belletristik" |
| `Korpus` | Source corpus | "kern" |

#### Why DWDS?

1. **Temporal coverage**: Spans 1900-present, enabling diachronic analysis
2. **Quality**: Curated, high-quality German texts
3. **Metadata**: Rich temporal and genre information
4. **Accessibility**: Public API with structured exports
5. **Representativeness**: Balanced across genres and time periods

#### Rate Limiting

- Delay: 0.6 seconds between requests
- Timeout: 30 seconds per request
- Implemented in `src/data/dwds/client.py`

---

## 4. Target Word Selection

### 4.1 Source of Target Words

The 750 target words for this project are sourced from:

> **Charbonnier, J., & Wartena, C. (2019).** *Predicting Word Concreteness and Imagery.*
> In Proceedings of the 13th International Conference on Computational Semantics (IWCS).
> https://aclanthology.org/W19-0415/

This word list provides:
- 750 German lemmas with concreteness ratings
- Psycholinguistic norms suitable for semantic research
- Coverage across concrete (e.g., "Apfel", "Bank") and abstract (e.g., "Angst", "Arbeit") words

**Why this source?**
- Established resource from peer-reviewed research
- Words selected for semantic/psycholinguistic properties
- Appropriate for studying sense variation and disambiguation
- Includes concreteness scores that may correlate with sense behavior

**Raw data file**: `data/raw/target_words/target_words_clean.xlsx`

### 4.2 Selection Criteria

From the 750 Charbonnier & Wartena words, we apply additional criteria for WSD:

#### Criterion 1: Polysemy

Words must have **2 or more distinct senses** in GermaNet to be useful for WSD evaluation.

| Polysemy Level | Sense Count | Suitability |
|----------------|-------------|-------------|
| Monosemous | 1 | Excluded |
| Low polysemy | 2-3 | Good baseline |
| Medium polysemy | 4-6 | Core evaluation set |
| High polysemy | 7+ | Challenging cases |

#### Criterion 2: Frequency

Words must be **sufficiently frequent** in DWDS to:
- Ensure enough contexts for annotation
- Enable per-decade stratification for diachronic analysis

**Minimum threshold**: At least 100 DWDS hits across all corpora

#### Criterion 3: Temporal Attestation

For diachronic analysis, words must be **attested across multiple decades** to track sense variation over time.

#### Criterion 4: Sense Distinguishability

Senses should be **meaningfully distinguishable** by human annotators:
- Not overly fine-grained technical distinctions
- Clear semantic differences (not just register/style)

#### Criterion 5: Part-of-Speech Distribution

Include a **balanced mix of POS categories**:
- Nouns (most polysemous in German)
- Verbs (complex sense structures)
- Adjectives (gradable meanings)

### 4.2 Selection Process

1. **Initial filtering**: Extract all GermaNet lemmas with 2+ senses
2. **Frequency check**: Query DWDS for corpus frequency
3. **Temporal check**: Verify attestation across decades
4. **Manual review**: Confirm sense distinguishability
5. **Balancing**: Ensure POS and polysemy level diversity

### 4.3 Current Target Word Statistics

**Total target words**: 752 unique lemmas

| POS | Count | Percentage |
|-----|-------|------------|
| NOUN | ~600 | 80% |
| VERB | ~100 | 13% |
| ADJ | ~50 | 7% |

#### Polysemy Distribution

| Sense Count | Word Count | Examples |
|-------------|------------|----------|
| 2 senses | ~300 | Adresse, Ampel, Angel |
| 3-4 senses | ~250 | Bank, Schloss, Dose |
| 5-7 senses | ~150 | Arbeit, Art, Rolle |
| 8+ senses | ~50 | halten (26), gehen (19), geben (11) |

### 4.4 Example Target Words

| Word | POS | Senses | Semantic Classes | Diachronic Interest |
|------|-----|--------|------------------|---------------------|
| Bank | NOUN | 3 | Artefakt, Ort | Stable |
| Maus | NOUN | 2 | Tier, Artefakt | Emergence (computer) |
| Netz | NOUN | 3 | Artefakt | Emergence (internet) |
| Rolle | NOUN | 5 | Artefakt, Geschehen | Metaphorical extension |
| sitzen | VERB | 7 | Various | Idiomatic uses |

---

## 5. Methodology: Benchmark Creation

### 5.1 Sense Inventory Construction

#### The Sparsity Problem

GermaNet provides the sense inventory (synsets) but with minimal textual descriptions:
- Only 4.8% of synsets have paraphrases
- Many senses are defined only by their position in the hierarchy

**Challenge**: How to represent senses for embedding-based disambiguation?

#### Our Solution: Semantic Relation Enrichment

We enrich sparse sense representations by leveraging GermaNet's semantic relations:

```
Enriched Sense Text = Lemma + Paraphrase (if available) + Hypernym Context + Semantic Class
```

**Example**:
- **Before**: "Bank" (no paraphrase)
- **After**: "Bank ist eine Art von Sitzgelegenheit. Bank ist ein Begriff aus Artefakt"

#### Enrichment Components

| Component | Source | Example |
|-----------|--------|---------|
| Lemma | Lexical unit | "Bank" |
| Paraphrase | Synset (when available) | "ein Geldinstitut" |
| Hypernym | has_hypernym relation | "ist eine Art von Sitzgelegenheit" |
| Meronyms | has_component_meronym | "hat Lehne und Sitzfläche" |
| Semantic class | Synset classification | "ist ein Begriff aus Artefakt" |

#### Implementation

See `src/data/germanet/transforms/sense_enrichment.py`:

```python
class SenseEnricher:
    def build_contextual_description(self, synset_id: str) -> str:
        """Build contextual description using semantic relations."""
        # 1. Get hypernym chain
        hypernym_chain = self.get_hypernym_chain(synset_id, max_depth=2)

        # 2. Get sibling synsets (same hypernym)
        siblings = self.get_sibling_synsets(synset_id)

        # 3. Get meronym relations
        components = self.get_related_lemmas(synset_id, ['has_component_meronym'])

        # 4. Combine into description
        return self._format_description(...)
```

#### Enrichment Statistics

| Metric | Value |
|--------|-------|
| Total enriched senses | 2,022 |
| Target words covered | 752 |
| Avg. words per sense text | 15.7 |
| Coverage (has description) | 100% |
| Senses with original paraphrase | ~5% |
| Senses enriched via relations | ~95% |

### 5.2 Context Sampling

#### Sampling Strategy

For each target word, we sample real corpus contexts from DWDS:

1. **Query construction**: Use lemma form with POS filter when needed
2. **Corpus selection**: Primarily Kernkorpus for temporal coverage
3. **Limit per word**: Configurable (default: 100 contexts)
4. **Deduplication**: Remove duplicate sentences

#### Temporal Stratification

For diachronic analysis, we optionally stratify samples by decade:

```
Target: 100 contexts per word
├── 1900-1909: 10 contexts
├── 1910-1919: 10 contexts
├── ...
├── 1980-1989: 10 contexts
└── 1990-1999: 10 contexts
```

**Rationale**: Ensures sufficient data per time slice for statistical analysis

#### Implementation

See `src/data/dwds/client.py`:

```python
class DWDSClient:
    def corpus_csv(self, term: str, corpus: str = "kern", limit: int = 100):
        """Fetch concordance hits as DataFrame."""
        # Returns: sentence, date, bibliography, genre, corpus
```

#### Current Corpus Statistics

| Metric | Value |
|--------|-------|
| Total sentences fetched | 4,809 |
| Words with contexts | 25 (pilot set) |
| Avg. contexts per word | ~190 |
| Date range | 1900-1999 |
| Primary corpus | Kernkorpus |

### 5.3 Annotation Process

#### Annotation Schema

For each context (sentence containing target word):

| Field | Description | Required |
|-------|-------------|----------|
| `instance_id` | Unique identifier | Yes |
| `target_word` | Lemma being annotated | Yes |
| `context` | Full sentence | Yes |
| `gold_sense_id` | GermaNet synset ID | Yes |
| `annotator` | Annotator identifier | Yes |
| `confidence` | Annotator confidence (1-5) | Optional |
| `notes` | Free-text comments | Optional |

#### Annotation Guidelines

1. **Read the full context**: Consider the entire sentence, not just adjacent words
2. **Consider all candidate senses**: Review GermaNet sense definitions before deciding
3. **Select the best-fitting sense**: Choose the sense that most accurately captures the intended meaning
4. **Mark uncertain cases**: Use the confidence field or flag for discussion
5. **Handle edge cases**:
   - If no sense fits, mark as "other/unclear"
   - If multiple senses apply, choose the primary one

#### Candidate Sense Presentation

For each target word, annotators see:

```
Target: Bank (NOUN)
Context: "Er setzte sich auf die Bank im Park."

Candidate senses:
1. [s12345] Artefakt - Sitzgelegenheit (eine Art von Möbel mit Sitzfläche)
2. [s23456] Ort - Geldinstitut (Finanzunternehmen für Geldgeschäfte)
3. [s34567] Ort - Sandbank (Ablagerung in Gewässern)

Select sense: [1]
```

#### Annotator Setup

**Current approach**: Single expert annotator (researcher)

**Future**: Multiple annotators for inter-annotator agreement (IAA)

#### Inter-Annotator Agreement (Planned)

For validation, we plan to:
- Have 2-3 annotators label a subset (10-20%)
- Calculate Cohen's Kappa or Fleiss' Kappa
- Resolve disagreements through discussion
- Target: κ > 0.7 (substantial agreement)

### 5.4 Dataset Statistics

#### Current Progress

| Metric | Value | Status |
|--------|-------|--------|
| Target words defined | 752 | Complete |
| Senses enriched | 2,022 | Complete |
| Contexts fetched | 4,809 | Pilot (25 words) |
| Contexts annotated | TBD | In progress |
| Words fully annotated | TBD | In progress |

#### Target Dataset Size

| Component | Target | Rationale |
|-----------|--------|-----------|
| Target words | 100-200 | Balance depth vs. breadth |
| Contexts per word | 50-100 | Statistical reliability |
| Total instances | 5,000-20,000 | Comparable to SemEval datasets |

#### Sense Distribution Goals

- Avoid extreme class imbalance (no sense > 80% of instances)
- Include rare senses when attested in corpus
- Track "most frequent sense" baseline per word

---

## 6. Methodology: Embedding-Based WSD

### 6.1 Sense Representation

#### Text-to-Embedding Pipeline

```
GermaNet Sense → Enriched Text → Embedding Model → Sense Vector
```

#### Current Approach

**Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

| Property | Value |
|----------|-------|
| Embedding dimension | 384 |
| Languages | 50+ (including German) |
| Architecture | MiniLM (distilled BERT) |
| Normalization | L2 normalized |

**Implementation**: `src/data/germanet/transforms/embedding_builder.py`

```python
class EmbeddingBuilder:
    DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def create_embeddings(self, sense_texts_df, text_column='text_for_embedding'):
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,  # L2 normalization
            batch_size=256
        )
        return embeddings  # Shape: (n_senses, 384)
```

#### Alternative Approaches to Compare

| Approach | Description | Text Input |
|----------|-------------|------------|
| **Baseline** | Lemma only | "Bank" |
| **Paraphrase** | Original GermaNet gloss | "ein Geldinstitut" |
| **Enriched** | Our enriched text | "Bank ist eine Art von..." |
| **Examples** | Include usage examples | Text + example sentences |
| **Hierarchical** | Include hypernym chain | Text + hypernym lemmas |

### 6.5 LLM-Generated Text Variants

In addition to template-based approaches, we use LLMs to generate natural language sense representations.

#### Module: `src/data/llm/`

| File | Purpose |
|------|---------|
| `client.py` | LLM client wrapper (OpenAI/Claude) |
| `sense_generator.py` | Generates definitions and examples per sense |

#### Prompt Templates: `config/prompts/`

| Prompt | Output |
|--------|--------|
| `definition.txt` | 1-2 sentence natural definition |
| `example_single.txt` | Single example sentence |
| `examples_multiple.txt` | N varied example sentences |

Each prompt receives the GermaNet baseline text (`sense_description`) to specify which sense to generate for.

#### Usage

```python
from src.data.llm import LLMClient, SenseTextGenerator

client = LLMClient(provider="openai")  # or "claude"
generator = SenseTextGenerator(client)

variants = generator.generate_all_variants(
    lemma="Bank",
    pos="NOUN",
    sense_description="Bank ist eine Art von Sitzgelegenheit..."
)
# Returns: {'definition': ..., 'example_1': ..., 'examples_5': [...]}
```

#### Script

```bash
python scripts/data_preparation/generate_llm_sense_texts.py --provider openai
```

Output: `data/evaluation/quick_eval/sense_variants.csv`

### 6.2 Disambiguation Algorithm

#### Context Embedding

For a sentence containing the target word:

```python
context_embedding = model.encode(sentence)  # 384-dim vector
```

#### Similarity Computation

Compare context embedding to all candidate sense embeddings:

```python
similarities = cosine_similarity(context_embedding, sense_embeddings)
predicted_sense = argmax(similarities)
```

#### Candidate Filtering

Only compare against senses that are valid for the target lemma:

```python
candidate_senses = get_senses_for_lemma(target_word)
similarity_scores = {sense: cosine_sim(context, sense_emb)
                     for sense in candidate_senses}
```

### 6.3 Baseline Strategies

#### Most Frequent Sense (MFS)

Always predict the most common sense in the training/gold data:

```python
def mfs_baseline(word, gold_data):
    return gold_data[gold_data['word'] == word]['sense'].mode()
```

**Expected performance**: Strong baseline (~50-60% for highly skewed distributions)

#### Random Baseline

Randomly select from candidate senses:

```python
def random_baseline(word, sense_inventory):
    candidates = sense_inventory[word]
    return random.choice(candidates)
```

**Expected performance**: 1/n where n = number of senses

#### First Sense Baseline

Always predict sense #1 in GermaNet (often the most common):

```python
def first_sense_baseline(word):
    return get_senses(word)[0]
```

### 6.4 Embedding Method Comparison Framework

#### Evaluation Setup

To compare different text-to-embedding approaches:

1. **Hold out evaluation set**: N annotated instances per word
2. **Generate sense embeddings**: Using each approach
3. **Run disambiguation**: Compare context to sense embeddings
4. **Compute metrics**: Accuracy, per-word accuracy, confidence correlation

#### Comparison Variables

| Variable | Options |
|----------|---------|
| Text construction | Lemma-only, paraphrase, enriched, +examples |
| Embedding model | MiniLM, mBERT, XLM-R, German BERT |
| Normalization | L2, none |
| Context window | Full sentence, target ± k words |

---

## 7. Methodology: Diachronic Analysis

### 7.1 Sense Distribution Over Time

#### Data Organization

For diachronic analysis, data is organized by time slice:

```
Word: "Netz"
├── 1900-1909: {fishing_net: 8, spider_web: 2, network: 0}
├── 1910-1919: {fishing_net: 7, spider_web: 3, network: 0}
├── ...
├── 1980-1989: {fishing_net: 3, spider_web: 2, network: 5}
└── 1990-1999: {fishing_net: 2, spider_web: 1, network: 7}
```

#### Metrics

| Metric | Definition | Use Case |
|--------|------------|----------|
| **Sense frequency** | Count per decade | Raw prevalence |
| **Sense proportion** | freq / total | Relative importance |
| **Dominance** | Max sense proportion | Polysemy balance |
| **Entropy** | -Σ p log p | Sense distribution evenness |

### 7.2 Semantic Change Detection

#### Types of Change We Track

**1. Sense Emergence**

A sense appears or becomes frequent:
```
1900s: 0% → 1990s: 40%
Example: "Maus" (computer mouse)
```

**2. Sense Decline**

A sense becomes rare or disappears:
```
1900s: 30% → 1990s: 5%
Example: Archaic meanings
```

**3. Sense Broadening**

A sense expands in usage contexts:
```
Detected via: Increased variety in context types
```

**4. Sense Narrowing**

A sense becomes more specialized:
```
Detected via: Decreased variety, technical contexts
```

### 7.3 Visualization Approaches

#### Time Series Plots

```
Sense proportion over decades
  │
  │    ●───●───●───●───●
  │   /                 \
  │  ●                   ●───●
  │
  └──────────────────────────→
     1900  1920  1940  1960  1980
```

#### Stacked Area Charts

Show how sense proportions change relative to each other over time.

#### Heat Maps

Matrix of (word × decade) showing dominant sense or entropy.

### 7.4 Validation

#### Lexicographic Evidence

Cross-reference detected changes with:
- DWDS dictionary entries (historical notes)
- Etymological sources
- Historical corpora annotations

#### Statistical Significance

For detected changes:
- Chi-square tests for distribution shifts
- Confidence intervals on proportions
- Multiple comparison corrections

---

## 8. Experiments & Evaluation

### 8.1 WSD Evaluation

#### Evaluation Protocol

1. **Train/test split**: By instance (not by word) for generalization
2. **Metrics**: Overall accuracy, per-word accuracy, macro-averaged F1
3. **Baselines**: MFS, random, first-sense
4. **Systems**: Embedding methods (Section 6.4)

#### Gold Standard Format

Required columns:
- `instance_id`: Unique identifier
- `target_word`: Lemma
- `gold_sense_id`: Correct GermaNet synset ID

#### Prediction Format

Required columns:
- `instance_id`: Matches gold standard
- `predicted_sense_id`: System's prediction
- `confidence`: Optional confidence score

#### Implementation

See `src/evaluation/wsd_evaluator.py`:

```python
class WSDEvaluator:
    def evaluate_predictions(self, predictions: pd.DataFrame) -> Dict:
        """
        Returns:
        - Overall accuracy
        - Per-word accuracy breakdown
        - Confidence analysis (if available)
        - Error analysis
        """
```

### 8.2 Preliminary Results

#### Current Baseline (Embedding-Based)

| Metric | Value | Notes |
|--------|-------|-------|
| Overall accuracy | 48.6% | On 25-word pilot set |
| Coverage | 100% | All instances have predictions |
| Avg. text length | 15.7 words | Enriched sense texts |
| Improvement over random | +42.4% | Significant |

#### Per-POS Performance

| POS | Accuracy | N instances | Avg. senses |
|-----|----------|-------------|-------------|
| NOUN | TBD | TBD | 2.8 |
| VERB | TBD | TBD | 4.2 |
| ADJ | TBD | TBD | 2.1 |

### 8.3 Diachronic Analysis Results

*To be completed after annotation*

#### Planned Analyses

1. **Case studies**: Deep dive on 5-10 words with interesting patterns
2. **Aggregate patterns**: Cross-word trends by semantic class
3. **Correlation with frequency**: Does overall frequency predict sense stability?

### 8.4 Error Analysis

#### Common Error Types

| Error Type | Description | Example |
|------------|-------------|---------|
| **Sense confusion** | Similar senses conflated | Bank (bench) vs Bank (river bank) |
| **Context poverty** | Insufficient context | Short sentences |
| **Rare sense miss** | Low-frequency senses underrepresented | Technical meanings |
| **Metaphorical use** | Figurative vs. literal | "Das sitzt!" (idiomatic) |

#### Error Analysis Tools

See `src/evaluation/error_analysis.py` for:
- Confusion matrix generation
- High-error word identification
- Sense pair confusion analysis

### 8.5 Evaluation Framework

#### Shared Modules (`src/evaluation/`)

The evaluation package provides reusable components for embedding-based WSD:

| Module | Class | Purpose |
|--------|-------|---------|
| `embedder.py` | `Embedder` | Unified text embedding using sentence-transformers |
| `similarity_wsd.py` | `SimilarityWSD` | Cosine similarity-based WSD predictor |
| `wsd_evaluator.py` | `WSDEvaluator` | Evaluates predictions against gold standard |
| `metrics.py` | `WSDMetrics` | Detailed metrics (precision, recall, F1, confusion matrix) |
| `error_analysis.py` | `ErrorAnalyzer` | Error pattern analysis |

##### Embedder

Unified text embedder with L2 normalization for cosine similarity:

```python
from src.evaluation import Embedder

embedder = Embedder(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    normalize=True  # L2 normalization for cosine similarity
)

# Encode texts
embeddings = embedder.encode(texts, batch_size=32, show_progress=True)
```

**Recommended models:**

| Model | Dimension | Speed | Quality |
|-------|-----------|-------|---------|
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Fast | Good |
| `paraphrase-multilingual-mpnet-base-v2` | 768 | Medium | Higher |
| `distiluse-base-multilingual-cased-v2` | 512 | Medium | Good |
| `intfloat/multilingual-e5-large` | 1024 | Slow | Best |

##### SimilarityWSD

Performs WSD by comparing context embeddings to sense embeddings:

```python
from src.evaluation import SimilarityWSD

# Create predictor from sense embeddings
wsd = SimilarityWSD.from_dataframe(
    sense_df=sense_variants,
    embeddings=sense_embeddings,
    sense_id_col="synset_id",
    lemma_col="lemma",
)

# Evaluate against gold standard
result = wsd.evaluate(
    context_embeddings=context_embeddings,
    lemmas=lemmas,
    gold_sense_ids=gold_sense_ids,
)

print(f"Accuracy: {result.accuracy:.3f}")
print(f"Top-3: {result.top3_accuracy:.3f}")
print(f"MRR: {result.mrr:.3f}")
```

#### Evaluation Scripts (`scripts/evaluation/`)

Three scripts for different evaluation purposes:

| Script | Purpose | Data Source | Output |
|--------|---------|-------------|--------|
| `compare_embedding_methods.py` | Benchmark models + text variants | `quick_eval/` gold annotations | Accuracy, Top-k, MRR |
| `real_dwds_wsd_evaluation.py` | Indicative evaluation on corpus | DWDS + GermaNet | MFS agreement, confidence |
| `evaluate_embeddings.py` | Quality analysis (not WSD accuracy) | GermaNet senses | Similarity stats, clustering |

##### compare_embedding_methods.py

**Purpose:** Benchmark different embedding models and text variants for WSD accuracy.

**Input:**
- `data/evaluation/quick_eval/sense_variants.csv` — sense definitions with text variants
- `data/evaluation/quick_eval/gold_annotations.csv` — human-annotated gold labels

**Process:**
1. For each model × text variant combination:
   - Embed all sense definition texts (with `passage:` prefix for E5 models)
   - Embed all context sentences (with `query:` prefix for E5 models)
   - Use cosine similarity to predict the best-matching sense
   - Compare predictions against gold labels
   - Aggressively free RAM after each variant and model

**Output:**
- `data/evaluation/quick_eval/embedding_comparison.csv`
- Console summary with best configuration

**Metrics reported:**
- **Accuracy** — % correct top-1 predictions
- **Top-3/Top-5** — % where correct sense is in top k predictions
- **MRR** — Mean Reciprocal Rank

**Usage:**
```bash
python scripts/evaluation/compare_embedding_methods.py                # Normal run
python scripts/evaluation/compare_embedding_methods.py --clear-cache  # Clear disk cache after each model
```

**Features:**
- **Memory safety**: Aggressive RAM cleanup after each text variant and model
- **E5 prefix handling**: Automatically applies `query:`/`passage:` prefixes for E5 models
- **Disk cache management**: `--clear-cache` flag deletes models from HuggingFace cache after evaluation
- **Fair comparison**: Only includes sentence-transformer models (excludes base BERT models)

**Models compared:**

| Model | Type | Notes |
|-------|------|-------|
| `paraphrase-multilingual-MiniLM-L12-v2` | Multilingual | Fast, 384 dim |
| `distiluse-base-multilingual-cased-v2` | Multilingual | Classic, 512 dim |
| `paraphrase-multilingual-mpnet-base-v2` | Multilingual | Higher quality, 768 dim |
| `LaBSE` | Multilingual | Cross-lingual, 768 dim |
| `multilingual-e5-base` | E5 | Retrieval-optimized, 768 dim |
| `multilingual-e5-large` | E5 | Best quality, 1024 dim |

**Benchmark Results (January 2026):**

| Rank | Model | Text Variant | Accuracy | MRR |
|------|-------|--------------|----------|-----|
| 1 | multilingual-e5-large | text_llm_examples_5 | **0.527** | 0.723 |
| 2 | multilingual-e5-large | text_llm_definition | 0.516 | 0.693 |
| 3 | mpnet-base-v2 | text_llm_examples_5 | 0.516 | 0.707 |
| 4 | distiluse-v2 | text_llm_examples_5 | 0.505 | 0.697 |

**Key findings:**
- LLM-generated examples (`text_llm_examples_5`) consistently outperform baseline and definitions
- E5 models benefit significantly from proper query/passage prefixes
- Best configuration: `multilingual-e5-large` + `text_llm_examples_5` → 52.7% accuracy

##### real_dwds_wsd_evaluation.py

**Purpose:** Evaluate WSD on real corpus sentences (indicative, no gold labels).

**Input:**
- DWDS corpus sentences (`data/dwds_contexts/`)
- GermaNet sense inventory (`data/processed/germanet/wsd_senses_enhanced.csv`)

**Process:**
1. Find polysemous words with sufficient corpus coverage
2. Embed senses and contexts
3. Predict senses via cosine similarity
4. Compute MFS agreement rate (how often model predicts most frequent sense)

**Output:**
- `data/processed/dwds/dwds_wsd_evaluation_results.json`

**Note:** Without human-annotated gold labels, this provides indicative results only. Use `compare_embedding_methods.py` for true accuracy evaluation.

**Usage:**
```bash
python scripts/evaluation/real_dwds_wsd_evaluation.py
```

##### evaluate_embeddings.py

**Purpose:** Exploratory quality analysis of sense embeddings (NOT WSD accuracy).

**Analyses:**
- Text quality metrics (word counts, sources)
- Semantic diversity (class distribution, POS distribution)
- Embedding similarity analysis (same-word vs different-word senses)
- Clustering analysis (silhouette score)

**Usage:**
```bash
python scripts/evaluation/evaluate_embeddings.py
```

### 8.6 Diachronic Analysis Pipeline

The diachronic analysis pipeline tracks how word sense distributions change over time (decades). It consists of four scripts that form a complete workflow.

#### Pipeline Overview

```
predict_dwds_senses.py → aggregate_temporal.py → compute_change_metrics.py → visualize_trajectories.py
```

| Script | Input | Output | Purpose |
|--------|-------|--------|---------|
| `predict_dwds_senses.py` | DWDS contexts + senses | `dwds_predictions.csv` | Predict sense for each context |
| `aggregate_temporal.py` | predictions | `temporal_*_sense_counts.csv`, `temporal_*_stats.csv` | Aggregate by decade |
| `compute_change_metrics.py` | sense counts | `metrics_*_entropy.csv`, `metrics_*_divergence.csv`, emergence/decline CSVs | Compute change metrics |
| `visualize_trajectories.py` | counts + metrics | PNG plots | Generate visualizations |

#### Script 1: predict_dwds_senses.py

**Purpose:** Apply embedding-based WSD to DWDS corpus contexts.

**Key features:**
- Configurable sense input file (`--senses-file`)
- Configurable text column (`--text-column`) — use `text_llm_examples_5` for best accuracy
- E5 model prefix handling (query:/passage:)
- Memory-safe with garbage collection

**Usage:**
```bash
# Default (baseline text)
python scripts/diachronic/predict_dwds_senses.py

# With LLM-generated text variants (recommended when available)
python scripts/diachronic/predict_dwds_senses.py \
    --senses-file data/evaluation/quick_eval/sense_variants.csv \
    --text-column text_llm_examples_5
```

#### Script 2: aggregate_temporal.py

**Purpose:** Aggregate predictions by time period (decade or year).

**Output:**
- `temporal_decade_sense_counts.csv` — sense counts and proportions per lemma per decade
- `temporal_decade_stats.csv` — summary stats (n_contexts, n_senses, dominant sense, mean confidence)

**Usage:**
```bash
python scripts/diachronic/aggregate_temporal.py
python scripts/diachronic/aggregate_temporal.py --period-column year  # aggregate by year
```

#### Script 3: compute_change_metrics.py

**Purpose:** Compute quantitative metrics for sense change.

**Metrics computed:**
- **Entropy** — higher entropy = more uniform distribution across senses (more ambiguous)
- **Jensen-Shannon Divergence** — measures how different distributions are between adjacent periods
- **Emergence events** — senses that appear above threshold in period N but were absent/low in N-1
- **Decline events** — senses that drop below threshold

**Usage:**
```bash
python scripts/diachronic/compute_change_metrics.py
python scripts/diachronic/compute_change_metrics.py --threshold 0.05  # more sensitive detection
```

#### Script 4: visualize_trajectories.py

**Purpose:** Generate publication-quality visualizations.

**Plot types:**
- Stacked area charts showing sense proportions over time
- Entropy over time line plots
- Sense-period heatmaps
- Summary statistics plots

**Usage:**
```bash
python scripts/diachronic/visualize_trajectories.py --lemma Bahn  # single word
python scripts/diachronic/visualize_trajectories.py --all-lemmas  # all words
```

#### Complete Pipeline Example

```bash
# Step 1: Predict senses for DWDS contexts
python scripts/diachronic/predict_dwds_senses.py

# Step 2: Aggregate by decade
python scripts/diachronic/aggregate_temporal.py

# Step 3: Compute change metrics
python scripts/diachronic/compute_change_metrics.py

# Step 4: Generate visualizations
python scripts/diachronic/visualize_trajectories.py --all-lemmas
```

**Output files:**
```
data/diachronic/
├── dwds_predictions.csv           # Raw predictions
├── temporal_decade_sense_counts.csv
├── temporal_decade_stats.csv
├── metrics_decade_entropy.csv
├── metrics_decade_divergence.csv
├── metrics_decade_emergence.csv
├── metrics_decade_decline.csv
└── plots/
    ├── trajectory_*.png
    ├── heatmap_*.png
    ├── entropy_*.png
    └── summary_entropy.png
```

---

## 9. Data Formats & Schema

### 9.1 Key Files

#### Sense Inventory

**File**: `data/processed/germanet/wsd_senses.csv`

| Column | Type | Description |
|--------|------|-------------|
| `lemma` | str | Target word |
| `pos` | str | Part of speech (NOUN/VERB/ADJ) |
| `synset_id` | str | GermaNet synset ID (e.g., "s10603") |
| `class` | str | Semantic class (e.g., "Artefakt") |
| `paraphrase` | str | Original GermaNet gloss (may be empty) |
| `text_for_embedding` | str | Enriched text for embedding |
| `text_source` | str | Source of text (direct_relations/example) |
| `sense_frequency` | int | Sense rank within word (1 = most common) |
| `embedding_word_count` | int | Word count of enriched text |
| `has_examples` | bool | Whether GermaNet examples exist |

#### Corpus Contexts

**File**: `data/dwds_contexts/fetch_progress_*.jsonl`

| Field | Type | Description |
|-------|------|-------------|
| `lemma` | str | Target word |
| `pos` | str | Part of speech |
| `sentence` | str | Full corpus sentence |
| `date` | str | Year or date of source |
| `genre` | str | Text genre (Belletristik, etc.) |
| `bibl` | str | Bibliographic reference |
| `corpus` | str | Source corpus (kern, etc.) |
| `year` | int | Extracted year |

#### Gold Standard Annotations

**File**: `data/evaluation/gold_standard.csv` (to be created)

| Column | Type | Description |
|--------|------|-------------|
| `instance_id` | str | Unique identifier |
| `target_word` | str | Lemma |
| `context` | str | Sentence |
| `gold_sense_id` | str | Correct synset ID |
| `annotator` | str | Annotator ID |
| `date_annotated` | str | Annotation date |
| `confidence` | int | Annotator confidence (1-5) |

### 9.2 Directory Structure

```
data/
├── raw/
│   └── target_word_lists/          # Source word lists
├── external/
│   └── germanet_v190/              # Original GermaNet XML
├── processed/
│   ├── germanet/
│   │   ├── wsd_senses.csv          # Main sense inventory
│   │   └── wsd_senses_enhanced.csv # Enhanced version
│   ├── germanet_base/
│   │   ├── synsets.parquet         # Parsed synsets
│   │   ├── lexical_units.parquet   # Parsed lexical units
│   │   ├── relations.parquet       # Semantic relations
│   │   └── lemma_senses.parquet    # Lemma-sense mappings
│   └── target_words/
│       └── target_words_with_germanet_sense_counts.tsv
├── dwds_contexts/
│   └── fetch_progress_*.jsonl      # DWDS corpus sentences
└── evaluation/
    ├── gold_standard.csv           # Annotated data
    └── results/                    # Evaluation outputs
```

### 9.3 Reproducibility

#### Environment Setup

```bash
# Clone repository
git clone [repo-url]
cd Worde-Sense-Disambiguation

# Create environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Pipeline Execution

```bash
# 1. Process GermaNet data
python scripts/data_preparation/process_germanet.py

# 2. Fetch DWDS contexts
python scripts/data_preparation/fetch_dwds_contexts.py

# 3. Build embeddings
python scripts/data_preparation/build_embeddings.py

# 4. Run WSD evaluation
python scripts/evaluation/run_wsd_evaluation.py
```

---

## 10. Limitations & Future Work

### 10.1 Current Limitations

#### Data Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Small pilot set** | Limited generalization | Expand to 100+ words |
| **Single annotator** | No IAA validation | Add annotators |
| **DWDS availability** | Some words lack contexts | Alternative corpora |
| **Temporal gaps** | Uneven decade coverage | Weighted sampling |

#### Methodological Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| **Enrichment quality** | Varies by synset | Manual review |
| **Embedding model** | May not capture German well | Test German BERT |
| **Context window** | Full sentence may be noisy | Experiment with window |

### 10.2 Future Work

#### Short-Term

1. **Complete annotation** of pilot word set (25 words)
2. **Add annotators** for inter-annotator agreement
3. **Expand corpus** to 100+ target words
4. **Compare embedding models** (German BERT, XLM-R)

#### Medium-Term

1. **Diachronic analysis** on fully annotated data
2. **Fine-tune models** on annotated data
3. **Release benchmark** for community use
4. **Submit paper** on methodology and findings

#### Long-Term

1. **Expand to full GermaNet** coverage
2. **Cross-lingual extension** (other Germanic languages)
3. **Dynamic sense tracking** with continuous annotation
4. **Integration with NLP pipelines** (spaCy, etc.)

---

## 11. References

### Lexical Resources

- **GermaNet**: Hamp, B., & Feldweg, H. (1997). GermaNet - a lexical-semantic net for German. In *Proceedings of the ACL workshop on Automatic Information Extraction and Building of Lexical Semantic Resources*.

- **DWDS**: Klein, W., & Geyken, A. (2010). Das Digitale Wörterbuch der Deutschen Sprache (DWDS). *Lexicographica*, 26, 79-96.

### Target Word Source

- **Charbonnier & Wartena (2019)**: Charbonnier, J., & Wartena, C. (2019). Predicting Word Concreteness and Imagery. In *Proceedings of the 13th International Conference on Computational Semantics (IWCS)*. https://aclanthology.org/W19-0415/

### Word Sense Disambiguation

- **WSD Survey**: Navigli, R. (2009). Word sense disambiguation: A survey. *ACM Computing Surveys*, 41(2), 1-69.

- **SensEval/SemEval**: Edmonds, P., & Cotton, S. (2001). SENSEVAL-2: Overview. In *Proceedings of SENSEVAL-2*.

### Diachronic Semantics

- **Semantic Change Detection**: Kutuzov, A., et al. (2018). Diachronic word embeddings and semantic shifts: a survey. In *Proceedings of COLING*.

- **Laws of Semantic Change**: Hamilton, W. L., et al. (2016). Diachronic word embeddings reveal statistical laws of semantic change. In *Proceedings of ACL*.

### Embedding Models

- **Sentence Transformers**: Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of EMNLP*.

- **Multilingual Models**: Conneau, A., et al. (2020). Unsupervised cross-lingual representation learning at scale. In *Proceedings of ACL*.

---

## Appendix A: Semantic Classes in GermaNet

| Class | German | Description |
|-------|--------|-------------|
| Artefakt | Artifact | Man-made objects |
| Attribut | Attribute | Properties, qualities |
| Besitz | Possession | Ownership-related |
| Form | Form | Shapes, structures |
| Gefuehl | Feeling | Emotions |
| Geschehen | Event | Processes, happenings |
| Gruppe | Group | Collections |
| Kognition | Cognition | Mental concepts |
| Koerper | Body | Body parts |
| Kommunikation | Communication | Speech acts |
| Menge | Quantity | Amounts |
| Mensch | Human | People |
| Motiv | Motive | Reasons |
| Nahrung | Food | Edibles |
| natPhaenomen | Natural Phenomenon | Natural events |
| Ort | Location | Places |
| Pflanze | Plant | Vegetation |
| privativ | Privative | Absences |
| Relation | Relation | Relationships |
| Substanz | Substance | Materials |
| Tier | Animal | Fauna |
| Tops | Top | Abstract tops |
| Verhalten | Behavior | Conduct |
| Zeit | Time | Temporal concepts |

---

## Appendix B: Example Annotated Instances

### Example 1: Bank (Clear Case)

```
Instance ID: bank_001
Target: Bank (NOUN)
Context: "Er setzte sich auf die Bank im Park und las ein Buch."
Translation: "He sat down on the bench in the park and read a book."

Candidate senses:
1. s12345 (Artefakt): Sitzgelegenheit - bench, seating
2. s23456 (Ort): Geldinstitut - financial institution
3. s34567 (Ort): Sandbank - sandbank

Gold sense: s12345 (bench)
Rationale: "setzte sich auf" (sat down on) + "im Park" (in the park) indicate physical seating furniture
Confidence: 5 (very confident)
```

### Example 2: Netz (Ambiguous Case)

```
Instance ID: netz_042
Target: Netz (NOUN)
Context: "Die Information verbreitete sich schnell im Netz."
Translation: "The information spread quickly on the net/network."

Candidate senses:
1. s11111 (Artefakt): Fangnetz - fishing/catching net
2. s22222 (Artefakt): Spinnennetz - spider web
3. s33333 (Artefakt): Computernetzwerk - computer network

Gold sense: s33333 (computer network)
Rationale: "Information verbreitete sich" (information spread) is typical of internet contexts
Confidence: 4 (confident, though "Netz" alone is ambiguous)
Date: 1995 (supports modern sense)
```

---

*Last updated: 2026-01-13*
*Version: 1.1*