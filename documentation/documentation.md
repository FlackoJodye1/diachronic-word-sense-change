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
9. [Project Architecture](#9-project-architecture)
10. [Data Formats & Schema](#10-data-formats--schema)
11. [Limitations & Future Work](#11-limitations--future-work)
12. [References](#12-references)

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

**Limitation**: Only 4.8% of synsets have paraphrases (definitions), making direct gloss-based disambiguation impossible.

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

**Total target words**: 750 unique lemmas

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

### 4.4 Diachronic Pilot Subset

For the diachronic pilot study, a strategic subset of 116 words was selected from the 750-word list to maximize coverage of interesting temporal patterns while keeping the initial analysis manageable.

#### Selection Strategy

1. **All high-polysemy words (8+ senses)**: 36 words
   - Hypothesis: More senses → more likely to show temporal variation
   - Examples: halten (26), gehen (19), schlagen (15), zeigen (15), Zug (14)

2. **Medium-polysemy sample (4-7 senses)**: 65 words
   - All 50 words with 6-7 senses (upper medium range)
   - Random sample of 15 words with 4-5 senses
   - Examples: Bahn (7), Fall (7), Feld (7), Art (4), Bass (4)

3. **Low-polysemy controls (2-3 senses)**: 15 words
   - Random sample for baseline comparison
   - Examples: Anfang (2), Berg (2), Flasche (2), Frosch (3)

#### Pilot Distribution

| Category | Sense Range | Count | Percentage |
|----------|-------------|-------|------------|
| High | 8+ senses | 36 | 31.0% |
| Medium | 4-7 senses | 65 | 56.0% |
| Low | 2-3 senses | 15 | 13.0% |
| **Total** | | **116** | **100%** |

**Rationale**: This distribution over-samples high-polysemy words (31% vs 4.8% in full set) to maximize the likelihood of detecting interesting temporal patterns in the pilot phase.

**File**: `data/raw/target_words/pilot_words_diachronic.txt`

### 4.5 Example Target Words

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

See `src/data/germanet/transforms/sense_builder.py` (`SenseTextBuilder`) and `relation_text_builder.py` (`RelationTextBuilder`):

```python
class SenseTextBuilder:
    def create_sense_texts(self, synsets_df, lexical_units_df) -> pd.DataFrame:
        """Build embedding-ready text for each synset."""
        # 1. Normalize lexical units (aggregate lemmas, examples per synset)
        # 2. Combine: paraphrase + lemmas + examples into text_for_embedding
        ...

class RelationTextBuilder:
    def build_enriched_text(self, synset_id: str, original_text: str) -> str:
        """Enrich sense text with related synset information (hypernyms, meronyms, etc.)."""
        # BFS traversal of the relation graph
        related_terms = self.get_related_text_terms(synset_id)
        return TEXT_SEPARATOR.join([original_text] + [t for t, _, _ in related_terms])
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
| Total sentences fetched (diachronic pilot v2) | 30,337 |
| Words with contexts | 90 (of 116 pilot words) |
| Avg. contexts per word | ~337 |
| Date range | 1900-1999 |
| Primary corpus | Kernkorpus |
| Fetch parameters | --max-tokens 60, --limit 1000 |

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
| Senses enriched (LLM) | 2,022 | Complete |
| Diachronic contexts fetched (v2) | 30,337 | Complete (90 words) |
| Atemporal gold annotations | 93 instances / 10 words | Complete |
| Second annotation set | 98 instances / 10 words | Fetched, not annotated |

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

#### Best Approach (from evaluation)

**Model**: `intfloat/multilingual-e5-large`

| Property | Value |
|----------|-------|
| Embedding dimension | 1024 |
| Languages | 100+ (including German) |
| Architecture | E5 (retrieval-optimized) |
| Normalization | L2 normalized |
| Special handling | Requires `query:` / `passage:` prefixes |

**Accuracy**: 54.8% on 93-instance gold eval set

**Implementation**: `src/wsd/embedder.py`, `src/wsd/embedding_strategies.py`

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
| `text_generator.py` | Generates definitions and examples per sense |

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
python scripts/data/enrich_senses_with_llm.py --provider openai
```

**Output:**
- `data/processed/germanet/target_word_senses_enhanced.csv` (CSV format)
- `data/processed/germanet/target_word_senses_enhanced.jsonl` (JSONL format - recommended)

Both formats contain the same data. JSONL format uses cleaner structure:
- Examples grouped as `llm_examples` array (not 5 separate columns)
- `baseline` field instead of `text_for_embedding`

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

See `src/wsd/wsd_evaluator.py`:

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

### 8.2 Evaluation Results

#### Dataset Summary

| Metric | Value |
|--------|-------|
| Total instances | 93 |
| Lemmas evaluated | 10 |
| Models compared | 5 |
| Text variants | 8 |

#### Best Model Configurations

| Rank | Model | Text Variant | Accuracy | Top-3 | Top-5 | MRR |
|------|-------|--------------|----------|-------|-------|-----|
| 1 | multilingual-e5-large | weighted (def + avg examples) | **54.8%** | 96.8% | 100% | 0.725 |
| 2 | bge-m3 | weighted (def + avg examples) | 51.6% | 96.8% | 100% | 0.717 |
| 3 | multilingual-e5-large | llm_example_4 | 47.3% | 90.3% | 100% | 0.679 |
| 4 | mpnet-base-v2 | weighted (def + avg examples) | 47.3% | 89.2% | 98.9% | 0.677 |
| 5 | LaBSE | weighted (def + avg examples) | 46.2% | 92.5% | 100% | 0.680 |

#### Model Comparison (Best Variant per Model)

| Model | Best Accuracy | Best Variant |
|-------|---------------|--------------|
| `intfloat/multilingual-e5-large` | 54.8% | weighted def+examples |
| `BAAI/bge-m3` | 51.6% | weighted def+examples |
| `paraphrase-multilingual-mpnet-base-v2` | 47.3% | weighted def+examples |
| `LaBSE` | 46.2% | weighted def+examples |
| `intfloat/multilingual-e5-base` | 46.2% | weighted def+examples |

#### Text Variant Comparison (E5-large)

| Text Variant | Accuracy | Notes |
|--------------|----------|-------|
| weighted (def + avg examples) | 54.8% | Best overall |
| llm_example_4 | 47.3% | Single example varies |
| llm_example_3 | 46.2% | |
| llm_definition | 41.9% | Definition alone |
| baseline | 36.6% | GermaNet enriched text |

#### Key Findings

1. **Weighted combination wins**: Combining definition embedding (0.5) with averaged example embeddings (0.5) consistently outperforms single-text approaches across all models.

2. **E5-large is best**: The `multilingual-e5-large` model achieves the highest accuracy (54.8%), with proper query/passage prefix handling.

3. **LLM examples help**: LLM-generated example sentences improve over baseline GermaNet text (+18.2 percentage points for E5-large).

4. **High Top-k scores**: Top-3 accuracy of 96.8% and Top-5 of 100% indicate the correct sense is almost always in the top candidates—the challenge is ranking it first.

5. **Individual examples vary**: Single example sentences (llm_example_0 through llm_example_4) show high variance (34-47% for E5-large), supporting the value of averaging multiple examples.

### 8.3 Diachronic Analysis Results

#### Pilot Study Setup

**Dataset:** 116 strategically selected words from 750-word target list
- 36 high-polysemy words (8+ senses) - most likely to show temporal change
- 65 medium-polysemy words (4-7 senses) - good variation
- 15 low-polysemy controls (2-3 senses) - baseline comparison

**Corpus Data:** 30,337 DWDS contexts (1900-1999), v2 re-fetch with --max-tokens 60
- Time period: 10 decades (1900s-1990s)
- Average: ~337 contexts per word (90 words with sufficient data)
- Only 22/894 word-decades sparse after min-5-contexts filter
- 0.3% truncation rate

**WSD Method:** Unsupervised prediction using best atemporal model
- Model: `intfloat/multilingual-e5-large` (54.8% atemporal accuracy)
- Text representation: weighted (0.5 × definition + 0.5 × avg(5 examples))
- No manual annotation required for pilot phase

#### Analysis Pipeline

1. **Sense Prediction**: Apply E5-large WSD to all 30,337 historical contexts → 24,325 predictions
2. **Temporal Aggregation**: Group predictions by lemma × decade
3. **Change Metrics**: Compute entropy, divergence, emergence/decline events (min 5 contexts/decade filter)
4. **Visualization**: Generate trajectory plots, heatmaps, summary statistics

#### Research Questions

1. **Temporal stability**: Which words show the most sense distribution change over time?
2. **Historical patterns**: Are there periods of high instability (e.g., 1940s war period)?
3. **Sense dynamics**: Which senses emerge or decline across the century?
4. **Polysemy effects**: Does polysemy level predict change rate?
5. **WSD reliability**: How does unsupervised WSD perform on historical data?

#### Key Findings (v2 pipeline)

- **518 emergence events**, **521 decline events** detected across 90 words
- **Max JSD: 0.835** — JSD=1.0 artifacts eliminated after v2 re-fetch
- **Only 22/894 word-decades** sparse (previously 371 in v1)
- Top-changed words (pending re-rank from v2): dominated by high-polysemy verbs
- Manual validation not yet started — detecting real change vs. WSD noise is the next step

### 8.4 Error Analysis

Detailed error analysis based on predictions from the best model (E5-large + weighted def+examples).

#### Gold Rank Distribution

| Gold Rank | Count | Interpretation |
|-----------|-------|----------------|
| 1 (correct) | 51 | Model predicted correctly |
| 2 (near-miss) | 27 | Correct sense was second choice |
| 3 | 10 | Correct sense ranked third |
| 4+ | 5 | Correct sense ranked low |

**Key insight**: 64% of errors (27/42) are near-misses where the correct sense was the model's second choice. This suggests the model captures relevant distinctions but struggles with final ranking.

#### Performance by Polysemy Level

| Polysemy | Lemmas | Instances | Accuracy | Avg Senses |
|----------|--------|-----------|----------|------------|
| Low (2-3) | 4 | 37 | 62.2% | 2.5 |
| Medium (4-6) | 4 | 37 | 59.5% | 4.8 |
| High (7+) | 2 | 19 | 31.6% | 9.0 |

**Key insight**: Accuracy drops sharply for high-polysemy words. Low and medium polysemy perform similarly, but words with 7+ senses are substantially harder.

#### Hardest Lemmas

The two worst-performing lemmas show systematic confusion patterns:

**Lemma 1** (0% accuracy, 9 errors):
- All errors: Gold=Kommunikation, Predicted=Mensch
- Pattern: Communication-act senses consistently confused with human-agent senses

**Lemma 2** (0% accuracy, 6 errors):
- All errors: Gold=Kommunikation, Predicted=Geschehen
- Pattern: Communication senses confused with event/process senses

**Root cause**: These lemmas have senses spanning semantic classes (Kommunikation, Mensch, Geschehen) that share contextual vocabulary, making disambiguation difficult with surface-level embeddings alone.

#### Common Error Types

| Error Type | Description | Example | Frequency |
|------------|-------------|---------|-----------|
| **Cross-class confusion** | Senses from different semantic classes conflated | Kommunikation → Mensch | High |
| **Near-miss ranking** | Correct sense is rank 2 | 64% of errors | High |
| **High-polysemy failure** | Too many candidate senses | Words with 7+ senses | Medium |
| **Context poverty** | Insufficient disambiguating context | Short sentences | Low |

#### Implications for Improvement

1. **Semantic class features**: Incorporating explicit semantic class information could help distinguish Kommunikation from Mensch senses.

2. **Reranking**: A two-stage approach (retrieve top-k, then rerank) could convert near-misses to correct predictions.

3. **Polysemy-aware models**: High-polysemy words may benefit from hierarchical sense clustering or specialized handling.

4. **More examples**: The high variance across individual LLM examples (34-47% for single examples) suggests more diverse examples could help.

#### Error Analysis Tools

See `analysis/error_analysis.ipynb` for interactive exploration:
- Per-lemma accuracy breakdown with polysemy visualization
- Confusion matrices for worst-performing lemmas
- Near-miss analysis with example sentences
- Gold rank distribution plots

See `src/wsd/wsd_evaluator.py` (`ErrorAnalyzer` class) for programmatic analysis:
- Confusion matrix generation
- High-error word identification
- Sense pair confusion analysis

### 8.5 Evaluation Framework

#### Shared Modules (`src/wsd/`)

The evaluation package provides reusable components for embedding-based WSD:

| Module | Class | Purpose |
|--------|-------|---------|
| `embedder.py` | `Embedder` | Unified text embedding using sentence-transformers |
| `similarity_wsd.py` | `SimilarityWSD` | Cosine similarity-based WSD predictor |
| `wsd_evaluator.py` | `WSDEvaluator` | Evaluates predictions against gold standard |
| `wsd_evaluator.py` | `WSDMetrics` | Detailed metrics (precision, recall, F1, confusion matrix) |
| `wsd_evaluator.py` | `ErrorAnalyzer` | Error pattern analysis |

##### Embedder

Unified text embedder with L2 normalization for cosine similarity:

```python
from src.wsd import Embedder

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
from src.wsd import SimilarityWSD

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

#### Evaluation Scripts (`scripts/atemporal/`)

Five scripts for different evaluation purposes:

| Script | Purpose | Data Source | Output |
|--------|---------|-------------|--------|
| `compare_embedding_methods.py` | Benchmark models + text variants | gold annotations | Accuracy, Top-k, MRR |
| `tune_representation_weights.py` | Optimize α in def+examples weighting | gold annotations | Best α, accuracy curve |
| `generate_predictions.py` | Generate per-instance predictions | gold annotations | Predictions with error analysis |
| `real_dwds_wsd_evaluation.py` | Indicative evaluation on corpus | DWDS + GermaNet | MFS agreement, confidence |
| `evaluate_embeddings.py` | Quality analysis (not WSD accuracy) | GermaNet senses | Similarity stats, clustering |

##### compare_embedding_methods.py

**Purpose:** Benchmark different embedding models and text variants for WSD accuracy.

**Input:**
- `data/evaluation/quick_eval/sense_text_representations.jsonl` — sense definitions with LLM-generated text (JSONL format with `llm_examples` as list)
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
python scripts/atemporal/compare_embedding_methods.py                # Normal run
python scripts/atemporal/compare_embedding_methods.py --clear-cache  # Clear disk cache after each model
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

**Benchmark Results:**

| Rank | Model | Text Variant | Accuracy | MRR |
|------|-------|--------------|----------|-----|
| 1 | multilingual-e5-large | Weighted (0.5×def + 0.5×avg examples) | **54.8%** | 0.725 |
| 2 | bge-m3 | Weighted (0.5×def + 0.5×avg examples) | 51.6% | 0.717 |
| 3 | mpnet-base-v2 | Weighted (0.5×def + 0.5×avg examples) | 47.3% | 0.677 |
| 4 | LaBSE | Weighted (0.5×def + 0.5×avg examples) | 46.2% | 0.680 |

**Key findings:**
- Weighted combination (α=0.5, validated via grid search) consistently best across all models
- LLM-generated examples outperform baseline GermaNet text by +18.2 pp
- E5 models require proper query/passage prefixes for best performance
- Top-3 accuracy: 96.8%, Top-5: 100% — correct sense almost always in top candidates

##### real_dwds_wsd_evaluation.py

**Purpose:** Evaluate WSD on real corpus sentences (indicative, no gold labels).

**Input:**
- DWDS corpus sentences (`data/dwds_contexts/`)
- GermaNet sense inventory (`data/processed/germanet/target_word_senses.csv`)

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
python scripts/atemporal/real_dwds_wsd_evaluation.py
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
python scripts/atemporal/evaluate_embeddings.py
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
├── pilot_predictions_weighted_v2.csv             # Step 1: predictions
├── pilot_weighted_v2_decade_sense_counts.csv     # Step 2: aggregated counts
├── pilot_weighted_v2_decade_stats.csv            # Step 2: per-decade stats
├── pilot_weighted_v2_filtered_decade_*.csv       # Step 3: change metrics (filtered)
└── plots/
    ├── trajectory_*.png
    ├── heatmap_*.png
    └── entropy_*.png
```

---

## 9. Project Architecture

### 9.1 Code Organization Principles

This project follows best practices for research software engineering by separating **reusable library code** (`src/`) from **executable entry points** (`scripts/`).

#### Design Philosophy

**`src/` = Library Code (Reusable Components)**
- Classes and functions that are **imported** by scripts, notebooks, and tests
- Core algorithms and business logic
- API clients and data processing utilities
- Pure functions with well-defined interfaces
- No CLI/argparse code

**`scripts/` = Executable Entry Points**
- Programs you **run** from the command line
- Data processing pipelines and experiment runners
- Contains `argparse`, `if __name__ == "__main__"`, logging setup
- Glue code that chains `src/` components together
- One-off analyses and utilities

**Golden Rule:** If you'd want to `import` it elsewhere, put it in `src/`.

### 9.2 Source Code Structure (`src/`)

```
src/
├── annotation/              # Annotation task management
│   ├── __init__.py
│   └── task.py             # AnnotationTask class
├── data/                    # Data processing modules
│   ├── __init__.py
│   ├── dwds/               # DWDS corpus integration
│   │   ├── __init__.py
│   │   ├── client.py       # DWDSClient API wrapper
│   │   ├── fetcher.py      # Context fetching & filtering
│   │   └── util/
│   │       └── context_utils.py # Text cleaning, deduplication
│   ├── germanet/           # GermaNet processing
│   │   ├── __init__.py
│   │   ├── make_dataset.py # GermaNetProcessor class
│   │   ├── parsing/        # XML parsing
│   │   ├── transforms/     # Data transformations
│   │   └── io/             # Data I/O
│   └── llm/                # LLM integration
│       ├── __init__.py
│       ├── client.py       # LLM API client
│       └── text_generator.py # Sense text generation
├── diachronic/             # Temporal analysis library
│   ├── __init__.py
│   ├── io.py              # Data loading utilities
│   ├── metrics.py         # Entropy, divergence, change detection
│   ├── aggregation.py     # Temporal aggregation functions
│   ├── predictor.py       # Sense prediction for temporal data
│   └── visualization.py   # Plotting functions
└── wsd/                    # WSD engine
    ├── __init__.py
    ├── embedder.py            # Embedding wrapper
    ├── embedding_strategies.py # Sense text representations
    ├── similarity_wsd.py      # WSD algorithm
    └── wsd_evaluator.py       # Evaluator, metrics, error analysis
```

#### Module Descriptions

**`src/data/dwds/`**
- `DWDSClient`: API client for fetching corpus data from the DWDS API
- `FetchOptions`: Configuration for context fetching (filtering, sampling)
- `fetch_contexts_for_word()`: Main fetching function with filtering pipeline
- Context utilities: cleaning, deduplication, temporal parsing

**`src/data/germanet/`**
- `GermaNetProcessor`: Complete pipeline for processing GermaNet XML
- Parsing: Extract synsets, lexical units, relations
- Transforms: Build sense texts, embeddings, candidate mappings
- I/O: Save/load processed data

**`src/diachronic/`**
- `load_*()`: Data loading functions for predictions, contexts, senses
- `compute_entropy_per_period()`: Sense distribution entropy metrics
- `compute_period_divergence()`: Jensen-Shannon divergence between periods
- `detect_sense_changes()`: Emergence and decline detection
- `aggregate_by_period()`: Group predictions by time
- `predict_senses()`: Apply WSD to temporal data
- Visualization functions: trajectories, heatmaps, entropy plots

**`src/wsd/`**
- `Embedder`: Wrapper for sentence transformer models
- `SimilarityWSD`: Cosine similarity-based WSD
- `WSDEvaluator`: Evaluation pipeline with metrics
- Error analysis utilities

### 9.3 Scripts Structure

```
scripts/
├── annotation/              # Annotation tools
│   ├── app.py                          # Flask annotation interface
│   ├── select_eval_words.py            # Select evaluation words
│   ├── generate_annotation_tasks.py    # Bundle contexts + senses into tasks
│   └── export_annotation_tasks_json.py # Convert pickle tasks to JSON
├── data/                    # Data preparation scripts
│   ├── fetch_dwds_contexts.py          # DWDS context fetcher (CLI)
│   ├── extract_eval_contexts.py        # Extract evaluation contexts
│   ├── process_germanet.py             # GermaNet processor (CLI)
│   ├── enrich_senses_with_llm.py       # LLM sense enrichment
│   └── setup/                          # One-off initialization scripts
│       ├── calculate_germanet_senses.py
│       ├── sample_from_wsd_senses.py
│       ├── convert_to_jsonl.py
│       └── rebuild_complete_wsd_senses.py
├── diachronic/             # Temporal analysis runners
│   ├── predict_dwds_senses.py          # Apply WSD to historical data
│   ├── aggregate_temporal.py           # Aggregate by time period
│   ├── compute_change_metrics.py       # Compute temporal metrics
│   └── visualize_trajectories.py       # Generate plots
└── atemporal/              # WSD evaluation runners
    ├── compare_embedding_methods.py
    ├── tune_representation_weights.py
    ├── generate_predictions.py
    ├── evaluate_embeddings.py
    └── real_dwds_wsd_evaluation.py
```

#### Typical Usage Pattern

```python
# In a script:
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from diachronic import load_dwds_contexts, predict_senses  # Import from src
from evaluation import Embedder, SimilarityWSD

def main():
    # Use library functions
    contexts = load_dwds_contexts(args.file)
    results = predict_senses(contexts, model_name=args.model)
    save_results(results)
```

### 9.4 Notebooks Structure

```
notebooks/
├── WSD-Atemporal.ipynb     # Main atemporal WSD analysis
├── analysis/
│   └── preliminary/
│       ├── combination_analysis.ipynb  # Embedding method comparison
│       └── error_analysis.ipynb        # Detailed error analysis
└── exploration/
    └── germanet.ipynb      # GermaNet data exploration
```

Notebooks import directly from `src/`:
```python
from evaluation import Embedder, WSDEvaluator
from diachronic.visualization import plot_sense_trajectory
```

### 9.5 Diachronic Module API Reference

The `src/diachronic/` module provides a complete toolkit for temporal sense analysis.

#### Data Loading (`io.py`)

```python
from diachronic import load_dwds_contexts, load_senses, load_predictions

# Load DWDS contexts from JSONL
contexts_df = load_dwds_contexts(Path("data/dwds_contexts/contexts.jsonl"))
# Returns: DataFrame with columns [lemma, sentence, year, decade, genre, bibl, corpus]

# Load GermaNet senses
senses_df = load_senses(
    filepath=Path("data/processed/germanet/target_word_senses.csv"),
    text_column="text_for_embedding",
    target_lemmas=["Bank", "Bahn"]  # Optional filter
)
# Returns: DataFrame with sense information for specified lemmas

# Load WSD predictions
predictions_df = load_predictions(Path("data/diachronic/dwds_predictions.csv"))
# Returns: DataFrame with predicted sense IDs and confidence scores
```

#### Sense Prediction (`predictor.py`)

```python
from diachronic import predict_senses

# Apply WSD to historical contexts
results_df = predict_senses(
    contexts_df=contexts_df,           # DWDS contexts
    senses_df=senses_df,               # GermaNet senses
    text_column="text_for_embedding",  # Column to embed
    model_name="intfloat/multilingual-e5-large"
)
# Returns: contexts_df with added columns: predicted_sense_id, confidence, sense_class
```

#### Temporal Aggregation (`aggregation.py`)

```python
from diachronic import aggregate_by_period, compute_period_stats

# Aggregate sense distributions by decade
agg_df = aggregate_by_period(predictions_df, period_column="decade")
# Returns: DataFrame with columns [lemma, decade, predicted_sense_id, count, total, proportion]

# Compute summary statistics per period
stats_df = compute_period_stats(predictions_df, period_column="decade")
# Returns: DataFrame with [lemma, decade, n_contexts, n_senses, dominant_sense,
#                          dominant_proportion, mean_confidence]
```

#### Change Metrics (`metrics.py`)

```python
from diachronic import (
    compute_entropy_per_period,
    compute_period_divergence,
    detect_sense_changes
)

# Compute entropy (sense ambiguity) per period
entropy_df = compute_entropy_per_period(agg_df, period_column="decade")
# Returns: [lemma, decade, entropy, max_entropy, normalized_entropy, n_senses]

# Compute Jensen-Shannon divergence between adjacent periods
divergence_df = compute_period_divergence(agg_df, period_column="decade")
# Returns: [lemma, decade_from, decade_to, jsd, n_senses_from, n_senses_to]

# Detect sense emergence and decline
emergence_df, decline_df = detect_sense_changes(
    agg_df,
    period_column="decade",
    emergence_threshold=0.1,  # Min proportion to count as "emerged"
    decline_threshold=0.1      # Max proportion to count as "declined"
)
# Returns: Two DataFrames with emergence/decline events
```

#### Visualization (`visualization.py`)

```python
from diachronic.visualization import (
    plot_sense_trajectory,
    plot_entropy_over_time,
    plot_sense_heatmap,
    plot_all_lemmas_summary
)

# Plot sense proportions over time (stacked area chart)
plot_sense_trajectory(
    counts_df=agg_df,
    lemma="Bank",
    period_column="decade",
    output_dir=Path("data/diachronic/plots")
)

# Plot entropy trends for multiple lemmas
plot_entropy_over_time(
    entropy_df=entropy_df,
    lemmas=["Bank", "Bahn", "Kraft"],
    period_column="decade",
    output_dir=Path("data/diachronic/plots")
)

# Plot heatmap of sense distributions
plot_sense_heatmap(
    counts_df=agg_df,
    lemma="Bank",
    period_column="decade",
    output_dir=Path("data/diachronic/plots")
)

# Plot summary statistics across all lemmas
plot_all_lemmas_summary(
    entropy_df=entropy_df,
    period_column="decade",
    output_dir=Path("data/diachronic/plots")
)
```

#### Complete Pipeline Example

```python
from pathlib import Path
from diachronic import *

# 1. Load data
contexts = load_dwds_contexts(Path("data/dwds_contexts/historical.jsonl"))
senses = load_senses(
    Path("data/processed/germanet/target_word_senses.csv"),
    text_column="text_for_embedding"
)

# 2. Predict senses
predictions = predict_senses(
    contexts, senses,
    text_column="text_for_embedding",
    model_name="intfloat/multilingual-e5-large"
)

# 3. Aggregate by decade
agg = aggregate_by_period(predictions, period_column="decade")
stats = compute_period_stats(predictions, period_column="decade")

# 4. Compute metrics
entropy = compute_entropy_per_period(agg)
divergence = compute_period_divergence(agg)
emergence, decline = detect_sense_changes(agg, emergence_threshold=0.1)

# 5. Visualize
output_dir = Path("data/diachronic/plots")
for lemma in agg["lemma"].unique():
    plot_sense_trajectory(agg, lemma, output_dir=output_dir)
    plot_sense_heatmap(agg, lemma, output_dir=output_dir)

plot_entropy_over_time(entropy, output_dir=output_dir)
plot_all_lemmas_summary(entropy, output_dir=output_dir)
```

---

## 10. Data Formats & Schema

### 10.1 Key Files

#### Sense Inventory

**File**: `data/processed/germanet/target_word_senses.csv`

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

### 10.2 Directory Structure

```
data/
├── raw/
│   └── target_words/               # Source word lists
├── external/
│   └── germanet_v190/              # Original GermaNet XML
├── processed/
│   ├── germanet/
│   │   └── target_word_senses.csv  # Main sense inventory with enriched text
│   ├── germanet_base/
│   │   ├── synsets.parquet         # Parsed synsets
│   │   ├── lexical_units.parquet   # Parsed lexical units
│   │   ├── relations.parquet       # Semantic relations
│   │   └── lemma_senses.parquet    # Lemma-sense mappings
│   ├── target_words/
│   │   └── target_words_sense_counts.tsv  # GermaNet sense counts per word
│   └── dwds/                       # Processed DWDS data
├── dwds_contexts/
│   └── fetch_progress_*.jsonl      # DWDS corpus sentences
├── diachronic/                     # Temporal analysis outputs
│   ├── dwds_predictions.csv
│   ├── temporal_decade_*.csv
│   ├── metrics_decade_*.csv
│   └── plots/
└── evaluation/
    ├── preliminary/                # Preliminary evaluation results
    │   └── input/
    │       └── sense_text_representations.jsonl
    ├── quick_eval/                 # Quick evaluation sets
    └── results/                    # Evaluation outputs
```

### 10.3 Reproducibility

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
python scripts/data/process_germanet.py

# 2. Fetch DWDS contexts
python scripts/data/fetch_dwds_contexts.py

# 3. Build embeddings
python scripts/data/create_embeddings.py

# 4. Run WSD evaluation
python scripts/atemporal/compare_embedding_methods.py
```

---

## 11. Limitations & Future Work

### 11.1 Current Limitations

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

### 11.2 Future Work

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

## 12. References

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

*Last updated: 2026-01-17*
*Version: 1.2*