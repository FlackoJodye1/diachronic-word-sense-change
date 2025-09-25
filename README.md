Worde-Sense-Disambiguation
==============================

This project implements a BERT-based Word Sense Disambiguation (WSD) system for German and applies it to study diachronic sense change across decades using the DWDS corpora and GermaNet as the sense inventory


Motivation
------------

* Word Sense Disambiguation (WSD) is well studied in English, but German lacks large-scale benchmarks.

* Diachronic change (how meanings shift over time) has only been explored for English (e.g., Hu et al. 2019).

* GermaNet and DWDS together provide the resources needed to bring diachronic WSD to German.

Project Goals
------------
* Build a German WSD pipeline aligned to GermaNet.

* Extract context examples per decade from DWDS corpora.

* Model sense distributions across time and measure sense emergence/decline.

* Evaluate with:

  - Intrinsic WSD accuracy (manual annotation)

  - Entropy/diversity of sense distributions

  - Human judgments (mini study)

* Produce case studies (e.g., Bank, Schloss, Zelle).

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


Dataset Description
--------

## Germanet

### What we want

We need formats that are fast to query and easy to join with DWDS contexts!

### What the raw data is

GermaNet is a lexical–semantic network for German, similar to WordNet.

- Synset (a “sense” node): a set of near-synonymous lexical units representing one meaning.

  - Has an ID (e.g., s123456), POS (adj, nomen, verb), a semantic class (e.g., Allgemein), and sometimes a paraphrase/definition.

- Lexical Unit (LU): a lemma-form inside a synset.

  - Has word form (orthForm), optional variants (orthVar), plus flags like namedEntity, etc.

- Relations: edges between synsets (e.g., hypernym, meronym, antonym…).

  - Given in gn_relations.xml as pairs from_synset → to_synset with relation_type.

We can think of it as a graph: synsets are nodes; lexical units attach words to nodes; relations connect nodes.

### How we transform the raw data

The original GermaNet release comes as a collection of XML files, split by POS (e.g. Verb) plus a gn_relations.xml file. 
These files are rich but not convenient for downstream NLP tasks.

Our data processing script (src/data/Make_Dataset.py) performs the following transformations:

1. Parse XML files

- Extract each <synset> with its ID, POS category, semantic class, and optional paraphrase (gloss).

- Extract each <lexUnit> with lemma (orthForm), variants (orthVar), example sentences, and flags (namedEntity, artificial, styleMarking).

- Extract semantic relations between synsets from gn_relations.xml.

2. Normalize attributes

- Map GermaNet categories (adj, nomen, subst, verb) to canonical POS tags (ADJ, NOUN, VERB).

- Store variants and examples as lists for JSON/Parquet, and pipe-separated strings for CSV.

3. Write clean tables

- Save each entity type (synsets, lexical units, relations) into JSON, CSV, and Parquet.

- Build a lemma–senses view by joining lexical units with their parent synsets to make WSD lookups efficient.

### Target Tables

1. Synsets
- One row per sense node.
- Columns: synset_id, pos, class, paraphrase.

2. Lexical Units
- One row per lemma inside a synset.
- Columns: lexunit_id, synset_id, lemma, variants, sense, source, named_entity, artificial, style_marking, examples, pos, class.

3. Lemma–Senses (derived view)
- One row per candidate sense of a lemma, used directly for word sense disambiguation.
- Columns: lemma, pos, synset_id, class, paraphrase, variants, examples.

4. Relations
- Semantic edges between synsets.
- Columns: relation_type, from_synset, to_synset, direction, inverse.

These tables provide a clear separation between the inventory (synsets, lexical units, relations) and the lookup view (lemma_senses), which is what the WSD model will actually query.