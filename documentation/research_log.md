# Research Log

A running log of decisions, experiments, and open questions for the German WSD project.

---

## Log Format

Each entry follows this structure:
```
### YYYY-MM-DD: Topic
**Status**: Decision Made / In Progress / Open Question
**Context**: Why this came up
**Options Considered**: What alternatives exist
**Decision**: What we chose (if decided)
**Rationale**: Why
**Follow-up**: Next steps or related questions
```

---

## 2025-01-09: Evaluation Set Design for Embedding Method Comparison

**Status**: Decision Made

**Context**:
We want to compare different embedding methods (different `text_to_embedding` approaches) to determine which produces the best sense representations for WSD. Before running a full evaluation, we need a smaller, well-designed eval set for quick iteration.

**Key Questions**:

### Q1: What words to pick?

**Options Considered**:
1. Random sample from 752 target words
2. Stratified sample by polysemy level
3. Stratified by polysemy + POS
4. Use existing DWDS-fetched words (25 words already have contexts)

**Current Thinking**:
Stratified sample by polysemy level AND POS is best because:
- Different polysemy levels test different aspects (low = baseline, high = stress test)
- Different POS have different sense structures (verbs are harder)
- Reveals where methods differ, not just overall accuracy

**Recommended Stratification**:

| Category | Sense Count | Purpose | Example Words |
|----------|-------------|---------|---------------|
| Low polysemy | 2-3 | Baseline - all methods should succeed | Dose, Keller, leer |
| Medium polysemy | 4-6 | Discriminative - reveals method differences | Stoff, Rolle, sprechen |
| High polysemy | 7+ | Stress test - exposes weaknesses | sitzen (7), halten (26), gehen (19) |

**POS Balance Target**:
- ~60% nouns (most common, moderate difficulty)
- ~30% verbs (harder, complex sense structures)
- ~10% adjectives (often gradable/context-dependent)

---

### Q2: How many words/instances?

**Options Considered**:

| Size | Words | Instances/Word | Total | Trade-off |
|------|-------|----------------|-------|-----------|
| Minimal | 5 | 5 | 25 | Too small for reliable metrics |
| Quick | 10 | 5-10 | 50-100 | Fast iteration, moderate reliability |
| Standard | 15 | 10 | 150 | Good per-word metrics |
| Full | 25+ | 15+ | 400+ | Publication quality, slow annotation |

**Current Thinking**:
**10-15 words, 5-10 instances each (50-150 total)** for the quick eval set.

**Rationale**:
- Enough words to cover polysemy/POS strata
- Enough instances per word for meaningful per-word accuracy (~±15% CI with 10 instances)
- Small enough to annotate quickly (1-2 hours manual work)
- Can expand later once best methods are identified

---

### Q3: Should we merge senses before evaluation?

**Options Considered**:

1. **No merging (fine-grained)**: Use all GermaNet synset distinctions as-is
2. **Merge to coarse-grained**: Group similar senses (e.g., by semantic class)
3. **Annotate fine-grained, analyze both ways**: Keep flexibility

**Current Thinking**:
**Option 3 - Annotate fine-grained, create merge mapping separately**

**Rationale**:
- Fine-grained annotation preserves maximum information
- Can always aggregate to coarse-grained at analysis time
- Reveals whether embedding methods struggle with fine distinctions vs. coarse categories
- Avoids re-annotation if we change merging strategy
- More comparable to GermaNet's actual structure

**Implementation Plan**:
1. Annotate using GermaNet synset IDs (fine-grained)
2. Create `sense_merge_mapping.csv`: `fine_sense_id → coarse_sense_id`
3. Evaluate both: fine-grained accuracy AND coarse-grained accuracy
4. Report difference (gap indicates fine-grained distinction difficulty)

**Merge Candidates**:
- Senses within same semantic class
- Senses with shared hypernym
- Senses distinguished only by register/formality

---

### Final Decisions Made

**Size**: 10 words × 10 instances = 100 total

**Strategy**: Stratified by polysemy + POS (fresh selection, not existing pilot data)

**Sense granularity**: Fine-grained annotation, merge mapping created separately

**Annotation**: Self-annotation only (quick iteration, not publication)

### Final Word Selection

| Word | POS | Senses | Category | Rationale |
|------|-----|--------|----------|-----------|
| Doktor | NOUN | 3 | Low | Academic title vs medical doctor |
| Antwort | NOUN | 2 | Low | Simple baseline case |
| wagen | VERB | 3 | Low | Low-polysemy verb baseline |
| Glas | NOUN | 4 | Medium | Material vs container vs eyeglasses |
| Schatz | NOUN | 4 | Medium | Treasure vs term of endearment |
| springen | VERB | 6 | Medium | Physical with metaphorical extensions |
| bitter | ADJ | 4 | Medium | Taste vs emotional |
| Bahn | NOUN | 7 | High | Railway/path/lane/orbit |
| sitzen | VERB | 7 | High | Physical/location/prison/mastery |
| hören | VERB | 8 | High | Perception/obey/learn |

**Note on word selection**: Original high-polysemy candidates fassen (12 senses) and werfen (9 senses) were replaced with sitzen (7) and hören (8). Rationale: With only 10 instances per word, extreme polysemy makes evaluation unreliable (most senses won't appear in sample).

**File**: `data/evaluation/quick_eval/words.csv`

### Context Fetching

**Initial attempt** (rejected):
- Strategy: Take first 10 DWDS results per word
- Problem: DWDS returns results sorted by date (newest first)
- Result: All 100 contexts from 1999, ~50% from single novel
- No temporal or source diversity

**Final approach**:
- Strategy: Random sampling with unique source filtering
- Implementation: Use DWDS `sort=random`, fetch 200 per word, deduplicate by source
- Result: 95 contexts with excellent diversity (after noise cleanup)

**Diversity achieved:**
| Metric | Value |
|--------|-------|
| Unique sources | 95 |
| Unique dates | ~73 |
| Date range | 1901-1999 |
| Genres | 4 (Belletristik, Gebrauchsliteratur, Zeitung, Wissenschaft) |

**Contexts per word**: Antwort (8), Glas (9), hören (9), sitzen (9), others (10)

**Removed contexts** (5 total):
- hören: Truncated sentence
- Antwort: Meta-label (`ANTWORT:`), reference style (`Antwort auf Frage 22`)
- Glas: Recipe list (`½ Glas Weinbrand...`)
- sitzen: Typo (`sietzen`)

**Known limitations**:
- Doktor: Mostly "Dr." title usage (limited sense diversity)
- Glas: Mostly "drinking vessel" sense
- Accepted for quick eval - reflects natural distribution

**File**: `data/evaluation/quick_eval/contexts.csv`

### Annotation Completed

**Date**: 2025-01-10

Successfully annotated 93 instances using the Flask annotation interface (`scripts/annotation/simple_app.py`).

**Output file**: `data/evaluation/quick_eval/gold_annotations.csv`

**Sense distribution per word**:
| Word | Instances | Senses Used | Distribution |
|------|-----------|-------------|--------------|
| Doktor | 10 | 1 | s32242 (10) - all academic title |
| Antwort | 7 | 2 | s30167 (6), s21188 (1) |
| wagen | 10 | 2 | s53262 (7), s54744 (3) |
| Glas | 9 | 2 | s6663 (7), s12229 (2) |
| Schatz | 10 | 3 | s109745 (5), s14559 (3), s109744 (2) |
| springen | 10 | 4 | s57843 (5), s57328 (3), s106710 (1), s58775 (1) |
| bitter | 10 | 3 | s530 (5), s108548 (3), s3166 (2) |
| Bahn | 10 | 4 | s107020 (5), s107675 (2), s9279 (2), s24471 (1) |
| sitzen | 9 | 2 | s57425 (8), s51903 (1) |
| hören | 8 | 4 | s58547 (4), s54965 (3), s54902 (1) |

**Observations**:
- Doktor: As expected, all contexts are "Dr." title usage (low sense diversity in corpus)
- Glas: Mostly drinking glass, some jars (missing material/eyeglasses senses)
- Good sense diversity in: Schatz, springen, Bahn, hören

### Quick Eval File Locations

| File | Description |
|------|-------------|
| `data/evaluation/quick_eval/words.csv` | Selected 10 words with metadata |
| `data/evaluation/quick_eval/contexts.csv` | 95 DWDS contexts |
| `data/evaluation/quick_eval/gold_annotations.csv` | 93 gold standard annotations |
| `data/evaluation/annotation_tasks.pkl` | Annotation task state (for Flask app) |
| `data/evaluation/annotation_tasks.json` | JSON backup of tasks |

**contexts.csv schema:**

| Column | Description |
|--------|-------------|
| `No.` | Row number from DWDS |
| `Date` | Date of source text |
| `Genre` | Text genre (Belletristik, etc.) |
| `Bibl` | Bibliographic reference |
| `Hit` | The sentence context |
| `lemma` | Target word |
| `pos` | Part of speech |

### Next Steps

1. [x] Select final 10 words for quick eval set
2. [x] Fetch DWDS contexts for selected words (95 contexts with source diversity)
3. [x] Annotate instances (93 gold annotations)
4. [ ] Run embedding comparison experiments
5. [ ] Document results in this log

---

## 2026-01-10: Repo Cleanup & Embedding Experiment Design

### Repo Cleanup

Reorganized repository structure:

**Deleted (redundant/broken):**
- `notebooks/data/Build_Embeddings.ipynb` - duplicate of scripts
- `notebooks/data/Map_Candidates.ipynb` - duplicate of scripts
- `notebooks/data/Map_Sense_Texts.ipynb` - duplicate of scripts
- `notebooks/exploration/dwds.ipynb` - simple API test
- `scripts/data_preparation/calculate_germanet_senses_750.py` - broken import
- `scripts/refetch_eval_contexts.py` - one-time utility
- `analysis/analyze_current_dwds_progress.py` - one-time progress check
- `documentation/eval_set_planning.md` - merged into this file

**Reorganized scripts/:**
- Moved annotation-related scripts → `scripts/annotation/`
- Moved data prep scripts → `scripts/data_preparation/`
- Kept only evaluation runners in `scripts/evaluation/`

### Embedding Text Variants Experiment

**Status**: In Design

**Goal**: Compare different text-to-embedding strategies for sense representations to determine which produces the best WSD accuracy.

**Current baseline approach**:
Template-based using GermaNet relations:
```
{lemma} ist eine Art von {hypernym}. {lemma} ist ein Begriff aus {class}
```
Example: "Bank ist eine Art von Sitzgelegenheit. Bank ist ein Begriff aus Artefakt"

**Proposed approaches to compare**:

| # | Method | Description | Text for Embedding |
|---|--------|-------------|-------------------|
| 0 | Baseline (GermaNet) | Template with hypernym + class | `{lemma} ist eine Art von {hypernym}...` |
| 1 | LLM Definition | LLM generates natural definition | "Eine Bank ist ein längliches Sitzmöbel..." |
| 2 | LLM 1 Example | LLM generates example sentence | "Die alte Frau setzte sich auf die Bank im Park." |
| 3 | LLM 5 Examples | Average embeddings of 5 examples | 5 varied usage sentences |

**Key design question**: How to specify the target sense when prompting the LLM?

For polysemous words like "Bank" (bench vs. financial institution vs. sandbank), the LLM needs context to know which sense to generate for. Options:
- Provide GermaNet hypernym (e.g., "Sitzgelegenheit")
- Provide semantic class (e.g., "Artefakt")
- Provide synonyms from the synset

This creates some circularity - LLM approaches still depend on GermaNet info for sense specification.

**Evaluation setup**:
- 10 eval words, 48 senses total
- 93 gold-annotated test instances
- Stratified by polysemy level (low/medium/high) and POS
- Metric: WSD accuracy (predicted sense matches gold sense)

**Evaluation process**:
1. Embed each sense representation (per method)
2. Embed each test sentence (context)
3. Predict sense = highest cosine similarity to context
4. Compare prediction to gold → compute accuracy per method

**Next steps**:
1. [x] Create LLM prompts for sense-specific generation
2. [x] Generate text variants for 48 senses (10 eval words)
3. [ ] Run WSD evaluation for each variant
4. [ ] Compare results and analyze

---

## 2026-01-13: LLM Text Generation Implementation

**Status**: Completed

### Implementation

Created LLM text generation module for sense representations:

**New files:**
- `src/data/llm/client.py` - LLM client wrapper (OpenAI/Claude)
- `src/data/llm/text_generator.py` - Sense text generator
- `config/prompts/definition.txt` - Definition prompt template
- `config/prompts/example_single.txt` - Single example prompt
- `config/prompts/examples_multiple.txt` - Multiple examples prompt
- `scripts/data_preparation/generate_llm_sense_texts.py` - Runner script

**Prompt design:**
Each prompt receives:
- `lemma` - The word (e.g., "Bank")
- `pos_german` - Part of speech in German (e.g., "Nomen")
- `sense_description` - Full GermaNet baseline text (e.g., "Bank ist eine Art von Sitzgelegenheit...")

The prompts instruct the LLM to generate text that clearly distinguishes this sense from other senses of the word.

### Generation Results

**Run details:**
- Model: `gpt-4o-mini`
- Temperature: 0.7
- Cost: ~$0.02
- Time: ~3 minutes

**Output:** `data/evaluation/quick_eval/sense_variants.csv`

| Column | Description |
|--------|-------------|
| `lemma` | Word |
| `pos` | Part of speech |
| `synset_id` | GermaNet sense ID |
| `class` | Semantic class |
| `text_baseline` | Original GermaNet template |
| `text_llm_definition` | LLM-generated definition |
| `text_llm_example_1` | LLM-generated single example |
| `text_llm_examples_5` | LLM-generated 5 examples (`;`-separated) |

**Stats:** 48 senses × 3 variants = 144 LLM-generated texts

### Next Steps

- [ ] Run WSD evaluation comparing all 4 text representations
- [ ] Analyze which representation works best per polysemy level

---

## 2026-01-30: Weight Tuning for Weighted Representation

**Status**: Completed

**Context**:
The weighted representation (α×definition + (1-α)×avg(examples)) achieved 54.8% accuracy with α=0.5. We wanted to verify whether 0.5 is optimal or if other weights perform better.

**Experiment Design**:
- **Model**: intfloat/multilingual-e5-large (best performer)
- **Search space**: α ∈ [0.0, 0.1, 0.2, ..., 1.0] (11 values)
- **Evaluation**: 93 gold-annotated instances from atemporal eval set
- **Metric**: WSD accuracy (top-1)

**Results**:

| α | Interpretation | Accuracy | vs. Baseline |
|---|---------------|----------|--------------|
| 0.0 | Pure examples | 51.6% | -3.2% |
| 0.1-0.3 | Heavy examples | 50.5% | -4.3% |
| 0.4 | Examples-leaning | 52.7% | -2.1% |
| **0.5** | **Equal weight** | **54.8%** | **baseline** ✓ |
| 0.6 | Definition-leaning | 51.6% | -3.2% |
| 0.7-0.8 | Heavy definition | 47.3-50.5% | -4.3 to -7.5% |
| 1.0 | Pure definition | 41.9% | -12.9% |

**Key Findings**:

1. **α=0.5 is optimal**: Performance peaks exactly at equal weighting
2. **Symmetric degradation**: Accuracy drops similarly when moving away from 0.5 in either direction
3. **Synergy effect**: Combined (54.8%) > best single source (51.6% examples)
4. **Definitions need examples**: Pure definition performs worst (41.9%)

**Decision**:
Keep α=0.5 for production pipeline. The 50/50 split is:
- Empirically optimal on eval set
- Simple and interpretable
- Conceptually sound (balancing abstract definition with concrete usage)

**Implementation**:
Weight is hardcoded in `src/data/embedding_strategies.py:144`

**Results file**: `data/evaluation/atemporal/results/weight_tuning.csv`

**Script**: `scripts/atemporal/tune_representation_weights.py`

---

## Open Questions

### OQ1: Which embedding models to compare?

**Candidates**:
- `paraphrase-multilingual-MiniLM-L12-v2` (current)
- `deepset/gbert-base` (German BERT)
- `xlm-roberta-base` (multilingual)
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (larger)

**Decision needed**: Which models are worth testing given compute/time constraints?

---

### OQ2: What text variations to test?

**Status**: Superseded by 2026-01-10 entry above.

New plan: Compare GermaNet template baseline vs. LLM-generated variants (definition, 1 example, 5 examples).

---

### OQ3: Context representation for disambiguation

How to represent the sentence context being disambiguated?

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| Full sentence | Embed entire sentence | Simple, captures all context | May include noise |
| Window ±k | k words around target | Focused | May miss relevant context |
| Target-masked | Replace target with [MASK] | Forces context reliance | Loses target info |

**Decision needed**: Start with full sentence (simplest) or test multiple?

---

## Experiment Ideas (Backlog)

- [ ] Compare embedding methods on quick eval set
- [ ] Analyze which sense pairs are most confused
- [ ] Test whether enrichment helps more for sparse vs. rich senses
- [ ] Correlation between sense frequency and accuracy
- [ ] Per-decade accuracy (does model favor modern senses?)

---

## References & Resources

- Main methodology: `documentation/documentation.md`
- Sense inventory: `data/processed/germanet/wsd_senses.csv`
- DWDS contexts: `data/dwds_contexts/`
- Evaluation code: `src/evaluation/wsd_evaluator.py`
- Embedding code: `src/data/germanet/transforms/embedding_builder.py`

---

*Last updated: 2026-02-18*

---

## 2026-02-18: Diachronic Corpus Quality Assessment

**Status**: Decision Made

**Context**:
Before proceeding with manual validation of the diachronic pilot results, we did a basic quality check of the fetched DWDS sentences (`data/dwds_contexts/pilot_diachronic_contexts.jsonl`) to assess whether they are suitable input for the WSD pipeline.

**Findings**:

| Metric | Value |
|--------|-------|
| Total sentences | 6,474 |
| Average length | 20.7 words |
| Median length | 20.0 words |
| Short sentences (< 10 words) | 383 (5.9%) |
| Truncated sentences (ends with ...) | 32 (0.5%) |
| Sentences with < 6 words | 0 |

**Verdict**: The corpus is **usable**. Sentence length is adequate for WSD, virtually nothing is truncated or trivially short.

**Known Issues**:

1. **Fixed expressions (~20% estimate)**: Many sentences contain idiomatic / frozen multi-word expressions where the target word's sense is effectively bleached. Examples for *tragen*: *Sorge tragen* ("to take care"), *Rechnung tragen* ("to take into account"), *zu Grabe tragen* ("to bury").

   **Why this is a problem for WSD**: In a fixed expression, the target word no longer carries an independently assignable sense — the meaning is compositionally opaque. The embedding of the sentence will reflect the idiomatic meaning of the whole phrase, not the target word in isolation. Our model will nonetheless force an assignment to the nearest GermaNet sense, which is essentially arbitrary for these cases.

   **Why this is a problem for diachronic analysis**: If the frequency of fixed expressions containing a word shifts across decades — e.g., because *Rechnung tragen* becomes more common in bureaucratic/scientific writing after the 1950s — the model will systematically push predictions toward whichever GermaNet sense happens to be closest to that idiom. This looks like a genuine sense shift in the metrics (rising JSD, sense emergence/decline) but is actually a genre or register artifact, not a change in word meaning. This is particularly hard to detect because the model's confidence will remain high — the embedding is clear and consistent, just not meaningful for our purposes.

2. **Sparse decade coverage**: Average of ~5.7 sentences per word-decade. Several word-decades have fewer than 5 sentences, making decade-level sense distributions unreliable. JSD=1.0 values in the metrics are often artifacts of this sparsity rather than genuine semantic change.

3. **Genre imbalance across decades**: The DWDS kern corpus is not evenly distributed by genre over time. Shifts in genre composition (e.g., more Zeitung in certain decades) can drive apparent sense shifts independently of actual language change.

**Decision: Run as-is, treat as known limitation.**

Filtering out fixed expressions was considered but rejected for the following reasons:
- Diachronic analysis is about *shifts* in sense distributions, not absolute frequencies. If idioms are roughly evenly distributed across decades for a given word, they add a constant bias but do not create fake shifts.
- Data is already sparse (~5.7 sentences per word-decade on average). Removing idioms would make this worse.
- This is a pilot study — the goal is to validate the pipeline and identify patterns, not to produce publication-ready clean results.
- Handling idioms and fixed expressions is a known open problem in WSD research; ignoring them is standard practice in comparable studies.

The one real risk is if genre composition in DWDS shifts across decades in a way that correlates with idiom frequency (e.g., more bureaucratic/scientific writing in later decades increasing *Rechnung/Sorge tragen*). This is hard to control for and is noted as a limitation.

**Follow-up**:
- Document as a limitation in any write-up.
- When manually validating top words, note if a suspicious sense shift is driven by idiom-heavy sentences.
- Future work: consider flagging known fixed expressions as a pre-processing step.

---

## 2026-02-19: Re-fetch with Relaxed Sentence Length Cap

**Status**: Done

**Context**:
The original fetch used `--max-tokens 35`, which was artificially limiting yield and truncating many DWDS sentences. Re-fetched all 116 pilot words with `--max-tokens 60`, `--limit 1000`, `--target-n 0`, `--max-per-doc 1`.

**Results**:
- Total sentences: **30,337** (up from 6,474 — ~5x increase)
- Output: `data/dwds_contexts/pilot_diachronic_contexts_v2.jsonl`

**Truncation check**: Only 0.3% of sentences hit the 60-token cap → cap is not limiting. Keep at 60.

---

## 2026-02-19: DWDS Kern Corpus — Temporal and Genre Imbalance

**Status**: Documented (known limitation, does not invalidate study)

**Context**:
Post-fetch analysis of genre and decade distribution in `pilot_diachronic_contexts_v2.jsonl`.

**Findings**:

*Sentences per decade:*

| Decade | Sentences |
|--------|-----------|
| 1900s | 3,143 |
| 1910s | 3,879 |
| 1920s | 4,393 |
| 1930s | 3,998 |
| 1940s | 2,912 |
| 1950s | 2,797 |
| 1960s | 2,347 |
| 1970s | 1,956 |
| 1980s | 2,271 |
| 1990s | 2,641 |

*Genre composition (%) per decade:*

| Decade | Belletristik | Gebrauchsliteratur | Wissenschaft | Zeitung |
|--------|-------------|-------------------|-------------|---------|
| 1900s | 29 | 21 | 17 | 32 |
| 1920s | 25 | 24 | 16 | 35 |
| 1950s | 11 | 34 | 23 | 32 |
| 1970s | 13 | 25 | 24 | 38 |
| 1990s | 21 | 29 | 24 | 27 |

**Interpretation**:
The DWDS kern corpus is officially designed to be balanced across decades (1900–1999) and across four genres (Belletristik, Zeitung, Wissenschaft, Gebrauchsliteratur). However, in practice the distribution in our fetched data is not perfectly uniform:

- Later decades (e.g. 1970s: 1,956 sentences) contain significantly fewer sentences than earlier ones (e.g. 1920s: 4,393). This is the opposite of what a recency-biased corpus would show, and likely reflects licensing and digitization constraints on older texts being more thoroughly digitized in this corpus.
- Genre composition shifts over time: Belletristik declines from ~30% (1900–1920s) to ~12% (1960–1980s), offset by rising Wissenschaft (17% → 27%). Zeitung is stable at 27–42%.

This temporal and genre imbalance is a **structural property of the corpus**, not a pipeline bug. It is a potential confound in the diachronic analysis — genre-specific vocabulary and sense usage may drive apparent sense shifts independently of genuine semantic change. It does not invalidate the study but should be reported as a limitation.

**Follow-up**: When interpreting diachronic results, treat shifts that coincide with strong genre composition changes with extra caution.

---

## 2026-02-19: Sparse-Data Filter Applied to Diachronic Metrics

**Status**: Decision Made

**Context**:
Many JSD=1.0 values in the diachronic metrics were suspected to be artifacts of word-decades with very few sentences (sometimes 1–2), rather than genuine semantic change.

**Decision**: Filter out word-decades with fewer than 5 contexts before computing JSD and other change metrics.

**Implementation**: Added `--min-contexts` flag to `scripts/diachronic/compute_change_metrics.py`. Filter applied to the `total` column in the sense counts CSV.

**Results**:

| Metric | Unfiltered | Filtered (n≥5) |
|--------|-----------|----------------|
| Word-decades included | 861 | 490 |
| Max JSD | 1.0 | 0.913 |
| Emergence events | 82 | 343 |
| Decline events | 84 | 318 |

The JSD=1.0 values are eliminated. Emergence/decline counts increased because proportional shifts in reliable decades are more meaningful.

**Output files**: `data/diachronic/pilot_weighted_filtered_decade_*.csv`

**Follow-up**: v1 metrics superseded by v2. Keep v1 files for reference only.

---

## 2026-02-19: v2 Predictions, Aggregation & Metrics

**Status**: Done

**Context**: After re-fetching with relaxed sentence cap (30,337 sentences vs 6,474), re-ran the full pipeline on the new data.

**Pipeline run:**
1. `predict_dwds_senses.py` → `pilot_predictions_weighted_v2.csv` (24,325 predictions, 90 words)
2. `aggregate_temporal.py` → `pilot_weighted_v2_decade_sense_counts.csv`, `pilot_weighted_v2_decade_stats.csv`
3. `compute_change_metrics.py --min-contexts 5` → `pilot_weighted_v2_filtered_decade_*.csv`

**Results vs v1:**

| Metric | v1 | v2 |
|--------|----|----|
| Total predictions | 5,155 | 24,325 |
| Word-decades filtered out (n<5) | 371/861 (43%) | 22/894 (2%) |
| Max JSD | 0.913 | 0.835 |
| Emergence events | 343 | 518 |
| Decline events | 318 | 521 |
| Mean confidence | 0.833 | ~0.831 |

The v2 data is substantially more reliable — nearly all word-decades now have sufficient data. The top 10 ranking needs to be re-computed from v2 metrics. Notebook updated to load v2 files.

**Known issue — 26 words skipped (no GermaNet senses):**
The following words are in `pilot_words_diachronic.txt` but absent from all GermaNet senses files (`target_word_senses.csv`, `target_word_senses_enhanced_*.csv`):

> Bauer, Becher, Bild, Fall, Farbe, Figur, Gang, Husten, Knoten, Paket, Seite, Stück, Wurzel, Zug, brennen, fahren, folgen, halten, heißen, lassen, legen, liegen, riechen, schließen, sehen, stehen

These words **are** present in the raw GermaNet lexical units (`lemma_senses.csv`) and in the 750-word target list. The gap was introduced during the GermaNet processing step (`process_germanet.py`) — likely a matching or filtering issue. Does not affect the current pilot (90 words are sufficient), but should be fixed before scaling up. Root cause investigation deferred to Phase 4.

---

## 2026-02-19: Temporal Resolution vs. Data Density Trade-off

**Status**: Decision Made

**Context**:
After applying the n≥5 filter, we found that the median is still only ~8.5 sentences per reliable word-decade, and the average word has ~5-6 GermaNet senses. To get reliable sense proportion estimates you need roughly 5 observations per sense, implying ~30 sentences per decade per word. The DWDS kern corpus simply cannot provide this at decade-level granularity — even the most data-rich words cap out at ~20 sentences in their best decade.

**Options Considered**:

1. **Re-fetch with higher `--per-decade-n`** — kern corpus ceiling is ~10-20/decade regardless of fetch limit. Not a real solution.
2. **Add more DWDS subcorpora** (Zeitung, Belletristik, Wissenschaft, Web) — more data, but mixing genres across decades introduces genre-composition confounds that can mimic semantic change. Right path for a full study, needs genre-balancing.
3. **Merge decades into larger time bins** — pools existing data, no re-fetching needed, methodologically clean since corpus composition stays consistent.

**Decision**: For the pilot study, merge to **25-year bins** (1900–1924, 1925–1949, 1950–1974, 1975–1999). This gives ~25-40 sentences per bin for most words — enough for reliable distributions — while retaining 4 data points per word (sufficient to track broad historical trends).

50-year bins were rejected as too coarse (only 2 data points = before/after comparison, not diachronic).

**Why not add subcorpora now**: Genre imbalance across decades is a harder confound than sparsity. If Belletristik dominates the early decades and Wissenschaft the later ones, distributional shifts in sense usage look like semantic change. This needs careful genre-stratified sampling and is a Phase 4 concern.

**Implementation**: Add a `--bin-size` option (or hardcode 25-year bins) to `aggregate_temporal.py`, re-run metrics on the binned data.

**Follow-up**:
- Implement 25-year binning in the aggregation pipeline
- Re-run metrics and update notebook to use binned results
- Adding subcorpora is the correct scaling path for the full study but requires genre-balancing