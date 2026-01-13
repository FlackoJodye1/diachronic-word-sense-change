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
- `src/data/llm/sense_generator.py` - Sense text generator
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

*Last updated: 2026-01-13*