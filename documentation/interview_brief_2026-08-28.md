# Project Audit — Precise Description of the WSD System

**Prepared:** 2026-08-28 · **Basis:** repository documentation, verified against source and result files.
**Method:** docs read first, then every load-bearing number recomputed from the artifacts on disk. Where the α grid could not be settled from files alone, the tuning experiment was re-run in memory (no repo files written) using the cached `intfloat/multilingual-e5-large` and the current sense file.

Everything below is either (a) quoted from a doc, (b) read off a file at a cited line, or (c) recomputed by me and labelled as such. Gaps are marked **UNDOCUMENTED** rather than filled.

---

## 0. Documentation inventory

### `documentation/` (12 files)

| File | Lines | What it is | Freshness |
|---|---|---|---|
| `documentation.md` | 1978 | Master methodology doc, 9 sections (research overview → architecture) | **Mixed.** §8.2 headline numbers current; §8.4 error analysis stale (see §7 below) |
| `research_log.md` | 1357 | Dated decision log, newest first (2026-05 → 2026-02) | Current; the most reliable narrative source |
| `roadmap.md` | 198 | Phase-by-phase status tracker | Header says "Last updated 2026-04-02"; contains a since-falsified α claim |
| `pipeline.md` | 289 | End-to-end runnable walkthrough | Current |
| `paper_plan.md` | 157 | Claims, section map, artifact map for the paper | Current (2026-05-07+); most accurate about the eval set |
| `coarse_clustering_plan.md` | 304 | Clustering design doc | **Explicitly marked "Planning (as of 2026-03-10)"** and never updated; superseded on two points (§7) |
| `data_catalog.md` | 198 | Artifact → producing-script → headline-number table | Current |
| `proposal.md` | 113 | Original project proposal | Historical |
| `llm_judge_clustering.md` | 208 | LLM-as-judge design | Marked PARKED (2026-05-07) |
| `survey_correlations_2026-05-15.md` | 77 | German survey × model correlations | Current |
| `survey_correlations_english_2026-05-15.md` | 94 | English replication vs. Hu et al. (2019) | Current |
| `baumann_replication_2026-05-15.md` | 127 | Baumann et al. (2023) replication + Hamilton bonus | Current |

### Other prose

| Location | Content |
|---|---|
| `README.md` (101 lines) | Status table, key numbers, best-config box, known limitations |
| `scripts/atemporal/README.md` (87 lines) | **The most candid document on the weighting question** — the 2026-04-07 note at line 79 |
| `scripts/clustering/ANNOTATION.md` (101) | Multi-annotator quickstart |
| `notebooks/analysis/*.ipynb` | 4 notebooks carry substantial markdown prose: `combination_analysis.ipynb` (model selection justification), `sense_granularity_analysis.ipynb` (fine vs. coarse), `error_analysis.ipynb`, `case_study_validation.ipynb`. **The first two contain superseded numbers** (§7) |
| `resources/wsd-project-report.pdf` | Prior project report (not read; PDF) |
| `config/prompts/*.txt` | The four LLM enrichment prompts — the operative spec for what the sense texts contain |

No thesis chapters, no `docs/` directory, no experiment-log directory beyond `research_log.md`.

---

## 1. Sense representation

**A sense is a GermaNet synset**, identified by `synset_id` (e.g. `s56941`), carrying four text fields generated per sense (`documentation/data_catalog.md:70-73`):

| Field | Content | Used for |
|---|---|---|
| `baseline` / `text_for_embedding` | GermaNet template text (hypernym chain) | the "text-only" ablation |
| `core_gloss` | 5–12 word denotative gloss | **clustering only** |
| `llm_definition` | verbose 1–2 sentence contrastive definition | **WSD** |
| `llm_examples` | list of 5 generated example sentences | **WSD** |

Generated with `gpt-4o-mini` by default (`src/data/llm/client.py:60`), temperature 0.7 (`scripts/data/enrich_senses_with_llm.py:265`), prompts in `config/prompts/`.

### Encoded separately, then combined as vectors — not concatenated

This is the direct answer to your question. In `src/wsd/embedding_strategies.py:131-155`:

- the definition is encoded as **one text** (line 137);
- each of the 5 examples is encoded as its **own text**, then the five vectors are **mean-pooled** (lines 143-152);
- the two resulting vectors are combined arithmetically (line 155).

There is no string concatenation on the WSD path. A concatenating variant exists (`llm_examples_concat`, lines 55-66) but it is not the deployed representation and appears in no reported result.

### Encoder and pooling

- **Model:** `intfloat/multilingual-e5-large` (`src/wsd/embedder.py:29`; set as the constant in every atemporal script, e.g. `scripts/atemporal/generate_predictions.py:37`).
- **Dim:** 1024.
- **Pooling:** **mean pooling over token embeddings**, followed by an L2-normalize module. Verified from the cached model config, not assumed: `~/.cache/huggingface/hub/models--intfloat--multilingual-e5-large/snapshots/*/1_Pooling/config.json` has `"pooling_mode_mean_tokens": true`, `"pooling_mode_cls_token": false`; `modules.json` lists `Transformer → Pooling → Normalize`. `max_seq_length` 512.
- **Prefixes:** E5 asymmetric prefixes are applied — `"passage: "` on sense texts, `"query: "` on the context sentence (`src/diachronic/predictor.py:119,133`; `scripts/atemporal/generate_predictions.py:74,171`). Applied per-text, i.e. each example gets its own `passage: `.
- **Normalization:** every individual encode is L2-normalized at source (`src/wsd/embedder.py:107`, `normalize_embeddings=self.normalize`, default `True`).

So the three pooling stages, in order: mean-pool tokens → L2-normalize (inside the encoder) → mean-pool the 5 example vectors → α-weighted sum with the definition vector → L2-normalize (at consumption).

**Context side:** the full sentence is embedded, with no target-word masking or windowing. `documentation/research_log.md:940-955` records this as open question OQ3 with "Decision needed" — the decision was never written down, but the code has only ever done full-sentence.

---

## 2. The weighting — location and exact formula

**Weighted combination of vectors, not of similarity scores.** There is exactly one similarity computation per (context, sense) pair, against the already-combined vector.

The formula exists in **four** independent implementations:

| File | Lines | α | Example columns | Re-normalizes after summing? |
|---|---|---|---|---|
| `src/wsd/embedding_strategies.py` | 131-155 | **hardcoded 0.5** at line 155 | `llm_example_1..5` (1-indexed) | No — but see below |
| `scripts/atemporal/generate_predictions.py` | 66-111 | constants `0.5`/`0.5`, lines 47-48 | `llm_example_0..4` (0-indexed) | Yes, lines 107-109 |
| `scripts/atemporal/tune_representation_weights.py` | 84-129 | parameterised `alpha` | 0-indexed, line 46 | Yes, lines 126-127 |
| `scripts/atemporal/compare_embedding_methods.py` | 195-210 | constants `0.5`/`0.5`, lines 66-67 | 0-indexed, line 64 | Yes, lines 208-210 |

The deployed diachronic path uses the **first** one (`src/diachronic/predictor.py:114-120`). As implemented there:

```
d_s      = E5("passage: " + llm_definition_s)                     # L2-normalized, 1024-d
e_{s,i}  = E5("passage: " + llm_example_{s,i})                    # L2-normalized, per example
ē_s      = (1/|X_s|) Σ_{i ∈ X_s} e_{s,i}    where X_s = non-empty examples (≤5)
v_s      = 0.5 · d_s + 0.5 · ē_s                                  # embedding_strategies.py:155
v̂_s      = v_s / ‖v_s‖                                            # similarity_wsd.py:75
q_c      = E5("query: " + sentence_c) / ‖·‖                       # similarity_wsd.py:113/186
ŝ(c)     = argmax_{s ∈ senses(lemma(c))}  ⟨q̂_c , v̂_s⟩             # similarity_wsd.py:116-126
```

Notes on faithfulness:

- `embedding_strategies.py` alone does **not** re-normalize (line 155 returns the raw sum). It is harmless because `SimilarityWSD.__init__` normalizes all sense vectors at construction (`src/wsd/similarity_wsd.py:75`). The two code paths are therefore numerically equivalent — I verified this by reproducing `predictions.csv` exactly (§3).
- A sense with zero usable examples gets `ē_s = 0` (`embedding_strategies.py:146`), so `v̂_s` collapses to the pure definition vector. Not documented anywhere.
- α is **not exposed as a parameter** on the production path. Changing it requires editing line 155. `RepresentationType` (line 13-21) has no α-carrying variant.
- The 0- vs 1-indexed example-column split between the two families is real but internally consistent in each (loaders at `src/diachronic/io.py:108-113` vs `scripts/atemporal/generate_predictions.py:54-57`). `coarse_clustering_plan.md:300` calls it out as a known trap.

---

## 3. Tuning of α

### What is recorded

**Grid:** the script default is `np.arange(0.0, 1.01, 0.1)` → 11 values, 0.0–1.0 step 0.1 (`tune_representation_weights.py:298`). A `--fine-grid` option exists for 0.30–0.70 step 0.05 (line 296); **no fine-grid result exists on disk.**
**Data:** the 93-instance gold set — `data/evaluation/atemporal/input/gold_annotations.csv`, 10 lemmas, single annotator (`annotator` column is `user` for all 93 rows). **This is the same set used to pick the model and to report the headline result. There is no train/dev/test split anywhere in the repo.**
**Metric:** fine-grained top-1 accuracy (`log_summary` maximises `accuracy`, line 238). Top-3/top-5/MRR are recorded but not optimised. Coarse accuracy is **not** computed by the tuning script.
**Result file:** `data/evaluation/atemporal/results/weight_tuning.csv`, mtime 2026-03-12 13:00.

Quoted verbatim (accuracy column rendered as percentages for readability; raw values in the file):

```
alpha,accuracy,top3,top5,mrr,evaluated,skipped
0.0,0.4838709677419355,0.9247311827956989,0.989247311827957,0.6853046594982081,93,0
0.1,0.4946236559139785,0.9247311827956989,1.0,0.6946236559139788,93,0
0.2,0.5053763440860215,0.9354838709677419,1.0,0.7014336917562726,93,0
0.3,0.5161290322580645,0.946236559139785,1.0,0.7077060931899645,93,0
0.4,0.4838709677419355,0.9247311827956989,1.0,0.6879928315412188,93,0
0.5,0.46236559139784944,0.9247311827956989,1.0,0.6784946236559143,93,0
0.6,0.41935483870967744,0.9032258064516129,0.978494623655914,0.6506400409626218,93,0
0.7,0.3870967741935484,0.8924731182795699,0.967741935483871,0.6240399385560678,93,0
0.8,0.3655913978494624,0.8494623655913979,0.967741935483871,0.604851510496672,93,0
0.9,0.3655913978494624,0.8172043010752689,0.9354838709677419,0.5899769585253457,93,0
1.0,0.34408602150537637,0.7956989247311828,0.9247311827956989,0.5705709165386587,93,0
```

**The maximum in this file is α = 0.3 at 51.6%. α = 0.5 scores 46.2%.** The file has no column recording which definition field was used.

### Which definition column produced that file

`scripts/atemporal/README.md:79` states plainly:

> "α=0.5 remains the chosen weight (originally validated; `weight_tuning.csv` contains the core_gloss tuning run, not the current llm_definition one)."

and `research_log.md:157` adds:

> "The original core_gloss comparison (51.6% vs 54.8%) is no longer valid with current data — the core_gloss run used `weight_tuning.csv` which predates the file re-generation"

**The second claim is false, and it matters.** `sense_text_representations.jsonl` has mtime 2026-03-12 12:52; `weight_tuning.csv` has mtime 2026-03-12 13:00 — eight minutes *later*. To settle it I re-ran the grid in memory against the current sense file. The `core_gloss` column reproduces `weight_tuning.csv` **to every decimal place** (0.4839, 0.4946, 0.5054, 0.5161, 0.4839, 0.4624, 0.4194, 0.3871, 0.3656, 0.3656, 0.3441). The recorded grid is a `core_gloss` run on *current* data, and is valid.

### The α = 0.5 selection has no surviving evidence — and my re-run contradicts it

There is **no recorded grid search over α with `llm_definition` on the current sense file**. `README.md:86`, `roadmap.md:46` and `documentation.md:1222` all assert α = 0.5 was "validated via grid search"; the only grid on disk is the core_gloss one, whose optimum is 0.3.

Re-running the grid myself (E5-large, current `sense_text_representations.jsonl`, same 93 instances, both definition columns; coarse accuracy added via `sense_clusters_reviewed.json`):

| α | `llm_definition` fine | coarse | MRR | | `core_gloss` fine | coarse |
|---|---|---|---|---|---|---|
| 0.0 (pure examples) | **48.4%** | 83.9% | 0.685 | | 48.4% | 83.9% |
| 0.1 | 46.2% | 86.0% | 0.677 | | 49.5% | 86.0% |
| 0.2 | 47.3% | 86.0% | 0.682 | | 50.5% | 85.0% |
| 0.3 | 44.1% | 86.0% | 0.669 | | **51.6%** | 85.0% |
| 0.4 | 43.0% | 86.0% | 0.667 | | 48.4% | 80.6% |
| **0.5 (deployed)** | **43.0%** | **85.0%** | 0.664 | | 46.2% | 79.6% |
| 0.6 | 45.2% | 86.0% | 0.676 | | 41.9% | 80.6% |
| 0.7 | 45.2% | 85.0% | 0.675 | | 38.7% | 74.2% |
| 0.8 | 41.9% | 81.7% | 0.655 | | 36.6% | 71.0% |
| 0.9 | 39.8% | 82.8% | 0.637 | | 36.6% | 69.9% |
| 1.0 (pure definition) | 44.1% | 83.9% | 0.651 | | 34.4% | 66.7% |

(α = 0.5 / `llm_definition` reproduces the shipped `predictions.csv` exactly: 43.0% fine, 84.95% coarse — so the reproduction is trustworthy.)

Three things follow, and they are the substance of what an interviewer will press on:

1. **α = 0.5 is not the optimum of its own grid, on either definition field.** For `llm_definition` it sits at the bottom of the 0.0–0.7 range on fine accuracy and is the *only* value in 0.1–0.7 that does not reach 86.0% coarse.
2. **Pure examples (α = 0.0) beat the deployed setting** by +5.4 pp fine on the very set the system is tuned and reported on. The definition contributes nothing positive at this weight.
3. **The rejected `core_gloss` + α = 0.3 configuration is the best on record** (51.6% fine, 85.0% coarse). It was rejected against the *pre-regeneration* 54.8% figure; after the 2026-04-07 recalibration to 43.0% the comparison inverted, and the decision was never revisited. `roadmap.md:61` still reads "core_gloss: 51.6% best (α=0.3) — worse than llm_definition", and the same stale comparison is baked into a code comment at `src/wsd/embedding_strategies.py:127-130`.

The honest framing for an interview: **α = 0.5 is an unvalidated default, chosen before a sense-file regeneration invalidated the numbers that justified it, and never re-tuned.** The spread across α is ~5 pp fine on n = 93 (≈ ±10 pp at 95% confidence), so none of these differences is individually significant — which is itself the correct defence, but it undercuts "validated via grid search" just as firmly.

---

## 4. Effect by sense frequency

**No such analysis exists.** It was planned twice and never executed. `documentation/research_log.md:960-964`, "Experiment Ideas (Backlog)", both unchecked:

```
- [ ] Test whether enrichment helps more for sparse vs. rich senses
- [ ] Correlation between sense frequency and accuracy
```

The machinery to do it exists and is **dead code**: `WSDMetrics.calculate_frequency_metrics` (`src/wsd/wsd_evaluator.py:92-121`) bins senses into `primary / secondary / tertiary / rare` and returns `primary_sense_accuracy` and `rare_sense_accuracy`. Grepping the repo, **`WSDEvaluator`, `WSDMetrics` and `ErrorAnalyzer` are never instantiated by any script or notebook** — the only references are the class definitions, the `__init__.py` re-export, and doc prose. `documentation.md:900-914` presents `wsd_evaluator.py` as *the* evaluation implementation; the numbers actually come from `generate_predictions.py` and `SimilarityWSD.evaluate`.

### Where the data is, if you want to run it

| Ingredient | Location |
|---|---|
| Per-instance correctness + gold rank + top-5 scores | `data/evaluation/atemporal/results/predictions.csv` (93 rows), `predictions_text_only.csv`, `predictions_random.csv` |
| A per-sense frequency-ish index | `data/processed/germanet/senses_base.csv`, column `sense_frequency` (2,022 rows), joinable on `synset_id` |
| Corpus-attested sense counts (the real frequency signal, 502 lemmas × decade) | `data/processed/diachronic/aggregated/pilot_v3_coarse_sense_counts.csv` — but these are *predicted*, not gold, so they cannot score accuracy |

**Important caveat about `sense_frequency`:** despite the name it is **not a corpus frequency**. It is the enumeration index of the sense within its lemma's GermaNet listing, assigned by `enumerate(...)` at `scripts/data/setup/rebuild_complete_wsd_senses.py:82,95`. Any "rare sense" analysis built on it is really a *sense-order* analysis, and inherits whatever ordering GermaNet happens to use. This is nowhere stated in the docs, and `wsd_evaluator.py:97-104` labels its bins `primary/secondary/tertiary/rare` as though it were frequency.

For completeness, here is the analysis nobody ran, computed by me on that index (n = 93, so this is indicative only):

| Gold sense's GermaNet index | n | α=0.0 | α=0.3 | **α=0.5 (deployed)** | α=1.0 |
|---|---|---|---|---|---|
| 1st-listed | 23 | 65.2% | 56.5% | 60.9% | 65.2% |
| 2nd-listed | 17 | 58.8% | 17.6% | **0.0%** | 11.8% |
| 3rd-listed | 31 | 35.5% | 41.9% | 38.7% | 51.6% |
| 4th-or-later | 22 | 40.9% | 54.5% | 63.6% | 36.4% |

No clean rare-vs-frequent story; the striking cell is that at α = 0.5 the system gets **0 of 17** second-listed senses right, where pure examples get 10. With 17 instances that is not a result, but it is a concrete pointer for a real experiment. There is no per-sense corpus frequency in the gold set at all, so the genuinely interesting question — does enrichment help senses that are rare *in the corpus* — **cannot currently be answered from these artifacts.**

---

## 5. "The classifier"

**There is no trained classifier.** Nothing is fit on labelled data anywhere in `src/` or `scripts/` — I grepped for `.fit(`, `train_test_split`, `LogisticRegression`, `SVC`, `classifier`; the single hit is `sm.OLS(...).fit()` in `scripts/analysis/baumann_replication.py:103`, which is the sociolinguistic regression, not the WSD system.

The system is **1-nearest-neighbour over cosine similarity, per lemma**, in `src/wsd/similarity_wsd.py`:

**"Training"** = building the index (`__init__`, lines 56-82): encode every sense of every lemma once into `v̂_s`, L2-normalize (line 75), build a `lemma → [sense indices]` dict (lines 80-82). No labels, no gradients, no fitted parameters.

**Inference** (`predict`, lines 91-135):
1. restrict candidates to `senses(lemma)` — the lemma is given, never predicted (line 108). Out-of-inventory lemma → `None`, counted as skipped.
2. embed the full sentence with the `query: ` prefix, L2-normalize (lines 112-113).
3. cosine = dot product against the candidate matrix (line 116).
4. predict `argmax` (lines 125-126); the raw cosine is stored as `confidence` (line 127) — it is a similarity, not a calibrated probability, and no doc claims otherwise.
5. `evaluate` (lines 165-264) additionally computes top-3, top-5 and MRR from the full ranking, and skips instances whose gold sense is not among the lemma's candidates (line 206).

**Hyperparameters — the complete list.** There are only these, and only the first is tuned at all:

| Parameter | Value | Where | Tuned? |
|---|---|---|---|
| α (definition weight) | 0.5 | `embedding_strategies.py:155` | see §3 — effectively no |
| Encoder | `intfloat/multilingual-e5-large` | `embedder.py:29` | selected on the 93-instance set (see §7 item 2) |
| Example count | 5 | prompt config; `llm_example_1..5` | never varied |
| Normalize | `True` | `embedder.py:44` | fixed |
| Batch size | 32 (64 for diachronic contexts) | `embedder.py:78`, `predictor.py:50` | irrelevant to results |
| Enrichment temperature | 0.7 | `enrich_senses_with_llm.py:265` | fixed |

**Coarse level.** The coarse number is not a second classifier. Fine predictions are mapped post-hoc through `sense_clusters_reviewed.json` and scored at cluster level (`generate_predictions.py:350-366`); the identical mapping is applied to gold and prediction. Clusters come from average-linkage agglomerative clustering on cosine distance over **`core_gloss`** embeddings, `distance_threshold = 0.10`, `min_clusters = 2` enforced only if the threshold yields 1 (`src/wsd/sense_clustering.py:66-85`, `scripts/clustering/compute_sense_clusters.py:52-53,58`), followed by full manual review by three annotators.

---

## 6. Results — everything on record

### 6.1 Atemporal WSD — the headline (93 gold instances, 10 lemmas)

Source: `README.md:27`, `research_log.md:110-120`, `documentation.md:958`. **All four numbers recomputed by me from the per-instance CSVs and confirmed exact.**

| Tier | Fine | Coarse | Recomputed |
|---|---|---|---|
| Random (analytical 1/n_senses, 1/n_clusters) | 24.5% | 50.5% | 24.51% / 50.54% ✓ |
| Text-only ablation (E5 + GermaNet hypernym text, no LLM) | 36.6% (34/93) | 75.3% (70/93) | 36.56% / 75.27% ✓ |
| **Weighted (E5 + LLM def + examples), deployed** | **43.0% (40/93)** | **84.9% (79/93)** | 43.01% / 84.95% ✓ |

Deltas as the docs frame them (`research_log.md:120`): encoder+lexicon buys +12.1 / +24.8 pp over random; LLM enrichment adds +6.4 / +9.6 pp on top.

Ranking metrics for the deployed system (`documentation.md:931`, recomputed ✓): **top-3 95.7%, top-5 100%, MRR 0.664**. Text-only: top-3 81.7%, MRR 0.617.

**Split:** there is only one split. The same 93 instances were used for encoder selection, α tuning, coarse-cluster validation and final reporting. The docs never state this. A second annotation set exists (98 instances, 10 different words — `data/evaluation/atemporal/annotation/eval_contexts.csv`) and is **unannotated** (`documentation.md:584`, `roadmap.md` §2b "NOT STARTED").

**Provenance of the eval contexts:** DWDS `kern`, years **1901–1999, median 1932**, genres Belletristik 27 / Gebrauchsliteratur 26 / Wissenschaft 22 / Zeitung 20 (computed from `contexts.csv`). `paper_plan.md:62` states this correctly; `README.md:101` and `research_log.md:454` contradict it (§7 item 4).

### 6.2 Encoder × representation grid (`embedding_comparison.csv`, 5 models × 8 variants = 40 rows)

Best variant per model, all on the same 93 instances:

| Model | Best variant | Fine acc | MRR |
|---|---|---|---|
| `BAAI/bge-m3` | weighted | **51.6%** | 0.717 |
| `paraphrase-multilingual-mpnet-base-v2` | weighted | 47.3% | 0.677 |
| `sentence-transformers/LaBSE` | weighted | 46.2% | 0.680 |
| `intfloat/multilingual-e5-base` | weighted | 46.2% | 0.676 |
| `intfloat/multilingual-e5-large` **(selected)** | weighted | **43.0%** | 0.664 |

The weighted variant is the best variant **for all five models** — that finding (`documentation.md:962`) holds in the data. E5-large single-variant rows: `llm_example_4` 47.3%, `llm_example_3` 46.2%, `llm_definition` 41.9%, `baseline` 36.6%, other single examples 30–39%.

**Caveat that the docs never state:** `research_log.md:149` records that after the 2026-04-07 regeneration only "`embedding_comparison.csv` updated (e5-large weighted row)". So **one row of this table is on the new sense file and the other 39 are on the old one.** The cross-model comparison is not internally comparable, and the 43.0-vs-51.6 gap is at least partly an artifact of that. This is why the table in `documentation.md:929-935` ranks 43.0% above 51.6% — but no doc says so.

### 6.3 Error analysis (deployed system, recomputed by me from `predictions.csv`)

Gold-rank distribution: rank 1 = 40, rank 2 = 27, rank 3 = 22, rank 4 = 3, rank 5 = 1. So **51% of errors (27/53) are rank-2 near-misses.** (`documentation.md:1023-1028` reports 51/27/10/5 and "64% of errors" — stale, see §7.)

Per-lemma, fine / coarse:

| Lemma | fine senses | clusters | fine acc | coarse acc |
|---|---|---|---|---|
| Glas | 4 | 2 | 88.9% | 100% |
| hören | 8 | 2 | 75.0% | 75.0% |
| sitzen | 7 | 6 | 66.7% | 66.7% |
| Schatz | 4 | 2 | 50.0% | 90.0% |
| bitter | 4 | 2 | 50.0% | 90.0% |
| Bahn | 7 | 2 | 30.0% | 90.0% |
| springen | 6 | 2 | 30.0% | 80.0% |
| wagen | 3 | 2 | 30.0% | 100% |
| Antwort | 2 | **1** | 14.3% | 100% |
| Doktor | 3 | 2 | 0.0% | 60.0% |

By polysemy band (recomputed): low 2-3 senses 14.8% (27 inst.), medium 4-6 53.8% (39), high 7+ 55.6% (27). **This inverts the documented finding** (`documentation.md:1034-1038`: 62.2 / 59.5 / 31.6) — see §7.

### 6.4 Coarse clustering

- Canonical mapping `sense_clusters_reviewed.json`: **503 lemmas, 1,813 fine senses → 1,135 clusters (37.4% reduction)** (counted by me). Docs quote the earlier 90-lemma pilot figure: 595 senses → 305 clusters, 39.5% reduction (`README.md:28`).
- Threshold t = 0.10, average linkage, cosine, min 2 clusters; all clusters human-reviewed and labelled; three annotators merged 2026-05-16.
- Effect on diachronic noise (v2 pipeline, `README.md` status table): max JSD 0.835 → 0.532; emergence+decline events 518 → 194 (~60% reduction).

### 6.5 Diachronic pipeline (combined v3, verified against the metric files)

| Quantity | Value | Source file |
|---|---|---|
| Coarse predictions | 318,611 rows, 502 lemmas | `predictions/pilot_predictions_v3_coarse.csv` ✓ |
| Contexts (v3) | 90,130 sentences, DWDS `kern` 1900–1999 | `documentation.md:981` / `README.md:26` |
| Mean inter-decade JSD | **0.0926** (4,488 decade pairs) | `metrics/pilot_v3_coarse_divergence.csv` ✓ |
| Mean entropy | **0.6583** (4,990 lemma-decades) | `metrics/pilot_v3_coarse_entropy.csv` ✓ |
| Emergence / decline events | **458 / 445** | corresponding metric CSVs ✓ |
| Per-lemma summary | 502 lemmas; mean clusters 2.26, mean entropy 0.669, mean JSD 0.086 | `analysis/per_lemma_summary.csv` ✓ |

(The per-lemma means differ slightly from the pooled means because one is a mean of lemma-means; both are correct, no doc explains the difference.)

Superseded runs, for the record: v1 → v2 24,325 predictions / 30,337 contexts / mean JSD 0.165 → v3 90-lemma 72,591 predictions / mean JSD 0.112 → combined 318,611 / 0.093.

### 6.6 Downstream correlational results

**German survey × model** (`documentation/survey_correlations_2026-05-15.md`), n = 73 lemmas (v3 pilot ∩ Grieshaber & Rubinskii, 78 participants × 544 lemmas):

| y vs. subjective polysemy | Spearman r | p | Pearson r | p |
|---|---|---|---|---|
| `mean_entropy` | **+0.345** | 0.0028 | +0.317 | 0.0063 |
| `n_clusters_coarse` | **+0.338** | 0.0035 | +0.269 | 0.0213 |
| `mean_jsd` | **+0.254** | 0.0299 | +0.268 | 0.0219 |
| `mean_jsd_normalized` | +0.094 | 0.4267 | +0.111 | 0.3510 |

Bonferroni α′ = 0.0125 → entropy and cluster-count survive, `mean_jsd` does not. Survey aggregation check: per-lemma mean-of-means 1.706 vs. G&R's reported 1.71.

**English replication vs. Hu et al. (2019)** (n = 557): `n_senses` +0.680, `mean_entropy` +0.586, `mean_jsd` +0.537, all p < 0.0001 — roughly 2× the German effect sizes.

**Baumann et al. (2023) replication** (n = 87, `entropy_slope` as diversification proxy):
- Bivariate: `log_frequency` −0.044 (p = 0.686, **fails to replicate**, and wrong sign in regression); `discriminability_desc` +0.220 (p = 0.041); `concreteness` +0.201 (p = 0.062).
- Model A (`~ log_frequency + discriminability_desc`): R² = 0.019, F p = 0.4476, nothing significant.
- Model B (`~ log_frequency + concreteness`): concreteness b = +0.326 (p = 0.0167), R² = 0.067, F p = 0.0533.
- **Hamilton bonus:** `n_clusters_coarse` ↔ `mean_jsd`, n = 89, Spearman **+0.714**, p < 0.0001.

**Baselines that were designed but never run:** MFS (most-frequent-sense) is specified at `documentation.md:718-730` and implemented at `scripts/atemporal/real_dwds_wsd_evaluation.py:126-215`, but **no MFS result file exists** anywhere under `data/`. The only baselines with numbers are random and text-only. For a WSD interview this is the conspicuous gap — MFS is the standard baseline in the field, and the system is currently reported without it.

---

## 7. Docs vs. code — discrepancies

Ordered by how much they'd matter if challenged.

1. **"α = 0.5 is optimal / validated via grid search" is contradicted by the only grid on disk.**
 `roadmap.md:43` — "alpha=0.5 is optimal (`weight_tuning.csv`)". `documentation.md:1222` — "Weighted combination (α=0.5, validated via grid search)". `README.md:86` presents it as the best configuration. `weight_tuning.csv` peaks at **α = 0.3 (51.6%)** and puts α = 0.5 at 46.2%; my re-run puts α = 0.5 at the bottom of the useful range for `llm_definition` too (§3). `scripts/atemporal/README.md:79` is the only document that concedes the tuning run doesn't cover the deployed configuration.

2. **"E5-large is best" is contradicted by its own table.**
 `documentation.md:964` — "E5-large is best: achieves the highest accuracy (43.0%)" — sits four lines below a table (`:931-932`) showing bge-m3 at 51.6%, and the table ranks 43.0% as #1 above 51.6% as #2. The real explanation (only the E5-large row was recomputed after the sense-file regeneration, `research_log.md:149`) is recorded in the log but never connected to the ranking table. **No document anywhere gives a rationale for keeping E5-large over bge-m3.**

3. **`documentation.md` §8.4 error analysis is from the superseded 54.8% run.**
 Rank distribution 51/27/10/5 (`:1023-1026`) sums to a 51/93 = 54.8% system; the current file gives 40/27/22/3/1. "64% of errors are near-misses" (`:1028`) is now 51%. The polysemy table (`:1034-1038`: low 62.2%, medium 59.5%, high 31.6%) recomputes to **low 14.8%, medium 53.8%, high 55.6%** — the documented conclusion "accuracy drops sharply for high-polysemy words" is **reversed** in the current data. Same for "Lemma 1 (0% accuracy, 9 errors)": the 0%-accuracy lemma is now Doktor with 10.

4. **The eval set's time period is stated three different ways.**
 `README.md:101` and `research_log.md:454` — "measured on contemporary DWDS texts... accuracy on 1900s data is unknown". `paper_plan.md:62` — "93-instance gold set spans 1901–1999 (median year 1932)". The data agrees with `paper_plan`: `contexts.csv` is DWDS kern, 1901–1999, median 1932. The limitation as written in the README is wrong in a way that *understates* the system's evidence base — but the real limitation (1980s missing, median 1932, so the 1990s end of the trajectories is untested) is only in `paper_plan`.

5. **`documentation.md` presents `wsd_evaluator.py` as the evaluation implementation; it is dead code.**
 `:900-914` and `:1096-1098` point at `WSDEvaluator` / `ErrorAnalyzer`. Neither, nor `WSDMetrics`, is instantiated anywhere outside its own module. Reported numbers come from `generate_predictions.py` + `SimilarityWSD.evaluate`.

6. **`coarse_clustering_plan.md` specifies a clustering setup that was not built.**
 Line 18 and line 47 specify the **`weighted`** representation for clustering ("the same weighted embedding already used for WSD"); line 55 proposes thresholds **0.14 / 0.16 / 0.18**. The code uses **`core_gloss`** (`compute_sense_clusters.py:52`) at **t = 0.10**. The doc is honestly marked "Status: Planning (as of 2026-03-10)" and `roadmap.md` §2c records the change, so this is stale-not-wrong — but the plan doc is the one a reader would open for the clustering rationale.

7. **`src/wsd/sense_clustering.py:4` docstring says clustering runs "on weighted E5-large embeddings".** The callers pass `core_gloss`. The module docstring is simply wrong.

8. **Stale claim frozen in a code comment.** `src/wsd/embedding_strategies.py:127-130` — "BEST METHOD — 54.8% accuracy … core_gloss was tested … but scored lower (best 51.6% at α=0.3)". Both halves are obsolete: the method scores 43.0%, and 51.6% is now *higher*. `research_log.md:157` acknowledges the comparison is invalid; the code comment and `roadmap.md:61` were never updated.

9. **Two notebooks still carry the pre-recalibration numbers while loading the recalibrated CSVs.**
 `combination_analysis.ipynb` — "Accuracy: 54.8% — The highest among all 45 configurations"; `sense_granularity_analysis.ipynb` — "coarse clustering improves atemporal accuracy from 54.8% → 81.7%". The files they read now say 43.0% → 84.9%. Also "45 configurations": the CSV has 40 rows.

10. **`documentation.md:696` says the context embedding is 384-dim.** That is MiniLM's dimension; the deployed encoder is 1024-dim. Vestigial from the original default model (`embedder.py:22`).

11. **Cluster totals in `README.md:28` (595 senses → 305 clusters) describe the 90-lemma pilot**, while the shipped mapping is 503 lemmas / 1,813 senses / 1,135 clusters. Not wrong, but unlabelled — a reader will take the smaller numbers as current.

12. **Two example-indexing conventions coexist** (`llm_example_0..4` in `scripts/atemporal/`, `llm_example_1..5` in `src/`). Documented as a trap (`coarse_clustering_plan.md:300`, `pipeline.md:269`) rather than fixed. Currently harmless — each family loads and consumes consistently — but it is a live footgun for anyone who moves a sense DataFrame between the two.

---

## 8. Limitations

### Stated in the documentation

From `README.md:96-101`, `paper_plan.md:69-73`, `research_log.md:440-460`, `coarse_clustering_plan.md`:

- 26 pilot words have no GermaNet senses (`fahren`, `sehen`, `halten`, `stehen`, …) — deferred to Phase 4.
- **DWDS `kern` genre drift** (Belletristik ↓, Wissenschaft ↑ across the century) — a structural confound for *any* diachronic claim made from this corpus. Named as unfixable.
- Words with < 100 total contexts are unreliable regardless of clustering (`Bass` 25, `Frosch` 109, `Knoten` 270, `Lappen` 189) and are excluded from case studies.
- GermaNet is not a learner's dictionary; coarse clusters are an approximation, and cluster boundaries are subjective (mitigated by manual review, not eliminated).
- Accuracy on early-20th-century text is unmeasured and "likely lower"; **no historical gold standard exists** to measure it (`research_log.md:454-455`).
- Therefore: only report words with large, monotonic changes over ≥3 consecutive decades; manual sentence-level validation is "not optional" (`research_log.md:458-461`). **That validation is still pending** — `case_study_validation.ipynb` is a scaffold with a rubric and no filled-in verdicts.
- `entropy_slope` is noisy: mean −0.0002, sd 0.0024, few individually significant slopes (`baumann_replication_2026-05-15.md:118-121`).
- Uncorrected p-values throughout the correlational work; Bonferroni thresholds stated but results reported uncorrected.
- 1980s missing from the gold eval set (`paper_plan.md:62`).

### Not stated, and worth owning before someone else raises them

- **No held-out data.** Encoder choice, α, and the coarse mapping were all decided on the same 93 instances the headline numbers are reported on. Every reported figure is a training-set figure. The 98-instance second set that would fix this exists and is unannotated.
- **n = 93, single annotator, no IAA.** `documentation.md:562-572` labels multi-annotator agreement as "Planned"; it never happened for the WSD gold set (it *did* happen for cluster review, which is a different task). One annotator's readings define the ceiling. A 1-instance change moves accuracy by 1.1 pp; the entire α grid spans ~5 pp.
- **The coarse label space is close to binary.** 8 of the 10 eval lemmas have exactly 2 clusters; **`Antwort` has 1**, so its 7 instances are unconditionally correct at coarse level. Excluding `Antwort`, coarse accuracy is 83.7% (n = 86). 84.9% against a 50.5% floor is a genuine result, but it is a result on a near-binary task, and the headline framing ("84.9% coarse") invites readers to hear something much stronger.
- **Mild circularity in the coarse metric.** The clusters were drawn and hand-reviewed by the same researcher who reports the coarse accuracy. `coarse_clustering_plan.md:299` anticipates exactly this ("do not tune the clusters to maximise this metric") and the mapping was frozen before scoring — worth saying out loud, because it is the first thing a reviewer will ask.
- **The definition may be dead weight.** At the deployed α, pure examples score higher on fine accuracy (48.4% vs 43.0%), and `llm_definition` alone is the weakest LLM variant for E5-large (41.9%). The system's own ablation data does not support 50% weight on the definition.
- **Sense texts are LLM-generated and non-deterministic.** A regeneration moved fine accuracy 54.8% → 43.0% (`research_log.md:141-147`) — an 11.8 pp swing from resampling the *sense inventory text*, larger than any effect the project reports. No seed, no cached generation manifest, no version field in the JSONL. Nothing in the repo would let you reproduce the 54.8% artifacts.
- **No MFS baseline was ever run** (§6.6), despite being specified and implemented. Against a random floor only, "+6.4 pp from LLM enrichment" is hard to situate — in WSD, MFS is the baseline that matters.
- **`sense_frequency` is a sense-order index, not a frequency** (§4), yet `wsd_evaluator.py` bins it as `primary…rare`. Any future frequency analysis built on that column without checking would silently measure the wrong thing.
- **Confidence is an uncalibrated raw cosine** (`similarity_wsd.py:127`); the observed range is narrow (~0.79–0.83 in the sample rows of `predictions.csv`), so it should not be used for thresholding or abstention without calibration. No doc proposes to, but the column name invites it.
- **The diachronic conclusions inherit an unmeasured error rate.** 84.9% coarse on 1901–1999 gold text is reassuring for the period, but per-decade accuracy is unknown, and the aggregation assumes errors are decade-independent. If E5-large degrades on 1900s orthography and vocabulary, error rates are *correlated with the independent variable* — the most serious threat to the diachronic claims, and the docs raise the concern (`research_log.md:454`) without quantifying it.

---

## Quick answers, if you have 60 seconds

- **Sense representation:** definition and 5 examples encoded **separately** by multilingual-E5-large (1024-d, mean-pooled tokens, L2-normalized, `passage:` prefix); the 5 example vectors are mean-pooled; then a weighted vector sum. No concatenation.
- **Weighting:** `v = 0.5·d + 0.5·mean(e_i)`, then L2-normalize — vectors, not scores. `src/wsd/embedding_strategies.py:155`, hardcoded.
- **Tuning:** α ∈ {0.0…1.0} step 0.1, on the 93-instance gold set, fine top-1 accuracy. The recorded grid peaks at α = 0.3 (51.6%) with `core_gloss`. **α = 0.5 with `llm_definition` was never validated on current data, and re-running the grid shows it is not optimal.**
- **Frequency effect:** never analysed; twice-listed backlog item; the per-instance data is in `data/evaluation/atemporal/results/predictions.csv` and the only "frequency" column available is really a sense-order index.
- **Classifier:** there isn't one — 1-NN cosine over per-lemma candidate senses, no training, no fitted parameters.
- **Headline:** 43.0% fine / 84.9% coarse on 93 instances, vs. 24.5% / 50.5% random and 36.6% / 75.3% text-only. All four verified against the per-instance CSVs. No held-out split; no MFS baseline.
