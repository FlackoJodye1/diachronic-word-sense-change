# Paper Plan — German WSD / Hu et al. Replication

**Status:** Skeleton (as of 2026-05-07)
**Target venue:** EMNLP
**Framing / USP:** German replication of Hu et al., evaluated against human survey data; extension via Hamilton-style frequency × change-rate correlation.

This document is the source of truth for *what the paper claims* and *where each piece of evidence lives*. Pipeline-level docs (`pipeline.md`, `coarse_clustering_plan.md`, `llm_judge_clustering.md`) describe how the system works; this document describes how those artifacts assemble into a paper.

---

## 1. Scope decisions (from 2026-05-07 meeting)

### Variant 1 — committed
- German replication of Hu et al.
- Evaluation via survey (German participants), correlated with model-derived sense distributions
- English validation using Amandine's data + parallel English survey to re-test Hu et al.

### Extension 2.1 — committed
- Hamilton-style replication: correlation between word frequency and rate of sense change (JSD-based) for German
- Anchors the paper in Hamilton et al. and Andreas's EMNLP empirical chapter (Ch. 5)

### Extension 2.2 — open
- "Correctness?" — meeting note unclear; revisit before scoping in.

### Extension 2.3 — likely out of scope
- Atemporal: human sensitivity to sense-frequency differences. Flagged as "vllt zu exotisch" in the meeting. Park unless v1 has space.

---

## 2. Claims (the things the paper argues)

> **TBD** — fill in once survey design is fixed. Working hypotheses below; treat as drafts, not commitments.

- **C1 (replication):** Hu et al.'s WSD-based diachronic methodology transfers to German with comparable internal validity.
- **C2 (human alignment):** Model-derived sense-frequency distributions correlate with human judgments of polysemy / sense prominence.
- **C3 (Hamilton replication):** German word frequency negatively correlates with rate of semantic change (operationalised as mean inter-decade JSD).
- **C4 (cross-lingual):** Hu et al.'s English-data conclusions hold (or fail to hold) when re-evaluated against the English survey.

---

## 3. Section-by-section plan

### 3.1 Introduction (½–¾ page, must fit on first page)
- Motivate diachronic WSD; one-paragraph summary of contributions; key result numbers.
- **Hook figure (page 1):** trajectory plot for `springen` or `Welle` showing sense distribution shifting across decades. Source: `notebooks/analysis/diachronic_pilot_results.ipynb`.

### 3.2 Related Work & SOTA (¾–1 page)
- Hu et al. (replication target)
- Hamilton et al. (frequency × semantic change)
- Andreas EMNLP empirical chapter (replication framing)
- German-specific: Charbonnier & Wartena 2019 (target word selection); GermaNet (Hamp & Feldweg)
- Standard contextual-WSD references (Loureiro, Blevins & Zettlemoyer, etc.)

### 3.3 Methods (largest section)
- **3.3.1 Sense inventory.** GermaNet v19.0 + LLM enrichment (`core_gloss`, `llm_definition`, `llm_examples`). 90 pilot lemmas, 595 senses. Cross-ref: `documentation/pipeline.md`.
- **3.3.2 Atemporal WSD.** E5-large multilingual + weighted representation (0.5·definition + 0.5·avg(5 examples)). Cross-ref: `scripts/atemporal/generate_predictions.py`.
- **3.3.3 Coarse-sense clustering.** Agglomerative on `core_gloss` embeddings, t=0.10, with LLM-judge verification. Cross-ref: `documentation/coarse_clustering_plan.md`, `documentation/llm_judge_clustering.md`.
- **3.3.4 Diachronic pipeline (v3).** DWDS `kern` corpus (1900–1999), 90,130 sentences, predict → coarse-map → aggregate per decade → JSD/entropy/emergence/decline. Cross-ref: `documentation/pipeline.md`.

### 3.4 Evaluation
- **3.4.1 Atemporal benchmark (3-tier).** Random 24.5/50.5 → text-only ablation 36.6/75.3 → weighted 43.0/84.9 (fine/coarse). 93-instance gold set.
- **3.4.2 Survey.** **TBD** — see §5.
- **3.4.3 Entropy × subjective polysemy.** For each lemma: compute sense-distribution entropy from coarse counts; correlate with mean subjective polysemy rating from the survey. Run for both German and English.

### 3.5 Application — Hamilton extension
- Per-lemma frequency from DWDS kern; per-lemma mean inter-decade JSD from `pilot_v3_coarse_divergence.csv`.
- Spearman correlation, controlling for polysemy (number of coarse senses).
- Robustness check on a frequency-balanced sub-sample to address kern genre drift.

### 3.6 Limitations
- GermaNet is not a learner's dictionary; coarse clusters are an approximation.
- DWDS kern genre drift (Belletristik ↓, Wissenschaft ↑); structural confound for any diachronic claim.
- 26 pilot words have no GermaNet senses (deferred to Phase 4 of the project).
- Coarse-cluster boundary subjectivity (mitigated by LLM judge + manual review, but not eliminated).
- Atemporal accuracy ceiling (43.0% fine) — discuss what this means for downstream diachronic interpretation.

---

## 4. Artifact ↔ section map

| Section | Artifact | Path | Status |
|---|---|---|---|
| 3.1 hook figure | springen / Welle trajectory plot | `notebooks/analysis/diachronic_pilot_results.ipynb` | Notebook exists; figure to be finalised |
| 3.3.1 | Sense inventory (90 lemmas, 595 senses) | `data/processed/germanet/senses_pilot.jsonl` | Done |
| 3.3.2 | Atemporal predictions | `scripts/atemporal/generate_predictions.py` | Done |
| 3.3.3 | Coarse mapping | `data/processed/germanet/sense_clusters_reviewed.json` | Done (t=0.10); judge calibration pending |
| 3.3.4 | v3 fine predictions (72,591 rows) | `data/processed/diachronic/predictions/pilot_predictions_v3.csv` | Done |
| 3.3.4 | v3 coarse predictions | `data/processed/diachronic/predictions/pilot_predictions_v3_coarse.csv` | Done |
| 3.3.4 | v3 metrics (JSD, entropy, emergence, decline) | `data/processed/diachronic/metrics/pilot_v3_coarse_*.csv` | Done |
| 3.4.1 | 3-tier atemporal table (24.5/50.5 → 36.6/75.3 → 43.0/84.9) | `data/evaluation/atemporal/` | Done |
| 3.4.2 | Survey design + raw responses | TBD | **Not started** |
| 3.4.3 | Entropy × subjective polysemy correlation | (depends on survey) | **Not started** |
| 3.5 | Frequency × change-rate analysis | (script TBD) | **Not started — inputs ready** |
| 3.5 (English) | Hu et al. cross-check on Amandine's data | TBD | **Not started — data handoff pending** |

---

## 5. Open questions / decisions needed

### 5.1 Survey (highest priority)
- **Task type:** ranking-by-frequency, Likert polysemy rating, sense discrimination, or combination? Decide by reading Hu et al.'s exact protocol.
- **Word set:** all 90 v3 lemmas, or a stratified subset by polysemy / frequency / change rate? Subset feels more realistic for participant load.
- **Participant pool & language proficiency criteria.**
- **English-version parallelism:** must mirror German exactly to support cross-lingual claims.
- **Statistical analysis plan:** correlations, agreement metrics, hypothesis tests — pre-register before data collection.

### 5.2 English replication
- Amandine data handoff: format, sense inventory (WordNet?), corpus.
- Whether English replication is a v1 submission requirement or a follow-up paper.

### 5.3 Coarse clustering
- LLM-judge calibration result decides whether to raise threshold above t=0.10 before paper-final numbers are frozen.

### 5.4 Case studies for the hook figure
- `springen` and `Welle` are current candidates. Decide after manual sentence-level validation.

---

## 6. Timeline (working draft)

| Week of | Milestone |
|---|---|
| 2026-05-07 | LLM-judge calibration run + decision on threshold |
| 2026-05-07 | Manual sentence-level validation of springen / Welle |
| 2026-05-11 | Multi-annotator setup: split 407 unlabelled lemmas / 921 clusters across 3 annotators (≈136 lemmas each). See `scripts/clustering/ANNOTATION.md`. |
| 2026-05-14 | Hamilton-extension analysis script + first plot |
| 2026-05-14 | Survey design v1 (task type, word set, analysis plan) |
| TBD | Amandine data handoff |
| TBD | Survey deployment (German) |
| TBD | Survey deployment (English) |
| TBD | First full draft (Methods → Evaluation → Application → Intro) |

---

## 7. Working notes from 2026-05-07 meeting (raw)

Preserved verbatim for traceability:

> Variante 1: Deutsche Replikationsstudie zu Hu et al + Evaluierung von den Survey Daten. Daran dranhängen: Evaluieren und vergleich mit menschlichen Rankings. Englische Daten von Amandine um Hu et al zu validieren.
>
> Variante 2 (Extensions): 2.1 Unterschiedliche Wandelraten (Korrelation mit Wortfrequenz, Hamilton-Replikation auf Deutsch); 2.2 Correctness?; 2.3 Atemporal: Sensitivität für Häufigkeitsunterschiede von Bedeutungen (vllt zu exotisch).
>
> Tipps für EMNLP: Deutsch als USP. Limitations: GermaNet ist kein Lexikon, Acc.
>
> Gliederung: Einleitung (¾–½ Seite, erste Seite, mit Hauptidee-Abbildung); Related Work & SOTA (¾–1 Seite); Methoden (viel Platz, Hu et al. nachbauen); Evaluierung (Survey, Baseline, Prozedere; optional englische Survey + Entropiemaß über subjektive Mehrdeutigkeitseinschätzungen — für jedes Wort Verteilung der Senses, Entropie hoch = gleichverteilt, niedrig = skewed); Anwendungsteil (Hamilton-Replikation, Korrelation Frequenz × Wandelrate via JSD; Andreas EMNLP empirischer Teil Kapitel 5).