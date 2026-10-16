# Paper Plan — German WSD / Hu et al. Replication

**Status:** v2 — robustness pass done (2026-10-16); C3 falls, C5 added. See `analysis/robustness/RESULTS.md`.
**Target venue:** EMNLP
**Framing / USP:** German replication of Hu et al., evaluated against human survey data; extension via Hamilton-style frequency × change-rate correlation.
**⚠ Framing needs revisiting (2026-10-16):** the frequency × change-rate correlation does not survive a JSD null baseline in any temporal binning. The polysemy × change-rate correlation does (at period resolution). If the extension stays the USP, it has to be rebuilt around the latter.

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

Empirical status flagged inline. C1 and C4 are still working hypotheses; C2 and C5 have empirical support; C3 was tested and does not hold.

- **C1 (replication):** Hu et al.'s WSD-based diachronic methodology transfers to German with comparable internal validity. *Status: working hypothesis.*
- **C2 (human alignment):** Model-derived sense-frequency distributions correlate with human judgments of polysemy / sense prominence.
  *Status: **supported, first-pass** (2026-05-14). `mean_entropy` × Grieshaber-Rubinskii subjective polysemy: Spearman r = +0.345 (p = 0.0028), n = 73. See `documentation/survey_correlations_2026-05-15.md`.*
  *Caveat (2026-10-16): the entropy variant carries no information beyond cluster count — partial r = −.005 once `n_clusters_coarse` is controlled. Phrase C2 in terms of cluster count, or drop the entropy framing.*
- **C3 (Hamilton replication):** German word frequency negatively correlates with rate of semantic change (operationalised as mean inter-decade JSD).
  *Status: **not supported** (2026-10-16). Against a JSD null baseline the effect is sampling noise in every temporal binning tested — raw r = −.348 at decade resolution, null-corrected excess ≈ 0 (n.s.) at 10 decades, 5 periods and 2 periods. Recommendation: drop C3 as a positive claim and report it as a null result with the noise analysis as the explanation. See `analysis/robustness/RESULTS.md` §5.*
- **C5 (Innovation, added 2026-10-16):** More polysemous words change faster. *Status: **supported at period resolution**. Null-corrected polysemy × JSD is +.21 (p < .001) at 2 periods — for cluster count, entropy and the independent survey ratings alike — but only +.04 (n.s.) at decade resolution, where 90% of the observed JSD is sampling noise. Report at period resolution with the noise share stated; the magnitude is about half the uncorrected +.37.*
- **C4 (cross-lingual):** Hu et al.'s English-data conclusions hold (or fail to hold) when re-evaluated against the English survey. *Status: Hu et al. per-word metrics extracted (2026-05-14, 3,220 lemmas); awaiting Amandine's English survey for the join.*

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
- **3.3.3 Coarse-sense clustering.** Agglomerative on `core_gloss` embeddings, t=0.10, then manual review by three annotators across all 503 lemmas. (LLM-judge approach prototyped but parked — see `documentation/llm_judge_clustering.md`.) Cross-ref: `documentation/coarse_clustering_plan.md`, `scripts/clustering/ANNOTATION.md`.
- **3.3.4 Diachronic pipeline (v3).** DWDS `kern` corpus (1900–1999), 90,130 sentences, predict → coarse-map → aggregate per decade → JSD/entropy/emergence/decline. Cross-ref: `documentation/pipeline.md`.

### 3.4 Evaluation
- **3.4.1 Atemporal benchmark (3-tier).** Random 24.5/50.5 → text-only ablation 36.6/75.3 → weighted 43.0/84.9 (fine/coarse). 93-instance gold set spans 1901–1999 (median year 1932, 1980s missing — note in Limitations).
- **3.4.2 Survey.** Grieshaber & Rubinskii German polysemy survey (78 participants × 544 lemmas, ratings 0–11 with mode 1). Aggregation: `scripts/analysis/aggregate_survey.py` → `data/processed/survey/survey_polysemy_per_lemma.csv` (per-lemma mean / median / std / n_responses; mean-of-means 1.706 vs. G&R reported 1.71 ✓).
- **3.4.3 Entropy × subjective polysemy (German, first pass).** `mean_entropy` × subjective polysemy: Spearman r = +0.345 (p = 0.003); `n_clusters_coarse` r = +0.338; `mean_jsd` r = +0.254; `mean_jsd_normalized` r = +0.094 (n.s.). n = 73 (intersection of v3 pilot ∩ survey). Two of three significant after Bonferroni. Details + 4-panel figure: `documentation/survey_correlations_2026-05-15.md`, `figures/survey_correlations.png`. English replication via Hu et al. metrics is set up but awaits Amandine's English survey.

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
| 3.3.3 | Coarse mapping (canonical, 503 lemmas) | `data/processed/germanet/sense_clusters_reviewed.json` | Paul's 136 lemmas done (2026-05-13); awaiting Amandine + Nadja |
| 3.3.3 | Multi-annotator tooling | `scripts/clustering/{split_annotation_work,review_app,merge_annotations}.py`, `scripts/clustering/ANNOTATION.md` | Done (2026-05-11) |
| 3.3.4 | v3 fine predictions (72,591 rows) | `data/processed/diachronic/predictions/pilot_predictions_v3.csv` | Done |
| 3.3.4 | v3 coarse predictions | `data/processed/diachronic/predictions/pilot_predictions_v3_coarse.csv` | Done (re-run after annotation merge) |
| 3.3.4 | v3 metrics (JSD, entropy, emergence, decline) | `data/processed/diachronic/metrics/pilot_v3_coarse_*.csv` | Done (re-run after annotation merge) |
| 3.3.4 | Per-lemma model summary | `data/processed/analysis/per_lemma_summary.csv` (90 lemmas) | Done (2026-05-14) |
| 3.4.1 | 3-tier atemporal table (24.5/50.5 → 36.6/75.3 → 43.0/84.9) | `data/evaluation/atemporal/` | Done |
| 3.4.2 | Survey per-lemma aggregate (544 lemmas) | `data/processed/survey/survey_polysemy_per_lemma.csv` | Done (2026-05-14) |
| 3.4.3 | Survey × model correlations + 4-panel figure | `data/processed/analysis/survey_correlations.csv`, `figures/survey_correlations.png`, `documentation/survey_correlations_2026-05-15.md` | **First pass done (2026-05-14)**; n = 73; primary r = +0.345 (p = 0.003) |
| 3.5 | Frequency × change-rate analysis | (script TBD: `scripts/analysis/frequency_change_correlation.py`) | **Not started — inputs ready** |
| 3.5 (English) | Hu et al. per-word metrics (3,220 lemmas) | `data/processed/analysis/hu_etal_per_word.csv` | Done (2026-05-14) |
| 3.5 (English) | Hu et al. × English survey correlation | TBD | **Awaiting Amandine's English survey** |

---

## 5. Open questions / decisions needed

### 5.1 Survey — first results in, framing decisions remain
- ✅ **Survey source decided:** G&R 2024 polysemy ratings (78 participants, 544 lemmas) — used as-is, no new survey collection needed.
- ✅ **First-pass correlation done** (2026-05-14): three significant Spearman correlations; mean_entropy primary at r=+0.345 (p=0.003).
- **Open: framing of mean_jsd_normalized.** Raw `mean_jsd` correlates (r=+0.254); the cluster-count-normalized variant does not (r=+0.094). Decide whether to frame this as "normalisation removes polysemy bias" or "polysemy was mediating raw JSD". Latter is more defensible.
- **Open: partial correlation entropy ↔ polysemy | n_clusters_coarse.** n_clusters_coarse and mean_entropy correlate near-identically with polysemy — partial correlation tells us whether entropy contributes signal beyond cluster count. Quick win for the methods section.
- **Open: multiple-comparison correction.** Report Bonferroni-adjusted thresholds in the paper. mean_entropy + n_clusters_coarse survive; mean_jsd doesn't.
- **Open: confidence intervals on correlations.** Wide at n=73; report bootstrap CIs.

### 5.2 English replication
- ✅ Hu et al. per-word metrics extracted (3,220 lemmas) — `data/processed/analysis/hu_etal_per_word.csv`.
- **Open: Amandine's English subjective-polysemy survey data** — format, lemma overlap with Hu et al., timing of handoff.
- **Open: whether English replication is v1 submission requirement** or follow-up paper.

### 5.3 Coarse clustering
- **LLM-judge approach parked** (see `documentation/llm_judge_clustering.md` — manual review by 3 annotators absorbed the motivation).
- **Open: re-run downstream pipeline after Amandine + Nadja submit their annotations.** Apply coarse mapping → aggregate → metrics, regenerate `per_lemma_summary.csv`. Numbers in §3.4.3 will refresh accordingly.

### 5.4 Case studies for the hook figure
- `springen` and `Welle` are current candidates. Decide after manual sentence-level validation (still pending).

### 5.5 Hamilton extension (direction changed 2026-10-16)
- Frequency × change-rate Spearman correlation: tested, does not survive the JSD null baseline (C3). Not worth building the extension around.
- Open decision: rebuild the extension around polysemy × change-rate (C5, holds at +.21 at period resolution), or report the frequency result as an explicit null with the noise analysis as the contribution. The second option is defensible on its own — "decade-resolution JSD is 90% sampling noise" is a methodological finding other replications need.
- Either way the analysis must run on 20-year or 50-year periods, not decades. Binning decision needed from co-authors.

---

## 6. Timeline

| Date | Milestone | Status |
|---|---|---|
| 2026-05-07 | Post-meeting scoping: Variant 1 + Ext 2.1 committed; LLM judge parked | ✅ |
| 2026-05-11 | Multi-annotator setup: split 407 unlabelled lemmas / 921 clusters across 3 annotators (≈136 lemmas each). See `scripts/clustering/ANNOTATION.md`. | ✅ |
| 2026-05-13 | Paul's annotation chunk: 136 lemmas / 258 clusters labelled (cluster count dropped from 307 due to merges) | ✅ |
| 2026-05-14 | Per-lemma model summary CSV (90 lemmas) for survey correlation | ✅ |
| 2026-05-14 | G&R survey aggregation (544 lemmas) + first-pass correlation analysis (n=73) | ✅ |
| 2026-05-14 | Hu et al. per-word metrics extraction (3,220 lemmas) | ✅ |
| Pending | Amandine + Nadja finish their annotation chunks (~135 each) | ⏳ blocked on humans |
| Pending | Merge → re-run downstream pipeline → updated metrics + correlations | ⏳ blocked on annotation |
| Pending | Manual sentence-level validation of `springen` / `Welle` (hook figure) | ⏳ |
| 2026-10-16 | Robustness pass: JSD null baseline, H2 partial correlation, survey split-half (.86), temporal-binning sensitivity. C3 falls, C5 added. `analysis/robustness/RESULTS.md` | ✅ |
| Pending | Hamilton-extension analysis script + first plot — rescope first (§5.5), must run on periods not decades | ⏳ blocked on binning decision |
| Pending | Amandine's English subjective-polysemy survey | ⏳ |
| Pending | First full paper draft (Methods → Evaluation → Application → Intro) | ⏳ |

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