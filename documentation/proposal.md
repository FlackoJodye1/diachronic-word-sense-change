# Project Proposal — Interdisciplinary Project

## Modeling Diachronic Sense Change in German Using Deep Contextualized Word Representations

**Author:** Paul Leon Schmitt (11714338)  
**Institute:** TU Wien  
**Supervisors:** Andreas Baumann, Gábor Recski (tbd)  
**Submission Date:** February 2025  

---

## 1. Introduction and Motivation

Word meanings evolve with social, cultural, and technological change. Quantitatively modeling such diachronic shifts enables analysis of innovation and obsolescence in language use. Prior work for English shows that contextualized embeddings (e.g., BERT) capture sense variation, but comparable, sense-annotated resources for German are scarce.

### Research Gap
German corpora like DWDS offer rich temporal coverage but lack mappings to sense inventories such as GermaNet. No dataset currently supports supervised or semi-supervised modeling of German sense dynamics. Bridging DWDS with GermaNet would enable computationally grounded historical linguistics.

### Research Questions
1. How can German word senses be aligned between DWDS and GermaNet?
2. Do BERT-style contextual embeddings capture diachronic sense variation across decades in German?
3. *(Optional)* Which factors (frequency, distinctiveness, competition) explain persistence vs. decline of senses?

---

## 2. Related Work

Deep contextual models have been used for diachronic sense tracking in English. GermaNet provides a structured German sense inventory. Cognitive and frequency-based explanations for sense decline are discussed in recent work. The DWDS corpus supplies temporally stratified German usage data.

---

## 3. Objectives

### 3.1 Dataset Preparation
- Select target words with diachronic variation (e.g., *Netz*, *Bank*, *Ball*) from DWDS.
- Sample contextual occurrences per decade.

### 3.2 Sense Mapping and Annotation
- Link target lemmas to GermaNet synsets (glosses/relations).
- Use embedding similarity or LLM-assisted matching to assign probable senses to DWDS occurrences.

### 3.3 Modeling Sense Change
- Generate contextual embeddings with multilingual BERT or sentence transformers.
- Cluster by period to approximate sense groupings and align clusters with GermaNet senses.
- Compare emergence, merging, or decline across periods.

### 3.4 Quantitative Analysis & Evaluation
- Measure sense distributions over time to identify broadening, narrowing, and replacement.
- Create visualizations and case studies.
- Compare to human judgments and linguistic descriptions (in coordination with other students).

---

## 4. Methodology and Implementation Plan

### 4.1 Data and Resources
- **DWDS:** temporal samples.
- **GermaNet:** sense inventory.
- **LLMs (e.g., Gemma):** support difficult mappings where lexical overlap is weak.

### 4.2 Workflow

**Stage 1 — Dataset Creation**
- Extract decade-wise contexts.
- Compute similarity between occurrence sentence embeddings and GermaNet gloss embeddings to assign senses.

**Stage 2 — Modeling**
- Encode occurrences with pretrained contextual models (no heavy fine-tuning).
- Cluster per decade and compute temporal sense distributions.

**Stage 3 — Evaluation**
- Compare distributions to human judgments from collaborating Germanistik students (questionnaire-based).
- Optionally analyze correlates (frequency/competition).

---

## 5. Datasets
- DWDS (decade-stratified German corpora).
- GermaNet (German lexical-semantic network).

---

## 6. Expected Results
1. A German diachronic WSD dataset aligned with GermaNet.
2. A framework for detecting sense emergence/merging/decline with entropy-based metrics.
3. Visualizations and case studies of sense trajectories.
4. A report and poster summarizing methods, findings, and implications.

---

## 7. Appendix

### 7.1 Timeline

| Period        | Phase        | Tasks                                                                 |
|---------------|--------------|-----------------------------------------------------------------------|
| Sept 2025     | Foundations  | Study papers; explore DWDS/GermaNet; define mapping strategy          |
| Oct–Nov 2025  | Dataset      | Extract samples by decade; align with GermaNet; build WSD dataset     |
| Nov–Dec 2025  | Modeling     | Contextual encoding; clustering; temporal sense alignment             |
| Jan 2026      | Evaluation   | Entropy metrics; human judgments; case studies                         |
| Feb 2026      | Reporting    | Poster; final report (10–15 pages)                                     |

---

## References

1. Hu, Li, Liang (2019). *Diachronic sense modeling with deep contextualized word embeddings*. ACL.  
2. Geyken & Gloning (2015). *The DWDS corpus*. Journal for Language Technology.  
3. Hamp & Feldweg (1997). *GermaNet: A lexical-semantic net for German*. ACL Workshop.  
4. Kali, Xu, Stevenson (2024). *Cognitive factors in word sense decline*. TACL.  
5. Devlin et al. (2019). *BERT*. NAACL-HLT.  
6. Reimers & Gurevych (2019). *Sentence-BERT*. EMNLP.  
7. Google DeepMind (2024). *Gemma*. arXiv:2403.08295.
