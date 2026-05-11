# LLM-as-Judge for Coarse Sense Clustering

**Status:** Calibration phase (as of 2026-05-07)
**Motivation:** Manual review is the bottleneck for raising the agglomerative-clustering threshold and producing more merged clusters. An LLM judge automates the *verification* step while leaving the geometric clustering and the final human sign-off intact.

---

## 1. Problem statement

The canonical mapping `sense_clusters_reviewed.json` was produced at threshold `t=0.10` and manually reviewed: every multi-member cluster has a hand-written `label`, and Antwort received a manual override merge. Across 503 lemmas / 1,813 fine senses, this yields 1,236 coarse clusters, of which only **336 are multi-member** (i.e. an actual merge happened) and **900 are singletons** (no merge).

To produce *more merged clusters*, the threshold needs to rise. At `t=0.12` and `t=0.14` the number of multi-member clusters grows but so does the per-lemma manual review burden — by historical count, roughly 50–150 *new* clusters that would need human inspection. The bottleneck is reviewer time, not compute.

This document describes how an LLM judge can absorb most of that review load while remaining defensible in a thesis context.

---

## 2. Design

### 2.1 What the judge does — and does not do

| Role | Owned by |
|---|---|
| Propose candidate clusters | Agglomerative clustering on E5-large `core_gloss` embeddings (geometric) |
| Verify each candidate (APPROVE / REJECT + confidence) | **LLM judge** |
| Generate a coarse label for APPROVED clusters | **LLM judge** |
| Final sign-off on disagreements & low-confidence cases | Human reviewer |

**The judge never invents merges.** Anything that ends up in the output mapping was first proposed by the agglomerative geometry. This keeps the methodology auditable and the LLM in a strict verifier role.

### 2.2 Prompt structure

Frozen system prompt — defines the task as German lexicographic judgment ("would these senses appear under one entry in a learner's dictionary that distinguishes 3–5 senses per polysemous word?"), spells out APPROVE / REJECT criteria, defines a 1–5 confidence scale, and includes three worked examples (`schlagen` → APPROVE, `Bank` → REJECT, `spielen` → REJECT-but-borderline).

Per-cluster user message — lemma + each member's `synset_id`, GermaNet `class`, `core_gloss`, `llm_definition`, and the first two `llm_examples`. Two examples per sense is enough context without bloating the request.

The system prompt is held under top-level `cache_control: {"type": "ephemeral"}` so it caches once per 5-minute window across all per-cluster calls. (At the current prompt length the prefix is below Opus 4.7's 4096-token cache minimum; the marker silently no-ops until the prompt grows. It is left in place to make caching automatic if more examples are added.)

### 2.3 Output schema

Strict JSON via `output_config: {"format": {"type": "json_schema", "schema": ...}}`:

```json
{
  "decision": "APPROVE" | "REJECT",
  "confidence": 1..5,
  "justification": "one English sentence",
  "suggested_label": "2–5 word label" | null
}
```

The schema is enforced by the API, not by post-hoc parsing — invalid responses 400 instead of silently corrupting the audit log.

### 2.4 Model & sampling

- **Model:** `claude-opus-4-7`. This is a bounded but intelligence-sensitive judgment; the marginal cost of using Opus over Sonnet is ~$1 across the full calibration pass and the quality difference matters for borderline cases.
- **Adaptive thinking:** on. The judge benefits from a moment to weigh the senses before answering.
- **Effort:** default (`high`). Not set explicitly.
- **Sampling parameters:** none (Opus 4.7 removes `temperature` / `top_p` / `top_k`).
- **`max_tokens`:** 1024. Output is small; thinking gets the extra headroom.

---

## 3. Calibration protocol

The calibration pass is the cheap, defensible first step: it asks the judge to rule on every cluster the human reviewer already approved, and measures agreement.

### 3.1 Inputs

- `data/processed/germanet/sense_clusters_reviewed.json` (current canonical mapping, 336 multi-member clusters)
- `data/processed/germanet/senses_pilot.jsonl` + `data/evaluation/atemporal/input/sense_text_representations.jsonl` (sense definitions and examples)

### 3.2 Procedure

```bash
# Smoke test on five clusters first
python scripts/clustering/calibrate_llm_judge.py --limit 5

# Full pass — ~336 calls, runs ~10–25 minutes depending on rate limits
python scripts/clustering/calibrate_llm_judge.py

# Resume after interruption (uses lemma::cluster_id keys already in the output JSONL)
python scripts/clustering/calibrate_llm_judge.py --resume

# Re-print the summary without re-calling
python scripts/clustering/calibrate_llm_judge.py --report-only
```

Output goes to `data/processed/sense_clustering/llm_judge/calibration_v1.jsonl`. One JSON object per cluster:

```json
{
  "lemma": "schlagen",
  "cluster_id": "schlagen_c0",
  "members": ["s56941", "s56959", ...],
  "human_label": "physical strike",
  "judge": {
    "decision": "APPROVE",
    "confidence": 5,
    "justification": "...",
    "suggested_label": "physical strike",
    "model": "claude-opus-4-7-...",
    "latency_seconds": 4.2,
    "usage": { "input_tokens": ..., "output_tokens": ..., ... }
  }
}
```

### 3.3 What we measure

- **Agreement rate** — % of human-approved merges the judge also approves. Headline number for the thesis.
- **Confidence distribution** — how confident is the judge across the population? A long left tail of confidence-1/2 APPROVEs is an honest "I don't really know" signal worth flagging for human review.
- **Disagreements** — every REJECT is a row to inspect. Either the human merge is questionable (informative) or the judge has a systematic blind spot (informative for prompt iteration).
- **Token usage / cost** — for budgeting the full pipeline run.

### 3.4 Decision gate

| Outcome | Interpretation | Next step |
|---|---|---|
| ≥ 90% agreement, confidence median ≥ 4 | Judge is reliable. | Proceed to Phase 4 (use judge on new candidates from higher-threshold clustering). |
| 70–90% agreement | Usable as a triage filter, not as a sole approver. | Inspect REJECTs. Tighten prompt or accept that all judge-flagged cases need human review. |
| < 70% agreement | Judge is not aligned with our review standard. | Inspect failures, iterate the prompt criteria, re-run. Do not deploy as a filter yet. |

**The threshold for the production clustering is NOT tuned to maximize agreement** — that would be circular. The threshold is chosen on linguistic grounds (Phase 1 of the original clustering plan). The judge filters whatever the geometry proposes at that threshold.

---

## 4. Triage pipeline (Phase 4 — after calibration passes)

Once the calibration agreement number is acceptable, the full pipeline for raising the threshold looks like:

```
                                   ┌── unchanged: carry over human label
1. Compute candidates at t=0.12    │
   (compute_sense_clusters.py)  ───┼── new merge:
                                   │      ↓
2. Diff vs current mapping         │   judge APPROVE high-conf  → auto-accept
                                   │      ↓
                                   │   judge APPROVE low-conf   → manual queue
                                   │      ↓
                                   │   judge REJECT             → manual queue
                                   │      ↓
                                   │   manually decided clusters → reviewed mapping
```

**Effective manual workload:** if the calibration shows ~90% agreement, then for ~50–150 *new* clusters at `t=0.12`, only ~5–15 hit the manual queue. That is the reduction we are buying.

Code for this triage step is **not yet written** — it depends on the calibration result and is left for Phase 4. The two interfaces it will need are already in place:

- `judge_cluster()` in `src/wsd/llm_judge.py` (per-cluster verdict)
- `compute_sense_clusters.py` with `--threshold` (candidate generation)

---

## 5. Defensibility (thesis-relevant notes)

**What goes in the methodology section:**

1. The judge is a *verifier*, not a clustering algorithm. The geometry is unchanged.
2. The judge was calibrated against N manually-approved merges and showed X% agreement.
3. The judge's prompt, schema, and full per-cluster responses are stored in the repo (`data/processed/sense_clustering/llm_judge/calibration_*.jsonl`) — every decision is auditable.
4. Any cluster the judge rejected, expressed low confidence on, or disagreed with the geometric proposal on, was either resolved by a human reviewer or excluded.
5. The merge threshold was chosen on linguistic grounds before judge agreement was computed; agreement is a downstream sanity check, not a tuning objective.

**What does NOT go in the methodology section:**

- "We picked t=X because it maximized judge agreement." (Circular — don't.)
- "The LLM made the final call on borderline cases." (False — borderline cases route to humans.)

**Why this is more defensible than just trusting the geometric threshold:**

The geometry tells us *which senses are similar in embedding space*. The judge tells us *whether that similarity crosses a categorical lexicographic boundary*. These are different signals. Combining them is a stronger argument than either alone, and the judge's per-cluster justifications give every reported result an explanation a reader can evaluate.

---

## 6. Caveats & honest limits

- **Anchoring:** the worked examples in the system prompt anchor the judge's behavior. They were chosen to illustrate clear APPROVE / clear REJECT / borderline REJECT, not to bias toward either decision. If after calibration the judge is systematically too lenient or too strict, the prompt needs to be revised — and the calibration must be re-run.
- **Self-consistency:** Opus 4.7 is largely deterministic on this kind of bounded judgment, but with adaptive thinking on there is some run-to-run variance. The audit log makes this visible. Aggregate numbers over 336 calls smooth it out.
- **No tool use:** the judge cannot consult external resources (DWDS, secondary literature). It rules on the GermaNet description plus 2 examples per sense. This is a deliberate scope: we are testing whether the *information already in our pipeline* is sufficient for a coarse-grain decision.
- **Cost estimate:** ~336 calls × ~1.5–2k input tokens × $5/M + ~336 × ~150 output tokens × $25/M ≈ **\$3–5 for the full calibration pass**. Re-runs after prompt changes are the same. Not a constraint.
- **Antwort.** The Antwort manual override is a single merge decision the human reviewer made that the geometry did not propose. The calibration will check whether the judge approves it. This is informative either way.

---

## 7. File index

| File | Purpose |
|---|---|
| `src/wsd/llm_judge.py` | Prompt assembly + `judge_cluster()` API call |
| `scripts/clustering/calibrate_llm_judge.py` | CLI: run calibration over reviewed mapping, write JSONL |
| `data/processed/sense_clustering/llm_judge/calibration_v1.jsonl` | Per-cluster judge responses (created on first run) |
| `documentation/coarse_clustering_plan.md` | The original clustering plan; this doc is a Phase-4 addendum |
| `documentation/research_log.md` | Decision log; calibration results recorded there |

---

## 8. Immediate next action

Run the smoke test: `python scripts/clustering/calibrate_llm_judge.py --limit 5`. Inspect the five JSONL records. If the decisions, confidences, and labels look sensible, run the full pass without `--limit`.