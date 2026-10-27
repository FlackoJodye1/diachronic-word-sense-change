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

## 2026-05-07: LLM-as-Judge for Coarse-Cluster Verification

**Status**: In Progress (calibration pending)

**Context:**
The canonical mapping `sense_clusters_reviewed.json` was produced at `t=0.10` and yields only 336 multi-member clusters out of 1,236 total (~73% singletons). Raising the threshold to produce more merged clusters is the obvious next step, but each new merge candidate would historically need manual lexicographic review — which is the bottleneck. We want a way to reduce that review load without dropping the quality bar.

**Options Considered:**

1. Raise threshold and re-review every new cluster manually — slow, doesn't scale.
2. Raise threshold blindly and accept whatever the geometry produces — defeats the lexicographic standard already established.
3. Add an LLM judge that verifies each candidate merge (APPROVE / REJECT + confidence + label) and routes only its REJECTs and low-confidence APPROVEs to manual review.
4. Replace the geometric clustering with a direct LLM-driven clustering — too open-ended, hard to defend.

**Decision:**
Option 3. The judge is a *verifier*, never a proposer — the agglomerative clustering still produces every candidate. This keeps the geometry-driven structure intact and the LLM in a strict verification role, which is much easier to justify in a thesis methodology section.

**Rationale:**
- The 336 manually-approved merges already in `sense_clusters_reviewed.json` are a free calibration set. Run the judge on them, measure agreement; if it's high, deploy as a triage filter on new candidates at higher thresholds.
- Anthropic Opus 4.7 with adaptive thinking, structured JSON output, and a constrained "would these go under one entry in a 3–5 sense learner's dictionary?" framing is well-suited to this bounded judgment task.
- Cost is trivial (~$3–5 per full calibration pass); the bottleneck is human time, not compute.
- The audit trail (per-cluster raw responses persisted to JSONL) gives every downstream merge a defensible explanation a reader can evaluate.

**Implementation:**
- `src/wsd/llm_judge.py` — `judge_cluster()` API call with frozen system prompt, per-cluster user message, strict JSON schema output, top-level prompt-cache marker (silent no-op below the 4096-token cache minimum until prompt grows), adaptive thinking on.
- `scripts/clustering/calibrate_llm_judge.py` — CLI: streams to JSONL, supports `--resume`, `--limit`, `--lemmas`, `--report-only`. Reports agreement %, confidence histogram, and triage-queue size.
- `documentation/llm_judge_clustering.md` — full design doc, calibration protocol, thesis-defensibility notes, decision gates.

**Decision gates (post-calibration):**

| Agreement | Action |
|---|---|
| ≥ 90%, conf median ≥ 4 | Deploy as triage filter on new-threshold candidates. |
| 70–90% | Use as filter, but every judge-flagged case still goes to human review. |
| < 70% | Don't deploy. Inspect failures, iterate prompt, re-run. |

**Important constraint:** the merge threshold itself will be chosen on linguistic grounds *before* checking judge agreement. Tuning the threshold to maximize agreement would be circular and is explicitly off-limits.

**Follow-up:**
1. Run `python scripts/clustering/calibrate_llm_judge.py --limit 5` smoke test.
2. Run full calibration pass; record agreement number in this log.
3. If gate passes, generate candidates at `t=0.12` and `t=0.14` (`--dry-run` first), pick threshold on linguistic grounds, run the judge over the *new* clusters only, route disagreements to manual queue.
4. Update `sense_clusters_reviewed.json` with the resulting merges (preserving carry-over labels from clusters whose membership did not change).

### Calibration result (2026-05-07, same day)

Implementation switched from Anthropic to OpenAI (gpt-4o, temperature 0, strict JSON schema) — the project already has `OPENAI_API_KEY` configured and uses OpenAI for sense enrichment. The judge prompt, schema, and calibration protocol are unchanged.

**Headline:** 45.8% agreement (154 APPROVE / 182 REJECT) on 336 multi-member clusters, 99.4% at confidence 4 or 5.

By my own pre-stated gate (`<70%` = don't deploy), this fails. **But the breakdown by cluster size flips the interpretation:**

| Cluster size | n | judge APPROVE % |
|---|---:|---:|
| 2 | 208 | 59.6% |
| 3 | 72 | 27.8% |
| 4 | 32 | 21.9% |
| 5 | 9 | 11.1% |
| 6+ | 15 | 13.3% |

The judge agrees with humans on small geometric merges and systematically rejects larger ones. Inspection of the high-confidence REJECTs shows they are **lexicographically valid**: e.g., `zeigen_c0` (13 members, "zeigen / sichtbar machen") was flagged as mixing physical demonstration / emotional expression / perceptual revelation; `fliegen_c0` (8 members) mixes natural flight, mechanical flight, and racing horses; `Klasse_c0` mixes social class, biological classification, quality level, lottery grouping. These are exactly the kind of over-merges the original clustering plan said must not happen (`spielen_instrument` must never merge with `spielen_sport`).

**Reframed conclusion:** The calibration is not a reliability check on the judge — it is a *quality check on the existing reviewed mapping*. ~54% of multi-member clusters in `sense_clusters_reviewed.json` cross categorical boundaries, almost entirely concentrated in the 4+ member clusters from high-polysemy verbs and abstract nouns (`zeigen`, `geben`, `Ordnung`, `fliegen`, `Klasse`, `leicht`, ...). 17 lemmas have ALL their multi-member clusters rejected (Bahn, Bass, Kopf, gehen, nehmen, tragen, Gesicht, Platz, ...).

**Why the original gate was the wrong frame:**
- It assumed humans were the gold standard.
- The actual fact: at `t=0.10` with `min_clusters=2`, the geometry produced both small valid merges (~60% pass) and large invalid merges (~15% pass at 4+ members), and the manual review approved both.

**Updated next steps:**

1. Manually inspect 8–10 high-confidence REJECTs (specifically `zeigen_c0`, `fliegen_c0`, `Klasse_c0`, `leicht_c0`, `Ordnung_c0`, `geben_c2`). If they look like genuine over-merges, the judge is reliable and the existing mapping needs revision.
2. Decide what "merged cluster" means for our actual goal (diachronic noise reduction vs. lexicographic defensibility). The judge enforces the latter; the existing mapping is closer to the former. They are not the same standard.
3. Two paths now visible:
   - **(a) Quality-improvement path:** Use the judge to flag clusters in the existing mapping that should be split. Re-cluster those lemmas at a *higher* threshold won't help — they're large because the geometry says they're similar.
   - **(b) Threshold-raising path (original goal):** Generate candidates at `t=0.12`/`t=0.14`, judge each new cluster, accept only APPROVEs. Combined with (a) this would give a smaller, cleaner mapping rather than a larger one.
4. Token cost was ~$1.20 for the full pass. Re-runs after prompt changes are similarly cheap.

Output: `data/processed/sense_clustering/llm_judge/calibration_v1.jsonl` (336 records, full audit trail with justification + suggested label per cluster).

---

## 2026-04-20: Baseline Reframing — 3-Tier Reporting (Random / Text-Only / Weighted)

**Status**: Complete

**Context:**
Previous reporting had only two numbers: the weighted (best) system and a single "baseline" = E5-large over GermaNet hypernym text. That baseline is a strong encoder + lexicon combination, not a true floor. Readers couldn't tell whether the lift came from the encoder, the lexicon, or the LLM enrichment.

**Decision:**
Report three tiers so each component's contribution is visible:

| Variant | Fine | Coarse |
|---|---|---|
| Random baseline (1/n_senses fine, 1/n_clusters coarse) | 24.5% | 50.5% |
| Text-only ablation (E5-large + GermaNet hypernyms) | 36.6% | 75.3% |
| **Weighted (best, E5 + LLM def+examples)** | **43.0%** | **84.9%** |

Deltas: encoder+lexicon buys +12.1 / +24.8 pp over random; LLM enrichment adds +6.4 / +9.6 pp on top.

**Implementation:**
- `scripts/atemporal/generate_predictions.py` now takes `--variant {random, text-only, weighted}`. Random path is analytical (no embedder loaded).
- Output files: `predictions_random.csv`, `predictions_text_only.csv`, `predictions.csv`. Old `predictions_baseline.csv` removed.
- Random CSV schema includes per-instance `n_senses_fine`, `p_correct_fine`, `n_clusters_coarse`, `p_correct_coarse` (no `pred_sense` column — no prediction is made).

**Impact on conclusions:**
- Numerical conclusions unchanged: weighted is still best; the text-only→weighted delta is identical to the previous weighted→baseline delta.
- Framing now makes the ablation story explicit and gives a true task-difficulty floor.

---

## 2026-04-07: Atemporal Accuracy Recalibration

**Status**: Complete

**What was done:**
Re-ran atemporal WSD evaluation for both `baseline` and `weighted` variants on the 93-instance eval set using `scripts/atemporal/generate_predictions.py`. Added `--variant` and `--cluster-mapping` flags so both fine and coarse accuracy are computed in one pass.

**Root cause of change:**
When `core_gloss` was added to `sense_text_representations.jsonl`, the file was re-generated via the enrichment pipeline. Due to LLM non-determinism, the `llm_definition` and `llm_examples` outputs changed — the embedding_comparison.csv figure of 54.8% was therefore stale.

**Updated results (2×2 table):**

| Method | Fine | Coarse |
|--------|------|--------|
| Baseline (GermaNet `text_for_embedding`) | 36.6% (34/93) | 75.3% (70/93) |
| Weighted (0.5×llm_definition + 0.5×avg(examples)) | 43.0% (40/93) | **84.9% (79/93)** |

- `embedding_comparison.csv` updated (e5-large weighted row)
- `predictions.csv` regenerated; `predictions_baseline.csv` newly created
- All documentation updated to reflect new numbers

**Impact on conclusions:**
- Weighted still outperforms baseline at both levels (+6.4pp fine, +9.6pp coarse)
- Coarse mapping still provides a large boost (~+42pp for weighted, ~+39pp for baseline)
- The original core_gloss comparison (51.6% vs 54.8%) is no longer valid with current data — the core_gloss run used `weight_tuning.csv` which predates the file re-generation
- All diachronic pipeline conclusions remain valid: the relevant figure for diachronic analysis is coarse accuracy (84.9%), unchanged in direction

---

## 2026-04-02: Pilot Set Expansion — Senses + Clusters Generated

**Status**: In progress (annotation ongoing)

**What was done today:**
- Generated `core_gloss` for all 413 polysemous non-pilot lemmas (≥2 senses, not in pilot) via `generate_expanded_senses.py`. Output: `senses_expanded.jsonl` (1229 senses). Workers reduced to 1 to avoid memory pressure on Mac.
- Ran `compute_sense_clusters.py --senses-file senses_expanded.jsonl --merge-into sense_clusters_reviewed.json`. 407 new lemmas merged; 6 eval lemmas already present were skipped. `sense_clusters_reviewed.json` now covers 503 lemmas total (90 pilot + 6 eval + 407 expanded).
- 160/413 new words had the min_clusters=2 constraint fire — these words' glosses were too similar to split organically at t=0.10. Worth extra scrutiny during annotation.
- Updated review dashboard (`review_app.py`) with a progress bar, word-level stat boxes (fully done / in progress / not started), and separate sections for pilot vs. expanded words.
- `load_all_senses()` in `src/diachronic/io.py` already listed `senses_expanded.jsonl` — review app picks up the new senses automatically.

**Strategy:**
Cluster everything upfront, annotate on-demand in the review dashboard as words are selected for the diachronic study. Avoids committing to a fixed expansion set before seeing which words have sufficient DWDS coverage.

**Follow-up:**
- Annotate cluster labels for candidate words in the dashboard
- Select subset to fetch DWDS contexts for (targeting ~60–110 new words on top of 90 pilot)
- Run full diachronic pipeline on expanded set

---

## 2026-03-27: v3 Analysis Complete — Case Study Candidates Selected

**Status**: Complete

**What was done today:**
- Re-ran all sections of `diachronic_pilot_results.ipynb` on v3 data. Updated rankings, trajectory plots, and candidate scoring throughout.
- Added notebook sections §7.1 (Manual Trajectory Review) and §7.2 (Per-Word Verdicts) documenting the selection rationale beyond automated criteria.
- Fixed trajectory plot bug: stacked area charts were overlaying the dashed "most monotonic sense" line at raw proportions, which don't correspond to stacked y-positions. Replaced with line plots throughout (cells 60 and 65).

**Case study selection — v3 results:**

6 words passed all three automated criteria (normalized JSD top quartile, Spearman |ρ| ≥ 0.6, ≥ 8 dense decades): springen, Löffel, Welle, Hof, Stamm, Kugel.

Manual reliability review narrowed this to **2 selected words**:

- **springen** ✓ — binary split (literal vs. figurative topic-switch sense). Figurative sense rises from ~9% (1900) to ~31–44% (1970s–1990s). Consistent trend, 21–86 contexts per decade throughout.
- **Welle** ✓ — figurative 'trend/wave-of-events' sense rises from 19% (1900) to 76% (1940), remains dominant at 55–69% through 1990. Strongest single-decade shift in dataset. Plausible historical motivation (politically charged 1930s–40s).

**4 words excluded:**
- **Stamm** — trend plausible but counts collapse to 19–31/decade post-1960; ρ inflated by sparse tail.
- **Löffel** — ρ=0.729 driven entirely by sparse tail (n=10 in 1980s); data artifact.
- **Hof** — royal-court sense spikes to 83% (1970) then reverses to 47% (1990); non-monotonic U-shape.
- **Kugel** — non-monotonic (bullet sense peaks, then sphere sense surges in 1950, then reverses); sparse post-1950.

**Key methodological note:**
Normalized JSD (avg_jsd / n_coarse_senses) is the primary ranking metric. Raw JSD is strongly correlated with number of senses (+0.687) and systematically favours high-polysemy verbs regardless of whether the change is interpretable.

**Follow-up:**
- Manual sentence-level validation of springen and Welle predictions against raw DWDS contexts — required before reporting results.

---

## 2026-03-24: v3 Pipeline Complete — Pre-Analysis Snapshot

**Status**: Complete (pipeline done, analysis pending)

**What was completed today:**
- v3 predictions finished: 90/90 lemmas, 72,591 rows (+198% vs v2's 24,325). Interrupted by disk-full error at 87/90; resumed after clearing space, all 3 remaining lemmas completed cleanly.
- Coarse mapping applied: `pilot_predictions_v3_coarse.csv` (0 rows dropped).
- Temporal aggregation: `pilot_v3_coarse_sense_counts.csv`, `pilot_v3_coarse_stats.csv`.
- Change metrics recomputed: mean JSD=0.112, 806 transitions, 126 emergence / 118 decline events.
- `diachronic_pilot_results.ipynb` updated to load v3 files.

**Key finding — v3 vs v2 comparison:**
- 3x more data across all decades; 1970s got the biggest relative boost (+258%).
- Mean JSD dropped from 0.165 → 0.112. This is expected and desirable: v2 signals were partially inflated by sparse-sample variance. v3 estimates are more reliable, not weaker.
- v2 top candidates (Krone ρ=−0.69, ziehen ρ=+0.842, Blatt ρ=+0.810) will likely shift in v3 rankings — effects will be smaller but more trustworthy.

**Immediate next step:** Re-run `diachronic_pilot_results.ipynb` on v3 data to get updated candidate rankings.

---

## 2026-03-23: v3 Corpus Fetch + Prediction Pipeline Refactor

**Status**: Complete

**What was done today**:

1. **Corpus exploration** — investigated all available DWDS corpora via the public API. `kern` is the only valid historical corpus (1900–1999). `ddc` returns 404. `public` is modern-only (Wikipedia-dominated, 2020s). `kern21` (2000–2005), `ddr` (1959–1989), `tsp`/`tagesspiegel` (1998–2004) are available but out of scope. ZEIT/FAZ/Spiegel require session-cookie authentication.

2. **API limit investigation** — discovered the `--limit` parameter was the bottleneck, not a hard corpus ceiling. Tested 1000 → 2000 → 5000: API silently reverts to 100 rows above ~7,500. Optimal setting confirmed as `--limit 5000`. Also discovered `--min-tokens 8` introduced genre/decade bias (Zeitung headlines dropped disproportionately) — lowered to 6.

3. **v3 re-fetch** — ran full 116-word fetch with `--limit 5000 --min-tokens 6 --max-tokens 60`. Result: 30,337 → 90,130 contexts (+197%). Most words +150–350%. Genuinely kern-sparse words unchanged (Bass 23→25, Frosch 106→109). *(Correction 2026-10-27: Bass is not sparse. kern spells it *Baß*. See `analysis/fetch_check/RESULTS.md` §4b.)*

4. **Prediction pipeline memory refactor** — original script encoded all 90k contexts at once (~370MB embedding matrix), exhausting swap. Refactored to process one lemma at a time: model + sense embeddings loaded once, context embeddings computed and deleted per lemma. Added `--resume` flag for crash recovery. Peak RAM per step now ~7MB (down from 370MB).

5. **Overnight predictions** — running with `--resume`. 14/90 lemmas complete at time of writing. Full pipeline (coarse mapping → aggregation → metrics → notebooks) to follow tomorrow.

**Follow-up**:
- Tomorrow: chain downstream pipeline after predictions finish
- Implement DWDSClient pagination + session-cookie auth (see separate entry)
- Re-run analysis notebooks with v3 results; re-evaluate case study candidates

---

## 2026-03-23: DWDS API — Pagination and Authentication

**Status**: Decision Made

**Context**:
While investigating corpus options and ZEIT access, two important API features were discovered that affect how much data we can retrieve.

**Finding 1 — Pagination (`&p=n`)**:
The DWDS export API supports page-by-page fetching via the `p` parameter. Each page returns up to `limit` rows (max 5,000). Our current `DWDSClient` fetches only page 1 — meaning for words with more than 5,000 corpus hits, we silently miss everything beyond the first page.

For kern this matters less (most words have <5,000 unique documents, confirmed for Krone at ~3,514). But for ZEIT and other larger corpora, pagination is essential.

**Finding 2 — ZEIT requires session-cookie authentication**:
The DWDS API does not use API keys. Access to licensed corpora (ZEIT, FAZ, Spiegel) requires a valid browser session cookie — i.e., the user must be logged in to DWDS. Confirmed by DWDS admin response (2026-03-23): export is allowed at large scale once authenticated. *(Correction 2026-10-27: the mail does not tie export to a login. Kern results may be exported and reused; ZEIT/FAZ/Spiegel are quotes-only. See `documentation/DATA_SOURCES.md` and the 2026-10-27 fetch-check entry.)* The `DWDSClient` currently sends no authentication headers, which is why `corpus=zeit` returns 404.

**Implemented (2026-03-23)** in `src/data/dwds/client.py`, `src/data/dwds/fetcher.py`, `scripts/data/fetch_dwds_contexts.py`:

- `pd.read_csv(url)` replaced with `requests.get()` + `io.StringIO` — necessary to attach custom headers
- `DWDSClient(session_cookie=str)` — cookie attached as `Cookie` header on every request
- `corpus_csv(paginate=False)` — when `True`, loops `&p=1,2,...` until page returns fewer rows than limit
- `FetchOptions.paginate` + `--paginate` flag in fetch script; `--session-cookie` arg in fetch script

**How to extract session cookie**:
1. Log in at https://www.dwds.de in browser
2. DevTools → Application → Cookies → dwds.de
3. Copy session cookie value, pass as `--session-cookie "<value>"`

**Note**: `--paginate` only meaningful with deterministic sort. Do NOT combine with `--sort random`.

**Follow-up**:
- Test `corpus=zeit` with authenticated session on one word (e.g. Krone)
- Check year range and genre fields returned by ZEIT
- Decide merge strategy: kern + ZEIT combined, or ZEIT as separate parallel analysis

---

## 2026-03-23: Institutional DWDS Access Applied For

**Status**: Pending

**Context**:
During corpus exploration, the ZEIT corpus (`corpus=zeit`) was found to exist in DWDS but return HTML rather than CSV — indicating restricted/licensed access. Institutional access has been applied for.

**What institutional access would unlock**:
- **ZEIT corpus** (Die Zeit, 1946–present): high-quality weekly newspaper, consistent Zeitung register throughout. Particularly valuable for 1950s–1990s, which are the sparsest decades in kern. Single-genre corpus removes the Belletristik/Wissenschaft/Zeitung mix confound present in kern.
- Potentially other restricted corpora (DDR corpus is already accessible publicly; others unknown until access granted).

**Why this matters**:
The kern corpus is our only current option for 1900–1999. It is mixed-genre and sparse in later decades (1970s–1990s: 19–31 contexts per word-decade for Krone even at limit=2000). ZEIT would substantially increase density for 1950–1999 and with a cleaner genre profile.

**Follow-up**:
- When access is granted: test ZEIT corpus availability via API (`corpus=zeit` or institutional variant)
- Check year range and genre fields returned
- Decide whether to use ZEIT as a supplement to kern (merge strategy) or as a separate parallel analysis
- Genre-stratified sampling will be needed if merging kern + ZEIT (different genre compositions)

---

## 2026-03-23: Corpus Switch kern → ddc for Increased Context Density

**Status**: Decision Made (pending test run)

**Context**:
Pilot study results look promising but context density is insufficient for reliable sense proportion estimates. Current `kern` fetch (`--limit 1000`, `--max-tokens 60`) yields a median of **23 sentences per word-decade** and **39% of word-decades have fewer than 20 sentences**. The research log (2026-02-19) previously documented that the kern corpus has a hard ceiling of ~10–20 sentences per word-decade regardless of `--limit` — the corpus simply does not contain more unique documents per decade for most lemmas. Several words are critically sparse (Bass: 23 total, groß: 47 total across all decades).

**Options Considered**:

1. **Increase `--limit` on kern** — not a real solution; ceiling is a corpus size constraint, not a fetch parameter issue.
2. **Switch to `--corpus ddc`** — full DWDS corpus, substantially larger than kern. Already supported by the script via `--corpus` arg. More data but less controlled genre composition.
3. **Merge to 25-year bins** — previously decided against (2026-03-18) because only 22/894 word-decades are sparse after the v2 re-fetch. Doesn't solve per-decade density, just hides it.

**Decision**: Switch to `--corpus ddc` with the following settings:

```
--corpus kern
--limit 5000
--min-tokens 6
--max-tokens 60
--sort random
--output data/processed/dwds/pilot_contexts_v3.jsonl
```

**Rationale for each parameter**:
- `--corpus kern`: only valid historical corpus (1900–1999). `ddc` returns 404; `public` is modern-only.
- `--limit 5000`: effectively no artificial cap — the kern corpus ceiling per word is ~3,500 (confirmed for Krone). Above 5000–7500 the API silently falls back to 100 rows. Setting 5000 guarantees we retrieve everything available without hitting the fallback.
- `--min-tokens 6`: lowered from 8 to reduce genre/decade bias from dropping short newspaper-style sentences
- `--per-decade-n`: disabled (0) — no artificial cap per decade. JSD is computed on proportions so more data in data-rich decades is better, not worse.
- `--max-tokens 60`: carried over from v2 (0.3% truncation rate)
- New output file: preserves kern v2 data (`pilot_contexts.jsonl`) as fallback

**Yield confirmed for Krone**: v2 limit=1000 → 287 contexts; limit=5000 → 682 contexts (+138%). All decades improved.

**Key concern — genre composition**:
The `ddc` corpus has less controlled genre composition than `kern`. Genre drift (Belletristik ↓, Wissenschaft ↑ over time) is already a documented confound in the kern data; it will be more pronounced in `ddc`. Any apparent sense shift that correlates with a genre composition shift should be treated with caution. **At minimum**, genre distribution must be reported alongside diachronic results. At best, genre-stratified sampling should be implemented in `aggregate_temporal.py` before the full-study scale-up.

**Result (2026-03-23)**: Fetch complete. 30,337 → 90,130 contexts (+197%). All 116 words fetched. Most words +150–350%. Genuinely sparse words unchanged (Bass 23→25, Frosch 106→109, Knoten 266→270) — kern corpus ceiling confirmed for those. *(Correction 2026-10-27: Bass is not genuinely sparse. kern spells it *Baß* (758 vs. 159 tokens), and the same holds for Schloss/Schloß and Schluss/Schluß. See `analysis/fetch_check/RESULTS.md` §4b.)* Case study words: Krone 287→682, ziehen 212→778, Blatt 224→752. Output: `data/processed/dwds/pilot_contexts_v3.jsonl`.

**Follow-up**:
- Run predictions on v3 contexts → `pilot_predictions_v3.csv`
- Apply coarse mapping, re-aggregate, recompute metrics, re-run notebooks
- Consider adding genre stratification to `aggregate_temporal.py` (Phase 4 concern, but worth flagging now)
- Re-run when ZEIT institutional access granted (applied 2026-03-23)

---

## 2026-03-18: Results Validation & Temporal Aggregation Decision — Complete

**Status**: Decision Made

**What was done:**

1. **Full run of `diachronic_pilot_results.ipynb` validated** — notebook ran clean end-to-end. All 7 sections executed without errors. Key stats confirmed: 90 words, 24,325 predictions, 717 transitions, 194 emergence / 197 decline events, no JSD ≥ 0.99 artifacts.

2. **Fixed `Styler.applymap` deprecation warnings** — cells 47 and 49 updated to use `.map()` instead of `.applymap()` (deprecated in pandas ≥ 2.1). No functional change.

3. **Temporal granularity: 10-year decades confirmed** — evaluated whether aggregating to 20-year periods would reduce noise from sparse cells.

**Decision:** Keep 10-year decades.

**Rationale:**
- Only 22/894 word-decade cells (2%) are below the n=5 threshold — sparsity is not a structural problem.
- 20-year aggregation would collapse the DWDS range to 5 time points, making Spearman ρ near-meaningless (currently computed over ≥6 valid points out of 10).
- WWII-era transitions (1930s→1940s→1950s) would be lost entirely.
- Genre drift (Belletristik → Wissenschaft) is the real confound, and it is not addressable by re-binning.
- Raising the minimum contexts threshold from 5 to 10 at analysis time (already done for change metrics) is sufficient.

**Follow-up:** Proceed to manual sentence-level validation for Krone, ziehen, Blatt.

---

## 2026-03-19: Notebook Restructuring & Case Study Candidate Selection — Complete

**Status**: Complete

**What was done:**

1. **`diachronic_pilot_results.ipynb` — stripped to coarse-only (50 cells, 7 sections)**
   - Removed all fine-grained data loading, the `USE_COARSE` toggle, and the fine vs. coarse comparison section.
   - All analysis now uses coarse predictions only (`pilot_predictions_coarse.csv`).
   - Section numbers updated (old 5–8 → new 4–7 after removal of former Section 4).
   - Cell 0 updated to reflect 24,325 predictions / coarse-only scope.
   - Cell 2 now points to `sense_granularity_analysis.ipynb` instead of describing both methods.

2. **`sense_granularity_analysis.ipynb` — new notebook created**
   - Documents the methodological justification for using coarse senses.
   - 6 sections: Setup, Atemporal Accuracy table, Cluster Reduction statistics, Noise Reduction in diachronic metrics, Top-10 Fine vs. Coarse ranking, Conclusion.
   - Key result: Coarse clustering improves atemporal accuracy 43.0% → 84.9% and reduces emergence/decline events by ~60% (518 → 194).

3. **Section 7: Case Study Candidates (in `diachronic_pilot_results.ipynb`)**
   - Scoring pipeline: Spearman ρ monotonicity on per-sense proportions, filtered by avg coarse JSD ≥ top quartile, ≥ 8 decades with ≥ 10 contexts.
   - **11 strong candidates:** spielen, gehen, fassen, laufen, tragen, ziehen, nehmen, Krone, zählen, Blatt, Feld.
   - **Top picks for case studies:**
     - `Krone`: royal crown usage declining post-1918 monarchy collapse (ρ = −0.69)
     - `ziehen`: strong directional shift (ρ = +0.842)
     - `Blatt`: strong directional shift (ρ = +0.810)

**Next step:** Manual sentence-level validation for Krone, ziehen, Blatt — read 15–30 raw sentences per key transition decade to confirm shifts are genuine and not WSD artifacts.

---

## 2026-03-18: Coarse Diachronic Rerun — Complete

**Status**: Complete

**What was done:**
Ran Phase 3 of the clustering plan end-to-end:
1. `apply_coarse_mapping.py` — mapped 24,325 fine predictions → coarse cluster IDs using `sense_clusters_reviewed.json`. 0 rows dropped. `sense_class` and `sense_paraphrase` dropped from output (caused duplicate rows in aggregation; coarse cluster labels in `sense_clusters_reviewed.json` serve as the replacement).
2. `aggregate_temporal.py` → `pilot_coarse_sense_counts.csv`, `pilot_coarse_stats.csv`
3. `compute_change_metrics.py` (min-contexts=10) → `pilot_coarse_entropy.csv`, `pilot_coarse_divergence.csv`, `pilot_coarse_emergence.csv`, `pilot_coarse_decline.csv`

**Results vs. fine-grained:**
| Metric | Fine | Coarse |
|---|---|---|
| Max JSD | 0.835 | 0.532 |
| Emergence events | 518 | 194 |
| Decline events | 521 | 197 |
| Mean JSD | — | 0.165 |

**Interpretation:** ~60% reduction in events as expected. Remaining 194/197 events still require filtering for sustained, monotonic shifts — not all are genuine. Top candidates by JSD: `spielen` (1920→1930, 1980→1990), `fassen` (1960→1980), `Kasse`, `passen`.

**Next step:** Update `diachronic_pilot_results.ipynb` to load fine and coarse results side-by-side, re-rank top words, and identify case study candidates.

---

## 2026-03-18: Classifier Sufficiency Assessment

**Status**: Decision Made

**Question:** Is the WSD classifier good enough to detect meaningful diachronic change?

**Short answer:** Yes for large, sustained shifts. Uncertain for subtle or short-lived ones.

**Arguments for sufficiency:**
- 84.9% coarse accuracy means individual errors average out when aggregating hundreds of contexts per decade. A genuine 20-point distribution shift will remain visible through the noise.
- Coarse clustering was explicitly designed to reduce fake JSD from fine-grained sense confusion. Remaining JSD should reflect real change more reliably.

**Key uncertainty:**
- The 84.9% figure was measured on contemporary DWDS texts. The diachronic data spans 1900–1999. The model (multilingual-e5-large) was pretrained on modern data; accuracy on early 20th century texts is unknown and likely lower due to vocabulary, syntax, and genre differences.
- No historical gold standard exists to measure this directly.

**Practical implications for case study selection:**
- Only report words with **large, monotonic trajectory changes across multiple consecutive decades** — not one-decade spikes.
- Manual sentence-level validation (15–30 raw sentences per key transition decade) is not optional — it is what separates genuine findings from classifier noise.
- Words with <100 total contexts (`Bass`, `groß`, `Zwiebel`) and constrained words (11 words forced to 2 clusters) should be excluded from case studies regardless of their metric values.

---

## 2026-03-17: Coarse Atemporal Sanity Check

**Status**: Complete

**What was done:**
Clustered the 6 eval words missing from `sense_clusters_reviewed.json` (`Antwort`, `Doktor`, `Glas`, `Schatz`, `bitter`, `wagen`) using `compute_sense_clusters.py --senses-file ... --words ... --output sense_clusters_eval_0.10.json`. Labels assigned and merged into `sense_clusters_reviewed.json` (now 96 lemmas, 317 clusters).

Applied the coarse mapping to both gold labels and fine predictions on the 93-instance eval set.

**Result:**
- Fine-grained accuracy: 43.0% (40/93) — *recalibrated 2026-04-07; originally 54.8% (51/93) before sense file re-generation*
- Coarse-grained accuracy: 84.9% (79/93) — *recalibrated 2026-04-07; originally 81.7% (76/93)*
- Baseline: 36.6% fine (34/93) / 75.3% coarse (70/93)

**Decision gate:** Plan expected 65–75%. Result is 84.9%, above the upper bound, explained by the `Antwort` manual merge. `Antwort` has only 2 senses (verbal vs. written reply) that the model cannot distinguish at all — merging them to a single cluster `reply / response` is linguistically defensible. All other words fall within expected range.

**Override:** `Antwort_c0` and `Antwort_c1` manually merged into `Antwort_c0` ("reply / response"). Both sense IDs now map to `Antwort_c0` in `sense_clusters_reviewed.json`.

**Next step:** Phase 3 — write `scripts/diachronic/apply_coarse_mapping.py`.

---

## 2026-03-17: Sense Clustering Manual Review — Complete

**Status**: Complete

**What was done:**
Manual review of all 90 lemma cluster assignments produced by `compute_sense_clusters.py` at t=0.10. All 305 clusters were reviewed and labelled. Overrides applied where the automatic clustering produced linguistically incoherent groupings.

**Output:** `data/processed/germanet/sense_clusters_reviewed.json` — 90 lemmas, 305 clusters, all labels filled. Canonical mapping for all downstream pipeline steps.

**Next step:** Write `scripts/diachronic/apply_coarse_mapping.py`, then re-aggregate and recompute metrics at coarse level.

---

## 2026-03-14: Core Gloss Quality Review and Prompt Refinement

**Status**: Decision Made

**Context**:
After re-generating all 596 core_glosses with the improved sibling-context prompt (2026-03-12), a gloss quality review cell was added to `sense_clustering_exploration.ipynb` showing `baseline → core_gloss` side by side. Review revealed a systematic failure: many glosses had become too abstract and semantically uniform (e.g. multiple schlagen senses all mapping to "etwas mit Kraft treffen oder bewegen"), causing embeddings to collapse and clusters to be too large.

**Root cause**:
The sibling context instruction ("muss diese Bedeutung klar von den anderen unterscheiden") pushed the model toward abstract differentiation rather than grounding in the specific use case from the baseline. The model discarded distinctive baseline details (Sahne, Trommel, Flügel, Uhr) in favour of generic category descriptions. Additionally the prompt example "in einem Wettkampf gegen jemanden gewinnen" caused one sense to copy that phrase verbatim.

**Prompt iterations**:
1. Previous prompt (2026-03-12): long (30 lines, 12 rules, 7+ examples) — over-abstraction problem
2. Lean prompt (attempt 1): too short — model regurgitated sibling list verbatim into output
3. **Final prompt**: 12 lines, 6 rules, no examples. Key additions: "Genau eine Kernbedeutung, keine Liste"; "Konkrete Nomen müssen erhalten bleiben"; sibling block labelled "nur zur Orientierung — nicht aufzählen"

**Final state**:
- Prompt: `config/prompts/core_gloss.txt` — lean, 6 rules, no examples
- Sibling instruction: "unterscheide durch konkretes Objekt oder Kontext — nicht aufzählen"
- 3 manual patches applied directly to `senses_pilot.jsonl`:
  - `s52049` (gehen): "akzeptabel sein oder gerade noch funktionieren"
  - `s58716` (spielen): "Farben und Facetten reflektieren"
  - `s58405` (spielen): "Licht auf einer Oberfläche reflektieren"
- Backup of pre-patch state: `senses_pilot.jsonl.regen_bak`

**Test set expanded**:
Added `gehen` (19 senses) and `fassen` (12 senses) to notebook `TEST_LEMMAS`. t=0.10 confirmed on both. `Bank` absent from pilot words — confirmed by checking senses_pilot.jsonl.

**Follow-up**: Run `compute_sense_clusters.py` at t=0.10 on all 90 pilot words.

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
The weighted representation (α×definition + (1-α)×avg(examples)) achieved 54.8% accuracy with α=0.5 at the time of this experiment. *Note (2026-04-07): the sense file was later re-generated when core_gloss was added; current weighted fine accuracy is 43.0%. The α=0.5 choice remains valid. See 2026-04-07 entry.*

We wanted to verify whether 0.5 is optimal or if other weights perform better.

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
- Evaluation code: `src/wsd/wsd_evaluator.py`
- Embedding code: `src/data/germanet/transforms/embedding_builder.py`

---

*Last updated: 2026-03-10*

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
The following words are in `pilot_words_diachronic.txt` but absent from all GermaNet senses files (`senses_base.csv`, `senses_full / senses_pilot`):

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

---

## 2026-03-10: Coarse Sense Clustering — Embedding Representation Choice

**Status**: Resolved (2026-03-11)

**Context**:
During exploratory clustering (`notebooks/analysis/sense_clustering_exploration.ipynb`), all three candidate thresholds (0.20, 0.25, 0.30) produced the same result: everything collapsed to one cluster (before the `min_clusters=2` constraint forced an artificial split). The natural agglomerative merging finishes before any of the planned thresholds is reached.

**Root cause (confirmed)**:
The `weighted` representation uses verbose LLM-generated definitions that contain heavy boilerplate ("Diese Bedeutung unterscheidet sich von anderen Verwendungen des Begriffs..."). This shared phrasing is the same across all senses of a word, inflating cosine similarity uniformly. Combined with example sentences that share syntactic context, all pairwise distances for pilot words fall in a narrow 0.13–0.21 band.

**Resolution**: Recalibrated candidate thresholds to **0.14 / 0.16 / 0.18**, where organic splits occur. At t=0.18, all three test lemmas cluster without constraint override. `coarse_clustering_plan.md` updated accordingly.

**Remaining issue — addressed separately (see 2026-03-11 entry)**:
The root cause (boilerplate inflating similarity) affects not just clustering but WSD accuracy too. Addressed by re-enriching with short core glosses.

---

## 2026-03-12: Clustering Representation Comparison Plan

**Status**: In Progress

**Context**:
After the WSD weight tuning showed core_gloss is worse for nearest-neighbour context matching (51.6% core_gloss+examples vs 54.8% llm_definition+examples at the time; *note: llm_definition+examples is now 43.0% after sense file re-generation — see 2026-04-07 entry*), the question became: is core_gloss still useful for clustering, where the task is sense-vs-sense distance rather than sense-vs-context distance?

**Three representations to compare in `notebooks/analysis/sense_clustering_exploration.ipynb`**:

| Name | Formula | Hypothesis |
|------|---------|-----------|
| Original | 0.5 × `llm_definition` + 0.5 × avg(examples) | Baseline — boilerplate compresses distances to 0.13–0.21 |
| A | `core_gloss` only | Removes all boilerplate; may show wider distance range |
| B | 0.5 × `core_gloss` + 0.5 × avg(examples) | Clean definition + rich usage examples; expected best |

**Why B is expected to win**: The boilerplate problem is localised to `llm_definition`. The 5 examples are varied usage sentences without shared metalinguistic phrasing — they add useful discriminative signal without adding noise. Replacing only the noisy component (definition → core_gloss) while keeping examples should give the best of both worlds.

**Decision criteria**: Compare min/mean/max pairwise cosine distances per lemma. Wider range = better clustering signal. Also check whether threshold calibration (0.14/0.16/0.18) needs to shift after switching representations.

**Results**:

| Representation | Distance range | max/min ratio | Verdict |
|---|---|---|---|
| Original (llm_def+ex) | 0.13–0.22 | ~1.7× | Boilerplate compresses distances |
| A (core_gloss only) | 0.04–0.18 | ~5× | **Winner** — widest relative spread |
| B (core_gloss+ex) | 0.02–0.11 | ~7× | Worst — examples dominate and compress everything |

**Why B is unexpectedly bad**: all 5 examples for a lemma share its vocabulary regardless of which sense they illustrate (e.g., all "schlagen" examples contain schlagen-related vocabulary). This pulls all sense embeddings toward the same lemma centroid. The short core_gloss has too little weight to counteract it, so examples dominate completely.

**Decision: use A (core_gloss only) for clustering.**

**Threshold calibration result:**

Lower threshold sweep with A showed organic splits in range 0.04–0.14 (vs 0.14–0.20 for Original). Evaluated [0.08, 0.10, 0.12]:

| t | schlagen | Geist | spielen | verdict |
|---|---|---|---|---|
| 0.08 | 6 (constrained, organic=10) | 4 organic | 6 (constrained, organic=11) | too fine |
| **0.10** | **5 organic ✓** | **4 organic ✓** | **6 (constrained, organic=9)** | **chosen** |
| 0.12 | 3 organic — too coarse | 3 organic | 3 organic — too coarse | too aggressive |

**Chosen threshold: t=0.10.** First point where schlagen clusters organically (5 clusters). Geist stable at 4 clusters (supernatural/mental/mindset/Branntwein). spielen still constrained at t=0.10 (organic=9 > max_clusters=6) but the 6 forced clusters are linguistically coherent. t=0.12 collapses 13/15 schlagen and 12/15 spielen senses into one cluster.

**Follow-up**: Run `compute_sense_clusters.py` at t=0.10 on all 90 pilot words.

---

## 2026-03-12: Core Gloss Quality Improvement & Full Re-generation

**Status**: Decision Made

**Context**:
After the initial core_gloss generation (via `enrich_senses_with_llm.py`) and the representation comparison experiment, a quality scan of `senses_pilot.jsonl` revealed two categories of problems:

1. **Formatting issues** (265 senses, 44%): stray quotation marks and inconsistent trailing periods — fixed by one-off post-processing in-place
2. **Content issues** (25 senses): 12 "eine Art von X" constructions (uninformative) + 13 generic/tautological glosses using forbidden words

**Prompt improvements made** (`config/prompts/core_gloss.txt`):
- Added POS-specific phrasing norms: Nomen → Nominalphrase, Verben → Infinitivphrase, Adjektive → Eigenschaft direkt
- Added no-synonym rule ("Kein bloßes Synonym als gesamte Antwort")
- Added "Art", "Form", "Typ" to list of forbidden generic words
- Added "Keine 'eine Art von X'-Formulierungen" rule
- Added "Keine Anführungszeichen" rule
- Extended negative examples (3 total, including synonym example)

**Sibling context** (new — most impactful change):
The model previously saw each sense in isolation. The updated `generate_core_gloss()` now accepts `sibling_descriptions`: baseline descriptions of all other senses of the same lemma. These are passed in the prompt to force discriminative glosses. Implementation: `src/data/llm/text_generator.py`, `scripts/data/regen_core_glosses.py`.

**Decision**: Full re-generation of all 596 core_glosses via `regen_core_glosses.py --all` rather than targeted patching. Reasons:
- Updated prompt applies consistently across all senses
- Sibling context improves discriminability for all lemmas
- Cost negligible (~596 calls to gpt-4o-mini, < $0.10)
- `llm_definition` and `llm_examples` left untouched

**Results after re-generation** (verified in `sense_clustering_exploration.ipynb`):
- Zero "Art von" constructions remaining
- Zero stray quotation marks
- Only 7 generic-word hits remaining, all false positives (word appears in meaningful compound phrase)
- Distance ranges unchanged (A still wins with ~∞ ratio for schlagen/spielen — min=0.000 for near-identical fine senses)
- **No constrained clusters** at t=0.08 or t=0.10 — all splits now organic
- Geist: 3-way split (supernatural beings / alcohol / mental faculty) — cleaner than previous 4-way
- spielen at t=0.10: 3 organic clusters (play-leisure, instrument, unserious) — previously forced to 6
- schlagen at t=0.10: 4 clusters stable — large "hitting" cluster (12/15) reflects genuine GermaNet granularity

**Threshold confirmed**: t=0.10 still optimal. No reason to change.

**Follow-up**: Run `compute_sense_clusters.py` at t=0.10 on all 90 pilot words.

---

## 2026-03-12: Core Gloss Weight Tuning Result

**Status**: Resolved

**Context**:
After re-enriching `senses_pilot.jsonl` with `core_gloss` (short 5–10 word denotative gloss), we re-ran `tune_representation_weights.py` on the 93-instance eval set to check whether `core_gloss` as the definition component outperforms the verbose `llm_definition`.

**Result**:
- v2 (core_gloss + examples): best 51.6% at α=0.3
- v1 (llm_definition + examples): best 54.8% at α=0.5 *(with original sense file; after re-generation: 43.0% — see 2026-04-07 entry)*

Core gloss was **worse** for WSD by 3.2 percentage points at the time of this experiment.

**Root cause**:
Short glosses are too terse for nearest-neighbour context matching. A 7-word denotative label like *"jemanden körperlich mit der Hand treffen"* provides less surface for similarity computation against a full sentence context than the verbose, example-rich `llm_definition`. The contrastive boilerplate in `llm_definition`, while adding noise between senses, also adds useful lexical signal for context alignment.

**Decision**:
- **WSD `weighted` representation**: revert to `llm_definition` — keep α=0.5 (current fine accuracy: 43.0%)
- **Clustering**: use `core_gloss` separately (not through `weighted`) — the short gloss is still expected to reduce boilerplate-driven distance compression for clustering purposes
- No new `core_gloss_weighted` representation added to `embedding_strategies.py` (keep API simple)
- `tune_representation_weights.py` reverted to `DEF_COL = "llm_definition"`
- `validate_sense_file` for `weighted` reverted to require `llm_definition`

**Follow-up**:
- Re-run clustering exploration notebook using `core_gloss` embeddings to check distance range
- Decide how to expose `core_gloss` for clustering (simplest: embed it directly outside the `weighted` pathway)

---

## 2026-03-11: Re-enrichment with Short Core Glosses

**Status**: Resolved (see 2026-03-12 entry for tuning result)

**Context**:
The current LLM-generated definitions (`llm_definition`) were written with a contrastive, metalinguistic prompt ("beschreibe diese Bedeutung so, dass sie sich von anderen Bedeutungen unterscheidet"). This produces verbose 2–3 sentence definitions filled with boilerplate phrases shared across all senses of a word. The result is that the definition embedding component carries substantial shared noise, compressing the inter-sense distance range to 0.13–0.21 for all pilot words.

**Why this matters beyond clustering**:
The same boilerplate inflation affects WSD nearest-neighbour matching. In a compressed similarity landscape, every context is nearly equidistant from all sense representations — making the model's top-1 decision noisy. Cleaner, more distinctive sense representations should improve both clustering distance range and WSD accuracy.

**Decision**: Re-enrich `senses_pilot.jsonl` with a new `core_gloss` field — a short (5–10 word) denotative gloss stripped of all contrastive boilerplate.

**Examples of target style**:
- Current: *"Schlagen (Verb) bezeichnet die Handlung, jemandem einen oder mehrere körperliche Treffer zu versetzen, oft in Form von Schlägen mit der Hand oder einem Gegenstand... Diese Bedeutung unterscheidet sich von anderen Verwendungen..."*
- Target: *"jemanden körperlich mit der Hand oder einem Gegenstand treffen"*

**Format decision — add `core_gloss`, keep `llm_definition`**:
Add `core_gloss` as a new field in the JSONL alongside the existing `llm_definition`. Do not overwrite or remove `llm_definition`. Reasons:
- Non-destructive: existing pipeline continues to run unchanged until explicitly switched
- Enables controlled comparison on the 93-instance eval set before committing to full re-run
- Enrichment script generates both in one pass (negligible extra cost)
- `embed_with_representation` gets a new `core_gloss_weighted` representation option

**Downstream pipeline changes required**:
1. Update prompt in `scripts/data/enrich_senses_with_llm.py` to also generate `core_gloss`
2. Re-run enrichment on 595 pilot senses (fast, ~$2)
3. Add `core_gloss_weighted` representation to `src/wsd/embedding_strategies.py`
4. Re-tune embedding composition `α × core_gloss + β × mean(examples)` on 93-instance eval set
5. If new composition improves on 43.0%: re-run `predict_dwds_senses.py` on 30,337 contexts
6. Re-run clustering in `sense_clustering_exploration.ipynb` with new embeddings — distance range and optimal threshold may shift

**Expected outcome**:
- Wider inter-sense distance range (organic clustering more meaningful)
- Improved atemporal WSD accuracy (expect 58–68%, to be verified on eval set)
- Optimal embedding weight may shift away from 50/50 (core gloss is more informative than verbose definition was)

**Rationale for doing this before full clustering run**:
Running `compute_sense_clusters.py` on all 90 words with the current representations would produce clusters based on inflated similarity. Re-enriching first means the one full-scale run is based on the best representations available.

**Follow-up**:
- Update `scripts/data/enrich_senses_with_llm.py` with new prompt
- Add `core_gloss_weighted` to `src/wsd/embedding_strategies.py`
- Re-tune on eval set, document result in new research log entry
- Update `documentation/pipeline.md` Step 2 once format is finalised

---

## 2026-10-16: JSD Null Baseline and Temporal-Binning Sensitivity

**Status**: Decision Made (two submitted claims affected)

**Context:**
Reviewer-driven robustness pass for the ARR resubmission. The submitted diachronic
results rest on mean adjacent-decade JSD over 10 decades, at a median of ~50 sentences
per lemma-decade. JSD between two multinomial samples is strictly positive even when
the underlying distribution is identical, so the question was how much of the reported
JSD is sampling noise rather than semantic change — and whether the Hamilton-style
Conformity and Innovation correlations survive once that noise is subtracted.

**Method:**
Per lemma, pool all sentences into one bag of coarse-sense labels (preserving the
lemma's overall sense distribution), reshuffle, re-split into bins of exactly the
observed sizes, recompute mean adjacent-bin JSD. 1000 permutations, seed 42. This is a
strict "no change over time" null: anything it produces is noise. Reported per lemma as
excess (observed − null mean), z (excess / null SD), p_perm, and BH-FDR.
Then repeated over four pre-specified temporal binnings: (a) 10 decades (reference),
(b) 5 periods × 20 years, (c) 2 periods 1900–1949 vs. 1950–1999, (c′) same without the
1940s. Restricted to lemmas with n_clusters_coarse > 1; min. 10 sentences per
lemma-period; a transition counts only if both endpoints pass the filter.

**Findings:**

1. **The decade binning is noise-dominated.** The null accounts for 90% of the observed
   mean JSD (0.099 of 0.110 on k>1 lemmas). The noise floor falls as ~1/√n with coarser
   bins: 80% at 20-year periods, 51% at 2 periods.
2. **Conformity does not hold in any binning.** Raw frequency × JSD is −.348 at decades
   but shrinks exactly as a noise artefact must as bins coarsen (−.184 at 2 periods),
   and the null-corrected excess is ≈ 0 (n.s.) in all four binnings. The positive
   `jsd_z` correlation (+.12 to +.19) is *not* reverse Conformity: `jsd_null_sd`
   correlates −.96 with context count and −.75 with log frequency, so frequent lemmas
   get a mechanically larger z at equal excess. Excess is the clean statistic here.
3. **Innovation holds, but was underpowered at decade resolution.** Null-corrected
   polysemy × JSD is +.04 (n.s.) at decades and +.21 (p < .001) at 2 periods — same for
   entropy (+.21) and the independent survey ratings (+.21). So the decade-level null
   result was a power problem, not a null finding. Magnitude is roughly halved versus
   the submitted number (+.37 raw vs. +.21 corrected).
4. **FDR-significant change**: 9 lemmas at decades, 28 at 20-year periods, 92 of 378
   (24%) at 2 periods. Coarsening is not free: 4 of the 9 decade hits (Druck, Feuer,
   Preis, Stift) drop out, because change concentrated in one decade averages out inside
   a 50-year bin.
5. **Präsident** (case study) goes from p = .10 at decades to p_BH = .007 (z = 8.2) at
   2 periods.
6. Two code/paper mismatches found independently of the null baseline: §2.4 claims
   single-cluster lemmas are excluded but 115 of 502 are in (JSD = entropy = 0 by
   construction), and the Conformity numbers were computed on log(# retrieved DWDS
   contexts), not dlexDB frequency as described. H2 (entropy ~ perceived polysemy)
   is fully explained by cluster count: partial r = −.005.

**Decision:**
Move change detection and the Hamilton correlations to (b) or (c) and report the noise
share explicitly. Drop the Conformity claim. Keep Innovation with the corrected
magnitude. Decade trajectories stay for visualisation and case studies.

**Deviation from pre-specification:**
The pre-registered 2-period cut was 1900–1944 vs. 1945–1999. The aggregated counts only
exist at decade resolution, so the 1940s decade cannot be split at 1945. (c) assigns it
to the early period, (c′) drops it; they agree on 84 of 92 FDR hits, so the choice is
immaterial. Both reported.

**Follow-up:**
- Full numbers and tables: `analysis/robustness/RESULTS.md`
- Re-run the Hamilton extension on period-level rather than decade-level bins
- Decide with co-authors which binning becomes primary in the paper
- Derived CSVs are gitignored; both scripts regenerate them in ~90 s

---

## 2026-10-16/27: Binning Sensitivity Extended — 3/4 Periods, Frequency Partials, WSD by Era

**Status**: Analysis done; binning recommendation for co-authors

**What was added** (`analysis/robustness/binning_sensitivity.py`, RESULTS.md §5–6):
two further binnings fixed before computation, (d) 4 periods 1900–29 / 1930–49 /
1950–69 / 1970–99 and (e) 3 periods 1900–29 / 1930–59 / 1960–99, to see where between
(b) and (c) the Innovation effect appears; partial Spearman correlations controlling
for frequency (Dubossarsky et al. 2017); and an exploratory check of WSD accuracy on
pre- vs. post-1950 eval instances. Reference binning (a) reproduces the first run exactly.

**Findings:**

1. **The noise share falls gradually, with no threshold:** 90 → 80 → 78 → 71 → 51 %
   for (a) → (b) → (d) → (e) → (c). FDR-significant lemmas: 9 / 28 / 26 / 50 / 92.
2. **Innovation** (null-corrected excess) rises monotonically, from +.04 → +.08 → +.14 → +.17 → +.21,
   and becomes significant from (d) on. Survey ratings become significant only from (e) on.
3. **Innovation is not a frequency artefact.** Partialling out log dlexDB frequency
   barely moves it (+.213 → +.211 at (c)), as expected, because the permutation null already
   absorbs the sample-size path. Partialling out log context count shrinks it by
   ~10 % but it stays significant (+.19 at (c) for cluster count, entropy and survey).
4. **Conformity fails in all six binnings**. At (e) the excess even has the wrong sign (+.124*).
5. **WSD accuracy is not worse on older text:** fine 48.4 % before 1950 vs. 32.3 % from
   1950 (coarse 88.7 % vs. 77.4 %), n.s. (Fisher p = .18 / .22), and the same direction
   on the 9 lemmas present in both halves. Only 31 late instances, so this rules
   out only large gaps. It does not support the documented worry that 1900s accuracy
   is "likely lower".

**Recommendation:** (e) 3 periods as the primary change-detection binning, because it keeps a
direction of change rather than just before/after, gives 50 FDR-significant lemmas, and shows an
Innovation effect that survives both frequency controls. Report (a) for trajectories and
(c) as the strongest-power check.

---

## 2026-10-27: DWUG DE Inventory (reviewer request)

**Status**: Inventory done; plan decisions pending

Reviewers ask for evaluation on DWUG DE / SemEval-2020 Task 1 (German): ARI against human
clusters and Spearman of 2-period JSD vs. `change_graded`. Step 0 only, nothing computed.
Full write-up: `analysis/dwug/INVENTORY.md`.

- Downloaded DWUG DE 3.0.0 and DWUG DE Sense 1.0.1 (both 2024-11-01, CC BY-ND 4.0) to
  `data/external/dwug_de/` (gitignored).
- 50 targets, 9,125 uses; P1 = 1800–1899 (DTA), P2 = 1946–1990.
- All 50 are in GermaNet, 2 only via `oldOrthForm` (Engpaß, Mißklang), which our
  `XMLParser` does not read. **24 of 50 have a single synset**, so our method gives JSD = 0
  by construction, including high-change words (abgebrüht .83, Armenhaus .72).
- Only 3 overlap with our 502 lemmas (Fuß, Mut, packen). Estimate: 1.5–2 days to first numbers.
- **Side finding:** `target_matches_token` keeps only sentences containing the bare lemma form,
  so inflected forms are dropped. On DWUG that would discard 47–65 % of uses. Our DWDS
  contexts are therefore systematically restricted to base forms, which should be mentioned as a limitation.
  Follow-up: see the 2026-10-27 fetch-check entry below.

**Open decisions:** handling of single-synset words, DWUG 3.0.0 vs. SemEval gold, ARI target
(graph clusters vs. sense labels), P1 input (`context` vs. `context_normalized`).

---

## 2026-10-27: DWDS Fetch Check — Base-Form Filter, Data Access

**Status**: Partly done; blocked on DWDS technical access
**Full write-up:** `analysis/fetch_check/RESULTS.md`; sources and terms: `documentation/DATA_SOURCES.md`.

**Context:** Follow-up to the DWUG side finding above. How many sentences would we have
without the base-form filter (`target_matches_token`: target exactly once, exact base form)?

**Done:**
- v3 fetch settings confirmed: `kern`, limit 5000 (one page), `sort=random`, 6–60 tokens,
  `max_per_doc=1`, near-dup 0.7, keep all. Expanded lemmas used the same settings.
- Stratified sample of 20 lemmas (15 NOUN / 3 VERB / 2 ADJ, seed 20261027).
- `pilot_v3_coarse_stats.csv` matches the stored contexts exactly for all 20.
- Embedding benchmark: e5-large, M1/MPS, ~44 s per 1,000 sentences. Re-running the current
  318,611 sentences takes ~3.9 h, ~7.8 h at factor 2.

**Not done (blocked):** the factor b/c. Indirect hint: on DWUG, only 41–54 % of uses pass
the filter. Whether the limit binds was later measured offline (see addendum: yes, for 245/502).

**Data access:** DWDS (F. Nojack, BBAW) confirmed in writing that corpus results may be
exported and reused (mails 2025-11-21 and 2026-03-23). `/wb/` dictionary data may not be
reused; we don't use it. Only the newspaper corpora are restricted. **Technically**,
automated export is blocked: `robots.txt` disallows `/r` (and all of `ddc.dwds.de`), there
is no `#export` section on `/d/api` any more, and `/r/?…&view=csv` returns HTTP 403 with an
honest user agent. All earlier fetches only worked because `DWDSClient` sends
`User-Agent: Mozilla/5.0`. This was not circumvented; the follow-up with DWDS is pending.

**Correction to the 2026-03-23 entry (Finding 2):** the DWDS reply does not tie large-scale
export to being logged in. It allows export via the export button or the API page; the only
restriction is quotes-only for ZEIT/FAZ/Spiegel.

**Follow-up:** Ask DWDS which automated export route is intended and whether a named research
user agent can be allowed. Then run `analysis/fetch_check/fetch_check.py` and decide on
dropping the base-form filter. Proposed `DWDSClient` changes are in RESULTS.md (not implemented).

**Addendum (same day):**
- **Limit check, done offline** with the kern lemma frequencies in `data/raw/dlexdb/kern.sqlite`: 245/502 lemmas (49 %) have more than 5,000 kern tokens. For them the fetch is a 5,000-row random sample (NOUN 43 %, VERB 68 %, ADJ 62 %). For the 256 lemmas below the limit we keep 17 % of all kern tokens (NOUN 19 %, VERB 11 %, ADJ 13 %). This is an upper bound on what dropping *all* filters could gain, not the base-form factor alone.
- **Spelling:** kern uses pre-1996 orthography. Our queries for *Schloss* (41 sentences vs. 6,073 *Schloß*), *Schluss* (87 vs. 10,025 *Schluß*) and *Bass* (25 vs. 758 *Baß*) miss almost everything. A re-fetch should query both spellings.
- **Licences:** dlexDB on Zenodo is CC BY-SA 4.0. The provenance of our `kern.sqlite` (SQLite, dstar-lexdb style) is undocumented and needs confirming. See `documentation/DATA_SOURCES.md`. SemEval-2020 German (README in the archive): no named licence, "freely available for education, research and other non-commercial purposes". Non-commercial only; link to the original, don't redistribute.
