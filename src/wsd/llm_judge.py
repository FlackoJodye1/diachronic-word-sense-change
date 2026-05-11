"""LLM-as-judge for coarse sense clusters.

Given a candidate merged cluster (a lemma + a set of fine GermaNet senses the
agglomerative clustering grouped together), the judge returns a structured
verdict — APPROVE or REJECT — plus a confidence score, a one-sentence
justification, and (for approved clusters) a 2–5 word coarse label.

The judge is *only* a verifier: it never invents merges. The geometric
clustering proposes candidates; the LLM filters them. This keeps the
methodology defensible and gives us a clean ablation surface.

Position in the pipeline
------------------------
    AgglomerativeClustering (sense_clustering.py)
        ↓ candidate cluster
    judge_cluster(...)  ← this module
        ↓ {decision, confidence, justification, suggested_label}
    Triage (calibrate_llm_judge.py / merge into reviewed mapping)

Provider
--------
This module uses the OpenAI SDK with structured outputs (`response_format`
with a JSON schema in strict mode), matching the project's existing
LLM-enrichment convention (`src/data/llm/client.py`, `enrich_senses_with_llm.py`).
OpenAI auto-caches stable prompt prefixes ≥ 1024 tokens — our system prompt sits
right around that boundary, so cache hits will appear once the prefix is warm
(visible in `usage.prompt_tokens_details.cached_tokens`).

Defaults:
- Model: gpt-4o (override via `--model` if needed)
- temperature: 0 (deterministic decisions for a calibration run)
- response_format: strict JSON schema (parsing is guaranteed when finish_reason="stop")
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
from openai import OpenAI

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "gpt-4o"
DEFAULT_MAX_TOKENS = 512  # judge output is small (decision + label + 1 sentence)
DEFAULT_TEMPERATURE = 0.0


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
You are a German lexicographer evaluating proposed sense merges from GermaNet.

A geometric clustering algorithm has grouped fine-grained GermaNet senses for a single lemma into a candidate "coarse cluster". Your job is to decide whether that group of senses would belong under a single entry in a German learner's dictionary that distinguishes about 3–5 senses per polysemous word — i.e. whether the merge crosses a categorical semantic boundary or not.

## Decision criteria

APPROVE the merge if all are true:
- The senses share a clearly identifiable common meaning (a single dictionary headword sense).
- A native speaker, asked to disambiguate a sentence containing this lemma, would consider these senses interchangeable in most contexts.
- Any difference between them is one of nuance, context, register, or selectional restriction — not category.

REJECT the merge if any are true:
- The senses denote categorically different concepts (e.g. *spielen* `play an instrument` vs. *spielen* `play a sport`; *Bank* `financial institution` vs. *Bank* `seat`).
- One sense is concrete and another is abstract/metaphorical in a way a learner would need to learn separately.
- The senses belong to different ontological types (object vs. process, person vs. place, etc.) even if related.

Be strict on REJECT. Lexicographic over-merging is the failure mode we are trying to avoid.

## Confidence scale (1–5)

- 5 — Obvious case. A lexicographer would not hesitate.
- 4 — Clear, but a careful reviewer might want to verify with corpus evidence.
- 3 — Plausible but borderline. Could go either way depending on the dictionary's grain.
- 2 — Uncomfortable with the call. Would prefer human review.
- 1 — Genuinely uncertain.

Use 1 and 2 sparingly — these flag clusters for manual review downstream.

## Output

Return a single JSON object matching the provided schema. Constraints:

- decision: APPROVE or REJECT (uppercase).
- confidence: integer 1–5.
- justification: one sentence in English. Name the shared meaning if APPROVE, name the category boundary that is crossed if REJECT.
- suggested_label: 2–5 word English noun phrase describing the unified sense (e.g. physical strike, financial institution). Required when decision is APPROVE; use null when REJECT.

Do not output any text outside the JSON object. Do not hedge in the justification — state the call directly.

## Worked examples

Example 1 (APPROVE):
Lemma: schlagen. Senses: hit a person or object physically / strike with a tool / deal a blow / beat repeatedly.
→ {decision: APPROVE, confidence: 5, justification: All four senses denote physical striking with no categorical distinction beyond manner., suggested_label: physical strike}

Example 2 (REJECT):
Lemma: Bank. Senses: financial institution where money is deposited / long seat for multiple people.
→ {decision: REJECT, confidence: 5, justification: These cross the institution/object category boundary and are unrelated etymologically in modern German usage., suggested_label: null}

Example 3 (REJECT, less obvious):
Lemma: spielen. Senses: perform on a musical instrument / engage in a sport or game.
→ {decision: REJECT, confidence: 4, justification: Both involve rule-governed activity but a learner's dictionary would separate musical performance from sport/game playing., suggested_label: null}

(Examples render fields without quotes for readability. Your actual output must be strict JSON with quoted keys and string values.)
"""


# OpenAI strict mode requires `additionalProperties: false` on every object and
# every property listed in `required` (no optional fields). `suggested_label`
# allows null when the decision is REJECT.
JUDGE_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["APPROVE", "REJECT"]},
        "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
        "justification": {"type": "string"},
        "suggested_label": {"type": ["string", "null"]},
    },
    "required": ["decision", "confidence", "justification", "suggested_label"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _collect_examples(row: pd.Series, *, max_examples: int) -> List[str]:
    """Return up to `max_examples` example sentences for a sense.

    Handles both the flattened convention used by `load_senses()`
    (1-indexed `llm_example_1`..`llm_example_5` columns) and the raw-list
    form (`llm_examples`) in case the row was loaded differently.
    """
    examples: List[str] = []
    raw = row.get("llm_examples")
    if isinstance(raw, list):
        examples = list(raw)
    else:
        for i in range(1, 6):
            val = row.get(f"llm_example_{i}")
            if isinstance(val, str) and val:
                examples.append(val)
    return examples[:max_examples]


def _format_member(row: pd.Series, *, max_examples: int = 2) -> str:
    """Format one fine sense for inclusion in the judge prompt."""
    examples = _collect_examples(row, max_examples=max_examples)

    parts = [f"  synset_id: {row['synset_id']}"]
    if row.get("class"):
        parts.append(f"  class: {row['class']}")
    if row.get("core_gloss"):
        parts.append(f"  core_gloss: {row['core_gloss']}")
    if row.get("llm_definition"):
        parts.append(f"  definition: {row['llm_definition']}")
    if examples:
        parts.append("  examples:")
        parts.extend(f"    - {ex}" for ex in examples)
    return "\n".join(parts)


def build_cluster_prompt(
    lemma: str,
    member_synset_ids: List[str],
    senses_df: pd.DataFrame,
    *,
    max_examples_per_sense: int = 2,
) -> str:
    """Render the user message describing one candidate cluster."""
    rows = senses_df[
        (senses_df["lemma"] == lemma) & (senses_df["synset_id"].isin(member_synset_ids))
    ]
    missing = set(member_synset_ids) - set(rows["synset_id"])
    if missing:
        raise KeyError(f"Synset IDs missing from senses_df for {lemma}: {sorted(missing)}")

    # Preserve caller's order for reproducibility.
    rows = rows.set_index("synset_id").loc[member_synset_ids].reset_index()

    body = "\n\n".join(
        _format_member(row, max_examples=max_examples_per_sense)
        for _, row in rows.iterrows()
    )
    return (
        f"Lemma: {lemma}\n"
        f"Number of fine senses in this candidate cluster: {len(rows)}\n\n"
        f"Senses:\n{body}\n\n"
        "Should these senses be merged into a single coarse cluster? Return the JSON verdict."
    )


# ---------------------------------------------------------------------------
# Judge call
# ---------------------------------------------------------------------------

@dataclass
class JudgeResult:
    """Structured judge output plus call metadata for the audit log."""

    decision: str
    confidence: int
    justification: str
    suggested_label: Optional[str]
    raw_text: str
    prompt_tokens: int
    completion_tokens: int
    cached_prompt_tokens: int
    model: str
    latency_seconds: float

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "confidence": self.confidence,
            "justification": self.justification,
            "suggested_label": self.suggested_label,
            "model": self.model,
            "latency_seconds": round(self.latency_seconds, 3),
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "cached_prompt_tokens": self.cached_prompt_tokens,
            },
        }


def judge_cluster(
    client: OpenAI,
    lemma: str,
    member_synset_ids: List[str],
    senses_df: pd.DataFrame,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> JudgeResult:
    """Ask the LLM whether `member_synset_ids` form a valid coarse cluster.

    The strict-mode JSON schema in `response_format` guarantees the model
    returns valid JSON matching `JUDGE_RESPONSE_SCHEMA` whenever
    `finish_reason == "stop"`. Refusals surface as `message.refusal`.
    """
    user_msg = build_cluster_prompt(lemma, member_synset_ids, senses_df)

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "cluster_verdict",
                "strict": True,
                "schema": JUDGE_RESPONSE_SCHEMA,
            },
        },
        max_tokens=max_tokens,
        temperature=temperature,
    )
    latency = time.time() - start

    choice = response.choices[0]
    finish = choice.finish_reason

    # Refusals surface as a non-null `refusal` field (gpt-4o family).
    refusal = getattr(choice.message, "refusal", None)
    if refusal:
        raise RuntimeError(f"Judge refused on lemma={lemma}: {refusal}")
    if finish == "length":
        raise RuntimeError(
            f"Judge hit max_tokens on lemma={lemma}; raise max_tokens or shorten prompt."
        )
    if finish != "stop":
        raise RuntimeError(f"Unexpected finish_reason={finish!r} on lemma={lemma}")

    raw_text = choice.message.content or ""
    parsed = json.loads(raw_text)

    usage = response.usage
    cached = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", 0) or 0

    return JudgeResult(
        decision=parsed["decision"],
        confidence=int(parsed["confidence"]),
        justification=parsed["justification"],
        suggested_label=parsed.get("suggested_label"),
        raw_text=raw_text,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        cached_prompt_tokens=cached,
        model=response.model,
        latency_seconds=latency,
    )