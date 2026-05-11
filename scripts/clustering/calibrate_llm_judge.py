#!/usr/bin/env python3
"""Calibrate the LLM-as-judge against the manually-reviewed cluster mapping.

Treats every multi-member cluster in `sense_clusters_reviewed.json` as a
human-approved positive example, asks the judge to rule on each, and reports
agreement metrics. Singleton clusters (one fine sense → one coarse cluster) are
skipped — there is no merge to evaluate.

Outputs are streamed to JSONL incrementally, so a crash or rate-limit blip
does not lose work; pass `--resume` to skip clusters already in the output
file.

This script does NOT modify `sense_clusters_reviewed.json`. It only writes
to its own output directory.

Usage:
    # Calibrate on the full reviewed mapping (~336 multi-member clusters)
    python scripts/clustering/calibrate_llm_judge.py

    # Smoke test on 5 clusters before committing
    python scripts/clustering/calibrate_llm_judge.py --limit 5

    # Restrict to specific lemmas
    python scripts/clustering/calibrate_llm_judge.py --lemmas schlagen spielen Bank

    # Resume after interruption
    python scripts/clustering/calibrate_llm_judge.py --resume

Pipeline position:
    sense_clusters_reviewed.json  ──▶  [this script]  ──▶  calibration_*.jsonl
                                                       └──▶ summary stats
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from openai import OpenAI, OpenAIError
from dotenv import load_dotenv

# Match the convention in src/data/llm/client.py: read .env at the project root
# so OPENAI_API_KEY (and other secrets) don't have to be exported per-shell.
load_dotenv()

from src.diachronic.io import load_senses
from src.wsd.llm_judge import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    JudgeResult,
    judge_cluster,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_REVIEWED = BASE_DIR / "data" / "processed" / "germanet" / "sense_clusters_reviewed.json"
# senses_full.jsonl covers all 1813 (lemma, synset_id) pairs in the reviewed
# mapping (503 lemmas); pilot is a 90-lemma subset. The eval file adds
# atemporal-only lemmas (Antwort, Doktor, Glas, Schatz, bitter, wagen) and is
# read second so it can override duplicate synsets if any.
DEFAULT_FULL_SENSES = BASE_DIR / "data" / "processed" / "germanet" / "senses_full.jsonl"
DEFAULT_EVAL_SENSES = BASE_DIR / "data" / "evaluation" / "atemporal" / "input" / "sense_text_representations.jsonl"
DEFAULT_OUTPUT = (
    BASE_DIR / "data" / "processed" / "sense_clustering" / "llm_judge" / "calibration_v1.jsonl"
)


# ---------------------------------------------------------------------------
# Cluster enumeration
# ---------------------------------------------------------------------------

def iter_multi_member_clusters(
    reviewed: Dict, lemmas: Optional[Set[str]] = None
) -> Iterator[Tuple[str, str, List[str], Optional[str]]]:
    """Yield (lemma, cluster_id, members, human_label) for every multi-member cluster."""
    for lemma, info in reviewed.items():
        if lemmas is not None and lemma not in lemmas:
            continue
        for cluster_id, cluster in info.get("clusters", {}).items():
            members = cluster.get("members", [])
            if len(members) < 2:
                continue
            yield lemma, cluster_id, list(members), cluster.get("label")


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_completed_keys(output_path: Path) -> Set[str]:
    """Read existing JSONL output and return the set of (lemma, cluster_id) already judged."""
    if not output_path.exists():
        return set()
    seen: Set[str] = set()
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed JSONL line in {output_path}")
                continue
            seen.add(f"{rec['lemma']}::{rec['cluster_id']}")
    return seen


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_agreement(records: List[Dict]) -> None:
    """Print agreement and confidence stats given the JSONL records."""
    n = len(records)
    if n == 0:
        logger.warning("No records to report on.")
        return

    approved = sum(1 for r in records if r["judge"]["decision"] == "APPROVE")
    rejected = n - approved
    conf_dist: Dict[int, int] = {c: 0 for c in range(1, 6)}
    for r in records:
        conf_dist[int(r["judge"]["confidence"])] += 1

    cached = sum(r["judge"]["usage"].get("cached_prompt_tokens", 0) for r in records)
    prompt_tokens = sum(r["judge"]["usage"]["prompt_tokens"] for r in records)
    completion_tokens = sum(r["judge"]["usage"]["completion_tokens"] for r in records)

    print()
    print("=" * 60)
    print("LLM-as-judge calibration summary")
    print("=" * 60)
    print(f"Multi-member clusters judged:         {n}")
    print(f"  APPROVE (agrees with human merge):  {approved} ({approved / n:.1%})")
    print(f"  REJECT  (disagrees):                {rejected} ({rejected / n:.1%})")
    print()
    print("Confidence distribution:")
    for c in range(5, 0, -1):
        bar = "█" * int(40 * conf_dist[c] / max(1, n))
        print(f"  {c}: {conf_dist[c]:4d}  {bar}")
    print()
    print("Recommended manual review queue:")
    low_conf_approves = sum(
        1 for r in records if r["judge"]["decision"] == "APPROVE" and r["judge"]["confidence"] <= 2
    )
    print(f"  REJECTs (all):                      {rejected}")
    print(f"  APPROVEs with confidence ≤ 2:       {low_conf_approves}")
    print()
    print("Token usage:")
    print(f"  Prompt tokens (total):              {prompt_tokens:,}")
    print(f"    of which cached:                  {cached:,}")
    print(f"  Completion tokens:                  {completion_tokens:,}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--reviewed", type=Path, default=DEFAULT_REVIEWED,
                        help="Path to sense_clusters_reviewed.json")
    parser.add_argument("--senses-files", type=Path, nargs="+",
                        default=[DEFAULT_FULL_SENSES, DEFAULT_EVAL_SENSES],
                        help="One or more senses JSONL files (concatenated; "
                             "default covers every lemma in reviewed mapping)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output JSONL path for raw judge responses")
    parser.add_argument("--lemmas", nargs="+", default=None,
                        help="Restrict to these lemmas (default: all)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after N clusters (smoke testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip clusters already present in the output file")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"OpenAI model (default: {DEFAULT_MODEL})")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help=f"Max output tokens (default: {DEFAULT_MAX_TOKENS})")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--report-only", action="store_true",
                        help="Re-print summary from existing output, no API calls")
    args = parser.parse_args()

    # Always print summary at the end if any results exist.
    def _maybe_summary() -> None:
        if not args.output.exists():
            return
        records = []
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        report_agreement(records)

    if args.report_only:
        _maybe_summary()
        return 0

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY is not set (checked env + .env); aborting before any API call.")
        return 2

    if not args.reviewed.exists():
        logger.error(f"Reviewed mapping not found: {args.reviewed}")
        return 1

    logger.info(f"Loading reviewed mapping from {args.reviewed}")
    with open(args.reviewed, "r", encoding="utf-8") as f:
        reviewed = json.load(f)
    logger.info(f"  {len(reviewed)} lemmas in mapping")

    logger.info(f"Loading senses from {len(args.senses_files)} file(s)")
    import pandas as pd
    frames = []
    for sp in args.senses_files:
        if not sp.exists():
            logger.warning(f"  senses file missing, skipping: {sp}")
            continue
        frames.append(load_senses(sp))
    if not frames:
        logger.error("No senses files loaded; aborting.")
        return 1
    senses_df = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["lemma", "synset_id"], keep="first"
    )
    logger.info(f"  {len(senses_df)} sense rows across {senses_df['lemma'].nunique()} lemmas")

    lemma_filter = set(args.lemmas) if args.lemmas else None
    if lemma_filter:
        missing = lemma_filter - set(reviewed.keys())
        if missing:
            logger.warning(f"Lemmas not in reviewed mapping: {sorted(missing)}")

    candidates = list(iter_multi_member_clusters(reviewed, lemma_filter))
    logger.info(f"Found {len(candidates)} multi-member clusters in scope")

    completed = load_completed_keys(args.output) if args.resume else set()
    if completed:
        logger.info(f"Resume: skipping {len(completed)} already-judged clusters")
        candidates = [
            c for c in candidates if f"{c[0]}::{c[1]}" not in completed
        ]

    if args.limit:
        candidates = candidates[: args.limit]
    logger.info(f"Will judge {len(candidates)} clusters")

    if not candidates:
        logger.info("Nothing to do.")
        _maybe_summary()
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI()

    n_done = 0
    n_errors = 0
    with open(args.output, "a", encoding="utf-8") as out_f:
        for i, (lemma, cluster_id, members, human_label) in enumerate(candidates, 1):
            try:
                result: JudgeResult = judge_cluster(
                    client,
                    lemma,
                    members,
                    senses_df,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
            except KeyError as e:
                # Cluster references senses that aren't in any of the loaded
                # senses files — skip with a warning rather than crashing the run.
                n_errors += 1
                logger.warning(
                    f"[{i}/{len(candidates)}] {lemma} {cluster_id}: missing senses {e}; skipping"
                )
                continue
            except (OpenAIError, RuntimeError, json.JSONDecodeError) as e:
                n_errors += 1
                logger.error(f"[{i}/{len(candidates)}] {lemma} {cluster_id}: {type(e).__name__}: {e}")
                continue

            record = {
                "lemma": lemma,
                "cluster_id": cluster_id,
                "members": members,
                "human_label": human_label,
                "judge": result.to_log_dict(),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            n_done += 1

            if i % 10 == 0 or i == len(candidates):
                logger.info(
                    f"[{i}/{len(candidates)}] last={lemma} {cluster_id} "
                    f"→ {result.decision} (conf={result.confidence}) "
                    f"[{result.latency_seconds:.1f}s]"
                )

    logger.info(f"Done. Judged {n_done}, errors {n_errors}. Output: {args.output}")
    _maybe_summary()
    return 0 if n_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
