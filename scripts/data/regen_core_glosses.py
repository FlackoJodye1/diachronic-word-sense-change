#!/usr/bin/env python3
"""
Re-generate core_gloss for targeted synset IDs or all senses.

Use this when specific glosses are low-quality (e.g. "Art von X" constructions,
tautological definitions, overly abstract paraphrases) or after updating the prompt.

Only core_gloss is updated; llm_definition and llm_examples are left untouched.

Sibling context is always passed: for each sense, the baseline descriptions of the
other senses of the same lemma are included in the prompt. The instruction is to
distinguish senses via concrete objects, tools, or contexts — not via abstract
reformulations. This avoids the failure mode where many similar senses collapse
into near-identical embeddings (e.g. "etwas mit Kraft treffen oder bewegen").

Runs with 10 parallel API workers by default.

Usage:
    # Re-generate all senses (recommended when updating the prompt)
    python scripts/data/regen_core_glosses.py --all --provider openai --yes

    # Re-generate only the hardcoded low-quality targets
    python scripts/data/regen_core_glosses.py --provider openai

    # Dry run
    python scripts/data/regen_core_glosses.py --all --dry-run
"""

import argparse
import json
import logging
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.llm import LLMClient, SenseTextGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
SENSES_FILE = BASE_DIR / "data" / "processed" / "germanet" / "senses_pilot.jsonl"

# Synset IDs to re-generate when not using --all (identified by quality scan 2026-03-12)
TARGET_SYNSET_IDS = {
    # "Art von X" constructions
    "s10251",   # Bahn
    "s11606",   # Bass
    "s46844",   # Bau
    "s25671",   # Blatt
    "s43550",   # Feld
    "s49954",   # Frosch
    "s25686",   # Herz
    "s15868",   # Herz
    "s112858",  # Nest
    "s110054",  # Stock
    "s75742",   # Ton
    "s25318",   # Welle
    # Generic / tautological
    "s14036",   # Art / Natur
    "s27075",   # Art
    "s112176",  # Geist
    "s38837",   # Geist
    "s40349",   # Land / Grund
    "s75298",   # Natur
    "s52123",   # rechnen
    "s53317",   # sammeln
    "s1295",    # schwarz
    "s56742",   # spielen
    "s52122",   # zählen
    # Wrong glosses from 2026-03-14 re-generation
    "s56941",   # schlagen — got competition-winning gloss instead of "Schlägerei"
    "s54197",   # schlagen — got generic hitting instead of "jemanden zum Ritter ernennen"
}


def clean_gloss(text: str) -> str:
    """Strip stray quotation marks and trailing periods."""
    text = text.strip()
    for q in ['"', '\u201c', '\u201e', '\u00bb', '\u00ab']:
        text = text.strip(q)
    text = text.strip()
    if text.endswith('.'):
        text = text[:-1].rstrip()
    return text


def build_sibling_map(senses: list[dict]) -> dict[str, list[str]]:
    """For each sense, collect baseline descriptions of all other senses of the same lemma."""
    lemma_baselines: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for s in senses:
        lemma_baselines[s["lemma"]].append((s["synset_id"], s.get("baseline", "")))

    sibling_map: dict[str, list[str]] = {}
    for lemma, entries in lemma_baselines.items():
        for sid, _ in entries:
            sibling_map[sid] = [
                baseline for other_sid, baseline in entries
                if other_sid != sid and baseline.strip()
            ]
    return sibling_map


def main():
    parser = argparse.ArgumentParser(description="Re-generate core_gloss for sense entries")
    parser.add_argument("--all", action="store_true", help="Re-generate all senses (not just targeted ones)")
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "claude"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel API workers")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    logger.info(f"Loading senses from {SENSES_FILE}")
    senses = []
    with open(SENSES_FILE, encoding="utf-8") as f:
        for line in f:
            senses.append(json.loads(line))
    logger.info(f"  Loaded {len(senses)} senses")

    if args.all:
        targets = senses
        logger.info(f"  Mode: --all ({len(targets)} senses)")
    else:
        targets = [s for s in senses if s["synset_id"] in TARGET_SYNSET_IDS]
        logger.info(f"  Mode: targeted ({len(targets)}/{len(TARGET_SYNSET_IDS)} synset IDs found)")
        for s in targets:
            logger.info(f"    {s['lemma']:15s} {s['synset_id']}  current: {s.get('core_gloss', '(none)')}")

    sibling_map = build_sibling_map(senses)

    if not args.dry_run:
        logger.info(f"\nInitializing {args.provider} client...")
        kwargs = {"provider": args.provider, "temperature": args.temperature}
        if args.model:
            kwargs["model"] = args.model
        client = LLMClient(**kwargs)
        generator = SenseTextGenerator(client)

        if not args.yes:
            confirm = input(f"\nRe-generate core_gloss for {len(targets)} senses? [y/N]: ")
            if confirm.lower() != "y":
                logger.info("Cancelled.")
                sys.exit(0)
    else:
        generator = None

    def process_sense(args_tuple):
        i, sense = args_tuple
        sid = sense["synset_id"]
        lemma = sense["lemma"]
        pos = sense["pos"]
        baseline = sense.get("baseline", "")
        siblings = sibling_map.get(sid, [])
        logger.info(f"[{i}/{len(targets)}] {lemma} ({sid})  siblings: {len(siblings)}")
        if args.dry_run:
            return sid, f"[DRY RUN] Gloss for {lemma}", sense.get("core_gloss", "(none)")
        new_gloss = generator.generate_core_gloss(
            lemma=lemma,
            pos=pos,
            sense_description=baseline,
            sibling_descriptions=siblings or None,
        )
        return sid, new_gloss, sense.get("core_gloss", "(none)")

    gloss_index = {s["synset_id"]: s for s in senses}
    n_workers = 1 if args.dry_run else min(args.workers, 3)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_sense, (i, s)): s for i, s in enumerate(targets, 1)}
        for future in as_completed(futures):
            sid, new_gloss, old_gloss = future.result()
            new_gloss = clean_gloss(new_gloss)
            if not args.dry_run:
                logger.info(f"  {old_gloss} → {new_gloss}")
            gloss_index[sid]["core_gloss"] = new_gloss

    if not args.dry_run:
        backup = SENSES_FILE.with_suffix(".jsonl.regen_bak")
        shutil.copy2(SENSES_FILE, backup)
        logger.info(f"\nBackup saved to {backup}")

    out_path = SENSES_FILE if not args.dry_run else SENSES_FILE.with_suffix(".jsonl.dryrun")
    with open(out_path, "w", encoding="utf-8") as f:
        for s in senses:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info(f"Saved to {out_path}")
    logger.info("Done.")


if __name__ == "__main__":
    main()