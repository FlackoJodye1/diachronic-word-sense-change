#!/usr/bin/env python3
"""
Generate core_gloss for polysemous non-pilot words and save to senses_expanded.jsonl.

Reads senses_full.jsonl, filters to words with >=2 senses that are not already in the
pilot study, generates core_gloss for each sense, and writes senses_expanded.jsonl.

The output file has the same format as senses_pilot.jsonl and can be used directly
with compute_sense_clusters.py.

Usage:
    # Generate all (prompts for confirmation)
    python scripts/data/generate_expanded_senses.py --provider openai --yes

    # Resume a previously interrupted run
    python scripts/data/generate_expanded_senses.py --provider openai --resume --yes

    # Dry run (no API calls, no files written)
    python scripts/data/generate_expanded_senses.py --dry-run
"""

import argparse
import json
import logging
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.llm import LLMClient, SenseTextGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
SENSES_FULL_FILE = BASE_DIR / "data" / "processed" / "germanet" / "senses_full.jsonl"
PILOT_FILE = BASE_DIR / "data" / "processed" / "germanet" / "senses_pilot.jsonl"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "germanet" / "senses_expanded.jsonl"


def clean_gloss(text: str) -> str:
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
    parser = argparse.ArgumentParser(
        description="Generate core_gloss for polysemous non-pilot words"
    )
    parser.add_argument("--provider", type=str, default="openai", choices=["openai", "claude"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--workers", type=int, default=1, help="Parallel API workers (default: 1)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip senses that already have core_gloss in the output file")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without making API calls or writing files")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    # Load pilot lemmas to exclude
    pilot_lemmas: set[str] = set()
    with open(PILOT_FILE, encoding="utf-8") as f:
        for line in f:
            pilot_lemmas.add(json.loads(line)["lemma"])
    logger.info(f"Pilot lemmas (excluded): {len(pilot_lemmas)}")

    # Load all senses from senses_full.jsonl
    all_senses_full: list[dict] = []
    with open(SENSES_FULL_FILE, encoding="utf-8") as f:
        for line in f:
            all_senses_full.append(json.loads(line))
    logger.info(f"Loaded {len(all_senses_full)} senses from senses_full.jsonl")

    # Filter to polysemous non-pilot words
    lemma_counts = Counter(s["lemma"] for s in all_senses_full)
    target_lemmas = {l for l, c in lemma_counts.items() if c >= 2 and l not in pilot_lemmas}
    target_senses = [s for s in all_senses_full if s["lemma"] in target_lemmas]
    logger.info(
        f"Target: {len(target_lemmas)} polysemous non-pilot lemmas, "
        f"{len(target_senses)} senses"
    )

    # Build working index — start from target_senses, overlay any already-generated glosses
    gloss_index: dict[str, dict] = {s["synset_id"]: dict(s) for s in target_senses}

    already_done: set[str] = set()
    if args.resume and OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                sid = entry["synset_id"]
                if sid in gloss_index and entry.get("core_gloss"):
                    gloss_index[sid]["core_gloss"] = entry["core_gloss"]
                    already_done.add(sid)
        logger.info(f"Resume: {len(already_done)} senses already have core_gloss — skipping")

    to_generate = [s for s in target_senses if s["synset_id"] not in already_done]
    logger.info(f"Senses to generate: {len(to_generate)}")

    if not to_generate:
        logger.info("Nothing to generate.")
        if not OUTPUT_FILE.exists():
            _write_output(gloss_index, target_senses)
        return

    if args.dry_run:
        logger.info("Dry run — sample of senses that would be processed:")
        for s in to_generate[:5]:
            logger.info(f"  {s['lemma']:20s} {s['synset_id']}  {s.get('baseline', '')[:60]}")
        if len(to_generate) > 5:
            logger.info(f"  ... and {len(to_generate) - 5} more")
        return

    # Sibling context is built from all senses_full entries (not just targets),
    # so multi-sense words have correct sibling lists even if some siblings are in the pilot.
    sibling_map = build_sibling_map(all_senses_full)

    kwargs: dict = {"provider": args.provider, "temperature": args.temperature}
    if args.model:
        kwargs["model"] = args.model
    client = LLMClient(**kwargs)
    generator = SenseTextGenerator(client)

    if not args.yes:
        confirm = input(f"\nGenerate core_gloss for {len(to_generate)} senses? [y/N]: ")
        if confirm.lower() != "y":
            logger.info("Cancelled.")
            sys.exit(0)

    def process_sense(args_tuple: tuple[int, dict]) -> tuple[str, str]:
        i, sense = args_tuple
        sid = sense["synset_id"]
        siblings = sibling_map.get(sid, [])
        logger.info(f"[{i}/{len(to_generate)}] {sense['lemma']} ({sid})  siblings: {len(siblings)}")
        new_gloss = generator.generate_core_gloss(
            lemma=sense["lemma"],
            pos=sense["pos"],
            sense_description=sense.get("baseline", ""),
            sibling_descriptions=siblings or None,
        )
        return sid, new_gloss

    n_workers = min(args.workers, 10)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_sense, (i, s)): s
            for i, s in enumerate(to_generate, 1)
        }
        for future in as_completed(futures):
            sid, new_gloss = future.result()
            new_gloss = clean_gloss(new_gloss)
            gloss_index[sid]["core_gloss"] = new_gloss

    _write_output(gloss_index, target_senses)
    logger.info("Done.")


def _write_output(gloss_index: dict[str, dict], target_senses: list[dict]) -> None:
    if OUTPUT_FILE.exists():
        shutil.copy2(OUTPUT_FILE, OUTPUT_FILE.with_suffix(".jsonl.bak"))
        logger.info(f"Backup saved to {OUTPUT_FILE.with_suffix('.jsonl.bak')}")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for s in target_senses:
            f.write(json.dumps(gloss_index[s["synset_id"]], ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(target_senses)} senses to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
