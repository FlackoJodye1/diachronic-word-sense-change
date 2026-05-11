#!/usr/bin/env python3
"""
Merge per-annotator cluster-review files back into the canonical
sense_clusters_reviewed.json.

Strategy:
  - Start from the current canonical sense_clusters_reviewed.json (carries
    the 96 lemmas already labelled by Paul before the split).
  - For each annotator's file `sense_clusters_reviewed_<name>.json`, take
    only the lemmas listed in `assignments/<name>.txt` and overwrite the
    canonical entry for those lemmas.
  - Refuses to merge if two annotators were assigned the same lemma (the
    split script should make this impossible, but we check anyway).

Usage:
    python scripts/clustering/merge_annotations.py paul anna ben
    # → updates data/processed/germanet/sense_clusters_reviewed.json in place

By default a backup of the pre-merge canonical file is written next to it
with a `.bak` suffix.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
GERMANET_DIR = BASE_DIR / "data" / "processed" / "germanet"
ASSIGNMENTS_DIR = BASE_DIR / "data" / "processed" / "sense_clustering" / "assignments"
CANONICAL = GERMANET_DIR / "sense_clusters_reviewed.json"


def load_assignment(name: str) -> set[str]:
    path = ASSIGNMENTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Assignment file not found: {path}")
    with path.open(encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def load_annotator_file(name: str) -> dict:
    path = GERMANET_DIR / f"sense_clusters_reviewed_{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Annotator file not found: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def count_labelled(payload: dict) -> tuple[int, int]:
    clusters = payload.get("clusters", {})
    labelled = sum(1 for c in clusters.values() if c.get("label"))
    return labelled, len(clusters)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("annotators", nargs="+", help="Annotator names (must match split_annotation_work.py).")
    ap.add_argument("--no-backup", action="store_true", help="Skip writing the .bak file.")
    ap.add_argument("--dry-run", action="store_true", help="Report what would change, but don't write.")
    args = ap.parse_args()

    # Sanity-check: each lemma must be assigned to exactly one annotator.
    assignments = {name: load_assignment(name) for name in args.annotators}
    all_assigned = Counter()
    for name, lemmas in assignments.items():
        for lemma in lemmas:
            all_assigned[lemma] += 1
    overlap = [l for l, c in all_assigned.items() if c > 1]
    if overlap:
        print("ERROR: lemmas assigned to multiple annotators:", overlap, file=sys.stderr)
        return 2

    with CANONICAL.open(encoding="utf-8") as f:
        canonical = json.load(f)

    summary: list[str] = []
    for name in args.annotators:
        data = load_annotator_file(name)
        lemmas = assignments[name]
        n_lemmas = 0
        n_clusters = 0
        n_labelled = 0
        for lemma in lemmas:
            if lemma not in data:
                print(f"WARN: lemma {lemma!r} (assigned to {name}) not in {name}'s file — skipped.",
                      file=sys.stderr)
                continue
            canonical[lemma] = data[lemma]
            labelled, total = count_labelled(data[lemma])
            n_lemmas += 1
            n_clusters += total
            n_labelled += labelled
        pct = round(100 * n_labelled / n_clusters) if n_clusters else 0
        summary.append(f"  {name:>12s}: {n_lemmas:>4d} lemmas merged, "
                       f"{n_labelled}/{n_clusters} clusters labelled ({pct}%)")

    print("Merge summary:")
    for line in summary:
        print(line)

    if args.dry_run:
        print("\n(dry run — canonical file not written)")
        return 0

    if not args.no_backup:
        bak = CANONICAL.with_suffix(CANONICAL.suffix + ".bak")
        shutil.copy2(CANONICAL, bak)
        print(f"\nBackup: {bak.relative_to(BASE_DIR)}")

    with CANONICAL.open("w", encoding="utf-8") as f:
        json.dump(canonical, f, ensure_ascii=False, indent=2)
    print(f"Wrote:  {CANONICAL.relative_to(BASE_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
