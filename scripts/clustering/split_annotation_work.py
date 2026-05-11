#!/usr/bin/env python3
"""
Split unlabelled cluster-review work across multiple annotators.

Reads `data/processed/germanet/sense_clusters_reviewed.json`, finds every
lemma that still has at least one unlabelled cluster, and divides those
lemmas across the named annotators using greedy load balancing on the
number of unlabelled clusters per lemma (so cluster-counts stay roughly
balanced, not just lemma-counts).

Writes one whitelist file per annotator:
    data/processed/sense_clustering/assignments/<annotator>.txt

Each file contains one lemma per line, ASCII-sorted. These files are then
passed to `review_app.py --lemmas-file ...` on each annotator's machine.

Usage:
    python scripts/clustering/split_annotation_work.py paul anna ben
    # → assignments/paul.txt, assignments/anna.txt, assignments/ben.txt

The split is deterministic given the same input file and annotator list.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
REVIEWED_FILE = BASE_DIR / "data" / "processed" / "germanet" / "sense_clusters_reviewed.json"
ASSIGNMENTS_DIR = BASE_DIR / "data" / "processed" / "sense_clustering" / "assignments"


def count_unlabelled(payload: dict) -> int:
    clusters = payload.get("clusters", {})
    return sum(1 for c in clusters.values() if not c.get("label"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("annotators", nargs="+", help="Annotator names (2+).")
    ap.add_argument("--dry-run", action="store_true", help="Print the split without writing files.")
    args = ap.parse_args()

    if len(args.annotators) < 2:
        ap.error("Need at least two annotator names.")
    if len(set(args.annotators)) != len(args.annotators):
        ap.error("Annotator names must be unique.")

    with REVIEWED_FILE.open(encoding="utf-8") as f:
        data = json.load(f)

    # Collect (lemma, unlabelled_cluster_count) for every lemma needing work.
    todo = [(lemma, count_unlabelled(payload)) for lemma, payload in data.items()]
    todo = [(lemma, n) for lemma, n in todo if n > 0]
    todo.sort(key=lambda x: (-x[1], x[0]))  # heaviest first; deterministic by lemma name on ties

    # Greedy bin-packing: assign each lemma to the annotator with the smallest load so far.
    loads: dict[str, int] = {name: 0 for name in args.annotators}
    buckets: dict[str, list[str]] = {name: [] for name in args.annotators}
    for lemma, n in todo:
        # Tie-break by annotator name order, so the split is fully deterministic.
        target = min(args.annotators, key=lambda a: (loads[a], a))
        buckets[target].append(lemma)
        loads[target] += n

    # Report.
    total_lemmas = sum(len(v) for v in buckets.values())
    total_clusters = sum(loads.values())
    print(f"Splitting {total_lemmas} lemmas / {total_clusters} unlabelled clusters across {len(args.annotators)} annotators:")
    for name in args.annotators:
        n_lemmas = len(buckets[name])
        n_clusters = loads[name]
        print(f"  {name:>12s}: {n_lemmas:>4d} lemmas, {n_clusters:>4d} clusters")

    if args.dry_run:
        print("\n(dry run — no files written)")
        return 0

    ASSIGNMENTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, lemmas in buckets.items():
        out = ASSIGNMENTS_DIR / f"{name}.txt"
        with out.open("w", encoding="utf-8") as f:
            for lemma in sorted(lemmas):
                f.write(f"{lemma}\n")
        print(f"  wrote {out.relative_to(BASE_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())