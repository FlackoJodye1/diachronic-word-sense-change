#!/usr/bin/env python3
"""
Assignment-driven merge of reviewed sense-cluster JSON files.

Combines the canonical pre-multi-annotator mapping with three
per-annotator review files (Paul, Amandine, Nadia) into one
`sense_clusters_reviewed.json` covering all 503 lemmas.

Per-annotator JSON files contain all 503 lemma keys (they were seeded
from the canonical at dashboard launch), but each annotator only edited
the lemmas listed in `data/processed/sense_clustering/assignments/<name>.txt`.
This script overlays each annotator's slice on top of the canonical
mapping using the assignment files as the source of truth for ownership.

Pre-merge validation:
  1. Schema check: all four files must share the same per-lemma layout
     (`mapping`, `clusters` keys; each cluster has `label` + `members`).
  2. Assignment-level overlap check: every lemma must be assigned to
     exactly one annotator (the assignment-split script guarantees this,
     but we re-verify).
  3. Empty / malformed flagging: print but do not stop.

Inputs:
  data/processed/germanet/sense_clusters_reviewed.json
  data/processed/germanet/sense_clusters_reviewed_paul.json
  data/processed/germanet/sense_clusters_reviewed_amandine.json
  data/processed/germanet/sense_clusters_reviewed_nadia.json
  data/processed/sense_clustering/assignments/{paul,amandine,nadia}.txt

Outputs:
  data/processed/germanet/sense_clusters_reviewed.json              (overwritten)
  data/processed/germanet/sense_clusters_reviewed_pilot_backup.json (pre-merge backup)
  data/processed/germanet/cluster_merge_diagnostic.csv              (per-lemma source map)
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
GERMANET_DIR    = BASE_DIR / "data/processed/germanet"
ASSIGNMENTS_DIR = BASE_DIR / "data/processed/sense_clustering/assignments"

CANONICAL_FILE = GERMANET_DIR / "sense_clusters_reviewed.json"
BACKUP_FILE    = GERMANET_DIR / "sense_clusters_reviewed_pilot_backup.json"
DIAGNOSTIC_CSV = GERMANET_DIR / "cluster_merge_diagnostic.csv"

ANNOTATORS = ["paul", "amandine", "nadia"]


def _load_json(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"ERROR: required input not found: {path.relative_to(BASE_DIR)}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _load_assignment(name: str) -> set[str]:
    path = ASSIGNMENTS_DIR / f"{name}.txt"
    if not path.exists():
        sys.exit(f"ERROR: assignment file not found: {path.relative_to(BASE_DIR)}")
    with path.open(encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------

EXPECTED_TOP_KEYS = {"mapping", "clusters"}


def _schema_check(files: dict[str, dict]) -> None:
    """Each JSON: dict[lemma -> {mapping, clusters}]. Cluster: {label, members}."""
    print("Schema check")
    print("============")
    differences = []
    for label, data in files.items():
        if not isinstance(data, dict):
            differences.append(f"  {label}: top-level is {type(data).__name__}, expected dict")
            continue
        if not data:
            differences.append(f"  {label}: empty")
            continue
        sample_lemma = next(iter(data))
        sample = data[sample_lemma]
        keys = set(sample.keys()) if isinstance(sample, dict) else set()
        print(f"  {label}: {len(data)} lemmas; first lemma {sample_lemma!r} keys = {sorted(keys)}")
        if keys != EXPECTED_TOP_KEYS:
            differences.append(f"  {label}: expected per-lemma keys {EXPECTED_TOP_KEYS}, got {keys}")
    if differences:
        print("\nERROR: schema differences:")
        for d in differences:
            print(d)
        sys.exit(1)
    print("  → all four files share the expected layout.")
    print()


def _assignment_overlap_check(assignments: dict[str, set[str]]) -> None:
    """No lemma may appear in more than one annotator's assignment file."""
    print("Assignment-level overlap check")
    print("==============================")
    counter: Counter = Counter()
    owner: dict[str, list[str]] = {}
    for name, lemmas in assignments.items():
        for lemma in lemmas:
            counter[lemma] += 1
            owner.setdefault(lemma, []).append(name)
    duplicates = {l: owner[l] for l, n in counter.items() if n > 1}
    if duplicates:
        print(f"ERROR: {len(duplicates)} lemmas appear in multiple annotators' assignments:")
        for lemma, owners in sorted(duplicates.items()):
            print(f"  {lemma}: {owners}")
        sys.exit(2)
    total = sum(len(v) for v in assignments.values())
    print(f"  → {total} assigned lemmas, no overlap across {len(assignments)} annotators.")
    print()


def _flag_empty_or_malformed(files: dict[str, dict]) -> None:
    """Print lemmas with empty/null cluster data. Do not stop."""
    print("Empty / malformed entries")
    print("=========================")
    any_flagged = False
    for label, data in files.items():
        flagged = []
        for lemma, payload in data.items():
            if not isinstance(payload, dict):
                flagged.append((lemma, "non-dict payload"))
                continue
            clusters = payload.get("clusters")
            if clusters is None:
                flagged.append((lemma, "missing 'clusters'"))
            elif not clusters:
                flagged.append((lemma, "empty 'clusters'"))
            else:
                empty_clusters = [cid for cid, c in clusters.items()
                                  if not isinstance(c, dict) or not c.get("members")]
                if empty_clusters:
                    flagged.append((lemma, f"empty members in {len(empty_clusters)} cluster(s)"))
        if flagged:
            any_flagged = True
            print(f"  {label}: {len(flagged)} flagged")
            for lemma, reason in flagged[:8]:
                print(f"    {lemma}: {reason}")
            if len(flagged) > 8:
                print(f"    ... ({len(flagged) - 8} more)")
    if not any_flagged:
        print("  → no empty or malformed entries found.")
    print()


# ----------------------------------------------------------------------
# Merge
# ----------------------------------------------------------------------

def _merge(files: dict[str, dict],
           assignments: dict[str, set[str]]) -> tuple[dict, list[dict]]:
    """Overlay each annotator's slice onto the canonical. Returns (merged_dict, diagnostic_rows)."""
    canonical = files["canonical"]
    merged: dict = {lemma: payload for lemma, payload in canonical.items()}

    source: dict[str, str] = {lemma: "canonical" for lemma in canonical}
    for name in ANNOTATORS:
        ann_data = files[name]
        for lemma in assignments[name]:
            if lemma not in ann_data:
                print(f"WARN: lemma {lemma!r} (assigned to {name}) is missing from {name}'s file — using canonical.")
                continue
            merged[lemma] = ann_data[lemma]
            source[lemma] = name

    diagnostic = []
    for lemma in sorted(merged):
        payload = merged[lemma]
        clusters = payload.get("clusters", {}) if isinstance(payload, dict) else {}
        n_fine = sum(len(c.get("members", [])) for c in clusters.values())
        diagnostic.append({
            "lemma":         lemma,
            "source_file":   source.get(lemma, "canonical"),
            "n_fine_senses": n_fine,
            "n_clusters":    len(clusters),
        })
    # Re-order merged dict alphabetically for stable output.
    merged_sorted = {lemma: merged[lemma] for lemma in sorted(merged)}
    return merged_sorted, diagnostic


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> int:
    # ---- Load inputs ----
    files = {
        "canonical": _load_json(CANONICAL_FILE),
        "paul":      _load_json(GERMANET_DIR / "sense_clusters_reviewed_paul.json"),
        "amandine":  _load_json(GERMANET_DIR / "sense_clusters_reviewed_amandine.json"),
        "nadia":     _load_json(GERMANET_DIR / "sense_clusters_reviewed_nadia.json"),
    }
    print("Lemma counts per input file:")
    for label, d in files.items():
        labelled = sum(1 for v in d.values() for c in v.get("clusters", {}).values() if c.get("label"))
        total    = sum(len(v.get("clusters", {})) for v in d.values())
        print(f"  {label:<10s}  lemmas = {len(d):>4d}   clusters labelled = {labelled:>4d} / {total:>4d}")
    print()

    assignments = {name: _load_assignment(name) for name in ANNOTATORS}
    print("Assignment file sizes:")
    for name in ANNOTATORS:
        print(f"  {name:<10s}  assigned lemmas = {len(assignments[name])}")
    print()

    # ---- Validation ----
    _schema_check(files)
    _assignment_overlap_check(assignments)
    _flag_empty_or_malformed(files)

    # ---- Backup canonical ----
    shutil.copy2(CANONICAL_FILE, BACKUP_FILE)
    print(f"Backup created: {BACKUP_FILE.relative_to(BASE_DIR)}")
    print()

    # ---- Merge ----
    merged, diagnostic = _merge(files, assignments)

    # ---- Write merged JSON ----
    with CANONICAL_FILE.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    # ---- Write diagnostic CSV (avoid extra deps; CSV is small + simple) ----
    import csv
    DIAGNOSTIC_CSV.parent.mkdir(parents=True, exist_ok=True)
    with DIAGNOSTIC_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lemma", "source_file", "n_fine_senses", "n_clusters"])
        writer.writeheader()
        for row in diagnostic:
            writer.writerow(row)

    # ---- Report ----
    src_counts = Counter(r["source_file"] for r in diagnostic)
    print(f"Wrote merged: {CANONICAL_FILE.relative_to(BASE_DIR)}  ({len(merged)} lemmas)")
    print(f"Wrote diagnostic: {DIAGNOSTIC_CSV.relative_to(BASE_DIR)}")
    print()
    print("Source-file distribution in merged result:")
    for src, n in sorted(src_counts.items()):
        print(f"  {src:<10s}  {n:>4d} lemmas")
    total_clusters  = sum(r["n_clusters"]     for r in diagnostic)
    total_fine      = sum(r["n_fine_senses"]  for r in diagnostic)
    labelled        = sum(1 for v in merged.values()
                            for c in v.get("clusters", {}).values() if c.get("label"))
    print()
    print(f"Merged totals: {total_clusters} clusters, {total_fine} fine senses, "
          f"{labelled}/{total_clusters} clusters labelled "
          f"({100*labelled/total_clusters:.1f} %).")
    return 0


if __name__ == "__main__":
    sys.exit(main())