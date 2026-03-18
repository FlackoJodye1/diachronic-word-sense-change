#!/usr/bin/env python3
"""
Generate per-lemma Markdown review sheets from a sense cluster mapping.

Each sheet lists clusters with their member senses (synset_id, core_gloss,
llm_definition) and a label placeholder for manual annotation.

Pipeline position:
    sense_clusters_0.10.json + senses_pilot.jsonl
        → [this script]
        → sense_clustering/t010/review/{lemma}_review.md (x90)

Usage:
    python scripts/clustering/review_sense_clusters.py

    python scripts/clustering/review_sense_clusters.py --threshold 0.12

    python scripts/clustering/review_sense_clusters.py --lemma schlagen spielen Geist
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.diachronic.io import load_senses

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
SENSES_FILE = BASE_DIR / "data" / "processed" / "germanet" / "senses_pilot.jsonl"
CLUSTERS_DIR = BASE_DIR / "data" / "processed" / "germanet"
SUMMARY_BASE_DIR = BASE_DIR / "data" / "processed" / "sense_clustering"


def render_review_sheet(
    lemma: str,
    lemma_data: dict,
    senses_lookup: dict,
    constrained: bool,
    n_fine: int,
) -> str:
    """Render a Markdown review sheet for one lemma."""
    clusters = lemma_data["clusters"]
    n_coarse = len(clusters)
    constrained_flag = "**Yes** ⚠" if constrained else "No"

    lines = [
        f"# {lemma} — Cluster Review",
        f"",
        f"**{n_fine} fine senses → {n_coarse} coarse clusters** | Constrained: {constrained_flag}",
        f"",
        f"---",
    ]

    for cluster_id in sorted(clusters.keys()):
        cluster = clusters[cluster_id]
        members = cluster["members"]
        lines += [
            f"",
            f"## {cluster_id} — label: ___________",
            f"",
        ]
        for sid in members:
            sense = senses_lookup.get(sid, {})
            core_gloss = sense.get("core_gloss", "—")
            llm_def = sense.get("llm_definition", "—")
            sense_class = sense.get("class", "—")
            lines += [
                f"- **{sid}** `{sense_class}` — {core_gloss}",
                f"  > {llm_def}",
                f"",
            ]
        lines.append("---")

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Generate Markdown review sheets for sense clusters"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.10,
        help="Threshold used when generating clusters (default: 0.10)"
    )
    parser.add_argument(
        "--lemma", nargs="+", default=None,
        help="Only generate sheets for these lemmas (default: all)"
    )
    args = parser.parse_args()

    t = args.threshold
    t_str = f"{t:.2f}".replace(".", "")
    clusters_file = CLUSTERS_DIR / f"sense_clusters_{t:.2f}.json"
    summary_file = SUMMARY_BASE_DIR / f"t{t_str}" / "cluster_summary.csv"
    review_dir = SUMMARY_BASE_DIR / f"t{t_str}" / "review"

    if not clusters_file.exists():
        logger.error(f"Cluster file not found: {clusters_file}")
        sys.exit(1)

    logger.info(f"Loading cluster mapping from {clusters_file}")
    with open(clusters_file, "r", encoding="utf-8") as f:
        mapping_dict = json.load(f)

    logger.info(f"Loading senses from {SENSES_FILE}")
    senses_df = load_senses(SENSES_FILE)
    senses_lookup = {row["synset_id"]: row.to_dict() for _, row in senses_df.iterrows()}

    # Load constrained flags from summary CSV
    constrained_map = {}
    n_fine_map = {}
    if summary_file.exists():
        import pandas as pd
        summary_df = pd.read_csv(summary_file)
        constrained_map = dict(zip(summary_df["lemma"], summary_df["constrained"]))
        n_fine_map = dict(zip(summary_df["lemma"], summary_df["n_fine_senses"]))

    lemmas = args.lemma if args.lemma else sorted(mapping_dict.keys())
    review_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {len(lemmas)} review sheets → {review_dir}")
    for lemma in lemmas:
        if lemma not in mapping_dict:
            logger.warning(f"  Lemma '{lemma}' not found in cluster mapping — skipping")
            continue
        sheet = render_review_sheet(
            lemma=lemma,
            lemma_data=mapping_dict[lemma],
            senses_lookup=senses_lookup,
            constrained=bool(constrained_map.get(lemma, False)),
            n_fine=int(n_fine_map.get(lemma, 0)),
        )
        out_path = review_dir / f"{lemma}_review.md"
        out_path.write_text(sheet, encoding="utf-8")

    logger.info(f"Done. {len(lemmas)} review sheets written to {review_dir}")


if __name__ == "__main__":
    main()