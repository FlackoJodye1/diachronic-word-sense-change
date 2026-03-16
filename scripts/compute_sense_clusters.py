#!/usr/bin/env python3
"""
Run coarse sense clustering on all pilot words and save the mapping + summary.

Clusters fine-grained GermaNet senses per lemma using agglomerative clustering
(average linkage, cosine distance) on E5-large embeddings of core_gloss.

No upper bound on cluster count — the threshold determines granularity organically.
min_clusters=2 ensures every word gets at least one split.
Sparse word-decade cells produced by high-polysemy words are filtered at analysis time.

Pipeline position:
    senses_pilot.jsonl → [this script] → sense_clusters_0.10.json
                                       → sense_clustering/t0.10/cluster_summary.csv

Usage:
    python scripts/compute_sense_clusters.py

    python scripts/compute_sense_clusters.py --threshold 0.12

    python scripts/compute_sense_clusters.py --threshold 0.10 --dry-run
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.diachronic.io import load_senses
from src.wsd.embedder import Embedder
from src.wsd.embedding_strategies import embed_with_representation
from src.wsd.sense_clustering import cluster_all_lemmas, compute_cluster_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
SENSES_FILE = BASE_DIR / "data" / "processed" / "germanet" / "senses_pilot.jsonl"
CLUSTERS_DIR = BASE_DIR / "data" / "processed" / "germanet"
SUMMARY_BASE_DIR = BASE_DIR / "data" / "processed" / "sense_clustering"

MODEL_NAME = "intfloat/multilingual-e5-large"
REPRESENTATION = "core_gloss"
PASSAGE_PREFIX = "passage: "


def main():
    parser = argparse.ArgumentParser(description="Cluster GermaNet senses for all pilot words")
    parser.add_argument("--threshold", type=float, default=0.10,
                        help="Cosine distance threshold (default: 0.10)")
    parser.add_argument("--min-clusters", type=int, default=2,
                        help="Minimum clusters per lemma (default: 2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load and embed senses but do not write output files")
    args = parser.parse_args()

    t = args.threshold
    t_str = f"{t:.2f}".replace(".", "")  # e.g. "010"
    clusters_out = CLUSTERS_DIR / f"sense_clusters_{t:.2f}.json"
    summary_dir = SUMMARY_BASE_DIR / f"t{t_str}"

    logger.info(f"Loading senses from {SENSES_FILE}")
    senses_df = load_senses(SENSES_FILE)
    n_lemmas = senses_df["lemma"].nunique()
    n_senses = len(senses_df)
    logger.info(f"  {n_senses} senses across {n_lemmas} lemmas")

    logger.info(f"Loading embedder: {MODEL_NAME}")
    embedder = Embedder(model_name=MODEL_NAME, normalize=True)

    logger.info(f"Embedding senses with representation='{REPRESENTATION}' (prefix='{PASSAGE_PREFIX}')")
    embeddings = embed_with_representation(
        senses_df, embedder, REPRESENTATION, passage_prefix=PASSAGE_PREFIX
    )
    logger.info(f"  Embeddings shape: {embeddings.shape}")

    logger.info(f"Clustering at threshold={t}, min_clusters={args.min_clusters}")
    mapping_dict = cluster_all_lemmas(
        senses_df, embeddings,
        threshold=t,
        min_clusters=args.min_clusters,
    )

    logger.info("Computing cluster summary")
    summary_df = compute_cluster_summary(mapping_dict, senses_df, embeddings, threshold=t)

    n_constrained = summary_df["constrained"].sum()
    mean_fine = summary_df["n_fine_senses"].mean()
    mean_coarse = summary_df["n_coarse_clusters"].mean()
    mean_reduction = summary_df["reduction_ratio"].mean()
    logger.info(f"  Words: {len(summary_df)}")
    logger.info(f"  Mean fine senses: {mean_fine:.1f}  →  Mean coarse clusters: {mean_coarse:.1f}")
    logger.info(f"  Mean reduction ratio: {mean_reduction:.1%}")
    logger.info(f"  Words where min_clusters constraint fired: {n_constrained}")

    if args.dry_run:
        logger.info("Dry run — no files written.")
        print(summary_df[["lemma", "n_fine_senses", "n_organic_clusters", "n_coarse_clusters",
                           "constrained", "reduction_ratio"]].to_string(index=False))
        return

    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_out = summary_dir / "cluster_summary.csv"

    logger.info(f"Writing cluster mapping to {clusters_out}")
    with open(clusters_out, "w", encoding="utf-8") as f:
        json.dump(mapping_dict, f, ensure_ascii=False, indent=2)

    logger.info(f"Writing cluster summary to {summary_out}")
    summary_df.to_csv(summary_out, index=False)

    logger.info("Done.")
    print(f"\nOutput files:")
    print(f"  {clusters_out}")
    print(f"  {summary_out}")


if __name__ == "__main__":
    main()