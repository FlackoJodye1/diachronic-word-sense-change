#!/usr/bin/env python3
"""
Map fine-grained sense predictions to coarse cluster IDs.

Reads pilot_predictions.csv and sense_clusters_reviewed.json, replaces
predicted_sense_id with the coarse cluster ID, and writes
pilot_predictions_coarse.csv. All other columns are preserved unchanged.

Rows whose predicted_sense_id is not found in the mapping are dropped with
a warning (should not happen if sense_clusters_reviewed.json is complete).

Pipeline position:
    pilot_predictions.csv + sense_clusters_reviewed.json
        → [this script]
        → pilot_predictions_coarse.csv

Usage:
    python scripts/diachronic/apply_coarse_mapping.py

    python scripts/diachronic/apply_coarse_mapping.py \\
        --input  data/processed/diachronic/predictions/pilot_predictions.csv \\
        --mapping data/processed/germanet/sense_clusters_reviewed.json \\
        --output data/processed/diachronic/predictions/pilot_predictions_coarse.csv
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT   = BASE_DIR / "data" / "processed" / "diachronic" / "predictions" / "pilot_predictions.csv"
DEFAULT_MAPPING = BASE_DIR / "data" / "processed" / "germanet" / "sense_clusters_reviewed.json"
DEFAULT_OUTPUT  = BASE_DIR / "data" / "processed" / "diachronic" / "predictions" / "pilot_predictions_coarse.csv"


def main():
    parser = argparse.ArgumentParser(description="Map fine sense predictions to coarse cluster IDs")
    parser.add_argument("--input",   type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output",  type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    logger.info(f"Loading predictions from {args.input}")
    df = pd.read_csv(args.input)
    logger.info(f"  {len(df)} rows, {df['lemma'].nunique()} lemmas")

    logger.info(f"Loading cluster mapping from {args.mapping}")
    with open(args.mapping, encoding="utf-8") as f:
        mapping_dict = json.load(f)

    # Build flat lookup: (lemma, fine_sense_id) -> coarse_cluster_id
    lookup = {
        (lemma, fine_id): coarse_id
        for lemma, data in mapping_dict.items()
        for fine_id, coarse_id in data["mapping"].items()
    }

    # Apply mapping
    def remap(row):
        return lookup.get((row["lemma"], row["predicted_sense_id"]))

    df["predicted_sense_id"] = df.apply(remap, axis=1)

    # Drop unmapped rows
    n_missing = df["predicted_sense_id"].isna().sum()
    if n_missing:
        missing = df[df["predicted_sense_id"].isna()][["lemma", "predicted_sense_id"]].value_counts()
        logger.warning(f"  {n_missing} rows could not be mapped — dropping:")
        logger.warning(f"\n{missing}")
        df = df.dropna(subset=["predicted_sense_id"])

    logger.info(f"  {len(df)} rows after mapping ({n_missing} dropped)")

    # Drop fine-grained sense metadata — don't map cleanly to coarse clusters
    # and cause duplicate rows in downstream aggregation.
    # Coarse cluster labels are in sense_clusters_reviewed.json.
    for col in ("sense_class", "sense_paraphrase"):
        if col in df.columns:
            df = df.drop(columns=[col])
            logger.info(f"  Dropped column '{col}' (fine-grained, not meaningful at coarse level)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    logger.info(f"Written to {args.output}")


if __name__ == "__main__":
    main()