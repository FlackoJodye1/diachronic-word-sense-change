#!/usr/bin/env python3
"""
Predict senses for DWDS corpus contexts using embedding similarity.

This script applies WSD to historical corpus data using the weighted representation
method (0.5 * definition + 0.5 * avg(examples)) for best accuracy.

Usage:
    python scripts/diachronic/predict_dwds_senses.py
    python scripts/diachronic/predict_dwds_senses.py --dwds-file path/to/contexts.jsonl
    python scripts/diachronic/predict_dwds_senses.py --model intfloat/multilingual-e5-large
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.diachronic import load_dwds_contexts, load_senses, predict_senses

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path("data")
DEFAULT_DWDS_FILE = DATA_DIR / "dwds_contexts" / "fetch_progress_25_words.jsonl"
DEFAULT_SENSES_FILE = DATA_DIR / "processed" / "germanet" / "wsd_senses.csv"
OUTPUT_DIR = DATA_DIR / "diachronic"
DEFAULT_MODEL = "intfloat/multilingual-e5-large"
# Always use weighted representation (0.5 * definition + 0.5 * avg(examples)) for best results
DEFAULT_REPRESENTATION = "weighted"


def log_summary(df: pd.DataFrame) -> None:
    """Log prediction summary."""
    total = len(df)
    predicted = df["predicted_sense_id"].notna().sum()

    logger.info("=" * 60)
    logger.info("PREDICTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total contexts: {total}")
    logger.info(f"Predicted: {predicted} ({predicted/total*100:.1f}%)")
    logger.info(f"Failed: {total - predicted}")

    if predicted > 0:
        logger.info("Confidence stats:")
        logger.info(f"  Mean: {df['confidence'].mean():.3f}")
        logger.info(f"  Min:  {df['confidence'].min():.3f}")
        logger.info(f"  Max:  {df['confidence'].max():.3f}")

    logger.info("Per-word breakdown:")
    for lemma in sorted(df["lemma"].unique()):
        lemma_df = df[df["lemma"] == lemma]
        n_senses = lemma_df["predicted_sense_id"].nunique()
        mean_conf = lemma_df["confidence"].mean()
        logger.info(f"  {lemma}: {len(lemma_df)} contexts, {n_senses} unique senses, conf={mean_conf:.3f}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict senses for DWDS corpus contexts"
    )
    parser.add_argument(
        "--dwds-file",
        type=Path,
        default=DEFAULT_DWDS_FILE,
        help="Path to DWDS contexts JSONL file",
    )
    parser.add_argument(
        "--senses-file",
        type=Path,
        default=DEFAULT_SENSES_FILE,
        help="Path to GermaNet senses CSV file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Embedding model to use",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: data/diachronic/dwds_predictions.csv)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Load data
    contexts_df = load_dwds_contexts(args.dwds_file)
    target_lemmas = contexts_df["lemma"].unique().tolist()
    senses_df = load_senses(
        args.senses_file,
        target_lemmas=target_lemmas,
    )

    # Check for words without senses
    words_with_senses = set(senses_df["lemma"].unique())
    words_without_senses = set(target_lemmas) - words_with_senses
    if words_without_senses:
        logger.warning(f"{len(words_without_senses)} words have no senses in file:")
        logger.warning(f"  {sorted(words_without_senses)}")
        contexts_df = contexts_df[contexts_df["lemma"].isin(words_with_senses)]

    # Predict senses using weighted representation (best method: 54.8% accuracy)
    logger.info(f"Using representation: {DEFAULT_REPRESENTATION}")
    results_df = predict_senses(
        contexts_df,
        senses_df,
        model_name=args.model,
        representation=DEFAULT_REPRESENTATION,
    )

    # Log summary
    log_summary(results_df)

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = args.output or OUTPUT_DIR / "dwds_predictions.csv"

    output_cols = [
        "lemma", "sentence", "year", "decade", "genre",
        "predicted_sense_id", "confidence", "sense_class", "sense_paraphrase",
        "date", "bibl", "corpus",
    ]
    output_cols = [c for c in output_cols if c in results_df.columns]
    results_df[output_cols].to_csv(output_path, index=False)

    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()