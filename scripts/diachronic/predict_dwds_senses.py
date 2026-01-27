#!/usr/bin/env python3
"""
Predict senses for DWDS corpus contexts using embedding similarity.

This script applies WSD to historical corpus data, producing sense predictions
that can be aggregated for diachronic analysis.

Usage:
    python scripts/diachronic/predict_dwds_senses.py
    python scripts/diachronic/predict_dwds_senses.py --dwds-file path/to/contexts.jsonl
    python scripts/diachronic/predict_dwds_senses.py --model intfloat/multilingual-e5-large
    python scripts/diachronic/predict_dwds_senses.py --senses-file data/evaluation/quick_eval/sense_variants.csv --text-column text_llm_examples_5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from diachronic import load_dwds_contexts, load_senses, predict_senses

# Configuration
DATA_DIR = Path("data")
DEFAULT_DWDS_FILE = DATA_DIR / "dwds_contexts" / "fetch_progress_25_words.jsonl"
DEFAULT_SENSES_FILE = DATA_DIR / "processed" / "germanet" / "wsd_senses.csv"
DEFAULT_TEXT_COLUMN = "text_for_embedding"  # Use 'text_llm_examples_5' for LLM variants
OUTPUT_DIR = DATA_DIR / "diachronic"
DEFAULT_MODEL = "intfloat/multilingual-e5-large"


def print_summary(df: pd.DataFrame) -> None:
    """Print prediction summary."""
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)

    total = len(df)
    predicted = df["predicted_sense_id"].notna().sum()

    print(f"Total contexts: {total}")
    print(f"Predicted: {predicted} ({predicted/total*100:.1f}%)")
    print(f"Failed: {total - predicted}")

    if predicted > 0:
        print(f"\nConfidence stats:")
        print(f"  Mean: {df['confidence'].mean():.3f}")
        print(f"  Min:  {df['confidence'].min():.3f}")
        print(f"  Max:  {df['confidence'].max():.3f}")

    print(f"\nPer-word breakdown:")
    for lemma in sorted(df["lemma"].unique()):
        lemma_df = df[df["lemma"] == lemma]
        n_senses = lemma_df["predicted_sense_id"].nunique()
        mean_conf = lemma_df["confidence"].mean()
        print(f"  {lemma}: {len(lemma_df)} contexts, {n_senses} unique senses, conf={mean_conf:.3f}")


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
        "--text-column",
        type=str,
        default=DEFAULT_TEXT_COLUMN,
        help="Column name for sense text to embed (default: text_for_embedding)",
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
        text_column=args.text_column,
        target_lemmas=target_lemmas,
    )

    # Check for words without senses
    words_with_senses = set(senses_df["lemma"].unique())
    words_without_senses = set(target_lemmas) - words_with_senses
    if words_without_senses:
        print(f"\nWarning: {len(words_without_senses)} words have no senses in file:")
        print(f"  {sorted(words_without_senses)}")
        contexts_df = contexts_df[contexts_df["lemma"].isin(words_with_senses)]

    # Predict senses
    results_df = predict_senses(
        contexts_df,
        senses_df,
        text_column=args.text_column,
        model_name=args.model,
    )

    # Print summary
    print_summary(results_df)

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = args.output or OUTPUT_DIR / "dwds_predictions.csv"

    # Select and order columns for output
    output_cols = [
        "lemma", "sentence", "year", "decade", "genre",
        "predicted_sense_id", "confidence", "sense_class", "sense_paraphrase",
        "date", "bibl", "corpus",
    ]
    # Only include columns that exist
    output_cols = [c for c in output_cols if c in results_df.columns]
    results_df[output_cols].to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()