#!/usr/bin/env python3
"""
Aggregate sense predictions by time period for diachronic analysis.

This script takes predictions from predict_dwds_senses.py and aggregates
sense distributions by decade for each lemma.

Usage:
    python scripts/diachronic/aggregate_temporal.py
    python scripts/diachronic/aggregate_temporal.py --input data/diachronic/dwds_predictions.csv
    python scripts/diachronic/aggregate_temporal.py --period-column year  # aggregate by year instead
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from diachronic import load_predictions, aggregate_by_period, compute_period_stats

# Configuration
DATA_DIR = Path("data")
DEFAULT_INPUT = DATA_DIR / "diachronic" / "dwds_predictions.csv"
OUTPUT_DIR = DATA_DIR / "diachronic"


def print_summary(agg_df: pd.DataFrame, stats_df: pd.DataFrame, period_column: str) -> None:
    """Print aggregation summary."""
    print("\n" + "=" * 60)
    print("AGGREGATION SUMMARY")
    print("=" * 60)

    print(f"\nPeriod column: {period_column}")
    print(f"Total lemmas: {agg_df['lemma'].nunique()}")
    print(f"Total periods: {agg_df[period_column].nunique()}")
    print(f"Period range: {agg_df[period_column].min()} - {agg_df[period_column].max()}")

    print(f"\nPer-lemma summary:")
    for lemma in sorted(agg_df["lemma"].unique()):
        lemma_stats = stats_df[stats_df["lemma"] == lemma]
        n_periods = len(lemma_stats)
        total_contexts = lemma_stats["n_contexts"].sum()
        avg_senses = lemma_stats["n_senses"].mean()
        print(f"  {lemma}: {n_periods} periods, {total_contexts} contexts, avg {avg_senses:.1f} senses/period")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Aggregate sense predictions by time period"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to predictions CSV from predict_dwds_senses.py",
    )
    parser.add_argument(
        "--period-column",
        type=str,
        default="decade",
        help="Column to aggregate by (default: decade)",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Output file prefix (default: auto-generated)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Load data
    predictions_df = load_predictions(args.input)

    # Validate period column
    if args.period_column not in predictions_df.columns:
        available = [c for c in predictions_df.columns if c in ["year", "decade"]]
        raise ValueError(
            f"Period column '{args.period_column}' not found. Available: {available}"
        )

    # Aggregate
    agg_df = aggregate_by_period(predictions_df, period_column=args.period_column)
    stats_df = compute_period_stats(predictions_df, period_column=args.period_column)

    # Print summary
    print_summary(agg_df, stats_df, args.period_column)

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix or f"temporal_{args.period_column}"

    agg_path = OUTPUT_DIR / f"{prefix}_sense_counts.csv"
    stats_path = OUTPUT_DIR / f"{prefix}_stats.csv"

    agg_df.to_csv(agg_path, index=False)
    stats_df.to_csv(stats_path, index=False)

    print(f"\nOutputs saved:")
    print(f"  Sense counts: {agg_path}")
    print(f"  Period stats: {stats_path}")


if __name__ == "__main__":
    main()