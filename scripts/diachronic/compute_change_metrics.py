#!/usr/bin/env python3
"""
Compute diachronic sense change metrics.

This script analyzes sense distribution changes over time using:
- Entropy of sense distributions (higher = more ambiguous)
- Jensen-Shannon divergence between adjacent periods
- Sense emergence and decline detection

Usage:
    python scripts/diachronic/compute_change_metrics.py
    python scripts/diachronic/compute_change_metrics.py --counts data/diachronic/temporal_decade_sense_counts.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.diachronic import (
    load_sense_counts,
    compute_entropy_per_period,
    compute_period_divergence,
    detect_sense_changes,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path("data")
DEFAULT_COUNTS_FILE = DATA_DIR / "diachronic" / "temporal_decade_sense_counts.csv"
OUTPUT_DIR = DATA_DIR / "diachronic"


def log_summary(
    entropy_df: pd.DataFrame,
    divergence_df: pd.DataFrame,
    emergence_df: pd.DataFrame,
    decline_df: pd.DataFrame,
    period_column: str,
) -> None:
    """Log metrics summary."""
    logger.info("=" * 60)
    logger.info("CHANGE METRICS SUMMARY")
    logger.info("=" * 60)
    logger.info("Entropy statistics:")
    logger.info(f"  Mean entropy: {entropy_df['entropy'].mean():.3f}")
    logger.info(f"  Max entropy: {entropy_df['entropy'].max():.3f}")
    logger.info(f"  Mean normalized entropy: {entropy_df['normalized_entropy'].mean():.3f}")

    if len(divergence_df) > 0:
        logger.info("Divergence statistics:")
        logger.info(f"  Mean JSD: {divergence_df['jsd'].mean():.3f}")
        logger.info(f"  Max JSD: {divergence_df['jsd'].max():.3f}")
        biggest = divergence_df.nlargest(3, "jsd")
        logger.info("  Biggest distribution shifts:")
        for _, row in biggest.iterrows():
            logger.info(f"    {row['lemma']}: {row[f'{period_column}_from']}->{row[f'{period_column}_to']} (JSD={row['jsd']:.3f})")

    logger.info("Sense dynamics:")
    logger.info(f"  Total emergence events: {len(emergence_df)}")
    logger.info(f"  Total decline events: {len(decline_df)}")

    if len(emergence_df) > 0:
        logger.info("  Notable emergences:")
        for _, row in emergence_df.nlargest(3, "change").iterrows():
            logger.info(f"    {row['lemma']} sense {row['sense_id']}: {row['proportion_from']:.2f} -> {row['proportion_to']:.2f}")

    if len(decline_df) > 0:
        logger.info("  Notable declines:")
        for _, row in decline_df.nsmallest(3, "change").iterrows():
            logger.info(f"    {row['lemma']} sense {row['sense_id']}: {row['proportion_from']:.2f} -> {row['proportion_to']:.2f}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute diachronic sense change metrics"
    )
    parser.add_argument(
        "--counts",
        type=Path,
        default=DEFAULT_COUNTS_FILE,
        help="Path to sense counts CSV from aggregate_temporal.py",
    )
    parser.add_argument(
        "--period-column",
        type=str,
        default="decade",
        help="Period column name (default: decade)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Proportion threshold for emergence/decline detection (default: 0.1)",
    )
    parser.add_argument(
        "--min-contexts",
        type=int,
        default=0,
        help="Minimum contexts per word-decade to include in metrics (default: 0 = no filter). "
             "Word-decades below this threshold are excluded before computing JSD and other metrics.",
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
    counts_df = load_sense_counts(args.counts)

    # Apply sparse-data filter
    if args.min_contexts > 0:
        wd_before = counts_df.groupby(["lemma", counts_df.columns[1]]).ngroups
        counts_df = counts_df[counts_df["total"] >= args.min_contexts]
        wd_after = counts_df.groupby(["lemma", counts_df.columns[1]]).ngroups
        logger.info(f"Sparse-data filter (min_contexts={args.min_contexts}):")
        logger.info(f"  Word-decades before: {wd_before}, after: {wd_after} (removed {wd_before - wd_after})")

    # Infer period column from data if needed
    period_column = args.period_column
    if period_column not in counts_df.columns:
        candidates = ["decade", "year"]
        for c in candidates:
            if c in counts_df.columns:
                period_column = c
                logger.info(f"Using period column: {period_column}")
                break

    # Compute metrics
    entropy_df = compute_entropy_per_period(counts_df, period_column)
    divergence_df = compute_period_divergence(counts_df, period_column)
    emergence_df, decline_df = detect_sense_changes(
        counts_df,
        period_column,
        emergence_threshold=args.threshold,
        decline_threshold=args.threshold,
    )

    # Log summary
    log_summary(entropy_df, divergence_df, emergence_df, decline_df, period_column)

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix or f"metrics_{period_column}"

    entropy_path = OUTPUT_DIR / f"{prefix}_entropy.csv"
    divergence_path = OUTPUT_DIR / f"{prefix}_divergence.csv"
    emergence_path = OUTPUT_DIR / f"{prefix}_emergence.csv"
    decline_path = OUTPUT_DIR / f"{prefix}_decline.csv"

    entropy_df.to_csv(entropy_path, index=False)
    divergence_df.to_csv(divergence_path, index=False)
    emergence_df.to_csv(emergence_path, index=False)
    decline_df.to_csv(decline_path, index=False)

    logger.info("Outputs saved:")
    logger.info(f"  Entropy: {entropy_path}")
    logger.info(f"  Divergence: {divergence_path}")
    logger.info(f"  Emergence: {emergence_path}")
    logger.info(f"  Decline: {decline_path}")


if __name__ == "__main__":
    main()