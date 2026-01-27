#!/usr/bin/env python3
"""
Visualize diachronic sense trajectories.

Creates plots showing how sense distributions change over time:
- Stacked area charts of sense proportions
- Entropy over time
- Heatmaps of sense-period distributions

Usage:
    python scripts/diachronic/visualize_trajectories.py
    python scripts/diachronic/visualize_trajectories.py --lemma Bahn
    python scripts/diachronic/visualize_trajectories.py --all-lemmas
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from diachronic.visualization import (
    plot_sense_trajectory,
    plot_entropy_over_time,
    plot_sense_heatmap,
    plot_all_lemmas_summary,
)

# Configuration
DATA_DIR = Path("data")
DEFAULT_COUNTS_FILE = DATA_DIR / "diachronic" / "temporal_decade_sense_counts.csv"
DEFAULT_ENTROPY_FILE = DATA_DIR / "diachronic" / "metrics_decade_entropy.csv"
OUTPUT_DIR = DATA_DIR / "diachronic" / "plots"


def load_data(
    counts_path: Path,
    entropy_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load sense counts and entropy data."""
    print(f"Loading data...")
    counts_df = pd.read_csv(counts_path)
    entropy_df = pd.read_csv(entropy_path)
    print(f"  Loaded {counts_df['lemma'].nunique()} lemmas")
    return counts_df, entropy_df


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize diachronic sense trajectories"
    )
    parser.add_argument(
        "--counts",
        type=Path,
        default=DEFAULT_COUNTS_FILE,
        help="Path to sense counts CSV",
    )
    parser.add_argument(
        "--entropy",
        type=Path,
        default=DEFAULT_ENTROPY_FILE,
        help="Path to entropy CSV",
    )
    parser.add_argument(
        "--lemma",
        type=str,
        default=None,
        help="Specific lemma to visualize",
    )
    parser.add_argument(
        "--all-lemmas",
        action="store_true",
        help="Generate plots for all lemmas",
    )
    parser.add_argument(
        "--period-column",
        type=str,
        default="decade",
        help="Period column name",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for plots",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Load data
    counts_df, entropy_df = load_data(args.counts, args.entropy)

    # Determine which lemmas to plot
    all_lemmas = sorted(counts_df["lemma"].unique())

    if args.lemma:
        lemmas_to_plot = [args.lemma]
    elif args.all_lemmas:
        lemmas_to_plot = all_lemmas
    else:
        # Default: plot first 3 lemmas as examples
        lemmas_to_plot = all_lemmas[:3]

    print(f"\nGenerating visualizations for {len(lemmas_to_plot)} lemmas...")

    # Generate per-lemma plots
    for lemma in lemmas_to_plot:
        print(f"\n  Processing: {lemma}")
        plot_sense_trajectory(counts_df, lemma, args.period_column, args.output_dir)
        plot_sense_heatmap(counts_df, lemma, args.period_column, args.output_dir)

    # Generate summary plots
    print(f"\nGenerating summary plots...")
    plot_entropy_over_time(entropy_df, lemmas_to_plot, args.period_column, args.output_dir)
    plot_all_lemmas_summary(entropy_df, args.period_column, args.output_dir)

    print(f"\nAll visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()