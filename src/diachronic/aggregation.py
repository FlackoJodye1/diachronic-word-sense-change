"""Temporal aggregation functions for diachronic analysis."""

from __future__ import annotations

import pandas as pd


def aggregate_by_period(
    df: pd.DataFrame,
    period_column: str = "decade",
) -> pd.DataFrame:
    """Aggregate sense distributions by time period.

    Args:
        df: DataFrame with predictions (must have 'lemma', 'predicted_sense_id', period_column)
        period_column: Column to group by (default: 'decade')

    Returns:
        DataFrame with aggregated sense counts per lemma per period
    """
    print(f"\nAggregating by {period_column}...")

    # Group by lemma, period, and sense
    agg = (
        df.groupby(["lemma", period_column, "predicted_sense_id"])
        .size()
        .reset_index(name="count")
    )

    # Add total per lemma-period for calculating proportions
    totals = (
        df.groupby(["lemma", period_column])
        .size()
        .reset_index(name="total")
    )
    agg = agg.merge(totals, on=["lemma", period_column])
    agg["proportion"] = agg["count"] / agg["total"]

    # Add sense metadata if available
    if "sense_class" in df.columns:
        sense_meta = df[["predicted_sense_id", "sense_class"]].drop_duplicates()
        agg = agg.merge(sense_meta, on="predicted_sense_id", how="left")

    print(f"  Created {len(agg)} lemma-period-sense combinations")
    return agg


def compute_period_stats(
    df: pd.DataFrame,
    period_column: str = "decade",
) -> pd.DataFrame:
    """Compute summary statistics per lemma per period.

    Args:
        df: DataFrame with predictions
        period_column: Column to group by

    Returns:
        DataFrame with stats per lemma per period
    """
    print(f"\nComputing period statistics...")

    stats = []
    for (lemma, period), group in df.groupby(["lemma", period_column]):
        sense_counts = group["predicted_sense_id"].value_counts()
        n_senses = len(sense_counts)
        n_contexts = len(group)
        dominant_sense = sense_counts.index[0] if len(sense_counts) > 0 else None
        dominant_count = sense_counts.iloc[0] if len(sense_counts) > 0 else 0
        mean_confidence = group["confidence"].mean() if "confidence" in group.columns else None

        stats.append({
            "lemma": lemma,
            period_column: period,
            "n_contexts": n_contexts,
            "n_senses": n_senses,
            "dominant_sense": dominant_sense,
            "dominant_proportion": dominant_count / n_contexts if n_contexts > 0 else 0,
            "mean_confidence": mean_confidence,
        })

    stats_df = pd.DataFrame(stats)
    print(f"  Computed stats for {len(stats_df)} lemma-period pairs")
    return stats_df