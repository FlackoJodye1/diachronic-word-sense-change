"""Visualization functions for diachronic analysis."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Style settings
plt.style.use("seaborn-v0_8-whitegrid")
FIGSIZE = (12, 6)
DPI = 150


def plot_sense_trajectory(
    counts_df: pd.DataFrame,
    lemma: str,
    period_column: str = "decade",
    output_dir: Optional[Path] = None,
) -> None:
    """Plot stacked area chart of sense proportions over time."""
    lemma_df = counts_df[counts_df["lemma"] == lemma].copy()

    if len(lemma_df) == 0:
        print(f"  No data for lemma: {lemma}")
        return

    # Pivot to get senses as columns
    pivot = lemma_df.pivot(
        index=period_column,
        columns="predicted_sense_id",
        values="proportion",
    ).fillna(0)

    # Sort columns by total proportion (most common first)
    col_order = pivot.sum().sort_values(ascending=False).index
    pivot = pivot[col_order]

    # Create figure
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Stacked area chart
    pivot.plot.area(ax=ax, alpha=0.7, linewidth=0.5)

    ax.set_xlabel(period_column.capitalize())
    ax.set_ylabel("Proportion")
    ax.set_title(f"Sense Distribution Over Time: {lemma}")
    ax.set_ylim(0, 1)
    ax.legend(title="Sense ID", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"trajectory_{lemma}.png"
        plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {output_path}")

    plt.close()


def plot_entropy_over_time(
    entropy_df: pd.DataFrame,
    lemmas: Optional[List[str]] = None,
    period_column: str = "decade",
    output_dir: Optional[Path] = None,
) -> None:
    """Plot entropy over time for selected lemmas."""
    if lemmas:
        df = entropy_df[entropy_df["lemma"].isin(lemmas)]
    else:
        df = entropy_df

    if len(df) == 0:
        print("  No entropy data to plot")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for lemma, group in df.groupby("lemma"):
        group_sorted = group.sort_values(period_column)
        ax.plot(
            group_sorted[period_column],
            group_sorted["entropy"],
            marker="o",
            label=lemma,
            linewidth=2,
            markersize=6,
        )

    ax.set_xlabel(period_column.capitalize())
    ax.set_ylabel("Entropy (bits)")
    ax.set_title("Sense Distribution Entropy Over Time")
    ax.legend(title="Lemma", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_".join(lemmas) if lemmas and len(lemmas) <= 3 else "all"
        output_path = output_dir / f"entropy_{suffix}.png"
        plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {output_path}")

    plt.close()


def plot_sense_heatmap(
    counts_df: pd.DataFrame,
    lemma: str,
    period_column: str = "decade",
    output_dir: Optional[Path] = None,
) -> None:
    """Plot heatmap of sense proportions over time."""
    lemma_df = counts_df[counts_df["lemma"] == lemma].copy()

    if len(lemma_df) == 0:
        print(f"  No data for lemma: {lemma}")
        return

    # Pivot to get senses as rows, periods as columns
    pivot = lemma_df.pivot(
        index="predicted_sense_id",
        columns=period_column,
        values="proportion",
    ).fillna(0)

    # Sort by total proportion
    row_order = pivot.sum(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[row_order]

    # Create figure
    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns)), max(6, len(pivot) * 0.5)))

    sns.heatmap(
        pivot,
        ax=ax,
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Proportion"},
    )

    ax.set_xlabel(period_column.capitalize())
    ax.set_ylabel("Sense ID")
    ax.set_title(f"Sense Proportion Heatmap: {lemma}")

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"heatmap_{lemma}.png"
        plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {output_path}")

    plt.close()


def plot_all_lemmas_summary(
    entropy_df: pd.DataFrame,
    period_column: str = "decade",
    output_dir: Optional[Path] = None,
) -> None:
    """Plot summary visualization of all lemmas."""
    # Average entropy per period across all lemmas
    period_avg = entropy_df.groupby(period_column)["entropy"].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Average entropy over time
    ax1 = axes[0]
    ax1.bar(period_avg[period_column], period_avg["entropy"], color="steelblue", alpha=0.7)
    ax1.set_xlabel(period_column.capitalize())
    ax1.set_ylabel("Mean Entropy (bits)")
    ax1.set_title("Average Sense Ambiguity Over Time")

    # Right: Box plot of entropy per period
    ax2 = axes[1]
    entropy_df.boxplot(column="entropy", by=period_column, ax=ax2)
    ax2.set_xlabel(period_column.capitalize())
    ax2.set_ylabel("Entropy (bits)")
    ax2.set_title("Entropy Distribution by Period")
    plt.suptitle("")  # Remove automatic title

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "summary_entropy.png"
        plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
        print(f"  Saved: {output_path}")

    plt.close()