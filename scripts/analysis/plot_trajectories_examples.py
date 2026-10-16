#!/usr/bin/env python3
"""
§6.1 Illustrative Trajectories — 2-panel stacked-area figure.

Lemmas: Start, Schlüssel
Time range: 1910–1990 (1900 dropped — too sparse for Start, n=8)
Smoothing: centred 3-decade rolling mean (visualisation only)

Inputs (read-only):
  data/processed/diachronic/aggregated/pilot_v3_coarse_sense_counts.csv
  data/processed/germanet/sense_clusters_reviewed.json

Output:
  figures/trajectories_examples.pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
COUNTS_FILE   = BASE / "data/processed/diachronic/aggregated/pilot_v3_coarse_sense_counts.csv"
CLUSTERS_FILE = BASE / "data/processed/germanet/sense_clusters_reviewed.json"
OUTPUT_FILE   = BASE / "figures/trajectories_examples.pdf"

LEMMAS         = ["Start", "Schlüssel"]
DECADE_MIN     = 1910
DECADE_MAX     = 1990
SMOOTH_WINDOW  = 3  # centred 3-decade rolling mean


def trajectory_table(counts: pd.DataFrame, lemma: str) -> pd.DataFrame:
    """Return wide proportion table (rows=cluster, cols=decade) for the given lemma."""
    sub = counts[counts["lemma"] == lemma]
    wide = (sub.pivot_table(index="predicted_sense_id",
                            columns="decade",
                            values="proportion",
                            aggfunc="sum")
              .fillna(0.0))
    decades = [d for d in sorted(wide.columns) if DECADE_MIN <= int(d) <= DECADE_MAX]
    return wide[decades]


def smooth(wide: pd.DataFrame, window: int) -> pd.DataFrame:
    """Centred rolling mean along the decade axis, edges handled with min_periods=1."""
    return wide.T.rolling(window=window, center=True, min_periods=1).mean().T


def order_clusters_by_mass(wide: pd.DataFrame) -> list[str]:
    """Largest mean proportion first → drawn at bottom of stack."""
    return wide.mean(axis=1).sort_values(ascending=False).index.tolist()


def main() -> int:
    counts = pd.read_csv(COUNTS_FILE)
    with CLUSTERS_FILE.open(encoding="utf-8") as f:
        clusters = json.load(f)

    plt.rcParams.update({
        "font.family":   "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    palettes = [plt.cm.tab10.colors, plt.cm.Set2.colors]
    labels_per_lemma: dict[str, list[tuple[str, str]]] = {}
    decades_used: list[int] = []

    for ax, lemma, palette in zip(axes, LEMMAS, palettes):
        wide   = trajectory_table(counts, lemma)
        smooth_wide = smooth(wide, SMOOTH_WINDOW)
        order  = order_clusters_by_mass(smooth_wide)
        smooth_wide = smooth_wide.loc[order]
        decades = [int(d) for d in smooth_wide.columns]
        if not decades_used:
            decades_used = decades

        # Labels from the reviewed cluster JSON
        cluster_labels = {
            cid: clusters[lemma]["clusters"][cid]["label"]
            for cid in order
        }
        labels_per_lemma[lemma] = [(cid, cluster_labels[cid]) for cid in order]

        ax.stackplot(
            decades,
            smooth_wide.values,
            labels=[cluster_labels[cid] for cid in order],
            colors=palette[:len(order)],
            edgecolor="white",
            linewidth=0.5,
        )

        ax.set_xlim(DECADE_MIN, DECADE_MAX)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(list(range(DECADE_MIN, DECADE_MAX + 1, 10)))
        ax.set_xlabel("Decade")
        ax.set_title(f"$\\it{{{lemma}}}$")
        ax.tick_params(axis="both", direction="out", length=3)

        # Legend: cleanest spot is below the plot, but for stacked area
        # the upper-right corner is usually OK as long as the dominant cluster
        # is at the bottom. We try outside-right to avoid overlap.
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
            handlelength=1.2,
            handletextpad=0.5,
        )

    axes[0].set_ylabel("Sense proportion")

    fig.tight_layout()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ----- Report -----
    size_kb = OUTPUT_FILE.stat().st_size / 1024
    print(f"Wrote: {OUTPUT_FILE.relative_to(BASE)}  ({size_kb:.1f} KB)")
    print()
    print("Cluster labels used (in stack order — dominant first):")
    for lemma in LEMMAS:
        print(f"  {lemma}:")
        for cid, label in labels_per_lemma[lemma]:
            print(f"    {cid:<14} → {label}")
    print()
    print(f"Decade range shown: {decades_used[0]}–{decades_used[-1]}  ({len(decades_used)} decades)")
    return 0


if __name__ == "__main__":
    sys.exit(main())