"""§6 case study: Präsident sense trajectory (1910-1990)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
COUNTS_FILE = BASE / "data/processed/diachronic/aggregated/pilot_v3_coarse_sense_counts.csv"
CLUSTERS_FILE = BASE / "data/processed/germanet/sense_clusters_reviewed.json"
OUTPUT_FILE = BASE / "figures/president_trajectory_plot.png"

LEMMA = "Präsident"
DECADE_MIN = 1910
DECADE_MAX = 1990
SMOOTH_WINDOW = 3
PALETTE = ["#1f77b4", "#ff7f0e"]  # tab10 blue/orange


def trajectory_table(counts: pd.DataFrame, lemma: str) -> pd.DataFrame:
    sub = counts[counts["lemma"] == lemma]
    wide = (sub.pivot_table(index="predicted_sense_id",
                            columns="decade",
                            values="proportion",
                            aggfunc="sum")
              .fillna(0.0))
    decades = [d for d in sorted(wide.columns) if DECADE_MIN <= int(d) <= DECADE_MAX]
    return wide[decades]


def smooth(wide: pd.DataFrame, window: int) -> pd.DataFrame:
    return wide.T.rolling(window=window, center=True, min_periods=1).mean().T


def order_by_mass(wide: pd.DataFrame) -> list[str]:
    return wide.mean(axis=1).sort_values(ascending=False).index.tolist()


def main() -> None:
    counts = pd.read_csv(COUNTS_FILE)
    with CLUSTERS_FILE.open(encoding="utf-8") as f:
        clusters = json.load(f)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    })

    wide = trajectory_table(counts, LEMMA)
    smooth_wide = smooth(wide, SMOOTH_WINDOW)
    order = order_by_mass(smooth_wide)
    smooth_wide = smooth_wide.loc[order]
    decades = [int(d) for d in smooth_wide.columns]
    labels = [clusters[LEMMA]["clusters"][cid]["label"] for cid in order]

    fig, ax = plt.subplots(figsize=(3.5, 3.4))

    ax.stackplot(
        decades,
        smooth_wide.values,
        labels=labels,
        colors=PALETTE[:len(order)],
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_xlim(DECADE_MIN, DECADE_MAX)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(list(range(DECADE_MIN, DECADE_MAX + 1, 20)))
    ax.set_xlabel("Decade")
    ax.set_ylabel("Sense proportion")
    ax.tick_params(axis="both", direction="out", length=3)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=1,
        frameon=False,
        handlelength=1.4,
        handletextpad=0.6,
    )

    fig.tight_layout()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.close(fig)

    from PIL import Image
    with Image.open(OUTPUT_FILE) as img:
        w_px, h_px = img.size
    print(f"Wrote: {OUTPUT_FILE.relative_to(BASE)}")
    print(f"Dimensions: {w_px}x{h_px} px at 300 dpi ({w_px/300:.2f}x{h_px/300:.2f} inches)")


if __name__ == "__main__":
    main()