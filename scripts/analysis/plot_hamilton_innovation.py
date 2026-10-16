"""Hamilton's Law of Innovation: cluster polysemy vs mean JSD."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/processed/analysis/per_lemma_summary.csv"
OUTPUT = ROOT / "figures/hamilton_innovation.pdf"


def main() -> None:
    df = pd.read_csv(INPUT)
    df = df.dropna(subset=["mean_jsd", "n_clusters_coarse"]).copy()
    n = len(df)

    rho, p = spearmanr(df["n_clusters_coarse"], df["mean_jsd"])

    rng = np.random.default_rng(42)
    df["x_jitter"] = df["n_clusters_coarse"] + rng.uniform(-0.18, 0.18, size=n)

    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    sns.regplot(
        x="x_jitter",
        y="mean_jsd",
        data=df,
        ax=ax,
        scatter_kws={"alpha": 0.4, "s": 14, "color": "#2b6cb0", "edgecolor": "none"},
        line_kws={"color": "#c53030", "linewidth": 1.5},
        ci=95,
    )

    ax.set_xlabel("Number of reviewed clusters")
    ax.set_ylabel("Mean Jensen-Shannon distance")

    xmin = int(df["n_clusters_coarse"].min())
    xmax = int(df["n_clusters_coarse"].max())
    ax.set_xticks(range(xmin, xmax + 1))

    ax.text(
        0.97,
        0.97,
        f"Spearman $r$ = +{rho:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.6", linewidth=0.6),
    )

    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUTPUT}")
    print(f"n = {n} (Spearman r = +{rho:.3f}, p = {p:.2e})")


if __name__ == "__main__":
    main()