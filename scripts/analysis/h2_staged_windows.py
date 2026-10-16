#!/usr/bin/env python3
"""
§4 H2 — Distribution-grounding test via staged historical windows.

Hypothesis: subjective polysemy ratings reflect present-day intuitions, so the
correlation between survey ratings and model-inferred entropy should be
strongest on the most recent decade alone and weaken (or at least not
strengthen) as older decades are added.

For each historical window D_k:
    k=1 → 1990s only
    k=2 → 1980s–1990s
    ...
    k=9 → 1910s–1990s
compute:
    - mean entropy per lemma over supported decades in D_k
    - Spearman correlation with subjective_polysemy_mean (inner-join)
    - n, two-tailed p, 95% CI via Fisher z-transform

Inputs (read-only):
  data/processed/diachronic/metrics/pilot_v3_coarse_entropy.csv
  data/processed/diachronic/aggregated/pilot_v3_coarse_stats.csv
  data/processed/survey/survey_polysemy_per_lemma.csv

Outputs:
  data/processed/analysis/h2_staged_correlations.csv
  figures/h2_staged_windows.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parents[2]

ENTROPY_FILE = BASE / "data/processed/diachronic/metrics/pilot_v3_coarse_entropy.csv"
STATS_FILE   = BASE / "data/processed/diachronic/aggregated/pilot_v3_coarse_stats.csv"
SURVEY_FILE  = BASE / "data/processed/survey/survey_polysemy_per_lemma.csv"

OUT_CSV = BASE / "data/processed/analysis/h2_staged_correlations.csv"
OUT_FIG = BASE / "figures/h2_staged_windows.pdf"

MIN_CONTEXTS = 10
WINDOW_END   = 1990
MAX_K        = 9   # k=1 → 1990s only; k=9 → 1910s–1990s


def fisher_ci(r: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """95% CI on Spearman r via Fisher z-transform."""
    if n < 4 or not -1.0 < r < 1.0:
        return float("nan"), float("nan")
    z   = np.arctanh(r)
    se  = 1.0 / np.sqrt(n - 3)
    crit = 1.959963984540054  # 95% two-tailed
    return float(np.tanh(z - crit * se)), float(np.tanh(z + crit * se))


def main() -> int:
    ent    = pd.read_csv(ENTROPY_FILE)[["lemma", "decade", "entropy"]]
    stats  = pd.read_csv(STATS_FILE)[["lemma", "decade", "n_contexts"]]
    survey = pd.read_csv(SURVEY_FILE)[["lemma", "subjective_polysemy_mean"]]

    supported = {
        (str(l), int(d))
        for l, d in stats.loc[stats["n_contexts"] >= MIN_CONTEXTS, ["lemma", "decade"]].values
    }
    ent = ent[[(l, int(d)) in supported for l, d in zip(ent["lemma"], ent["decade"])]]

    rows = []
    for k in range(1, MAX_K + 1):
        start = WINDOW_END - 10 * (k - 1)
        label = "1990s only" if k == 1 else f"{start}s-1990s"

        window_ent = ent[(ent["decade"] >= start) & (ent["decade"] <= WINDOW_END)]
        mean_h = (window_ent.groupby("lemma")["entropy"]
                            .mean()
                            .rename("mean_entropy_window")
                            .reset_index())
        joined = mean_h.merge(survey, on="lemma", how="inner").dropna()
        if len(joined) < 4:
            continue

        sr, sp = spearmanr(joined["subjective_polysemy_mean"],
                           joined["mean_entropy_window"])
        ci_lo, ci_hi = fisher_ci(float(sr), len(joined))
        rows.append({
            "window_label": label,
            "window_start": start,
            "window_end":   WINDOW_END,
            "k":            k,
            "n_lemmas":     int(len(joined)),
            "spearman_r":   float(sr),
            "spearman_p":   float(sp),
            "ci_lo":        ci_lo,
            "ci_hi":        ci_hi,
        })

    df = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df[["window_label", "window_start", "window_end", "k",
        "n_lemmas", "spearman_r", "spearman_p"]].to_csv(OUT_CSV, index=False)

    # ---------- Figure ----------
    plt.rcParams.update({
        "font.family":     "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.titlesize":  11,
        "axes.labelsize":  10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
    })
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = df["window_start"].values
    y = df["spearman_r"].values

    ax.fill_between(x, df["ci_lo"], df["ci_hi"],
                    color="steelblue", alpha=0.18, label="95% CI (Fisher z)")
    ax.plot(x, y, "-o", color="steelblue", linewidth=1.8, markersize=7, label="Spearman r")

    full_r = float(df.loc[df["k"] == MAX_K, "spearman_r"].iloc[0])
    ax.axhline(full_r, linestyle="--", color="gray", alpha=0.7,
               label=f"Full-window r = {full_r:+.3f}")

    ax.invert_xaxis()  # most-recent window on the left
    ax.set_xticks([WINDOW_END - 10 * (k - 1) for k in range(1, MAX_K + 1)])
    ax.set_xlabel("Earliest decade in window")
    ax.set_ylabel("Spearman r (subjective polysemy × mean entropy)")
    ax.set_title("H2: Subjective polysemy × mean entropy, by historical window")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", frameon=False)

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------- Report ----------
    print("Outputs:")
    print(f"  {OUT_CSV.relative_to(BASE)}")
    print(f"  {OUT_FIG.relative_to(BASE)}  ({OUT_FIG.stat().st_size/1024:.1f} KB)")
    print()
    print("Correlation table")
    print("-" * 78)
    print(f"  {'window':<14} {'k':>3} {'n':>5} {'r':>10} {'p':>11}   {'95% CI':<18}")
    print("-" * 78)
    for _, r in df.iterrows():
        ci = f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"
        print(f"  {r['window_label']:<14} {int(r['k']):>3} {int(r['n_lemmas']):>5} "
              f"{r['spearman_r']:>+10.4f} {r['spearman_p']:>11.3g}   {ci:<18}")

    # Interpretation
    rs = df["spearman_r"].tolist()
    diffs = [rs[i + 1] - rs[i] for i in range(len(rs) - 1)]
    monotone_weaken     = all(d <= 0 for d in diffs)
    monotone_strengthen = all(d >= 0 for d in diffs)
    if monotone_weaken:
        headline = (f"Spearman r weakens monotonically as the window extends backward "
                    f"({rs[0]:+.3f} at k=1 → {rs[-1]:+.3f} at k={MAX_K}) — consistent with H2.")
    elif monotone_strengthen:
        headline = (f"Spearman r strengthens monotonically as the window extends backward "
                    f"({rs[0]:+.3f} at k=1 → {rs[-1]:+.3f} at k={MAX_K}) — opposite of H2.")
    else:
        peak_k = int(np.argmax(rs)) + 1
        direction = "weaker" if rs[-1] < rs[0] else "stronger"
        headline = (f"Non-monotonic: r peaks at k={peak_k} (r={max(rs):+.3f}) and is "
                    f"{direction} at the full window ({rs[-1]:+.3f}) than at k=1 ({rs[0]:+.3f}).")
    print()
    print(f"Headline: {headline}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
