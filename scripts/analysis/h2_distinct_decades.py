#!/usr/bin/env python3
"""
§4 H2 — Distinct-decades variant.

Like h2_staged_windows.py, but instead of cumulative windows ending in 1990
we use each decade's entropy IN ISOLATION. For each decade d ∈ {1910, ..., 1990}
we compute Spearman(subjective_polysemy_mean, entropy_d) across the lemmas
with n_contexts >= 10 in that decade.

Inputs (read-only):
  data/processed/diachronic/metrics/pilot_v3_coarse_entropy.csv
  data/processed/diachronic/aggregated/pilot_v3_coarse_stats.csv
  data/processed/survey/survey_polysemy_per_lemma.csv

Outputs:
  data/processed/analysis/h2_distinct_decades.csv
  figures/h2_distinct_decades.pdf
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

OUT_CSV = BASE / "data/processed/analysis/h2_distinct_decades.csv"
OUT_FIG = BASE / "figures/h2_distinct_decades.pdf"

MIN_CONTEXTS = 10
DECADES = list(range(1910, 2000, 10))  # 1910 .. 1990


def fisher_ci(r: float, n: int) -> tuple[float, float]:
    if n < 4 or not -1.0 < r < 1.0:
        return float("nan"), float("nan")
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    crit = 1.959963984540054
    return float(np.tanh(z - crit * se)), float(np.tanh(z + crit * se))


def main() -> int:
    ent    = pd.read_csv(ENTROPY_FILE)[["lemma", "decade", "entropy"]]
    stats  = pd.read_csv(STATS_FILE)[["lemma", "decade", "n_contexts"]]
    survey = pd.read_csv(SURVEY_FILE)[["lemma", "subjective_polysemy_mean"]]

    supported = {(str(l), int(d))
                 for l, d in stats.loc[stats["n_contexts"] >= MIN_CONTEXTS,
                                       ["lemma", "decade"]].values}

    rows = []
    for d in DECADES:
        ent_d = ent[ent["decade"] == d][["lemma", "entropy"]]
        ent_d = ent_d[[(l, d) in supported for l in ent_d["lemma"]]]
        joined = ent_d.merge(survey, on="lemma", how="inner").dropna()
        if len(joined) < 4:
            rows.append({"decade": d, "n_lemmas": int(len(joined)),
                         "spearman_r": float("nan"), "spearman_p": float("nan"),
                         "ci_lo": float("nan"), "ci_hi": float("nan")})
            continue
        sr, sp = spearmanr(joined["subjective_polysemy_mean"], joined["entropy"])
        lo, hi = fisher_ci(float(sr), len(joined))
        rows.append({
            "decade":     d,
            "n_lemmas":   int(len(joined)),
            "spearman_r": float(sr),
            "spearman_p": float(sp),
            "ci_lo":      lo,
            "ci_hi":      hi,
        })

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df[["decade", "n_lemmas", "spearman_r", "spearman_p"]].to_csv(OUT_CSV, index=False)

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
    x = df["decade"].values
    y = df["spearman_r"].values

    ax.fill_between(x, df["ci_lo"], df["ci_hi"],
                    color="darkorange", alpha=0.18, label="95% CI (Fisher z)")
    ax.plot(x, y, "-o", color="darkorange", linewidth=1.8, markersize=7,
            label="Spearman r (per-decade entropy)")

    # Reference line: full-window r from h2_staged_windows.csv (1910–1990), if present
    staged_csv = BASE / "data/processed/analysis/h2_staged_correlations.csv"
    if staged_csv.exists():
        staged = pd.read_csv(staged_csv)
        ref_r = float(staged.loc[staged["k"] == staged["k"].max(), "spearman_r"].iloc[0])
        ax.axhline(ref_r, linestyle="--", color="gray", alpha=0.7,
                   label=f"Full-window r = {ref_r:+.3f}")

    ax.set_xticks(DECADES)
    ax.set_xlabel("Decade")
    ax.set_ylabel("Spearman r (subjective polysemy × decade entropy)")
    ax.set_title("H2: Subjective polysemy × per-decade entropy")
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
    print("Per-decade correlation table")
    print("-" * 72)
    print(f"  {'decade':<7} {'n':>5} {'r':>10} {'p':>11}   {'95% CI':<18}")
    print("-" * 72)
    for _, r in df.iterrows():
        ci = f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]" if pd.notna(r['ci_lo']) else 'n/a'
        rstr = f"{r['spearman_r']:+10.4f}" if pd.notna(r['spearman_r']) else "    n/a   "
        pstr = f"{r['spearman_p']:>11.3g}" if pd.notna(r['spearman_p']) else "        n/a"
        print(f"  {int(r['decade']):<7} {int(r['n_lemmas']):>5} {rstr} {pstr}   {ci:<18}")

    # Headline
    rs = df["spearman_r"].dropna().tolist()
    decs = df.loc[df["spearman_r"].notna(), "decade"].tolist()
    if rs:
        peak_idx = int(np.argmax(rs))
        rng = max(rs) - min(rs)
        print()
        print(f"Headline: per-decade r ranges from {min(rs):+.3f} to {max(rs):+.3f} "
              f"(Δ={rng:.3f}); peak in {decs[peak_idx]}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())