#!/usr/bin/env python3
"""
Survey × model correlation analysis (EMNLP §4 Survey Evaluation).

Tests whether model-inferred sense-distribution statistics track human-perceived
polysemy from the Grieshaber & Rubinskii survey.

Inputs:
  data/processed/survey/survey_polysemy_per_lemma.csv  (544 lemmas)
  data/processed/analysis/per_lemma_summary.csv        (90 lemmas)

Outputs:
  data/processed/analysis/survey_correlations.csv      — long-format table
  figures/survey_correlations.png                      — 4-panel figure

For each (x = subjective_polysemy_mean) × y in {n_clusters_coarse, mean_entropy,
mean_jsd, mean_jsd_normalized}, computes Spearman + Pearson correlations
with two-tailed p-values. Reports both an all-lemmas and an
n_decades_with_data >= 5 robustness variant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display backend; we only save
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

BASE_DIR = Path(__file__).resolve().parents[2]

SURVEY_FILE = BASE_DIR / "data/processed/survey/survey_polysemy_per_lemma.csv"
MODEL_FILE  = BASE_DIR / "data/processed/analysis/per_lemma_summary.csv"

OUT_CSV     = BASE_DIR / "data/processed/analysis/survey_correlations.csv"
OUT_FIG     = BASE_DIR / "figures/survey_correlations.png"

X_VAR = "subjective_polysemy_mean"
Y_VARS = [
    "n_clusters_coarse",
    "mean_entropy",
    "mean_jsd",
    "mean_jsd_normalized",
]

MIN_DECADES = 5  # robustness filter


def _require(path: Path) -> Path:
    if not path.exists():
        sys.exit(f"ERROR: required input not found: {path.relative_to(BASE_DIR)}")
    return path


def _corr_row(df: pd.DataFrame, y: str, filter_label: str) -> dict:
    """Compute one row of the correlation table, after dropping NaN in (x, y)."""
    sub = df[[X_VAR, y]].dropna()
    n = len(sub)
    if n < 3:
        return {
            "pair_x": X_VAR, "pair_y": y, "n": n,
            "spearman_r": np.nan, "spearman_p": np.nan,
            "pearson_r": np.nan,  "pearson_p": np.nan,
            "filter_applied": filter_label,
        }
    sr, sp = spearmanr(sub[X_VAR], sub[y])
    pr, pp = pearsonr(sub[X_VAR], sub[y])
    return {
        "pair_x": X_VAR, "pair_y": y, "n": n,
        "spearman_r": float(sr), "spearman_p": float(sp),
        "pearson_r":  float(pr), "pearson_p":  float(pp),
        "filter_applied": filter_label,
    }


def _ascii_table(df: pd.DataFrame) -> str:
    """Compact fixed-width rendering of the correlation table."""
    cols = ["pair_y", "n", "spearman_r", "spearman_p", "pearson_r", "pearson_p", "filter_applied"]
    out = df[cols].copy()
    out["spearman_r"] = out["spearman_r"].map(lambda v: f"{v:+.3f}" if pd.notna(v) else "  nan")
    out["spearman_p"] = out["spearman_p"].map(lambda v: f"{v:.4f}"  if pd.notna(v) else " nan")
    out["pearson_r"]  = out["pearson_r"].map( lambda v: f"{v:+.3f}" if pd.notna(v) else "  nan")
    out["pearson_p"]  = out["pearson_p"].map( lambda v: f"{v:.4f}"  if pd.notna(v) else " nan")
    return out.to_string(index=False)


def _plot(joined: pd.DataFrame, results_all: list[dict]) -> None:
    """4-panel scatter + regression with top/bottom-3 lemma labels (x-axis)."""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    by_y = {r["pair_y"]: r for r in results_all if r["filter_applied"] == "none"}

    for ax, y in zip(axes, Y_VARS):
        sub = joined[["lemma", X_VAR, y]].dropna().copy()
        sns.regplot(
            data=sub, x=X_VAR, y=y, ax=ax,
            scatter_kws={"alpha": 0.6, "s": 40, "color": "#1f77b4"},
            line_kws={"color": "#d62728", "linewidth": 1.5},
            ci=95,
        )

        # Top 3 + bottom 3 by x-axis value (subjective_polysemy_mean)
        top3    = sub.nlargest(3, X_VAR)
        bottom3 = sub.nsmallest(3, X_VAR)
        for _, row in pd.concat([top3, bottom3]).iterrows():
            ax.annotate(
                row["lemma"],
                xy=(row[X_VAR], row[y]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                color="#444",
            )

        r = by_y.get(y, {})
        title = (f"{y}: Spearman r = {r.get('spearman_r', float('nan')):.3f} "
                 f"(p = {r.get('spearman_p', float('nan')):.4f}), n = {r.get('n', 0)}")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("subjective_polysemy_mean")
        ax.set_ylabel(y)

    fig.suptitle(
        "Survey-derived polysemy vs. model-derived sense statistics",
        fontsize=13, y=1.00,
    )
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    survey = pd.read_csv(_require(SURVEY_FILE))
    model  = pd.read_csv(_require(MODEL_FILE))

    # ------------------------------------------------------------------
    # Inner join + diagnostics
    # ------------------------------------------------------------------
    survey_lemmas = set(survey["lemma"])
    model_lemmas  = set(model["lemma"])

    joined = model.merge(survey, on="lemma", how="inner")
    dropped_from_model  = sorted(model_lemmas  - survey_lemmas)
    dropped_from_survey = sorted(survey_lemmas - model_lemmas)

    n_joined = len(joined)

    # ------------------------------------------------------------------
    # Correlations: all lemmas, and n_decades_with_data >= MIN_DECADES
    # ------------------------------------------------------------------
    results: list[dict] = []
    for y in Y_VARS:
        results.append(_corr_row(joined, y, filter_label="none"))

    filt = joined[joined["n_decades_with_data"] >= MIN_DECADES]
    for y in Y_VARS:
        results.append(_corr_row(filt, y, filter_label=f"n_decades_with_data>={MIN_DECADES}"))

    results_df = pd.DataFrame(results)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUT_CSV, index=False)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    _plot(joined, results)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    n_nan_jsd = int(joined["mean_jsd"].isna().sum())

    print(f"\nSurvey × model correlation analysis")
    print(f"===================================")
    print(f"Inner join: n = {n_joined} lemmas overlap")
    print(f"  Dropped from model side ({len(dropped_from_model)}):  {dropped_from_model}")
    print(f"  Dropped from survey side ({len(dropped_from_survey)}): "
          f"{dropped_from_survey[:10]}{' ...' if len(dropped_from_survey) > 10 else ''}")
    print(f"  NaN mean_jsd in joined set: {n_nan_jsd}")
    print(f"  Lemmas with n_decades_with_data >= {MIN_DECADES}: {len(filt)} / {n_joined}")
    print()
    print(_ascii_table(results_df))
    print()
    print(f"Outputs:")
    print(f"  CSV:    {OUT_CSV.relative_to(BASE_DIR)}")
    print(f"  Figure: {OUT_FIG.relative_to(BASE_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())