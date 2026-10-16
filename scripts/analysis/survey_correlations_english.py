#!/usr/bin/env python3
"""
English survey × Hu et al. (2019) correlation analysis.

Cross-linguistic counterpart to `survey_correlations.py` (German). Joins
Amandine Grieshaber's English subjective polysemy ratings with Hu et al.'s
per-word entropy / JSD metrics and tests whether the same C2 (human ↔ model)
alignment we found in German also holds in English.

Inputs:
  data/processed/survey/survey_english_per_lemma.csv     (558 lemmas)
  data/processed/analysis/hu_etal_per_word.csv           (3,220 lemmas)

Outputs:
  data/processed/analysis/survey_correlations_english.csv  — long-format table
  figures/survey_correlations_english.png                  — 3-panel figure

For each (x = subjective_polysemy_mean) × y in {n_senses, mean_entropy,
mean_jsd}, computes Spearman + Pearson correlations with two-tailed p-values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

BASE_DIR = Path(__file__).resolve().parents[2]

SURVEY_FILE = BASE_DIR / "data/processed/survey/survey_english_per_lemma.csv"
MODEL_FILE  = BASE_DIR / "data/processed/analysis/hu_etal_per_word.csv"

OUT_CSV     = BASE_DIR / "data/processed/analysis/survey_correlations_english.csv"
OUT_FIG     = BASE_DIR / "figures/survey_correlations_english.png"

X_VAR = "subjective_polysemy_mean"
Y_VARS = [
    "n_senses",        # sanity: do speakers perceive Hu et al.'s sense count?
    "mean_entropy",    # primary: cross-linguistic counterpart to German r = +0.345
    "mean_jsd",        # diachronic redistribution
]


def _require(path: Path) -> Path:
    if not path.exists():
        sys.exit(f"ERROR: required input not found: {path.relative_to(BASE_DIR)}")
    return path


def _corr_row(df: pd.DataFrame, y: str) -> dict:
    sub = df[[X_VAR, y]].dropna()
    n = len(sub)
    if n < 3:
        return {
            "pair_x": X_VAR, "pair_y": y, "n": n,
            "spearman_r": np.nan, "spearman_p": np.nan,
            "pearson_r":  np.nan, "pearson_p":  np.nan,
        }
    sr, sp = spearmanr(sub[X_VAR], sub[y])
    pr, pp = pearsonr(sub[X_VAR], sub[y])
    return {
        "pair_x": X_VAR, "pair_y": y, "n": n,
        "spearman_r": float(sr), "spearman_p": float(sp),
        "pearson_r":  float(pr), "pearson_p":  float(pp),
    }


def _ascii_table(df: pd.DataFrame) -> str:
    cols = ["pair_y", "n", "spearman_r", "spearman_p", "pearson_r", "pearson_p"]
    out = df[cols].copy()
    out["spearman_r"] = out["spearman_r"].map(lambda v: f"{v:+.3f}" if pd.notna(v) else "  nan")
    out["spearman_p"] = out["spearman_p"].map(lambda v: f"{v:.4f}"  if pd.notna(v) else " nan")
    out["pearson_r"]  = out["pearson_r"].map( lambda v: f"{v:+.3f}" if pd.notna(v) else "  nan")
    out["pearson_p"]  = out["pearson_p"].map( lambda v: f"{v:.4f}"  if pd.notna(v) else " nan")
    return out.to_string(index=False)


def _plot(joined: pd.DataFrame, results: list[dict]) -> None:
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    by_y = {r["pair_y"]: r for r in results}

    for ax, y in zip(axes, Y_VARS):
        sub = joined[["lemma", X_VAR, y]].dropna().copy()
        sns.regplot(
            data=sub, x=X_VAR, y=y, ax=ax,
            scatter_kws={"alpha": 0.5, "s": 28, "color": "#1f77b4"},
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
        "English survey-derived polysemy vs. Hu et al. (2019) sense statistics",
        fontsize=13, y=1.02,
    )
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- German reference numbers for side-by-side reporting --------------
GERMAN_REF = [
    ("n_clusters_coarse / n_senses", "+0.338", "0.0035", "73"),
    ("mean_entropy",                  "+0.345", "0.0028", "73"),
    ("mean_jsd",                      "+0.254", "0.0299", "73"),
]


def main() -> int:
    survey = pd.read_csv(_require(SURVEY_FILE))
    model  = pd.read_csv(_require(MODEL_FILE))

    survey_lemmas = set(survey["lemma"])
    model_lemmas  = set(model["lemma"])

    joined = model.merge(survey, on="lemma", how="inner")
    dropped_from_model  = sorted(model_lemmas  - survey_lemmas)
    dropped_from_survey = sorted(survey_lemmas - model_lemmas)
    n_joined = len(joined)

    n_nan_jsd = int(joined["mean_jsd"].isna().sum())

    results = [_corr_row(joined, y) for y in Y_VARS]

    results_df = pd.DataFrame(results)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUT_CSV, index=False)

    _plot(joined, results)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print(f"\nEnglish survey × Hu et al. correlation analysis")
    print(f"================================================")
    print(f"Inner join: n = {n_joined} lemmas overlap")
    print(f"  Dropped from model side (Hu et al., {len(dropped_from_model)} of {len(model_lemmas)}): too many to list; first 10:")
    print(f"    {dropped_from_model[:10]}")
    print(f"  Dropped from survey side ({len(dropped_from_survey)} of {len(survey_lemmas)}): first 10:")
    print(f"    {dropped_from_survey[:10]}")
    print(f"  NaN mean_jsd in joined set: {n_nan_jsd}")
    print()
    print(_ascii_table(results_df))
    print()

    # Side-by-side German vs. English
    print(f"=== German vs. English (Spearman) ===")
    print(f"{'pair':<35} {'German r':>9} {'p':>8}  n  | {'English r':>10} {'p':>8}  n")
    print("-" * 90)
    eng_pairs = {
        "n_clusters_coarse / n_senses": next((r for r in results if r["pair_y"] == "n_senses"), {}),
        "mean_entropy":                  next((r for r in results if r["pair_y"] == "mean_entropy"), {}),
        "mean_jsd":                      next((r for r in results if r["pair_y"] == "mean_jsd"), {}),
    }
    for label, ger_r, ger_p, ger_n in GERMAN_REF:
        eng = eng_pairs.get(label, {})
        eng_r = f"{eng.get('spearman_r', float('nan')):+.3f}" if eng else "    nan"
        eng_p = f"{eng.get('spearman_p', float('nan')):.4f}"  if eng else "    nan"
        eng_n = str(eng.get("n", 0))                          if eng else "0"
        print(f"{label:<35} {ger_r:>9} {ger_p:>8}  {ger_n}  | {eng_r:>10} {eng_p:>8}  {eng_n}")

    print()
    print(f"Outputs:")
    print(f"  CSV:    {OUT_CSV.relative_to(BASE_DIR)}")
    print(f"  Figure: {OUT_FIG.relative_to(BASE_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())