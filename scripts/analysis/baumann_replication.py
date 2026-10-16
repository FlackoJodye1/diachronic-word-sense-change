#!/usr/bin/env python3
"""
§5 Baumann replication — bivariate correlations, multiple regression, and
Hamilton bonus on the German diachronic pilot.

Tests Baumann et al. (2023)'s predictions for sense diversification rate
(operationalised here as `entropy_slope`):
  - log_frequency           Baumann predicts NEGATIVE  (common words diversify slower)
  - discriminability_desc   Baumann predicts POSITIVE  (more discriminable → faster)
  - concreteness            Baumann predicts POSITIVE  (concrete words diversify faster)

Hamilton bonus: lemma-level Law of Innovation — does sense count predict
mean diachronic redistribution (`mean_jsd`)?

Inputs (read-only):
  data/processed/analysis/baumann_predictors.csv
  data/processed/analysis/per_lemma_summary.csv

Outputs:
  data/processed/analysis/baumann_bivariate.csv
  data/processed/analysis/baumann_regression.csv
  data/processed/analysis/hamilton_innovation.csv
  figures/baumann_replication.png
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
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr

BASE_DIR = Path(__file__).resolve().parents[2]

PREDICTORS_FILE = BASE_DIR / "data/processed/analysis/baumann_predictors.csv"
PER_LEMMA_FILE  = BASE_DIR / "data/processed/analysis/per_lemma_summary.csv"

OUT_BIVARIATE   = BASE_DIR / "data/processed/analysis/baumann_bivariate.csv"
OUT_REGRESSION  = BASE_DIR / "data/processed/analysis/baumann_regression.csv"
OUT_HAMILTON    = BASE_DIR / "data/processed/analysis/hamilton_innovation.csv"
OUT_FIG         = BASE_DIR / "figures/baumann_replication.png"

NUMERIC_COLS = ["entropy_slope", "log_frequency", "discriminability_desc", "concreteness"]

# Baumann's directional predictions
BAUMANN_PREDICTED_SIGN = {
    "log_frequency":         "-",
    "discriminability_desc": "+",
    "concreteness":          "+",
}


def _require(path: Path) -> Path:
    if not path.exists():
        sys.exit(f"ERROR: required input not found: {path.relative_to(BASE_DIR)}")
    return path


def _z_standardise(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Sample-std (ddof=1) z-standardisation per column."""
    out = df.copy()
    for c in cols:
        v = out[c].astype(float)
        out[c] = (v - v.mean()) / v.std(ddof=1)
    return out


def _bivariate(df_z: pd.DataFrame) -> pd.DataFrame:
    """Spearman + Pearson correlations of entropy_slope with each predictor.

    Note: Spearman is rank-based and Pearson r is scale-invariant, so
    standardisation does not change r or p — but we operate on the
    z-standardised frame for consistency with the regression step.
    """
    rows = []
    for pred in ["log_frequency", "discriminability_desc", "concreteness"]:
        sub = df_z[["entropy_slope", pred]].dropna()
        sr, sp = spearmanr(sub["entropy_slope"], sub[pred])
        pr, pp = pearsonr( sub["entropy_slope"], sub[pred])
        rows.append({
            "predictor":               pred,
            "spearman_r":              float(sr),
            "spearman_p":              float(sp),
            "pearson_r":               float(pr),
            "pearson_p":               float(pp),
            "n":                       int(len(sub)),
            "baumann_predicted_sign":  BAUMANN_PREDICTED_SIGN[pred],
        })
    return pd.DataFrame(rows)


def _fit_ols(df_z: pd.DataFrame, predictors: list[str], model_name: str) -> tuple[pd.DataFrame, dict]:
    """Fit OLS on z-standardised columns. Returns coefficient table + summary stats."""
    y = df_z["entropy_slope"]
    X = sm.add_constant(df_z[predictors])
    res = sm.OLS(y, X).fit()

    ci = res.conf_int()
    rows = []
    for name in X.columns:
        rows.append({
            "model":     model_name,
            "predictor": name,
            "b":         float(res.params[name]),
            "se":        float(res.bse[name]),
            "t":         float(res.tvalues[name]),
            "p":         float(res.pvalues[name]),
            "ci_low":    float(ci.loc[name, 0]),
            "ci_high":   float(ci.loc[name, 1]),
        })
    summary = {
        "n":             int(res.nobs),
        "r_squared":     float(res.rsquared),
        "adj_r_squared": float(res.rsquared_adj),
        "f_pvalue":      float(res.f_pvalue),
    }
    return pd.DataFrame(rows), summary


def _hamilton(per_lemma: pd.DataFrame) -> pd.DataFrame:
    """Spearman + Pearson between n_clusters_coarse and mean_jsd."""
    sub = per_lemma[["lemma", "n_clusters_coarse", "mean_jsd"]].dropna()
    sr, sp = spearmanr(sub["n_clusters_coarse"], sub["mean_jsd"])
    pr, pp = pearsonr( sub["n_clusters_coarse"], sub["mean_jsd"])
    return pd.DataFrame([{
        "pair_x":    "n_clusters_coarse",
        "pair_y":    "mean_jsd",
        "n":         int(len(sub)),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
        "pearson_r":  float(pr),
        "pearson_p":  float(pp),
    }])


def _plot(df: pd.DataFrame, bivariate: pd.DataFrame) -> None:
    """3-panel scatter + regression line, with top/bottom-3 entropy_slope labels."""
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    by_pred = {row["predictor"]: row for _, row in bivariate.iterrows()}

    for ax, pred in zip(axes, ["log_frequency", "discriminability_desc", "concreteness"]):
        sub = df[["lemma", pred, "entropy_slope"]].dropna().copy()
        sns.regplot(
            data=sub, x=pred, y="entropy_slope", ax=ax,
            scatter_kws={"alpha": 0.6, "s": 36, "color": "#1f77b4"},
            line_kws={"color": "#d62728", "linewidth": 1.5},
            ci=95,
        )
        # Top 3 + bottom 3 by entropy_slope (y axis)
        top3    = sub.nlargest(3, "entropy_slope")
        bottom3 = sub.nsmallest(3, "entropy_slope")
        for _, row in pd.concat([top3, bottom3]).iterrows():
            ax.annotate(
                row["lemma"],
                xy=(row[pred], row["entropy_slope"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                color="#444",
            )

        r = by_pred[pred]
        title = (f"{pred} vs. entropy_slope: "
                 f"Spearman r = {r['spearman_r']:+.3f} (p = {r['spearman_p']:.4f})")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(pred)
        ax.set_ylabel("entropy_slope")
        ax.axhline(0, color="grey", linewidth=0.7, linestyle="--", alpha=0.5)

    fig.suptitle("Baumann replication on German diachronic pilot (z-standardised predictors)",
                 fontsize=12, y=1.02)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _ascii_bivariate(df: pd.DataFrame) -> str:
    out = df.copy()
    out["spearman_r"] = out["spearman_r"].map(lambda v: f"{v:+.3f}")
    out["spearman_p"] = out["spearman_p"].map(lambda v: f"{v:.4f}")
    out["pearson_r"]  = out["pearson_r"].map( lambda v: f"{v:+.3f}")
    out["pearson_p"]  = out["pearson_p"].map( lambda v: f"{v:.4f}")
    return out[["predictor", "spearman_r", "spearman_p",
                "pearson_r", "pearson_p", "n", "baumann_predicted_sign"]].to_string(index=False)


def _ascii_regression(df: pd.DataFrame, summary: dict, model_label: str) -> str:
    rows = df.copy()
    rows["b"]      = rows["b"].map(     lambda v: f"{v:+.3f}")
    rows["se"]     = rows["se"].map(    lambda v: f"{v:.3f}")
    rows["t"]      = rows["t"].map(     lambda v: f"{v:+.2f}")
    rows["p"]      = rows["p"].map(     lambda v: f"{v:.4f}")
    rows["ci_low"] = rows["ci_low"].map(lambda v: f"{v:+.3f}")
    rows["ci_high"]= rows["ci_high"].map(lambda v: f"{v:+.3f}")
    body = rows[["predictor", "b", "se", "t", "p", "ci_low", "ci_high"]].to_string(index=False)
    header = (f"{model_label}  (n={summary['n']}, "
              f"R²={summary['r_squared']:.3f}, "
              f"adj-R²={summary['adj_r_squared']:.3f}, "
              f"F p={summary['f_pvalue']:.4f})")
    return header + "\n" + body


def main() -> int:
    pred = pd.read_csv(_require(PREDICTORS_FILE))
    per_lemma = pd.read_csv(_require(PER_LEMMA_FILE))

    # ------------------------------------------------------------------
    # 1. Restrict to complete cases + z-standardise
    # ------------------------------------------------------------------
    complete = pred.dropna(subset=NUMERIC_COLS).copy().reset_index(drop=True)
    n = len(complete)
    df_z = _z_standardise(complete, NUMERIC_COLS)

    # ------------------------------------------------------------------
    # 2. Bivariate correlations
    # ------------------------------------------------------------------
    bivariate = _bivariate(df_z)
    OUT_BIVARIATE.parent.mkdir(parents=True, exist_ok=True)
    bivariate.to_csv(OUT_BIVARIATE, index=False)

    # ------------------------------------------------------------------
    # 3. Multiple regressions (z-standardised → b are standardised coefficients)
    # ------------------------------------------------------------------
    a_df, a_summary = _fit_ols(df_z, ["log_frequency", "discriminability_desc"], "A_disc")
    b_df, b_summary = _fit_ols(df_z, ["log_frequency", "concreteness"],          "B_concr")

    reg_rows = []
    for d, s in [(a_df, a_summary), (b_df, b_summary)]:
        for _, row in d.iterrows():
            row_out = dict(row)
            row_out["n"]             = s["n"]
            row_out["r_squared"]     = s["r_squared"]
            row_out["adj_r_squared"] = s["adj_r_squared"]
            reg_rows.append(row_out)
    regression = pd.DataFrame(reg_rows)[
        ["model", "predictor", "b", "se", "t", "p", "ci_low", "ci_high",
         "n", "r_squared", "adj_r_squared"]
    ]
    regression.to_csv(OUT_REGRESSION, index=False)

    # ------------------------------------------------------------------
    # 4. Hamilton bonus
    # ------------------------------------------------------------------
    hamilton = _hamilton(per_lemma)
    hamilton.to_csv(OUT_HAMILTON, index=False)

    # ------------------------------------------------------------------
    # 5. Figure (on z-standardised data — Spearman and Pearson invariant)
    # ------------------------------------------------------------------
    # For interpretability of x-axes we plot on z-standardised data
    # (predictor scales differ by orders of magnitude otherwise).
    _plot(df_z.assign(lemma=complete["lemma"]), bivariate)

    # ------------------------------------------------------------------
    # 6. Report
    # ------------------------------------------------------------------
    print(f"\nBaumann replication — German diachronic pilot")
    print(f"=============================================")
    print(f"Lemmas used (complete cases across the four predictors): n = {n}")
    print()
    print("Bivariate correlations (z-standardised, entropy_slope vs. predictor):")
    print(_ascii_bivariate(bivariate))
    print()
    print(_ascii_regression(a_df, a_summary, "Model A:  entropy_slope ~ log_frequency + discriminability_desc"))
    print()
    print(_ascii_regression(b_df, b_summary, "Model B:  entropy_slope ~ log_frequency + concreteness"))
    print()
    print("Hamilton bonus (n_clusters_coarse ↔ mean_jsd):")
    h = hamilton.iloc[0]
    print(f"  Spearman r = {h['spearman_r']:+.3f} (p = {h['spearman_p']:.4f}), n = {int(h['n'])}")
    print(f"  Pearson  r = {h['pearson_r']:+.3f} (p = {h['pearson_p']:.4f})")
    print()

    # ------------------------------------------------------------------
    # 7. Headline: which Baumann predictions replicate at p<0.05?
    # ------------------------------------------------------------------
    replicated = []
    failed = []
    for _, row in bivariate.iterrows():
        sign = "+" if row["spearman_r"] > 0 else "-"
        # "Replicates" = correct sign AND p < 0.05
        if sign == row["baumann_predicted_sign"] and row["spearman_p"] < 0.05:
            replicated.append(row["predictor"])
        else:
            failed.append(row["predictor"])
    print(f"HEADLINE: Baumann predictions replicating at p<0.05 ({len(replicated)}/3): "
          f"{replicated if replicated else 'NONE'}. Not replicating: {failed}.")
    print()
    print(f"Outputs:")
    print(f"  Bivariate:  {OUT_BIVARIATE.relative_to(BASE_DIR)}")
    print(f"  Regression: {OUT_REGRESSION.relative_to(BASE_DIR)}")
    print(f"  Hamilton:   {OUT_HAMILTON.relative_to(BASE_DIR)}")
    print(f"  Figure:     {OUT_FIG.relative_to(BASE_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
