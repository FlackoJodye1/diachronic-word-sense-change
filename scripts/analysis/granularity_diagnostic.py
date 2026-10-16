#!/usr/bin/env python3
"""
Fine vs. coarse granularity diagnostic for the German Survey × Model correlation.

Question:
  Is the moderate German correlation (r ≈ 0.345 for mean_entropy × subjective
  polysemy, n = 73) partly caused by the coarse-clustering layer collapsing
  sense distinctions that survey participants implicitly count?

Approach:
  Recompute the same correlations on fine-grained GermaNet sense IDs (one row
  per `predicted_sense_id` from `pilot_predictions_v3.csv`, pre-coarse-mapping)
  and compare side-by-side with the already-computed coarse metrics.

Why we compute fine metrics here rather than load them from disk:
  No canonical v3 fine-grained metrics CSV exists. The Feb-19 `pilot_*.csv`
  files are v2 (smaller, older pipeline). Mixing v2 fine with v3 coarse would
  confound the diagnostic with a version change. So fine metrics are derived
  on the fly from `pilot_predictions_v3.csv` (the raw v3 predictions) using
  the same support filter (`n_contexts >= 10`) and aggregation logic as the
  coarse `per_lemma_summary.csv`.

Inputs:
  data/processed/diachronic/predictions/pilot_predictions_v3.csv
  data/processed/diachronic/aggregated/pilot_v3_coarse_stats.csv  (n_contexts)
  data/processed/germanet/senses_pilot.jsonl                       (fine sense count)
  data/processed/analysis/per_lemma_summary.csv                    (coarse metrics)
  data/processed/survey/survey_polysemy_per_lemma.csv              (survey)

Output:
  data/processed/analysis/granularity_diagnostic.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, spearmanr

BASE_DIR = Path(__file__).resolve().parents[2]

PREDICTIONS_FILE = BASE_DIR / "data/processed/diachronic/predictions/combined_predictions_v3.csv"
STATS_FILE       = BASE_DIR / "data/processed/diachronic/aggregated/pilot_v3_coarse_stats.csv"
SENSES_FILE      = BASE_DIR / "data/processed/germanet/senses_combined.jsonl"
COARSE_FILE      = BASE_DIR / "data/processed/analysis/per_lemma_summary.csv"
SURVEY_FILE      = BASE_DIR / "data/processed/survey/survey_polysemy_per_lemma.csv"

OUTPUT_FILE      = BASE_DIR / "data/processed/analysis/granularity_diagnostic.csv"

MIN_CONTEXTS = 10  # support filter, same as per_lemma_summary.py
X_VAR = "subjective_polysemy_mean"


def _require(path: Path) -> Path:
    if not path.exists():
        sys.exit(f"ERROR: required input not found: {path.relative_to(BASE_DIR)}")
    return path


def _fine_sense_counts(senses_path: Path) -> pd.Series:
    """Count fine GermaNet senses per lemma from senses_pilot.jsonl."""
    counts: dict[str, int] = {}
    with senses_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            counts[row["lemma"]] = counts.get(row["lemma"], 0) + 1
    return pd.Series(counts, name="n_senses_fine")


def _fine_metrics_from_predictions(predictions_path: Path,
                                   supported: set[tuple[str, int]]) -> pd.DataFrame:
    """Compute fine-grained mean_entropy_fine + mean_jsd_fine per lemma.

    Same support filter as per_lemma_summary.py: a (lemma, decade) cell is
    included only if n_contexts >= MIN_CONTEXTS. A JSD transition is included
    only if both endpoint decades pass the filter.
    """
    preds = pd.read_csv(predictions_path, usecols=["lemma", "decade", "predicted_sense_id"])

    # Per (lemma, decade, sense) → proportion
    counts = (preds.groupby(["lemma", "decade", "predicted_sense_id"])
                   .size().reset_index(name="count"))
    totals = counts.groupby(["lemma", "decade"])["count"].transform("sum")
    counts["proportion"] = counts["count"] / totals

    # Pivot to wide per lemma so each decade is a column and each sense is a row.
    rows: list[dict] = []
    for lemma, lemma_df in counts.groupby("lemma"):
        # Wide: rows = sense_id, cols = decade, vals = proportion (zero-fill missing senses).
        wide = lemma_df.pivot(index="predicted_sense_id", columns="decade",
                              values="proportion").fillna(0.0)
        # Restrict to supported decades.
        supported_decades_for_lemma = [d for d in wide.columns if (lemma, int(d)) in supported]
        if not supported_decades_for_lemma:
            rows.append({"lemma": lemma,
                         "mean_entropy_fine": np.nan,
                         "mean_jsd_fine": np.nan})
            continue
        wide_sup = wide[supported_decades_for_lemma]

        # Entropy per supported decade column.
        ents = entropy(wide_sup.values, base=2, axis=0)
        mean_h = float(np.mean(ents)) if len(ents) else np.nan

        # JSD across consecutive supported decades (sorted ascending).
        sorted_cols = sorted(supported_decades_for_lemma)
        jsds = []
        for c0, c1 in zip(sorted_cols[:-1], sorted_cols[1:]):
            jsds.append(jensenshannon(wide_sup[c0].values, wide_sup[c1].values, base=2))
        mean_jsd = float(np.mean(jsds)) if jsds else np.nan

        rows.append({"lemma": lemma,
                     "mean_entropy_fine": mean_h,
                     "mean_jsd_fine": mean_jsd})

    return pd.DataFrame(rows)


def _corr(joined: pd.DataFrame, y: str) -> tuple[float, float, int]:
    sub = joined[[X_VAR, y]].dropna()
    if len(sub) < 3:
        return float("nan"), float("nan"), len(sub)
    sr, sp = spearmanr(sub[X_VAR], sub[y])
    return float(sr), float(sp), len(sub)


def main() -> int:
    # ------------------------------------------------------------------
    # Load existing artifacts
    # ------------------------------------------------------------------
    coarse = pd.read_csv(_require(COARSE_FILE))   # 90 lemmas
    survey = pd.read_csv(_require(SURVEY_FILE))   # 544 lemmas
    stats  = pd.read_csv(_require(STATS_FILE))[["lemma", "decade", "n_contexts"]]
    supported: set[tuple[str, int]] = {
        (l, int(d)) for l, d in stats[stats["n_contexts"] >= MIN_CONTEXTS][["lemma", "decade"]].values
    }

    # ------------------------------------------------------------------
    # Compute fine-grained metrics on the fly
    # ------------------------------------------------------------------
    fine = _fine_metrics_from_predictions(_require(PREDICTIONS_FILE), supported)
    fine_counts = _fine_sense_counts(_require(SENSES_FILE))
    fine = fine.merge(fine_counts.rename_axis("lemma").reset_index(), on="lemma", how="left")

    # ------------------------------------------------------------------
    # Join coarse + fine + survey
    # ------------------------------------------------------------------
    combined = (coarse
                .merge(fine, on="lemma", how="inner")
                .merge(survey, on="lemma", how="inner"))
    n = len(combined)

    # ------------------------------------------------------------------
    # Correlations: coarse vs. fine
    # ------------------------------------------------------------------
    pairs = [
        ("n_senses",     "n_clusters_coarse", "n_senses_fine"),
        ("mean_entropy", "mean_entropy",      "mean_entropy_fine"),
        ("mean_jsd",     "mean_jsd",          "mean_jsd_fine"),
    ]

    rows = []
    for label, coarse_col, fine_col in pairs:
        c_r, c_p, c_n = _corr(combined, coarse_col)
        f_r, f_p, f_n = _corr(combined, fine_col)
        rows.append({"granularity": "coarse", "pair_y": coarse_col, "n": c_n,
                     "spearman_r": c_r, "spearman_p": c_p})
        rows.append({"granularity": "fine",   "pair_y": fine_col,   "n": f_n,
                     "spearman_r": f_r, "spearman_p": f_p})

    out = pd.DataFrame(rows)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)

    # ------------------------------------------------------------------
    # Side-by-side report
    # ------------------------------------------------------------------
    print(f"\nFine vs. coarse granularity diagnostic")
    print(f"=======================================")
    print(f"Joined dataset: n = {n} lemmas")
    print()
    print(f"{'metric':<20} {'coarse r':>10} {'coarse p':>10}  {'fine r':>10} {'fine p':>10} {'Δ (fine−coarse)':>20}")
    print("-" * 92)
    deltas: dict[str, float] = {}
    for label, coarse_col, fine_col in pairs:
        c_r, c_p, _ = _corr(combined, coarse_col)
        f_r, f_p, _ = _corr(combined, fine_col)
        delta = f_r - c_r
        deltas[label] = delta
        print(f"{label:<20} {c_r:>+10.3f} {c_p:>10.4f}  {f_r:>+10.3f} {f_p:>10.4f} {delta:>+20.3f}")

    # ------------------------------------------------------------------
    # Interpretation flag — based on the entropy pair
    # ------------------------------------------------------------------
    d_entropy = deltas["mean_entropy"]
    print()
    if d_entropy > 0.05:
        flag = ("Fine-grained correlations are notably stronger — suggests coarse clustering "
                "is reducing alignment with survey ratings.")
    elif abs(d_entropy) <= 0.05:
        flag = ("Coarse and fine correlations are similar — clustering is not driving the "
                "magnitude difference vs. English.")
    else:
        flag = ("Coarse correlations are stronger — clustering improves alignment, possibly "
                "by reducing noise from near-synonymous senses.")
    print(f"Interpretation: {flag}")
    print()
    print(f"Output: {OUTPUT_FILE.relative_to(BASE_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())