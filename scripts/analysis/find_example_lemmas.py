#!/usr/bin/env python3
"""
Find candidate lemmas for §6.1 Illustrative Trajectories.

Produces three sorted candidate lists across three modes of sense change:
  A. Directional shift   — one cluster rises while another falls clearly
  B. Diversification     — entropy grows over time, distribution becomes more even
  C. Stable polysemy     — high polysemy but stable distribution across decades

Inputs (read-only):
  data/processed/diachronic/aggregated/pilot_v3_coarse_sense_counts.csv
  data/processed/diachronic/aggregated/pilot_v3_coarse_stats.csv
  data/processed/diachronic/metrics/pilot_v3_coarse_entropy.csv
  data/processed/analysis/per_lemma_summary.csv

Support filter (matches per_lemma_summary.py):
  - A (lemma, decade) cell is "supported" if n_contexts >= 10.
  - A lemma is eligible if n_decades_with_data >= 8.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

BASE = Path(__file__).resolve().parents[2]

COUNTS_FILE  = BASE / "data/processed/diachronic/aggregated/pilot_v3_coarse_sense_counts.csv"
STATS_FILE   = BASE / "data/processed/diachronic/aggregated/pilot_v3_coarse_stats.csv"
ENTROPY_FILE = BASE / "data/processed/diachronic/metrics/pilot_v3_coarse_entropy.csv"
SUMMARY_FILE = BASE / "data/processed/analysis/per_lemma_summary.csv"

MIN_CONTEXTS = 10
MIN_DECADES  = 8
TOP_N        = 15


def _supported_cells(stats: pd.DataFrame) -> set[tuple[str, int]]:
    sup = stats[stats["n_contexts"] >= MIN_CONTEXTS][["lemma", "decade"]]
    return {(str(l), int(d)) for l, d in sup.values}


def list_a_directional(counts: pd.DataFrame,
                       supported: set[tuple[str, int]],
                       eligible: set[str],
                       summary: pd.DataFrame) -> pd.DataFrame:
    """Max |Δproportion| across clusters between first and last supported decade."""
    rows = []
    counts_eligible = counts[counts["lemma"].isin(eligible)].copy()
    counts_eligible["__sup"] = [
        (l, int(d)) in supported for l, d in zip(counts_eligible["lemma"], counts_eligible["decade"])
    ]
    counts_eligible = counts_eligible[counts_eligible["__sup"]].drop(columns=["__sup"])

    for lemma, sub in counts_eligible.groupby("lemma"):
        decs = sorted(sub["decade"].unique())
        d_first, d_last = decs[0], decs[-1]
        # Wide: rows = cluster, cols = decade, vals = proportion (zero-fill missing)
        wide = (sub.pivot_table(index="predicted_sense_id",
                                columns="decade",
                                values="proportion",
                                aggfunc="sum")
                  .fillna(0.0))
        if d_first not in wide.columns or d_last not in wide.columns:
            continue
        delta = wide[d_last] - wide[d_first]
        max_abs = delta.abs().max()
        rising = delta.idxmax()
        falling = delta.idxmin()
        rows.append({
            "lemma": lemma,
            "n_clusters": int(wide.shape[0]),
            "first_decade": int(d_first),
            "last_decade":  int(d_last),
            "max_abs_change": float(max_abs),
            "cluster_rising":  rising,
            "Δ_rising":       float(delta[rising]),
            "cluster_falling": falling,
            "Δ_falling":      float(delta[falling]),
        })
    out = pd.DataFrame(rows)
    out = out.merge(summary[["lemma", "mean_jsd"]], on="lemma", how="left")
    return out.sort_values("max_abs_change", ascending=False).head(TOP_N)


def list_b_diversification(entropy: pd.DataFrame,
                           supported: set[tuple[str, int]],
                           eligible: set[str],
                           summary: pd.DataFrame) -> pd.DataFrame:
    """Slope of entropy ~ decade. Top positive = strongest diversification."""
    n_clu = summary.set_index("lemma")["n_clusters_coarse"].to_dict()
    eligible_b = {l for l in eligible if n_clu.get(l, 0) >= 3}

    ent = entropy[entropy["lemma"].isin(eligible_b)].copy()
    ent["__sup"] = [
        (l, int(d)) in supported for l, d in zip(ent["lemma"], ent["decade"])
    ]
    ent = ent[ent["__sup"]].drop(columns=["__sup"])

    rows = []
    for lemma, sub in ent.groupby("lemma"):
        if len(sub) < 3:
            continue
        sub = sub.sort_values("decade")
        res = linregress(sub["decade"].astype(float), sub["entropy"].astype(float))
        rows.append({
            "lemma": lemma,
            "n_clusters": int(n_clu[lemma]),
            "n_decades_used": int(len(sub)),
            "entropy_slope": float(res.slope),
            "entropy_slope_p": float(res.pvalue),
            "entropy_start": float(sub["entropy"].iloc[0]),
            "entropy_end":   float(sub["entropy"].iloc[-1]),
            "decade_start":  int(sub["decade"].iloc[0]),
            "decade_end":    int(sub["decade"].iloc[-1]),
        })
    out = pd.DataFrame(rows)
    return out.sort_values("entropy_slope", ascending=False).head(TOP_N)


def list_c_stable(entropy: pd.DataFrame,
                  supported: set[tuple[str, int]],
                  eligible: set[str],
                  summary: pd.DataFrame) -> pd.DataFrame:
    """Lowest variance of entropy across decades, among genuinely polysemous lemmas."""
    summ = summary.set_index("lemma")
    eligible_c = {
        l for l in eligible
        if summ.loc[l, "n_clusters_coarse"] >= 4
        and pd.notna(summ.loc[l, "mean_entropy"])
        and summ.loc[l, "mean_entropy"] > 1.0
    }

    ent = entropy[entropy["lemma"].isin(eligible_c)].copy()
    ent["__sup"] = [
        (l, int(d)) in supported for l, d in zip(ent["lemma"], ent["decade"])
    ]
    ent = ent[ent["__sup"]].drop(columns=["__sup"])

    rows = []
    for lemma, sub in ent.groupby("lemma"):
        if len(sub) < 3:
            continue
        rows.append({
            "lemma": lemma,
            "n_clusters":     int(summ.loc[lemma, "n_clusters_coarse"]),
            "mean_entropy":   float(summ.loc[lemma, "mean_entropy"]),
            "entropy_var":    float(sub["entropy"].var(ddof=1)),
            "entropy_std":    float(sub["entropy"].std(ddof=1)),
            "entropy_min":    float(sub["entropy"].min()),
            "entropy_max":    float(sub["entropy"].max()),
            "n_decades_used": int(len(sub)),
            "mean_jsd":       float(summ.loc[lemma, "mean_jsd"]) if pd.notna(summ.loc[lemma, "mean_jsd"]) else np.nan,
        })
    out = pd.DataFrame(rows)
    return out.sort_values("entropy_var", ascending=True).head(TOP_N)


def main() -> int:
    counts  = pd.read_csv(COUNTS_FILE)
    stats   = pd.read_csv(STATS_FILE)
    entropy = pd.read_csv(ENTROPY_FILE)
    summary = pd.read_csv(SUMMARY_FILE)

    supported = _supported_cells(stats)
    eligible  = {
        l for l in summary.loc[summary["n_decades_with_data"] >= MIN_DECADES, "lemma"]
        if summary.set_index("lemma").loc[l, "n_clusters_coarse"] >= 2
    }

    print(f"Total lemmas in summary:   {len(summary)}")
    print(f"Eligible (n_decades >= {MIN_DECADES}, n_clusters >= 2): {len(eligible)}")
    print()

    A = list_a_directional(counts, supported, eligible, summary)
    B = list_b_diversification(entropy, supported, eligible, summary)
    C = list_c_stable(entropy, supported, eligible, summary)

    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda v: f"{v:+.4f}" if isinstance(v, float) else str(v))

    print("=" * 100)
    print("LIST A — Directional shifts  (max |Δproportion| across clusters, first → last supported decade)")
    print("        Eligibility: n_clusters >= 2, n_decades_with_data >= 8")
    print("=" * 100)
    print(A.to_string(index=False))
    print()

    print("=" * 100)
    print("LIST B — Diversification  (entropy_slope > 0 = entropy grows over time)")
    print("        Eligibility: n_clusters >= 3, n_decades_with_data >= 8")
    print("=" * 100)
    print(B.to_string(index=False))
    print()

    print("=" * 100)
    print("LIST C — Stable polysemy  (lowest entropy variance, among genuinely polysemous lemmas)")
    print("        Eligibility: n_clusters >= 4, mean_entropy > 1.0, n_decades_with_data >= 8")
    print("=" * 100)
    print(C.to_string(index=False))
    print()

    print("=" * 100)
    print("Suggested candidates for paper §6.1 review")
    print("=" * 100)
    print(f"  Directional shift   →  {', '.join(A['lemma'].head(5).tolist())}")
    print(f"  Diversification     →  {', '.join(B['lemma'].head(5).tolist())}")
    print(f"  Stable polysemy     →  {', '.join(C['lemma'].head(5).tolist())}")
    print()
    print("Pick one from each list and inspect cluster trajectories manually before committing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())