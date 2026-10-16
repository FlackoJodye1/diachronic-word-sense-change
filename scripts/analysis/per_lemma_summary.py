#!/usr/bin/env python3
"""
Per-lemma summary table for survey correlation analysis (EMNLP §4).

Aggregates v3 *coarse* diachronic metrics to one row per lemma, joining
in the manually-reviewed cluster mapping. Pure summarisation — does NOT
re-run any model, clustering, or aggregation step.

Outputs:
  lemma                 — string
  n_clusters_coarse     — # distinct reviewed clusters from sense_clusters_reviewed.json
  mean_entropy          — mean entropy across decades with n_contexts >= 10
  mean_jsd              — mean adjacent-decade JSD where both endpoints have n_contexts >= 10
  mean_jsd_normalized   — mean_jsd / n_clusters_coarse  (the divergence CSV has no
                          pre-computed normalised JSD column, so we compute it here)
  n_decades_with_data   — count of decades for this lemma with n_contexts >= 10

Inputs (read-only):
  data/processed/diachronic/metrics/pilot_v3_coarse_entropy.csv
  data/processed/diachronic/metrics/pilot_v3_coarse_divergence.csv
  data/processed/diachronic/aggregated/pilot_v3_coarse_stats.csv  (for n_contexts)
  data/processed/germanet/sense_clusters_reviewed.json

Output:
  data/processed/analysis/per_lemma_summary.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

ENTROPY_FILE    = BASE_DIR / "data/processed/diachronic/metrics/pilot_v3_coarse_entropy.csv"
DIVERGENCE_FILE = BASE_DIR / "data/processed/diachronic/metrics/pilot_v3_coarse_divergence.csv"
STATS_FILE      = BASE_DIR / "data/processed/diachronic/aggregated/pilot_v3_coarse_stats.csv"
CLUSTERS_FILE   = BASE_DIR / "data/processed/germanet/sense_clusters_reviewed.json"

OUTPUT_FILE     = BASE_DIR / "data/processed/analysis/per_lemma_summary.csv"

MIN_CONTEXTS = 10  # support filter for decade-level metrics


def _require(path: Path) -> Path:
    if not path.exists():
        sys.exit(f"ERROR: required input not found: {path.relative_to(BASE_DIR)}")
    return path


def main() -> int:
    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    entropy_df    = pd.read_csv(_require(ENTROPY_FILE))
    divergence_df = pd.read_csv(_require(DIVERGENCE_FILE))
    stats_df      = pd.read_csv(_require(STATS_FILE))[["lemma", "decade", "n_contexts"]]

    with _require(CLUSTERS_FILE).open(encoding="utf-8") as f:
        clusters_data = json.load(f)

    # ------------------------------------------------------------------
    # Cluster counts: n distinct cluster IDs per lemma from the reviewed mapping.
    # The JSON covers 503 lemmas; we only join in the ones present in the metrics.
    # ------------------------------------------------------------------
    n_clusters_by_lemma: dict[str, int] = {
        lemma: len(payload.get("clusters", {}))
        for lemma, payload in clusters_data.items()
    }

    # ------------------------------------------------------------------
    # Support filter on (lemma, decade): keep decades with n_contexts >= MIN_CONTEXTS
    # ------------------------------------------------------------------
    supported = stats_df[stats_df["n_contexts"] >= MIN_CONTEXTS][["lemma", "decade"]]
    supported_set = set(map(tuple, supported.values.tolist()))

    # n_decades_with_data: per-lemma count of supported decades. Built from ALL
    # lemmas present in the stats file (which is the 90 v3 pilot lemmas).
    n_decades_with_data = (
        supported.groupby("lemma").size().rename("n_decades_with_data")
    )

    # ------------------------------------------------------------------
    # mean_entropy: filter entropy rows to supported (lemma, decade), then mean.
    # ------------------------------------------------------------------
    ent_mask = entropy_df.apply(
        lambda r: (r["lemma"], r["decade"]) in supported_set, axis=1
    )
    mean_entropy = (
        entropy_df[ent_mask].groupby("lemma")["entropy"].mean().rename("mean_entropy")
    )

    # ------------------------------------------------------------------
    # mean_jsd: a transition counts only if BOTH endpoint decades pass the support filter.
    # ------------------------------------------------------------------
    div_mask = divergence_df.apply(
        lambda r: ((r["lemma"], r["decade_from"]) in supported_set)
                  and ((r["lemma"], r["decade_to"])   in supported_set),
        axis=1,
    )
    mean_jsd = (
        divergence_df[div_mask].groupby("lemma")["jsd"].mean().rename("mean_jsd")
    )

    # ------------------------------------------------------------------
    # Assemble final table. Base set = lemmas present in the v3 pilot stats file
    # (i.e. the 90 lemmas with diachronic data). Don't drop any of them, even if
    # they fail the support filter for every decade.
    # ------------------------------------------------------------------
    base_lemmas = sorted(stats_df["lemma"].unique())
    out = pd.DataFrame({"lemma": base_lemmas})
    out["n_clusters_coarse"] = out["lemma"].map(n_clusters_by_lemma)
    out = out.merge(n_decades_with_data, how="left", on="lemma")
    out["n_decades_with_data"] = out["n_decades_with_data"].fillna(0).astype(int)
    out = out.merge(mean_entropy, how="left", on="lemma")
    out = out.merge(mean_jsd,     how="left", on="lemma")
    out["mean_jsd_normalized"] = out["mean_jsd"] / out["n_clusters_coarse"]

    # Final column order
    out = out[[
        "lemma",
        "n_clusters_coarse",
        "mean_entropy",
        "mean_jsd",
        "mean_jsd_normalized",
        "n_decades_with_data",
    ]]

    # ------------------------------------------------------------------
    # Save + report
    # ------------------------------------------------------------------
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)

    print(f"\nWrote: {OUTPUT_FILE.relative_to(BASE_DIR)}")
    print(f"Rows : {len(out)} lemmas")
    print()
    print("=== Head (20) ===")
    print(out.head(20).to_string(index=False))
    print()
    print("=== Tail ===")
    print(out.tail().to_string(index=False))
    print()
    print("=== Summary stats ===")
    print(out.describe(include="all").to_string())
    print()
    n_missing_clusters = int(out["n_clusters_coarse"].isna().sum())
    n_no_support      = int((out["n_decades_with_data"] == 0).sum())
    n_nan_entropy     = int(out["mean_entropy"].isna().sum())
    n_nan_jsd         = int(out["mean_jsd"].isna().sum())
    print(f"Lemmas without entry in clusters JSON: {n_missing_clusters}")
    print(f"Lemmas with 0 supported decades:       {n_no_support}")
    print(f"Lemmas with NaN mean_entropy:          {n_nan_entropy}")
    print(f"Lemmas with NaN mean_jsd:              {n_nan_jsd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())