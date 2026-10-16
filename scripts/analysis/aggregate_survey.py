#!/usr/bin/env python3
"""
Aggregate the Grieshaber & Rubinskii polysemy survey to per-lemma statistics.

Input:
  data/external/survey/survey_polysemy_per_lemma.xlsx (sheet 'Tabelle1')
    — 78 participants × 593 columns. First column is the interview ID.
    — Other columns are lemmas. Some lemmas appear up to 3 times
      (control items / repeats) with pandas-suffixed names like
      `Nadel.1`, `Nadel.2`. Responses are integer polysemy ratings.

For each unique lemma (after stripping `.N` suffixes), pool all numeric
responses across all duplicate columns and all participants, drop NaN,
and compute mean / median / std / n_responses.

Output:
  data/processed/survey/survey_polysemy_per_lemma.csv

Expected: ~544 lemmas, per-lemma mean of subjective_polysemy_mean ≈ 1.71.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE  = BASE_DIR / "data/external/survey/survey_polysemy_per_lemma.xlsx"
SHEET_NAME  = "Tabelle1"
ID_COL      = "Interview-Nummer (fortlaufend)"
OUTPUT_FILE = BASE_DIR / "data/processed/survey/survey_polysemy_per_lemma.csv"

# Pandas suffix on duplicated column names: trailing ".<digits>"
_DUP_SUFFIX = re.compile(r"\.\d+$")


def main() -> int:
    if not INPUT_FILE.exists():
        sys.exit(f"ERROR: survey file not found: {INPUT_FILE.relative_to(BASE_DIR)}")

    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

    # Drop the participant-ID column.
    if ID_COL not in df.columns:
        sys.exit(f"ERROR: expected ID column {ID_COL!r} not found in sheet {SHEET_NAME!r}.")
    lemma_df = df.drop(columns=[ID_COL])

    # Group columns by lemma (stripping pandas .N duplicate suffix).
    lemma_to_columns: dict[str, list[str]] = defaultdict(list)
    for col in lemma_df.columns:
        base = _DUP_SUFFIX.sub("", str(col))
        lemma_to_columns[base].append(col)

    # Aggregate per lemma: pool all numeric cells across duplicate columns +
    # all participants, drop NaN, compute mean / median / std / n.
    rows = []
    for lemma in sorted(lemma_to_columns):
        cols = lemma_to_columns[lemma]
        values = lemma_df[cols].values.flatten()
        values = values[~np.isnan(values.astype(float))]
        if len(values) == 0:
            rows.append({
                "lemma": lemma,
                "subjective_polysemy_mean": np.nan,
                "subjective_polysemy_median": np.nan,
                "subjective_polysemy_std": np.nan,
                "n_responses": 0,
            })
            continue
        rows.append({
            "lemma": lemma,
            "subjective_polysemy_mean":   float(np.mean(values)),
            "subjective_polysemy_median": float(np.median(values)),
            "subjective_polysemy_std":    float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
            "n_responses":                int(len(values)),
        })

    out = pd.DataFrame(rows)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    n_lemmas = len(out)
    n_with_responses = int((out["n_responses"] > 0).sum())
    mean_of_means    = float(out["subjective_polysemy_mean"].mean())
    median_of_means  = float(out["subjective_polysemy_mean"].median())
    n_duplicates     = sum(1 for cs in lemma_to_columns.values() if len(cs) > 1)

    print(f"Wrote: {OUTPUT_FILE.relative_to(BASE_DIR)}")
    print(f"Lemmas:                  {n_lemmas}")
    print(f"Lemmas with ≥1 response: {n_with_responses}")
    print(f"Lemmas with duplicates:  {n_duplicates}")
    print(f"Mean of per-lemma means:   {mean_of_means:.3f}")
    print(f"Median of per-lemma means: {median_of_means:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())