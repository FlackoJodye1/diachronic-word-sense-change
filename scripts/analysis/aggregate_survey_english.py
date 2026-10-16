#!/usr/bin/env python3
"""
Aggregate Amandine Grieshaber's English polysemy survey to per-lemma statistics.

Input:
  data/external/survey/survey_english.xlsx (sheet 'SurveyEnglishPolysemy_Data')
    Raw shape: 69 × 619. Two header rows (no data header inferred — read with
    `header=None`):
      Row 0: SoSci column codes (WA01_01, …). The first column is `CASE`.
      Row 1: human-readable mapping; lemma cells are 'Senses: <lemma>'.
    Rows 2+ are participant responses (67 rows).
  Some lemmas appear up to 3× (control items / repeats). 558 unique lemmas.

For each unique lemma, pool all numeric responses across every column whose
row-1 label is 'Senses: <lemma>', across every participant; drop NaN; compute
mean / median / std / n_responses.

Output:
  data/processed/survey/survey_english_per_lemma.csv

Expected: 558 lemmas, mean of per-lemma means ≈ 2.04 (matches G&R report).
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR    = Path(__file__).resolve().parents[2]

INPUT_FILE  = BASE_DIR / "data/external/survey/survey_english.xlsx"
SHEET_NAME  = "SurveyEnglishPolysemy_Data"
OUTPUT_FILE = BASE_DIR / "data/processed/survey/survey_english_per_lemma.csv"

_LEMMA_PAT = re.compile(r"^Senses:\s*(.+)$")


def main() -> int:
    if not INPUT_FILE.exists():
        sys.exit(f"ERROR: survey file not found: {INPUT_FILE.relative_to(BASE_DIR)}")

    raw = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME, header=None)

    # Build the column-index → lemma mapping from row 1.
    metadata_row = raw.iloc[1]
    col_to_lemma: dict[int, str] = {}
    for col_idx, v in enumerate(metadata_row):
        if isinstance(v, str):
            m = _LEMMA_PAT.match(v)
            if m:
                col_to_lemma[col_idx] = m.group(1).strip()

    if not col_to_lemma:
        sys.exit("ERROR: no 'Senses: <lemma>' cells found in row 1 of the metadata.")

    # Response block: row 2 onward, lemma columns only.
    response = raw.iloc[2:, :].reset_index(drop=True)
    lemma_cols = sorted(col_to_lemma.keys())
    block = response.iloc[:, lemma_cols].apply(pd.to_numeric, errors="coerce")

    # Group columns by lemma (pooling duplicate columns).
    lemma_to_idxs: dict[str, list[int]] = defaultdict(list)
    for col_idx in lemma_cols:
        lemma_to_idxs[col_to_lemma[col_idx]].append(col_idx)

    # Build positional index inside `block` (block's columns are 0..N-1 in lemma_cols order).
    block.columns = lemma_cols  # so .loc[:, lemma_cols] works

    rows = []
    for lemma in sorted(lemma_to_idxs):
        cols = lemma_to_idxs[lemma]
        values = block[cols].values.flatten()
        values = values[~np.isnan(values.astype(float))]
        if len(values) == 0:
            rows.append({
                "lemma": lemma,
                "subjective_polysemy_mean":   np.nan,
                "subjective_polysemy_median": np.nan,
                "subjective_polysemy_std":    np.nan,
                "n_responses":                0,
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
    n_lemmas         = len(out)
    n_with_responses = int((out["n_responses"] > 0).sum())
    mean_of_means    = float(out["subjective_polysemy_mean"].mean())
    median_of_means  = float(out["subjective_polysemy_mean"].median())
    n_duplicates     = sum(1 for cs in lemma_to_idxs.values() if len(cs) > 1)
    n_participants   = len(response)

    print(f"Wrote: {OUTPUT_FILE.relative_to(BASE_DIR)}")
    print(f"Participants (response rows):  {n_participants}")
    print(f"Lemmas:                        {n_lemmas}")
    print(f"Lemmas with ≥1 response:       {n_with_responses}")
    print(f"Lemmas with duplicate columns: {n_duplicates}")
    print(f"Mean of per-lemma means:       {mean_of_means:.3f}")
    print(f"Median of per-lemma means:     {median_of_means:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())