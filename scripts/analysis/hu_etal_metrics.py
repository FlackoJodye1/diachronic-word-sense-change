#!/usr/bin/env python3
"""
Per-word entropy and JSD metrics from Hu et al. (2019) prob_fitting_10.data.

Computes analogues of our German `per_lemma_summary.csv` so they can later be
joined with Amandine Grieshaber's English subjective-polysemy survey.

Input:
  data/external/hu_etal_2019/prob_fitting_10.data — pickle from Hu et al.'s
  GitHub. Top-level dict keyed by lemma. Each value is itself a dict keyed by
  sense_id (e.g. `body_1_noun_5`). Each sense dict has:
    - definition: str
    - x: list[int] of decade labels
    - y: list[float] of proportions for that sense across decades
    - y_fitting: smoothed version of y (we ignore this — same operationalisation
      as the German pipeline, no smoothing)
  Within a lemma, all senses share the same x. Column sums of the stacked
  matrix (across senses, per decade) are 1.0 (verified at write time).

Per lemma:
  n_senses      — number of senses (rows of the stacked matrix)
  mean_entropy  — Shannon entropy (bits) of the per-decade sense distribution,
                  averaged across decades. scipy.stats.entropy handles 0s.
  mean_jsd      — Jensen-Shannon distance (bits, sqrt of JS divergence) between
                  consecutive decade columns, averaged.
  n_decades     — # decades for this lemma

Output:
  data/processed/analysis/hu_etal_per_word.csv

Edge cases:
  - 1 sense  → mean_entropy = 0     (degenerate; entropy of a delta is 0)
  - 1 decade → mean_jsd     = NaN   (no consecutive pair)
  - Empty payload (no senses) → lemma skipped entirely
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

BASE_DIR    = Path(__file__).resolve().parents[2]
INPUT_FILE  = BASE_DIR / "data/external/hu_etal_2019/prob_fitting_10.data"
OUTPUT_FILE = BASE_DIR / "data/processed/analysis/hu_etal_per_word.csv"


def _stack_lemma_matrix(payload: dict) -> tuple[np.ndarray, list[int]] | None:
    """Stack a lemma's sense dicts into a (n_senses, n_decades) matrix.

    Assumes all senses share the same x (verified empirically across all 3,220
    populated lemmas in the pickle).

    Returns:
        (matrix of proportions, decade list) or None if the payload is empty.
    """
    if not payload:
        return None
    senses = list(payload.values())
    x = senses[0]["x"]
    y = np.array([s["y"] for s in senses], dtype=float)
    return y, list(x)


def _metrics(mat: np.ndarray) -> tuple[float, float]:
    """Compute (mean_entropy_bits, mean_jsd_bits) from a (n_senses, n_decades) matrix.

    Entropy averaged across decade columns. JSD averaged across consecutive
    decade-column pairs. Returns NaN for jsd if only one decade.
    """
    # Entropy per decade column. Single-sense lemma → 0.
    if mat.shape[0] == 1:
        mean_entropy = 0.0
    else:
        per_decade_entropy = entropy(mat, base=2, axis=0)
        mean_entropy = float(np.mean(per_decade_entropy))

    # JSD between consecutive decade columns.
    if mat.shape[1] < 2:
        mean_jsd = float("nan")
    else:
        jsds = [
            jensenshannon(mat[:, j], mat[:, j + 1], base=2)
            for j in range(mat.shape[1] - 1)
        ]
        mean_jsd = float(np.mean(jsds))

    return mean_entropy, mean_jsd


def main() -> int:
    if not INPUT_FILE.exists():
        sys.exit(f"ERROR: input not found: {INPUT_FILE.relative_to(BASE_DIR)}")

    with INPUT_FILE.open("rb") as f:
        data = pickle.load(f)

    n_total = len(data)
    n_skipped_empty = 0
    rows: list[dict] = []

    for lemma, payload in data.items():
        stacked = _stack_lemma_matrix(payload)
        if stacked is None:
            n_skipped_empty += 1
            continue
        mat, decades = stacked
        mean_h, mean_jsd = _metrics(mat)
        rows.append({
            "lemma":        lemma,
            "n_senses":     int(mat.shape[0]),
            "mean_entropy": mean_h,
            "mean_jsd":     mean_jsd,
            "n_decades":    len(decades),
        })

    out = pd.DataFrame(rows)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    n_nan_jsd = int(out["mean_jsd"].isna().sum())
    n_nan_ent = int(out["mean_entropy"].isna().sum())  # should be 0 by construction

    print(f"\nWrote: {OUTPUT_FILE.relative_to(BASE_DIR)}")
    print(f"Pickle total lemmas:    {n_total}")
    print(f"Empty payloads skipped: {n_skipped_empty}")
    print(f"Lemmas in output CSV:   {len(out)}")
    print()
    print(f"Mean entropy across lemmas: {out['mean_entropy'].mean():.3f} bits")
    print(f"Mean JSD across lemmas:     {out['mean_jsd'].mean():.4f} bits")
    print(f"Lemmas with NaN mean_jsd:     {n_nan_jsd}  (single-decade lemmas)")
    print(f"Lemmas with NaN mean_entropy: {n_nan_ent}")
    print()
    print("=== First 10 rows ===")
    print(out.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
