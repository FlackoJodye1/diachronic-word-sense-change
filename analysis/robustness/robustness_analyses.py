#!/usr/bin/env python3
"""
Robustness analyses for the ARR resubmission (Oct 2026).

Read-only on all existing data. Writes only to analysis/robustness/.

  1. Single-cluster check   — are k=1 lemmas (JSD = 0, entropy = 0 by construction)
                              in the Hamilton / survey correlations?
  2. H2 partial correlation — does entropy explain perceived polysemy beyond n_clusters?
  3. JSD null baseline      — per lemma, permute the time-bin labels of the sentences
                              (bin sizes fixed), recompute mean adjacent-bin JSD
                              exactly as per_lemma_summary.py does (both endpoints
                              n_contexts >= 10); excess = observed - mean(null), z, p.
  4. Split-half reliability of the survey ratings.

The time-binning primitives in part 3 are shared with binning_sensitivity.py,
which re-runs the same null baseline over coarser period schemes.

Inputs (read-only):
  data/processed/diachronic/aggregated/pilot_v3_coarse_sense_counts.csv
  data/processed/analysis/per_lemma_summary.csv
  data/processed/analysis/dlexdb_kern_freq.csv
  data/processed/analysis/baumann_predictors.csv
  data/processed/survey/survey_polysemy_per_lemma.csv
  data/external/survey/survey_polysemy_per_lemma.xlsx

Outputs:
  analysis/robustness/jsd_null_baseline.csv
  analysis/robustness/per_lemma_robustness.csv
  analysis/robustness/correlations.csv

Usage:  python analysis/robustness/robustness_analyses.py [--perms 1000] [--seed 42]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import rankdata, spearmanr, t as tdist, wilcoxon

BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "analysis/robustness"

COUNTS_FILE  = BASE_DIR / "data/processed/diachronic/aggregated/pilot_v3_coarse_sense_counts.csv"
SUMMARY_FILE = BASE_DIR / "data/processed/analysis/per_lemma_summary.csv"
SURVEY_FILE  = BASE_DIR / "data/processed/survey/survey_polysemy_per_lemma.csv"
SURVEY_RAW   = BASE_DIR / "data/external/survey/survey_polysemy_per_lemma.xlsx"
DLEXDB_FILE  = BASE_DIR / "data/processed/analysis/dlexdb_kern_freq.csv"
# log_frequency = log(# retrieved DWDS contexts), i.e. capped per lemma — this is
# what the submitted Conformity numbers were actually computed on (see RESULTS.md §1).
BAUMANN_FILE = BASE_DIR / "data/processed/analysis/baumann_predictors.csv"

OUT_NULL         = OUT_DIR / "jsd_null_baseline.csv"
OUT_PER_LEMMA    = OUT_DIR / "per_lemma_robustness.csv"
OUT_CORRELATIONS = OUT_DIR / "correlations.csv"

MIN_CONTEXTS = 10  # support filter for bin-level metrics


def _require(path: Path) -> Path:
    if not path.exists():
        sys.exit(f"ERROR: required input not found: {path.relative_to(BASE_DIR)}")
    return path


# ----------------------------------------------------------------------
# Time-binned JSD + permutation null (shared with binning_sensitivity.py)
# ----------------------------------------------------------------------

def mean_adjacent_jsd(counts: np.ndarray, keep: list[bool]) -> float:
    """Mean JSD (base 2) between adjacent rows of a bin x sense count matrix.

    `keep[i]` selects the transition from row i to row i+1; it encodes the
    support filter, so unsupported bins never contribute (and never trigger a
    divide-by-zero on an all-empty row).
    """
    props = counts / counts.sum(axis=1, keepdims=True)
    jsds = [
        jensenshannon(props[i], props[i + 1], base=2)
        for i in range(len(props) - 1)
        if keep[i]
    ]
    return float(np.mean(jsds)) if jsds else np.nan


def permutation_null(
    counts: np.ndarray, keep: list[bool], perms: int, rng: np.random.Generator
) -> np.ndarray:
    """Null distribution of mean adjacent-bin JSD under "no change over time".

    Pools the lemma's sentences into one bag of sense labels (preserving the
    lemma's overall sense distribution), reshuffles it, and re-splits it into
    bins of exactly the observed sizes. Any JSD that survives is sampling noise.
    """
    n_senses = counts.shape[1]
    labels = np.repeat(np.arange(n_senses), counts.sum(axis=0))
    cuts = np.cumsum(counts.sum(axis=1))[:-1]
    nulls = np.empty(perms)
    for i in range(perms):
        parts = np.split(rng.permutation(labels), cuts)
        shuffled = np.stack([np.bincount(p, minlength=n_senses) for p in parts])
        nulls[i] = mean_adjacent_jsd(shuffled, keep)
    return nulls


def jsd_null_baseline(
    counts_df: pd.DataFrame,
    bin_col: str = "decade",
    perms: int = 1000,
    seed: int = 42,
    min_contexts: int = MIN_CONTEXTS,
) -> pd.DataFrame:
    """Per-lemma observed vs. null mean adjacent-bin JSD.

    `counts_df` needs columns (lemma, <bin_col>, predicted_sense_id, count).
    Bin support is derived from the counts themselves (the per-bin count total
    equals n_contexts in the stats file). Transitions count only when *both*
    endpoint bins have >= `min_contexts` sentences, matching per_lemma_summary.py.

    One RNG for the whole pass, lemmas in groupby (alphabetical) order; lemmas
    with no usable transition or a single sense draw no random numbers.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for lemma, group in counts_df.groupby("lemma"):
        pivot = group.pivot_table(
            index=bin_col, columns="predicted_sense_id", values="count", fill_value=0
        ).sort_index()
        counts = pivot.values.astype(int)
        supported = counts.sum(axis=1) >= min_contexts
        keep = [bool(supported[i] and supported[i + 1]) for i in range(len(counts) - 1)]

        observed = mean_adjacent_jsd(counts, keep)

        # Degenerate: no supported transition, or a single sense (JSD = 0 always).
        if np.isnan(observed) or counts.shape[1] < 2:
            rows.append(dict(
                lemma=lemma,
                jsd_obs=observed,
                jsd_null_mean=np.nan if np.isnan(observed) else 0.0,
                jsd_null_sd=np.nan,
                jsd_excess=observed,
                jsd_z=np.nan,
                p_perm=np.nan,
            ))
            continue

        nulls = permutation_null(counts, keep, perms, rng)
        null_mean, null_sd = nulls.mean(), nulls.std(ddof=1)
        rows.append(dict(
            lemma=lemma,
            jsd_obs=observed,
            jsd_null_mean=null_mean,
            jsd_null_sd=null_sd,
            jsd_excess=observed - null_mean,
            jsd_z=(observed - null_mean) / null_sd if null_sd > 0 else np.nan,
            p_perm=(1 + (nulls >= observed).sum()) / (perms + 1),
        ))

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Correlation helpers
# ----------------------------------------------------------------------

def spearman(df: pd.DataFrame, x: str, y: str) -> tuple[int, float, float]:
    sub = df[[x, y]].dropna()
    result = spearmanr(sub[x], sub[y])
    return len(sub), float(result.statistic), float(result.pvalue)


def partial_spearman(df: pd.DataFrame, x: str, y: str, z: str) -> tuple[int, float, float]:
    """Spearman correlation of x and y with z partialled out (rank residuals)."""
    sub = df[[x, y, z]].dropna()
    rank_x, rank_y, rank_z = (rankdata(sub[c]) for c in (x, y, z))
    resid_x = rank_x - np.polyval(np.polyfit(rank_z, rank_x, 1), rank_z)
    resid_y = rank_y - np.polyval(np.polyfit(rank_z, rank_y, 1), rank_z)
    r = float(np.corrcoef(resid_x, resid_y)[0, 1])
    n = len(sub)
    t_stat = r * np.sqrt((n - 3) / (1 - r * r))
    return n, r, float(2 * tdist.sf(abs(t_stat), n - 3))


# ----------------------------------------------------------------------
# Survey split-half reliability
# ----------------------------------------------------------------------

def split_half_reliability(n_iter: int = 1000, seed: int = 1) -> tuple[int, int, np.ndarray]:
    """Spearman-Brown corrected split-half reliability over random participant splits.

    Duplicate lemma columns (the survey asked some lemmas twice) are averaged
    per participant first, so each lemma contributes one rating per person.
    """
    raw = pd.read_excel(_require(SURVEY_RAW), sheet_name="Tabelle1")
    ratings = (
        raw.drop(columns=["Interview-Nummer (fortlaufend)"])
        .apply(pd.to_numeric, errors="coerce")
    )
    per_lemma = (
        ratings.T.groupby([re.sub(r"\.\d+$", "", str(c)) for c in ratings.columns])
        .mean()
        .T
    )

    rng = np.random.default_rng(seed)
    reliabilities = []
    for _ in range(n_iter):
        order = rng.permutation(per_lemma.shape[0])
        half = len(order) // 2
        a = per_lemma.iloc[order[:half]].mean()
        b = per_lemma.iloc[order[half:]].mean()
        both = a.notna() & b.notna()
        r = spearmanr(a[both], b[both]).statistic
        reliabilities.append(2 * r / (1 + r))

    return per_lemma.shape[0], per_lemma.shape[1], np.array(reliabilities)


# ----------------------------------------------------------------------
# Correlation battery
# ----------------------------------------------------------------------

# (test label, x, y) — "raw" y's are the submitted metrics, jsd_excess / jsd_z
# the null-corrected ones.
CORRELATION_TESTS = [
    ("Conformity", "log_n_contexts",           "mean_jsd"),
    ("Conformity", "log_freq_dlexdb",          "mean_jsd"),
    ("Conformity", "log_n_contexts",           "mean_jsd_normalized"),
    ("Conformity", "log_freq_dlexdb",          "mean_jsd_normalized"),
    ("Conformity", "log_freq_dlexdb",          "jsd_excess"),
    ("Conformity", "log_freq_dlexdb",          "jsd_z"),
    ("Innovation", "n_clusters_coarse",        "mean_jsd"),
    ("Innovation", "n_clusters_coarse",        "jsd_null_mean"),
    ("Innovation", "n_clusters_coarse",        "jsd_excess"),
    ("Innovation", "n_clusters_coarse",        "jsd_z"),
    ("Innovation", "mean_entropy",             "mean_jsd"),
    ("Innovation", "mean_entropy",             "jsd_excess"),
    ("Innovation", "mean_entropy",             "jsd_z"),
    ("Innovation", "subjective_polysemy_mean", "mean_jsd"),
    ("Innovation", "subjective_polysemy_mean", "jsd_excess"),
    ("Innovation", "subjective_polysemy_mean", "jsd_z"),
    ("H1",         "subjective_polysemy_mean", "n_clusters_coarse"),
    ("H2",         "subjective_polysemy_mean", "mean_entropy"),
]


def build_per_lemma_table(null_df: pd.DataFrame) -> pd.DataFrame:
    """Join the null baseline onto the submitted per-lemma metrics + predictors."""
    survey = pd.read_csv(_require(SURVEY_FILE))[["lemma", "subjective_polysemy_mean"]]
    dlexdb = pd.read_csv(_require(DLEXDB_FILE))[["lemma", "log_freq_dlexdb"]]
    baumann = (
        pd.read_csv(_require(BAUMANN_FILE))[["lemma", "log_frequency"]]
        .rename(columns={"log_frequency": "log_n_contexts"})
    )

    merged = (
        pd.read_csv(_require(SUMMARY_FILE))
        .merge(null_df, on="lemma", how="left")
        .merge(survey, on="lemma", how="left")
        .merge(dlexdb, on="lemma", how="left")
        .merge(baumann, on="lemma", how="left")
    )

    # Guard: the null baseline must reproduce the submitted mean_jsd exactly.
    max_diff = (merged["jsd_obs"] - merged["mean_jsd"]).abs().max()
    assert max_diff < 1e-9, f"null-baseline JSD does not reproduce mean_jsd (max diff {max_diff:.2e})"
    return merged


def run_correlations(per_lemma: pd.DataFrame) -> pd.DataFrame:
    subsets = [("all", per_lemma), ("k>1", per_lemma[per_lemma["n_clusters_coarse"] > 1])]
    rows = []
    for subset_name, subset in subsets:
        for test, x, y in CORRELATION_TESTS:
            n, r, p = spearman(subset, x, y)
            rows.append(dict(subset=subset_name, test=test, x=x, y=y, n=n, spearman_r=r, p=p))
        n, r, p = partial_spearman(
            subset, "subjective_polysemy_mean", "mean_entropy", "n_clusters_coarse"
        )
        rows.append(dict(
            subset=subset_name, test="H2 partial | n_clusters",
            x="subjective_polysemy_mean", y="mean_entropy", n=n, spearman_r=r, p=p,
        ))
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perms", type=int, default=1000, help="permutations per lemma")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for the null baseline")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 3. JSD null baseline (decade binning, as submitted)
    # ------------------------------------------------------------------
    counts_df = pd.read_csv(_require(COUNTS_FILE))
    null_df = jsd_null_baseline(counts_df, bin_col="decade", perms=args.perms, seed=args.seed)
    null_df.to_csv(OUT_NULL, index=False)

    per_lemma = build_per_lemma_table(null_df)
    per_lemma.to_csv(OUT_PER_LEMMA, index=False)

    # ------------------------------------------------------------------
    # 1.-2. Correlation battery (all lemmas vs. k > 1 only)
    # ------------------------------------------------------------------
    results = run_correlations(per_lemma)
    results.to_csv(OUT_CORRELATIONS, index=False)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    scored = per_lemma[per_lemma["jsd_z"].notna()]
    n_single = int((per_lemma["n_clusters_coarse"] == 1).sum())
    n_single_hamilton = int(
        ((per_lemma["n_clusters_coarse"] == 1) & per_lemma["mean_jsd"].notna()).sum()
    )
    print(f"Single-cluster lemmas: {n_single} of {len(per_lemma)}; "
          f"in Hamilton n=485: {n_single_hamilton}")
    print(f"Mean null JSD / mean observed JSD: "
          f"{per_lemma['jsd_null_mean'].mean() / per_lemma['jsd_obs'].mean():.3f}")
    print(f"Lemmas with p_perm < .05: {(scored['p_perm'] < .05).sum()} of {len(scored)}; "
          f"Wilcoxon excess > 0: p = {wilcoxon(scored['jsd_excess']).pvalue:.1e}")
    print(results.to_string(index=False))

    n_participants, n_lemmas, reliabilities = split_half_reliability()
    print(f"Split-half reliability ({n_participants} participants, {n_lemmas} lemmas): "
          f"{reliabilities.mean():.3f} "
          f"[{np.percentile(reliabilities, 2.5):.3f}, {np.percentile(reliabilities, 97.5):.3f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
