#!/usr/bin/env python3
"""
Sensitivity of the diachronic JSD results to the temporal binning (Oct 2026).

The JSD null baseline (robustness_analyses.py §3) showed that ~90% of the mean
adjacent-decade JSD is sampling noise at ~50 sentences per lemma-decade. Coarser
time bins pool more sentences per bin, so the noise floor should drop roughly as
1/sqrt(n). This script asks whether any *observed* change survives once the bins
are coarse enough for the noise floor to be low — i.e. whether the Hamilton-style
Conformity and Innovation effects are an artefact of the decade binning.

Binning schemes (pre-specified before running):
  (a) 10 decades, 1900..1990                                    — reference, as submitted
  (b) 5 periods x 20 years: 1900-19, 1920-39, 1940-59, 1960-79, 1980-99
  (c) 2 periods: 1900-1949 vs 1950-1999
  (c') 2 periods with the 1940s dropped: 1900-1939 vs 1950-1999  — tie-break check
  (d) 4 periods: 1900-1929, 1930-1949, 1950-1969, 1970-1999
  (e) 3 periods: 1900-1929, 1930-1959, 1960-1999

NOTE on (c): the pre-specified cut was 1900-1944 vs 1945-1999, but the aggregated
counts exist only at decade resolution, so the 1940s decade cannot be split at
1945. (c) keeps it in the early period (closest realizable split that uses all
data); (c') drops it instead, giving a clean pre-/post-war separation at the cost
of 9.9% of the sentences. Both are reported.

Per binning and lemma: observed mean JSD between adjacent supported periods,
permutation null baseline (period sizes fixed), excess, z, p_perm, BH-FDR.
Restricted to lemmas with n_clusters_coarse > 1 (a single cluster forces JSD = 0).
`mean_entropy` stays the decade-based value from per_lemma_summary.csv — it is
used here as a fixed lemma-level polysemy predictor, not as an outcome, so it is
deliberately not recomputed per binning.

Each binning also gets partial Spearman correlations of the three polysemy
predictors with jsd_excess, controlling for frequency (dlexDB) and for the
lemma's sample size. Dubossarsky et al. (2017) argue the Innovation effect is a
frequency artefact; these ask whether it survives that control.

Finally (§6, exploratory): WSD accuracy on the 93-instance gold set split at
1950, i.e. is sense assignment worse on older German? This is an aside, not part
of the binning analysis — it reads the atemporal evaluation outputs and is
skipped with a warning if they are absent.

Read-only on all existing data. Writes only to analysis/robustness/.

Inputs (read-only):
  data/processed/diachronic/aggregated/pilot_v3_coarse_sense_counts.csv
  data/processed/analysis/per_lemma_summary.csv
  data/processed/analysis/dlexdb_kern_freq.csv
  data/processed/survey/survey_polysemy_per_lemma.csv
  data/evaluation/atemporal/results/predictions.csv        — §6 only
  data/evaluation/atemporal/input/contexts.csv             — §6 only
  data/processed/germanet/sense_clusters_reviewed.json     — §6 only

Outputs:
  analysis/robustness/binning_sensitivity.csv              — per lemma x binning
  analysis/robustness/binning_sensitivity_correlations.csv — per binning x test
  analysis/robustness/binning_sensitivity_wsd_era.csv      — §6 era split

Usage:  python analysis/robustness/binning_sensitivity.py [--perms 1000] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr

# Same directory; reuses the exact JSD + permutation primitives of the submitted
# null baseline, so binning (a) reproduces the published mean_jsd by construction.
# `partial_spearman` is the same rank-residual implementation used for H2 in §3.
from robustness_analyses import MIN_CONTEXTS, jsd_null_baseline, partial_spearman

BASE_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = BASE_DIR / "analysis/robustness"

COUNTS_FILE  = BASE_DIR / "data/processed/diachronic/aggregated/pilot_v3_coarse_sense_counts.csv"
SUMMARY_FILE = BASE_DIR / "data/processed/analysis/per_lemma_summary.csv"
DLEXDB_FILE  = BASE_DIR / "data/processed/analysis/dlexdb_kern_freq.csv"
SURVEY_FILE  = BASE_DIR / "data/processed/survey/survey_polysemy_per_lemma.csv"

# §6 only (exploratory WSD era check)
PREDICTIONS_FILE   = BASE_DIR / "data/evaluation/atemporal/results/predictions.csv"
EVAL_CONTEXTS_FILE = BASE_DIR / "data/evaluation/atemporal/input/contexts.csv"
CLUSTERS_FILE      = BASE_DIR / "data/processed/germanet/sense_clusters_reviewed.json"

OUT_CSV          = OUT_DIR / "binning_sensitivity.csv"
OUT_CORRELATIONS = OUT_DIR / "binning_sensitivity_correlations.csv"
OUT_WSD_ERA      = OUT_DIR / "binning_sensitivity_wsd_era.csv"

ERA_CUT = 1950  # §6: sentences before this year vs. from this year on

# binning label -> {decade: period label}. Decades absent from the map are dropped.
DECADES = list(range(1900, 2000, 10))

BINNINGS: dict[str, dict[int, str]] = {
    "a_10_decades": {d: str(d) for d in DECADES},
    "b_5_periods_20y": {
        d: f"{1900 + 20 * ((d - 1900) // 20)}-{1900 + 20 * ((d - 1900) // 20) + 19}"
        for d in DECADES
    },
    "c_2_periods": {d: "1900-1949" if d < 1950 else "1950-1999" for d in DECADES},
    "c2_2_periods_no1940s": {
        d: "1900-1939" if d < 1940 else "1950-1999" for d in DECADES if d != 1940
    },
    # Period labels are sortable as strings (all start with the 4-digit start year),
    # which is what fixes the adjacency order in jsd_null_baseline.
    "d_4_periods": {
        d: next(
            label for lo, hi, label in [
                (1900, 1929, "1900-1929"), (1930, 1949, "1930-1949"),
                (1950, 1969, "1950-1969"), (1970, 1999, "1970-1999"),
            ] if lo <= d <= hi
        )
        for d in DECADES
    },
    "e_3_periods": {
        d: next(
            label for lo, hi, label in [
                (1900, 1929, "1900-1929"), (1930, 1959, "1930-1959"),
                (1960, 1999, "1960-1999"),
            ] if lo <= d <= hi
        )
        for d in DECADES
    },
}

BINNING_LABELS = {
    "a_10_decades":         "(a) 10 decades (reference)",
    "b_5_periods_20y":      "(b) 5 periods x 20y",
    "c_2_periods":          "(c) 2 periods, 1940s early",
    "c2_2_periods_no1940s": "(c') 2 periods, 1940s dropped",
    "d_4_periods":          "(d) 4 periods",
    "e_3_periods":          "(e) 3 periods",
}

# (test label, predictor) — each is correlated against jsd_obs, jsd_excess and jsd_z.
PREDICTORS = [
    ("Conformity", "log_freq_dlexdb"),
    ("Innovation", "n_clusters_coarse"),
    ("Innovation", "mean_entropy"),
    ("Innovation", "subjective_polysemy_mean"),
]
OUTCOMES = ["jsd_obs", "jsd_excess", "jsd_z"]

# Partial correlations: predictor ~ jsd_excess | control, per binning.
# jsd_excess is the null-corrected outcome, so the only remaining worry is that
# the predictor stands in for frequency (Dubossarsky et al. 2017). Two controls:
# dlexDB corpus frequency (external) and the lemma's own sample size in this
# binning (what actually drives the noise floor).
PARTIAL_PREDICTORS = ["n_clusters_coarse", "mean_entropy", "subjective_polysemy_mean"]
PARTIAL_CONTROLS = ["log_freq_dlexdb", "log_n_contexts"]
PARTIAL_OUTCOME = "jsd_excess"


def _require(path: Path) -> Path:
    if not path.exists():
        sys.exit(f"ERROR: required input not found: {path.relative_to(BASE_DIR)}")
    return path


def benjamini_hochberg(p: pd.Series) -> pd.Series:
    """BH-adjusted p-values; NaN in, NaN out (NaNs excluded from the family)."""
    adjusted = pd.Series(np.nan, index=p.index, dtype=float)
    valid = p.dropna().sort_values()
    if valid.empty:
        return adjusted
    m = len(valid)
    raw = valid.to_numpy() * m / np.arange(1, m + 1)
    adjusted.loc[valid.index] = np.minimum.accumulate(raw[::-1])[::-1].clip(max=1.0)
    return adjusted


def bin_support(counts_df: pd.DataFrame) -> pd.DataFrame:
    """Per lemma x period: total sentences and whether the period passes the filter."""
    totals = counts_df.groupby(["lemma", "period"])["count"].sum().rename("n_contexts")
    out = totals.reset_index()
    out["supported"] = out["n_contexts"] >= MIN_CONTEXTS
    return out


def run_binning(
    counts_df: pd.DataFrame, binning: str, perms: int, seed: int
) -> pd.DataFrame:
    """Null baseline for one binning scheme, plus per-lemma bin bookkeeping."""
    mapping = BINNINGS[binning]
    binned = counts_df[counts_df["decade"].isin(mapping)].copy()
    binned["period"] = binned["decade"].map(mapping)
    binned = (
        binned.groupby(["lemma", "period", "predicted_sense_id"], as_index=False)["count"]
        .sum()
    )

    null_df = jsd_null_baseline(
        binned, bin_col="period", perms=perms, seed=seed, min_contexts=MIN_CONTEXTS
    )

    support = bin_support(binned)
    bookkeeping = support.groupby("lemma").agg(
        n_periods=("period", "size"),
        n_periods_supported=("supported", "sum"),
        n_contexts_total=("n_contexts", "sum"),
    )
    # A transition counts only when both endpoint periods are supported, over the
    # lemma's periods in sort order — the same rule jsd_null_baseline applies.
    transitions = {}
    for lemma, group in support.sort_values(["lemma", "period"]).groupby("lemma"):
        flags = group["supported"].to_numpy()
        transitions[lemma] = int((flags[:-1] & flags[1:]).sum())

    out = null_df.merge(bookkeeping, on="lemma", how="left")
    out["n_transitions_used"] = out["lemma"].map(transitions)
    out["binning"] = binning
    out["p_bh"] = benjamini_hochberg(out["p_perm"])
    # Sample-size control for the partial correlations. Over all decades this is
    # the submitted `log_frequency` of baumann_predictors.csv (= log of the number
    # of retrieved DWDS contexts); recomputed here so it tracks each binning's
    # own sentence pool (binning (c') drops the 1940s).
    out["log_n_contexts"] = np.log(out["n_contexts_total"])
    # Median sentences per supported lemma-period cell, over the lemmas that
    # actually contribute a JSD — the quantity that sets the noise floor.
    # Attached per binning, so it is identical on every row.
    usable_lemmas = set(out.loc[out["jsd_obs"].notna(), "lemma"])
    out["median_n_contexts_supported"] = support.loc[
        support["supported"] & support["lemma"].isin(usable_lemmas), "n_contexts"
    ].median()
    return out


def summarise(df: pd.DataFrame) -> dict:
    """Noise share and significance counts for one binning (k > 1 lemmas)."""
    scored = df[df["jsd_z"].notna()]
    usable = df[df["jsd_obs"].notna()]
    return {
        "n_lemmas": len(usable),
        "n_lemmas_scored": len(scored),
        "median_n_contexts_supported": usable["median_n_contexts_supported"].iloc[0],
        "mean_n_contexts_per_period": (
            usable["n_contexts_total"] / usable["n_periods"]
        ).mean(),
        "mean_jsd_obs": usable["jsd_obs"].mean(),
        "mean_jsd_null": usable["jsd_null_mean"].mean(),
        "mean_jsd_excess": usable["jsd_excess"].mean(),
        "null_share_of_observed": usable["jsd_null_mean"].mean() / usable["jsd_obs"].mean(),
        "n_p_perm_lt_05": int((scored["p_perm"] < .05).sum()),
        "n_bh_lt_05": int((scored["p_bh"] < .05).sum()),
        "frac_excess_gt_0": float((usable["jsd_excess"] > 0).mean()),
    }


def correlations(df: pd.DataFrame, binning: str) -> list[dict]:
    rows = []
    for test, x in PREDICTORS:
        for y in OUTCOMES:
            sub = df[[x, y]].dropna()
            if len(sub) < 3:
                rows.append(dict(binning=binning, test=test, method="spearman",
                                 x=x, y=y, control="", n=len(sub),
                                 spearman_r=np.nan, p=np.nan))
                continue
            result = spearmanr(sub[x], sub[y])
            rows.append(dict(binning=binning, test=test, method="spearman",
                             x=x, y=y, control="", n=len(sub),
                             spearman_r=float(result.statistic), p=float(result.pvalue)))
    return rows


def partial_correlations(df: pd.DataFrame, binning: str) -> list[dict]:
    """Polysemy ~ jsd_excess with frequency / sample size partialled out.

    Ranks everything, residualises predictor and outcome on the control, then
    correlates the residuals; p from t with df = n - 3 (`partial_spearman`).
    """
    rows = []
    for x in PARTIAL_PREDICTORS:
        for control in PARTIAL_CONTROLS:
            sub = df[[x, PARTIAL_OUTCOME, control]].dropna()
            if len(sub) < 4:
                rows.append(dict(binning=binning, test="Innovation partial",
                                 method="partial_spearman", x=x, y=PARTIAL_OUTCOME,
                                 control=control, n=len(sub),
                                 spearman_r=np.nan, p=np.nan))
                continue
            n, r, p = partial_spearman(sub, x, PARTIAL_OUTCOME, control)
            rows.append(dict(binning=binning, test="Innovation partial",
                             method="partial_spearman", x=x, y=PARTIAL_OUTCOME,
                             control=control, n=n, spearman_r=r, p=p))
    return rows


# ----------------------------------------------------------------------
# §6 (exploratory): is WSD accuracy worse on older German?
# ----------------------------------------------------------------------

def flat_coarse_mapping() -> dict[str, str]:
    """sense_id -> coarse cluster id, mirroring generate_predictions.py."""
    with open(CLUSTERS_FILE) as f:
        clusters = json.load(f)
    flat: dict[str, str] = {}
    for lemma_data in clusters.values():
        flat.update(lemma_data["mapping"])
    return flat


def wsd_era_check() -> pd.DataFrame | None:
    """Fine and coarse accuracy on the 93-instance gold set, split at ERA_CUT.

    The submitted accuracies (43.0% fine / 84.9% coarse) were measured on this
    set as a whole; the set spans 1901-1999 (median 1932). Splitting it asks
    whether the embedder does worse on early-century German. n is tiny, so this
    is a direction check, not an estimate — hence Fisher's exact test rather
    than a reported effect size.

    Returns None (with a warning) if the evaluation outputs are not present.
    """
    missing = [p for p in (PREDICTIONS_FILE, EVAL_CONTEXTS_FILE, CLUSTERS_FILE) if not p.exists()]
    if missing:
        print("  skipped: missing " + ", ".join(str(p.relative_to(BASE_DIR)) for p in missing))
        return None

    predictions = pd.read_csv(PREDICTIONS_FILE)
    contexts = pd.read_csv(EVAL_CONTEXTS_FILE)[["lemma", "Hit", "Date"]]

    # contexts.csv holds the sentence under `Hit` and is the only source of dates.
    merged = predictions.merge(
        contexts, left_on=["lemma", "sentence"], right_on=["lemma", "Hit"], how="left"
    )
    assert len(merged) == len(predictions), "eval date join is not 1:1"
    assert merged["Date"].notna().all(), "some eval instances have no date"

    flat = flat_coarse_mapping()
    merged["gold_coarse"] = merged["gold_sense"].map(flat)
    merged["pred_coarse"] = merged["pred_sense"].map(flat)
    assert merged[["gold_coarse", "pred_coarse"]].notna().all().all(), "unmapped sense in eval set"

    merged["year"] = merged["Date"].str.slice(0, 4).astype(int)
    merged["era"] = np.where(merged["year"] < ERA_CUT, f"<{ERA_CUT}", f">={ERA_CUT}")
    merged["correct_fine"] = merged["correct"].astype(bool)
    merged["correct_coarse"] = merged["gold_coarse"] == merged["pred_coarse"]

    # The gold set covers only 10 lemmas, and `hoeren` has no post-1950 instance,
    # so an era gap could be lemma composition rather than anything about the
    # language. The second subset holds coverage roughly constant by keeping only
    # the lemmas attested in both eras.
    eras_per_lemma = merged.groupby("lemma")["era"].nunique()
    both_eras = set(eras_per_lemma[eras_per_lemma == 2].index)
    subsets = [
        ("all", merged),
        ("lemmas_in_both_eras", merged[merged["lemma"].isin(both_eras)]),
    ]

    rows = []
    for subset, frame in subsets:
        for granularity in ("fine", "coarse"):
            hits = frame.groupby("era")[f"correct_{granularity}"].agg(["sum", "size"])
            early, late = hits.loc[f"<{ERA_CUT}"], hits.loc[f">={ERA_CUT}"]
            # 2x2 of correct/incorrect by era; exact test because the cells are small.
            _, p = fisher_exact([
                [int(early["sum"]), int(early["size"] - early["sum"])],
                [int(late["sum"]), int(late["size"] - late["sum"])],
            ])
            rows.append({
                "subset": subset,
                "granularity": granularity,
                "n_lemmas": int(frame["lemma"].nunique()),
                "n_early": int(early["size"]),
                "n_correct_early": int(early["sum"]),
                "accuracy_early": early["sum"] / early["size"],
                "n_late": int(late["size"]),
                "n_correct_late": int(late["sum"]),
                "accuracy_late": late["sum"] / late["size"],
                "accuracy_diff_late_minus_early": (
                    late["sum"] / late["size"] - early["sum"] / early["size"]
                ),
                "fisher_p": float(p),
                "n_total": int(hits["size"].sum()),
                "accuracy_total": hits["sum"].sum() / hits["size"].sum(),
            })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perms", type=int, default=1000, help="permutations per lemma")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for the null baseline")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load inputs; restrict to lemmas with more than one coarse cluster
    # ------------------------------------------------------------------
    counts_df = pd.read_csv(_require(COUNTS_FILE))
    summary = pd.read_csv(_require(SUMMARY_FILE))
    dlexdb = pd.read_csv(_require(DLEXDB_FILE))[["lemma", "log_freq_dlexdb"]]
    survey = pd.read_csv(_require(SURVEY_FILE))[["lemma", "subjective_polysemy_mean"]]

    polysemous = summary[summary["n_clusters_coarse"] > 1]
    counts_df = counts_df[counts_df["lemma"].isin(set(polysemous["lemma"]))]

    predictors = (
        polysemous[["lemma", "n_clusters_coarse", "mean_entropy", "mean_jsd"]]
        .merge(dlexdb, on="lemma", how="left")
        .merge(survey, on="lemma", how="left")
    )

    # ------------------------------------------------------------------
    # Per-binning null baselines
    # ------------------------------------------------------------------
    per_lemma, corr_rows, summaries = [], [], {}
    for binning in BINNINGS:
        print(f"Running {BINNING_LABELS[binning]} ... ", end="", flush=True)
        result = run_binning(counts_df, binning, args.perms, args.seed)
        result = result.merge(predictors, on="lemma", how="left")

        if binning == "a_10_decades":
            # Guard: the reference binning must reproduce the submitted mean_jsd.
            max_diff = (result["jsd_obs"] - result["mean_jsd"]).abs().max()
            assert max_diff < 1e-9, (
                f"reference binning does not reproduce mean_jsd (max diff {max_diff:.2e})"
            )

        summaries[binning] = summarise(result)
        corr_rows += correlations(result, binning)
        corr_rows += partial_correlations(result, binning)
        per_lemma.append(result)
        print(f"{summaries[binning]['n_lemmas']} lemmas")

    columns = [
        "binning", "lemma", "n_clusters_coarse", "n_periods", "n_periods_supported",
        "n_transitions_used", "n_contexts_total", "jsd_obs", "jsd_null_mean",
        "jsd_null_sd", "jsd_excess", "jsd_z", "p_perm", "p_bh",
        "mean_entropy", "log_freq_dlexdb", "log_n_contexts", "subjective_polysemy_mean",
    ]
    out = pd.concat(per_lemma, ignore_index=True)[columns]
    out.to_csv(OUT_CSV, index=False)

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUT_CORRELATIONS, index=False)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    summary_tbl = pd.DataFrame(summaries).T
    summary_tbl.index = [BINNING_LABELS[b] for b in summary_tbl.index]
    print(f"\n=== Noise share and significance per binning ({args.perms} perms, seed {args.seed}) ===")
    print(summary_tbl.to_string(float_format=lambda v: f"{v:.3f}"))

    cells = corr_df.assign(
        cell=corr_df.apply(lambda r: f"{r['spearman_r']:+.3f} (p={r['p']:.3f})", axis=1)
    )

    def pivot(sub: pd.DataFrame, index: list[str]) -> str:
        # pivot_table sorts columns alphabetically (c2_ before c_); restore the
        # declared binning order so the printed table matches RESULTS.md.
        table = sub.pivot_table(index=index, columns="binning", values="cell", aggfunc="first")
        return table.reindex(columns=[b for b in BINNINGS if b in table.columns]).to_string()

    print("\n=== Spearman correlations per binning ===")
    print(pivot(cells[cells["method"] == "spearman"], ["test", "x", "y"]))

    print("\n=== Partial Spearman correlations per binning (y = jsd_excess) ===")
    print(pivot(cells[cells["method"] == "partial_spearman"], ["x", "control"]))

    print(f"\n=== 6. WSD accuracy on the 93-instance gold set, split at {ERA_CUT} (exploratory) ===")
    wsd_era = wsd_era_check()
    if wsd_era is not None:
        wsd_era.to_csv(OUT_WSD_ERA, index=False)
        print(wsd_era.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print(f"\nOutputs:")
    print(f"  {OUT_CSV.relative_to(BASE_DIR)}")
    print(f"  {OUT_CORRELATIONS.relative_to(BASE_DIR)}")
    if wsd_era is not None:
        print(f"  {OUT_WSD_ERA.relative_to(BASE_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
