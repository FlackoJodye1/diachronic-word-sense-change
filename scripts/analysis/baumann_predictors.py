#!/usr/bin/env python3
"""
Per-lemma Baumann-style predictors for the §5 application analysis.

Predictors computed:
  - entropy_slope, entropy_slope_p, n_decades_used
        Linear regression entropy ~ decade over (lemma, decade) cells that
        meet n_contexts >= MIN_CONTEXTS. Lemmas with < 3 supported decades
        get NaN. Baumann's ∂_t D analogue: positive slope = diversifying.
  - log_frequency
        log(total DWDS Kernkorpus contexts) per lemma.
  - discriminability_desc
        1 − mean pairwise cosine similarity among the lemma's sense
        embeddings (E5-large `weighted` representation, the same one used
        by `scripts/atemporal/generate_predictions.py`). Higher = more
        discriminable.
  - concreteness
        From Charbonnier & Wartena 2019 (sheet 'Concreteness deveL' of
        `data/raw/target_words/target_words_clean.xlsx`, column conc_score).

Embeddings are computed on the fly the first time the script runs and
cached at `data/processed/embeddings/senses_pilot_weighted_e5_large.npy`
(+ a metadata CSV). Subsequent runs reuse the cache.

Inputs (read-only):
  data/processed/diachronic/metrics/pilot_v3_coarse_entropy.csv
  data/processed/diachronic/aggregated/pilot_v3_coarse_stats.csv  (n_contexts)
  data/processed/dwds/pilot_contexts_v3.jsonl
  data/processed/germanet/senses_pilot.jsonl
  data/raw/target_words/target_words_clean.xlsx                    (concreteness)

Outputs:
  data/processed/analysis/baumann_predictors.csv
  data/processed/embeddings/senses_pilot_weighted_e5_large.npy     (cached)
  data/processed/embeddings/senses_pilot_weighted_e5_large.csv     (metadata)
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

ENTROPY_FILE  = BASE_DIR / "data/processed/diachronic/metrics/pilot_v3_coarse_entropy.csv"
STATS_FILE    = BASE_DIR / "data/processed/diachronic/aggregated/pilot_v3_coarse_stats.csv"
CONTEXTS_FILE = BASE_DIR / "data/processed/dwds/combined_contexts_v3.jsonl"
SENSES_FILE   = BASE_DIR / "data/processed/germanet/senses_combined.jsonl"
CONCR_FILE    = BASE_DIR / "data/raw/target_words/target_words_clean.xlsx"
CONCR_SHEET   = "Concreteness deveL"

EMB_FILE      = BASE_DIR / "data/processed/embeddings/senses_combined_weighted_e5_large.npy"
EMB_META_FILE = BASE_DIR / "data/processed/embeddings/senses_combined_weighted_e5_large.csv"

OUTPUT_FILE   = BASE_DIR / "data/processed/analysis/baumann_predictors.csv"

MIN_CONTEXTS = 10  # same support filter as per_lemma_summary.py


def _require(path: Path) -> Path:
    if not path.exists():
        sys.exit(f"ERROR: required input not found: {path.relative_to(BASE_DIR)}\n"
                 f"  Directory listing: {sorted(p.name for p in path.parent.iterdir())[:20] if path.parent.exists() else '(parent missing)'}")
    return path


# ----------------------------------------------------------------------
# Schema check (printed before any computation)
# ----------------------------------------------------------------------

def _schema_check() -> None:
    print("Schema check")
    print("============")

    ent = pd.read_csv(_require(ENTROPY_FILE))
    print(f"\n[1] {ENTROPY_FILE.relative_to(BASE_DIR)}")
    print(f"    rows={len(ent)}, lemmas={ent['lemma'].nunique()}, columns={list(ent.columns)}")
    print(ent.head(3).to_string(index=False))

    stats = pd.read_csv(_require(STATS_FILE))
    print(f"\n[2] {STATS_FILE.relative_to(BASE_DIR)}")
    print(f"    rows={len(stats)}, lemmas={stats['lemma'].nunique()}, columns={list(stats.columns)}")
    print(stats.head(3).to_string(index=False))

    n_contexts_seen = 0
    keys_first = None
    with _require(CONTEXTS_FILE).open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if keys_first is None:
                keys_first = list(r.keys())
            n_contexts_seen += 1
    print(f"\n[3] {CONTEXTS_FILE.relative_to(BASE_DIR)}")
    print(f"    rows={n_contexts_seen}, first-row keys={keys_first}")

    n_senses = 0
    lemmas_seen = set()
    with _require(SENSES_FILE).open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            lemmas_seen.add(r["lemma"])
            n_senses += 1
    print(f"\n[4] {SENSES_FILE.relative_to(BASE_DIR)}")
    print(f"    senses={n_senses}, lemmas={len(lemmas_seen)}")

    concr = pd.read_excel(_require(CONCR_FILE), sheet_name=CONCR_SHEET)
    print(f"\n[5] {CONCR_FILE.relative_to(BASE_DIR)} :: '{CONCR_SHEET}'")
    print(f"    rows={len(concr)}, columns={list(concr.columns)}")
    print(concr.head(3).to_string(index=False))
    print()


# ----------------------------------------------------------------------
# 1. Entropy slope per lemma
# ----------------------------------------------------------------------

def _entropy_slopes() -> pd.DataFrame:
    ent   = pd.read_csv(ENTROPY_FILE)[["lemma", "decade", "entropy"]]
    stats = pd.read_csv(STATS_FILE)[["lemma", "decade", "n_contexts"]]

    df = ent.merge(stats, on=["lemma", "decade"], how="left")
    df = df[df["n_contexts"] >= MIN_CONTEXTS]

    rows = []
    for lemma, sub in df.groupby("lemma"):
        n = len(sub)
        if n < 3:
            rows.append({"lemma": lemma, "entropy_slope": np.nan,
                         "entropy_slope_p": np.nan, "n_decades_used": int(n)})
            continue
        res = linregress(sub["decade"].astype(float), sub["entropy"].astype(float))
        rows.append({"lemma": lemma,
                     "entropy_slope":   float(res.slope),
                     "entropy_slope_p": float(res.pvalue),
                     "n_decades_used":  int(n)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 2. log_frequency per lemma
# ----------------------------------------------------------------------

def _log_frequency() -> pd.DataFrame:
    counts: dict[str, int] = {}
    with CONTEXTS_FILE.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            l = r["lemma"]
            counts[l] = counts.get(l, 0) + 1
    df = pd.DataFrame({"lemma": list(counts), "n_contexts_total": list(counts.values())})
    df["log_frequency"] = np.log(df["n_contexts_total"].astype(float))
    return df[["lemma", "log_frequency"]]


# ----------------------------------------------------------------------
# 3. Sense embeddings → discriminability per lemma
# ----------------------------------------------------------------------

def _compute_or_load_embeddings(force: bool = False) -> tuple[np.ndarray, pd.DataFrame]:
    if EMB_FILE.exists() and EMB_META_FILE.exists() and not force:
        print(f"  Loading cached embeddings from {EMB_FILE.relative_to(BASE_DIR)}")
        return np.load(EMB_FILE), pd.read_csv(EMB_META_FILE)

    # Lazy imports — only needed when actually computing.
    from src.diachronic.io import load_senses
    from src.wsd.embedder import Embedder
    from src.wsd.embedding_strategies import embed_with_representation

    print(f"  Computing E5-large 'weighted' embeddings for senses_pilot.jsonl …")
    senses = load_senses(SENSES_FILE)
    embedder = Embedder("multilingual-e5-large")
    emb = embed_with_representation(senses, embedder, "weighted",
                                    show_progress=False, passage_prefix="passage: ")

    # L2-normalise so cosine = dot product (matches WSD pipeline).
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.where(norms == 0, 1, norms)

    EMB_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMB_FILE, emb)
    meta = senses[["lemma", "synset_id"]].copy()
    meta.to_csv(EMB_META_FILE, index=False)
    print(f"  Cached embeddings: shape={emb.shape} → {EMB_FILE.relative_to(BASE_DIR)}")
    return emb, meta


def _discriminability(emb: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    """1 - mean pairwise cosine sim per lemma. Single-sense lemmas → NaN."""
    rows = []
    for lemma, sub in meta.reset_index().groupby("lemma"):
        idxs = sub["index"].values
        if len(idxs) < 2:
            rows.append({"lemma": lemma, "discriminability_desc": np.nan})
            continue
        vecs = emb[idxs]  # (n_senses, dim)
        # All pairwise cosines (vectors are L2-normalised, so dot = cosine).
        sims = vecs @ vecs.T
        # Upper triangle excluding diagonal.
        iu = np.triu_indices_from(sims, k=1)
        mean_sim = float(sims[iu].mean())
        rows.append({"lemma": lemma, "discriminability_desc": 1.0 - mean_sim})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 4. Concreteness from Charbonnier & Wartena
# ----------------------------------------------------------------------

def _concreteness() -> pd.DataFrame:
    df = pd.read_excel(CONCR_FILE, sheet_name=CONCR_SHEET)
    # NOTE: the column names in this sheet are misleading.
    # 'Concreteness' is the lemma string; 'num_senses' is actually the
    # 1–7 concreteness *rating* (Elefant 6.88, Geist 2.44); 'conc_score'
    # is actually the integer *sense count* (schwarz 8, Apfel 2). We
    # use `num_senses` as the concreteness rating and coerce to numeric
    # (the column is dtype=object due to occasional missing/text cells).
    out = df[["Concreteness", "num_senses"]].rename(
        columns={"Concreteness": "lemma", "num_senses": "concreteness"}
    )
    out["concreteness"] = pd.to_numeric(out["concreteness"], errors="coerce")
    return out


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def _summary_table(df: pd.DataFrame) -> str:
    cols = [c for c in df.columns if c != "lemma"]
    rows = []
    for c in cols:
        col = pd.to_numeric(df[c], errors="coerce")
        n = int(col.notna().sum())
        rows.append({
            "column":   c,
            "n_nonnan": n,
            "mean":     float(col.mean()) if n else float("nan"),
            "std":      float(col.std())  if n > 1 else float("nan"),
            "min":      float(col.min())  if n else float("nan"),
            "max":      float(col.max())  if n else float("nan"),
        })
    out = pd.DataFrame(rows)
    out["mean"] = out["mean"].map(lambda v: f"{v:.4f}" if pd.notna(v) else "nan")
    out["std"]  = out["std"].map( lambda v: f"{v:.4f}" if pd.notna(v) else "nan")
    out["min"]  = out["min"].map( lambda v: f"{v:.4f}" if pd.notna(v) else "nan")
    out["max"]  = out["max"].map( lambda v: f"{v:.4f}" if pd.notna(v) else "nan")
    return out.to_string(index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recompute-embeddings", action="store_true",
                    help="Force re-computation of sense embeddings (ignores cache).")
    args = ap.parse_args()

    _schema_check()

    print("Computing predictors")
    print("====================")

    print("[a] entropy_slope from coarse entropy + stats (support filter n_contexts >= 10)")
    slopes = _entropy_slopes()
    print(f"    -> {len(slopes)} lemmas")

    print("[b] log_frequency from DWDS contexts")
    freq = _log_frequency()
    print(f"    -> {len(freq)} lemmas")

    print("[c] discriminability_desc from E5-large weighted sense embeddings")
    emb, emb_meta = _compute_or_load_embeddings(force=args.recompute_embeddings)
    disc = _discriminability(emb, emb_meta)
    print(f"    -> {len(disc)} lemmas")

    print("[d] concreteness from Charbonnier & Wartena 2019")
    concr = _concreteness()
    print(f"    -> {len(concr)} lemmas in source")

    # Assemble final table. Base set = lemmas in the entropy CSV (the v3 pilot).
    base_lemmas = sorted(pd.read_csv(ENTROPY_FILE)["lemma"].unique())
    out = pd.DataFrame({"lemma": base_lemmas})
    out = (out.merge(slopes, on="lemma", how="left")
              .merge(freq,   on="lemma", how="left")
              .merge(disc,   on="lemma", how="left")
              .merge(concr,  on="lemma", how="left"))

    out = out[[
        "lemma", "entropy_slope", "entropy_slope_p", "n_decades_used",
        "log_frequency", "discriminability_desc", "concreteness",
    ]]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_FILE, index=False)

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    needed = ["entropy_slope", "log_frequency", "discriminability_desc", "concreteness"]
    complete_mask = out[needed].notna().all(axis=1)
    n_complete = int(complete_mask.sum())

    print()
    print(f"Wrote: {OUTPUT_FILE.relative_to(BASE_DIR)}")
    print(f"Rows : {len(out)} lemmas (v3 pilot base set)")
    print()
    print("Per-column summary:")
    print(_summary_table(out))
    print()
    print(f"Lemmas with COMPLETE data (no NaN across the four predictors): {n_complete} / {len(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())