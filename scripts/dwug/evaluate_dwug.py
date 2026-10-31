#!/usr/bin/env python3
"""
Step 3 of the DWUG DE evaluation: cluster agreement (ARI) and graded change (Spearman).

Input: data/processed/dwug/predictions/dwug_predictions.csv (scripts/dwug/predict_dwug.py)

A) ARI per word between our clusters and
   - DWUG DE 3.0.0 graph clusters (clusters/opt, usages with -1 excluded), 50 words
   - DWUG DE Sense 1.0.1 maj_2 labels (usages with -1 = no majority excluded), 24 words
   Random baseline: per word, uniform random labels with our number of clusters, 1000x.
B) JSD (base 2) of our cluster distributions P1 vs P2; permutation null as in
   analysis/robustness (period labels permuted, sizes fixed, 1000x, seed 42) -> jsd_excess.
   Spearman (p, 95% bootstrap CI over words) against SemEval-2020 graded gold (48 words)
   and DWUG 3.0.0 change_graded (50 words), all words and Zusatz 2 (>= 2 clusters).
   Baseline: number of our clusters as change predictor.
C) Robustness: P1 = context_normalized; without Seminar and vergönnen.
D) Rank deviations for the error analysis.

Outputs (analysis/dwug/, CSVs gitignored like analysis/robustness):
  per_word.csv, ari.csv, spearman.csv, error_analysis.csv, evaluate_log.txt

Usage:
    python scripts/dwug/evaluate_dwug.py [--perms 1000] [--boot 10000] [--seed 42]
"""

import argparse
import csv
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "robustness"))
from robustness_analyses import jsd_null_baseline  # noqa: E402

PRED = ROOT / "data/processed/dwug/predictions/dwug_predictions.csv"
DWUG = ROOT / "data/external/dwug_de/dwug_de"
DWUG_SENSE = ROOT / "data/external/dwug_de/dwug_de_sense"
SEMEVAL = ROOT / "data/external/semeval2020_ulscd_ger/extracted/semeval2020_ulscd_ger"
OUT = ROOT / "analysis/dwug"

KNOWN_DEFINITION_ERRORS = ["Seminar", "vergönnen"]  # analysis/dwug/review_notes.md


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def read_tsv(path: Path) -> pd.DataFrame:
    """TSV as strings; cells stripped (the DWUG Sense files have CRLF line endings)."""
    d = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
    d.columns = [c.strip() for c in d.columns]
    return d.apply(lambda col: col.str.strip())


# ---------------------------------------------------------------- gold

def load_graph_clusters() -> pd.DataFrame:
    frames = []
    for f in sorted((DWUG / "clusters/opt").glob("*.csv")):
        d = read_tsv(f)
        d["dwug_lemma"] = nfc(f.stem)
        frames.append(d)
    g = pd.concat(frames, ignore_index=True)
    g["cluster"] = g["cluster"].astype(int)
    return g[["dwug_lemma", "identifier", "cluster"]]


def load_sense_labels() -> pd.DataFrame:
    """maj_2 labels plus a flag for the per-word 'andere' sense (its id differs per word)."""
    frames = []
    for d in sorted((DWUG_SENSE / "labels").iterdir()):
        f = d / "maj_2" / "labels_senses.csv"
        if f.exists():
            x = read_tsv(f)
            senses = read_tsv(DWUG_SENSE / "data" / d.name / "senses.csv")
            andere = set(senses.loc[senses.description_sense == "andere", "identifier_sense"])
            x["dwug_lemma"] = nfc(d.name)
            x["is_andere"] = x["label"].isin(andere)
            frames.append(x)
    return pd.concat(frames, ignore_index=True)


def load_change_graded() -> pd.Series:
    s = read_tsv(DWUG / "stats/opt/stats_groupings.csv")
    s["lemma"] = s["lemma"].map(nfc)
    return s.set_index("lemma")["change_graded"].astype(float)


def load_semeval_graded() -> pd.Series:
    rows = [l.split("\t") for l in (SEMEVAL / "truth/graded.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    return pd.Series({nfc(w): float(v) for w, v in rows})


# ---------------------------------------------------------------- metrics

def random_ari(gold: np.ndarray, k: int, n: int, rng: np.random.Generator) -> float:
    return float(np.mean([adjusted_rand_score(gold, rng.integers(0, k, size=len(gold))) for _ in range(n)]))


def spearman_ci(x, y, boot: int, rng: np.random.Generator) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    r, p = spearmanr(x, y)
    rs = []
    for _ in range(boot):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) > 0 and np.ptp(y[i]) > 0:
            rs.append(spearmanr(x[i], y[i])[0])
    lo, hi = np.percentile(rs, [2.5, 97.5])
    return {"n": len(x), "rho": r, "p": p, "ci_lo": lo, "ci_hi": hi}


def per_word_change(pred: pd.DataFrame, perms: int, seed: int) -> pd.DataFrame:
    counts = (pred.groupby(["dwug_lemma", "grouping", "cluster_id"]).size()
              .rename("count").reset_index().rename(columns={"cluster_id": "predicted_sense_id"}))
    null = jsd_null_baseline(counts.rename(columns={"dwug_lemma": "lemma"}), bin_col="grouping",
                             perms=perms, seed=seed)
    # Cross-check the observed JSD against a direct computation.
    for lemma, g in counts.groupby("dwug_lemma"):
        pv = g.pivot_table(index="grouping", columns="predicted_sense_id", values="count", fill_value=0)
        p = pv.values / pv.values.sum(axis=1, keepdims=True)
        direct = jensenshannon(p[0], p[1], base=2) if p.shape[1] > 1 else 0.0
        obs = null.loc[null.lemma == lemma, "jsd_obs"].item()
        assert np.isclose(direct, obs), (lemma, direct, obs)
    return null.rename(columns={"lemma": "dwug_lemma"})


# ---------------------------------------------------------------- run

def evaluate(pred_variant: pd.DataFrame, gold_graph, gold_sense, gold_sense_no_andere, change_graded, semeval,
             exclude: list[str], args, tag: str, log) -> tuple[pd.DataFrame, list, list]:
    pred = pred_variant[~pred_variant.germanet_lemma.isin(exclude)]
    rng = np.random.default_rng(args.seed)

    # ---- A) ARI
    ari_rows = []
    for target, gold in [("graph_clusters", gold_graph), ("sense_labels_maj2", gold_sense),
                         ("sense_labels_maj2_no_andere", gold_sense_no_andere)]:
        m = pred.merge(gold, on=["dwug_lemma", "identifier"], how="inner")
        for lemma, g in m.groupby("dwug_lemma"):
            ari_rows.append({"variant": tag, "target": target, "dwug_lemma": lemma,
                             "n_usages": len(g), "n_clusters_ours": g.n_clusters.iloc[0],
                             "n_gold_classes": g.gold.nunique(),
                             "ari": adjusted_rand_score(g.gold, g.cluster_id),
                             "ari_random": random_ari(g.gold.values, int(g.n_clusters.iloc[0]), args.perms, rng)})
    ari = pd.DataFrame(ari_rows)

    # ---- B) change
    pw = per_word_change(pred, args.perms, args.seed)
    info = pred.groupby("dwug_lemma").agg(germanet_lemma=("germanet_lemma", "first"),
                                          status=("status", "first"), n_clusters=("n_clusters", "first"),
                                          n_p1=("grouping", lambda s: (s == 1).sum()),
                                          n_p2=("grouping", lambda s: (s == 2).sum()))
    pw = pw.merge(info, left_on="dwug_lemma", right_index=True)
    pw["change_graded"] = pw.dwug_lemma.map(change_graded)
    pw["semeval_graded"] = pw.dwug_lemma.map(semeval)
    pw["variant"] = tag

    sp_rows = []
    for gold_name in ["semeval_graded", "change_graded"]:
        for subset in ["all", "zusatz2"]:
            d = pw.dropna(subset=[gold_name])
            if subset == "zusatz2":
                d = d[d.n_clusters >= 2]
            for pred_name in ["jsd_obs", "jsd_excess", "n_clusters"]:
                r = spearman_ci(d[pred_name], d[gold_name], args.boot, np.random.default_rng(args.seed))
                sp_rows.append({"variant": tag, "gold": gold_name, "subset": subset, "predictor": pred_name, **r})
    sp = pd.DataFrame(sp_rows)

    log(f"\n=== {tag} (excluded: {exclude or '-'}) ===")
    for target, g in ari.groupby("target"):
        z2 = g[g.n_clusters_ours >= 2]
        log(f"ARI {target}: all n={len(g)} mean={g.ari.mean():.3f} (random {g.ari_random.mean():.3f}); "
            f">=2 clusters n={len(z2)} mean={z2.ari.mean():.3f} (random {z2.ari_random.mean():.3f})")
    log(sp.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return pw, ari, sp


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--perms", type=int, default=1000)
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    pred = pd.read_csv(PRED, dtype={"identifier": str})
    pred["dwug_lemma"] = pred["dwug_lemma"].map(nfc)
    gg = load_graph_clusters()
    gold_graph = gg[gg.cluster != -1].rename(columns={"cluster": "gold"})
    log(f"Graph clusters: {len(gg)} usages, {int((gg.cluster == -1).sum())} with -1 excluded")
    sl = load_sense_labels()
    sl = sl[sl.label != "-1"].rename(columns={"label": "gold"})
    gold_sense = sl[["dwug_lemma", "identifier", "gold"]]
    gold_sense_no_andere = sl.loc[~sl.is_andere, ["dwug_lemma", "identifier", "gold"]]
    log(f"Sense labels maj_2: {len(sl)} usages after excluding -1 (no majority), "
        f"{int(sl.is_andere.sum())} of them 'andere' (kept as a class; check without them)")
    change_graded = load_change_graded()
    semeval = load_semeval_graded()
    log(f"Gold: change_graded {len(change_graded)} words, SemEval graded {len(semeval)} words")

    runs = [("main", pred[pred.variant == "context"], []),
            ("p1_normalized", pd.concat([pred[(pred.variant == "context") & (pred.grouping == 2)],
                                         pred[pred.variant == "normalized"]]), []),
            ("without_seminar_vergoennen", pred[pred.variant == "context"], KNOWN_DEFINITION_ERRORS)]
    all_pw, all_ari, all_sp = [], [], []
    for tag, d, excl in runs:
        assert d.groupby(["dwug_lemma", "identifier"]).size().max() == 1
        pw, ari, sp = evaluate(d, gold_graph, gold_sense, gold_sense_no_andere, change_graded, semeval,
                               excl, args, tag, log)
        all_pw.append(pw); all_ari.append(ari); all_sp.append(sp)

    pw = pd.concat(all_pw); ari = pd.concat(all_ari); sp = pd.concat(all_sp)
    pw.to_csv(OUT / "per_word.csv", index=False)
    ari.to_csv(OUT / "ari.csv", index=False)
    sp.to_csv(OUT / "spearman.csv", index=False)

    # ---- D) rank deviation (main run, our JSD vs SemEval gold)
    m = pw[(pw.variant == "main")].dropna(subset=["semeval_graded"]).copy()
    m["rank_ours"] = m.jsd_obs.rank()
    m["rank_gold"] = m.semeval_graded.rank()
    m["rank_diff"] = m.rank_ours - m.rank_gold
    m = m.merge(ari[(ari.variant == "main") & (ari.target == "graph_clusters")][["dwug_lemma", "ari", "n_gold_classes"]],
                on="dwug_lemma", how="left")
    m.reindex(m.rank_diff.abs().sort_values(ascending=False).index).to_csv(OUT / "error_analysis.csv", index=False)
    (OUT / "evaluate_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
