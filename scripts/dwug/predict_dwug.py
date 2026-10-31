#!/usr/bin/env python3
"""
Step 2 of the DWUG DE evaluation: sense and cluster predictions for all DWUG usages.

Same method as in the paper:
- multilingual-e5-large, "query: " for contexts, "passage: " for sense texts
- sense representation "weighted" = 0.5 * llm_definition + 0.5 * mean(5 llm_examples)
- cosine argmax over the GermaNet senses of the lemma
- fine sense -> reviewed coarse cluster

DWUG specifics (decisions in analysis/dwug/INVENTORY.md):
- all usages of both periods, no base-form filter
- whole sentence as context; sentences longer than 60 tokens are cut to a
  60-token window around the target (position from indexes_target_token)
- P1 is predicted twice: `context` (main) and `context_normalized` (check).
  P2 has no normalized text and always uses `context`.
- words with one GermaNet sense get one cluster for all usages (no model call)

Only DWUG usage text is read here. No DWUG clusters, labels or gold values.

The output has no sentence text (DWUG is CC BY-ND 4.0); the window can be rebuilt
from the identifier and window_start/window_end.

Usage:
    python scripts/dwug/predict_dwug.py
"""

import argparse
import csv
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np  # noqa: F401  (load numpy before torch, avoids a libomp clash on macOS)
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.dwds.util.context_utils import clean_context  # noqa: E402
from src.diachronic.io import load_senses  # noqa: E402
from src.wsd import Embedder, SimilarityWSD  # noqa: E402
from src.wsd.embedding_strategies import embed_with_representation  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DWUG_DATA = ROOT / "data/external/dwug_de/dwug_de/data"
DWUG_DIR = ROOT / "data/processed/dwug"
LEMMA_MAP = DWUG_DIR / "lemma_map.csv"
SENSES_DWUG = DWUG_DIR / "senses_dwug.jsonl"
CLUSTERS_DWUG = DWUG_DIR / "sense_clusters_dwug_reviewed.json"
SENSES_502 = ROOT / "data/processed/germanet/senses_combined.jsonl"
CLUSTERS_502 = ROOT / "data/processed/germanet/sense_clusters_reviewed.json"
OUT_DIR = DWUG_DIR / "predictions"

MODEL = "intfloat/multilingual-e5-large"
MAX_TOKENS = 60


def read_uses(dwug_lemma: str) -> pd.DataFrame:
    path = DWUG_DATA / dwug_lemma / "uses.csv"
    return pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)


def window(n: int, target: int, size: int = MAX_TOKENS) -> tuple[int, int]:
    """Token window [start, end) of at most `size` tokens, target roughly centred."""
    if n <= size:
        return 0, n
    start = max(0, min(target - size // 2, n - size))
    return start, start + size


def build_input_context(context: str, target_span: str) -> dict:
    """Window over whitespace tokens of the original text (as count_words in the paper)."""
    tokens = list(re.finditer(r"\S+", context))
    char_start = int(target_span.split(":")[0])
    target = next(
        (i for i, m in enumerate(tokens) if m.start() <= char_start < m.end()),
        next((i for i, m in enumerate(tokens) if m.start() >= char_start), len(tokens) - 1),
    )
    start, end = window(len(tokens), target)
    text = " ".join(m.group() for m in tokens[start:end])
    return {"n_tokens": len(tokens), "target_token": target, "window_start": start,
            "window_end": end, "text": clean_context(text)}


def build_input_normalized(normalized: str, target_index: str) -> dict:
    """Window over the tokens of context_normalized (space-separated, punctuation is a token)."""
    tokens = normalized.split(" ")
    target = int(target_index)
    start, end = window(len(tokens), target)
    return {"n_tokens": len(tokens), "target_token": target, "window_start": start,
            "window_end": end, "text": clean_context(" ".join(tokens[start:end]))}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=OUT_DIR / "dwug_predictions.csv")
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    lemma_map = pd.read_csv(LEMMA_MAP)
    clusters = {**json.load(open(CLUSTERS_502, encoding="utf-8")),
                **json.load(open(CLUSTERS_DWUG, encoding="utf-8"))}

    multi = lemma_map[lemma_map["status"] != "single_sense"]
    senses = pd.concat([
        load_senses(SENSES_DWUG, target_lemmas=multi.loc[multi.status == "prepare", "germanet_lemma"].tolist()),
        load_senses(SENSES_502, target_lemmas=multi.loc[multi.status == "existing_502", "germanet_lemma"].tolist()),
    ], ignore_index=True)

    # Sense inventory must match the reviewed cluster mapping exactly.
    for gl in multi["germanet_lemma"]:
        got = set(senses.loc[senses.lemma == gl, "synset_id"])
        if got != set(clusters[gl]["mapping"]):
            raise ValueError(f"{gl}: senses {sorted(got)} != cluster mapping {sorted(clusters[gl]['mapping'])}")

    # Build all inputs.
    rows = []
    for _, lm in lemma_map.iterrows():
        uses = read_uses(lm.dwug_lemma)
        for _, u in uses.iterrows():
            base = {"dwug_lemma": lm.dwug_lemma, "germanet_lemma": lm.germanet_lemma,
                    "status": lm.status, "identifier": u.identifier,
                    "grouping": int(u.grouping), "date": u.date}
            ctx = build_input_context(u.context, u.indexes_target_token)
            rows.append({**base, "variant": "context", "input_source": "context", **ctx})
            if u.grouping == "1":
                if u.context_normalized:
                    norm = build_input_normalized(u.context_normalized, u.indexes_target_token_tokenized)
                    rows.append({**base, "variant": "normalized", "input_source": "context_normalized", **norm})
                else:  # empty in DWUG for a few P1 usages -> original text
                    rows.append({**base, "variant": "normalized", "input_source": "context_fallback", **ctx})
    df = pd.DataFrame(rows)
    df["truncated"] = df["n_tokens"] > MAX_TOKENS
    logger.info(f"{len(df)} inputs: {df.variant.value_counts().to_dict()}, "
                f"truncated {df.truncated.mean():.1%}, fallback {(df.input_source == 'context_fallback').sum()}")

    # Model predictions for lemmas with >= 2 GermaNet senses.
    embedder = Embedder(model_name=MODEL, normalize=True)
    sense_emb = embed_with_representation(senses, embedder, representation="weighted",
                                          show_progress=True, passage_prefix="passage: ")
    wsd = SimilarityWSD(sense_embeddings=sense_emb, sense_ids=senses["synset_id"].tolist(),
                        sense_lemmas=senses["lemma"].tolist())

    is_multi = df["status"] != "single_sense"
    texts = df.loc[is_multi, "text"].drop_duplicates().tolist()
    logger.info(f"Encoding {len(texts)} unique contexts ...")
    emb = embedder.encode([f"query: {t}" for t in texts], show_progress=True, batch_size=args.batch_size)
    text_idx = {t: i for i, t in enumerate(texts)}

    sub = df[is_multi]
    preds = wsd.predict_batch(context_embeddings=emb[[text_idx[t] for t in sub["text"]]],
                              lemmas=sub["germanet_lemma"].tolist())
    df.loc[is_multi, "predicted_sense_id"] = [p.predicted_sense_id for p in preds]
    df.loc[is_multi, "confidence"] = [p.confidence for p in preds]

    # Fine sense -> reviewed cluster; single-sense words -> one cluster.
    def to_cluster(r):
        if r.status == "single_sense":
            return f"{r.germanet_lemma}_single"
        return clusters[r.germanet_lemma]["mapping"][r.predicted_sense_id]
    df["cluster_id"] = df.apply(to_cluster, axis=1)
    n_cl = {gl: len(c["clusters"]) for gl, c in clusters.items()}
    df["n_clusters"] = [1 if s == "single_sense" else n_cl[gl] for s, gl in zip(df.status, df.germanet_lemma)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cols = ["dwug_lemma", "germanet_lemma", "status", "identifier", "grouping", "date", "variant",
            "input_source", "n_tokens", "target_token", "window_start", "window_end", "truncated",
            "predicted_sense_id", "confidence", "cluster_id", "n_clusters"]
    df[cols].to_csv(args.output, index=False)

    meta = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": MODEL,
            "representation": "weighted", "max_tokens": MAX_TOKENS, "n_rows": len(df),
            "n_usages": int((df.variant == "context").sum()), "n_lemmas": int(df.dwug_lemma.nunique()),
            "n_unique_contexts_encoded": len(texts), "device": str(embedder.model.device),
            "dwug_version": "DWUG DE 3.0.0"}
    args.output.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"Saved {len(df)} rows -> {args.output}")


if __name__ == "__main__":
    main()
