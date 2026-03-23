#!/usr/bin/env python3
"""
Predict senses for DWDS corpus contexts using embedding similarity.

Processes one lemma at a time to avoid OOM on large context files.
Results are written incrementally to disk; use --resume to continue
after a crash without reprocessing completed lemmas.

Usage:
    python scripts/diachronic/predict_dwds_senses.py
    python scripts/diachronic/predict_dwds_senses.py --dwds-file path/to/contexts.jsonl
    python scripts/diachronic/predict_dwds_senses.py --resume
"""

import argparse
import gc
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.diachronic import load_dwds_contexts, load_senses, predict_senses_for_lemma
from src.diachronic.predictor import is_e5_model
from src.wsd import Embedder, SimilarityWSD
from src.wsd.embedding_strategies import embed_with_representation

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
DEFAULT_DWDS_FILE = DATA_DIR / "processed" / "dwds" / "pilot_contexts.jsonl"
DEFAULT_SENSES_FILE = DATA_DIR / "processed" / "germanet" / "senses_pilot.jsonl"
OUTPUT_DIR = DATA_DIR / "processed" / "diachronic" / "predictions"
DEFAULT_MODEL = "intfloat/multilingual-e5-large"
DEFAULT_REPRESENTATION = "weighted"

OUTPUT_COLS = [
    "lemma", "sentence", "year", "decade", "genre",
    "predicted_sense_id", "confidence", "sense_class", "sense_paraphrase",
    "date", "bibl", "corpus",
]


def load_done_lemmas(output_path: Path) -> set:
    if not output_path.exists():
        return set()
    try:
        df = pd.read_csv(output_path, usecols=["lemma"])
        return set(df["lemma"].unique())
    except Exception:
        return set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict senses for DWDS corpus contexts")
    parser.add_argument("--dwds-file", type=Path, default=DEFAULT_DWDS_FILE)
    parser.add_argument("--senses-file", type=Path, default=DEFAULT_SENSES_FILE)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Skip lemmas already present in output file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = args.output or OUTPUT_DIR / "dwds_predictions.csv"

    # Load contexts and senses
    contexts_df = load_dwds_contexts(args.dwds_file)
    target_lemmas = sorted(contexts_df["lemma"].unique().tolist())
    senses_df = load_senses(args.senses_file, target_lemmas=target_lemmas)

    words_with_senses = set(senses_df["lemma"].unique())
    words_without_senses = set(target_lemmas) - words_with_senses
    if words_without_senses:
        logger.warning(f"{len(words_without_senses)} words have no senses — skipping:")
        logger.warning(f"  {sorted(words_without_senses)}")
    lemmas_to_process = [l for l in target_lemmas if l in words_with_senses]

    # Resume: skip already-done lemmas
    done_lemmas = set()
    if args.resume:
        done_lemmas = load_done_lemmas(output_path)
        if done_lemmas:
            logger.info(f"Resuming: {len(done_lemmas)} lemmas already done, skipping.")
    lemmas_to_process = [l for l in lemmas_to_process if l not in done_lemmas]

    if not lemmas_to_process:
        logger.info("Nothing to do.")
        return

    # Load model once
    logger.info(f"Loading model: {args.model}")
    embedder = Embedder(model_name=args.model, normalize=True)
    use_e5 = is_e5_model(args.model)
    if use_e5:
        logger.info("  (E5 model: applying query/passage prefixes)")

    # Encode all senses once (596 rows — tiny)
    logger.info(f"Encoding senses using representation: '{DEFAULT_REPRESENTATION}'...")
    sense_embeddings = embed_with_representation(
        senses_df, embedder,
        representation=DEFAULT_REPRESENTATION,
        show_progress=True,
        passage_prefix="passage: " if use_e5 else "",
    )
    wsd = SimilarityWSD(
        sense_embeddings=sense_embeddings,
        sense_ids=senses_df["synset_id"].tolist(),
        sense_lemmas=senses_df["lemma"].tolist(),
    )
    del sense_embeddings
    gc.collect()

    # Write header if starting fresh
    write_header = not args.resume or not output_path.exists()

    total_done = len(done_lemmas)
    total = total_done + len(lemmas_to_process)

    with output_path.open("a" if args.resume else "w", encoding="utf-8") as out_f:
        for i, lemma in enumerate(lemmas_to_process, start=total_done + 1):
            lemma_ctx = contexts_df[contexts_df["lemma"] == lemma].copy()
            logger.info(f"[{i}/{total}] {lemma}: {len(lemma_ctx)} contexts")

            try:
                results = predict_senses_for_lemma(
                    lemma=lemma,
                    contexts_df=lemma_ctx,
                    senses_df=senses_df,
                    embedder=embedder,
                    wsd=wsd,
                    use_e5=use_e5,
                )
                cols = [c for c in OUTPUT_COLS if c in results.columns]
                results[cols].to_csv(out_f, index=False, header=write_header)
                write_header = False  # only write header once
                logger.info(f"  → {results['predicted_sense_id'].notna().sum()} predicted, "
                             f"conf={results['confidence'].mean():.3f}")
            except Exception as e:
                logger.error(f"  ERROR on {lemma}: {e}")

            del lemma_ctx
            gc.collect()

    del embedder, wsd
    gc.collect()
    logger.info(f"Done. Results saved to: {output_path}")


if __name__ == "__main__":
    main()