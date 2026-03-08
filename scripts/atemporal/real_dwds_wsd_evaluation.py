#!/usr/bin/env python3
"""
German WSD Evaluation using DWDS corpus sentences and enhanced GermaNet embeddings.

This script evaluates WSD performance using real corpus data from DWDS,
combined with sense definitions from GermaNet. Since we don't have human-annotated
sense labels for DWDS sentences, it uses the Most Frequent Sense (MFS) baseline
as a reference point.

Note: This provides indicative results, not true gold-standard evaluation.
For proper evaluation, use compare_embedding_methods.py with human-annotated data.
"""

import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.wsd import Embedder, SimilarityWSD

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path("data")
DWDS_CONTEXTS_DIR = DATA_DIR / "dwds_contexts"
GERMANET_DIR = DATA_DIR / "processed" / "germanet"
OUTPUT_DIR = DATA_DIR / "processed" / "dwds"

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MIN_SENTENCES_PER_WORD = 20
MAX_SENTENCES_PER_WORD = 100


def load_dwds_corpus() -> list[dict[str, Any]]:
    """Load real DWDS corpus sentences."""
    logger.info("Loading DWDS corpus data...")
    progress_file = DWDS_CONTEXTS_DIR / "fetch_progress_25_words.jsonl"

    contexts = []
    with open(progress_file, "r", encoding="utf-8") as f:
        for line in f:
            contexts.append(json.loads(line.strip()))

    logger.info(f"  Loaded {len(contexts):,} corpus sentences")
    return contexts


def load_germanet_senses() -> pd.DataFrame:
    """Load enhanced GermaNet sense data."""
    logger.info("Loading GermaNet senses...")
    df = pd.read_csv(GERMANET_DIR / "wsd_senses_enhanced.csv")
    logger.info(f"  Loaded {len(df)} senses for {df['lemma'].nunique()} words")
    return df


def find_evaluation_candidates(
    contexts: list[dict[str, Any]],
    germanet_df: pd.DataFrame,
    min_sentences: int = MIN_SENTENCES_PER_WORD,
) -> dict[str, dict[str, Any]]:
    """Find words suitable for evaluation (polysemous + have corpus data)."""
    logger.info("Finding evaluation candidates...")

    # Count corpus sentences per word
    corpus_counts = Counter(ctx["lemma"] for ctx in contexts)

    # Find polysemous words in GermaNet
    sense_counts = germanet_df.groupby("lemma").size()
    polysemous = set(sense_counts[sense_counts >= 2].index)

    candidates = {}
    for word in polysemous:
        if word in corpus_counts and corpus_counts[word] >= min_sentences:
            word_senses = germanet_df[germanet_df["lemma"] == word]
            candidates[word] = {
                "num_senses": len(word_senses),
                "corpus_sentences": corpus_counts[word],
            }

    logger.info(f"  Found {len(candidates)} evaluation candidates:")
    for word, info in candidates.items():
        logger.info(f"    {word}: {info['num_senses']} senses, {info['corpus_sentences']} sentences")

    return candidates


def prepare_test_data(
    contexts: list[dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
    max_sentences: int = MAX_SENTENCES_PER_WORD,
) -> pd.DataFrame:
    """Prepare test data from DWDS corpus sentences."""
    logger.info("Preparing test data...")

    test_records = []
    for word in candidates:
        word_contexts = [ctx for ctx in contexts if ctx["lemma"] == word]

        # Sample if too many
        if len(word_contexts) > max_sentences:
            word_contexts = random.sample(word_contexts, max_sentences)

        for i, ctx in enumerate(word_contexts):
            test_records.append({
                "context_id": f"{word}_{i}",
                "lemma": word,
                "sentence": ctx["sentence"],
                "year": ctx.get("year"),
                "genre": ctx.get("genre", ""),
            })

    df = pd.DataFrame(test_records)
    logger.info(f"  Created {len(df)} test cases")
    return df


def evaluate_mfs_baseline(
    germanet_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, float]:
    """Evaluate Most Frequent Sense baseline.

    Since DWDS sentences don't have gold labels, we compute:
    - Random baseline: 1 / num_senses per word
    - MFS would predict the most frequent sense (frequency=1) for all instances
    """
    word_num_senses = germanet_df.groupby("lemma")["synset_id"].nunique()

    # Random baseline: average of 1/num_senses
    random_scores = []
    for lemma in test_df["lemma"]:
        if lemma in word_num_senses:
            random_scores.append(1.0 / word_num_senses[lemma])
    random_baseline = np.mean(random_scores) if random_scores else 0.0

    return {
        "random_baseline": random_baseline,
        "num_test_cases": len(test_df),
    }


def run_similarity_evaluation(
    germanet_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Run similarity-based WSD evaluation.

    Note: Without gold labels, we report:
    - Distribution of prediction confidences
    - Agreement with MFS (does the model predict sense_frequency=1?)
    - Confidence statistics
    """
    logger.info(f"Running similarity evaluation with {model_name}...")

    embedder = Embedder(model_name=model_name, normalize=True)

    logger.info("  Encoding sense definitions...")
    sense_texts = germanet_df["text_for_embedding"].fillna("").tolist()
    sense_embeddings = embedder.encode(sense_texts, show_progress=True)

    logger.info("  Encoding context sentences...")
    context_texts = test_df["sentence"].fillna("").tolist()
    context_embeddings = embedder.encode(context_texts, show_progress=True)

    # Create WSD predictor
    wsd = SimilarityWSD.from_dataframe(
        sense_df=germanet_df,
        embeddings=sense_embeddings,
        sense_id_col="synset_id",
        lemma_col="lemma",
    )

    # Get predictions
    predictions = wsd.predict_batch(
        context_embeddings=context_embeddings,
        lemmas=test_df["lemma"].tolist(),
        context_ids=test_df["context_id"].tolist(),
    )

    # Analyze predictions
    # Build sense_id -> frequency lookup
    sense_freq_map = dict(zip(germanet_df["synset_id"], germanet_df["sense_frequency"]))

    confidences = []
    mfs_agreement = 0
    valid_predictions = 0

    for pred in predictions:
        if pred is None:
            continue

        valid_predictions += 1
        confidences.append(pred.confidence)

        # Check if prediction matches MFS (sense_frequency == 1)
        pred_freq = sense_freq_map.get(pred.predicted_sense_id)
        if pred_freq == 1:
            mfs_agreement += 1

    mfs_agreement_rate = mfs_agreement / valid_predictions if valid_predictions > 0 else 0.0

    return {
        "model": model_name,
        "valid_predictions": valid_predictions,
        "mfs_agreement_rate": mfs_agreement_rate,
        "confidence_mean": np.mean(confidences) if confidences else 0.0,
        "confidence_std": np.std(confidences) if confidences else 0.0,
        "confidence_min": np.min(confidences) if confidences else 0.0,
        "confidence_max": np.max(confidences) if confidences else 0.0,
    }


def log_results(
    baselines: dict[str, float],
    similarity_results: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
) -> None:
    """Log evaluation results."""
    logger.info("=" * 60)
    logger.info("DWDS WSD EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info("Dataset:")
    logger.info(f"  Words: {len(candidates)}")
    logger.info(f"  Test cases: {baselines['num_test_cases']}")
    logger.info("Baselines:")
    logger.info(f"  Random: {baselines['random_baseline']:.3f}")
    logger.info(f"Similarity-based WSD ({similarity_results['model']}):")
    logger.info(f"  Valid predictions: {similarity_results['valid_predictions']}")
    logger.info(f"  MFS agreement rate: {similarity_results['mfs_agreement_rate']:.3f}")
    logger.info(f"  Confidence: {similarity_results['confidence_mean']:.3f} +/- {similarity_results['confidence_std']:.3f}")
    logger.info("=" * 60)
    logger.info("NOTE: These results are indicative only.")
    logger.info("Without human-annotated gold labels, we cannot compute true accuracy.")
    logger.info("MFS agreement rate shows how often the model predicts the most frequent sense.")
    logger.info("=" * 60)


def main() -> None:
    """Main entry point."""
    random.seed(42)
    np.random.seed(42)

    # Load data
    contexts = load_dwds_corpus()
    germanet_df = load_germanet_senses()

    # Find evaluation candidates
    candidates = find_evaluation_candidates(contexts, germanet_df)

    if len(candidates) < 3:
        logger.warning(f"Insufficient polysemous words: {len(candidates)} found (need at least 3).")
        return

    # Prepare test data
    test_df = prepare_test_data(contexts, candidates)

    # Compute baselines
    baselines = evaluate_mfs_baseline(germanet_df, test_df)

    # Run similarity evaluation
    similarity_results = run_similarity_evaluation(germanet_df, test_df)

    log_results(baselines, similarity_results, candidates)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_file = OUTPUT_DIR / "dwds_wsd_evaluation_results.json"

    results = {
        "baselines": baselines,
        "similarity_results": similarity_results,
        "candidates": {k: v for k, v in candidates.items()},
        "evaluation_date": pd.Timestamp.now().isoformat(),
    }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
