"""Sense prediction functions for diachronic WSD."""

import gc
import logging
from typing import List

import pandas as pd

from src.wsd import Embedder, SimilarityWSD
from src.wsd.embedding_strategies import embed_with_representation, RepresentationType

logger = logging.getLogger(__name__)


def is_e5_model(model_name: str) -> bool:
    """Check if model is an E5 model requiring query/passage prefixes."""
    return "e5" in model_name.lower()


def add_prefix(texts: List[str], prefix: str) -> List[str]:
    """Add prefix to texts for E5 models."""
    return [f"{prefix}{t}" for t in texts]


def predict_senses(
    contexts_df: pd.DataFrame,
    senses_df: pd.DataFrame,
    model_name: str = "intfloat/multilingual-e5-large",
    representation: RepresentationType = "weighted",
) -> pd.DataFrame:
    """Predict senses for all contexts using embedding similarity.

    Args:
        contexts_df: DataFrame with DWDS contexts (must have 'lemma', 'sentence')
        senses_df: DataFrame with senses (must have 'lemma', 'synset_id')
        model_name: Embedding model to use
        representation: Representation type (baseline, weighted, etc.)

    Returns:
        DataFrame with predictions added
    """
    logger.info(f"\nLoading model: {model_name}")
    embedder = Embedder(model_name=model_name, normalize=True)

    use_e5 = is_e5_model(model_name)
    if use_e5:
        logger.info("  (E5 model: applying query/passage prefixes)")

    # Encode all sense texts once
    logger.info(f"Encoding senses using representation: '{representation}'...")
    sense_embeddings = embed_with_representation(
        senses_df,
        embedder,
        representation=representation,
        show_progress=True,
        passage_prefix="passage: " if use_e5 else "",
    )

    # Create WSD predictor
    wsd = SimilarityWSD(
        sense_embeddings=sense_embeddings,
        sense_ids=senses_df["synset_id"].tolist(),
        sense_lemmas=senses_df["lemma"].tolist(),
    )

    # Encode contexts in batches
    logger.info("Encoding and predicting contexts...")
    context_texts = contexts_df["sentence"].fillna("").tolist()
    if use_e5:
        context_texts = add_prefix(context_texts, "query: ")

    context_embeddings = embedder.encode(context_texts, show_progress=True, batch_size=64)

    # Get predictions
    predictions = wsd.predict_batch(
        context_embeddings=context_embeddings,
        lemmas=contexts_df["lemma"].tolist(),
    )

    # Extract results
    predicted_senses = []
    confidences = []

    for pred in predictions:
        if pred is not None:
            predicted_senses.append(pred.predicted_sense_id)
            confidences.append(pred.confidence)
        else:
            predicted_senses.append(None)
            confidences.append(None)

    # Add to dataframe
    results_df = contexts_df.copy()
    results_df["predicted_sense_id"] = predicted_senses
    results_df["confidence"] = confidences

    # Add sense metadata (columns vary between wsd_senses.csv and sense_variants.csv)
    meta_cols = ["synset_id", "class"]
    if "paraphrase" in senses_df.columns:
        meta_cols.append("paraphrase")
    sense_meta = senses_df[meta_cols].drop_duplicates()
    results_df = results_df.merge(
        sense_meta,
        left_on="predicted_sense_id",
        right_on="synset_id",
        how="left",
    )
    results_df = results_df.drop(columns=["synset_id"])
    rename_map = {"class": "sense_class"}
    if "paraphrase" in results_df.columns:
        rename_map["paraphrase"] = "sense_paraphrase"
    results_df = results_df.rename(columns=rename_map)

    # Cleanup
    del embedder, wsd, sense_embeddings, context_embeddings
    gc.collect()

    return results_df