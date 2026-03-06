"""Sense prediction functions for diachronic WSD."""

from __future__ import annotations

import gc
from typing import List, Optional

import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation import Embedder, SimilarityWSD
from src.evaluation.embedding_strategies import embed_with_representation, RepresentationType


def is_e5_model(model_name: str) -> bool:
    """Check if model is an E5 model requiring query/passage prefixes."""
    return "e5" in model_name.lower()


def add_prefix(texts: List[str], prefix: str) -> List[str]:
    """Add prefix to texts for E5 models."""
    return [f"{prefix}{t}" for t in texts]


def predict_senses(
    contexts_df: pd.DataFrame,
    senses_df: pd.DataFrame,
    text_column: Optional[str] = None,
    model_name: str = "intfloat/multilingual-e5-large",
    representation: Optional[RepresentationType] = None,
) -> pd.DataFrame:
    """Predict senses for all contexts using embedding similarity.

    Args:
        contexts_df: DataFrame with DWDS contexts (must have 'lemma', 'sentence')
        senses_df: DataFrame with senses (must have 'lemma', 'synset_id')
        text_column: Column name containing text for embedding (ignored if representation is set)
        model_name: Embedding model to use
        representation: Representation type (baseline, weighted, etc.). If provided, uses
                       embed_with_representation() instead of text_column.

    Returns:
        DataFrame with predictions added
    """
    print(f"\nLoading model: {model_name}")
    embedder = Embedder(model_name=model_name, normalize=True)

    use_e5 = is_e5_model(model_name)
    if use_e5:
        print("  (E5 model: applying query/passage prefixes)")

    # Encode all sense texts once
    if representation:
        print(f"Encoding senses using representation: '{representation}'...")
        sense_embeddings = embed_with_representation(
            senses_df,
            embedder,
            representation=representation,
            show_progress=True
        )
    else:
        if text_column is None:
            raise ValueError("Must provide either text_column or representation parameter")
        print(f"Encoding sense texts from '{text_column}'...")
        sense_texts = senses_df[text_column].fillna("").tolist()
        if use_e5:
            sense_texts = add_prefix(sense_texts, "passage: ")
        sense_embeddings = embedder.encode(sense_texts, show_progress=True)

    # Create WSD predictor
    wsd = SimilarityWSD(
        sense_embeddings=sense_embeddings,
        sense_ids=senses_df["synset_id"].tolist(),
        sense_lemmas=senses_df["lemma"].tolist(),
    )

    # Encode contexts in batches
    print("Encoding and predicting contexts...")
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