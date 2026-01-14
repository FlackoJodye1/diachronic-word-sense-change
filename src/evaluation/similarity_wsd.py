# -*- coding: utf-8 -*-
"""
Cosine similarity-based Word Sense Disambiguation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class WSDPrediction:
    """A single WSD prediction with ranking information."""

    context_id: str
    lemma: str
    predicted_sense_id: str
    confidence: float
    all_scores: Dict[str, float] = field(default_factory=dict)
    rank_of_gold: Optional[int] = None


@dataclass
class WSDResult:
    """Aggregated WSD evaluation results."""

    accuracy: float
    top3_accuracy: float
    top5_accuracy: float
    mrr: float
    evaluated: int
    skipped: int
    predictions: List[WSDPrediction] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "accuracy": self.accuracy,
            "top3_accuracy": self.top3_accuracy,
            "top5_accuracy": self.top5_accuracy,
            "mrr": self.mrr,
            "evaluated": self.evaluated,
            "skipped": self.skipped,
        }


class SimilarityWSD:
    """Cosine similarity-based Word Sense Disambiguation.

    Given pre-computed embeddings for senses and contexts, performs WSD
    by finding the most similar sense for each context within the same lemma.
    """

    def __init__(
        self,
        sense_embeddings: np.ndarray,
        sense_ids: Sequence[str],
        sense_lemmas: Sequence[str],
    ):
        """Initialize with sense embeddings.

        Args:
            sense_embeddings: Array of shape (n_senses, embedding_dim)
            sense_ids: Sequence of sense IDs corresponding to each embedding
            sense_lemmas: Sequence of lemmas corresponding to each sense
        """
        if len(sense_embeddings) != len(sense_ids) or len(sense_ids) != len(sense_lemmas):
            raise ValueError(
                f"Length mismatch: {len(sense_embeddings)} embeddings, "
                f"{len(sense_ids)} sense_ids, {len(sense_lemmas)} lemmas"
            )

        self.sense_embeddings = self._normalize(sense_embeddings)
        self.sense_ids = list(sense_ids)
        self.sense_lemmas = list(sense_lemmas)

        # Build lemma -> sense indices lookup
        self.lemma_to_indices: Dict[str, List[int]] = {}
        for idx, lemma in enumerate(self.sense_lemmas):
            self.lemma_to_indices.setdefault(lemma, []).append(idx)

    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize embeddings."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return embeddings / norms

    def predict(
        self,
        context_embedding: np.ndarray,
        lemma: str,
    ) -> Optional[WSDPrediction]:
        """Predict sense for a single context.

        Args:
            context_embedding: Embedding vector for the context (1D or 2D)
            lemma: Target lemma to disambiguate

        Returns:
            WSDPrediction or None if lemma not found
        """
        if lemma not in self.lemma_to_indices:
            return None

        sense_indices = self.lemma_to_indices[lemma]
        sense_vecs = self.sense_embeddings[sense_indices]

        # Ensure context is 2D and normalized
        context_vec = context_embedding.reshape(1, -1)
        context_vec = self._normalize(context_vec)

        # Compute cosine similarities
        scores = np.dot(context_vec, sense_vecs.T).flatten()
        ranking = np.argsort(-scores)

        # Get ranked sense IDs and scores
        ranked_sense_ids = [self.sense_ids[sense_indices[i]] for i in ranking]
        all_scores = {
            self.sense_ids[sense_indices[i]]: float(scores[i]) for i in range(len(scores))
        }

        best_idx = ranking[0]
        predicted_sense_id = self.sense_ids[sense_indices[best_idx]]
        confidence = float(scores[best_idx])

        return WSDPrediction(
            context_id="",
            lemma=lemma,
            predicted_sense_id=predicted_sense_id,
            confidence=confidence,
            all_scores=all_scores,
        )

    def predict_batch(
        self,
        context_embeddings: np.ndarray,
        lemmas: Sequence[str],
        context_ids: Optional[Sequence[str]] = None,
    ) -> List[Optional[WSDPrediction]]:
        """Predict senses for multiple contexts.

        Args:
            context_embeddings: Array of shape (n_contexts, embedding_dim)
            lemmas: Sequence of lemmas to disambiguate
            context_ids: Optional sequence of context identifiers

        Returns:
            List of WSDPrediction (or None for unknown lemmas)
        """
        if context_ids is None:
            context_ids = [str(i) for i in range(len(lemmas))]

        predictions = []
        for i, (embedding, lemma) in enumerate(zip(context_embeddings, lemmas)):
            pred = self.predict(embedding, lemma)
            if pred is not None:
                pred.context_id = context_ids[i]
            predictions.append(pred)

        return predictions

    def evaluate(
        self,
        context_embeddings: np.ndarray,
        lemmas: Sequence[str],
        gold_sense_ids: Sequence[str],
        context_ids: Optional[Sequence[str]] = None,
    ) -> WSDResult:
        """Evaluate WSD performance against gold standard.

        Args:
            context_embeddings: Array of shape (n_contexts, embedding_dim)
            lemmas: Sequence of lemmas to disambiguate
            gold_sense_ids: Sequence of gold standard sense IDs
            context_ids: Optional sequence of context identifiers

        Returns:
            WSDResult with accuracy metrics
        """
        if context_ids is None:
            context_ids = [str(i) for i in range(len(lemmas))]

        context_embeddings = self._normalize(context_embeddings)

        evaluated = 0
        correct = 0
        top3_correct = 0
        top5_correct = 0
        mrr_sum = 0.0
        skipped = 0
        predictions: List[WSDPrediction] = []

        for i, (embedding, lemma, gold_sense, ctx_id) in enumerate(
            zip(context_embeddings, lemmas, gold_sense_ids, context_ids)
        ):
            if lemma not in self.lemma_to_indices:
                skipped += 1
                continue

            sense_indices = self.lemma_to_indices[lemma]
            lemma_sense_ids = [self.sense_ids[idx] for idx in sense_indices]

            if gold_sense not in lemma_sense_ids:
                skipped += 1
                continue

            # Compute similarities
            sense_vecs = self.sense_embeddings[sense_indices]
            context_vec = embedding.reshape(1, -1)
            scores = np.dot(context_vec, sense_vecs.T).flatten()
            ranking = np.argsort(-scores)

            ranked_senses = [lemma_sense_ids[idx] for idx in ranking]
            all_scores = {
                lemma_sense_ids[idx]: float(scores[idx]) for idx in range(len(scores))
            }

            # Calculate metrics
            evaluated += 1
            predicted_sense = ranked_senses[0]

            if predicted_sense == gold_sense:
                correct += 1
            if gold_sense in ranked_senses[:3]:
                top3_correct += 1
            if gold_sense in ranked_senses[:5]:
                top5_correct += 1

            rank = ranked_senses.index(gold_sense) + 1
            mrr_sum += 1.0 / rank

            pred = WSDPrediction(
                context_id=ctx_id,
                lemma=lemma,
                predicted_sense_id=predicted_sense,
                confidence=float(scores[ranking[0]]),
                all_scores=all_scores,
                rank_of_gold=rank,
            )
            predictions.append(pred)

        if evaluated == 0:
            return WSDResult(
                accuracy=0.0,
                top3_accuracy=0.0,
                top5_accuracy=0.0,
                mrr=0.0,
                evaluated=0,
                skipped=skipped,
                predictions=[],
            )

        return WSDResult(
            accuracy=correct / evaluated,
            top3_accuracy=top3_correct / evaluated,
            top5_accuracy=top5_correct / evaluated,
            mrr=mrr_sum / evaluated,
            evaluated=evaluated,
            skipped=skipped,
            predictions=predictions,
        )

    def to_predictions_df(self, predictions: List[WSDPrediction]) -> pd.DataFrame:
        """Convert predictions to DataFrame format for WSDEvaluator.

        Args:
            predictions: List of WSDPrediction objects

        Returns:
            DataFrame with columns: instance_id, predicted_sense_id, confidence
        """
        records = []
        for pred in predictions:
            if pred is not None:
                records.append({
                    "instance_id": pred.context_id,
                    "predicted_sense_id": pred.predicted_sense_id,
                    "confidence": pred.confidence,
                })
        return pd.DataFrame(records)

    @classmethod
    def from_dataframe(
        cls,
        sense_df: pd.DataFrame,
        embeddings: np.ndarray,
        sense_id_col: str = "synset_id",
        lemma_col: str = "lemma",
    ) -> "SimilarityWSD":
        """Create SimilarityWSD from a DataFrame and embeddings.

        Args:
            sense_df: DataFrame with sense metadata
            embeddings: Pre-computed sense embeddings
            sense_id_col: Column name for sense IDs
            lemma_col: Column name for lemmas

        Returns:
            SimilarityWSD instance
        """
        return cls(
            sense_embeddings=embeddings,
            sense_ids=sense_df[sense_id_col].tolist(),
            sense_lemmas=sense_df[lemma_col].tolist(),
        )