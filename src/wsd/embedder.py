# -*- coding: utf-8 -*-
"""
Unified text embedding module using sentence-transformers.
"""

import logging
from typing import List, Optional, Sequence, Union

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


logger = logging.getLogger(__name__)


# Common multilingual models for German WSD
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

RECOMMENDED_MODELS = {
    "paraphrase-multilingual-MiniLM-L12-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "paraphrase-multilingual-mpnet-base-v2": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "distiluse-base-multilingual-cased-v2": "sentence-transformers/distiluse-base-multilingual-cased-v2",
    "multilingual-e5-base": "intfloat/multilingual-e5-base",
    "multilingual-e5-large": "intfloat/multilingual-e5-large",
}


class Embedder:
    """Unified text embedder using sentence-transformers.

    Provides a consistent interface for encoding texts into dense vectors,
    supporting batched encoding and multiple model backends.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        """Initialize the embedder.

        Args:
            model_name: HuggingFace model identifier or short name from RECOMMENDED_MODELS
            device: Device to use ('cpu', 'cuda', 'mps'). None for auto-detection.
            normalize: Whether to L2-normalize embeddings (recommended for cosine similarity)
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )

        # Resolve short names to full model paths
        resolved_name = RECOMMENDED_MODELS.get(model_name, model_name)

        logger.info(f"Loading embedding model: {resolved_name}")
        self.model = SentenceTransformer(resolved_name, device=device)
        self.model_name = resolved_name
        self.normalize = normalize
        self._dimension: Optional[int] = None

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        if self._dimension is None:
            # Encode a dummy text to get dimension
            dummy_embedding = self.model.encode(["test"], convert_to_numpy=True)
            self._dimension = dummy_embedding.shape[1]
        return self._dimension

    def encode(
        self,
        texts: Union[str, Sequence[str]],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode texts into dense embeddings.

        Args:
            texts: Single text or sequence of texts to encode
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        # Handle empty input
        if len(texts) == 0:
            return np.array([]).reshape(0, self.dimension)

        # Replace None/NaN with empty strings
        texts = [t if t and isinstance(t, str) else "" for t in texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )

        return embeddings

    def encode_with_metadata(
        self,
        texts: Sequence[str],
        ids: Optional[Sequence[str]] = None,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> tuple[np.ndarray, List[str]]:
        """Encode texts and return embeddings with their IDs.

        Args:
            texts: Sequence of texts to encode
            ids: Optional sequence of IDs corresponding to texts.
                 If None, uses indices as IDs.
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (embeddings array, list of IDs)
        """
        if ids is None:
            ids = [str(i) for i in range(len(texts))]

        if len(ids) != len(texts):
            raise ValueError(f"Length mismatch: {len(texts)} texts vs {len(ids)} ids")

        embeddings = self.encode(texts, batch_size=batch_size, show_progress=show_progress)

        return embeddings, list(ids)

    def __repr__(self) -> str:
        return f"Embedder(model='{self.model_name}', normalize={self.normalize})"


def get_available_models() -> dict[str, str]:
    """Return dictionary of recommended models with descriptions."""
    return {
        "paraphrase-multilingual-MiniLM-L12-v2": "Fast, good quality (384 dim)",
        "paraphrase-multilingual-mpnet-base-v2": "Higher quality, slower (768 dim)",
        "distiluse-base-multilingual-cased-v2": "Classic distilUSE model (512 dim)",
        "multilingual-e5-base": "E5 base model, good for retrieval (768 dim)",
        "multilingual-e5-large": "E5 large model, best quality (1024 dim)",
    }