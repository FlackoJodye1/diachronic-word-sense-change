# -*- coding: utf-8 -*-
"""
Embedding building functionality for GermaNet data.

This module handles creation of embeddings for synset texts using various embedding models,
primarily sentence transformers for multilingual semantic embeddings.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
import warnings

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    warnings.warn("sentence-transformers not available. Install it to use EmbeddingBuilder.")


class EmbeddingBuilder:
    """Builder for creating embeddings from GermaNet sense texts."""
    
    DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the embedding builder.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.model_name = self.config.get('model_name', self.DEFAULT_MODEL)
        self.batch_size = self.config.get('batch_size', 256)
        self.normalize_embeddings = self.config.get('normalize_embeddings', True)
        self.convert_to_numpy = self.config.get('convert_to_numpy', True)
        self.dtype = self.config.get('dtype', 'float32')
        self.show_progress = self.config.get('show_progress', True)
        
        self.model = None
        self.embeddings = None
        self.metadata = None
        
        self.logger = logging.getLogger(__name__)
    
    def load_model(self, model_name: Optional[str] = None) -> None:
        """Load the embedding model.
        
        Args:
            model_name: Optional model name to override config
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required but not installed")
        
        model_name = model_name or self.model_name
        self.logger.info(f'Loading embedding model: {model_name}')
        
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        
        self.logger.info(f'Model loaded successfully: {model_name}')
    
    def create_embeddings(self, sense_texts_df: pd.DataFrame, 
                         text_column: str = 'text_for_embedding') -> np.ndarray:
        """Create embeddings for sense texts.
        
        Args:
            sense_texts_df: DataFrame containing sense texts
            text_column: Column name containing text to embed
            
        Returns:
            Array of embeddings with shape (n_senses, embedding_dim)
        """
        if sense_texts_df is None or sense_texts_df.empty:
            raise ValueError("sense_texts_df cannot be None or empty")
        
        if text_column not in sense_texts_df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        if self.model is None:
            self.load_model()
        
        self.logger.info(f'Creating embeddings for {len(sense_texts_df)} sense texts')
        
        # Extract texts for embedding
        texts = sense_texts_df[text_column].fillna("").astype(str).tolist()
        
        # Create embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=self.convert_to_numpy,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=self.show_progress
        )
        
        # Convert to specified dtype
        if self.dtype:
            embeddings = embeddings.astype(self.dtype)
        
        self.embeddings = embeddings
        
        # Create metadata DataFrame
        metadata_columns = ['synset_id']
        if 'pos' in sense_texts_df.columns:
            metadata_columns.append('pos')
        if 'lemmas' in sense_texts_df.columns:
            metadata_columns.append('lemmas')
        
        available_columns = [col for col in metadata_columns if col in sense_texts_df.columns]
        self.metadata = sense_texts_df[available_columns].copy()
        
        self.logger.info(f'Created embeddings with shape {embeddings.shape}')
        return embeddings
    
    def save_embeddings(self, output_dir: Path, 
                       embeddings_filename: str = "sense_embeddings.npz",
                       metadata_filename: str = "sense_meta.csv") -> Tuple[Path, Path]:
        """Save embeddings and metadata to disk.
        
        Args:
            output_dir: Directory to save files
            embeddings_filename: Name for embeddings file
            metadata_filename: Name for metadata file
            
        Returns:
            Tuple of (embeddings_path, metadata_path)
        """
        if self.embeddings is None:
            raise ValueError("No embeddings to save. Run create_embeddings first.")
        
        if self.metadata is None:
            raise ValueError("No metadata to save. Run create_embeddings first.")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        embeddings_path = output_dir / embeddings_filename
        metadata_path = output_dir / metadata_filename
        
        # Save embeddings with metadata
        np.savez_compressed(
            embeddings_path,
            X=self.embeddings,
            model=np.array(self.model_name),
            created=np.array(datetime.utcnow().isoformat() + "Z"),
            shape=np.array(self.embeddings.shape),
            dtype=np.array(str(self.embeddings.dtype))
        )
        
        # Save metadata
        self.metadata.to_csv(metadata_path, index=False)
        
        self.logger.info(f'Saved embeddings to {embeddings_path} with shape {self.embeddings.shape}')
        self.logger.info(f'Saved metadata to {metadata_path} with {len(self.metadata)} rows')
        
        return embeddings_path, metadata_path
    
    def load_embeddings(self, embeddings_path: Path, 
                       metadata_path: Optional[Path] = None) -> Tuple[np.ndarray, Optional[pd.DataFrame]]:
        """Load embeddings and metadata from disk.
        
        Args:
            embeddings_path: Path to embeddings file
            metadata_path: Optional path to metadata file
            
        Returns:
            Tuple of (embeddings, metadata_df)
        """
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
        
        # Load embeddings
        data = np.load(embeddings_path)
        embeddings = data['X']
        
        # Load metadata if available
        if 'model' in data:
            model_name = str(data['model'])
            self.logger.info(f'Loaded embeddings created with model: {model_name}')
        
        if 'created' in data:
            created = str(data['created'])
            self.logger.info(f'Embeddings created at: {created}')
        
        self.embeddings = embeddings
        
        # Load metadata if path provided
        metadata = None
        if metadata_path and metadata_path.exists():
            metadata = pd.read_csv(metadata_path)
            self.metadata = metadata
            self.logger.info(f'Loaded metadata with {len(metadata)} rows')
        
        self.logger.info(f'Loaded embeddings with shape {embeddings.shape}')
        return embeddings, metadata
    
    def get_embedding_statistics(self, embeddings: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Get statistics about the embeddings.
        
        Args:
            embeddings: Optional embeddings array (uses self.embeddings if None)
            
        Returns:
            Dictionary with statistics
        """
        embeddings_to_analyze = embeddings if embeddings is not None else self.embeddings
        
        if embeddings_to_analyze is None:
            return {
                'shape': None,
                'dtype': None,
                'memory_usage_mb': 0,
                'mean_norm': 0,
                'std_norm': 0
            }
        
        # Basic statistics
        shape = embeddings_to_analyze.shape
        dtype = embeddings_to_analyze.dtype
        memory_usage_mb = embeddings_to_analyze.nbytes / (1024 * 1024)
        
        # Norm statistics (if embeddings are normalized, should be close to 1.0)
        norms = np.linalg.norm(embeddings_to_analyze, axis=1)
        mean_norm = float(np.mean(norms))
        std_norm = float(np.std(norms))
        
        return {
            'shape': shape,
            'dtype': str(dtype),
            'memory_usage_mb': round(memory_usage_mb, 2),
            'mean_norm': round(mean_norm, 4),
            'std_norm': round(std_norm, 4),
            'min_norm': round(float(np.min(norms)), 4),
            'max_norm': round(float(np.max(norms)), 4)
        }
    
    def validate_embeddings(self, embeddings: Optional[np.ndarray] = None,
                           metadata: Optional[pd.DataFrame] = None) -> bool:
        """Validate the embeddings and metadata.
        
        Args:
            embeddings: Optional embeddings array (uses self.embeddings if None)
            metadata: Optional metadata DataFrame (uses self.metadata if None)
            
        Returns:
            True if validation passes, False otherwise
        """
        embeddings_to_validate = embeddings if embeddings is not None else self.embeddings
        metadata_to_validate = metadata if metadata is not None else self.metadata
        
        if embeddings_to_validate is None:
            self.logger.error("No embeddings to validate")
            return False
        
        # Check for NaN or infinite values
        if np.isnan(embeddings_to_validate).any():
            self.logger.error("Found NaN values in embeddings")
            return False
        
        if np.isinf(embeddings_to_validate).any():
            self.logger.error("Found infinite values in embeddings")
            return False
        
        # Check metadata alignment if available
        if metadata_to_validate is not None:
            if len(embeddings_to_validate) != len(metadata_to_validate):
                self.logger.error(
                    f"Embeddings and metadata length mismatch: "
                    f"{len(embeddings_to_validate)} vs {len(metadata_to_validate)}"
                )
                return False
            
            # Check for required columns
            if 'synset_id' not in metadata_to_validate.columns:
                self.logger.error("Missing 'synset_id' column in metadata")
                return False
        
        self.logger.info("Embedding validation completed successfully")
        return True
    
    def search_similar(self, query_embedding: np.ndarray, 
                      top_k: int = 10,
                      embeddings: Optional[np.ndarray] = None) -> List[Tuple[int, float]]:
        """Find most similar embeddings to a query.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            embeddings: Optional embeddings array (uses self.embeddings if None)
            
        Returns:
            List of (index, similarity_score) tuples
        """
        embeddings_to_search = embeddings if embeddings is not None else self.embeddings
        
        if embeddings_to_search is None:
            raise ValueError("No embeddings available for search")
        
        # Normalize query if embeddings are normalized
        if self.normalize_embeddings:
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Compute similarities (cosine similarity for normalized embeddings)
        similarities = np.dot(embeddings_to_search, query_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(int(idx), float(similarities[idx])) for idx in top_indices]


def create_embeddings(sense_texts_df: pd.DataFrame,
                     output_dir: Optional[Path] = None,
                     config: Optional[Dict] = None) -> Tuple[np.ndarray, pd.DataFrame]:
    """Convenience function to create embeddings.
    
    Args:
        sense_texts_df: DataFrame containing sense texts
        output_dir: Optional directory to save embeddings
        config: Optional configuration dictionary
        
    Returns:
        Tuple of (embeddings, metadata_df)
    """
    builder = EmbeddingBuilder(config)
    embeddings = builder.create_embeddings(sense_texts_df)
    
    if output_dir:
        builder.save_embeddings(output_dir)
    
    return embeddings, builder.metadata