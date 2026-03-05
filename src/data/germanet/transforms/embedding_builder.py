"""Creates embeddings for GermaNet sense texts using sentence transformers."""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
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
    