# -*- coding: utf-8 -*-
"""
Sense text building functionality for GermaNet data.

This module handles creation of embedding-ready text for each synset by combining
paraphrases, examples, lemmas, and variants from the raw GermaNet data.
"""

import logging
import pandas as pd
from typing import Dict, Optional


class SenseTextBuilder:
    """Builder for creating embedding-ready sense texts from GermaNet data."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the sense text builder.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.text_separator = self.config.get('text_separator', ' ; ')
        self.logger = logging.getLogger(__name__)
    
    def create_sense_texts(self, synsets_df: pd.DataFrame, lexical_units_df: pd.DataFrame) -> pd.DataFrame:
        """Create embedding-ready text for each synset by combining all available information.
        
        Args:
            synsets_df: DataFrame containing synset data
            lexical_units_df: DataFrame containing lexical unit data
            
        Returns:
            DataFrame with synset_id and text_for_embedding columns
        """
        if synsets_df is None or synsets_df.empty:
            raise ValueError("synsets_df cannot be None or empty")
        if lexical_units_df is None or lexical_units_df.empty:
            raise ValueError("lexical_units_df cannot be None or empty")
        
        self.logger.info('Creating sense texts for embeddings')
        
        # Normalize missing values in lexical units
        lexical_units_clean = self._normalize_lexical_units(lexical_units_df)
        
        # Aggregate lemmas/variants/examples per synset
        agg_by_sense = self._aggregate_by_synset(lexical_units_clean)
        
        # Merge with synsets
        senses = synsets_df.merge(agg_by_sense, on="synset_id", how="left")
        
        # Create embedding texts
        sense_texts = self._create_embedding_texts(senses)
        
        self.logger.info(f'Created {len(sense_texts)} sense texts for embeddings')
        return sense_texts
    
    def _normalize_lexical_units(self, lexical_units_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize missing values in lexical units DataFrame."""
        lexical_units_clean = lexical_units_df.copy()
        
        for col in ["variants", "examples", "lemma"]:
            if col in lexical_units_clean:
                lexical_units_clean[col] = lexical_units_clean[col].fillna("")
        
        return lexical_units_clean
    
    def _aggregate_by_synset(self, lexical_units_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate lemmas, variants, and examples by synset."""
        agg_by_sense = lexical_units_df.groupby("synset_id").agg(
            lemmas=("lemma", lambda s: self.text_separator.join(sorted(set(s.astype(str))))),
            variants=("variants", lambda s: self._aggregate_list_column(s)),
            examples=("examples", lambda s: self._aggregate_list_column(s))
        ).reset_index()
        
        return agg_by_sense
    
    def _aggregate_list_column(self, series) -> str:
        """Helper to aggregate list columns with proper flattening."""
        items = []
        for sublist in series:
            if isinstance(sublist, list):
                items.extend([str(item) for item in sublist if item and str(item) != ""])
            elif sublist and str(sublist) != "":
                items.append(str(sublist))
        return self.text_separator.join(sorted(set(items)))
    
    def _create_embedding_texts(self, senses_df: pd.DataFrame) -> pd.DataFrame:
        """Create the final embedding texts by combining all components."""
        # Normalize missing values in merged data
        for col in ["paraphrase", "lemmas", "variants", "examples"]:
            if col in senses_df:
                senses_df[col] = senses_df[col].fillna("").astype(str)
        
        # Text-clean the input for embedding
        cols = ["paraphrase", "examples", "lemmas", "variants"]
        senses_df[cols] = (senses_df[cols]
            .fillna("")
            .astype(str)
            .replace(r"^\s*\[\s*\]\s*$", "", regex=True)
            .apply(lambda s: s.str.strip())
        )
        
        # Create combined text for embedding
        senses_df["text_for_embedding"] = (
            senses_df["paraphrase"].fillna("") + self.text_separator +
            senses_df["examples"] + self.text_separator +
            senses_df["lemmas"] + self.text_separator +
            senses_df["variants"]
        ).str.strip(self.text_separator)
        
        # Create final sense_texts dataframe
        sense_texts = senses_df[["synset_id", "text_for_embedding"]].copy()
        
        # Check for any null values
        null_count = sense_texts["text_for_embedding"].isnull().sum()
        if null_count > 0:
            self.logger.warning(f"Found {null_count} null text_for_embedding values")
        
        return sense_texts
    
    def get_statistics(self, sense_texts_df: pd.DataFrame) -> Dict:
        """Get statistics about the created sense texts.
        
        Args:
            sense_texts_df: DataFrame with sense texts
            
        Returns:
            Dictionary with statistics
        """
        if sense_texts_df is None or sense_texts_df.empty:
            return {'total': 0, 'empty_texts': 0, 'avg_length': 0}
        
        # Calculate text lengths
        text_lengths = sense_texts_df['text_for_embedding'].str.len()
        empty_texts = (sense_texts_df['text_for_embedding'].str.strip() == '').sum()
        
        return {
            'total': len(sense_texts_df),
            'empty_texts': int(empty_texts),
            'avg_length': float(text_lengths.mean()),
            'min_length': int(text_lengths.min()),
            'max_length': int(text_lengths.max()),
            'median_length': float(text_lengths.median())
        }
    
    def validate_sense_texts(self, sense_texts_df: pd.DataFrame) -> bool:
        """Validate the created sense texts.
        
        Args:
            sense_texts_df: DataFrame with sense texts to validate
            
        Returns:
            True if validation passes, False otherwise
        """
        if sense_texts_df is None or sense_texts_df.empty:
            self.logger.error("Sense texts DataFrame is None or empty")
            return False
        
        required_columns = ['synset_id', 'text_for_embedding']
        missing_columns = [col for col in required_columns if col not in sense_texts_df.columns]
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for null synset_ids
        null_synset_ids = sense_texts_df['synset_id'].isnull().sum()
        if null_synset_ids > 0:
            self.logger.warning(f"Found {null_synset_ids} null synset_ids")
        
        # Check for duplicate synset_ids
        duplicates = sense_texts_df['synset_id'].duplicated().sum()
        if duplicates > 0:
            self.logger.warning(f"Found {duplicates} duplicate synset_ids")
        
        self.logger.info("Sense texts validation completed")
        return True


def create_sense_texts(synsets_df: pd.DataFrame, lexical_units_df: pd.DataFrame, 
                      config: Optional[Dict] = None) -> pd.DataFrame:
    """Convenience function to create sense texts.
    
    Args:
        synsets_df: DataFrame containing synset data
        lexical_units_df: DataFrame containing lexical unit data
        config: Optional configuration dictionary
        
    Returns:
        DataFrame with synset_id and text_for_embedding columns
    """
    builder = SenseTextBuilder(config)
    return builder.create_sense_texts(synsets_df, lexical_units_df)