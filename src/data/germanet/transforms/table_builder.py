# -*- coding: utf-8 -*-
"""
Table building functionality for GermaNet data.

This module handles creation of derived tables like lemma_senses by joining
and transforming the raw GermaNet synset and lexical unit data.
"""

import logging
import pandas as pd
from typing import Dict, List, Optional


class TableBuilder:
    """Builder for creating derived tables from GermaNet data."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the table builder.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def create_lemma_senses(self, synsets_df: pd.DataFrame, lexical_units_df: pd.DataFrame) -> pd.DataFrame:
        """Create lemma_senses table by joining lexical units with synsets.
        
        Args:
            synsets_df: DataFrame containing synset data
            lexical_units_df: DataFrame containing lexical unit data
            
        Returns:
            DataFrame with lemma-synset mappings
        """
        if synsets_df is None or synsets_df.empty:
            raise ValueError("synsets_df cannot be None or empty")
        if lexical_units_df is None or lexical_units_df.empty:
            raise ValueError("lexical_units_df cannot be None or empty")
        
        self.logger.info('Creating lemma_senses table')
        
        # Join lexical units with synsets on synset_id
        lemma_senses = lexical_units_df.merge(
            synsets_df[['synset_id', 'paraphrase']], 
            on='synset_id', 
            how='left'
        )
        
        # Select and rename columns for final output
        lemma_senses = lemma_senses[[
            'lemma', 'pos', 'synset_id', 'class', 'paraphrase', 
            'variants', 'examples'
        ]].copy()
        
        # Remove duplicates (same lemma-synset combinations)
        lemma_senses = lemma_senses.drop_duplicates(subset=['lemma', 'synset_id'])
        
        self.logger.info(f'Created {len(lemma_senses)} unique lemma-synset pairs')
        return lemma_senses
    
    def create_candidate_map(self, lemma_senses_df: pd.DataFrame) -> Dict:
        """Create candidate mapping dictionary from lemma_senses table.
        
        This creates a dictionary mapping (lemma, pos) tuples to lists of synset_ids,
        which is useful for WSD candidate generation.
        
        Args:
            lemma_senses_df: DataFrame with lemma-synset mappings
            
        Returns:
            Dictionary mapping (lemma, pos) -> [synset_id, ...]
        """
        if lemma_senses_df is None or lemma_senses_df.empty:
            raise ValueError("lemma_senses_df cannot be None or empty")
        
        self.logger.info('Creating candidate mapping dictionary')
        
        candidate_map = {}
        
        # Group by (lemma, pos) and collect synset_ids
        grouped = lemma_senses_df.groupby(['lemma', 'pos'])['synset_id'].apply(list)
        
        for (lemma, pos), synset_ids in grouped.items():
            candidate_map[(lemma, pos)] = synset_ids
        
        self.logger.info(f'Created candidate mappings for {len(candidate_map)} lemma-pos pairs')
        return candidate_map
    
    def create_sense_meta(self, synsets_df: pd.DataFrame) -> pd.DataFrame:
        """Create sense metadata table with essential synset information.
        
        Args:
            synsets_df: DataFrame containing synset data
            
        Returns:
            DataFrame with essential synset metadata
        """
        if synsets_df is None or synsets_df.empty:
            raise ValueError("synsets_df cannot be None or empty")
        
        self.logger.info('Creating sense metadata table')
        
        # Select essential columns for sense metadata
        sense_meta = synsets_df[['synset_id', 'pos', 'class', 'paraphrase']].copy()
        
        # Add derived fields
        sense_meta['has_paraphrase'] = sense_meta['paraphrase'].notna()
        sense_meta['paraphrase_length'] = sense_meta['paraphrase'].str.len().fillna(0).astype(int)
        
        self.logger.info(f'Created metadata for {len(sense_meta)} synsets')
        return sense_meta
    
    def get_statistics(self, lemma_senses_df: pd.DataFrame) -> Dict:
        """Get statistics about the lemma_senses table.
        
        Args:
            lemma_senses_df: DataFrame with lemma-synset mappings
            
        Returns:
            Dictionary with statistics
        """
        if lemma_senses_df is None or lemma_senses_df.empty:
            return {
                'total_pairs': 0,
                'unique_lemmas': 0,
                'unique_synsets': 0,
                'by_pos': {},
                'avg_senses_per_lemma': 0
            }
        
        # Basic counts
        total_pairs = len(lemma_senses_df)
        unique_lemmas = lemma_senses_df['lemma'].nunique()
        unique_synsets = lemma_senses_df['synset_id'].nunique()
        
        # Distribution by POS
        by_pos = lemma_senses_df['pos'].value_counts().to_dict()
        
        # Average senses per lemma
        senses_per_lemma = lemma_senses_df.groupby(['lemma', 'pos']).size()
        avg_senses_per_lemma = float(senses_per_lemma.mean())
        
        # Polysemy analysis
        polysemous_lemmas = (senses_per_lemma > 1).sum()
        monosemous_lemmas = (senses_per_lemma == 1).sum()
        
        return {
            'total_pairs': total_pairs,
            'unique_lemmas': unique_lemmas,
            'unique_synsets': unique_synsets,
            'by_pos': by_pos,
            'avg_senses_per_lemma': avg_senses_per_lemma,
            'polysemous_lemmas': int(polysemous_lemmas),
            'monosemous_lemmas': int(monosemous_lemmas),
            'polysemy_ratio': float(polysemous_lemmas / len(senses_per_lemma)) if len(senses_per_lemma) > 0 else 0
        }
    
    def validate_lemma_senses(self, lemma_senses_df: pd.DataFrame) -> bool:
        """Validate the lemma_senses table.
        
        Args:
            lemma_senses_df: DataFrame to validate
            
        Returns:
            True if validation passes, False otherwise
        """
        if lemma_senses_df is None or lemma_senses_df.empty:
            self.logger.error("Lemma senses DataFrame is None or empty")
            return False
        
        required_columns = ['lemma', 'pos', 'synset_id', 'class', 'paraphrase', 'variants', 'examples']
        missing_columns = [col for col in required_columns if col not in lemma_senses_df.columns]
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for null values in key columns
        key_columns = ['lemma', 'pos', 'synset_id']
        for col in key_columns:
            null_count = lemma_senses_df[col].isnull().sum()
            if null_count > 0:
                self.logger.warning(f"Found {null_count} null values in {col}")
        
        # Check for duplicate (lemma, synset_id) pairs
        duplicates = lemma_senses_df.duplicated(subset=['lemma', 'synset_id']).sum()
        if duplicates > 0:
            self.logger.warning(f"Found {duplicates} duplicate lemma-synset pairs")
        
        self.logger.info("Lemma senses validation completed")
        return True
    
    def filter_by_pos(self, lemma_senses_df: pd.DataFrame, pos_list: List[str]) -> pd.DataFrame:
        """Filter lemma_senses by part-of-speech tags.
        
        Args:
            lemma_senses_df: DataFrame with lemma-synset mappings
            pos_list: List of POS tags to keep
            
        Returns:
            Filtered DataFrame
        """
        if lemma_senses_df is None or lemma_senses_df.empty:
            return lemma_senses_df
        
        filtered_df = lemma_senses_df[lemma_senses_df['pos'].isin(pos_list)].copy()
        self.logger.info(f'Filtered to {len(filtered_df)} pairs for POS: {pos_list}')
        return filtered_df
    
    def get_ambiguous_lemmas(self, lemma_senses_df: pd.DataFrame, min_senses: int = 2) -> pd.DataFrame:
        """Get lemmas with multiple senses (ambiguous words).
        
        Args:
            lemma_senses_df: DataFrame with lemma-synset mappings
            min_senses: Minimum number of senses to be considered ambiguous
            
        Returns:
            DataFrame with ambiguous lemmas and their sense counts
        """
        if lemma_senses_df is None or lemma_senses_df.empty:
            return pd.DataFrame(columns=['lemma', 'pos', 'sense_count'])
        
        # Count senses per lemma-pos combination
        sense_counts = lemma_senses_df.groupby(['lemma', 'pos']).size().reset_index(name='sense_count')
        
        # Filter to ambiguous lemmas
        ambiguous = sense_counts[sense_counts['sense_count'] >= min_senses].copy()
        ambiguous = ambiguous.sort_values('sense_count', ascending=False)
        
        self.logger.info(f'Found {len(ambiguous)} ambiguous lemmas with {min_senses}+ senses')
        return ambiguous


def create_lemma_senses(synsets_df: pd.DataFrame, lexical_units_df: pd.DataFrame, 
                       config: Optional[Dict] = None) -> pd.DataFrame:
    """Convenience function to create lemma_senses table.
    
    Args:
        synsets_df: DataFrame containing synset data
        lexical_units_df: DataFrame containing lexical unit data
        config: Optional configuration dictionary
        
    Returns:
        DataFrame with lemma-synset mappings
    """
    builder = TableBuilder(config)
    return builder.create_lemma_senses(synsets_df, lexical_units_df)


def create_candidate_map(lemma_senses_df: pd.DataFrame, config: Optional[Dict] = None) -> Dict:
    """Convenience function to create candidate mapping dictionary.
    
    Args:
        lemma_senses_df: DataFrame with lemma-synset mappings
        config: Optional configuration dictionary
        
    Returns:
        Dictionary mapping (lemma, pos) -> [synset_id, ...]
    """
    builder = TableBuilder(config)
    return builder.create_candidate_map(lemma_senses_df)