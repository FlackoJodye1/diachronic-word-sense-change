# -*- coding: utf-8 -*-
"""
Candidate mapping functionality for GermaNet data.

This module creates a fast lookup from (lemma, POS) → [synset_id, ...] using lemma_senses.csv.
This trims the search space for WSD to only the senses that actually belong to the lemma.
"""

import logging
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any


class CandidateMapper:
    """Creates and manages candidate mappings from (lemma, POS) to synset IDs."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the candidate mapper.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.candidates = {}
    
    def create_candidate_map(self, lemma_senses_df: pd.DataFrame) -> Dict[Tuple[str, str], List[str]]:
        """Create candidate mapping from lemma_senses DataFrame.
        
        Args:
            lemma_senses_df: DataFrame containing lemma, pos, and synset_id columns
            
        Returns:
            Dictionary mapping (lemma, pos) tuples to lists of synset IDs
        """
        if lemma_senses_df is None or lemma_senses_df.empty:
            raise ValueError("lemma_senses_df cannot be None or empty")
        
        required_columns = ['lemma', 'pos', 'synset_id']
        missing_columns = [col for col in required_columns if col not in lemma_senses_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        self.logger.info('Creating candidate mapping from lemma_senses data')
        
        candidates = {}
        for (lemma, pos), sub in lemma_senses_df.groupby(["lemma", "pos"]):
            ids = sub["synset_id"].dropna().astype(str).unique().tolist()
            if ids:
                candidates[(lemma, pos)] = ids
        
        self.candidates = candidates
        self.logger.info(f'Created candidate mapping with {len(candidates)} (lemma, pos) keys')
        return candidates
    
    def save_candidate_map(self, output_path: Path, candidates: Optional[Dict] = None) -> None:
        """Save candidate mapping to pickle file.
        
        Args:
            output_path: Path to save the pickle file
            candidates: Optional candidate map to save (uses self.candidates if None)
        """
        candidates_to_save = candidates or self.candidates
        if not candidates_to_save:
            raise ValueError("No candidates to save. Run create_candidate_map first.")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "wb") as f:
            pickle.dump(candidates_to_save, f)
        
        self.logger.info(f'Saved candidate mapping to {output_path} with {len(candidates_to_save)} keys')
    
    def load_candidate_map(self, input_path: Path) -> Dict[Tuple[str, str], List[str]]:
        """Load candidate mapping from pickle file.
        
        Args:
            input_path: Path to the pickle file
            
        Returns:
            Dictionary mapping (lemma, pos) tuples to lists of synset IDs
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Candidate map file not found: {input_path}")
        
        with open(input_path, "rb") as f:
            candidates = pickle.load(f)
        
        self.candidates = candidates
        self.logger.info(f'Loaded candidate mapping from {input_path} with {len(candidates)} keys')
        return candidates
    
    def get_candidates(self, lemma: str, pos: str) -> List[str]:
        """Get candidate synset IDs for a given lemma and POS.
        
        Args:
            lemma: The lemma to look up
            pos: The part-of-speech tag
            
        Returns:
            List of synset IDs for the given (lemma, pos) pair
        """
        return self.candidates.get((lemma, pos), [])
    
    def get_statistics(self, candidates: Optional[Dict] = None) -> Dict[str, Any]:
        """Get statistics about the candidate mapping.
        
        Args:
            candidates: Optional candidate map to analyze (uses self.candidates if None)
            
        Returns:
            Dictionary with statistics
        """
        candidates_to_analyze = candidates or self.candidates
        if not candidates_to_analyze:
            return {
                'total_pairs': 0,
                'empty_candidate_lists': 0,
                'avg_candidates_per_pair': 0,
                'max_candidates': 0,
                'min_candidates': 0
            }
        
        candidate_counts = [len(ids) for ids in candidates_to_analyze.values()]
        empty_lists = sum(1 for ids in candidates_to_analyze.values() if not ids)
        
        # Check for non-synset IDs (should start with 's')
        non_synset_pairs = sum(
            1 for ids in candidates_to_analyze.values() 
            if any(not str(sid).startswith("s") for sid in ids)
        )
        
        return {
            'total_pairs': len(candidates_to_analyze),
            'empty_candidate_lists': empty_lists,
            'avg_candidates_per_pair': sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0,
            'max_candidates': max(candidate_counts) if candidate_counts else 0,
            'min_candidates': min(candidate_counts) if candidate_counts else 0,
            'pairs_with_non_synset_ids': non_synset_pairs
        }
    
    def validate_candidate_map(self, candidates: Optional[Dict] = None) -> bool:
        """Validate the candidate mapping.
        
        Args:
            candidates: Optional candidate map to validate (uses self.candidates if None)
            
        Returns:
            True if validation passes, False otherwise
        """
        candidates_to_validate = candidates or self.candidates
        if not candidates_to_validate:
            self.logger.error("No candidates to validate")
            return False
        
        # Check that all keys are (lemma, pos) tuples
        invalid_keys = [
            key for key in candidates_to_validate.keys() 
            if not isinstance(key, tuple) or len(key) != 2
        ]
        if invalid_keys:
            self.logger.error(f"Found {len(invalid_keys)} invalid keys (not (lemma, pos) tuples)")
            return False
        
        # Check that all values are lists of strings
        invalid_values = [
            key for key, value in candidates_to_validate.items()
            if not isinstance(value, list) or not all(isinstance(sid, str) for sid in value)
        ]
        if invalid_values:
            self.logger.error(f"Found {len(invalid_values)} invalid values (not lists of strings)")
            return False
        
        # Check for synset ID format (should start with 's')
        stats = self.get_statistics(candidates_to_validate)
        if stats['pairs_with_non_synset_ids'] > 0:
            self.logger.warning(f"Found {stats['pairs_with_non_synset_ids']} pairs with non-synset IDs")
        
        self.logger.info("Candidate mapping validation completed successfully")
        return True
    
    def inspect_candidates(self, lemma: str, pos: str, sense_texts_df: Optional[pd.DataFrame] = None) -> None:
        """Inspect candidates for a specific lemma and POS.
        
        Args:
            lemma: The lemma to inspect
            pos: The part-of-speech tag
            sense_texts_df: Optional DataFrame with synset texts for display
        """
        candidates = self.get_candidates(lemma, pos)
        print(f"{lemma}/{pos}: {len(candidates)} candidates -> {candidates}")
        
        if sense_texts_df is not None and not candidates:
            return
        
        if sense_texts_df is not None:
            matching_senses = sense_texts_df[sense_texts_df['synset_id'].isin(candidates)]
            if 'paraphrase' in matching_senses.columns:
                print(matching_senses[['synset_id', 'paraphrase']])
            elif 'text_for_embedding' in matching_senses.columns:
                for _, row in matching_senses.iterrows():
                    preview = str(row['text_for_embedding'])[:90]
                    print(f"  - {row['synset_id']}: {preview}")


def create_candidate_map(lemma_senses_df: pd.DataFrame, 
                        output_path: Optional[Path] = None,
                        config: Optional[Dict] = None) -> Dict[Tuple[str, str], List[str]]:
    """Convenience function to create and optionally save candidate mapping.
    
    Args:
        lemma_senses_df: DataFrame containing lemma, pos, and synset_id columns
        output_path: Optional path to save the mapping as pickle file
        config: Optional configuration dictionary
        
    Returns:
        Dictionary mapping (lemma, pos) tuples to lists of synset IDs
    """
    mapper = CandidateMapper(config)
    candidates = mapper.create_candidate_map(lemma_senses_df)
    
    if output_path:
        mapper.save_candidate_map(output_path, candidates)
    
    return candidates