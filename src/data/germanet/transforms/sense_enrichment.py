# -*- coding: utf-8 -*-
"""
Sense enrichment functionality for GermaNet data.

This module provides methods to enrich sparse GermaNet sense information by:
1. Building contextual descriptions from semantic relations
2. Leveraging sibling synsets and hypernym chains
3. Aggregating co-occurring lemmas for disambiguation
4. Using WordNet translations where available
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, Counter


class SenseEnricher:
    """Enriches sparse GermaNet sense information for better WSD performance."""
    
    def __init__(self, lemma_senses_df: pd.DataFrame, relations_df: pd.DataFrame):
        """Initialize the sense enricher.
        
        Args:
            lemma_senses_df: DataFrame with lemma-sense mappings
            relations_df: DataFrame with semantic relations
        """
        self.lemma_senses = lemma_senses_df
        self.relations = relations_df
        self.logger = logging.getLogger(__name__)
        
        # Build lookup indexes for efficient access
        self._build_indexes()
    
    def _build_indexes(self):
        """Build internal indexes for efficient lookups."""
        self.logger.info("Building lookup indexes for sense enrichment")
        
        # Synset -> lemmas mapping
        self.synset_to_lemmas = defaultdict(list)
        for _, row in self.lemma_senses.iterrows():
            self.synset_to_lemmas[row['synset_id']].append(row['lemma'])
        
        # Relation indexes
        self.hypernyms = defaultdict(list)  # child -> [parents]
        self.hyponyms = defaultdict(list)   # parent -> [children] 
        self.related = defaultdict(list)    # synset -> [related_synsets]
        
        for _, rel in self.relations.iterrows():
            if rel['relation_type'] == 'has_hypernym':
                self.hypernyms[rel['from_synset']].append(rel['to_synset'])
                self.hyponyms[rel['to_synset']].append(rel['from_synset'])
            elif rel['relation_type'] in ['is_related_to', 'has_component_meronym', 
                                         'has_substance_meronym', 'has_member_meronym']:
                self.related[rel['from_synset']].append(rel['to_synset'])
                self.related[rel['to_synset']].append(rel['from_synset'])
        
        self.logger.info(f"Built indexes: {len(self.synset_to_lemmas)} synsets, "
                        f"{len(self.hypernyms)} hypernym relations")
    
    def get_hypernym_chain(self, synset_id: str, max_depth: int = 3) -> List[str]:
        """Get the hypernym chain for a synset (bottom-up hierarchy).
        
        Args:
            synset_id: Target synset ID
            max_depth: Maximum depth to traverse
            
        Returns:
            List of synset IDs from specific to general
        """
        chain = [synset_id]
        current = synset_id
        
        for _ in range(max_depth):
            parents = self.hypernyms.get(current, [])
            if not parents:
                break
            # Take the first hypernym (most direct parent)
            parent = parents[0]
            chain.append(parent)
            current = parent
            
        return chain
    
    def get_sibling_synsets(self, synset_id: str) -> List[str]:
        """Get sibling synsets (same direct hypernym).
        
        Args:
            synset_id: Target synset ID
            
        Returns:
            List of sibling synset IDs
        """
        siblings = []
        parents = self.hypernyms.get(synset_id, [])
        
        for parent in parents:
            # Get all children of this parent
            children = self.hyponyms.get(parent, [])
            # Add siblings (excluding the original synset)
            siblings.extend([child for child in children if child != synset_id])
        
        return list(set(siblings))
    
    def get_related_lemmas(self, synset_id: str, relation_types: List[str] = None) -> List[str]:
        """Get lemmas from related synsets.
        
        Args:
            synset_id: Target synset ID
            relation_types: Types of relations to consider
            
        Returns:
            List of related lemmas
        """
        if relation_types is None:
            # Use all available related synsets
            related_synsets = self.related.get(synset_id, [])
        else:
            # Filter by specific relation types
            related_synsets = []
            for _, rel in self.relations[self.relations['from_synset'] == synset_id].iterrows():
                if rel['relation_type'] in relation_types:
                    related_synsets.append(rel['to_synset'])
        
        # Collect lemmas from related synsets
        related_lemmas = []
        for related_synset in related_synsets:
            related_lemmas.extend(self.synset_to_lemmas.get(related_synset, []))
        
        return related_lemmas
    
    def build_contextual_description(self, synset_id: str) -> str:
        """Build a contextual description for a synset using semantic relations.
        
        Args:
            synset_id: Target synset ID
            
        Returns:
            Generated contextual description
        """
        # Get synset info
        synset_info = self.lemma_senses[self.lemma_senses['synset_id'] == synset_id].iloc[0]
        synset_lemmas = self.synset_to_lemmas.get(synset_id, [])
        
        description_parts = []
        
        # 1. Basic info: semantic class and lemmas
        pos = synset_info['pos']
        semantic_class = synset_info['class']
        description_parts.append(f"A {pos.lower()} in the semantic class '{semantic_class}'")
        
        if len(synset_lemmas) > 1:
            description_parts.append(f"including synonyms: {', '.join(synset_lemmas[:5])}")
        
        # 2. Hypernym context (hierarchical position)
        hypernym_chain = self.get_hypernym_chain(synset_id, max_depth=2)
        if len(hypernym_chain) > 1:
            parent_synset = hypernym_chain[1]
            parent_lemmas = self.synset_to_lemmas.get(parent_synset, [])
            if parent_lemmas:
                description_parts.append(f"which is a type of {'/'.join(parent_lemmas[:3])}")
        
        # 3. Sibling context (distinguishing features)
        siblings = self.get_sibling_synsets(synset_id)
        if siblings:
            sibling_lemmas = []
            for sibling in siblings[:3]:  # Limit to 3 siblings
                sibling_lemmas.extend(self.synset_to_lemmas.get(sibling, [])[:2])
            if sibling_lemmas:
                description_parts.append(f"related to {', '.join(sibling_lemmas[:6])}")
        
        # 4. Component/meronym relations
        component_lemmas = self.get_related_lemmas(synset_id, 
                                                 ['has_component_meronym', 'has_substance_meronym'])
        if component_lemmas:
            description_parts.append(f"composed of/containing {', '.join(component_lemmas[:4])}")
        
        return "; ".join(description_parts) + "."
    
    def enrich_synset(self, synset_id: str) -> Dict[str, any]:
        """Generate enriched information for a single synset.
        
        Args:
            synset_id: Target synset ID
            
        Returns:
            Dictionary with enriched information
        """
        # Get original synset data
        synset_data = self.lemma_senses[self.lemma_senses['synset_id'] == synset_id].iloc[0].to_dict()
        
        # Add enrichment
        enrichment = {
            'original_paraphrase': synset_data.get('paraphrase'),
            'contextual_description': self.build_contextual_description(synset_id),
            'hypernym_chain': self.get_hypernym_chain(synset_id),
            'sibling_synsets': self.get_sibling_synsets(synset_id),
            'related_lemmas': self.get_related_lemmas(synset_id),
            'synset_lemmas': self.synset_to_lemmas.get(synset_id, []),
            'enrichment_score': self._calculate_enrichment_score(synset_id)
        }
        
        # Combine original and enriched data
        return {**synset_data, **enrichment}
    
    def _calculate_enrichment_score(self, synset_id: str) -> float:
        """Calculate an enrichment quality score for a synset.
        
        Args:
            synset_id: Target synset ID
            
        Returns:
            Score from 0-1 indicating richness of available information
        """
        synset_info = self.lemma_senses[self.lemma_senses['synset_id'] == synset_id].iloc[0]
        score = 0.0
        
        # Base information
        if pd.notna(synset_info.get('paraphrase')):
            score += 0.4
        if pd.notna(synset_info.get('english_paraphrase')):
            score += 0.3
        if synset_info.get('examples') and len(synset_info.get('examples', [])) > 0:
            score += 0.2
        
        # Relational context
        if self.hypernyms.get(synset_id):
            score += 0.05
        if self.get_sibling_synsets(synset_id):
            score += 0.03
        if self.related.get(synset_id):
            score += 0.02
        
        return min(score, 1.0)
    
    def enrich_lemma_senses(self, target_lemmas: Set[str] = None) -> pd.DataFrame:
        """Enrich sense information for specified lemmas or all lemmas.
        
        Args:
            target_lemmas: Set of lemmas to enrich (None for all)
            
        Returns:
            DataFrame with enriched sense information
        """
        self.logger.info("Enriching lemma-sense mappings with contextual descriptions")
        
        # Filter to target lemmas if specified
        if target_lemmas:
            df_subset = self.lemma_senses[self.lemma_senses['lemma'].isin(target_lemmas)]
        else:
            df_subset = self.lemma_senses
        
        enriched_rows = []
        
        for _, row in df_subset.iterrows():
            synset_id = row['synset_id']
            enriched_data = self.enrich_synset(synset_id)
            enriched_rows.append(enriched_data)
        
        enriched_df = pd.DataFrame(enriched_rows)
        
        self.logger.info(f"Enriched {len(enriched_df)} synset entries")
        return enriched_df
    
    def generate_disambiguation_features(self, lemma: str) -> Dict[str, List]:
        """Generate features for disambiguating senses of a polysemous lemma.
        
        Args:
            lemma: Target lemma
            
        Returns:
            Dictionary with disambiguation features for each sense
        """
        lemma_senses = self.lemma_senses[self.lemma_senses['lemma'] == lemma]
        
        if len(lemma_senses) <= 1:
            return {}
        
        features = {}
        
        for _, sense_row in lemma_senses.iterrows():
            synset_id = sense_row['synset_id']
            
            # Key distinguishing features
            sense_features = {
                'semantic_class': sense_row['class'],
                'contextual_description': self.build_contextual_description(synset_id),
                'hypernym_context': [self.synset_to_lemmas.get(h, []) for h in self.get_hypernym_chain(synset_id)],
                'related_domains': self.get_related_lemmas(synset_id),
                'distinguishing_siblings': [self.synset_to_lemmas.get(s, []) for s in self.get_sibling_synsets(synset_id)[:3]]
            }
            
            features[synset_id] = sense_features
        
        return features


def load_and_enrich_senses(data_dir: Path, target_words_file: Path = None) -> pd.DataFrame:
    """Load GermaNet data and enrich sense information.
    
    Args:
        data_dir: Path to processed GermaNet data directory
        target_words_file: Optional file with target words to focus on
        
    Returns:
        DataFrame with enriched sense information
    """
    logger = logging.getLogger(__name__)
    
    # Load data - try multiple possible locations
    lemma_senses_file = data_dir / 'lemma_senses.parquet'
    relations_file = data_dir / 'relations.parquet'
    
    # Fallback to external data if needed
    if not lemma_senses_file.exists():
        external_dir = data_dir.parent.parent.parent / 'data' / 'external' / 'germanet_v190'
        lemma_senses_file = external_dir / 'lemma_senses.parquet'
    if not relations_file.exists():
        external_dir = data_dir.parent.parent.parent / 'data' / 'external' / 'germanet_v190' 
        relations_file = external_dir / 'relations.parquet'
    
    lemma_senses = pd.read_parquet(lemma_senses_file)
    relations = pd.read_parquet(relations_file)
    
    logger.info(f"Loaded {len(lemma_senses)} lemma-senses and {len(relations)} relations")
    
    # Load target words if specified
    target_lemmas = None
    if target_words_file and target_words_file.exists():
        target_df = pd.read_csv(target_words_file, sep='\t' if target_words_file.suffix == '.tsv' else ',')
        target_lemmas = set(target_df.iloc[:, 0].tolist())
        logger.info(f"Focusing on {len(target_lemmas)} target words")
    
    # Initialize enricher and generate enriched senses
    enricher = SenseEnricher(lemma_senses, relations)
    enriched_senses = enricher.enrich_lemma_senses(target_lemmas)
    
    return enriched_senses


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Load and enrich data for target words
    project_dir = Path(__file__).resolve().parents[3]
    data_dir = project_dir / 'data' / 'processed' / 'germanet_v190'
    target_words_file = project_dir / 'data' / 'processed' / 'dwds' / 'target_words.tsv'
    
    enriched_df = load_and_enrich_senses(data_dir, target_words_file)
    
    # Save enriched data
    output_file = data_dir / 'enriched_lemma_senses.parquet'
    enriched_df.to_parquet(output_file, index=False)
    print(f"Saved enriched senses to {output_file}")
    
    # Demo: show enrichment for polysemous words
    test_words = ['Bank', 'Schloss', 'Zelle']
    for word in test_words:
        word_senses = enriched_df[enriched_df['lemma'] == word]
        if len(word_senses) > 0:
            print(f"\n=== Enriched senses for '{word}' ===")
            for _, sense in word_senses.iterrows():
                print(f"Synset {sense['synset_id']} ({sense['class']}):")
                print(f"  {sense['contextual_description']}")
                if pd.notna(sense.get('original_paraphrase')):
                    print(f"  Original: {sense['original_paraphrase']}")
                print()