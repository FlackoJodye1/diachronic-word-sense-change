# -*- coding: utf-8 -*-
"""
Relation-aware text building for GermaNet embeddings.

This module enhances sense texts by incorporating information from semantically
related synsets before embedding generation, creating richer representations.
"""

import logging
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict


class RelationTextBuilder:
    """Builds relation-enriched texts for embedding generation."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the relation text builder.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Configuration for relation inclusion
        self.relation_config = {
            # “is a kind of”: dog → animal
            'has_hypernym': {'include': True, 'max_depth': 2, 'weight': 1.0},
            # “is a type of”: animal → dog
            'has_hyponym': {'include': True, 'max_depth': 1, 'weight': 0.8},
            # “has as a physical part”: car → engine
            'has_component_meronym': {'include': True, 'max_depth': 1, 'weight': 0.6},
            # “has as a member”: team → player
            'has_member_meronym': {'include': True, 'max_depth': 1, 'weight': 0.6},
            # “is made of”: statue → marble
            'has_substance_meronym': {'include': True, 'max_depth': 1, 'weight': 0.6},
            # “has as a portion”: cake → slice
            'has_portion_meronym': {'include': True, 'max_depth': 1, 'weight': 0.6},
            # is_related_to: loosely connected conceptually
            'is_related_to': {'include': True, 'max_depth': 1, 'weight': 0.5},
            # one concept causes another: virus → disease
            'causes': {'include': True, 'max_depth': 1, 'weight': 0.4},
            # one implies another: snoring → sleeping
            'entails': {'include': True, 'max_depth': 1, 'weight': 0.4}
        }
        self.relation_config.update(self.config.get('relation_config', {}))
        
        self.text_separator = self.config.get('text_separator', ' ; ')
        self.max_related_terms = self.config.get('max_related_terms', 10)
        self.include_relation_types = self.config.get('include_relation_types', False)
        
        self.relations_df = None
        self.relation_graph = None
        self.synset_texts = {}
        
        self.logger = logging.getLogger(__name__)
    
    def load_relations(self, relations_df: pd.DataFrame) -> None:
        """Load semantic relations into a graph structure.
        
        Args:
            relations_df: DataFrame with relation data
        """
        self.relations_df = relations_df.copy()
        
        # Build directed graph
        self.relation_graph = nx.DiGraph()
        
        for _, row in relations_df.iterrows():
            rel_type = row['relation_type']
            from_synset = row['from_synset']
            to_synset = row['to_synset']
            
            if rel_type in self.relation_config and self.relation_config[rel_type]['include']:
                self.relation_graph.add_edge(
                    from_synset, to_synset,
                    relation=rel_type,
                    weight=self.relation_config[rel_type]['weight']
                )
        
        self.logger.info(f'Loaded {len(self.relation_graph.edges)} relevant semantic relations')
    
    def load_synset_texts(self, sense_texts_df: pd.DataFrame) -> None:
        """Load original synset texts.
        
        Args:
            sense_texts_df: DataFrame with synset_id and text_for_embedding columns
        """
        self.synset_texts = dict(zip(
            sense_texts_df['synset_id'], 
            sense_texts_df['text_for_embedding']
        ))
        self.logger.info(f'Loaded texts for {len(self.synset_texts)} synsets')
    
    def get_related_text_terms(self, synset_id: str) -> List[Tuple[str, str, float]]:
        """Get text terms from related synsets.
        
        Args:
            synset_id: Source synset ID
            
        Returns:
            List of (term, relation_type, weight) tuples
        """
        if not self.relation_graph or synset_id not in self.relation_graph:
            return []
        
        related_terms = []
        
        # Traverse relations according to configuration
        for rel_type, config in self.relation_config.items():
            if not config['include']:
                continue
            
            max_depth = config['max_depth']
            base_weight = config['weight']
            
            # BFS traversal for this relation type
            visited = set()
            queue = [(synset_id, 0)]
            
            while queue:
                current_synset, depth = queue.pop(0)
                
                if depth >= max_depth or current_synset in visited:
                    continue
                
                visited.add(current_synset)
                
                # Find neighbors with this relation type
                for neighbor in self.relation_graph.neighbors(current_synset):
                    edge_data = self.relation_graph[current_synset][neighbor]
                    
                    if edge_data.get('relation') == rel_type:
                        # Get text for this related synset
                        if neighbor in self.synset_texts:
                            related_text = self.synset_texts[neighbor]
                            if related_text and related_text.strip():
                                # Weight decreases with depth
                                weight = base_weight * (0.7 ** depth)
                                related_terms.append((related_text, rel_type, weight))
                        
                        # Continue traversal
                        if depth + 1 < max_depth:
                            queue.append((neighbor, depth + 1))
        
        # Sort by weight and limit number of terms
        related_terms.sort(key=lambda x: x[2], reverse=True)
        return related_terms[:self.max_related_terms]
    
    def build_enriched_text(self, synset_id: str, original_text: str) -> str:
        """Build relation-enriched text for a synset.
        
        Args:
            synset_id: Synset ID
            original_text: Original text for embedding
            
        Returns:
            Enriched text incorporating related synset information
        """
        if not self.relation_graph:
            return original_text
        
        # Start with original text
        text_parts = [original_text] if original_text and original_text.strip() else []
        
        # Add related terms
        related_terms = self.get_related_text_terms(synset_id)
        
        for related_text, rel_type, weight in related_terms:
            # Clean and format related text
            cleaned_text = related_text.strip()
            
            if cleaned_text and cleaned_text not in text_parts:
                if self.include_relation_types:
                    # Include relation type as context
                    formatted_text = f"[{rel_type}] {cleaned_text}"
                else:
                    formatted_text = cleaned_text
                
                text_parts.append(formatted_text)
        
        # Combine all parts
        enriched_text = self.text_separator.join(text_parts)
        return enriched_text
    
    def enrich_sense_texts(self, sense_texts_df: pd.DataFrame) -> pd.DataFrame:
        """Enrich all sense texts with relation information.
        
        Args:
            sense_texts_df: DataFrame with synset texts
            
        Returns:
            DataFrame with enriched texts
        """
        if self.relation_graph is None:
            self.logger.warning("No relations loaded. Returning original texts.")
            return sense_texts_df.copy()
        
        enriched_df = sense_texts_df.copy()
        
        self.logger.info(f'Enriching {len(sense_texts_df)} sense texts with relations')
        
        enriched_texts = []
        for i, row in sense_texts_df.iterrows():
            if i % 10000 == 0:
                self.logger.info(f'Processed {i}/{len(sense_texts_df)} texts')
            
            synset_id = row['synset_id']
            original_text = row['text_for_embedding']
            
            enriched_text = self.build_enriched_text(synset_id, original_text)
            enriched_texts.append(enriched_text)
        
        enriched_df['text_for_embedding'] = enriched_texts
        
        self.logger.info('Sense text enrichment completed')
        return enriched_df
    
    def get_enrichment_statistics(self, original_df: pd.DataFrame, enriched_df: pd.DataFrame) -> Dict:
        """Get statistics about the enrichment process.
        
        Args:
            original_df: Original sense texts DataFrame
            enriched_df: Enriched sense texts DataFrame
            
        Returns:
            Dictionary with enrichment statistics
        """
        original_lengths = original_df['text_for_embedding'].str.len()
        enriched_lengths = enriched_df['text_for_embedding'].str.len()
        
        expansion_ratios = enriched_lengths / (original_lengths + 1)  # +1 to avoid division by zero
        
        return {
            'total_synsets': len(original_df),
            'avg_original_length': float(original_lengths.mean()),
            'avg_enriched_length': float(enriched_lengths.mean()),
            'avg_expansion_ratio': float(expansion_ratios.mean()),
            'max_expansion_ratio': float(expansion_ratios.max()),
            'min_expansion_ratio': float(expansion_ratios.min()),
            'synsets_with_relations': len(enriched_df[enriched_lengths > original_lengths]),
            'avg_length_increase': float((enriched_lengths - original_lengths).mean())
        }


def enrich_sense_texts_with_relations(sense_texts_df: pd.DataFrame,
                                     relations_df: pd.DataFrame,
                                     config: Optional[Dict] = None) -> pd.DataFrame:
    """Convenience function to enrich sense texts with relations.
    
    Args:
        sense_texts_df: DataFrame with synset texts
        relations_df: DataFrame with semantic relations
        config: Optional configuration
        
    Returns:
        DataFrame with relation-enriched texts
    """
    builder = RelationTextBuilder(config)
    builder.load_relations(relations_df)
    builder.load_synset_texts(sense_texts_df)
    return builder.enrich_sense_texts(sense_texts_df)