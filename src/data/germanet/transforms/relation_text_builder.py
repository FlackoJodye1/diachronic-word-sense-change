"""Enriches sense texts with information from semantically related synsets."""

import logging
import pandas as pd
import networkx as nx
from typing import List, Tuple


class RelationTextBuilder:
    """Builds relation-enriched texts for embedding generation."""

    TEXT_SEPARATOR = ' ; '
    MAX_RELATED_TERMS = 10
    INCLUDE_RELATION_TYPES = False

    # Which relation types to follow, how deep, and how much to weight them
    RELATION_CONFIG = {
        'has_hypernym':          {'include': True, 'max_depth': 2, 'weight': 1.0},  # dog → animal
        'has_hyponym':           {'include': True, 'max_depth': 1, 'weight': 0.8},  # animal → dog
        'has_component_meronym': {'include': True, 'max_depth': 1, 'weight': 0.6},  # car → engine
        'has_member_meronym':    {'include': True, 'max_depth': 1, 'weight': 0.6},  # team → player
        'has_substance_meronym': {'include': True, 'max_depth': 1, 'weight': 0.6},  # statue → marble
        'has_portion_meronym':   {'include': True, 'max_depth': 1, 'weight': 0.6},  # cake → slice
        'is_related_to':         {'include': True, 'max_depth': 1, 'weight': 0.5},  # loose conceptual link
        'causes':                {'include': True, 'max_depth': 1, 'weight': 0.4},  # virus → disease
        'entails':               {'include': True, 'max_depth': 1, 'weight': 0.4},  # snoring → sleeping
    }

    def __init__(self):
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
            
            if rel_type in self.RELATION_CONFIG and self.RELATION_CONFIG[rel_type]['include']:
                self.relation_graph.add_edge(
                    from_synset, to_synset,
                    relation=rel_type,
                    weight=self.RELATION_CONFIG[rel_type]['weight']
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
        for rel_type, config in self.RELATION_CONFIG.items():
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
        return related_terms[:self.MAX_RELATED_TERMS]
    
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
                if self.INCLUDE_RELATION_TYPES:
                    # Include relation type as context
                    formatted_text = f"[{rel_type}] {cleaned_text}"
                else:
                    formatted_text = cleaned_text
                
                text_parts.append(formatted_text)
        
        # Combine all parts
        enriched_text = self.TEXT_SEPARATOR.join(text_parts)
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
    