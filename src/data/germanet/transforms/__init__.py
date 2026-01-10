# -*- coding: utf-8 -*-
"""
GermaNet transforms module.

This module provides functionality for transforming raw GermaNet data into
derived tables and embedding-ready formats.

Example usage:
    from germanet.transforms import SenseTextBuilder, TableBuilder
    
    # Create sense texts
    sense_builder = SenseTextBuilder()
    sense_texts = sense_builder.create_sense_texts(synsets_df, lexunits_df)
    
    # Create lemma_senses table
    table_builder = TableBuilder()
    lemma_senses = table_builder.create_lemma_senses(synsets_df, lexunits_df)
    
    # Or use convenience functions
    from germanet.transforms import create_sense_texts, create_lemma_senses
    sense_texts = create_sense_texts(synsets_df, lexunits_df)
    lemma_senses = create_lemma_senses(synsets_df, lexunits_df)
"""

from .sense_builder import SenseTextBuilder, create_sense_texts
from .table_builder import TableBuilder, create_lemma_senses
from .candidate_mapper import CandidateMapper, create_candidate_map
from .embedding_builder import EmbeddingBuilder, create_embeddings
from .relation_text_builder import RelationTextBuilder, enrich_sense_texts_with_relations

__all__ = [
    'SenseTextBuilder',
    'TableBuilder', 
    'CandidateMapper',
    'EmbeddingBuilder',
    'RelationTextBuilder',
    'create_sense_texts',
    'create_lemma_senses',
    'create_candidate_map',
    'create_embeddings',
    'enrich_sense_texts_with_relations'
]