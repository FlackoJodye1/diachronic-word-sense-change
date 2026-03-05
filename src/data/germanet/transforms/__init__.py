"""GermaNet transforms: builds sense texts, candidate maps, and embeddings."""

from .sense_builder import SenseTextBuilder
from .table_builder import TableBuilder
from .candidate_mapper import CandidateMapper
from .embedding_builder import EmbeddingBuilder
from .relation_text_builder import RelationTextBuilder

__all__ = [
    'SenseTextBuilder',
    'TableBuilder',
    'CandidateMapper',
    'EmbeddingBuilder',
    'RelationTextBuilder',
]