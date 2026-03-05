"""GermaNet parsing: synsets, lexical units, and semantic relations."""

from .xml_parser import XMLParser
from .relation_parser import RelationParser

__all__ = ['XMLParser', 'RelationParser']