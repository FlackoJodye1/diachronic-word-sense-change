# -*- coding: utf-8 -*-
"""
GermaNet parsing module.

This module provides functionality for parsing GermaNet XML files to extract
synsets, lexical units, and semantic relations.

Example usage:
    from germanet.parsing import XMLParser, RelationParser
    
    # Parse synsets and lexical units
    xml_parser = XMLParser()
    synsets, lexunits = xml_parser.parse_xml_files(xml_dir)
    
    # Parse relations
    relation_parser = RelationParser()
    relations = relation_parser.parse_relations_from_dir(xml_dir)
"""

from .xml_parser import XMLParser, SynsetParser, LexicalUnitParser
from .relation_parser import RelationParser

__all__ = [
    'XMLParser',
    'SynsetParser', 
    'LexicalUnitParser',
    'RelationParser'
]