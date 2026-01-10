# -*- coding: utf-8 -*-
"""
Relation parsing functionality for GermaNet data.

This module handles parsing of GermaNet semantic relations from gn_relations.xml.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
from lxml import etree


class RelationParser:
    """Parser for GermaNet semantic relations."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the relation parser.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def parse_relations(self, relations_file: Path) -> List[Dict]:
        """Parse the semantic relations from gn_relations.xml.
        
        Args:
            relations_file: Path to gn_relations.xml file
            
        Returns:
            List of relation dictionaries
        """
        self.logger.info(f'Parsing relations from {relations_file}')
        
        relations = []
        
        try:
            parser = etree.XMLParser(recover=True, huge_tree=True)
            tree = etree.parse(str(relations_file), parser)
            root = tree.getroot()
            
            for rel in root.xpath('//con_rel'):
                relation_data = {
                    'relation_type': rel.get('name'),
                    'from_synset': rel.get('from'),
                    'to_synset': rel.get('to'),
                    'direction': rel.get('dir'),
                    'inverse': rel.get('inv')
                }
                relations.append(relation_data)
                
        except Exception as e:
            self.logger.error(f'Error parsing relations: {e}')
        
        self.logger.info(f'Parsed {len(relations)} semantic relations')
        return relations
    
    def parse_relations_from_dir(self, xml_dir: Path) -> List[Dict]:
        """Parse relations from the standard gn_relations.xml file in a directory.
        
        Args:
            xml_dir: Directory containing gn_relations.xml
            
        Returns:
            List of relation dictionaries
        """
        relations_file = xml_dir / 'gn_relations.xml'
        
        if not relations_file.exists():
            self.logger.warning(f'Relations file not found: {relations_file}')
            return []
        
        return self.parse_relations(relations_file)
    
    def get_relation_statistics(self, relations: List[Dict]) -> Dict:
        """Get statistics about the parsed relations.
        
        Args:
            relations: List of relation dictionaries
            
        Returns:
            Dictionary with relation statistics
        """
        if not relations:
            return {'total': 0, 'by_type': {}}
        
        # Count relations by type
        type_counts = {}
        for rel in relations:
            rel_type = rel.get('relation_type', 'unknown')
            type_counts[rel_type] = type_counts.get(rel_type, 0) + 1
        
        return {
            'total': len(relations),
            'by_type': type_counts,
            'unique_types': len(type_counts)
        }
    
    def filter_relations_by_type(self, relations: List[Dict], relation_types: List[str]) -> List[Dict]:
        """Filter relations by specific types.
        
        Args:
            relations: List of relation dictionaries
            relation_types: List of relation types to keep
            
        Returns:
            Filtered list of relations
        """
        return [rel for rel in relations if rel.get('relation_type') in relation_types]
    
    def get_synset_relations(self, relations: List[Dict], synset_id: str) -> List[Dict]:
        """Get all relations involving a specific synset.
        
        Args:
            relations: List of relation dictionaries
            synset_id: Synset ID to find relations for
            
        Returns:
            List of relations involving the synset
        """
        return [rel for rel in relations 
                if rel.get('from_synset') == synset_id or rel.get('to_synset') == synset_id]