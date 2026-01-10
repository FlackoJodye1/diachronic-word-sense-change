# -*- coding: utf-8 -*-
"""
XML parsing functionality for GermaNet data.

This module handles parsing of GermaNet XML files to extract synsets and lexical units.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from lxml import etree


class XMLParser:
    """Parser for GermaNet XML files."""
    
    DEFAULT_POS_MAPPING = {
        'adj': 'ADJ',
        'nomen': 'NOUN', 
        'subst': 'NOUN',
        'verb': 'VERB',
        'verben': 'VERB'
    }
    
    DEFAULT_SKIP_PATTERNS = ['gn_relations', 'interlingualindex', '.dtd', 'germanet_']
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the XML parser.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.pos_mapping = self.config.get('pos_mapping', self.DEFAULT_POS_MAPPING)
        self.skip_patterns = self.config.get('skip_patterns', self.DEFAULT_SKIP_PATTERNS)
        self.logger = logging.getLogger(__name__)
    
    def normalize_pos(self, category: str) -> str:
        """Normalize POS tags to standard format."""
        if not category:
            return category
        return self.pos_mapping.get(category.lower(), category.upper())
    
    def get_xml_files(self, xml_dir: Path) -> List[Path]:
        """Get list of XML files to process, excluding skip patterns."""
        xml_files = [f for f in xml_dir.glob('*.xml') 
                     if not any(pattern in f.name.lower() for pattern in self.skip_patterns)]
        self.logger.info(f'Found {len(xml_files)} XML files to process')
        return xml_files
    
    def parse_xml_files(self, xml_dir: Path) -> Tuple[List[Dict], List[Dict]]:
        """Parse all GermaNet XML files to extract synsets and lexical units.
        
        Args:
            xml_dir: Path to directory containing XML files
            
        Returns:
            Tuple of (synsets_list, lexical_units_list)
        """
        self.logger.info(f'Parsing XML files from {xml_dir}')
        
        synsets = []
        lexical_units = []
        xml_files = self.get_xml_files(xml_dir)
        
        for xml_file in xml_files:
            self.logger.info(f'Processing {xml_file.name}')
            
            try:
                file_synsets, file_lexunits = self._parse_single_xml_file(xml_file)
                synsets.extend(file_synsets)
                lexical_units.extend(file_lexunits)
                
                self.logger.info(f'{xml_file.name}: {len(file_synsets)} synsets, {len(file_lexunits)} lexical units')
                        
            except Exception as e:
                self.logger.error(f'Error processing {xml_file.name}: {e}')
                continue
        
        self.logger.info(f'Parsed {len(synsets)} total synsets and {len(lexical_units)} lexical units')
        return synsets, lexical_units
    
    def _parse_single_xml_file(self, xml_file: Path) -> Tuple[List[Dict], List[Dict]]:
        """Parse a single XML file and return synsets and lexical units.
        
        Args:
            xml_file: Path to XML file to parse
            
        Returns:
            Tuple of (synsets_list, lexical_units_list)
        """
        # Use forgiving parser for malformed XML
        parser = etree.XMLParser(recover=True, huge_tree=True)
        tree = etree.parse(str(xml_file), parser)
        root = tree.getroot()
        
        synsets = []
        lexical_units = []
        
        for synset in root.xpath('//synset'):
            synset_id = synset.get('id')
            category = synset.get('category')
            synset_class = synset.get('class')
            normalized_pos = self.normalize_pos(category) if category else None
            
            # Extract paraphrase if present
            paraphrase_elem = synset.find('paraphrase')
            paraphrase = paraphrase_elem.text if paraphrase_elem is not None else None
            
            synset_data = {
                'synset_id': synset_id,
                'pos': normalized_pos,
                'class': synset_class,
                'paraphrase': paraphrase
            }
            synsets.append(synset_data)
            
            # Extract lexical units for this synset
            for lexunit in synset.xpath('lexUnit'):
                lexunit_data = self._parse_lexical_unit(lexunit, synset_id, normalized_pos, synset_class)
                lexical_units.append(lexunit_data)
        
        return synsets, lexical_units
    
    def _parse_lexical_unit(self, lexunit, synset_id: str, pos: str, synset_class: str) -> Dict:
        """Parse a single lexical unit element.
        
        Args:
            lexunit: XML element for lexical unit
            synset_id: ID of parent synset
            pos: Normalized POS tag
            synset_class: Synset class
            
        Returns:
            Dictionary with lexical unit data
        """
        lexunit_id = lexunit.get('id')
        sense = lexunit.get('sense')
        source = lexunit.get('source')
        named_entity = lexunit.get('namedEntity') == 'yes'
        artificial = lexunit.get('artificial') == 'yes'
        style_marking = lexunit.get('styleMarking') == 'yes'
        
        # Extract orthographic form (lemma)
        orth_form = lexunit.find('orthForm')
        lemma = orth_form.text if orth_form is not None else None
        
        # Extract ALL variants as list
        variants = []
        for orth_var in lexunit.xpath('orthVar'):
            if orth_var.text:
                variants.append(orth_var.text)
        
        # Extract ALL examples as list
        examples = []
        for example in lexunit.xpath('example/text'):
            if example.text:
                examples.append(example.text)
        
        return {
            'lexunit_id': lexunit_id,
            'synset_id': synset_id,
            'lemma': lemma,
            'variants': variants,
            'named_entity': named_entity,
            'artificial': artificial,
            'style_marking': style_marking,
            'examples': examples,
            'pos': pos,
            'class': synset_class,
            'sense': sense,
            'source': source
        }


class SynsetParser(XMLParser):
    """Specialized parser focused on synset extraction."""
    
    def parse_synsets_only(self, xml_dir: Path) -> List[Dict]:
        """Parse XML files and return only synsets (no lexical units).
        
        Args:
            xml_dir: Path to directory containing XML files
            
        Returns:
            List of synset dictionaries
        """
        synsets, _ = self.parse_xml_files(xml_dir)
        return synsets


class LexicalUnitParser(XMLParser):
    """Specialized parser focused on lexical unit extraction."""
    
    def parse_lexical_units_only(self, xml_dir: Path) -> List[Dict]:
        """Parse XML files and return only lexical units (no synsets).
        
        Args:
            xml_dir: Path to directory containing XML files
            
        Returns:
            List of lexical unit dictionaries
        """
        _, lexical_units = self.parse_xml_files(xml_dir)
        return lexical_units