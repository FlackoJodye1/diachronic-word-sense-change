"""Relation parsing for GermaNet semantic relations from gn_relations.xml."""

import logging
from pathlib import Path
from typing import Dict, List
from lxml import etree


class RelationParser:
    """Parser for GermaNet semantic relations."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def parse_relations(self, relations_file: Path) -> List[Dict]:
        """Parse semantic relations from gn_relations.xml."""
        self.logger.info(f'Parsing relations from {relations_file}')

        relations = []

        try:
            parser = etree.XMLParser(recover=True, huge_tree=True)
            tree = etree.parse(str(relations_file), parser)
            root = tree.getroot()

            for rel in root.xpath('//con_rel'):
                relations.append({
                    'relation_type': rel.get('name'),
                    'from_synset': rel.get('from'),
                    'to_synset': rel.get('to'),
                    'direction': rel.get('dir'),
                    'inverse': rel.get('inv'),
                })

        except Exception as e:
            self.logger.error(f'Error parsing relations: {e}')

        self.logger.info(f'Parsed {len(relations)} semantic relations')
        return relations

    def parse_relations_from_dir(self, xml_dir: Path) -> List[Dict]:
        """Parse relations from the standard gn_relations.xml file in a directory."""
        relations_file = xml_dir / 'gn_relations.xml'

        if not relations_file.exists():
            self.logger.warning(f'Relations file not found: {relations_file}')
            return []

        return self.parse_relations(relations_file)