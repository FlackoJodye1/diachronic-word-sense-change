# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

from .parsing import XMLParser, RelationParser
from .transforms import SenseTextBuilder, TableBuilder, CandidateMapper, EmbeddingBuilder, RelationTextBuilder
from .io import DataSaver


class GermaNetProcessor:
    """Process GermaNet XML data into structured DataFrames for WSD tasks."""
    
    def __init__(self, xml_dir: Path, output_dir: Path, config: Optional[Dict] = None):
        """Initialize the GermaNet processor.        """
        self.xml_dir = Path(xml_dir)
        self.output_dir = Path(output_dir)
        self.config = config or {}
        
        # Configuration
        self.text_separator = self.config.get('text_separator', ' ; ')
        
        # Data storage
        self.synsets_df: Optional[pd.DataFrame] = None
        self.lexical_units_df: Optional[pd.DataFrame] = None
        self.relations_df: Optional[pd.DataFrame] = None
        self.lemma_senses_df: Optional[pd.DataFrame] = None
        self.sense_texts_df: Optional[pd.DataFrame] = None
        self.candidate_map: Optional[Dict] = None
        self.embeddings: Optional[np.ndarray] = None
        self.embedding_metadata: Optional[pd.DataFrame] = None
        
        # Initialize parsers, transformers, and I/O
        self.xml_parser = XMLParser(self.config)
        self.relation_parser = RelationParser(self.config)
        self.sense_builder = SenseTextBuilder(self.config)
        self.table_builder = TableBuilder(self.config)
        self.candidate_mapper = CandidateMapper(self.config)
        self.relation_text_builder = RelationTextBuilder(self.config)
        self.embedding_builder = EmbeddingBuilder(self.config)
        self.data_saver = DataSaver(self.output_dir, self.config)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def parse_xml_files(self) -> None:
        """Parse all GermaNet XML files to extract synsets and lexical units."""
        synsets, lexical_units = self.xml_parser.parse_xml_files(self.xml_dir)
        
        # Convert to DataFrames
        self.synsets_df = pd.DataFrame(synsets)
        self.lexical_units_df = pd.DataFrame(lexical_units)
    
    def parse_relations(self) -> None:
        """Parse the semantic relations from gn_relations.xml."""
        relations = self.relation_parser.parse_relations_from_dir(self.xml_dir)
        self.relations_df = pd.DataFrame(relations)
    
    def create_lemma_senses(self) -> pd.DataFrame:
        """Create lemma_senses table by joining lexical units with synsets."""
        if self.synsets_df is None or self.lexical_units_df is None:
            raise ValueError("Must parse XML files first")
        
        lemma_senses = self.table_builder.create_lemma_senses(self.synsets_df, self.lexical_units_df)
        self.lemma_senses_df = lemma_senses
        return lemma_senses
    
    def create_sense_texts(self) -> pd.DataFrame:
        """Create embedding-ready text for each synset by combining all available information."""
        if self.synsets_df is None or self.lexical_units_df is None:
            raise ValueError("Must parse XML files first")
        
        sense_texts = self.sense_builder.create_sense_texts(self.synsets_df, self.lexical_units_df)
        self.sense_texts_df = sense_texts
        return sense_texts
    
    def create_candidate_map(self) -> Dict:
        """Create candidate mapping from (lemma, POS) to synset IDs."""
        if self.lemma_senses_df is None:
            raise ValueError("Must create lemma_senses first")
        
        candidate_map = self.candidate_mapper.create_candidate_map(self.lemma_senses_df)
        self.candidate_map = candidate_map
        return candidate_map
    
    def enrich_sense_texts_with_relations(self) -> pd.DataFrame:
        """Enrich sense texts with semantic relation information."""
        if self.sense_texts_df is None:
            raise ValueError("Must create sense_texts first")
        if self.relations_df is None:
            raise ValueError("Must parse relations first")
        
        # Load relations and sense texts into the builder
        self.relation_text_builder.load_relations(self.relations_df)
        self.relation_text_builder.load_synset_texts(self.sense_texts_df)
        
        # Enrich the sense texts
        enriched_texts = self.relation_text_builder.enrich_sense_texts(self.sense_texts_df)
        
        # Replace the original sense texts with enriched ones
        self.sense_texts_df = enriched_texts
        return enriched_texts
    
    def create_embeddings(self) -> Tuple[np.ndarray, pd.DataFrame]:
        """Create embeddings for sense texts."""
        if self.sense_texts_df is None:
            raise ValueError("Must create sense_texts first")
        
        embeddings = self.embedding_builder.create_embeddings(self.sense_texts_df)
        self.embeddings = embeddings
        self.embedding_metadata = self.embedding_builder.metadata
        return embeddings, self.embedding_metadata
    
    def save_table(self, name: str, df: pd.DataFrame) -> bool:
        """Save a single table as CSV."""
        return self.data_saver.save_table(name, df)
    
    def save_all(self) -> Dict[str, bool]:
        """Save all processed tables as CSV."""
        tables = {
            'synsets': self.synsets_df,
            'lexical_units': self.lexical_units_df,
            'relations': self.relations_df,
            'lemma_senses': self.lemma_senses_df,
            'sense_texts': self.sense_texts_df
        }
        
        return self.data_saver.save_all_tables(tables)
    
    def process_all(self, create_embeddings: bool = False, enrich_with_relations: bool = True) -> None:
        """Run the complete processing pipeline.
        
        Args:
            create_embeddings: Whether to create embeddings (requires sentence-transformers)
            enrich_with_relations: Whether to enrich sense texts with semantic relations
        """
            
        self.logger.info('Starting complete GermaNet processing pipeline')
        
        # Step 1: Parse XML files
        self.parse_xml_files()
        
        # Step 2: Parse relations
        self.parse_relations()
        
        # Step 3: Create derived tables
        self.create_lemma_senses()
        self.create_sense_texts()
        
        # Step 3.5: Enrich sense texts with relations (optional but recommended)
        if enrich_with_relations:
            try:
                self.enrich_sense_texts_with_relations()
                self.logger.info('Successfully enriched sense texts with semantic relations')
            except Exception as e:
                self.logger.warning(f'Failed to enrich sense texts with relations: {e}')
                self.logger.info('Continuing with basic sense texts')
        
        self.create_candidate_map()
        
        # Step 4: Create embeddings (optional)
        if create_embeddings:
            try:
                self.create_embeddings()
                # Save embeddings
                embeddings_path, metadata_path = self.embedding_builder.save_embeddings(self.output_dir)
                self.logger.info(f'Saved embeddings to {embeddings_path} and {metadata_path}')
            except ImportError as e:
                self.logger.warning(f'Skipping embeddings due to missing dependency: {e}')
            except Exception as e:
                self.logger.error(f'Failed to create embeddings: {e}')
        
        # Step 5: Save everything
        self.save_all()
        
        # Step 6: Save candidate map as pickle
        candidate_map_path = self.output_dir / "candidate_map.pkl"
        self.candidate_mapper.save_candidate_map(candidate_map_path, self.candidate_map)
        
        self.logger.info('GermaNet processing pipeline completed successfully')
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the processed data."""
        stats = {}
        
        if self.synsets_df is not None:
            stats['synsets'] = {
                'total': len(self.synsets_df),
                'with_paraphrase': len(self.synsets_df[self.synsets_df['paraphrase'].notna()]),
                'by_pos': self.synsets_df['pos'].value_counts().to_dict()
            }
        
        if self.lexical_units_df is not None:
            stats['lexical_units'] = {
                'total': len(self.lexical_units_df),
                'unique_lemmas': self.lexical_units_df['lemma'].nunique(),
                'by_pos': self.lexical_units_df['pos'].value_counts().to_dict()
            }
        
        if self.relations_df is not None:
            stats['relations'] = {
                'total': len(self.relations_df),
                'by_type': self.relations_df['relation_type'].value_counts().to_dict()
            }
        
        return stats


if __name__ == '__main__':
    import sys
    from pathlib import Path
    
    # Add project root to path for relative imports
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))
    
    from src.data.germanet.cli import main
    main()