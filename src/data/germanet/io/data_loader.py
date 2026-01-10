# -*- coding: utf-8 -*-
"""
Simple data loading for GermaNet processed data.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


class DataLoader:
    """Simple handler for loading processed GermaNet data from CSV files."""
    
    def __init__(self, data_dir: Path, config=None):
        self.data_dir = Path(data_dir)
        self.germanet_dir = self.data_dir / 'germanet_v190'
        self.logger = logging.getLogger(__name__)
    
    def load_table(self, name: str) -> Optional[pd.DataFrame]:
        """Load a single table from CSV."""
        if not self.germanet_dir.exists():
            self.logger.error(f"Data directory does not exist: {self.germanet_dir}")
            return None
        
        try:
            filepath = self.germanet_dir / f'{name}.csv'
            if not filepath.exists():
                self.logger.error(f"File not found: {filepath}")
                return None
                
            df = pd.read_csv(filepath, encoding='utf-8')
            
            # Convert | separated strings back to lists for known list columns
            list_columns = ['variants', 'examples']
            for col in list_columns:
                if col in df.columns:
                    df[col] = df[col].apply(
                        lambda x: x.split('|') if isinstance(x, str) and x else []
                    )
            
            self.logger.info(f'Loaded {name}: {len(df)} rows')
            return df
            
        except Exception as e:
            self.logger.error(f'Failed to load {name}.csv: {e}')
            return None
    
    def load_all_tables(self, table_names: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Load multiple tables."""
        if table_names is None:
            table_names = ['synsets', 'lexical_units', 'relations', 'lemma_senses', 'sense_texts']
        
        tables = {}
        for name in table_names:
            df = self.load_table(name)
            if df is not None:
                tables[name] = df
        
        self.logger.info(f'Loaded {len(tables)}/{len(table_names)} tables successfully')
        return tables