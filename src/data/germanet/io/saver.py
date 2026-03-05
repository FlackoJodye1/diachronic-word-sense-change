"""Simple data saving for GermaNet processed data."""

import logging
from pathlib import Path
from typing import Dict
import pandas as pd


def _cell_to_csv(x):
    if isinstance(x, list):
        return '|'.join(map(str, x)) if x else ''
    if x is None or pd.isna(x):
        return ''
    return x


class DataSaver:
    """Simple handler for saving processed GermaNet data as CSV files."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.germanet_dir = self.output_dir / 'base-tables'
        self.logger = logging.getLogger(__name__)
    
    def save_table(self, name: str, df: pd.DataFrame) -> bool:
        """Save a single table as CSV."""
        if df is None or df.empty:
            self.logger.warning(f"Skipping {name}: DataFrame is None or empty")
            return False
        
        self.germanet_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f'Saving {name}: {len(df)} rows')
        
        try:
            df_csv = df.copy()
            for col in df_csv.columns:
                if df_csv[col].dtype == 'object':
                    df_csv[col] = df_csv[col].apply(_cell_to_csv)
            
            filepath = self.germanet_dir / f'{name}.csv'
            df_csv.to_csv(filepath, index=False, encoding='utf-8')
            return True
        except Exception as e:
            self.logger.error(f'Failed to save {name}.csv: {e}')
            return False
    
    def save_all_tables(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, bool]:
        """Save multiple tables as CSV files."""
        results = {}
        for name, df in tables.items():
            results[name] = self.save_table(name, df)
        
        self.logger.info(f'Saved {sum(results.values())}/{len(results)} tables successfully')
        self.logger.info(f'All data saved to {self.germanet_dir}')
        return results