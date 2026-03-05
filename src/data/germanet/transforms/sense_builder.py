"""Creates embedding-ready sense texts by combining paraphrases, examples, and lemmas."""

import logging
import pandas as pd
from typing import Dict, Optional


class SenseTextBuilder:
    """Builder for creating embedding-ready sense texts from GermaNet data."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the sense text builder.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.text_separator = self.config.get('text_separator', ' ; ')
        self.logger = logging.getLogger(__name__)
    
    def create_sense_texts(self, synsets_df: pd.DataFrame, lexical_units_df: pd.DataFrame) -> pd.DataFrame:
        """Create embedding-ready text for each synset by combining all available information.
        
        Args:
            synsets_df: DataFrame containing synset data
            lexical_units_df: DataFrame containing lexical unit data
            
        Returns:
            DataFrame with synset_id and text_for_embedding columns
        """
        if synsets_df is None or synsets_df.empty:
            raise ValueError("synsets_df cannot be None or empty")
        if lexical_units_df is None or lexical_units_df.empty:
            raise ValueError("lexical_units_df cannot be None or empty")
        
        self.logger.info('Creating sense texts for embeddings')
        
        # Normalize missing values in lexical units (fills NaN in variants, examples, lemma with "")
        lexical_units_clean = self._normalize_lexical_units(lexical_units_df)
        
        # Aggregate lemmas/variants/examples per synset
        agg_by_sense = self._aggregate_by_synset(lexical_units_clean)
        
        # Merge with synsets
        senses = synsets_df.merge(agg_by_sense, on="synset_id", how="left")
        
        # Create embedding texts
        sense_texts = self._create_embedding_texts(senses)
        
        self.logger.info(f'Created {len(sense_texts)} sense texts for embeddings')
        return sense_texts
    
    def _normalize_lexical_units(self, lexical_units_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize missing values in lexical units DataFrame."""
        lexical_units_clean = lexical_units_df.copy()
        
        for col in ["variants", "examples", "lemma"]:
            if col in lexical_units_clean:
                lexical_units_clean[col] = lexical_units_clean[col].fillna("")
        
        return lexical_units_clean
    
    def _aggregate_by_synset(self, lexical_units_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate lemmas, variants, and examples by synset."""
        agg_by_sense = lexical_units_df.groupby("synset_id").agg(
            lemmas=("lemma", lambda s: self.text_separator.join(sorted(set(s.astype(str))))),
            variants=("variants", lambda s: self._aggregate_list_column(s)),
            examples=("examples", lambda s: self._aggregate_list_column(s))
        ).reset_index()
        
        return agg_by_sense
    
    def _aggregate_list_column(self, series) -> str:
        """Helper to aggregate list columns with proper flattening."""
        items = []
        for sublist in series:
            if isinstance(sublist, list):
                items.extend([str(item) for item in sublist if item and str(item) != ""])
            elif sublist and str(sublist) != "":
                items.append(str(sublist))
        return self.text_separator.join(sorted(set(items)))
    
    def _create_embedding_texts(self, senses_df: pd.DataFrame) -> pd.DataFrame:
        """Create the final embedding texts by combining all components."""
        # Normalize missing values in merged data
        for col in ["paraphrase", "lemmas", "variants", "examples"]:
            if col in senses_df:
                senses_df[col] = senses_df[col].fillna("").astype(str)
        
        # Text-clean the input for embedding
        cols = ["paraphrase", "examples", "lemmas", "variants"]
        senses_df[cols] = (senses_df[cols]
            .fillna("")
            .astype(str)
            .replace(r"^\s*\[\s*\]\s*$", "", regex=True)
            .apply(lambda s: s.str.strip())
        )
        
        # Create combined text for embedding
        senses_df["text_for_embedding"] = (
            senses_df["paraphrase"].fillna("") + self.text_separator +
            senses_df["examples"] + self.text_separator +
            senses_df["lemmas"] + self.text_separator +
            senses_df["variants"]
        ).str.strip(self.text_separator)
        
        # Create final sense_texts dataframe
        sense_texts = senses_df[["synset_id", "text_for_embedding"]].copy()
        
        # Check for any null values
        null_count = sense_texts["text_for_embedding"].isnull().sum()
        if null_count > 0:
            self.logger.warning(f"Found {null_count} null text_for_embedding values")
        
        return sense_texts
    