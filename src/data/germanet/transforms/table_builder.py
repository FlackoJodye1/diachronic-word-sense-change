"""Creates the lemma_senses table by joining lexical units with synsets."""

import logging
import pandas as pd


class TableBuilder:
    """Joins synsets and lexical units into a flat lemma_senses DataFrame."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_lemma_senses(self, synsets_df: pd.DataFrame, lexical_units_df: pd.DataFrame) -> pd.DataFrame:
        """Join lexical units with synsets to create the lemma_senses table."""
        if synsets_df is None or synsets_df.empty:
            raise ValueError("synsets_df cannot be None or empty")
        if lexical_units_df is None or lexical_units_df.empty:
            raise ValueError("lexical_units_df cannot be None or empty")

        self.logger.info('Creating lemma_senses table')

        lemma_senses = lexical_units_df.merge(
            synsets_df[['synset_id', 'paraphrase']],
            on='synset_id',
            how='left',
        )

        lemma_senses = lemma_senses[[
            'lemma', 'pos', 'synset_id', 'class', 'paraphrase',
            'variants', 'examples'
        ]].copy()

        lemma_senses = lemma_senses.drop_duplicates(subset=['lemma', 'synset_id'])

        self.logger.info(f'Created {len(lemma_senses)} unique lemma-synset pairs')
        return lemma_senses