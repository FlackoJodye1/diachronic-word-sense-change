"""Creates the (lemma, POS) → [synset_id, ...] candidate lookup used by the WSD predictor."""

import logging
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple


class CandidateMapper:
    """Creates and saves the candidate mapping from (lemma, POS) to synset IDs."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_candidate_map(self, lemma_senses_df: pd.DataFrame) -> Dict[Tuple[str, str], List[str]]:
        """Create candidate mapping from lemma_senses DataFrame."""
        if lemma_senses_df is None or lemma_senses_df.empty:
            raise ValueError("lemma_senses_df cannot be None or empty")

        required_columns = ['lemma', 'pos', 'synset_id']
        missing_columns = [col for col in required_columns if col not in lemma_senses_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        self.logger.info('Creating candidate mapping from lemma_senses data')

        candidates = {}
        for (lemma, pos), sub in lemma_senses_df.groupby(["lemma", "pos"]):
            ids = sub["synset_id"].dropna().astype(str).unique().tolist()
            if ids:
                candidates[(lemma, pos)] = ids

        self.logger.info(f'Created candidate mapping with {len(candidates)} (lemma, pos) keys')
        return candidates

    def save_candidate_map(self, output_path: Path, candidates: Dict) -> None:
        """Save candidate mapping to pickle file."""
        if not candidates:
            raise ValueError("No candidates to save. Run create_candidate_map first.")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            pickle.dump(candidates, f)

        self.logger.info(f'Saved candidate mapping to {output_path} with {len(candidates)} keys')