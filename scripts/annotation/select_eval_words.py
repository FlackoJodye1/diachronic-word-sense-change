#!/usr/bin/env python3
"""
Select evaluation words from the 750-word dataset with a 3-4-3 difficulty distribution.
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def select_evaluation_words():
    """Select 10 words with 3-4-3 easy/medium/hard distribution."""
    data_path = DATA_DIR / "processed" / "target_words" / "target_words_with_germanet_sense_counts.tsv"
    df = pd.read_csv(data_path, sep="\t")
    logger.info(f"Loaded {len(df)} words")

    easy = df[df["num_senses"] <= 2]
    medium = df[(df["num_senses"] >= 3) & (df["num_senses"] <= 4)]
    hard = df[df["num_senses"] >= 5]
    logger.info(f"Available: {len(easy)} easy, {len(medium)} medium, {len(hard)} hard")

    selected = {
        "easy": easy.sample(3, random_state=42)["word"].tolist(),
        "medium": medium.sample(4, random_state=43)["word"].tolist(),
        "hard": hard.sample(3, random_state=44)["word"].tolist(),
    }

    all_words = []
    for difficulty, words in selected.items():
        logger.info(f"{difficulty.capitalize()}: {words}")
        all_words.extend(words)

    output_path = DATA_DIR / "processed" / "target_words" / "eval_selection.tsv"
    df[df["word"].isin(all_words)].to_csv(output_path, sep="\t", index=False)
    logger.info(f"Saved to {output_path.name}")

    return selected


if __name__ == "__main__":
    select_evaluation_words()